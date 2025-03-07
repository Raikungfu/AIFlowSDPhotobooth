import requests
import io
import json
import pika
from flask_cors import CORS
from threading import Thread
from flask import Flask, jsonify
from PIL import Image
from enums.prompt_type import PromptType
from enums.user_info import Gender
from utils.background_prcessing import place_on_new_background, remove_background
from utils.face_alignment import align_face
from utils.firebase_utils import initialize_firebase
from utils.image_upload import upload_image, inpaint_image
from utils.masks import generate_face_lower_mask
from utils.optimize_image import optimize_image

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


RABBITMQ_URL = "amqps://gzuozlgd:oFx2YLFVunNoMijR4FFRz97K2n2gA4gr@armadillo.rmq.cloudamqp.com/gzuozlgd"
QUEUE_NAME = "image_processing_queue"
REPLY_QUEUE = "image_response_queue"


def download_image(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    return None
    

def process_image(image_url, correlation_id, options, form_data):
    try:
        image = download_image(image_url)
        if image is None:
            raise Exception("Invalid image URL")

        aligned_image = align_face(image)
        optimized_image = optimize_image(aligned_image)
        image_outputs = [upload_image(optimized_image, bucket)]

        if options.get("remove_background"):
            image, image_mask = remove_background(optimized_image)
            image_outputs.append(upload_image(image_mask, bucket))
        else:
            image_mask = None

        if image_mask:
            new_background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = place_on_new_background(image, new_background).convert("RGB")

        mask_face, mask_clothes, mask_combined = generate_face_lower_mask(image, image_mask)
        if not mask_face or not mask_clothes:
            raise Exception("No face detected, create mask failed")

        image_outputs.extend([
            upload_image(mask_clothes, bucket),
            upload_image(mask_face, bucket)
        ])

        edit_strength = form_data.get('edit_strength', {'face': 1.0, 'clothes': 1.0, 'combine': 1.0})
        guidance_scale = form_data.get('guidance_scale', {'face': 7.5, 'clothes': 7.5, 'combine': 7.5})
        seek_value = form_data.get('seek', 0)
        gender = Gender(form_data.get('gender', 'male').lower())

        edit_types = {
            PromptType.FACE: (mask_face, edit_strength.get('face', 1.0), guidance_scale.get('face', 7.5)),
            PromptType.CLOTHES: (mask_clothes, edit_strength.get('clothes', 1.0), guidance_scale.get('clothes', 7.5)),
            PromptType.COMBINE: (mask_combined, edit_strength.get('combine', 1.0), guidance_scale.get('combine', 7.5))
        }

        for edit_type, (mask, strength, scale) in edit_types.items():
            inpainted_image = inpaint_image(image, mask, seek_value, edit_type, gender, strength, scale)
            if inpainted_image:
                image_outputs.append(upload_image(inpainted_image, bucket))
            else:
                raise Exception(f"Failed to inpaint {edit_type.name.lower()}")

        response = {"CorrelationId": correlation_id, "ProcessedImageUrl": image_outputs}
        send_response_to_queue(response)

    except Exception as e:
        send_response_to_queue({"CorrelationId": correlation_id, "error": str(e)})
        print(f"Error: {e}")



def send_response_to_queue(response):
    params = pika.URLParameters(RABBITMQ_URL)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.queue_declare(queue=REPLY_QUEUE, durable=False, exclusive=False, auto_delete=False)

    channel.basic_publish(
        exchange="",
        routing_key=REPLY_QUEUE,
        body=json.dumps(response)
    )

    connection.close()
    print(f"Sent response: {response}")


def callback(ch, method, properties, body):
    message = json.loads(body)
    image_url = message.get("ImageUrl")
    correlation_id = message.get("CorrelationId")
    options = message.get("options", {})
    form_data = message.get("personal_info", {})

    print(f"Received: {image_url}, CorrelationId: {correlation_id}")

    process_image(image_url, correlation_id, options, form_data)


def start_worker():
    params = pika.URLParameters(RABBITMQ_URL)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()

    channel.queue_declare(queue=QUEUE_NAME, durable=False, exclusive=False, auto_delete=False, arguments={"x-message-ttl": 300000})
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=QUEUE_NAME, auto_ack=True, on_message_callback=callback)

    print("Worker started...")
    channel.start_consuming()



if __name__ == '__main__':
    cred_path = 'firebase/social-networking-8c187-firebase-adminsdk-cbnj3-5738be94d4.json'
    firebaseConfig = {
        'storageBucket': "social-networking-8c187.appspot.com"
    }
    
    bucket = initialize_firebase(cred_path, firebaseConfig)


    Thread(target=start_worker).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
