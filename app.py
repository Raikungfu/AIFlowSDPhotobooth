import json
import pika
from threading import Thread
from flask import Flask
from utils.firebase_utils import initialize_firebase
from const.style import QUEUE_NAME, RABBITMQ_URL, REPLY_QUEUE
from const.config import cred_path, firebaseConfig
import traceback
from utils.image_upload import download_image, upload_image
from flow.photo_style_flow import process_artistic_portrait, process_cartoon, process_fashion_style, process_id_photo, process_vintage


STYLE_PROCESSORS = {
    "id_photo": process_id_photo,
    "artistic_portrait": process_artistic_portrait,
    "cartoon": process_cartoon,
    "vintage": process_vintage,
    "fashion_style": process_fashion_style,
}

app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})


def process_image(image_url, correlation_id, form_data, bucket):
    try:
        image = download_image(image_url)
        if image is None:
            raise Exception("Invalid image URL")
        
        photo_style = form_data.get("photoStyle", "id_photo")
        processor = STYLE_PROCESSORS.get(photo_style)
        if not processor:
            raise Exception(f"Unknown photo style: {photo_style}")
        
        processed_images = processor(image, bucket)

        if not processed_images:
            raise Exception("Failed to process image")
        
        result = []

        for i, img in enumerate(processed_images):
            processed_url = upload_image(img, bucket)
            result.append(processed_url)
            print(f"Processed image {i}: {processed_url}")

        response = {"CorrelationId": correlation_id, "ProcessedImageUrl": result}
        send_response_to_queue(response)
    except Exception as e:
        send_response_to_queue({"CorrelationId": correlation_id, "error": str(e)})
        traceback.print_exc()
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
    print(f"Received: {message}")

    image_url = message.get("imageUrl")
    correlation_id = message.get("correlationId")
    form_data = message.get("formData", {})

    print(f"Received: {image_url}, CorrelationId: {correlation_id}")

    process_image(image_url, correlation_id, form_data, bucket)


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
    bucket = initialize_firebase(cred_path, firebaseConfig)


    Thread(target=start_worker).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
