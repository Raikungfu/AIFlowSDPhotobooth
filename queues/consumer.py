import json
import pika
from utils.firebase_utils import initialize_firebase
import traceback
from configs.const.style import RABBITMQ_URL, REPLY_QUEUE
from utils.image_upload import download_image, upload_image
from effects.photo_style_flow import process_artistic, process_anime, process_cartoon, process_fantasy, process_fashion_style, process_id_photo, process_modern_style, process_nature_style, process_vintage
from configs.const.config import cred_path, firebaseConfig


STYLE_PROCESSORS = {
    "anime": process_anime,
    "artistic_portrait": process_artistic,
    "cartoon": process_cartoon,
    "fantasy": process_fantasy,
    "fashion_style": process_fashion_style,
    "id_photo": process_id_photo,
    "modern": process_modern_style,
    "nature": process_nature_style,
    "vintage": process_vintage,
}

bucket = initialize_firebase(cred_path, firebaseConfig)


def process_image(image_url, correlation_id, photo_style, bucket):
    try:
        image = download_image(image_url)
        if image is None:
            raise Exception("Invalid image URL")
        
        processor = STYLE_PROCESSORS.get(photo_style)
        if not processor:
            raise Exception(f"Unknown photo style: {photo_style}")
        
        processed_images = processor(image)

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
    photo_style = message.get("photoStyle")

    print(f"Received: {image_url}, CorrelationId: {correlation_id}")

    process_image(image_url, correlation_id, photo_style, bucket)