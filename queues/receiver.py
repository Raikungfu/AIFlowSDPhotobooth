import json
import traceback
from queues.connection import RabbitMQConnection
from queues.sender import RabbitMQSender
from utils.firebase_utils import initialize_firebase
from utils.image_upload import download_image, upload_image
from effects.photo_style_flow import (
    process_artistic, process_anime, process_cartoon, process_fantasy,
    process_fashion_style, process_id_photo, process_modern_style,
    process_nature_style, process_vintage
)
from configs.const.config import cred_path, firebaseConfig
from configs.const.style import REPLY_QUEUE

STYLE_PROCESSORS = {
    "anime": process_anime,
    "artistic": process_artistic,
    "cartoon": process_cartoon,
    "fantasy": process_fantasy,
    "fashion style": process_fashion_style,
    "id photo": process_id_photo,
    "id photo (controlnet)": process_id_photo,
    "modern": process_modern_style,
    "nature": process_nature_style,
    "vintage": process_vintage,
    "cyberpunk": process_vintage,
    "watercolor painting": process_vintage,
    "black & white classic": process_vintage,
    "oil painting": process_vintage,
}

bucket = initialize_firebase(cred_path, firebaseConfig)

class RabbitMQReceiver:
    def __init__(self, queue_name):
        self.queue = queue_name
        self.connection = RabbitMQConnection()
        self.channel = self.connection.get_channel()
        self.channel.queue_declare(queue=self.queue, durable=False, exclusive=False, auto_delete=False)
    
    def start_listening(self):
        self.channel.basic_consume(queue=self.queue, on_message_callback=self.callback, auto_ack=True)
        print(f"Listening for messages on '{self.queue}'...")
        self.channel.start_consuming()

    def callback(ch, method, properties, body):
        try:
            message = json.loads(body)
            print(f"Received: {message}")

            image_url = message.get("imageUrl")
            correlation_id = message.get("correlationId")
            photo_style = message.get("photoStyle")

            if not image_url or not correlation_id or not photo_style:
                raise ValueError("Invalid message format")

            process_image(image_url, correlation_id, photo_style)
            #ch.basic_ack(delivery_tag=method.delivery_tag)

        except json.JSONDecodeError:
            print("Invalid JSON received.")
        except Exception as e:
            print(f"Unexpected error: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            traceback.print_exc()


def process_image(image_url, correlation_id, photo_style):
    try:
        image = download_image(image_url)
        if image is None:
            raise ValueError("Invalid image URL")

        processor = STYLE_PROCESSORS.get(photo_style)
        if not processor:
            raise ValueError(f"Unknown photo style: {photo_style}")

        processed_images = processor(image)
        if not processed_images:
            raise ValueError("Failed to process image")

        result = [upload_image(img, bucket) for img in processed_images]
        print(f"Processed {len(result)} images.")

        with RabbitMQSender(REPLY_QUEUE) as sender:
            sender.send_response({"CorrelationId": correlation_id, "ProcessedImageUrl": result})
    except Exception as e:
        error_message = str(e)
        print(f"Error: {error_message}")
        traceback.print_exc()
        with RabbitMQSender(REPLY_QUEUE) as sender:
            sender.send_response({"CorrelationId": correlation_id, "error": error_message})
