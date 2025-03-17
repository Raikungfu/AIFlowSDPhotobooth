import json
import pika
import time
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
        rabbitmq_client.send_response(response)
    except Exception as e:
        rabbitmq_client.send_response({"CorrelationId": correlation_id, "error": str(e)})
        traceback.print_exc()
        print(f"Error: {e}")


class RabbitMQClient:
    def __init__(self, url, queue):
        self.url = url
        self.queue = queue
        self.connection = None
        self.channel = None
        self.connect()

    def connect(self):
        while True:
            try:
                params = pika.ConnectionParameters(
                    host="rabbitmq-server",
                    heartbeat=600,
                    blocked_connection_timeout=300
                )
                self.connection = pika.BlockingConnection(params)
                self.channel = self.connection.channel()
                self.channel.queue_declare(queue=self.queue, durable=False, exclusive=False, auto_delete=False)
                print("RabbitMQ connected!")
                break
            except pika.exceptions.AMQPConnectionError as e:
                print(f"Failed to connect to RabbitMQ: {e}, retrying in 5s...")
                time.sleep(5)

    def send_response(self, response):
        try:
            if self.connection is None or self.connection.is_closed:
                print("Reconnecting to RabbitMQ...")
                self.connect()

            self.channel.basic_publish(
                exchange="",
                routing_key=self.queue,
                body=json.dumps(response)
            )
            print(f"Sent response: {response}")
        except pika.exceptions.AMQPConnectionError:
            print("Connection lost, retrying in 5 seconds...")
            time.sleep(5)
            self.send_response(response)
        except Exception as e:
            print(f"Failed to send response: {e}")
            traceback.print_exc()

rabbitmq_client = RabbitMQClient(RABBITMQ_URL, REPLY_QUEUE)


def callback(ch, method, properties, body):
    try:
        message = json.loads(body)

        image_url = message.get("imageUrl")
        correlation_id = message.get("correlationId")
        photo_style = message.get("photoStyle")

        print(f"Received: {image_url}, CorrelationId: {correlation_id}")

        process_image(image_url, correlation_id, photo_style, bucket)
    except Exception as e:
        print(f"Error in callback: {e}")
        traceback.print_exc()