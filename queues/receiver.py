import json
import traceback
from DTO.requestDTO import extract_message_data, validate_message
from effects.flow import process_image
from queues.connection import RabbitMQConnection
from queues.sender import RabbitMQSender
from utils.firebase_utils import initialize_firebase
from utils.image_upload import download_image, parallel_upload_images
from configs.const.config import cred_path, firebaseConfig
from configs.const.style import REPLY_QUEUE

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
            print(f"Received message: {message}")

            data = extract_message_data(message)
            validate_message(data)

            image = download_image(data["image_url"])
            if image is None:
                raise ValueError("Invalid image URL")

            processed_images = process_image(
                image,
                data["photo_style"],
                data["prompt"],
                data["negative_prompt"],
                data["controlnets"],
                data["num_images"],
                data["num_inference_steps"],
                data["guidance_scale"],
                data["strength"],
                data["face_image"],
                data["background_remover"],
                data["height"],
                data["width"],
                data["mode"]
            )

            if not processed_images:
                raise ValueError("Failed to process image")
            
            uploaded_urls = parallel_upload_images(processed_images, bucket)
            print(f"Processed and uploaded {len(uploaded_urls)} images.")

            send_response({
                "CorrelationId": data["correlation_id"],
                "ProcessedImageUrl": uploaded_urls,
                "SessionCode": data["session_code"],
                "PhotoStyleId": data["photo_style_id"]
            })

        except json.JSONDecodeError:
            print("Invalid JSON received.")
        except Exception as e:
            print(f"Unexpected error: {e}")
            traceback.print_exc()
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)


def send_response(response):
    with RabbitMQSender(REPLY_QUEUE) as sender:
        sender.send_response(response)