import json
import traceback
from configs.enums.inpaint_mode import InpaintMode
from effects.flow import process_image
from queues.connection import RabbitMQConnection
from queues.sender import RabbitMQSender
from utils.firebase_utils import initialize_firebase
from utils.image_upload import download_image, upload_image
from configs.const.config import cred_path, firebaseConfig
from configs.const.style import REPLY_QUEUE
from utils.process_input import parse_controlnets

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
            photo_style_id = message.get("photoStyleId")
            session_code = message.get("sessionCode")
            photo_style = message.get("photoStyle")
            prompt = message.get("prompt")
            negative_prompt = message.get("negativePrompt")
            controlnets = parse_controlnets(message.get("controlnets")) if message.get("controlnets") else None
            numImages = message.get("numImagesPerGen", 1)
            guidance_scale = message.get("guidanceScale", 7.5)
            strength = message.get("strength", 0.5)
            face_image = message.get("faceImage")
            background_remover = message.get("backgroundRemover", False)
            height = message.get("height", None)
            width = message.get("width", None)
            mode_map = [InpaintMode.NONE, InpaintMode.KEEP_FACE, InpaintMode.KEEP_STRUCTURE]
            mode = mode_map[message.get("mode", 0)]
            numInferenceSteps = message.get("numInferenceSteps", 30)

            if not image_url or not correlation_id or not photo_style:
                raise ValueError("Invalid message format")
            
            try:
                image = download_image(image_url)
                if image is None:
                    raise ValueError("Invalid image URL")

                processed_images = process_image(image, photo_style, prompt, negative_prompt, controlnets, numImages, numInferenceSteps, guidance_scale, strength, face_image, background_remover, height, width, mode)
                if not processed_images:
                    raise ValueError("Failed to process image")

                result = [upload_image(img, bucket) for img in processed_images]
                print(f"Processed {len(result)} images.")

                with RabbitMQSender(REPLY_QUEUE) as sender:
                    sender.send_response({"CorrelationId": correlation_id, "ProcessedImageUrl": result, "SessionCode": session_code, "PhotoStyleId": photo_style_id})
            except Exception as e:
                error_message = str(e)
                print(f"Error: {error_message}")
                traceback.print_exc()
                with RabbitMQSender(REPLY_QUEUE) as sender:
                    sender.send_response({"CorrelationId": correlation_id, "error": error_message})

        except json.JSONDecodeError:
            print("Invalid JSON received.")
        except Exception as e:
            print(f"Unexpected error: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            traceback.print_exc()