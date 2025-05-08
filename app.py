import threading
import time
import traceback
import pika
from flask import Flask
from configs.const.style import QUEUE_NAME, RABBITMQ_URL
from queues.receiver import RabbitMQReceiver
from pipelines.inpainting import load_inpainting_controlnet_pipeline, load_inpainting_pipeline
from pipelines import pipeline_instance
from image_input_controlnets.depth import load_zoe_model
from image_input_controlnets.open_pose import load_openpose_processor

app = Flask(__name__)

def start_worker():
    while True:
        try:
            print("Connecting to RabbitMQ...")
            receiver = RabbitMQReceiver(queue_name=QUEUE_NAME, max_workers=3)
            receiver.start_listening()
            print("Worker connected and listening...")

        except (pika.exceptions.AMQPError, pika.exceptions.ConnectionClosed) as e:
            print(f"[RabbitMQ] Connection error: {e}. Reconnecting in 5 seconds...")
            traceback.print_exc()
            time.sleep(5)
        except Exception as e:
            print(f"[Worker] Unexpected error: {e}")
            traceback.print_exc()
            time.sleep(5)


if __name__ == '__main__':
    try:
        # Loading multiple pipelines into CPU/GPU — can cause heavy memory usage
        # Consider limiting the number of pipelines or optimizing loading if needed
        pipeline_instance.controlnet_canny_pose_pipeline = load_inpainting_controlnet_pipeline(controlnets=["canny", "pose"])
        pipeline_instance.zoe_image_processor, pipeline_instance.zoe_model, pipeline_instance.device = load_zoe_model()
        pipeline_instance.openpose_processor = load_openpose_processor()
        pipeline_instance.controlnet_depth_pose_pipeline = load_inpainting_controlnet_pipeline(controlnets=[ "depth" , "pose"])
        pipeline_instance.controlnet_canny_depth_pipeline = load_inpainting_controlnet_pipeline(controlnets=["canny", "depth"])
        # Load basic inpainting pipeline
        pipeline_instance.inpainting_pipeline = load_inpainting_pipeline()
        threading.Thread(target=start_worker, daemon=True).start()
        app.run(host='0.0.0.0', port=5000, threaded=True)
    except Exception as e:
        traceback.print_exc()
        print(f"Error starting worker: {e}")