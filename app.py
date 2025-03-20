import threading
import time
import traceback
import pika
from flask import Flask
from configs.const.style import QUEUE_NAME, RABBITMQ_URL
from pipelines.inpainting import inpainting
from queues.receiver import RabbitMQReceiver

app = Flask(__name__)

inpainting.inpainting_pipeline = inpainting.load_inpainting_pipeline()

if inpainting.inpainting_pipeline is None:
    print("❌ Failed to load pipeline. Exiting...")
    exit(1)

print("✅ Pipeline loaded successfully!")

def start_worker():
    while True:
        try:
            print("Connecting to RabbitMQ...")
            params = pika.URLParameters(RABBITMQ_URL)
            connection = pika.BlockingConnection(params)
            channel = connection.channel()

            channel.queue_declare(queue=QUEUE_NAME, durable=False, exclusive=False, auto_delete=False, arguments={"x-message-ttl": 300000})
            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(queue=QUEUE_NAME, auto_ack=True, on_message_callback=RabbitMQReceiver.callback)

            print("Worker connected and listening...")

            while connection.is_open:
                connection.process_data_events(time_limit=1)
                time.sleep(0.5)

        except (pika.exceptions.AMQPError, pika.exceptions.ConnectionClosed) as e:
            print(f"Connect error: {e}, try to connect after 5 second...")
            try:
                if connection.is_open:
                    return
            except:
                pass
            time.sleep(5)

if __name__ == '__main__':
    try:
        threading.Thread(target=start_worker, daemon=True).start()
        app.run(host='0.0.0.0', port=5000, threaded=True)
    except Exception as e:
        traceback.print_exc()
        print(f"Error starting worker: {e}")