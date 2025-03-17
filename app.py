import pika
from threading import Thread
from flask import Flask
from configs.const.style import QUEUE_NAME, RABBITMQ_URL
from queues.consumer import callback


app = Flask(__name__)


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
    Thread(target=start_worker).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)