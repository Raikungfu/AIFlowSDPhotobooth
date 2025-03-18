import time
import pika
from configs.const.style import RABBITMQ_URL

class RabbitMQConnection:
    def __init__(self):
        self.url = RABBITMQ_URL
        self.connection = None
        self.channel = None
        self.connect()

    def connect(self, max_retries=5):
        retry_delay = 2
        for attempt in range(max_retries):
            try:
                params = pika.URLParameters(self.url)

                params.heartbeat = 60
                params.connection_attempts = 5
                params.retry_delay = 3

                self.connection = pika.BlockingConnection(params)
                self.channel = self.connection.channel()
                print("RabbitMQ connected!")
                return
            except pika.exceptions.AMQPConnectionError as e:
                print(f"RabbitMQ connection failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(retry_delay)
                retry_delay *= 2

        print("Could not connect to RabbitMQ after multiple attempts.")
        raise Exception("RabbitMQ connection failed")

    def get_channel(self):
        if self.channel is None or self.channel.is_closed:
            if self.connection is None or self.connection.is_closed:
                self.connect()
            self.channel = self.connection.channel()
        return self.channel

    def close(self):
        if self.connection and self.connection.is_open:
            self.connection.close()
            print("RabbitMQ connection closed!")
