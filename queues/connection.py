import time
import pika
from configs.const.style import RABBITMQ_URL
import ssl
from pika.exceptions import ConnectionClosedByBroker, AMQPConnectionError


class RabbitMQConnection:
    def __init__(self):
        self.connect()

    def connect(self, max_retries=5):
        retry_delay = 2
        for attempt in range(max_retries):
            try:
                params = pika.URLParameters(RABBITMQ_URL)
                params.heartbeat = 30
                params.connection_attempts = 5
                params.retry_delay = 3

                self.connection = pika.BlockingConnection(params)
                self.channel = self.connection.channel()
                print("RabbitMQ connected!")
                return
            except (ConnectionClosedByBroker, AMQPConnectionError, ssl.SSLEOFError) as e:
                print(f"RabbitMQ connection failed (attempt {attempt + 1}/{max_retries}): {e}")
                self.connection = pika.BlockingConnection(params)
                self.channel = self.connection.channel()
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
    
    def reconnect(self):
        if self.connection and self.connection.is_open:
            self.connection.close()
        self.connection = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=self.queue, durable=False, exclusive=False, auto_delete=False)
        print("RabbitMQ reconnected!")

    def close(self):
        if self.connection and self.connection.is_open:
            self.connection.close()
            print("RabbitMQ connection closed!")
