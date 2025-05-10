import time
import pika
import ssl
from pika.exceptions import ConnectionClosedByBroker, AMQPConnectionError
from configs.const.style import RABBITMQ_URL

class RabbitMQConnection:
    def __init__(self, queue_name):
        self.queue = queue_name
        self.connection = None
        self.channel = None
        self.connect()

    def connect(self, max_retries=5):
        retry_delay = 2
        context = ssl.create_default_context()
        params = pika.URLParameters(RABBITMQ_URL)
        params.heartbeat = 30
        params.connection_attempts = 5
        params.retry_delay = 3
        params.ssl_options = pika.SSLOptions(context)

        for attempt in range(max_retries):
            try:
                self.connection = pika.BlockingConnection(params)
                self.channel = self.connection.channel()
                print("RabbitMQ connected!")
                return
            except (ConnectionClosedByBroker, AMQPConnectionError, ssl.SSLEOFError) as e:
                print(f"RabbitMQ connection failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(retry_delay)
                retry_delay *= 2

        print("Could not connect to RabbitMQ after multiple attempts.")
        raise Exception("RabbitMQ connection failed")

    def get_channel(self):
        if not self.channel or self.channel.is_closed:
            print("Reconnecting to RabbitMQ...")
            self.reconnect()
        return self.channel

    def reconnect(self):
        if self.connection and self.connection.is_open:
            self.connection.close()
            print("Closing existing RabbitMQ connection...")

        self.connect()
        print("Reconnected to RabbitMQ!")

    def close(self):
        if self.connection and self.connection.is_open:
            self.connection.close()
            print("RabbitMQ connection closed!")
