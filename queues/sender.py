import json
import time
import pika
import traceback
from queues.connection import RabbitMQConnection
import ssl
from pika.exceptions import ConnectionClosedByBroker, AMQPConnectionError
from configs.const.style import REPLY_QUEUE


class RabbitMQSender:
    def __init__(self, queue_name):
        self.queue = queue_name
        self.connection = RabbitMQConnection()
        self.channel = self.connection.get_channel()
        self.channel.queue_declare(queue=self.queue, durable=False, exclusive=False, auto_delete=False)

    def send_response(self, response):
        try:
            properties = pika.BasicProperties(
                delivery_mode=2,
                correlation_id=response.get("CorrelationId"),
                reply_to=self.queue
            )

            self.channel.basic_publish(
                exchange="",
                routing_key=self.queue,
                body=json.dumps(response),
                properties=properties
            )
            print(f"Sent response: {response}")
        except (ConnectionClosedByBroker, AMQPConnectionError, ssl.SSLEOFError) as e:
            print("Connection lost, retrying in 5 seconds....Error: {e}")
            RabbitMQConnection(REPLY_QUEUE).reconnect()
            time.sleep(5)
            self.send_response(response)
        except Exception as e:
            print(f"Failed to send response: {e}")
            traceback.print_exc()
        finally:
            self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()