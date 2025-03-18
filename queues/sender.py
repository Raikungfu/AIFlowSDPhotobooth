import json
import time
import pika.exceptions
import traceback
from queues.connection import RabbitMQConnection


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
                reply_to=self.reply_queue
            )


            self.channel.basic_publish(
                exchange="",
                routing_key=self.queue,
                body=json.dumps(response),
                auto_ack=True
            )
            print(f"Sent response: {response}")
        except pika.exceptions.AMQPConnectionError:
            print("Connection lost, retrying in 5 seconds...")
            time.sleep(5)
            self.send_response(response)
        except Exception as e:
            print(f"Failed to send response: {e}")
            traceback.print_exc()