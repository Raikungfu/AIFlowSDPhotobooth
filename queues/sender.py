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


import json
import pika


class RabbitMQSender:
    def __init__(self, queue_name, reply_queue, rabbitmq_url="amqp://guest:guest@localhost:5672/"):
        self.queue_name = queue_name
        self.reply_queue = reply_queue
        self.connection_params = pika.URLParameters(rabbitmq_url)
        self.connection = pika.BlockingConnection(self.connection_params)
        self.channel = self.connection.channel()

        self.channel.queue_declare(queue=self.queue_name, durable=False, exclusive=False, auto_delete=False, 
                                   arguments={"x-message-ttl": 300000})

    def send_message(self, response, correlation_id):
        properties = pika.BasicProperties(
            correlation_id=correlation_id,
            reply_to=self.reply_queue
        )

        self.channel.basic_publish(
            exchange="",
            routing_key=self.queue_name,
            body=json.dumps(response),
            properties=properties
        )

        print(f"✅ Sent message to {self.queue_name}: {response}")