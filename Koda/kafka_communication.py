from kafka import KafkaProducer
import json


def get_kafka_producer():
    return KafkaProducer(
        bootstrap_servers='192.168.1.12:29092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )


def send_to_kafka(producer, topic, data):
    producer.send(topic, value=data)
    producer.flush()
