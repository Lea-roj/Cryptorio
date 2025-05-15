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


if __name__ == "__main__":
    producer = get_kafka_producer()

    with open("../coindesk_news.json", "r", encoding="utf-8") as f:
        articles = json.load(f)

    for article in articles:
        send_to_kafka(producer, "analyzed_articles", article)
        print(f"Sent: {article['title']}")
