from kafka import KafkaConsumer
import json


def start_consumer():
    consumer = KafkaConsumer(
        'analyzed_articles',
        bootstrap_servers='192.168.1.12:29092',
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='nlp-analysis-consumer-group',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    print("Listening on topic: analyzed_articles")
    for message in consumer:
        data = message.value
        print("\n========================")

        if 'title' in data and 'analysis' in data:
            print(f"Title: {data['title']}")
            print(f"Published: {data.get('published', 'N/A')}")
            print(f"Link: {data.get('link', 'N/A')}")
            print(f"Sentiment (BERT): {data['analysis'].get('bert_sentiment_summary', 'N/A')}")
            print(f"Crypto scores: {data['analysis'].get('crypto_scores', {})}")
        else:
            print("Received unexpected message:", data)

        print("========================\n")


if __name__ == "__main__":
    start_consumer()
