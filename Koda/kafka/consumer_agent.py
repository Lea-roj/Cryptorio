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

    for message in consumer:
        data = message.value
        print("\n========================")

        try:
            print(f"Title: {data.get('title', 'N/A')}")
            print(f"Published: {data.get('published', 'N/A')}")
            print(f"Link: {data.get('link', 'N/A')}")
            print(f"Summary: {data.get('summary', 'N/A')}")
            print(f"Sentiment (BERT): {data['analysis'].get('bert_sentiment_summary', 'N/A')}")
            print(f"Crypto scores: {data['analysis'].get('crypto_scores', {})}")

            print(f"Entities: {data['analysis'].get('entities', [])}")
            print(f"Top keywords (TF-IDF): {data['analysis'].get('top_keywords_tfidf', [])}")
            print(f"Vader sentiment: {data['analysis'].get('vader_sentiment', {})}")
            print(f"Entity sentiment: {data['analysis'].get('entity_sentiment', {})}")
            print(f"Content snippet: {data.get('content', '')[:300]}...")

        except Exception as e:
            print(f"Error processing message: {e}")
            print("Full message:", data)

        print("========================\n")


if __name__ == "__main__":
    start_consumer()
