import feedparser
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime, timezone
from helper_functions import analyze_text
from kafka_communication import get_kafka_producer, send_to_kafka

RSS_URL = "https://www.coindesk.com/arc/outboundfeeds/rss/"


def fetch_rss_entries():
    feed = feedparser.parse(RSS_URL)
    return feed.entries


def extract_full_article(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.content, 'html.parser')

        paragraphs = soup.find_all("p")
        article_text = " ".join(p.get_text() for p in paragraphs if p.get_text())
        return article_text.strip()
    except Exception as e:
        return e


def log_article(new_data, log_path="logs/coindesk_news.json"):
    new_data["timestamp"] = datetime.now(timezone.utc).isoformat()

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        existing_data = []

    existing_data.append(new_data)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)


def main():
    entries = fetch_rss_entries()
    producer = get_kafka_producer()

    for entry in entries:
        full_text = extract_full_article(entry.link)

        content = full_text

        result = analyze_text(content)
        data = {
            "title": entry.title,
            "link": entry.link,
            "published": entry.published,
            "summary": entry.summary,
            "content": content,
            "analysis": result
        }

        # log_article(data)
        send_to_kafka(producer, 'analyzed_articles', data)


if __name__ == "__main__":
    main()
