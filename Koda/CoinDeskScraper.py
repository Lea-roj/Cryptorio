import feedparser
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime, timezone
from global_params import *


def fetch_rss_entries():
    feed = feedparser.parse(RSS_URL)
    return feed.entries


def extract_full_article(url):
    try:
        response = requests.get(url, timeout=HTTP_TIMEOUT_SECONDS)
        soup = BeautifulSoup(response.content, 'html.parser')

        paragraphs = soup.find_all("p")
        article_text = " ".join(p.get_text() for p in paragraphs if p.get_text())
        return article_text.strip()
    except Exception as e:
        print(f"Could not fetch article from {url}: {e}")
        return None


def log_article(new_data, log_path=DEFAULT_LOG_PATH):
    new_data["timestamp"] = datetime.now(timezone.utc).isoformat()

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        existing_data = []

    existing_data.append(new_data)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)