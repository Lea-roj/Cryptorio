from CoinDeskScraper import *
from helper_functions import analyze_text

if __name__ == "__main__":
    entries = fetch_rss_entries()
    # producer = get_kafka_producer()

    for entry in entries:
        full_text = extract_full_article(entry.link)

        if not full_text:
            continue

        content = full_text

        result = analyze_text(content)
        data = {
            "title": entry.title,
            "link": entry.link,
            "published": entry.published,
            "content": content,
            "analysis": result
        }

        log_article(data)
        # send_to_kafka(producer, 'analyzed_articles', data)