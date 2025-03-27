import json

import requests


def fetch_and_save_exchange_list(file_path="exchange_list.json"):
    url = "https://api.coingecko.com/api/v3/exchanges"
    response = requests.get(url)
    data = response.json()

    exchange_names = set()

    for ex in data:
        name = ex.get("name", "").strip()
        if name:
            exchange_names.add(name)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(sorted(exchange_names), f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    fetch_and_save_exchange_list()
