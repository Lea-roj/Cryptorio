import requests
import json


def fetch_full_coin_list(file_path="full_crypto_list.json"):
    url = "https://api.coingecko.com/api/v3/coins/list"
    response = requests.get(url)

    if response.status_code == 200:
        coins = response.json()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(coins, f, indent=4, ensure_ascii=False)
        print(f"Saved {len(coins)} coins to {file_path}")
    else:
        print(f"Failed to fetch data: {response.status_code}")


fetch_full_coin_list()
