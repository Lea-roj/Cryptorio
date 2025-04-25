import requests
import json


def fetch_full_coin_list(file_path="full_crypto_list_coinmarketcap.json"):
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/map"
    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": "b0b12977-3dd7-41fa-81e1-7a8796d46f00"
    }
    params = {
        "listing_status": "active",
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        coins = data.get("data", [])

        cleaned = [{"name": coin["name"], "symbol": coin["symbol"], "slug": coin["slug"]} for coin in coins]

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=4, ensure_ascii=False)
    else:
        print(f"Failed to fetch data: {response.status_code}")
        print(response.text)


fetch_full_coin_list()
