import requests
import json


def fetch_and_save_crypto_list(file_path="crypto_list.json"):
    url = "https://api.coingecko.com/api/v3/coins/list"
    response = requests.get(url)
    data = response.json()

    names_and_symbols = set()

    for coin in data:
        names_and_symbols.add(coin["name"].title().strip())
        names_and_symbols.add(coin["symbol"].title().strip())

    final_list = sorted(set(names_and_symbols))

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(final_list, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    fetch_and_save_crypto_list()