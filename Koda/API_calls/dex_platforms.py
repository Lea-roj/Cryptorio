import requests
import json


def fetch_dex_names_from_defillama(save_path="dex_list.json"):
    url = "https://api.llama.fi/protocols"
    response = requests.get(url)
    data = response.json()

    dex_names = {
        entry['name'].strip().lower()
        for entry in data
        if 'category' in entry and 'dex' in entry['category'].lower()
    }

    dex_names = sorted(dex_names)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(dex_names, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(dex_names)} DEX names to {save_path}")
    return dex_names


if __name__ == "__main__":
    fetch_dex_names_from_defillama()
