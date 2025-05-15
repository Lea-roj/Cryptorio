import json
import re

from crypto_probability import is_probably_crypto
from global_params import *

with open(DEX_LIST_PATH, "r", encoding="utf-8") as f:
    KNOWN_DEX_NAMES = set(json.load(f))

with open(CRYPTO_LIST_PATH, "r", encoding="utf-8") as f:
    crypto_data = json.load(f)

with open(EXCHANGE_LIST_PATH, "r", encoding="utf-8") as f:
    exchange_names = set(name.lower() for name in json.load(f))


def extract_name_symbol_pairs(text, crypto_names, crypto_symbols, entity_map):
    pattern = r"(\b[A-Z][a-zA-Z0-9\s-]{2,}\b)\s*\(\s*([A-Z0-9]{2,5})\s*\)"
    for match in re.findall(pattern, text):
        name, symbol = match[0].strip(), match[1].strip()
        if name in crypto_names and symbol in crypto_symbols:
            entity_map[name] = "CRYPTO"
            entity_map[symbol] = "CRYPTO"
            print(f"[MATCHED PAIR] '{name} ({symbol})' found in text --> both labeled as CRYPTO")


def process_named_entities(doc, text, entity_map, crypto_names, crypto_symbols,
                           name_to_symbol, symbol_to_name, disambiguation_threshold):
    for ent in doc.ents:
        name = ent.text.strip()

        # Skip too short or already labeled entities
        if len(name) < 2 or name in entity_map:
            continue

        if name.lower() in KNOWN_DEX_NAMES:
            entity_map[name] = "DEX"
            print(f"[DEX MATCH] '{name}' matched known DEX list --> labeled as DEX")
            continue

        if ent.label_ == "CARDINAL":
            continue

        # Special case: PERSON label, but actually a known crypto name
        if ent.label_ == "PERSON":
            possible_symbol = name_to_symbol.get(name)
            if possible_symbol and possible_symbol in text:
                print(f"[SYMBOL OVERRIDE] '{name}' labeled as PERSON, but '{possible_symbol}' found in text --> CRYPTO")
                entity_map[name] = "CRYPTO"
            else:
                entity_map[name] = "PERSON"
            continue

        classify_entity(name, text, doc, entity_map,
                        crypto_names, crypto_symbols, ent, symbol_to_name, disambiguation_threshold)


def classify_entity(name, text, doc, entity_map,
                    crypto_names, crypto_symbols, ent, symbol_to_name, disambiguation_threshold):
    is_exact_crypto = name in crypto_names or name in crypto_symbols
    is_short = len(name) <= 3
    is_numeric = name.isnumeric()

    if is_exact_crypto:
        if is_numeric:
            full_name = symbol_to_name.get(name)
            if full_name and full_name not in text:
                print(f"[REJECTED NUMERIC] '{name}' found, but full name '{full_name}' not in text.")
                entity_map[name] = ent.label_
                return

        if any(kw in name.lower() for kw in REGULATORY_KEYWORDS):
            print(f"[REGULATORY BLOCK] '{name}' contains regulatory keyword --> labeled as ORG")
            entity_map[name] = "ORG"
            return

        if is_short and name not in TRUSTED_CRYPTO:
            context_window = [s.text for s in doc.sents if name in s.text or len(s.text.split()) > 5]
            context = " ".join(context_window[:3]) if context_window else text
            confidence = is_probably_crypto(name, context)

            if confidence >= disambiguation_threshold:
                entity_map[name] = "CRYPTO"
            else:
                # print(f"[FILTERED] '{name}' looked like crypto but got {confidence:.2f} confidence.")
                entity_map[name] = ent.label_
        else:
            entity_map[name] = "CRYPTO"
    elif name.lower() in exchange_names:
        entity_map[name] = "EXCHANGE"
    else:
        entity_map[name] = ent.label_


def fix_possessive_crypto_relationships(text, entity_map):
    # Look for patterns like "Uniswap's UNI"
    pos_pattern = r"([A-Z][a-zA-Z0-9]+)'s\s+([A-Z0-9]{2,5})"
    poss_matches = re.findall(pos_pattern, text)
    for parent, child in poss_matches:
        if parent in entity_map and child in entity_map:
            if entity_map[parent] == "CRYPTO" and entity_map[child] == "CRYPTO":
                entity_map[parent] = "ORG"
                print(f"[POSSESSIVE FIX] '{parent}' reclassified as ORG because of possessive relationship with '{child}'")
