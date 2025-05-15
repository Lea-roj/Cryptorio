import json
import re
import nltk
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import textwrap
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from global_params import *
from crypto_probability import is_probably_crypto

nltk.download('vader_lexicon')
nlp = spacy.load("en_core_web_sm")
sid = SentimentIntensityAnalyzer()
sentiment_pipeline_twitter_based = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
sentiment_pipeline_finance = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

with open(DEX_LIST_PATH, "r", encoding="utf-8") as f:
    KNOWN_DEX_NAMES = set(json.load(f))

with open(CRYPTO_LIST_PATH, "r", encoding="utf-8") as f:
    crypto_data = json.load(f)

with open(EXCHANGE_LIST_PATH, "r", encoding="utf-8") as f:
    exchange_names = set(name.lower() for name in json.load(f))


def get_top_tfidf_keywords(text, top_n=3):
    """
    Returns top-N keywords from the input text using TF-IDF.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])

    feature_names = vectorizer.get_feature_names_out()
    scores = X.toarray()[0]

    top_indices = np.argsort(scores)[::-1][:top_n]
    top_keywords = [feature_names[i] for i in top_indices]

    return top_keywords


def preprocess_text(text):
    return ' '.join(text.split())


def extract_entities(text, tfidf_keywords, disambiguation_threshold=CRYPTO_DISAMBIGUATION_THRESHOLD):
    doc = nlp(text)
    entity_map = {}

    symbol_to_name = {coin["symbol"]: coin["name"] for coin in crypto_data}
    name_to_symbol = {coin["name"]: coin["symbol"] for coin in crypto_data}
    crypto_names = set(name_to_symbol.keys())
    crypto_symbols = set(symbol_to_name.keys())

    # Extract "Name (Symbol)" pairs --> Ethereum (ETH)
    pattern = r"(\b[A-Z][a-zA-Z0-9\s-]{2,}\b)\s*\(\s*([A-Z0-9]{2,5})\s*\)"
    for match in re.findall(pattern, text):
        name, symbol = match[0].strip(), match[1].strip()
        if name in crypto_names and symbol in crypto_symbols:
            entity_map[name] = "CRYPTO"
            entity_map[symbol] = "CRYPTO"
            print(f"[MATCHED PAIR] '{name} ({symbol})' found in text --> both labeled as CRYPTO")

    for ent in doc.ents:
        name = ent.text.strip()

        # Skip too short or already labeled entities
        if len(name) < 2 or name in entity_map:
            continue

        if name.lower() in KNOWN_DEX_NAMES:
            entity_map[name] = "DEX"
            print(f"[DEX MATCH] '{name}' matched known DEX list --> labeled as DEX")
            continue

        # Skip cardinal numbers like "Four", "Two", ...
        if ent.label_ == "CARDINAL":
            continue

        # Special case: PERSON label, but actually a known crypto name
        if ent.label_ == "PERSON":
            # Check if the matching symbol is in the text --> then it's CRYPTO
            possible_symbol = name_to_symbol.get(name)
            if possible_symbol and possible_symbol in text:
                print(f"[SYMBOL OVERRIDE] '{name}' labeled as PERSON, but '{possible_symbol}' found in text --> CRYPTO")
                entity_map[name] = "CRYPTO"
                continue
            else:
                entity_map[name] = "PERSON"
                continue

        is_exact_crypto = name in crypto_names or name in crypto_symbols
        is_short = len(name) <= 3
        is_numeric = name.isnumeric()

        if is_exact_crypto:
            if is_numeric:
                full_name = symbol_to_name.get(name)
                if full_name and full_name not in text:
                    print(f"[REJECTED NUMERIC] '{name}' found, but full name '{full_name}' not in text.")
                    entity_map[name] = ent.label_
                    continue

            if any(kw in name.lower() for kw in REGULATORY_KEYWORDS):
                print(f"[REGULATORY BLOCK] '{name}' contains regulatory keyword --> labeled as ORG")
                entity_map[name] = "ORG"
                continue

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

    # Look for patterns like "Uniswap's UNI"
    pos_pattern = r"([A-Z][a-zA-Z0-9]+)'s\s+([A-Z0-9]{2,5})"
    poss_matches = re.findall(pos_pattern, text)
    for parent, child in poss_matches:
        if parent in entity_map and child in entity_map:
            if entity_map[parent] == "CRYPTO" and entity_map[child] == "CRYPTO":
                entity_map[parent] = "ORG"
                print(
                    f"[POSSESSIVE FIX] '{parent}' reclassified as ORG because of possessive relationship with '{child}'")

    entity_map = override_entities_with_person_context(text, entity_map)

    return list(entity_map.items())


def override_entities_with_person_context(text, entity_map):
    """
    Reclassifies entities as PERSON if strong personal context is detected.
    """
    person_like_patterns = [
        r"\b([A-Z][a-z]+)\s+focuses on",
        r"\b([A-Z][a-z]+)\s+is (originally )?from",
        r"\b([A-Z][a-z]+)\s+studied",
        r"\b([A-Z][a-z]+)\s+graduated from",
        r"\b([A-Z][a-z]+)\s+is a journalist",
        r"\b([A-Z][a-z]+)\s+works at"
    ]

    for pattern in person_like_patterns:
        matches = re.findall(pattern, text)
        for name in matches:
            if name in entity_map and entity_map[name] != "PERSON":
                entity_map[name] = "PERSON"
                print(f"[PERSON CONTEXT FIX] Reclassified '{name}' as PERSON based on surrounding context.")

    return entity_map


def vader_sentiment(text):
    """
    Analyzes sentiment using VADER, returns compound score and label.
    """
    scores = sid.polarity_scores(text)
    compound_score = scores['compound']

    if compound_score > VADER_POS_THRESHOLD:
        label = 'positive'
    elif compound_score < -VADER_NEG_THRESHOLD:
        label = 'negative'
    else:
        label = 'neutral'

    return scores, label


def bert_sentiment_chunks_finance(text, chunk_size=DEFAULT_CHUNK_SIZE):
    """
    Finance-specific sentiment using FinBERT. Output includes confidence.
    """
    chunks = textwrap.wrap(text, chunk_size)
    results = sentiment_pipeline_finance(chunks)

    return [
        {
            "label": result["label"].lower(),
            "score": result["score"]
        }
        for result in results
    ]


def topic_modeling(texts, n_topics=LDA_TOPICS):
    """
    Uses LDA to extract key topics from a list of texts, excluding named entities and dates to reduce noise.
    """
    filtered_texts = []

    for text in texts:
        doc = nlp(text)

        # Remove named entities
        filtered_tokens = [
            token.text for token in doc
            if not any(ent.text == token.text and ent.label_ in EXCLUDED_LABELS_FROM_TOPIC_MODELING for ent in doc.ents)
               and not token.is_stop
               and token.is_alpha
        ]

        cleaned_text = " ".join(filtered_tokens)
        filtered_texts.append(cleaned_text)

    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(filtered_texts)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)

    topics = []
    for topic in lda.components_:
        top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-LDA_TOP_WORDS:]]
        topics.append(top_words)

    return topics


def entity_level_sentiment(text, entities):
    """
    For each CRYPTO entity, run sentiment analysis on surrounding sentences.
    """
    entity_sentiments = {}
    sentences = list(nlp(text).sents)

    for name, label in entities:
        if label != "CRYPTO":
            continue

        all_sentences_where_entity_is_mentioned = [sent.text for sent in sentences if name in sent.text]

        if not all_sentences_where_entity_is_mentioned:
            continue

        combined = " ".join(all_sentences_where_entity_is_mentioned)
        sentiment_results = bert_sentiment_chunks_finance(combined)

        overall_label = Counter([r["label"] for r in sentiment_results]).most_common(1)[0][0]
        avg_conf = np.mean([r["score"] for r in sentiment_results])

        entity_sentiments[name] = {
            "sentiment": overall_label,
            "confidence": round(avg_conf, 4),
            "mentions": len(all_sentences_where_entity_is_mentioned)
        }

    return entity_sentiments


def score_cryptos(entity_sentiment):
    """
    Calculates final score for each crypto entity based on sentiment and frequency.
    """
    score_map = {}
    for name, data in entity_sentiment.items():
        label = data["sentiment"]
        conf = data["confidence"]
        mention_boost = min(0.1 * data["mentions"], 0.5)

        sentiment_weight = {
            "positive": 1,
            "neutral": 0.2,
            "negative": -1
        }[label]

        score = round((sentiment_weight * conf) + mention_boost, 4)
        score_map[name] = score

    return dict(sorted(score_map.items(), key=lambda x: x[1], reverse=True))


##############################

def analyze_text(text):
    text = preprocess_text(text)

    top_keywords = get_top_tfidf_keywords(text, top_n=TFIDF_TOP_N)
    entities = extract_entities(text, top_keywords)

    vader_scores, vader_label = vader_sentiment(text)
    bert_scores = bert_sentiment_chunks_finance(text)
    entity_sentiments = entity_level_sentiment(text, entities)
    crypto_scores = score_cryptos(entity_sentiments)

    overall = Counter([s['label'] for s in bert_scores])
    most_common = overall.most_common(1)[0][0]

    topics = topic_modeling([text])

    return {
        "entities": entities,
        "vader_sentiment": {
            "scores": vader_scores,
            "label": vader_label
        },
        "bert_sentiment": bert_scores,
        "bert_sentiment_summary": most_common,
        "entity_sentiment": entity_sentiments,
        "crypto_scores": crypto_scores,
        "topics": topics,
        "top_keywords_tfidf": top_keywords
    }
