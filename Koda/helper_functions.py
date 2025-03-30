import json

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

nltk.download('vader_lexicon')
nlp = spacy.load("en_core_web_sm")
sid = SentimentIntensityAnalyzer()
sentiment_pipeline_twitter_based = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
sentiment_pipeline_finance = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")


def get_top_tfidf_keywords(text, top_n=3):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])

    feature_names = vectorizer.get_feature_names_out()
    scores = X.toarray()[0]

    top_indices = np.argsort(scores)[::-1][:top_n]
    top_keywords = [feature_names[i] for i in top_indices]

    return top_keywords


def preprocess_text(text):
    return ' '.join(text.split())


def extract_entities(text, tfidf_keywords):
    doc = nlp(text)
    entity_map = {}

    with open("api_lists/full_crypto_list.json", "r", encoding="utf-8") as f:
        crypto_data = json.load(f)
    with open("api_lists/exchange_list.json", "r", encoding="utf-8") as f:
        exchange_names = set(name.lower() for name in json.load(f))

    crypto_names_and_symbols = set()
    for coin in crypto_data:
        crypto_names_and_symbols.add(coin["name"].lower())
        crypto_names_and_symbols.add(coin["symbol"].lower())

    for ent in doc.ents:
        name = ent.text.strip()
        name_lower = name.lower()

        if len(name_lower) < 2:
            continue

        if name_lower in crypto_names_and_symbols:
            if len(name_lower) <= 4 or name_lower in tfidf_keywords:
                entity_map[name] = "CRYPTO"
        elif name_lower in exchange_names:
            entity_map[name] = "EXCHANGE"
        else:
            entity_map[name] = ent.label_

    return list(entity_map.items())


def vader_sentiment(text):
    scores = sid.polarity_scores(text)
    label = 'positive' if scores['compound'] > 0.05 else ('negative' if scores['compound'] < -0.05 else 'neutral')
    return scores, label


def bert_sentiment_chunks_twitter(text, chunk_size=500):
    LABEL_MAP = {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive"
    }

    chunks = textwrap.wrap(text, chunk_size)
    results = sentiment_pipeline_twitter_based(chunks)

    return [
        {
            "label": LABEL_MAP[result["label"]],
            "score": result["score"]
        }
        for result in results
    ]


def bert_sentiment_chunks_finance(text, chunk_size=500):
    chunks = textwrap.wrap(text, chunk_size)
    results = sentiment_pipeline_finance(chunks)

    return [
        {
            "label": result["label"].lower(),
            "score": result["score"]
        }
        for result in results
    ]


def topic_modeling(texts, n_topics=1):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)

    topics = []
    for topic in lda.components_:
        top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
        topics.append(top_words)
    return topics


def analyze_text(text):
    text = preprocess_text(text)

    top_keywords = get_top_tfidf_keywords(text, top_n=10)
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


def entity_level_sentiment(text, entities):
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
