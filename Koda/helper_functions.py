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


def is_crypto(name):
    with open("api_lists/crypto_list.json", "r", encoding="utf-8") as f:
        crypto_names = set(json.load(f))

    return name.strip() in crypto_names


def get_custom_label(name):
    with open("api_lists/crypto_list.json", "r", encoding="utf-8") as f:
        crypto_names = set(json.load(f))

    with open("api_lists/exchange_list.json", "r", encoding="utf-8") as f:
        exchange_names = set(json.load(f))

    name_clean = name.strip()
    if name_clean in crypto_names:
        return "CRYPTO"
    elif name_clean in exchange_names:
        return "EXCHANGE"
    else:
        return None


def extract_entities(text):
    doc = nlp(text)
    entity_map = {}

    for ent in doc.ents:
        name = ent.text.strip()
        custom_label = get_custom_label(name)
        label = custom_label if custom_label else ent.label_

        if name not in entity_map or custom_label:
            entity_map[name] = label

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

    entities = extract_entities(text)
    vader_scores, vader_label = vader_sentiment(text)
    bert_scores = bert_sentiment_chunks_finance(text)

    overall = Counter([s['label'] for s in bert_scores])
    most_common = overall.most_common(1)[0][0]

    topics = topic_modeling([text])
    top_keywords = get_top_tfidf_keywords(text)

    return {
        "entities": entities,
        "vader_sentiment": {"scores": vader_scores, "label": vader_label},
        "bert_sentiment": bert_scores,
        "bert_sentiment_summary": most_common,
        "topics": topics,
        "top_keywords_tfidf": top_keywords
    }
