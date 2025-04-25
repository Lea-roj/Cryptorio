from transformers import pipeline

zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def is_probably_crypto(name, context):
    candidate_labels = [
        "cryptocurrency",
        "stock",
        "fiat currency",
        "blockchain project",
        "financial institution",
        "person",
        "company",
        "government organization",
        "technology term",
        "random word"
    ]

    result = zero_shot_classifier(
        sequences=context,
        candidate_labels=candidate_labels,
        hypothesis_template=f"{name} is a type of {{}}."
    )

    scores = dict(zip(result["labels"], result["scores"]))
    crypto_score = scores.get("cryptocurrency", 0.0)

    return round(crypto_score, 4)
