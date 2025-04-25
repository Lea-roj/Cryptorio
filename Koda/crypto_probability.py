from transformers import pipeline

zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def is_probably_crypto(name, context):
    candidate_labels = ["cryptocurrency", "person", "organization", "financial regulator", "author", "random word"]

    result = zero_shot_classifier(
        sequences=context,
        candidate_labels=candidate_labels,
        hypothesis_template=f"{name} is being mentioned as a type of {{}} in financial or crypto news."
    )

    scores = dict(zip(result["labels"], result["scores"]))
    crypto_score = scores.get("cryptocurrency", 0.0)

    return round(crypto_score, 4)
