import difflib
import json
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


entity_types = ["LOC", "ORG", "PERS",  "MON", "PCT", "DATE", "TIME", "PERIOD", "JOB", "DOC", "QUANT", "ART", "MISC"]


def find_closest_phrase(org_text, predicted_text, threshold=0.8):
    if predicted_text in org_text:
        return predicted_text

    words = org_text.split()
    predicted_len = len(predicted_text)

    best_match = None
    best_ratio = 0.0

    for start_idx in range(len(words)):
        for end_idx in range(start_idx + 1, len(words) + 1):
            candidate = " ".join(words[start_idx:end_idx])
            if abs(len(candidate) - predicted_len) <= 3:
                ratio = difflib.SequenceMatcher(None, candidate, predicted_text).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = candidate

    best_res = best_match if best_ratio >= threshold else None
    # print(predicted_text, "->", best_res, f"\n{org_text}", "-" * 100)
    return best_res


def find_closest_label(pred_label: str, threshold: float = 0.6) -> str:
    pred_label = pred_label.upper()

    vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(entity_types + [pred_label])

    similarity_scores = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
    valid_indices = np.where(similarity_scores >= threshold)[0]

    if valid_indices.size > 0:
        closest_label_index = valid_indices[np.argmax(similarity_scores[valid_indices])]
        closest_label = entity_types[closest_label_index]
        # print(f"{pred_label} -> {closest_label}")
        return closest_label
    # print("No close match found:", pred_label)


def validate_entity(label):
    try:
        if label not in entity_types:
            return find_closest_label(label)
        else:
            return label
    except ValueError:
        print(f"Validation error: {label}")


def basic_post_processing(row):
    _, org_text, _, preds = row

    cleaned = []
    for pred in json.loads(preds):
        if (new_label := validate_entity(pred["label"])) and (new_text := find_closest_phrase(org_text, pred["text"])):
            cleaned.append({"label": new_label, "text": new_text})

    deduplicated = list({tuple(d.items()): d for d in cleaned}.values())

    return json.dumps(deduplicated, indent=4, ensure_ascii=False)
