import os
import re
import json
import spacy
import pandas as pd

from spacy.tokens import DocBin
from spacy.training import Example
from spacy.scorer import Scorer


def evaluate_ner(df, is_llm=True):
    def map_predictions(text, preds):
        sorted_preds = sorted(preds, key=lambda x: len(x["text"]), reverse=True)
        used = [False] * len(text)
        results = []
        for pred in sorted_preds:
            entity, label = pred["text"], pred["label"]
            for match in re.finditer(re.escape(entity), text):
                start, end = match.span()
                if not any(used[start:end]):
                    used[start:end] = [True] * (end - start)
                    results.append((start, end, label))
                    break
        return sorted(results, key=lambda x: x[0])

    nlp = spacy.blank("en")
    pred_docs = []
    for _, row in df.iterrows():
        _, text, _, predictions = row
        if is_llm:
            predictions = map_predictions(text, json.loads(predictions))
        else:
            predictions = [(pred["start"], pred["end"], pred["label"]) for pred in json.loads(predictions)]
        doc = nlp(text)
        doc.ents = [span for start, end, label in predictions
                    if (span := doc.char_span(start, end, label=label)) is not None]
        pred_docs.append(doc)

    gold_docs = list(DocBin().from_disk("../data/test.spacy").get_docs(nlp.vocab))
    examples = [Example(pred, gold) for pred, gold in zip(pred_docs, gold_docs)]
    scores = Scorer().score(examples)
    return {k: v for k, v in scores.items() if k in {"ents_p", "ents_r", "ents_f", "ents_per_type"}}


def read_and_format_results(base_dir):
    experiments = ["zero_shot", "few_shot", "cot"]
    results = {}

    for exp in experiments:
        exp_path = os.path.join(base_dir, exp)
        for file in os.listdir(exp_path):
            if file.endswith(".csv"):
                results.setdefault(file, {"Zero-Shot": "TBD", "Few-Shot": "TBD", "Cot": "TBD"})
                df = pd.read_csv(os.path.join(exp_path, file))
                res = evaluate_ner(df, not ("gliner" in file or "NuNER" in file))
                f1 = round(res["ents_f"], 2)
                results[file][exp.replace("_", "-").title()] = f"{f1:.2f}"

    table = (
        "\\textbf{Model} & \\textbf{Zero-Shot (F1 Score)} & "
        "\\textbf{Few-Shot (F1 Score)} & \\textbf{CoT (F1 Score)} \\\\ \\hline\n"
    )
    for model, scores in results.items():
        table += (
            f"{model} & {scores['Zero-Shot']} & {scores['Few-Shot']} & "
            f"{scores['Cot']} \\\\ \\hline\n"
        )
    return table
