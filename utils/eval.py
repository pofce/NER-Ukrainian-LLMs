import json
import pandas as pd
import os
from collections import defaultdict


def calculate_f1_score(df):
    correct = 0
    total_predicted = 0
    total_gold = 0

    entity_correct = defaultdict(int)
    entity_total_predicted = defaultdict(int)
    entity_total_gold = defaultdict(int)

    for _, row in df.iterrows():
        gold_entities = json.loads(row['entities'])
        pred_entities = json.loads(row['pred'])

        gold_set = {(ent['label'], ent['text']) for ent in gold_entities}
        pred_set = {(ent['label'], ent['text']) for ent in pred_entities}

        correct += len(gold_set & pred_set)
        total_gold += len(gold_set)
        total_predicted += len(pred_set)

        for label, text in gold_set:
            entity_total_gold[label] += 1
        for label, text in pred_set:
            entity_total_predicted[label] += 1
        for label, text in gold_set & pred_set:
            entity_correct[label] += 1

    precision = correct / total_predicted if total_predicted else 0.0
    recall = correct / total_gold if total_gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    entity_scores = {}
    for entity in entity_total_gold.keys() | entity_total_predicted.keys():
        e_precision = entity_correct[entity] / entity_total_predicted[entity] if entity_total_predicted[entity] else 0.0
        e_recall = entity_correct[entity] / entity_total_gold[entity] if entity_total_gold[entity] else 0.0
        e_f1 = (2 * e_precision * e_recall / (e_precision + e_recall)) if (e_precision + e_recall) else 0.0
        entity_scores[entity] = {'precision': e_precision, 'recall': e_recall, 'f1': e_f1}

    return json.dumps({'overall': {'precision': precision, 'recall': recall, 'f1': f1},
                       'per_entity': entity_scores}, indent=2, ensure_ascii=False)


def read_and_format_results(base_dir):
    experiments = ["zero_shot", "few_shot", "cot"]
    model_mappings = {
        "Llama-3b.csv": "Llama-3.2-3B-Instruct",
        "gemma_2b.csv": "Gemma-2-2B-IT",
        "gemma_9b.csv": "Gemma-9-9B-IT",
        "4o.csv": "GPT-4o",
        "phi_4.csv": "Phi-4",
        "Phi-3-mini-4k.csv": "Phi-3-mini-4k-instruct",
        "qwen_3b.csv": "Qwen-2.5-3B",
        "qwen_7b.csv": "Qwen-2.5-7B",
        "qwen_14b.csv": "Qwen-2.5-14B"
    }

    results = {model: {"Zero-Shot": "TBD", "Few-Shot": "TBD", "Cot": "TBD"} for model in model_mappings.values()}

    for experiment in experiments:
        experiment_path = os.path.join(base_dir, experiment)
        for file in os.listdir(experiment_path):

            if file in model_mappings:
                model_name = model_mappings[file]
                file_path = os.path.join(experiment_path, file)
                df = pd.read_csv(file_path)
                res = calculate_f1_score(df)

                f1 = round(json.loads(res)["overall"]["f1"], 2)
                results[model_name][experiment.replace("_", "-").title()] = f"{f1:.2f}"

    table = "\\textbf{Model} & \\textbf{Zero-Shot (F1 Score)} & \\textbf{Few-Shot (F1 Score)} & \\textbf{CoT (F1 Score)} \\\\ \\hline\n"
    for model, scores in results.items():
        table += f"{model} & {scores['Zero-Shot']} & {scores['Few-Shot']} & {scores['Cot']} \\\\ \\hline\n"

    return table
