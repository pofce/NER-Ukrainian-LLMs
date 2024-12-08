import json


def calculate_f1_score(df):
    correct = 0
    total_predicted = 0
    total_gold = 0

    for _, row in df.iterrows():
        gold_entities = json.loads(row['entities'])
        pred_entities = json.loads(row['pred'])

        gold_set = {(ent['label'], ent['text']) for ent in gold_entities}
        pred_set = {(ent['label'], ent['text']) for ent in pred_entities}

        correct += len(gold_set & pred_set)
        total_gold += len(gold_set)
        total_predicted += len(pred_set)

    precision = correct / total_predicted if total_predicted else 0.0
    recall = correct / total_gold if total_gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return precision, recall, f1
