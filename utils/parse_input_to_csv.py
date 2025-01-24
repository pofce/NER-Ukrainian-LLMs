import hashlib
import pandas as pd
import json
import re


def parse_iob_file_to_dataframe(file_path):
    # Read the entire IOB file content
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split the content into blocks based on two or more consecutive newlines
    sentence_blocks = re.split(r'\n{2,}', content.strip())

    hashes = []
    texts = []
    all_entities = []

    for block in sentence_blocks:
        lines = block.strip().split('\n')
        sentence = []
        entities = []

        for line in lines:
            if line.strip() == "":
                continue
            word, tag = line.split()
            sentence.append(word)
            if tag.startswith("B-"):
                label = tag[2:]
                entities.append({"label": label, "text": word})
            elif tag.startswith("I-") and entities:
                # Append to the last entity if it's part of the same label
                entities[-1]["text"] += f" {word}"

        if sentence:
            sentence_text = " ".join(sentence)
            texts.append(sentence_text)
            all_entities.append(entities)

            # Generate a hash for the sentence
            sentence_hash = hashlib.sha256(sentence_text.encode('utf-8')).hexdigest()
            hashes.append(sentence_hash)

    df = pd.DataFrame({
        "hash": hashes,
        "text": texts,
        "entities": [json.dumps(e, ensure_ascii=False) for e in all_entities]  # Use json.dumps here
    })
    return df
