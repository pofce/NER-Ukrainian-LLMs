import json
import os


def read_iob_file(file_path):
    sentences = []
    current_sentence = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line == "":
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                token, tag = line.split()
                current_sentence.append((token, tag))
        if current_sentence:
            sentences.append(current_sentence)
    return sentences


def iob_to_gliner_format(sentences):
    data = []
    for sentence in sentences:
        tokenized_text = [token for token, tag in sentence]
        ner = []
        entity_start = None
        entity_label = None

        for i, (token, tag) in enumerate(sentence):
            if tag.startswith("B-"):
                if entity_start is not None:
                    ner.append([entity_start, i - 1, entity_label])
                entity_start = i
                entity_label = tag[2:]
            elif tag.startswith("I-") and entity_label == tag[2:]:
                continue
            else:
                if entity_start is not None:
                    ner.append([entity_start, i - 1, entity_label])
                    entity_start = None
                    entity_label = None

        if entity_start is not None:
            ner.append([entity_start, len(sentence) - 1, entity_label])

        data.append({
            "tokenized_text": tokenized_text,
            "ner": ner
        })
    return data


def convert_iob_to_gliner(input_file, output_file):
    sentences = read_iob_file(input_file)
    data = iob_to_gliner_format(sentences)
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    input_files = ["train.iob", "dev.iob"]
    for input_file in input_files:
        output_file = f"{os.path.splitext(input_file)[0]}_gliner.json"
        convert_iob_to_gliner(input_file, output_file)
        print(f"Converted {input_file} to {output_file}")
