# Improving Named Entity Recognition for Low-Resource Languages Using Large Language Models: A Ukrainian Case Study

This repository contains experiments and evaluation pipelines for Named Entity Recognition (NER) on Ukrainian texts using large language models (LLMs). It supports fine-tuning, prompting, and post-processing for various LLMs, with a focus on low-resource language scenarios.

## 📁 Project Structure

```
NER-Ukrainian-LLMs/
│
├── data/                            # Raw and processed datasets
│   ├── bruk/                        # BRUK part of corpus 
│   ├── dev-test-split/              # Dev/test partitions
│   ├── ng/                          # NG part of corpus 
│   ├── train.csv|iob|spacy          # Train sets in various formats
│   └── test.csv|iob|spacy           # Test sets in various formats
│
├── experiments/                     # Model training and evaluation pipelines
│   ├── encoders_tuning/             # Transformer encoder-only fine-tuning + evaluation
│   ├── prompting/                   # Prompting-based methods experiments
│   ├── sft/                         # Supervised fine-tuning experiments
│   └── eda.ipynb                    # Exploratory data analysis
│
├── results/
│   ├── prompting/
│   │    ├── bronze/                 # Raw outputs
│   │    │   ├── cot/                
│   │    │   ├── few_shot/
│   │    │   └── zero_shot/
│   │    ├── silver/                 # Lightly cleaned outputs
│   │    │   ├── cot/
│   │    │   ├── few_shot/
│   │    │   └── zero_shot/
│   │    └── gold/                   # Fully cleaned and rule-filtered outputs
│   │        ├── cot/
│   │        ├── few_shot/
│   │        └── zero_shot/
│   └── sft/                         # Supervised fine-tuning results
│
├── utils/                           # Utility scripts
│   ├── ner_eval.py                  # Evaluation metrics
│   ├── basic_post_processing.py     # Silver level transformations
│   ├── advanced_post_processing.py  # Gold level transformations
│   ├── parse_input_to_csv.py        # Input format converter
│   └── __init__.py
│
├── .env                             # Environment variables (create your own)
├── .gitignore                 
└── README.md                        # Project documentation
```

---

## ⚙️ Repo Setup

1. Clone the repo:

```bash
git clone <repo-url>
cd NER-Ukrainian-LLMs
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your `.env` file.

---

## 🧪 Experiments

### 🔹 Encoder-Only Models

Located in `experiments/encoders_tuning/`. Includes fine-tuning transformer-based encoders like `roberta-large`.

### 🔹 Prompt-Based

Located in `experiments/prompting/`. Inference is done using zero-shot, few-shot, or chain-of-thought (CoT) prompts. Results are post-processed using Bronze/Silver/Gold strategies.

### 🔹 Supervised Fine-Tuning (SFT)

Located in `experiments/sft/`. Full fine-tuning of LLMs using labeled training data.

---
