import json


def load_secrets(root="../", file_path="secrets.json"):
    with open(root + file_path, 'r') as file:
        secrets = json.load(file)
    return secrets

