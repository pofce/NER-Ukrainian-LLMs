import re
import json
import stanza
import logging
import warnings

warnings.simplefilter("ignore", category=FutureWarning)

nlp = stanza.Pipeline('uk', verbose=False)


def contains_pronoun_stanza(text):
    doc = nlp(text)
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.upos == 'PRON':  # Checks if the word is a pronoun
                return True
    return False


def clean_PERS(entity: str) -> str | None:
    """
    PERS Rules:
      1) Every word starts with a capital letter (no all-lower or digits).
      2) No numbers.
      3) Don't have quotes.
      4) Not in ("Він", "Вона").
    """
    text = entity.strip()
    # Rule 3: Don't have quotes.
    if re.search(r'[«»]', text):
        return None

    # Rule 2: No digits
    if any(ch.isdigit() for ch in text):
        return None

    tokens = text.split()
    # Rule 1: Every token should start with a capital letter
    for token in tokens:
        if not token or not token[0].isupper():
            return None

    # Rule 4: Exclude if the word is a pronoun
    if contains_pronoun_stanza(text):
        return None

    return entity


def clean_DOC(entity: str) -> str | None:
    """
    DOC Rules:
      1) Not one single lowercased word.
         (If there's exactly one token and it's all-lower, discard.)
    """
    text = entity.strip()
    tokens = text.split()

    # If there's only one token and it is all lowercase => discard
    if len(tokens) == 1 and tokens[0].islower():
        return None

    return text


def clean_QUANT(entity: str) -> str | None:
    """
    QUANT Rules:
      1) Not '%'
         (Discard if the entire text is just '%'.)
    """
    text = entity.strip()
    if "%" in text:
        return None
    return text


def clean_ART(entity: str) -> str | None:
    """
    ART Rules:
      1) Remove quotes.
      2) No pure numeric.
      3) Starts from capital letter.
    """
    text = entity.strip()

    # 1) Remove quotes
    text = re.sub(r'^[\'"“”«»]+|[\'"“”«»]+$', '', text)
    text = text.strip()

    # 2) No pure numeric
    if all(ch.isdigit() for ch in text if ch.isalnum()):
        return None

    # 3) Starts from capital letter
    if not text or not text[0].isupper():
        return None

    return text


def clean_TIME(entity: str) -> str | None:
    """
    TIME Rules:
      1) Only limited punctuation (':' or '.').
      2) Not '%'.
         (Discard if any '%' is present.)
    """
    text = entity.strip()

    # 2) Not '%'
    if '%' in text:
        return None

    # 1) Only allow letters, digits, spaces, ':' or '.' as punctuation
    #    If we find other punctuation, discard.
    if re.search(r'[^A-Za-zА-Яа-я0-9:\.\s]', text):
        return None

    return text


def clean_JOB(entity: str) -> str | None:
    """
    JOB Rules:
      1) No numbers.
    """
    text = entity.strip()
    if any(ch.isdigit() for ch in text):
        return None
    return text


def clean_MISC(entity: str) -> str | None:
    """
    MISC Rules:
      1) Is not one single lowercased word without numbers/punctuation.
         (If exactly one token, all-lower, purely alphabetic => discard.)
    """
    text = entity.strip()
    tokens = text.split()

    if len(tokens) == 1:
        token = tokens[0]
        # purely alphabetic and all-lower => discard
        if token.isalpha() and token.islower():
            return None

    return text


def clean_PCT(entity: str) -> str | None:
    """
    PCT Rules:
      1) Must contain '%' or 'відсот'/'процен' etc.
      2) Not a single number.
      3) Not a single word (must be at least two tokens or a token with non-digit text).
    """
    text = entity.strip().lower()

    # 1) Must contain '%' or partial match "відсот" or "процен"
    if '%' not in text and not re.search(r'(відсот|процен)', text):
        return None

    # 2) Not a single number
    if text.isdigit():
        return None

    # 3) Not a single word:
    tokens = text.split()
    if len(tokens) == 1:
        # If it's a single token, it should have more than just digits
        # e.g., "10%" is still one token, but valid because it has '%'
        # So let's allow single-token "10%" but not "10" alone.
        if not re.search(r'\d+%', text):
            return None

    # Return the original (mixed case) version
    return entity.strip()


def clean_ORG(entity: str) -> str | None:
    """
    ORG Rules:
      1) At least one word starts with a capital letter.
      2) Avoid pure numeric.
    """
    text = entity.strip()

    tokens = text.split()
    # Check that at least one token starts with a capital letter
    if not any(t and t[0].isupper() for t in tokens):
        return None

    # Avoid pure numeric
    # If after removing all spaces the entire thing is just digits
    if all(ch.isdigit() for ch in text.replace(" ", "")):
        return None

    return text


def clean_LOC(entity: str) -> str | None:
    """
    LOC Rules:
      1) Must contain at least one capitalized word.
      2) Avoid pure numeric.
    """
    text = entity.strip()
    tokens = text.split()

    if entity.upper() == entity:
        return None

    if not any(t and t[0].isupper() for t in tokens):
        return None

    if all(ch.isdigit() for ch in text.replace(" ", "")):
        return None

    return text


def clean_PERIOD(entity: str) -> str | None:
    """
    PERIOD Rules:
      1) From punctuation, only [.`-,:] are allowed.
         (If there's punctuation outside these characters, discard.)
    """
    text = entity.strip()

    # If we find punctuation that isn't one of . - , : `, discard
    # Note: we allow letters, digits, spaces, and these punctuation marks
    if re.search(r'[^A-Za-zА-Яа-я0-9\s\.\-\:\,`]', text):
        return None

    return text


def clean_MON(entity: str) -> str | None:
    """
    MON Rules:
      1) No single-word all-lowercase prediction without numbers.
      2) No solo numeric prediction.
    """
    text = entity.strip()
    tokens = text.split()

    # 1) If exactly one token, all-lowercase, no digits => discard
    if len(tokens) == 1:
        token = tokens[0]
        if token.islower() and not any(ch.isdigit() for ch in token):
            return None

    # 2) If exactly one token that is purely numeric => discard
    if len(tokens) == 1 and tokens[0].isdigit():
        return None

    return text


def clean_DATE(entity: str) -> str | None:
    """
    DATE Rules:
      1) no '%'.
         (Discard if text contains '-' or '%'.)
    """
    text = entity.strip()
    if '%' in text:
        return None
    return text


def clean_entity(entity_type: str, entity: str) -> str | None:
    """
    Dispatcher to call the right cleaner based on entity_type.
    Returns cleaned text or None if it fails the rules.
    """
    match entity_type:
        case "PERS": return clean_PERS(entity)
        case "DOC": return clean_DOC(entity)
        case "QUANT": return clean_QUANT(entity)
        case "ART": return clean_ART(entity)
        case "TIME": return clean_TIME(entity)
        case "JOB": return clean_JOB(entity)
        case "MISC": return clean_MISC(entity)
        case "PCT": return clean_PCT(entity)
        case "ORG": return clean_ORG(entity)
        case "LOC": return clean_LOC(entity)
        case "PERIOD": return clean_PERIOD(entity)
        case "MON": return clean_MON(entity)
        case "DATE": return clean_DATE(entity)


def advanced_post_processing(preds: str) -> str | None:
    return json.dumps(
        [pred for pred in json.loads(preds) if clean_entity(pred["label"], pred["text"])],
        ensure_ascii=False)
