import re


def clean(text: str) -> str:
    text = re.sub(r"\{\{.*?\}\}", " ", str(text))  # drop template placeholders
    return re.sub(r"\s+", " ", text).strip().lower() 