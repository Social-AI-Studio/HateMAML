def preprocess_en_text(text):
    return text


def preprocess_text(text, lang):
    if lang == "en":
        text = preprocess_en_text(text)
    else:
        raise ValueError(f"Unknown language {lang}")
    return text
