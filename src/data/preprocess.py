from src.data.preprocess_en import preprocess_en_text

import html


def unescape_text_html(text):
    text = html.unescape(text)
    return text


def preprocess_text(text, lang):
    text = unescape_text_html(text)
    if lang == "en":
        text = preprocess_en_text(text)
    else:
        raise ValueError(f"Unknown language {lang}")
    return text
