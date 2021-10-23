from src.data.preprocess_en import preprocess_en_text
from src.data.preprocess_es import preprocess_es_text
from src.data.preprocess_da import preprocess_da_text
from src.data.preprocess_ar import preprocess_ar_text

import html


def unescape_text_html(text):
    text = html.unescape(text)
    return text


def preprocess_text(text, lang):
    text = unescape_text_html(text)
    if lang == "en":
        text = preprocess_en_text(text)
    elif lang == "es":
        text = preprocess_es_text(text)
    elif lang == "da":
        text = preprocess_da_text(text)
    elif lang == "ar":
        text = preprocess_ar_text(text)
    else:
        raise ValueError(f"Unknown language {lang}")
    return text
