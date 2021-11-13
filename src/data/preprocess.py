from src.data.preprocess_en import preprocess_en_text
from src.data.preprocess_es import preprocess_es_text
from src.data.preprocess_da import preprocess_da_text
from src.data.preprocess_ar import preprocess_ar_text
from src.data.preprocess_gr import preprocess_gr_text
from src.data.preprocess_tr import preprocess_tr_text
from src.data.preprocess_hi import preprocess_hi_text
from src.data.preprocess_de import preprocess_de_text
from src.data.preprocess_it import preprocess_it_text

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
    elif lang == "gr":
        text = preprocess_gr_text(text)
    elif lang == "tr":
        text = preprocess_tr_text(text)
    elif lang == "de":
        text = preprocess_de_text(text)
    elif lang == "hi":
        text = preprocess_hi_text(text)
    elif lang == 'it':
        text = preprocess_it_text(text)
    else:
        raise ValueError(f"Unknown language {lang}")
    return text
