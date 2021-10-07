# this code uses an external library ekphrasis:
# (https://github.com/cbaziotis/ekphrasis/)
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

emoticons_processed = {}
for k, v in emoticons.items():
    emoticons_processed[k] = v.strip("<>")
emoticons = emoticons_processed
ekphrasis_text_processor_es = TextPreProcessor(
    # terms that will be normalized
    normalize=[
        "url",
        "email",
        "percent",
        "money",
        "phone",
        "user",
        "time",
        "date",
        "number",
    ],
    # terms that will be annotated
    annotate={},
    # annotate={"hashtag", "allcaps", "elongated", "repeated",
    #    'emphasis', 'censored'},
    fix_html=False,
    # fix_html=True,  # fix HTML tokens
    unpack_hashtags=False,  # perform word segmentation on hashtags
    unpack_contractions=False,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons],
)


def encode_ekphrasis_es(text):
    global ekphrasis_text_processor_en
    text = " ".join(ekphrasis_text_processor_es.pre_process_doc(text))
    text = text.replace("<user>", "user")
    text = text.replace("<url>", "url")
    text = text.replace("<email>", "email")
    text = text.replace("<percent>", "percent")
    text = text.replace("<money>", "money")
    text = text.replace("<phone>", "phone")
    text = text.replace("<time>", "time")
    text = text.replace("<date>", "date")
    text = text.replace("<number>", "number")
    return text


def preprocess_es_text(text):
    text = encode_ekphrasis_es(text)
    return text
