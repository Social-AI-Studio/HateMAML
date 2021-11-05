import re


def preprocess_de_text(text):
    text = text.replace("-", " - ")

    # Dealing with Hastags
    # https://stackoverflow.com/a/29920015/5909675

    matches = re.finditer(
        ".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", text.replace("#", " ")
    )
    text = " ".join([m.group(0) for m in matches])
    # Dealing with Links
    text = re.sub(r"http\S+", " url ", text)
    text = re.sub(r"\d+", "number", text)

    # Dealing with @User
    names = re.compile("@[A-Za-z0-9_]+")
    text = re.sub(names, "user", text)
    text = re.sub(" +", " ", text)

    return text.lower()