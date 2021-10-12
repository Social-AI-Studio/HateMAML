import re


def preprocess_text_en(text):

    text=text.replace('-',' - ')
    text = text.replace('.', ' ')

    # Dealing with Hastags
    # https://stackoverflow.com/a/29920015/5909675
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', text.replace("#", " "))
    text=" ".join([m.group(0) for m in matches])

    # Dealing with Links
    text = re.sub(r'http\S+', ' <Link> ', text)

    # remove numbers
    text = re.sub(r'\d+', '<NUM>', text)



    # Dealing with @User
    names = re.compile('@[A-Za-z0-9_]+')
    text = re.sub(names, ' @USER ', text)

    text=re.sub(' +',' ',text)

    return text.lower()

def preprocess_text_es(text):

    text = text.replace('-', ' - ')

    # Dealing with Hastags
    # https://stackoverflow.com/a/29920015/5909675

    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', text.replace("#", " "))
    text=" ".join([m.group(0) for m in matches])

    # Dealing with Links
    text = re.sub(r'http\S+', ' <Link> ', text)

    text = re.sub(r'\d+', '<NUM>', text)

    # Dealing with @User
    names = re.compile('@[A-Za-z0-9_]+')
    text = re.sub(names, '@USER', text)

    text = re.sub(' +', ' ', text)

    return text.lower()

#print(preprocess_text_en('My name is @Eshaan #LiveLife #Blessed #OMG'))





