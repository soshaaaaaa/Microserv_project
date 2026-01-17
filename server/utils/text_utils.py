PUNCTUATION = ".,!?;:()[]{}\"'"


def simple_tokenize(text):
    text = text.lower()
    for ch in PUNCTUATION:
        text = text.replace(ch, " ")
    return [token for token in text.split() if token]


def build_vocabulary(tokenized_docs):
    vocab = sorted({token for doc in tokenized_docs for token in doc})
    return vocab


def read_corpus(path):
    with open(path, encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]
