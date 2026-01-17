import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer


def ensure_nltk_resources():
    nltk.download("punkt")
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("maxent_ne_chunker")
    nltk.download("words")


_stemmer = PorterStemmer()
_lemmatizer = WordNetLemmatizer()


def tokenize_text(text):
    ensure_nltk_resources()
    return word_tokenize(text)


def stem_text(text):
    ensure_nltk_resources()
    tokens = tokenize_text(text)
    return [_stemmer.stem(word) for word in tokens]


def lemmatize_text(text, pos="n"):
    ensure_nltk_resources()
    tokens = tokenize_text(text)
    return [_lemmatizer.lemmatize(word, pos=pos) for word in tokens]


def pos_tag_text(text):
    ensure_nltk_resources()
    tokens = tokenize_text(text)
    return pos_tag(tokens)


def ner_text(text):
    ensure_nltk_resources()
    tags = pos_tag_text(text)
    tree = ne_chunk(tags)

    entities = []
    for chunk in tree:
        if hasattr(chunk, "label"):
            name = " ".join(token for token, _ in chunk.leaves())
            entities.append({"entity": name, "label": chunk.label()})
    return entities
