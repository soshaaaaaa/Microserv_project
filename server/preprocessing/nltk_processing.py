import nltk
from nltk import ne_chunk, pos_tag, sent_tokenize, word_tokenize
from nltk.corpus import brown, gutenberg, stopwords
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.util import ngrams

PUNCTUATION = ".,!?;:()[]{}\"'"


def ensure_nltk_resources():
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("averaged_perceptron_tagger_ru")
    nltk.download("maxent_ne_chunker")
    nltk.download("maxent_ne_chunker_tab")
    nltk.download("words")
    nltk.download("tagsets")


_stemmer_en = PorterStemmer()
_stemmer_ru = SnowballStemmer("russian")
_lemmatizer = WordNetLemmatizer()


def word_tokens(text, language="english"):
    ensure_nltk_resources()
    return word_tokenize(text, language=language)


def sentence_tokens(text, language="english"):
    ensure_nltk_resources()
    return sent_tokenize(text, language=language)


def remove_stopwords(text, language="english"):
    ensure_nltk_resources()
    tokens = word_tokens(text, language)
    stop_words = set(stopwords.words(language))
    return [word for word in tokens if word.lower() not in stop_words]


def stem_text(text, language="english"):
    ensure_nltk_resources()
    tokens = word_tokens(text, language)
    stemmer = _stemmer_ru if language == "russian" else _stemmer_en
    return [stemmer.stem(word) for word in tokens]


def lemmatize_text(text, pos=""):
    ensure_nltk_resources()
    tokens = word_tokens(text, "english")
    return [_lemmatizer.lemmatize(word, pos=pos) for word in tokens]


def pos_tag_text(text, language="english"):
    ensure_nltk_resources()
    tokens = word_tokens(text, language)
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


def freq_dist(text, language="english", top_n=10):
    ensure_nltk_resources()
    tokens = remove_stopwords(text, language)
    dist = FreqDist(tokens)
    return dist.most_common(top_n)


def ngram_tokens(text, n=2, language="english"):
    ensure_nltk_resources()
    tokens = word_tokens(text, language)
    return list(ngrams(tokens, n))


def analysis_pipeline(text):
    ensure_nltk_resources()
    sentences = sentence_tokens(text, "english")
    words = word_tokens(text, "english")
    stop_words = set(stopwords.words("english"))

    clean_words = [
        word.lower()
        for word in words
        if word.lower() not in stop_words and word not in PUNCTUATION
    ]
    lemmas = [_lemmatizer.lemmatize(word) for word in clean_words]
    dist = FreqDist(lemmas)

    lexical_diversity = len(set(lemmas)) / len(lemmas)

    return {
        "num_sentences": len(sentences),
        "num_words": len(words),
        "num_clean_words": len(clean_words),
        "top_10_words": dist.most_common(10),
        "lexical_diversity": lexical_diversity,
    }


def corpora_info():
    nltk.download("gutenberg")
    nltk.download("brown")

    gutenberg_files = gutenberg.fileids()
    emma_len = len(gutenberg.words("austen-emma.txt"))

    news_len = len(brown.words(categories="news"))
    fiction_len = len(brown.words(categories="fiction"))

    return {
        "gutenberg_files": gutenberg_files[:5],
        "emma_words": emma_len,
        "brown_news_words": news_len,
        "brown_fiction_words": fiction_len,
    }
