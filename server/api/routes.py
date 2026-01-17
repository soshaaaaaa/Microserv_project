from fastapi import APIRouter
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from server.models.bag_of_words import bag_of_words
from server.models.lsa import lsa_transform
from server.models.tfidf import tf_idf
from server.models.word2vec_svd import word2vec_svd
from server.preprocessing.nltk_processing import (
    analysis_pipeline,
    corpora_info,
    freq_dist,
    lemmatize_text,
    ner_text,
    ngram_tokens,
    pos_tag_text,
    remove_stopwords,
    sentence_tokens,
    stem_text,
    word_tokens,
)

router = APIRouter()


class CorpusRequest(BaseModel):
    texts: list = [
        "Я люблю NLP и обработку текста",
        "FastAPI помогает быстро сделать API",
        "NLP извлекает смысл из документов",
    ]


class LsaRequest(BaseModel):
    texts: list = [
        "I love natural language processing",
        "FastAPI makes APIs simple",
        "NLP finds patterns in documents",
    ]
    n_components: int = 2


class Word2VecRequest(BaseModel):
    texts: list = [
        "I love natural language processing",
        "FastAPI makes APIs simple",
        "NLP finds patterns in documents",
    ]
    window_size: int = 2
    n_components: int = 2


class TextRequest(BaseModel):
    text: str = "FastAPI and NLP are useful"
    language: str = "english"
    pos: str = "n"
    n: int = 2
    top_n: int = 10


@router.get("/")
def root():
    return RedirectResponse(url="/docs")


@router.post("/bag-of-words")
def bag_of_words_endpoint(payload: CorpusRequest):
    vocab, matrix = bag_of_words(payload.texts)
    return {"vocab": vocab, "matrix": matrix.tolist()}


@router.post("/tf-idf")
def tf_idf_endpoint(payload: CorpusRequest):
    vocab, matrix, idf = tf_idf(payload.texts)
    return {"vocab": vocab, "matrix": matrix.tolist(), "idf": idf.tolist()}


@router.post("/lsa")
def lsa_endpoint(payload: LsaRequest):
    vocab, doc_vectors, components, explained_variance = lsa_transform(
        payload.texts, payload.n_components
    )
    return {
        "vocab": vocab,
        "doc_vectors": doc_vectors.tolist(),
        "components": components.tolist(),
        "explained_variance": explained_variance,
    }


@router.post("/word2vec")
def word2vec_endpoint(payload: Word2VecRequest):
    vocab, vectors = word2vec_svd(
        payload.texts,
        window_size=payload.window_size,
        n_components=payload.n_components,
    )
    return {"vocab": vocab, "vectors": vectors.tolist()}


@router.post("/text_nltk/tokenize")
def tokenize_endpoint(payload: TextRequest):
    return {"tokens": word_tokens(payload.text, payload.language)}


@router.post("/text_nltk/sent_tokenize")
def sent_tokenize_endpoint(payload: TextRequest):
    return {"sentences": sentence_tokens(payload.text, payload.language)}


@router.post("/text_nltk/stopwords")
def stopwords_endpoint(payload: TextRequest):
    return {"filtered": remove_stopwords(payload.text, payload.language)}


@router.post("/text_nltk/stem")
def stem_endpoint(payload: TextRequest):
    return {"stems": stem_text(payload.text, payload.language)}


@router.post("/text_nltk/lemmatize")
def lemmatize_endpoint(payload: TextRequest):
    return {"lemmas": lemmatize_text(payload.text, payload.pos)}


@router.post("/text_nltk/pos")
def pos_endpoint(payload: TextRequest):
    tags = pos_tag_text(payload.text, payload.language)
    return {"pos": [{"token": token, "tag": tag} for token, tag in tags]}


@router.post("/text_nltk/ner")
def ner_endpoint(payload: TextRequest):
    return {"entities": ner_text(payload.text)}


@router.post("/text_nltk/freq")
def freq_endpoint(payload: TextRequest):
    return {"freq": freq_dist(payload.text, payload.language, payload.top_n)}


@router.post("/text_nltk/ngrams")
def ngrams_endpoint(payload: TextRequest):
    return {"ngrams": ngram_tokens(payload.text, payload.n, payload.language)}


@router.post("/text_nltk/pipeline")
def pipeline_endpoint(payload: TextRequest):
    return analysis_pipeline(payload.text)


@router.post("/text_nltk/corpora")
def corpora_endpoint():
    return corpora_info()
