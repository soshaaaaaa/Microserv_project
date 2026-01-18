from fastapi import APIRouter
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from server.models.bag_of_words import bag_of_words
from server.models.lsa import lsa_transform
from server.models.tfidf import tf_idf
from server.models.word2vec_svd import word2vec_svd
from server.preprocessing.nltk_processing import (
    lemmatize_text,
    ner_text,
    pos_tag_text,
    stem_text,
    tokenize_text,
)
from server.utils.text_utils import read_corpus

router = APIRouter()

DEFAULT_CORPUS = read_corpus()
DEFAULT_TEXT = (
    "I was born in the year 1632, in the city of York, of a good family, though not "
    "of that country; my father being a foreigner of Bremen, who settled first at Hull. "
    "He got a good estate by merchandise, and leaving off his trade, lived afterwards "
    "at York, from whence he had married my mother, whose relations were named Robinson."
)


class CorpusRequest(BaseModel):
    texts: list = DEFAULT_CORPUS


class LsaRequest(BaseModel):
    texts: list = DEFAULT_CORPUS
    n_components: int = 2


class Word2VecRequest(BaseModel):
    texts: list = DEFAULT_CORPUS
    window_size: int = 2
    n_components: int = 2


class TextRequest(BaseModel):
    text: str = DEFAULT_TEXT
    pos: str = "n"


@router.get("/")
def root():
    return RedirectResponse(url="/docs")


@router.post("/bag-of-words")
def bag_of_words_endpoint(payload: CorpusRequest = CorpusRequest()):
    vocab, matrix = bag_of_words(payload.texts)
    return {"vocab": vocab, "matrix": matrix.tolist()}


@router.post("/tf-idf")
def tf_idf_endpoint(payload: CorpusRequest = CorpusRequest()):
    vocab, matrix, _ = tf_idf(payload.texts)
    return {"vocab": vocab, "tfidf": matrix.tolist()}


@router.post("/lsa")
def lsa_endpoint(payload: LsaRequest = LsaRequest()):
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
def word2vec_endpoint(payload: Word2VecRequest = Word2VecRequest()):
    vocab, vectors = word2vec_svd(
        payload.texts,
        window_size=payload.window_size,
        n_components=payload.n_components,
    )
    return {"vocab": vocab, "vectors": vectors.tolist()}


@router.post("/text_nltk/tokenize")
def tokenize_endpoint(payload: TextRequest = TextRequest()):
    return {"tokens": tokenize_text(payload.text)}


@router.post("/text_nltk/stem")
def stem_endpoint(payload: TextRequest = TextRequest()):
    return {"stems": stem_text(payload.text)}


@router.post("/text_nltk/lemmatize")
def lemmatize_endpoint(payload: TextRequest = TextRequest()):
    return {"lemmas": lemmatize_text(payload.text, payload.pos)}


@router.post("/text_nltk/pos")
def pos_endpoint(payload: TextRequest = TextRequest()):
    tags = pos_tag_text(payload.text)
    return {"pos": [{"token": token, "tag": tag} for token, tag in tags]}


@router.post("/text_nltk/ner")
def ner_endpoint(payload: TextRequest = TextRequest()):
    return {"entities": ner_text(payload.text)}
