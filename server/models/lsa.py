from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def lsa_transform(texts, n_components=2):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(texts)

    svd = TruncatedSVD(n_components=n_components, random_state=0)
    doc_vectors = svd.fit_transform(tfidf)
    components = svd.components_

    vocab = vectorizer.get_feature_names_out().tolist()
    explained_variance = svd.explained_variance_ratio_.tolist()

    return vocab, doc_vectors, components, explained_variance
