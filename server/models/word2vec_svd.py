import numpy as np
from sklearn.decomposition import TruncatedSVD

from server.utils.text_utils import build_vocabulary, simple_tokenize


def _build_cooccurrence(tokenized_docs, vocab_index, window_size):
    vocab_size = len(vocab_index)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=float)

    for tokens in tokenized_docs:
        indices = [vocab_index[token] for token in tokens]
        for i, center in enumerate(indices):
            start = max(0, i - window_size)
            end = min(len(indices), i + window_size + 1)
            for j in range(start, end):
                if j == i:
                    continue
                context = indices[j]
                co_matrix[center, context] += 1.0

    return co_matrix


def word2vec_svd(texts, window_size=2, n_components=10):
    tokenized_docs = [simple_tokenize(text) for text in texts]
    vocab = build_vocabulary(tokenized_docs)
    vocab_index = {word: idx for idx, word in enumerate(vocab)}
    co_matrix = _build_cooccurrence(tokenized_docs, vocab_index, window_size)

    svd = TruncatedSVD(n_components=n_components, random_state=0)
    word_vectors = svd.fit_transform(co_matrix)

    return vocab, word_vectors
