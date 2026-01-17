from collections import Counter

import numpy as np

from server.utils.text_utils import build_vocabulary, simple_tokenize


def tf_idf(texts):
    tokenized_docs = [simple_tokenize(text) for text in texts]
    vocab = build_vocabulary(tokenized_docs)

    vocab_index = {word: idx for idx, word in enumerate(vocab)}
    n_docs = len(texts)
    n_terms = len(vocab)

    tf = np.zeros((n_docs, n_terms), dtype=float)
    df = np.zeros(n_terms, dtype=float)

    for doc_idx, tokens in enumerate(tokenized_docs):
        counts = Counter(tokens)
        total_terms = sum(counts.values()) or 1
        for token, count in counts.items():
            term_idx = vocab_index[token]
            tf[doc_idx, term_idx] = count / total_terms
            df[term_idx] += 1

    idf = np.log((n_docs + 1) / (df + 1)) + 1.0
    tfidf = tf * idf

    return vocab, tfidf, idf
