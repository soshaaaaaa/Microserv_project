from collections import Counter

import numpy as np

from server.utils.text_utils import build_vocabulary, simple_tokenize


def bag_of_words(texts):
    tokenized_docs = [simple_tokenize(text) for text in texts]
    vocab = build_vocabulary(tokenized_docs)
    vocab_index = {word: idx for idx, word in enumerate(vocab)}
    matrix = np.zeros((len(texts), len(vocab)), dtype=int)

    for doc_idx, tokens in enumerate(tokenized_docs):
        counts = Counter(tokens)
        for token, count in counts.items():
            matrix[doc_idx, vocab_index[token]] = count

    return vocab, matrix
