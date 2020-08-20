from nltk import FreqDist
from nltk.util import ngrams
from nltk.tokenize import word_tokenize


def get_frequency_distribution(docs, n=1):
    """ Get the n-gram terms frequency distribution from a list of strings

    Paramaters:
        docs: list of strings
        n: an integer

    Returns:
        nltk.FreqDist
    """
    ngram_freq_dist = FreqDist()

    for doc in docs:
        if isinstance(doc, str):
            tokens = word_tokenize(doc)
            ngram_tokens = ngrams(tokens, n)
            ngram_freq_dist.update(ngram_tokens)

    return ngram_freq_dist


def convert_ngrams_docs(docs, n=2):
    ngrams_docs = []
    for doc in docs:
        ngrams_docs.append(' '.join(["_".join(w)
                                     for w in ngrams(word_tokenize(doc), 2)]))
    return ngrams_docs
