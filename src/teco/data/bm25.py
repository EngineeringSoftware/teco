import warnings
from typing import Optional

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer


class BM25(BaseEstimator):
    """
    Okapi BM25 scoring function.

    Adapted from: https://gist.github.com/koreyou/f3a8a0470d32aa56b32f198f49a9f2b8
    """

    def __init__(self, b=0.75, k1=1.6, vectorizer_kwargs: Optional[dict] = None):
        if vectorizer_kwargs is None:
            vectorizer_kwargs = {}
        if "norm" in vectorizer_kwargs:
            warnings.warn(
                f"BM25 will ignore the norm={vectorizer_kwargs['norm']} argument passed to the vectorizer"
            )
        vectorizer_kwargs["norm"] = None
        if "smooth_idf" in vectorizer_kwargs:
            warnings.warn(
                f"BM25 will ignore the smooth_idf={vectorizer_kwargs['smooth_idf']} argument passed to the vectorizer"
            )
        vectorizer_kwargs["smooth_idf"] = False
        self.vectorizer = TfidfVectorizer(**vectorizer_kwargs)
        self.b = b
        self.k1 = k1

    def fit(self, X):
        """Fit IDF to documents X"""
        y = self.vectorizer.fit_transform(X)
        self.avdl = y.sum(1).mean()

    def transform(self, q, X):
        """Calculate BM25 score between query q and documents X"""
        b, k1, avdl = self.b, self.k1, self.avdl

        # apply vectorizer
        X = self.vectorizer.transform(X)
        len_X = X.sum(1).A1
        (q,) = self.vectorizer.transform([q])
        assert sparse.isspmatrix_csr(q)

        # convert to csc for better column slicing
        X = X.tocsc()[:, q.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.0
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)
        return (numer / denom).sum(1).A1
