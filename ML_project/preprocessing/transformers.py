import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from util.data_preprocessing import (
    normalize, remove_background, reduce_domain, interp_linear, extract_pca,
    max_A, max_N_A,
)


class SpectrumNormalizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        out = X.copy()
        normalize(out)
        return out

class BackgroundSubtractor(BaseEstimator, TransformerMixin):
    def __init__(self, background: pd.DataFrame):
        self.background = background.copy()
        normalize(self.background)

    def fit(self, X, y=None): return self
    def transform(self, X):
        out = X.copy()
        remove_background(out, self.background)
        return out

class DomainReducer(BaseEstimator, TransformerMixin):
    def __init__(self, bounds: tuple[float, float]):
        self.bounds = bounds

    def fit(self, X, y=None): return self
    def transform(self, X):
        out = X.copy()
        reduce_domain(self.bounds, out)
        return out

class LinearInterpolator(BaseEstimator, TransformerMixin):
    def __init__(self, n_samples: int):
        self.n_samples = n_samples

    def fit(self, X, y=None): return self
    def transform(self, X):
        return interp_linear(X, self.n_samples)

class SpectrumFlipper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X * -1.0

class PCAExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, K: int):
        self.K = K

    def fit(self, X, y=None):
        _, self.artifacts_ = extract_pca(X, self.K, artifacts=None)
        return self

    def transform(self, X):
        features, _ = extract_pca(X, self.K, artifacts=self.artifacts_)
        return features

class MaxPeakExtractor(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            window: int,
            threshold: float,
            test_idx: np.ndarray | None = None):
        self.window = window
        self.threshold = threshold
        self.test_idx = test_idx

    def fit(self, X, y=None): return self

    def transform(self, X):
        return max_A(X, self.window, self.threshold, self.test_idx)

class MultiPeakExtractor(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            N: int,
            window: int,
            threshold: float,
            test_idx: np.ndarray | None = None):
        self.N = N
        self.window = window
        self.threshold = threshold
        self.test_idx = test_idx

    def fit(self, X, y=None): return self

    def transform(self, X):
        return max_N_A(X, self.N, self.window, self.threshold, self.test_idx)
