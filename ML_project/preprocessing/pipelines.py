import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from preprocessing.transformers import (
    SpectrumNormalizer,
    BackgroundSubtractor,
    DomainReducer,
    LinearInterpolator,
    SpectrumFlipper,
    PCAExtractor,
    MaxPeakExtractor,
    MultiPeakExtractor,
)

def build_pca_pipeline(
    K: int,
    bounds: tuple[float, float],
    n_interp: int,
    background: pd.DataFrame | None = None,
    flip: bool = False,
) -> Pipeline:
    steps = [('normalize', SpectrumNormalizer())]

    if background is not None:
        steps.append(('remove_bg', BackgroundSubtractor(background)))

    steps += [
        ('crop',        DomainReducer(bounds=bounds)),
        ('interpolate', LinearInterpolator(n_samples=n_interp)),
    ]

    if flip:
        steps.append(('flip', SpectrumFlipper()))

    steps.append(('pca', PCAExtractor(K=K)))
    return Pipeline(steps)

def build_peak_pipeline(
        bounds: tuple[float, float],
        n_interp: int,
        window: int,
        threshold: float,
        test_idx: np.ndarray | None = None,
        background: pd.DataFrame | None = None,
        flip: bool = False,
) -> Pipeline:
    steps = [('normalize', SpectrumNormalizer())]

    if background is not None:
        steps.append(('remove_bg', BackgroundSubtractor(background)))

    steps += [
        ('crop',        DomainReducer(bounds=bounds)),
        ('interpolate', LinearInterpolator(n_samples=n_interp)),
    ]

    if flip:
        steps.append(('flip', SpectrumFlipper()))

    steps.append(('peak', MaxPeakExtractor(window=window, threshold=threshold, test_idx=test_idx)))
    return Pipeline(steps)

def build_multi_peak_pipeline(
        N: int,
        bounds: tuple[float, float],
        n_interp: int,
        window: int,
        threshold: float,
        test_idx: np.ndarray | None = None,
        background: pd.DataFrame | None = None,
        flip: bool = False,
) -> Pipeline:
    steps = [('normalize', SpectrumNormalizer())]

    if background is not None:
        steps.append(('remove_bg', BackgroundSubtractor(background)))

    steps += [
        ('crop',        DomainReducer(bounds=bounds)),
        ('interpolate', LinearInterpolator(n_samples=n_interp)),
    ]

    if flip:
        steps.append(('flip', SpectrumFlipper()))

    steps.append(('peak', MultiPeakExtractor(N=N, window=window, threshold=threshold, test_idx=test_idx)))
    return Pipeline(steps)
