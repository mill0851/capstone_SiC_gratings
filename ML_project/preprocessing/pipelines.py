import pandas as pd
from sklearn.pipeline import Pipeline
from preprocessing.transformers import (
    SpectrumNormalizer,
    BackgroundSubtractor,
    DomainReducer,
    LinearInterpolator,
    SpectrumFlipper,
    PCAExtractor,
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
