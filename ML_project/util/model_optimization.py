import numpy as np
from sklearn.model_selection import KFold


def generate_kfold(
        dataset, 
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        test_ratio: float = 0.15
        ):
    
    """
    Returns a KFold iterable based on the indices of the 
    dataset object passed to the function

    Returns:
        iterable: KFold iterable that stores n_split tuples (train_idx, val_idx)
    """

    # Get dataset indices
    N = len(dataset)
    idx = np.arange(N)

    # Shuffle before KFOLD
    if shuffle:
        rng = np.random.RandomState(random_state)
        rng.shuffle(idx)

    # Set aside indices for testing
    split = int(test_ratio * len(idx))
    test_indices = idx[:split]
    kfold_indices = idx[split:]

    # Generate KFOLD iterable
    kf = KFold(
        n_splits=n_splits,
        shuffle=False
    )

    # Split indices
    iterable = kf.split(kfold_indices)

    return iterable, test_indices



