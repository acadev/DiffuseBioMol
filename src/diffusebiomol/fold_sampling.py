"""Deterministic fold-balanced source presentations."""
from collections import defaultdict

import numpy as np


def epoch_sources(training, corpus, epoch, seed, *, fold_sampling="source",
                  presentations_per_epoch=None):
    if fold_sampling not in ("source", "uniform_fold"):
        raise ValueError("fold_sampling must be source or uniform_fold")
    if not training:
        raise ValueError("Empty training split")
    if fold_sampling == "source":
        if presentations_per_epoch is not None:
            raise ValueError("presentations_per_epoch requires uniform_fold sampling")
        return list(training)
    count = len(training) if presentations_per_epoch is None else presentations_per_epoch
    if count < 1:
        raise ValueError("presentations_per_epoch must be positive")
    groups = defaultdict(list)
    for index in training:
        fold = corpus.entries[index].get("fold_id")
        if fold is None:
            raise ValueError(f"Corpus entry {index} lacks fold_id")
        groups[fold].append(index)
    folds = sorted(groups)
    rng = np.random.default_rng(seed + epoch)
    return [int(rng.choice(groups[folds[int(rng.integers(len(folds)))]]))
            for _ in range(count)]
