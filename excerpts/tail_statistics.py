"""Empirical upper-tail statistic, extracted from the research implementation.

alpha is the upper-tail mass in (0, 1], not the confidence level.
values is a nonempty finite sequence of completed-episode costs.
The returned coefficients have mean 1; estimate = mean(coeff * values).
"""

from typing import Iterable

import numpy as np


def empirical_upper_tail(values: Iterable[float], alpha: float):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(-values, kind="stable")
    remaining = float(alpha) * len(values)
    coeff = np.zeros(len(values), dtype=np.float64)
    for idx in order:
        if remaining <= 1e-12:
            break
        take = min(1.0, remaining)
        coeff[idx] = take / float(alpha)
        remaining -= take
    estimate = float(np.dot(coeff, values) / len(values))
    boundary = min(
        max(int(np.ceil(float(alpha) * len(values))) - 1, 0),
        len(values) - 1,
    )
    return estimate, coeff, float(values[order[boundary]])
