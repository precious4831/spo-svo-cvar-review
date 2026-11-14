"""Beta-distribution moments used inside the bounded SVO belief.

alpha and beta are supplied positive tensors. This is a statistical
excerpt, with the parameter-producing networks and recurrent filter omitted.
"""


def beta_moments(alpha, beta):
    total = alpha + beta
    mean = alpha / total
    variance = (
        alpha * beta / (total.pow(2) * (total + 1.0))
    ).clamp_min(1e-10)
    return mean, variance, variance.sqrt()
