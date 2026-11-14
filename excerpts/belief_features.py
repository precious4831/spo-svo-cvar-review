"""Selected non-parametric expressions from interaction attention.

mu and sigma are supplied posterior moments in degrees, shaped (B, N).
prior_sigma > 0; evidence_strength is supplied externally, shaped (B, N).
These functions do not estimate beliefs or construct attention tokens.
"""

import numpy as np
import torch


def posterior_features(svo_mu, svo_sigma, prior_sigma):
    mu_rad = svo_mu * (np.pi / 180.0)
    svo_input = torch.stack([
        torch.cos(mu_rad),
        torch.sin(mu_rad),
        svo_sigma / prior_sigma,     # 归一化 (不再硬编码25.0)
        torch.log(svo_sigma + 1e-3),      # log尺度特征
    ], dim=-1)                             # (B, N, 4)
    return svo_input


def posterior_confidence(svo_sigma, prior_sigma, evidence_strength):
    uncertainty_ratio = svo_sigma / prior_sigma
    confidence = (
        1.0 - uncertainty_ratio.square()
    ).clamp(0.0, 1.0) * evidence_strength.clamp(0.0, 1.0)
    return confidence
