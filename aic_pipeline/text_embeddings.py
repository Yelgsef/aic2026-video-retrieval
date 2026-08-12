from __future__ import annotations

import numpy as np
import torch


E5_MODEL_ID = "intfloat/multilingual-e5-small"
E5_ENCODER_ID = "intfloat_multilingual_e5_small"


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def robust_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    std = float(values.std())
    return (values - float(values.mean())) / (std if std >= 1e-8 else 1.0)
