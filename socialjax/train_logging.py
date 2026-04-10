"""Optional TensorBoard scalar logging alongside Weights & Biases."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import jax
import numpy as np
import wandb


def maybe_create_tensorboard_writer(config: Dict[str, Any]):
    """Return a tensorboardX SummaryWriter if ``TENSORBOARD_DIR`` is set, else None."""
    path = config.get("TENSORBOARD_DIR") or ""
    if not str(path).strip():
        return None
    try:
        from tensorboardX import SummaryWriter
    except ImportError as e:
        raise ImportError(
            "TensorBoard logging needs tensorboardX. Install: pip install tensorboardX"
        ) from e
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(str(p))


def log_metrics_wandb_tensorboard(metric: Dict[str, Any], tb_writer) -> None:
    """``wandb.log`` plus optional TensorBoard scalars (host-side; use inside ``jax.debug.callback``)."""
    wandb.log(metric)
    if tb_writer is None:
        return
    step = int(
        metric.get(
            "env_step",
            metric.get("update_step", metric.get("update_steps", 0)),
        )
    )
    skip = {"Episode GIF", "update_step", "update_steps", "env_step"}
    for k, v in metric.items():
        if k in skip:
            continue
        try:
            x = jax.device_get(v)
            arr = np.asarray(x)
            if arr.shape == () and np.issubdtype(arr.dtype, np.number):
                fv = float(arr)
                if np.isfinite(fv):
                    tb_writer.add_scalar(str(k), fv, global_step=step)
        except (TypeError, ValueError):
            pass
    tb_writer.flush()
