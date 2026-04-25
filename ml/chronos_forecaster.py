"""
Share-Forge - Chronos foundation-model forecaster.

Wraps Amazon's Chronos-T5 family (Apache 2.0). Pretrained on 100K+ real
time-series, Chronos delivers strong zero-shot forecasts on financial data
without ever seeing TATAGOLD.NS during pretraining. Optional fine-tuning
(see `train_chronos_finetune.py`) further specialises it.

Available variants:
  - amazon/chronos-t5-tiny   (~8M params)   default, fast on Apple Silicon
  - amazon/chronos-t5-mini   (~20M params)
  - amazon/chronos-t5-small  (~46M params)
  - amazon/chronos-t5-base   (~200M params)

The wrapper exposes the same interface as our LSTM forecaster: given a
1-D context of close prices and a horizon, returns mean + percentile
bands for the projected price path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

DEFAULT_MODEL_ID = "amazon/chronos-t5-tiny"
DEFAULT_LOCAL_DIR = Path(__file__).parent.parent / "checkpoints" / "chronos"


def auto_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


@dataclass
class ChronosResult:
    horizon: int
    method: str
    mean: List[float]
    p05: List[float]
    p25: List[float]
    p50: List[float]
    p75: List[float]
    p95: List[float]


class ChronosForecaster:
    """
    Thin wrapper over `chronos.ChronosPipeline`.

    The pipeline tokenises continuous values into a vocabulary of bins, runs
    a T5 encoder-decoder, and decodes back to continuous values. Sampling
    `num_samples` paths per call gives us the percentile bands.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        torch_dtype: Optional[str] = None,
        local_path: Optional[str] = None,
    ):
        try:
            import torch
            from chronos import ChronosPipeline
        except ImportError as e:
            raise RuntimeError(
                "chronos-forecasting not installed. Run "
                "`pip install chronos-forecasting`"
            ) from e

        resolved_device = auto_device() if device == "auto" else device
        dtype = torch.float32
        if torch_dtype == "fp16":
            dtype = torch.float16
        elif torch_dtype == "bf16":
            dtype = torch.bfloat16

        load_target = local_path or model_id
        self.pipeline = ChronosPipeline.from_pretrained(
            load_target,
            device_map=resolved_device,
            torch_dtype=dtype,
        )
        self.model_id = model_id
        self.device = resolved_device
        self.local_path = local_path

    def predict(
        self,
        prices: np.ndarray,
        horizon: int,
        n_samples: int = 200,
    ) -> ChronosResult:
        """
        Predict next `horizon` close prices given a context of recent closes.

        Args:
            prices:    1-D numpy array of past close prices (e.g. last 64 bars).
            horizon:   Number of forward steps to forecast.
            n_samples: Number of Monte Carlo samples drawn from the model's
                       learned distribution.

        Returns:
            ChronosResult with mean and percentile arrays of length `horizon`.
        """
        import torch

        prices = np.asarray(prices, dtype=np.float32)
        if prices.ndim != 1 or prices.size < 8:
            raise ValueError(f"Need a 1-D context with >=8 bars, got shape {prices.shape}")

        context = torch.tensor(prices, dtype=torch.float32)
        try:
            forecast = self.pipeline.predict(
                context,
                prediction_length=horizon,
                num_samples=n_samples,
            )
        except TypeError:
            forecast = self.pipeline.predict(
                context,
                prediction_length=horizon,
            )

        if hasattr(forecast, "cpu"):
            arr = forecast[0].cpu().float().numpy()
        elif isinstance(forecast, (list, tuple)):
            arr = np.asarray(forecast[0])
        else:
            arr = np.asarray(forecast)
            if arr.ndim == 3:
                arr = arr[0]

        return ChronosResult(
            horizon=horizon,
            method=("chronos_ft" if self.local_path else "chronos_zs"),
            mean=arr.mean(axis=0).tolist(),
            p05=np.percentile(arr, 5, axis=0).tolist(),
            p25=np.percentile(arr, 25, axis=0).tolist(),
            p50=np.percentile(arr, 50, axis=0).tolist(),
            p75=np.percentile(arr, 75, axis=0).tolist(),
            p95=np.percentile(arr, 95, axis=0).tolist(),
        )


_CACHED: Dict[str, ChronosForecaster] = {}


def get_forecaster(
    model_id: str = DEFAULT_MODEL_ID,
    finetuned: bool = False,
    device: str = "auto",
) -> ChronosForecaster:
    """Lazy-load and cache forecasters keyed by (model_id, finetuned)."""
    cache_key = f"{model_id}|ft={finetuned}"
    if cache_key in _CACHED:
        return _CACHED[cache_key]

    local_path = None
    if finetuned:
        ft_dir = Path(os.getenv("CHRONOS_FT_DIR", str(DEFAULT_LOCAL_DIR)))
        if ft_dir.exists():
            local_path = str(ft_dir)

    forecaster = ChronosForecaster(
        model_id=model_id,
        device=device,
        local_path=local_path,
    )
    _CACHED[cache_key] = forecaster
    return forecaster


def chronos_available() -> bool:
    try:
        import chronos  # noqa: F401
        return True
    except ImportError:
        return False


def chronos_finetuned_available() -> bool:
    ft_dir = Path(os.getenv("CHRONOS_FT_DIR", str(DEFAULT_LOCAL_DIR)))
    return ft_dir.exists() and any(ft_dir.iterdir())
