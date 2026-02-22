from __future__ import annotations
from typing import Union, Literal, Optional

from attacks.attack_text import ScoreCalculator
from attacks.utils.metrics_utils import Metrics

import math
import numpy as np

'''
    Calculate the target Score
'''

AllowedGroup = Literal["mink", "mink++", "perplexity"]
AllowedPPL   = Literal["variance", "std", "range", "skewness", "kurtosis"]

def compute_single_metric(
    calc: ScoreCalculator,
    text: str,
    metric_group: AllowedGroup,
    subkey: Union[str, float],
    *,
    on_error: Literal["raise", "nan"] = "raise",
    min_tokens: int = 2,
) -> float:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text can not be empty")
    
    mg = str(metric_group).strip()
    if mg not in ("mink", "mink++", "perplexity"):
        raise ValueError(f"Unknown metric_group: {metric_group}")
    
    # ruler subkey
    try:
        if mg in ("mink", "mink++"):
            r = float(f"{float(subkey):.1f}")
            r = max(0.0, min(1.0, r))
            norm_subkey: Union[str, float] = r
        else:
            sk = str(subkey).strip().lower()
            if sk not in {"variance", "std", "range", "skewness", "kurtosis"}:
                if on_error == "nan": 
                    return float("nan")
                raise ValueError(f"perplexity subkey is illegal: {subkey}")
            norm_subkey = sk
    except Exception as e:
        if on_error == "nan": 
            return float("nan")
        raise ValueError(f"subkey parsers failed: {subkey}") from e
    
    try:
        val = calc.calculate_single_metric(text, mg, norm_subkey)
    except KeyError as e:
        if on_error == "nan":
            return float("nan")
        raise KeyError(f"Not Found Key: ({mg}, {norm_subkey}): {e}") from e
    except RuntimeError as e:
        if on_error == "nan":
            return float("nan")
        raise RuntimeError(f"Model forward failed : {e}") from e
    except Exception as e:
        if on_error == "nan":
            return float("nan")
        raise

    # check
    try:
        val = float(val)
        if math.isnan(val) or math.isinf(val):
            return float("nan") if on_error == "nan" else (_raise(ValueError("The score is not available data")))
        return val
    except Exception:
        return float("nan") if on_error == "nan" else (_raise(ValueError("The score is not available data")))
    
def _raise(err: Exception):
    raise err