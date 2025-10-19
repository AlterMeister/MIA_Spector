'''
    读取yaml文件
'''

from __future__ import annotations
from typing import Any, Dict, Optional, Union
from pathlib import Path

import json
import math
import bisect
import yaml
import numpy as np

# define the structure of data
YamlCfg = Dict[str, Any]

def load_yaml_config(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _ratio_str(x: Union[str, float]) -> str:
    return f"{float(x):.1f}"

# 把拍平的 perplexity_* 顶层键“虚拟”成一个分组
def _build_perplexity_group(thresholds: Dict[str, Any]) -> Dict[str, Any]:
    mapping = {
        "variance":  "perplexity_variance",
        "std":       "perplexity_std",
        "range":     "perplexity_range",
        "skewness":  "perplexity_skewness",
        "kurtosis":  "perplexity_kurtosis",
    }
    group = {}
    for short, full in mapping.items():
        if full in thresholds:
            group[short] = thresholds[full]
    return group

def _pick_entry(cfg: YamlCfg, metric_group: str, subkey: Union[str, float]) -> Dict[str, Any]:
    thresholds = cfg.get("thresholds", {})
    if metric_group in ("mink++", "mink"):
        if metric_group not in thresholds:
            raise KeyError(f"YAML.thresholds 缺少分组: {metric_group}")
        group = thresholds[metric_group]
        k_norm = _ratio_str(subkey)
        # 兼容 YAML 键可能是 float 或 str 的情况
        mapping = { _ratio_str(k): k for k in group.keys() }
        if k_norm not in mapping:
            raise KeyError(f"{metric_group} 未找到比例 {k_norm}；可用: {sorted(mapping.keys())}")
        return group[mapping[k_norm]]

    # 兼容两种用法：
    # a) 你前端传 metric_group="perplexity" + subkey in {"variance","std",...}
    # b) 直接传 metric_group="perplexity_std"（无 subkey 概念）
    if metric_group == "perplexity":
        group = _build_perplexity_group(thresholds)
        sk = str(subkey).strip().lower()
        if sk not in group:
            raise KeyError(f"perplexity 未找到子键 {sk}；可用: {list(group.keys())}")
        return group[sk]

    # 直接支持拍平顶层键（如 "perplexity_std"）
    if metric_group in thresholds and metric_group.startswith("perplexity_"):
        return thresholds[metric_group]

    raise KeyError(f"未知的 metric_group: {metric_group}；可用: {list(thresholds.keys())}")

def _sign(dir_str: str) -> int:
    return +1 if str(dir_str).strip() == "+" else -1

def decide_single(
    cfg: YamlCfg,
    metric_group: str,
    subkey: Union[str, float],
    *,
    score_raw: float,
    mode: str = "bestJ",
    abstain_margin: float | None = None
) -> Dict[str, Any]:
    entry = _pick_entry(cfg, metric_group, subkey)
    direction = entry["direction"]
    sgn = _sign(direction)
    tau_raw = entry["threshold_bestJ"] if mode == "bestJ" else entry["threshold_fpr_alpha"]

    # 正向化
    s_pos  =  score_raw if sgn == +1 else -score_raw
    tau_pos = tau_raw   if sgn == +1 else -tau_raw

    # 判决
    if abstain_margin is not None and abs(s_pos - tau_pos) <= abstain_margin:
        decision = "Uncertain"
    else:
        decision = "Yes" if s_pos >= tau_pos else "No"

    # 置信度：优先 calibrator，退回 ECDF，再退回阈值距离
    def _sigmoid(z: float) -> float:
        try: return 1.0/(1.0+math.exp(-z))
        except OverflowError: return 0.0 if z < 0 else 1.0

    conf = None
    calib = entry.get("calibrator")
    if isinstance(calib, dict) and "a" in calib and "b" in calib:
        conf = _sigmoid(float(calib["a"]) * s_pos + float(calib["b"]))
    else:
        ec = entry.get("ecdf", {})
        non = ec.get("non_member_posdir_ecdf", {})
        xs = np.asarray(non.get("xs", []), dtype=float)
        cdf = np.asarray(non.get("cdf", []), dtype=float)
        if xs.size and xs.size == cdf.size:
            i = np.searchsorted(xs, s_pos, side="right")
            F = 0.0 if i == 0 else (1.0 if i >= xs.size else float(cdf[i-1]))
            conf = max(0.0, 1.0 - F)
        else:
            conf = _sigmoid((s_pos - tau_pos)/0.1)

    # 规范 subkey 输出
    subkey_out = (_ratio_str(subkey) if metric_group in ("mink++","mink")
                  else (str(subkey) if metric_group == "perplexity"
                        else metric_group.replace("perplexity_", "")))

    return {
        "metric_group": metric_group,
        "subkey": subkey_out,
        "direction": direction,
        "mode": mode,
        "threshold": float(tau_raw),
        "score": float(score_raw),
        "decision": decision,
        "confidence": float(conf),
    }
