'''
    Some tools to compute the LVLMs
'''

import torch
import os
import json
import torch.nn.functional as F
import numpy as np

from typing import List, Tuple
from sklearn.metrics import roc_curve, auc, roc_auc_score

def compute_diffs_from_logits(logits, logits_p):
    """
    logits: torch.Tensor shape [1, T, V] or [T, V]
    logits_p: 同上（加噪后）
    returns dict of aggregated metrics
    """
    # ensure shape [T, V]
    if logits.dim() == 3 and logits.shape[0] == 1:
        logits = logits[0]
    if logits_p.dim() == 3 and logits_p.shape[0] == 1:
        logits_p = logits_p[0]

    T = min(logits.shape[0], logits_p.shape[0])  # 对齐到最小长度

    # 对齐长度
    logits = logits[:T]
    logits_p = logits_p[:T]

    p = F.softmax(logits, dim=-1)
    p_p = F.softmax(logits_p, dim=-1)

    logit_l2 = torch.norm(logits - logits_p, dim=-1)  # per-step logit L2 [T]

    kl = (p * (torch.log(p + 1e-12) - torch.log(p_p + 1e-12))).sum(-1)  # KL

    top1 = torch.argmax(p, dim=-1)  # top1 change
    top1p = torch.argmax(p_p, dim=-1)
    top1_change = (top1 != top1p).float()

    # entropy change
    H = -(p * torch.log(p + 1e-12)).sum(-1)
    H_p = -(p_p * torch.log(p_p + 1e-12)).sum(-1)
    H_diff = torch.abs(H - H_p)

    res = {
        "logit_l2_mean":        float(logit_l2.mean().cpu().item()),
        "logit_l2_max":         float(logit_l2.max().cpu().item()),
        "kl_mean":              float(kl.mean().cpu().item()),
        "top1_change_rate":     float(top1_change.mean().cpu().item()),
        "entropy_diff_mean":    float(H_diff.mean().cpu().item()),
    }
    return res

from sklearn.metrics import roc_curve, auc
import numpy as np
from typing import List, Tuple

class Metrics:
    @staticmethod
    def calculate_metrics(scores: List[float], labels: List[int]) -> Tuple[
        float, float, float, np.ndarray, np.ndarray, float, float]:
        """
        Calculate AUROC, FPR@TPR=95%, TPR@FPR=5%, 
        and return the best threshold (Youden J criterion).

        Returns:
            auroc: area under ROC curve
            fpr95: FPR when TPR >= 0.95
            tpr05: TPR when FPR <= 0.05
            fpr_list, tpr_list: full ROC points
            best_thresh: threshold giving max (TPR - FPR)
            best_j: value of (TPR - FPR)
        """
        scores = np.array(scores, dtype=float)
        labels = np.array(labels, dtype=int)

        valid_indices = ~np.isnan(scores)
        scores = scores[valid_indices]
        labels = labels[valid_indices]

        if len(np.unique(labels)) < 2:
            return 0.0, 1.0, 0.0, np.array([]), np.array([]), np.nan, np.nan

        # ROC curve
        fpr_list, tpr_list, thresholds = roc_curve(labels, scores)
        auroc = auc(fpr_list, tpr_list)

        # FPR@TPR=95%
        tpr_95_idx = np.where(tpr_list >= 0.95)[0]
        fpr95 = float(fpr_list[tpr_95_idx[0]]) if len(tpr_95_idx) > 0 else 1.0

        # TPR@FPR=5%
        fpr_5_idx = np.where(fpr_list <= 0.05)[0]
        tpr05 = float(tpr_list[fpr_5_idx[-1]]) if len(fpr_5_idx) > 0 else 0.0

        # 最优阈值（Youden J = TPR – FPR 最大处）
        j_scores = tpr_list - fpr_list
        best_idx = int(np.argmax(j_scores))
        best_thresh = float(thresholds[best_idx]) if len(thresholds) > 0 else np.nan
        best_j = float(j_scores[best_idx]) if len(j_scores) > 0 else np.nan

        return auroc, fpr95, tpr05, fpr_list, tpr_list, best_thresh, best_j

