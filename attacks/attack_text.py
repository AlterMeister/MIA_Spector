import os
import argparse
import numpy as np
import pandas as pd
import zlib
import matplotlib.pyplot as plt
import seaborn as sns
import json
import torch
import torch.nn.functional as F
import random

from scipy.fft import fft
from sklearn.metrics import roc_curve, auc, roc_auc_score
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Union
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

'''
    Analysis for the text LLMs
'''
MINK_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

class DatasetLoader:
    """Handles dataset loading and preprocessing."""
    @staticmethod
    def load_dataset(dataset_name: str, max_samples: int = None) -> List[Dict[str, Any]]:
        """Load and convert dataset to list format."""
        if 'bookmia' in dataset_name.lower():
            with open('data/BookMIA.json', 'r') as f:
                dataset = json.load(f)
            for entry in dataset:
                entry['summary'] = entry['book'][:-4]
        elif 'wikimia' in dataset_name.lower() and 'perturbed' not in dataset_name.lower():
            if '32' in dataset_name.lower():
                with open('/home/wanghuili/MIA-Spector/datasets/WIKI/wikimia32_summary.jsonl', 'r') as f:
                    dataset = [json.loads(line) for line in f]
            elif '64' in dataset_name.lower():
                with open('/home/wanghuili/MIA-Spector/datasets/WIKI/wikimia64_summary.jsonl', 'r') as f:
                    dataset = [json.loads(line) for line in f]
            elif '128' in dataset_name.lower():
                with open('/home/wanghuili/MIA-Spector/datasets/WIKI/wikimia128_summary.jsonl', 'r') as f:
                    dataset = [json.loads(line) for line in f]
            elif '256' in dataset_name.lower():
                with open('/home/wanghuili/MIA-Spector/datasets/WIKI/wikimia256_summary.jsonl', 'r') as f:
                    dataset = [json.loads(line) for line in f]
            else:
                raise ValueError(f"Invalid dataset name: {dataset_name}")
        elif 'wikimia' in dataset_name.lower() and 'perturbed' in dataset_name.lower():
            dataset = load_dataset('zjysteven/WikiMIA_paraphrased_perturbed', split=dataset_name)
        elif 'syntheic_prompt' in dataset_name.lower():
            with open('data/syntheic_prompt.jsonl', 'r') as f:
                dataset = [json.loads(line) for line in f]
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
        dataset = DatasetLoader._convert_to_list(dataset)
        if max_samples is not None:
            dataset = dataset[:max_samples]
        return dataset

    @staticmethod
    def load_dataset_byself(
        dataset_path: str, max_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Load dataset from a local file."""
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            dataset = [json.loads(line) for line in f]

        dataset = DatasetLoader._convert_to_list(dataset)
        
        if max_samples is not None:
            dataset = dataset[:max_samples]
        
        return dataset

    @staticmethod
    def _convert_to_list(dataset) -> List[Dict[str, Any]]:
        """Convert HuggingFace dataset to list of dictionaries."""
        return [dataset[i] for i in range(len(dataset))]


class ScoreCalculator:
    def __init__(self, model: Any, tokenizer: Any, ref_model:Any=None, ref_tokenizer:Any=None):
        self.model = model
        self.tokenizer = tokenizer
        self.ref_model = ref_model
        self.ref_tokenizer = ref_tokenizer

    def calculate_single_metric(self, text: str, metric_group: str, subkey: Union[float, str]) -> float:
        input_ids = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids, output_hidden_states=False)
            logits = outputs.logits

        if metric_group in ("mink", "mink++"):
            r = float(f"{float(subkey):.1f}")
            out = self._calculate_mink_scores(input_ids, logits)
            key = ("mink++_" if metric_group == "mink++" else "mink_") + f"{r:.1f}"
            if key not in out:
                raise KeyError(f"no key {key} in result")
            return out[key]
        elif metric_group == "perplexity":
            sc = self._calculate_token_perplexity_scores(input_ids, logits)
            mapping = {"variance":"perplexity_variance", "std":"perplexity_std", "range":"perplexity_range"}
            k = mapping[str(subkey)]
            return float(sc[k])
        else:
            raise ValueError(metric_group)

    def calculate_scores(self, text:str) -> Dict[str, float]:
        input_ids = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0)
        input_ids = input_ids.to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids, output_hidden_states=True)
            loss = outputs.loss
            logits = outputs.logits
        ll = -loss.item()  # log-likelihood

        log_probs = F.log_softmax(logits[0, :-1], dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
        token_perplexities = torch.exp(-token_log_probs).to(torch.float).cpu().numpy()

        perplexity = torch.exp(loss).item()

        last_hidden_state_l2 = float(torch.norm(outputs.hidden_states[-1]).item())
        embedding_l2 = float(torch.norm(outputs.hidden_states[0]).item())

        scores = {
            'loss': ll,
            'perplexity': perplexity,
            'zlib': ll / len(zlib.compress(bytes(text, 'utf-8'))),
            'last_hidden_state_l2': last_hidden_state_l2,
            'embedding_l2': embedding_l2
        }

        scores.update(self._calculate_mink_scores(input_ids, logits))
        scores.update(self._calculate_token_perplexity_scores(input_ids, logits))

        if self.ref_model and self.ref_tokenizer:
            ref_input_ids = torch.tensor(self.ref_tokenizer.encode(text)).unsqueeze(0)
            ref_input_ids = ref_input_ids.to(self.ref_model.device)

            with torch.no_grad():
                ref_outputs = self.ref_model(ref_input_ids, labels=ref_input_ids)
                ref_logits = ref_outputs.logits
                ref_loss = ref_outputs.loss
                ref_loss = -ref_loss.item()  # log-likelihood

            scores['ref_loss_ratio'] = self._calculate_loss_ratio(ll, ref_loss)
            scores['ref_loss_diff'] = self._calculate_loss_diff(ll, ref_loss)

        return scores, token_perplexities

    
    def calculate_set_level_scores(self, dataset: List[Dict[str, str]], K: int, N: int) -> Dict[str, float]:
        """
        计算 Set-level MIA Scores。通过随机选择 K 条样本，重复 N 次来计算每组的 AUROC等指标。
        
        :param dataset: 数据集（每条样本是一个字典，包含 'input' 和 'label'）
        :param K: 每组选择的样本数
        :param N: 重复选择的次数（每次计算一组）
        :return:
        """
        all_scores = {key: [] for key in ['loss', 'perplexity', 'zlib', 'last_hidden_state_l2', 'embedding_l2']}

        for n in range(N):
            random_samples = random.sample(dataset, K)
            input_texts = [sample['input'] for sample in random_samples]

            group_scores = {key : [] for key in all_scores}
            for text in input_texts:
                scores, _ = self.calculate_scores(text)
                for key in scores:
                    group_scores[key].append(scores[key])

            for key in group_scores:
                all_scores[key].append(np.mean(group_scores[key]))

        # 计算最终平均值
        final_scores = {key: np.mean(all_scores[key]) for key in all_scores}

        return final_scores
    

    def _calculate_mink(self, text:str) -> Dict[str, float]:
        """仅仅计算Mink"""
        scores = {}

        input_ids = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0)
        input_ids = input_ids.to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids, output_hidden_states=True)
            loss = outputs.loss
            logits = outputs.logits
        ll = -loss.item()
        
        # Prepare
        input_ids = input_ids[0][1:].unsqueeze(-1)
        probs = F.softmax(logits[0, :-1], dim=-1)
        log_probs = F.log_softmax(logits[0, :-1], dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
        
        # Min-K% scores
        for ratio in MINK_RATIOS:
            k_length = int(len(token_log_probs) * ratio)
            topk = np.sort(token_log_probs.to(torch.float).cpu())[:k_length]
            scores[f'mink_{ratio}'] = np.mean(topk).item()

        return scores
    
    def _calculate_mink_total(self, text:str) -> Dict[str, List]:
        input_ids = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0)
        input_ids = input_ids.to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids, output_hidden_states=True)
            loss = outputs.loss
            logits = outputs.logits
        ll = -loss.item()
        
        # Prepare
        input_ids = input_ids[0][1:].unsqueeze(-1)
        probs = F.softmax(logits[0, :-1], dim=-1)
        log_probs = F.log_softmax(logits[0, :-1], dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)

        return token_log_probs


    def _calculate_mink_scores(self, input_ids: torch.Tensor, logits: torch.Tensor) -> Dict[str, float]:
        """Calculate Min-K% and Min-K%++ scores."""
        scores = {}
        
        # Prepare token-level probabilities
        input_ids = input_ids[0][1:].unsqueeze(-1)
        probs = F.softmax(logits[0, :-1], dim=-1)
        log_probs = F.log_softmax(logits[0, :-1], dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
        
        # Calculate statistics for Min-K%++
        mu = (probs * log_probs).sum(-1)
        sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
        
        # Min-K% scores
        for ratio in MINK_RATIOS:
            k_length = int(len(token_log_probs) * ratio)
            topk = np.sort(token_log_probs.to(torch.float).cpu())[:k_length]
            scores[f'mink_{ratio}'] = np.mean(topk).item()
        
        # Min-K%++ scores
        safe_sigma = torch.clamp(sigma, min=1e-8)
        mink_plus = (token_log_probs - mu) / safe_sigma.sqrt()
        for ratio in MINK_RATIOS:
            k_length = int(len(mink_plus) * ratio)
            topk = np.sort(mink_plus.to(torch.float).cpu())[:k_length]
            scores[f'mink++_{ratio}'] = np.mean(topk).item()
        
        return scores
    
    def _calculate_loss_ratio(self, target_loss, reference_loss):
        '''
        This function takes a list of lists and returns the ratio of the mean loss to the perplexity of a reference model
        input:
            target_loss: a tensor of loss
            reference_loss: a tensor of loss
        '''
        ratio = target_loss/reference_loss
        return ratio

    def _calculate_loss_diff(self, target_loss, reference_loss):
        '''
        This function takes a list of lists and returns the difference of the mean loss to the perplexity of a reference model
        input:
            target_loss: a tensor of loss
            reference_loss: a tensor of loss
        '''
        diff = target_loss - reference_loss
        return diff
    
    def get_token_perplexities(self, text: str) -> Tuple[np.ndarray, List[str]]:
        """Get token-level perplexities and token strings for visualization."""
        # Get model outputs
        input_ids = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0)
        input_ids = input_ids.to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]
        
        # Get token-level perplexities
        input_ids = input_ids[0][1:].unsqueeze(-1)  # Remove BOS token
        log_probs = F.log_softmax(logits[0, :-1], dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
        token_perplexities = torch.exp(-token_log_probs).cpu().numpy()
        
        # Get token strings
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(-1).cpu().numpy())
        
        return token_perplexities, tokens

    def _calculate_token_perplexity_scores(self, input_ids: torch.Tensor, logits: torch.Tensor) -> Dict[str, float]:
        """Calculate token-level perplexity relationship scores for MIA."""
        scores = {}
        
        # Prepare token-level data
        input_ids = input_ids[0][1:].unsqueeze(-1)  # Remove BOS token
        probs = F.softmax(logits[0, :-1], dim=-1)
        log_probs = F.log_softmax(logits[0, :-1], dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)

        token_perplexities = (-token_log_probs).to(torch.float).cpu().numpy()
        
        if len(token_perplexities) < 2:
            # Handle very short sequences
            scores.update({
                'perplexity_variance': 0.0,
                'perplexity_std': 0.0,
                'perplexity_range': 0.0,
                'perplexity_trend': 0.0,
                'perplexity_correlation': 0.0,
                'perplexity_autocorr': 0.0,
                'perplexity_entropy': 0.0,
                'perplexity_skewness': 0.0,
                'perplexity_kurtosis': 0.0,
                'perplexity_norm1': 0.0,
            })
            return scores
        
        # 1. Perplexity variance and standard deviation
        scores['perplexity_variance'] = float(np.var(token_perplexities))
        scores['perplexity_std'] = float(np.std(token_perplexities))
        
        # 2. Perplexity range (max - min)
        scores['perplexity_range'] = float(np.max(token_perplexities) - np.min(token_perplexities))
        
        # 3. Perplexity trend (linear regression slope)
        x = np.arange(len(token_perplexities))
        if len(x) > 1:
            try:
                slope, _ = np.polyfit(x, token_perplexities, 1)
                scores['perplexity_trend'] = float(np.clip(slope, -1e6, 1e6))
            except:
                scores['perplexity_trend'] = 0.0
        else:
            scores['perplexity_trend'] = 0.0
        
        # 4. Perplexity correlation with position
        if len(token_perplexities) > 1:
            try:
                correlation = np.corrcoef(x, token_perplexities)[0, 1]
                scores['perplexity_correlation'] = float(correlation) if not np.isnan(correlation) else 0.0
            except:
                scores['perplexity_correlation'] = 0.0
        else:
            scores['perplexity_correlation'] = 0.0
        
        # 5. Perplexity autocorrelation (lag-1)
        if len(token_perplexities) > 2:
            try:
                autocorr = np.corrcoef(token_perplexities[:-1], token_perplexities[1:])[0, 1]
                scores['perplexity_autocorr'] = float(autocorr) if not np.isnan(autocorr) else 0.0
            except:
                scores['perplexity_autocorr'] = 0.0
        else:
            scores['perplexity_autocorr'] = 0.0
        
        # 6. Perplexity entropy (measure of uncertainty distribution)
        # Normalize perplexities to create a probability distribution
        perplexity_sum = np.sum(token_perplexities)
        if perplexity_sum > 0:
            perplexity_norm = token_perplexities / perplexity_sum
            perplexity_norm = perplexity_norm[perplexity_norm > 1e-10]  # Remove very small values
            if len(perplexity_norm) > 0:
                try:
                    entropy = -np.sum(perplexity_norm * np.log(perplexity_norm))
                    scores['perplexity_entropy'] = float(np.clip(entropy, 0, 100))
                except:
                    scores['perplexity_entropy'] = 0.0
            else:
                scores['perplexity_entropy'] = 0.0
        else:
            scores['perplexity_entropy'] = 0.0
        
        # 7. Perplexity skewness and kurtosis (distribution shape)
        if len(token_perplexities) > 2:
            from scipy import stats
            try:
                skewness = stats.skew(token_perplexities)
                kurtosis = stats.kurtosis(token_perplexities)
                scores['perplexity_skewness'] = float(np.clip(skewness, -1e6, 1e6)) if not np.isnan(skewness) else 0.0
                scores['perplexity_kurtosis'] = float(np.clip(kurtosis, -1e6, 1e6)) if not np.isnan(kurtosis) else 0.0
            except:
                scores['perplexity_skewness'] = 0.0
                scores['perplexity_kurtosis'] = 0.0
        else:
            scores['perplexity_skewness'] = 0.0
            scores['perplexity_kurtosis'] = 0.0

        # 8. Perplexity norm1
        scores['perplexity_norm1'] = float(np.linalg.norm(token_perplexities, ord=1))

        mu = (probs * log_probs).sum(-1)
        sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)

        token_log_probs_np = token_log_probs.to(torch.float).cpu().numpy()
        token_log_probs_residual = np.abs(token_log_probs_np - np.mean(token_log_probs_np))
        '''
        for ratio in MINK_RATIOS:
            k_length = int(len(token_log_probs_residual) * ratio)
            min_topk = np.sort(token_log_probs_residual)[:k_length]
            scores[f'perplexity_1_mink++_{ratio}'] = float(np.mean(min_topk))
            scores[f'perplexity_2_mink++_{ratio}'] = float(np.sqrt(np.mean(np.square(min_topk))))
            scores[f'perplexity_3_mink++_{ratio}'] = float(np.power(np.mean(np.power(min_topk, 3)), 1/3))
            scores[f'perplexity_4_mink++_{ratio}'] = float(np.power(np.mean(np.power(min_topk, 4)), 1/4))
            max_topk = np.sort(token_log_probs_residual)[-k_length:]
            scores[f'perplexity_1_maxk++_{ratio}'] = float(np.mean(max_topk))
            scores[f'perplexity_2_maxk++_{ratio}'] = float(np.sqrt(np.mean(np.square(max_topk))))
            scores[f'perplexity_3_maxk++_{ratio}'] = float(np.power(np.mean(np.power(max_topk, 3)), 1/3))
            scores[f'perplexity_4_maxk++_{ratio}'] = float(np.power(np.mean(np.power(max_topk, 4)), 1/4))
        '''

        # ====== Fourier Features for Perplexity Sequence ======
        fft_feats = self._calculate_fourier_features(token_perplexities)
        scores.update(fft_feats)
        
        # 8. Advanced perplexity relationship features
        scores.update(self._calculate_advanced_perplexity_features(token_perplexities))
        
        return scores
    
    def _calculate_advanced_perplexity_features(self, token_perplexities: np.ndarray) -> Dict[str, float]:
        """Calculate advanced perplexity relationship features."""
        scores = {}
        
        if len(token_perplexities) < 3:
            # Handle short sequences
            scores.update({
                'perplexity_stability': 0.0,
                'perplexity_clustering': 0.0,
                'perplexity_pattern': 0.0,
                'perplexity_smoothness': 0.0,
                'perplexity_consistency': 0.0,
                'perplexity_volatility': 0.0
            })
            return scores
        
        try:
            # 9. Perplexity stability (inverse of variance)
            mean_perplexity = np.mean(token_perplexities)
            variance = np.var(token_perplexities)
            stability = 1.0 / (1.0 + variance) if variance < 1e10 else 0.0
            scores['perplexity_stability'] = float(np.clip(stability, 0, 1))
            
            # 10. Perplexity clustering (how well perplexities cluster around mean)
            # Calculate coefficient of variation
            if mean_perplexity > 1e-10:
                cv = np.std(token_perplexities) / mean_perplexity
                clustering = 1.0 / (1.0 + cv) if cv < 1e6 else 0.0
                scores['perplexity_clustering'] = float(np.clip(clustering, 0, 1))
            else:
                scores['perplexity_clustering'] = 0.0
            
            # 11. Perplexity pattern (regularity in perplexity changes)
            # Calculate differences between consecutive perplexities
            perplexity_diffs = np.diff(token_perplexities)
            pattern_std = np.std(perplexity_diffs)
            pattern_score = 1.0 / (1.0 + pattern_std) if pattern_std < 1e6 else 0.0
            scores['perplexity_pattern'] = float(np.clip(pattern_score, 0, 1))
            
            # 12. Perplexity smoothness (how smooth the perplexity curve is)
            # Calculate second differences (acceleration)
            if len(perplexity_diffs) > 1:
                second_diffs = np.diff(perplexity_diffs)
                smoothness_std = np.std(second_diffs)
                smoothness = 1.0 / (1.0 + smoothness_std) if smoothness_std < 1e6 else 0.0
                scores['perplexity_smoothness'] = float(np.clip(smoothness, 0, 1))
            else:
                scores['perplexity_smoothness'] = 0.0
            
            # 13. Perplexity consistency (how consistent perplexities are)
            # Calculate relative standard deviation
            if mean_perplexity > 1e-10:
                relative_std = np.std(token_perplexities) / mean_perplexity
                consistency = 1.0 / (1.0 + relative_std) if relative_std < 1e6 else 0.0
                scores['perplexity_consistency'] = float(np.clip(consistency, 0, 1))
            else:
                scores['perplexity_consistency'] = 0.0
            
            # 14. Perplexity volatility (frequency of large changes)
            # Count significant changes (more than 1 std from mean)
            threshold = np.std(token_perplexities)
            if threshold > 1e-10 and len(perplexity_diffs) > 0:
                volatile_changes = np.sum(np.abs(perplexity_diffs) > threshold)
                volatility = volatile_changes / len(perplexity_diffs)
                scores['perplexity_volatility'] = float(np.clip(volatility, 0, 1))
            else:
                scores['perplexity_volatility'] = 0.0
                
        except Exception as e:
            # Fallback values if any calculation fails
            scores.update({
                'perplexity_stability': 0.0,
                'perplexity_clustering': 0.0,
                'perplexity_pattern': 0.0,
                'perplexity_smoothness': 0.0,
                'perplexity_consistency': 0.0,
                'perplexity_volatility': 0.0
            })
        
        return scores

    def _calculate_fourier_features(self, ppl_seq, top_n=3, high_freq_ratio=0.5):
        """
        ppl_seq: 1D array-like, token-level perplexity sequence
        top_n: int, number of top frequencies for主频能量比
        high_freq_ratio: float, high frequency threshold (e.g. 0.5 means后半段为高频)
        """
        ppl_seq = np.asarray(ppl_seq)
        n = len(ppl_seq)
        if n < 2:
            raise ValueError("ppl_seq must be at least 2 tokens long")
        # 去均值
        ppl_seq = ppl_seq - np.mean(ppl_seq)
        # FFT
        fft_vals = fft(ppl_seq)
        fft_mags = np.abs(fft_vals)[:n//2]  # 只取正频率部分
        total_energy = np.sum(fft_mags)
        if total_energy == 0:
            total_energy = 1e-8

        # 1. 主频能量比
        topk = np.sort(fft_mags)[-top_n:]
        main_freq_energy_ratio = np.sum(topk) / total_energy

        # 2. 高频能量占比
        high_freq_start = int(n//2 * high_freq_ratio)
        high_freq_energy = np.sum(fft_mags[high_freq_start:])
        high_freq_energy_ratio = high_freq_energy / total_energy

        # 3. 谱熵
        p = fft_mags / np.sum(fft_mags)
        p = p[p > 0]
        spectral_entropy = -np.sum(p * np.log(p))

        # 4. 最大幅值频率
        max_freq_idx = np.argmax(fft_mags)

        # 5. 谱质心
        freqs = np.arange(len(fft_mags))
        spectral_centroid = np.sum(freqs * fft_mags) / np.sum(fft_mags)

        # 6. 低/高频能量比
        low_freq_energy = np.sum(fft_mags[:high_freq_start])
        low_high_energy_ratio = low_freq_energy / (high_freq_energy + 1e-8)

        return {
            'main_freq_energy_ratio': main_freq_energy_ratio,
            'high_freq_energy_ratio': high_freq_energy_ratio,
            'spectral_entropy': spectral_entropy,
            'max_freq_idx': max_freq_idx,
            'spectral_centroid': spectral_centroid,
            'low_high_energy_ratio': low_high_energy_ratio
        }