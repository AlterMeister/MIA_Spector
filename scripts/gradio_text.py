# gradio_text.py
'''
    Gradio 界面（支持 Pythia / LLaMA）
'''
from __future__ import annotations
from functools import lru_cache
from typing import Any, Dict, Optional, Union, Literal

import os, math, yaml, json
import numpy as np

# ─────────────────────────────────────────────────────────
# 运行在无 GUI 的远程机器时，给 Gradio 一个稳定的临时目录
USER_TMP = os.path.expanduser("~/.cache/gradio_tmp")
os.environ["GRADIO_TEMP_DIR"] = USER_TMP
os.environ["TMPDIR"] = USER_TMP
os.environ["GRADIO_CACHE_DIR"] = USER_TMP
os.makedirs(USER_TMP, exist_ok=True)

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== 你的已有模块（不在此重复实现）=====
from attacks.attack_text import ScoreCalculator
from src.metric_score import compute_single_metric
from src.load_yaml import decide_single as decide_single, load_yaml_config
from src.auto_token_selector import select_token_size  # 如未使用可移除

# ─────────────────────────────────────────────────────────
# 本地模型注册（按需改成你的路径）
PYTHIA: Dict[str, str] = {
    "pythia-410m": "/home/wanghuili/MIA-Spector/models/pythia/pythia-410m",
    "pythia-1.4b": "/home/wanghuili/MIA-Spector/models/pythia/pythia-1.4b",
    "pythia-2.8b": "/home/wanghuili/MIA-Spector/models/pythia/pythia-2.8b",
}
LLAMA: Dict[str, str] = {
    "llama-13b": "/home/wanghuili/MIA-Spector/models/llama/llama-13b-hf",
    "llama-30b": "/home/wanghuili/MIA-Spector/models/llama/llama-30b-hf",
}

CFG_PYTHIA: Dict[str, str] = {
    "WikiMIA_length32":  "/home/wanghuili/MIA-Spector/configs/text/pythia/threshold_WikiMIA_length32.yaml",
    "WikiMIA_length64":  "/home/wanghuili/MIA-Spector/configs/text/pythia/threshold_WikiMIA_length64.yaml",
    "WikiMIA_length128": "/home/wanghuili/MIA-Spector/configs/text/pythia/threshold_WikiMIA_length128.yaml",
    "WikiMIA_length256": "/home/wanghuili/MIA-Spector/configs/text/pythia/threshold_WikiMIA_length256.yaml",
}
CFG_LLAMA: Dict[str, str] = {
    "WikiMIA_length32":  "/home/wanghuili/MIA-Spector/configs/text/llama/threshold_WikiMIA_length32.yaml",
    "WikiMIA_length64":  "/home/wanghuili/MIA-Spector/configs/text/llama/threshold_WikiMIA_length64.yaml",
    "WikiMIA_length128": "/home/wanghuili/MIA-Spector/configs/text/llama/threshold_WikiMIA_length128.yaml",
    "WikiMIA_length256": "/home/wanghuili/MIA-Spector/configs/text/llama/threshold_WikiMIA_length256.yaml",
}

FAMILIES = ("pythia", "llama")

# ─────────────────────────────────────────────────────────
# Lazy loaders：按 family 载入模型与 tokenizer
@lru_cache(maxsize=8)
def get_calc(family: str, model_key: str) -> ScoreCalculator:
    family = family.lower().strip()
    if family == "pythia":
        hf_id = PYTHIA[model_key]
        use_fast = True
        trust_remote = False
    elif family == "llama":
        hf_id = LLAMA[model_key]
        # LLaMA 社区权重常见：fast tokenizer 不完全可用；同时可能需要 trust_remote_code
        use_fast = False  # ← NEW for LLaMA
        trust_remote = True  # ← NEW for LLaMA（若你是官方 Meta HF 仓库且无自定义 code，可 False）
    else:
        raise KeyError(f"Unknown family: {family}")

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(hf_id, use_fast=use_fast, trust_remote_code=trust_remote)
    # LLaMA 常见坑：无 pad_token，需要对齐到 eos；padding 左对齐便于自回归
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Model
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    mdl = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=trust_remote,  # ← NEW for LLaMA
    )
    mdl.eval()
    return ScoreCalculator(model=mdl, tokenizer=tok)

@lru_cache(maxsize=32)
def get_cfg(family: str, cfg_key: str) -> Dict[str, Any]:
    family = family.lower().strip()
    if family == "pythia":
        cfg_map = CFG_PYTHIA
    elif family == "llama":
        cfg_map = CFG_LLAMA
    else:
        raise KeyError(f"Unknown family: {family}")

    if cfg_key not in cfg_map:
        raise KeyError(f"Unknown Config for {family}: {cfg_key}")
    return load_yaml_config(cfg_map[cfg_key])

# ─────────────────────────────────────────────────────────
def infer(
    text: str,
    family: str,
    model_key: str,
    cfg_key: str,
    metric_group: str,
    ratio: str,
    ppl_key: str,
    mode: str,
    abstain_margin: str
):
    print("[infer] raw text repr:", repr(text))
    if not text or not text.strip():
        return "Please input text!", None

    family = family.lower().strip()
    if family not in FAMILIES:
        return f"Unknown family: {family}", None

    # 加载模型与配置
    try:
        calc = get_calc(family, model_key)
        cfg = get_cfg(family, cfg_key)
    except Exception as e:
        return f"Load Config/Model failed: {e}", None

    # subkey
    subkey = ratio if metric_group in ("mink++", "mink") else ppl_key

    # 单指标分数
    try:
        score = compute_single_metric(calc, text, metric_group, subkey)
    except TypeError:
        # 兼容没有 on_error 参数的旧封装
        score = compute_single_metric(calc, text, metric_group, subkey)

    print(score)

    if not (isinstance(score, float) and np.isfinite(score)):
        return "Cannot calculate the score (text too short or invalid).", None

    # 阈值模式与弃答带
    abstain = None
    if abstain_margin and str(abstain_margin).strip():
        try:
            abstain = float(abstain_margin)
        except Exception:
            return "❗abstain margin should be a number.", None

    # Debug keys
    ths = cfg.get("thresholds", {})
    print("[debug] family:", family)
    print("[debug] model:", model_key)
    print("[debug] metric_group:", metric_group)
    print("[debug] subkey(raw):", subkey)
    print("[debug] groups available:", list(ths.keys()))
    if metric_group in ths:
        print("[debug] keys under group:", list(ths[metric_group].keys()))

    # 判决
    out = decide_single(cfg, metric_group, subkey, score_raw=score, mode=mode, abstain_margin=abstain)

    # 展示
    conf_val = out.get("confidence", 0.0)
    try:
        conf_val = float(conf_val)
        if not np.isfinite(conf_val):
            conf_val = 0.0
    except Exception:
        conf_val = 0.0
    conf_pct = f"{conf_val*100:.1f}%"

    md = (
        f"**Decision:** {out['decision']}\n"
        f"**Confidence:** {conf_pct}\n\n"
        f"- family: `{family}`\n"
        f"- model: `{model_key}`\n"
        f"- cfg: `{cfg_key}`\n"
        f"- metric_group: `{out['metric_group']}`\n"
        f"- subkey: `{out['subkey']}`\n"
        f"- direction: `{out['direction']}`\n"
        f"- mode: `{out['mode']}`\n"
        f"- score: `{out['score']:.6f}`\n"
        f"- threshold: `{out['threshold']:.6f}`\n"
    )
    return md, json.dumps(out, ensure_ascii=False, indent=2)

# ─────────────────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧪 MIA-Inspector (Text)")
        gr.Markdown("Input text → choose **family/model/config** → choose metric → get **Yes/No/Uncertain** + confidence.")

        with gr.Row():
            text = gr.Textbox(lines=6, label="Input Text", placeholder="Paste your text here...")

        with gr.Row():
            family = gr.Dropdown(choices=list(FAMILIES), value="pythia", label="Model Family")

            # family 变更时动态刷新
            model = gr.Dropdown(choices=list(PYTHIA.keys()), value="pythia-410m", label="Model")
            cfg   = gr.Dropdown(choices=list(CFG_PYTHIA.keys()), value="WikiMIA_length128", label="Threshold Config (YAML)")

            metric_group = gr.Dropdown(choices=["mink++","mink","perplexity"], value="mink++", label="Metric Group")
            mode  = gr.Dropdown(choices=["bestJ","fpr_alpha"], value="bestJ", label="Threshold Mode")

        with gr.Row():
            ratio = gr.Dropdown(choices=["0.1","0.2","0.3","0.4","0.5","0.6"], value="0.3",
                                label="Ratio (for mink/mink++)", visible=True)
            ppl_key = gr.Dropdown(choices=["variance","std","range","skewness","kurtosis"], value="range",
                                  label="Perplexity Key", visible=False)
            abstain_margin = gr.Textbox(label="Abstain Margin (optional)", placeholder="e.g., 0.02")

        btn = gr.Button("Decide", variant="primary")
        out_md = gr.Markdown()
        out_json = gr.Code(label="Raw JSON Result", language="json")

        # 切换 metric_group 展示 ratio / ppl_key
        def toggle_group(group):
            show_ratio = group in ("mink++","mink")
            show_ppl   = group == "perplexity"
            return gr.update(visible=show_ratio), gr.update(visible=show_ppl)

        metric_group.change(toggle_group, inputs=[metric_group], outputs=[ratio, ppl_key])

        # family → 更新 model/cfg
        def toggle_family(fam: str):
            fam = fam.lower().strip()
            if fam == "pythia":
                return (
                    gr.update(choices=list(PYTHIA.keys()), value=list(PYTHIA.keys())[0]),
                    gr.update(choices=list(CFG_PYTHIA.keys()), value="WikiMIA_length128"),
                )
            elif fam == "llama":
                return (
                    gr.update(choices=list(LLAMA.keys()), value=list(LLAMA.keys())[0]),
                    gr.update(choices=list(CFG_LLAMA.keys()), value="WikiMIA_length128"),
                )
            else:
                return gr.update(), gr.update()

        family.change(toggle_family, inputs=[family], outputs=[model, cfg])

        btn.click(
            infer,
            inputs=[text, family, model, cfg, metric_group, ratio, ppl_key, mode, abstain_margin],
            outputs=[out_md, out_json]
        )
    return demo

# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=8000, inbrowser=False, share=False)


# python -m scripts.gradio_text
# ssh -L 8000:127.0.0.1:8000 -p 5110 wanghuili@101.6.70.28
# 然后在本地浏览器打开: http://127.0.0.1:8000
