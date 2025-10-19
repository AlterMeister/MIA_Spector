'''
    Gradio 界面
'''

from __future__ import annotations
from functools import lru_cache
from typing import Any, Dict, Optional, Union, Literal
from transformers import AutoTokenizer, AutoModelForCausalLM

import os, math, yaml, json
import numpy as np

os.environ.setdefault("GRADIO_TEMP_DIR", os.path.expanduser("~/.cache/gradio_tmp"))
os.makedirs(os.environ["GRADIO_TEMP_DIR"], exist_ok=True)
os.environ.setdefault("TMPDIR", os.environ["GRADIO_TEMP_DIR"])

import gradio as gr
import torch

from attacks.attack_text import ScoreCalculator
from src.metric_score import compute_single_metric
from src.load_yaml import decide_single, load_yaml_config
from src.auto_token_selector import select_token_size

# Model List:
PYTHIA: Dict[str, str] = {
    "pythia-410m": "/home/wanghuili/MIA-Spector/models/pythia/pythia-410m",
    "pythia-1.4b": "/home/wanghuili/MIA-Spector/models/pythia/pythia-1.4b",
    "pythia-2.8b": "/home/wanghuili/MIA-Spector/models/pythia/pythia-2.8b",
}

CFG_FILES: Dict[str, str] = {
    "WikiMIA_length32":  "/home/wanghuili/MIA-Spector/configs/text/threshold_WikiMIA_length32.yaml",
    "WikiMIA_length64":  "/home/wanghuili/MIA-Spector/configs/text/threshold_WikiMIA_length64.yaml",
    "WikiMIA_length128": "/home/wanghuili/MIA-Spector/configs/text/threshold_WikiMIA_length128.yaml",
    "WikiMIA_length256": "/home/wanghuili/MIA-Spector/configs/text/threshold_WikiMIA_length128.yaml",
}

@lru_cache(maxsize=4)
def get_calc(model_key: str) -> ScoreCalculator:
    hf_id = PYTHIA[model_key]
    tok = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float16, device_map="auto")
    mdl.eval()
    return ScoreCalculator(model=mdl, tokenizer=tok)

@lru_cache(maxsize=16)
def get_cfg(cfg_key: str) -> Dict[str, Any]:
    if cfg_key not in CFG_FILES:
        raise KeyError(f"Unknown Config : {cfg_key}")
    return load_yaml_config(CFG_FILES[cfg_key])

def infer(
        text : str,
        model_key: str,
        cfg_key: str,
        metric_group: str,
        ratio: str,
        ppl_key: str,
        mode: str,
        abstain_margin: str
    ):
    print("[infer] raw text repr:", repr(text))

    if not text : return "Please input text!", None

    try:
        calc = get_calc(model_key)
        cfg = get_cfg(cfg_key)
    except Exception as e:
        return f"Load Config/Model failed!", None
    
    # choose subkey
    subkey = ratio if metric_group in ("mink++", "mink") else ppl_key

    # Calculate the single scores
    try:
        score = compute_single_metric(calc, text, metric_group, subkey, on_error="nan")
    except TypeError:
        score = compute_single_metric(calc, text, metric_group, subkey)
    
    if not (isinstance(score, float) and np.isfinite(score)):
        return "Cannot Calculate the Score!", None
    
    # Judge according to YAML
    abstain = None
    if abstain_margin and abstain_margin.strip():
        try:
            abstain = float(abstain_margin)
        except Exception:
            return "❗abstain margin should be a number。", None

    # Check Here
    print("[debug] metric_group:", metric_group)
    print("[debug] subkey(raw):", subkey)

    ths = cfg.get("thresholds", {})
    print("[debug] groups available:", list(ths.keys()))
    if metric_group in ths:
        print("[debug] keys under group:", list(ths[metric_group].keys()))

    out = decide_single(cfg, metric_group, subkey, score_raw=score, mode=mode, abstain_margin=abstain)
    
    conf_pct = f"{out['confidence']*100:.1f}%"
    md = (
        f"**Decision:** {out['decision']}  (confidence **{conf_pct}**)\n\n"
        f"- metric_group: `{out['metric_group']}`\n"
        f"- subkey: `{out['subkey']}`\n"
        f"- direction: `{out['direction']}`\n"
        f"- mode: `{out['mode']}`\n"
        f"- score: `{out['score']:.6f}`\n"
        f"- threshold: `{out['threshold']:.6f}`\n"
    )
    
    return md, json.dumps(out, ensure_ascii=False, indent=2)

def build_ui() -> gr.block:
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧪 MIA-Inspector (Text)")
        gr.Markdown("Input text → choose model and method → Return **Yes/No/Uncertain** + confidence percent.")

        with gr.Row():
            text = gr.Textbox(lines=6, label="Input Text", placeholder="Paste your text here...")

        with gr.Row():
            model = gr.Dropdown(choices=list(PYTHIA.keys()), value="pythia-410m", label="Model")
            cfg   = gr.Dropdown(choices=list(CFG_FILES.keys()), value="WikiMIA_length128", label="Threshold Config (YAML)")
            metric_group = gr.Dropdown(choices=["mink++","mink","perplexity"], value="mink++", label="Metric Group")
            mode  = gr.Dropdown(choices=["bestJ","fpr_alpha"], value="bestJ", label="Threshold Mode")

        with gr.Row():
            ratio = gr.Dropdown(choices=["0.1","0.2","0.3","0.4","0.5","0.6"], value="0.3", label="Ratio (for mink/mink++)", visible=True)
            ppl_key = gr.Dropdown(choices=["variance","std","range","skewness","kurtosis"], value="range", label="Perplexity Key", visible=False)
            abstain_margin = gr.Textbox(label="Abstain Margin (optional)", placeholder="e.g., 0.02")

        btn = gr.Button("Decide", variant="primary")
        out_md = gr.Markdown()
        out_json = gr.Code(label="Raw JSON Result", language="json")

        def toggle(group):
            show_ratio = group in ("mink++","mink")
            show_ppl   = group == "perplexity"
            return gr.update(visible=show_ratio), gr.update(visible=show_ppl)

        metric_group.change(toggle, inputs=[metric_group], outputs=[ratio, ppl_key])

        btn.click(infer,
                  inputs=[text, model, cfg, metric_group, ratio, ppl_key, mode, abstain_margin],
                  outputs=[out_md, out_json])
    return demo

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=8000, inbrowser=False, share=False)

# python -m scripts.gradio_text
# ssh -L 8000:127.0.0.1:8000 -p 5110 wanghuili@101.6.70.28
# http://127.0.0.1:8000
