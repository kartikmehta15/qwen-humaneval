#!/usr/bin/env python3
"""
4_qwen_eval_assignment.py
Final assignment runner:
- Uses fixed prompt/decode/post-processing settings
- Runs inference synchronously (no async)
- Evaluates pass@1 with HumanEval
"""

import os, sys, json, time
from pathlib import Path

# --- repo import path ---
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_DIR   = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# --- local imports ---
from load_datasets import load_humaneval
from prompts import get_header
from postprocessing import PostProcessor
from eval_utils import dump_for_eval, eval_pass1
from experiments import sync_infer_one   # already defined for sync inference
from decode_variants import DECODE_VARIANTS


# -------------------------
# Config
# -------------------------
API_BASE = os.getenv("API_BASE", "http://XX.XX.XX.XX:8001/v1")
MODEL_ID = os.getenv("MODEL_ID", "qwen-coder-0_5b-instruct")
TOKEN    = os.getenv("VLLM_API_KEY", "<RANDOM_PASSWORD>")
USE_CHAT = True

RUN_DIR = Path("he_runs"); RUN_DIR.mkdir(parents=True, exist_ok=True)

# Choose defaults
PROMPT_ID   = "hardened_v2"       # best performing
DECODE      = DECODE_VARIANTS[4]  # baseline decode config
PP_VERSION  = "v3"                # best post-processing version


def main():
    # --- load full dataset ---
    ds = load_humaneval(run_sample=False, n=0)
    print(f"[data] {len(ds)} tasks (full)")

    # --- get header / pp ---
    header_str = get_header(PROMPT_ID)
    PostProcessor.set_version(PP_VERSION)

    # --- run inference (sync) ---
    print(f"\n=== Inference: profile=baseline | prompt={PROMPT_ID} | decode={DECODE['name']} | pp={PP_VERSION} ===")
    t0 = time.time()
    records = []
    for ex in ds:
        rec = sync_infer_one(
            ex,
            header_str,
            DECODE,
            api_base=API_BASE,
            model_id=MODEL_ID,
            token=TOKEN,
            use_chat=USE_CHAT,
        )
        body = PostProcessor.normalize_body(rec["raw_text"])
        records.append({**rec, "completion": body})
    gen_s = time.time() - t0

    # --- dump combined ---
    tag = f"final__{PROMPT_ID}__{DECODE['name']}__{PP_VERSION}"
    combined = RUN_DIR / f"combined_{tag}.jsonl"
    with combined.open("w") as w:
        for r in records:
            w.write(json.dumps(r) + "\n")

    # --- evaluation ---
    samples, probs, N, cr, avg, med = dump_for_eval(combined, RUN_DIR, tag)
    t1 = time.time()
    pass1 = eval_pass1(str(samples), str(probs), n_workers=8)
    eval_s = time.time() - t1

    # --- results table ---
    print("\n=== Results ===")
    print("profile   | pass@1 | compile |   N | avg_len | median | gen_s | eval_s | path")
    print("----------------------------------------------------------------------------------------")
    print(f"{'final':<9} | {pass1:.3f} | {cr:.3f} | {N:>3} | {avg:>7} | {med:>6} | "
          f"{gen_s:>5.2f} | {eval_s:>6.2f} | {combined}")


if __name__ == "__main__":
    main()
