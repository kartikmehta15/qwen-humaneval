#!/usr/bin/env python3
"""
4_qwen_eval_assignment.py
=========================
End-to-end HumanEval pipeline for Qwen models served via vLLM.

Steps:
1. Load HumanEval dataset (full or sampled).
2. Generate completions via vLLM (chat/completions API).
3. Post-process outputs (normalize, strip fences/imports).
4. Evaluate with official HumanEval functional correctness.
5. Print summary metrics.

Artifacts are saved in `he_runs/` as combined JSONL + eval files.
"""

import os, sys, json, time, asyncio
from pathlib import Path
import nest_asyncio

nest_asyncio.apply()

# --- repo import path ---
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_DIR   = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# --- local helpers ---
from load_datasets import load_humaneval
from postprocessing import PostProcessor
from predictor import infer_async
from eval_utils import dump_for_eval, eval_pass1
from prompts import get_header

# -------------------------
# Config
# -------------------------
API_BASE = os.getenv("API_BASE", "http://10.182.0.6:8001/v1")
MODEL_ID = os.getenv("MODEL_ID", "qwen-coder-0_5b-instruct")
TOKEN    = os.getenv("VLLM_API_KEY", "RANDOM_PASSWORD")
USE_CHAT = True

RUN_DIR = Path("he_runs"); RUN_DIR.mkdir(parents=True, exist_ok=True)

USE_SAMPLE  = False
TOTAL_ITEMS = 164
SAMPLE_FRAC = 0.20
N_ITEMS     = int(round(TOTAL_ITEMS * SAMPLE_FRAC)) if USE_SAMPLE else TOTAL_ITEMS

# PROFILE = dict(
#     name="baseline",
#     temperature=0.2,
#     top_p=0.95,
#     max_tokens=520,
#     stop=["</sol>"],
#     concurrency=16,
#     eval_workers=8,
# )

PROFILE = dict(
    name="baseline",
    temperature=0.0,
    top_p=1.0,
    max_tokens=512,
    stop=["</sol>"],
    concurrency=16,
    eval_workers=8,
)

def main():
    # --- Load dataset ---
    ds = load_humaneval(run_sample=USE_SAMPLE, n=N_ITEMS, shuffle=True, seed=42)
    print(f"[data] {len(ds)} tasks{' (~sample)' if USE_SAMPLE else ' (full)'}")

    # --- Header/prompt style ---
    prompt_id = "hardened_v2"
    header_str = get_header(prompt_id)
    PostProcessor.set_version("v3")

    # --- Run inference ---
    print(f"\n=== Inference: profile={PROFILE['name']} | prompt={prompt_id} ===")
    t0 = time.time()
    loop = asyncio.get_event_loop()
    records = loop.run_until_complete(
        infer_async(
            ds,
            API_BASE,
            MODEL_ID,
            TOKEN,
            USE_CHAT,
            header_str,
            PROFILE  # <-- pass whole dict
        )
    )
    gen_s = time.time() - t0

    # --- Save combined outputs ---
    tag = f"{PROFILE['name']}__{prompt_id}"
    combined = RUN_DIR / f"combined_{tag}.jsonl"
    with combined.open("w") as w:
        for rec in records:
            body = PostProcessor.normalize_body(rec.get("raw_text", rec.get("completion", "")))
            # body = PostProcessor.normalize_body(rec["raw_text"])
            w.write(json.dumps({
                "task_id": rec.get("task_id", ""),
                "prompt": rec.get("prompt", ""),
                "entry_point": rec.get("entry_point", ""),
                "canonical_solution": rec.get("canonical_solution", ""),
                "test": rec.get("test", ""),
                "raw_text": rec.get("raw_text", ""),
                "completion": body,
            }) + "\n")

            # w.write(json.dumps({**rec, "completion": body}) + "\n")

    # --- Evaluate ---
    samples, probs, N, cr, avg, med = dump_for_eval(combined, RUN_DIR, tag)
    t1 = time.time()
    pass1 = eval_pass1(str(samples), str(probs), n_workers=PROFILE["eval_workers"])
    eval_s = time.time() - t1

    # --- Report ---
    print("\n=== Results ===")
    print("profile   | pass@1 | compile |   N | avg_len | median | gen_s | ex/s | eval_s | path")
    print("----------------------------------------------------------------------------------------")
    print(
        f"{PROFILE['name']:<9} | {pass1:.3f} | {cr:.3f} | {N:>3} | {avg:>7} | {med:>6} | "
        f"{gen_s:>5.2f} | {round(N/gen_s,2) if gen_s>0 else None:>4} | {eval_s:>6.2f} | {combined}"
    )

if __name__ == "__main__":
    main()
