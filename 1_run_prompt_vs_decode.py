#!/usr/bin/env python3
# 1_run_prompt_vs_decode.py  (two sequential sweeps)

import os, sys, json
from pathlib import Path

# --- repo import path ---
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_DIR   = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# --- local helpers ---
from load_datasets import load_humaneval
from decode_variants import DECODE_VARIANTS
from experiments import generate_and_eval

# -------------------------
# Config
# -------------------------
API_BASE = os.getenv("API_BASE", "http://10.XX.XX.XX:8001/v1")
MODEL_ID = os.getenv("MODEL_ID", "qwen-coder-0_5b-instruct")
TOKEN    = os.getenv("VLLM_API_KEY", "<RANDOM_PASSWORD>")
USE_CHAT = True

PROMPT_VARIANTS = ["raw", "hardened_v1", "hardened_v2", "icl_v2"]
DECODE_CHOICES  = DECODE_VARIANTS

USE_SAMPLE  = True
TOTAL_ITEMS = 164
SAMPLE_FRAC = 0.2
N_ITEMS     = int(round(TOTAL_ITEMS * SAMPLE_FRAC)) if USE_SAMPLE else TOTAL_ITEMS

RUN_DIR = Path("he_runs"); RUN_DIR.mkdir(parents=True, exist_ok=True)


def print_summary(rows, sweep_name: str):
    """Pretty print a summary table for a sweep."""
    print(f"\n=== Summary: {sweep_name} ===")
    print("prompt_id      | decode        | pass@1 | compile | avg_len | median | gen_s | out")
    print("---------------------------------------------------------------------------------")
    for r in rows:
        out = Path(r['combined_path']).name
        print(f"{r['prompt_id']:<14} | {r['decode']:<12} | {r['pass@1']:.3f} | {r['compile_rate']:.3f} | "
              f"{r['avg_len']:<7} | {r['median_len']:<6} | {r['gen_time_s']:<5} | {out}")


def sweep_prompts(ds, fixed_decode):
    """Sweep through various prompts on sample data to determine best prompt."""
    rows = []
    print(f"\n=== Sweep 1: Fixed decode={fixed_decode['name']} | varying prompts ===")
    for prompt_id in PROMPT_VARIANTS:
        stats = generate_and_eval(
            ds=ds,
            prompt_id=prompt_id,
            dec=fixed_decode,
            run_dir=RUN_DIR,
            api_base=API_BASE,
            model_id=MODEL_ID,
            token=TOKEN,
            use_chat=USE_CHAT,
            n_workers=8,
        )
        print(json.dumps(stats, indent=2))
        rows.append(stats)
        sys.stdout.flush()

    print_summary(rows, f"Fixed decode={fixed_decode['name']} | varying prompts")
    return rows


def sweep_decodes(ds, fixed_prompt):
    """Sweep through various decoding params on sample data to determine best one."""
    rows = []
    print(f"\n=== Sweep 2: Fixed prompt={fixed_prompt} | varying decodes ===")
    for dec in DECODE_CHOICES:
        stats = generate_and_eval(
            ds=ds,
            prompt_id=fixed_prompt,
            dec=dec,
            run_dir=RUN_DIR,
            api_base=API_BASE,
            model_id=MODEL_ID,
            token=TOKEN,
            use_chat=USE_CHAT,
            n_workers=8,
        )
        print(json.dumps(stats, indent=2))
        rows.append(stats)
        sys.stdout.flush()

    print_summary(rows, f"Fixed prompt={fixed_prompt} | varying decodes")
    return rows


def main():
    ds = load_humaneval(run_sample=USE_SAMPLE, n=N_ITEMS, shuffle=True, seed=42)
    print(f"[data] {len(ds)} tasks{' (~sample)' if USE_SAMPLE else ' (full)'}")

    # Sweep 1: fix decode[0], vary prompts
    sweep_prompts(ds, DECODE_CHOICES[0])

    # Sweep 2: fix prompt=hardened_v2, vary decodes
    sweep_decodes(ds, "hardened_v2")


if __name__ == "__main__":
    main()
