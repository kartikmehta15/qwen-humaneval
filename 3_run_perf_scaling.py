#!/usr/bin/env python3
"""
3_run_perf_scaling.py
=====================
Compare baseline vs optimized inference profiles on HumanEval.

Sync (non-async) version:
- Calls vLLM endpoint via requests (no event loop).
- Runs sequentially through dataset.
- Saves results and evaluates pass@1, compile rate, etc.
"""

import os, sys, json, time, requests
from pathlib import Path

# --- repo import path ---
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_DIR   = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# --- local imports ---
from load_datasets import load_humaneval
from postprocessing import PostProcessor
from eval_utils import dump_for_eval, eval_pass1
from prompts import get_header

# -------------------------
# Config
# -------------------------
API_BASE = os.getenv("API_BASE", "http://10.XX.XX.XX:8001/v1")
MODEL_ID = os.getenv("MODEL_ID", "qwen-coder-0_5b-instruct")
TOKEN    = os.getenv("VLLM_API_KEY", "<RANDOM_PASSWORD>")
USE_CHAT = True

RUN_DIR = Path("he_runs"); RUN_DIR.mkdir(parents=True, exist_ok=True)

BASELINE = dict(
    name="baseline",
    temperature=0.2,
    top_p=1.0,
    max_tokens=512,
    stop=None,
    eval_workers=8,
)

OPTIMIZED = dict(
    name="optimized",
    temperature=0.2,
    top_p=0.95,
    max_tokens=320,
    stop=["</sol>"],
    eval_workers=32,
)


# -------------------------
# Helper: build request
# -------------------------
def _make_instr(def_src: str, header_str: str) -> str:
    return header_str.rstrip() + "\n\n" + f"{def_src.rstrip()}\n" + "<sol>\n"


def sync_infer_one(ex, header_str, prof):
    """Call vLLM sync and return record with raw_text"""
    url = f"{API_BASE}/chat/completions" if USE_CHAT else f"{API_BASE}/completions"
    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

    payload = {
        "model": MODEL_ID,
        "max_tokens": prof["max_tokens"],
        "temperature": prof["temperature"],
        "top_p": prof["top_p"],
    }
    if prof.get("stop"):
        payload["stop"] = prof["stop"]

    if USE_CHAT:
        payload["messages"] = [
            {"role": "system", "content": "You are a precise Python coding assistant. Reply with code only."},
            {"role": "user", "content": _make_instr(ex["prompt"], header_str)},
        ]
    else:
        payload["prompt"] = _make_instr(ex["prompt"], header_str)

    r = requests.post(url, headers=headers, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    ch   = data["choices"][0]
    text = (ch.get("message") or {}).get("content") or ch.get("text") or ""

    return {
        "task_id": ex.get("task_id", ""),
        "prompt": ex.get("prompt", ""),
        "entry_point": ex.get("entry_point", ""),
        "canonical_solution": ex.get("canonical_solution", ""),
        "test": ex.get("test", ""),
        "raw_text": text,
    }


# -------------------------
# Main
# -------------------------
def main():
    ds = load_humaneval(run_sample=True, n=32, shuffle=True, seed=42)
    print(f"[data] {len(ds)} tasks (~sample)")

    fixed_prompt = "hardened_v2"
    header_str = get_header(fixed_prompt)
    PostProcessor.set_version("v3")

    rows = []
    for prof in (BASELINE, OPTIMIZED):
        print(f"\n=== Running profile: {prof['name']} | prompt={fixed_prompt} ===")
        t0 = time.time()

        records = []
        for ex in ds:
            rec = sync_infer_one(ex, header_str, prof)
            body = PostProcessor.normalize_body(rec.get("raw_text", ""))
            records.append({**rec, "completion": body})

        gen_s = time.time() - t0

        tag = f"perf_{prof['name']}"
        combined = RUN_DIR / f"combined_{tag}.jsonl"
        with combined.open("w") as w:
            for r in records:
                w.write(json.dumps(r) + "\n")

        samples, probs, N, cr, avg, med = dump_for_eval(combined, RUN_DIR, tag)
        t1 = time.time()
        pass1 = eval_pass1(str(samples), str(probs), n_workers=prof["eval_workers"])
        eval_s = time.time() - t1

        rows.append(
            (
                prof["name"],
                pass1,
                cr,
                N,
                avg,
                med,
                gen_s,
                round(N / gen_s, 2) if gen_s > 0 else None,
                eval_s,
                str(combined),
            )
        )

    # --- Final summary ---
    print(f"\n=== Baseline vs Optimized (prompt={fixed_prompt}) ===")
    print("profile   | pass@1 | compile |   N | avg_len | median | gen_s | ex/s | eval_s | path")
    print("----------------------------------------------------------------------------------------")
    for name, p1, cr, N, avg, med, gs, exs, es, path in rows:
        print(
            f"{name:<9} | {p1:.3f} | {cr:.3f} | {N:>3} | {avg:>7} | {med:>6} | "
            f"{gs:>5.2f} | {exs!s:>4} | {es:>6.2f} | {path}"
        )


if __name__ == "__main__":
    main()
