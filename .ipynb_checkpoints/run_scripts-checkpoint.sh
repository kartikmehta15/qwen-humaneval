#!/usr/bin/env bash
# ==============================================================
# run_all.sh
# Master script to run all HumanEval experiments in sequence
# ==============================================================
# Usage:
#   bash run_all.sh
#
# Notes:
# - Make sure your vLLM server is running before starting.
# - All results are written to the `he_runs/` folder.
# - Each section logs its stdout/stderr to timestamped log files.
# ==============================================================

set -e  # exit if any command fails
set -o pipefail

# --- Config ---
RUN_DIR="he_runs"
mkdir -p $RUN_DIR
TS=$(date +"%Y%m%d_%H%M%S")

# --- 1. Prompt and Decode sweep ablation ---
echo ">>> [2/4] Running 1_run_prompt_vs_decode.py (prompt & decode sweeps)"
python 1_run_prompt_vs_decode.py 2>&1 | tee $RUN_DIR/log_prompt_decode_$TS.txt

# === Summary: Fixed prompt=hardened_v2 | varying decodes ===
# prompt_id      | decode        | pass@1 | compile | avg_len | median | gen_s | out
# ---------------------------------------------------------------------------------
# hardened_v2    | t0.2_p0.95_len512 | 0.455 | 0.879 | 209.4   | 195    | 115.9 | combined_hardened_v2__t0.2_p0.95_len512.jsonl
# hardened_v2    | t0.2_p1.0_len512 | 0.394 | 0.818 | 192.9   | 168    | 102.48 | combined_hardened_v2__t0.2_p1.0_len512.jsonl
# hardened_v2    | t0.1_p0.95_len512 | 0.455 | 0.879 | 196.4   | 187    | 109.22 | combined_hardened_v2__t0.1_p0.95_len512.jsonl
# hardened_v2    | t0.3_p0.9_len512 | 0.394 | 0.848 | 216.9   | 218    | 114.17 | combined_hardened_v2__t0.3_p0.9_len512.jsonl
# hardened_v2    | t0.0_p1.0_len512 | 0.485 | 0.848 | 213.8   | 218    | 99.34 | combined_hardened_v2__t0.0_p1.0_len512.jsonl
# hardened_v2    | t0.7_p0.9_len512 | 0.273 | 0.909 | 170.8   | 136    | 119.32 | combined_hardened_v2__t0.7_p0.9_len512.jsonl
# hardened_v2    | t1.0_p1.0_len512 | 0.273 | 0.879 | 170.7   | 109    | 92.23 | combined_hardened_v2__t1.0_p1.0_len512.jsonl

# === Summary: Fixed decode=t0.2_p0.95_len512 | varying prompts ===
# prompt_id      | decode        | pass@1 | compile | avg_len | median | gen_s | out
# ---------------------------------------------------------------------------------
# raw            | t0.2_p0.95_len512 | 0.364 | 0.939 | 126.4   | 81     | 94.25 | combined_raw__t0.2_p0.95_len512.jsonl
# hardened_v1    | t0.2_p0.95_len512 | 0.364 | 0.848 | 164.2   | 147    | 106.16 | combined_hardened_v1__t0.2_p0.95_len512.jsonl
# hardened_v2    | t0.2_p0.95_len512 | 0.485 | 0.909 | 218.5   | 214    | 107.18 | combined_hardened_v2__t0.2_p0.95_len512.jsonl
# icl_v2         | t0.2_p0.95_len512 | 0.364 | 0.939 | 160.9   | 150    | 113.32 | combined_icl_v2__t0.2_p0.95_len512.jsonl

# --- 2. Post-processing ablation ---
# You may need to edit --src path below to point to a recent combined_*.jsonl
LATEST_COMBINED=$(ls -t $RUN_DIR/combined_*.jsonl | head -n1)
echo ">>> [3/4] Running 2_run_postprocess_ablation.py on $LATEST_COMBINED"
LATEST_COMBINED=he_runs/combined_hardened_v2__t0.0_p1.0_len512.jsonl
python 2_run_postprocess_ablation.py --src $LATEST_COMBINED --versions v1 v2 v3 2>&1 | tee $RUN_DIR/log_postprocess_$TS.txt

# --- 3. Performance scaling baseline vs optimized ---
echo ">>> [4/4] Running 3_run_perf_scaling.py (baseline vs optimized)"
python 3_run_perf_scaling.py 2>&1 | tee $RUN_DIR/log_perf_scaling_$TS.txt

# === Baseline vs Optimized (prompt=hardened_v2) ===
# profile   | pass@1 | compile |   N | avg_len | median | gen_s | ex/s | eval_s | path
# ----------------------------------------------------------------------------------------
# baseline  | 0.500 | 1.000 |  32 |   696.0 |    614 | 97.56 | 0.33 |   0.82 | he_runs/combined_perf_baseline.jsonl
# optimized | 0.531 | 0.938 |  32 |   673.0 |    664 | 107.12 |  0.3 |   1.25 | he_runs/combined_perf_optimized.jsonl

# --- 4. End-to-end baseline run (sanity check) ---
echo ">>> [1/4] Running qwen_eval_assignment.py (end-to-end pipeline)"
python 4_qwen_eval_assignment.py 2>&1 | tee $RUN_DIR/log_assignment_$TS.txt

# decode                      | post_processing  | promptid    | pass@1 | compile |   N | avg_len | median | gen_s | eval_s |
# ----------------------------------------------------------------------------------------
# temperature=0.2, top_p=0.95 | v3               | hardened_v2 | 0.543  | 0.994   | 164 |   679.1 |    610 | 562.68 |  18.31 
# temperature=0, top_p=1.0    | v3               | hardened_v2 | 0.549  | 1.000   | 164 |   696.9 |    623 | 498.14 |  18.26

# --- Done ---
echo "=============================================================="
echo "All experiments finished. Logs and outputs are in $RUN_DIR/"
echo "Latest logs tagged with timestamp: $TS"
echo "=============================================================="
