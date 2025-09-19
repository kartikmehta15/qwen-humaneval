#!/usr/bin/env python3
import os, sys, json, argparse
from pathlib import Path

# --- repo import path ---
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_DIR   = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# --- local imports ---
from postprocessing import PostProcessor
from eval_utils import dump_for_eval, eval_pass1


def rewrite_with_pp(src_path: Path, dst_path: Path, version: str):
    """Re-process a combined jsonl file with a given PostProcessor version."""
    PostProcessor.set_version(version)
    with src_path.open() as r, dst_path.open("w") as w:
        for line in r:
            rec = json.loads(line)
            raw = rec.get("raw_text", rec.get("completion", ""))
            rec["completion"] = PostProcessor.normalize_body(raw)
            w.write(json.dumps(rec) + "\n")


def sweep_pp_versions(src: Path, run_dir: Path, versions):
    rows = []
    for ver in versions:
        dst = run_dir / f"pp_{ver}__{src.name}"
        rewrite_with_pp(src, dst, ver)
        tag = f"pp_{ver}__{src.stem}"
        samples, probs, N, cr, avg_len, med = dump_for_eval(dst, run_dir, tag)
        pass1 = eval_pass1(str(samples), str(probs), n_workers=12)
        rows.append({
            "version": ver,
            "pass@1": pass1,
            "compile_rate": cr,
            "N": N,
            "avg_len": avg_len,
            "median_len": med,
            "path": str(dst)
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="combined_*.jsonl to re-process")
    ap.add_argument("--run-dir", default="he_runs")
    ap.add_argument("--versions", nargs="+", default=["v1", "v2", "v3"])
    args = ap.parse_args()

    src = Path(args.src)
    run_dir = Path(args.run_dir); run_dir.mkdir(parents=True, exist_ok=True)

    rows = sweep_pp_versions(src, run_dir, args.versions)

    print("\n=== Post-processing sweep (no inference) ===")
    print("version | pass@1 | compile |   N | avg_len | median | path")
    print("----------------------------------------------------------------")
    for r in rows:
        print(f"{r['version']:>7} | {r['pass@1']:.3f} | {r['compile_rate']:.3f} | "
              f"{r['N']:>3} | {r['avg_len']:>7} | {r['median_len']:>6} | {r['path']}")


if __name__ == "__main__":
    main()
