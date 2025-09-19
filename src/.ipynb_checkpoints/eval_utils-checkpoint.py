import json, statistics as stats
from pathlib import Path

# Ensure HumanEval repo once
def _ensure_repo():
    repo_root = Path.home() / "human_eval_official"
    if not repo_root.exists():
        subprocess.run(
            ["git","clone","--depth","1","https://github.com/openai/human-eval.git", str(repo_root)],
            check=True
        )
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from human_eval.evaluation import evaluate_functional_correctness  # noqa
    return evaluate_functional_correctness

def write_samples_and_probs(combined_path: Path, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = out_dir / f"samples_{tag}.jsonl"
    probs   = out_dir / f"problems_{tag}.jsonl"
    attempted, lengths, comp_ok = 0, [], 0

    with open(combined_path) as r, samples.open("w") as sw, probs.open("w") as pw:
        for line in r:
            obj = json.loads(line); attempted += 1
            body = obj["completion"]; lengths.append(len(body.strip()))
            try:
                compile(obj["prompt"] + body, "<chk>", "exec"); comp_ok += 1
            except Exception:
                pass
            sw.write(json.dumps({"task_id": obj["task_id"], "completion": body}) + "\n")
            pw.write(json.dumps({
                "task_id": obj["task_id"],
                "prompt": obj["prompt"],
                "entry_point": obj["entry_point"],
                "canonical_solution": obj["canonical_solution"],
                "test": obj["test"],
            }) + "\n")

    return {
        "samples_path": str(samples),
        "problems_path": str(probs),
        "attempted": attempted,
        "compile_rate": comp_ok/attempted if attempted else 0.0,
        "avg_len": round(stats.mean(lengths), 1) if lengths else 0.0,
        "median_len": int(stats.median(lengths)) if lengths else 0,
    }

def run_humaneval(samples_path: str, problems_path: str, n_workers: int = 8, timeout: int = 15) -> float:
    evaluate_functional_correctness = _ensure_repo()
    sig = inspect.signature(evaluate_functional_correctness)
    kwargs = {}
    if "k" in sig.parameters: kwargs["k"] = [1]
    if "n_workers" in sig.parameters: kwargs["n_workers"] = n_workers
    elif "n_processes" in sig.parameters: kwargs["n_processes"] = n_workers
    if "timeout" in sig.parameters: kwargs["timeout"] = timeout
    if "problem_file" in sig.parameters: kwargs["problem_file"] = problems_path
    os.environ.setdefault("TMPDIR", "/dev/shm")
    res = evaluate_functional_correctness(samples_path, **kwargs)
    return res.get("pass@1") or res.get("pass@1,exact") or 0.0



# src/eval_utils.py
import os
import json
import inspect
import statistics as stats
from pathlib import Path
from typing import Tuple

# -- lazy import of HumanEval evaluator (and auto-clone if missing)
def _ensure_humaneval_repo() -> Path:
    repo_root = Path.home() / "human_eval_official"
    if not repo_root.exists():
        import subprocess
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/openai/human-eval.git", str(repo_root)],
            check=True,
        )
    return repo_root

def _import_evaluator():
    repo_root = _ensure_humaneval_repo()
    import sys
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from human_eval.evaluation import evaluate_functional_correctness
    return evaluate_functional_correctness

def dump_for_eval(
    combined_path: Path,
    run_dir: Path,
    tag: str,
) -> Tuple[Path, Path, int, float, float, int]:
    """
    Prepare HumanEval-compatible files (samples + problems) from a combined predictions JSONL.

    Args:
        combined_path: Path to combined_*.jsonl with fields (task_id, prompt, completion, etc.)
        run_dir: output directory
        tag: string tag for filenames

    Returns:
        (samples_path, probs_path, attempted, compile_rate, avg_len, med_len)
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    samples = run_dir / f"samples_{tag}.jsonl"
    probs   = run_dir / f"probs_{tag}.jsonl"

    attempted, lengths, comp_ok = 0, [], 0

    with open(combined_path) as r, samples.open("w") as sw, probs.open("w") as pw:
        for line in r:
            obj = json.loads(line)
            attempted += 1

            body = obj.get("completion", "")
            lengths.append(len(body.strip()))

            # compile gate against the original prompt + body
            try:
                compile(obj["prompt"] + body, "<chk>", "exec")
                comp_ok += 1
            except Exception:
                pass

            # HumanEval expected inputs
            sw.write(json.dumps({"task_id": obj["task_id"], "completion": body}) + "\n")
            pw.write(json.dumps({
                "task_id": obj["task_id"],
                "prompt": obj["prompt"],
                "entry_point": obj["entry_point"],
                "canonical_solution": obj["canonical_solution"],
                "test": obj["test"],
            }) + "\n")

    avg_len = round(stats.mean(lengths), 1) if lengths else 0.0
    med_len = int(stats.median(lengths)) if lengths else 0

    return samples, probs, attempted, (comp_ok / attempted if attempted else 0.0), avg_len, med_len


def eval_pass1(
    samples_path: Path,
    probs_path: Path,
    n_workers: int = 8,
    timeout: int = 15,
) -> float:
    """
    Run HumanEval functional correctness on prepared files and return pass@1 as float.

    Args:
        samples_path: Path to samples_*.jsonl generated by dump_for_eval
        probs_path:   Path to probs_*.jsonl generated by dump_for_eval
        n_workers:    Parallel workers/processes for the evaluator
        timeout:      Seconds per task

    Returns:
        pass@1 as float (0.0 - 1.0)
    """
    evaluate_functional_correctness = _import_evaluator()

    # Prefer using a faster temp filesystem if available to reduce FS overhead
    os.environ.setdefault("TMPDIR", "/dev/shm" if Path("/dev/shm").exists() else "/tmp")

    sig = inspect.signature(evaluate_functional_correctness)
    kwargs = {}

    # Handle arg name differences across versions
    if "k" in sig.parameters:
        kwargs["k"] = [1]
    if "n_workers" in sig.parameters:
        kwargs["n_workers"] = n_workers
    elif "n_processes" in sig.parameters:
        kwargs["n_processes"] = n_workers
    if "timeout" in sig.parameters:
        kwargs["timeout"] = timeout
    if "problem_file" in sig.parameters:
        kwargs["problem_file"] = str(probs_path)

    results = evaluate_functional_correctness(str(samples_path), **kwargs)
    return float(results.get("pass@1") or results.get("pass@1,exact") or 0.0)
