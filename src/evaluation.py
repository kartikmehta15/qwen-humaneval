from pathlib import Path
import json, statistics as stats, inspect, os, subprocess, sys

def _ensure_repo():
    repo_root = Path.home() / "human_eval_official"
    if not repo_root.exists():
        subprocess.run(["git","clone","--depth","1","https://github.com/openai/human-eval.git", str(repo_root)], check=True)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from human_eval.evaluation import evaluate_functional_correctness  # noqa
    return evaluate_functional_correctness

def write_combined(records, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as w:
        for r in records:
            w.write(json.dumps(r) + "\n")

def write_samples_and_probs(combined_path: Path, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = out_dir / f"samples_{tag}.jsonl"
    probs   = out_dir / f"problems_{tag}.jsonl"
    attempted, lengths, ok = 0, [], 0

    with open(combined_path) as r, samples.open("w") as sw, probs.open("w") as pw:
        for line in r:
            obj = json.loads(line); attempted += 1
            body = obj["completion"]; lengths.append(len(body.strip()))
            try:
                compile(obj["prompt"] + body, "<chk>", "exec"); ok += 1
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
        "compile_rate": ok/attempted if attempted else 0.0,
        "avg_len": round(stats.mean(lengths),1) if lengths else 0.0,
        "median_len": int(stats.median(lengths)) if lengths else 0,
    }

def run_humaneval(samples_path: str, problems_path: str, n_workers: int = 8, timeout: int = 15):
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
