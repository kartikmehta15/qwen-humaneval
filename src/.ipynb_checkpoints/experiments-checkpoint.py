# src/experiments.py
#!/usr/bin/env python3
import time, json, requests
from pathlib import Path

from postprocessing import PostProcessor, extract_def_from_prompt
from prompts import get_header
from eval_utils import dump_for_eval, eval_pass1


def _make_instr(def_src: str, header_str: str) -> str:
    """Header + def stub + open <sol> tag (model must close it)."""
    return header_str.rstrip() + "\n\n" + f"{def_src.rstrip()}\n" + "<sol>\n"


def sync_infer_one(
    ex: dict,
    header_str: str,
    dec: dict,
    *,
    api_base: str,
    model_id: str,
    token: str,
    use_chat: bool = True,
):
    """
    Synchronous single-sample inference against an OpenAI-compatible endpoint.
    """
    url = f"{api_base}/chat/completions" if use_chat else f"{api_base}/completions"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # NOTE: use the static method on PostProcessor (not a free function):
    def_src = extract_def_from_prompt(ex["prompt"], ex["entry_point"])
    # def_src = PostProcessor.extract_def_from_prompt(ex["prompt"], ex["entry_point"])

    payload = {
        "model": model_id,
        "max_tokens": dec["max_tokens"],
        "temperature": dec["temperature"],
        "top_p": dec["top_p"],
    }
    if dec.get("stop"):
        payload["stop"] = dec["stop"]

    if use_chat:
        payload["messages"] = [
            {"role": "system", "content": "You are a precise Python coding assistant. Reply with code only."},
            {"role": "user", "content": _make_instr(def_src, get_header(header_str)) if isinstance(header_str, str) else _make_instr(def_src, get_header(header_str))},
        ]
    else:
        payload["prompt"] = _make_instr(def_src, get_header(header_str) if not isinstance(header_str, str) else header_str)

    r = requests.post(url, headers=headers, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    ch   = data["choices"][0]
    text = (ch.get("message") or {}).get("content") or ch.get("text") or ""

    return {
        "task_id": ex["task_id"],
        "prompt": ex["prompt"],
        "entry_point": ex["entry_point"],
        "canonical_solution": ex["canonical_solution"],
        "test": ex["test"],
        "raw_text": text,
    }


def generate_and_eval(
    ds,
    prompt_id: str,
    dec: dict,
    *,
    run_dir: Path,
    api_base: str,
    model_id: str,
    token: str,
    use_chat: bool = True,
    n_workers: int = 8,
):
    """
    Mini-experiment:
      - build header from prompts.get_header(prompt_id)
      - loop sync inference
      - postprocess with PostProcessor.normalize_body
      - write combined jsonl
      - dump evaluator files and compute pass@1
    """
    header_str = get_header(prompt_id)  # prompt_id like "raw", "hardened_v2", "icl_v2"
    tag = f"{prompt_id}__{dec['name']}"
    combined_path = run_dir / f"combined_{tag}.jsonl"

    t0 = time.time()
    records = []
    for ex in ds:
        rec = sync_infer_one(
            ex, header_str, dec,
            api_base=api_base, model_id=model_id, token=token, use_chat=use_chat
        )
        body = PostProcessor.normalize_body(rec["raw_text"])
        records.append({**rec, "completion": body})

    with combined_path.open("w") as w:
        for r in records:
            w.write(json.dumps(r) + "\n")

    gen_secs = time.time() - t0
    samples_path, probs_path, attempted, compile_rate, avg_len, med_len = dump_for_eval(combined_path, run_dir, tag)
    pass1 = eval_pass1(str(samples_path), str(probs_path), n_workers=n_workers)

    return {
        "tag": tag,
        "prompt_id": prompt_id,
        "decode": dec["name"],
        "attempted": attempted,
        "pass@1": pass1,
        "compile_rate": compile_rate,
        "avg_len": round(avg_len, 1),
        "median_len": med_len,
        "gen_time_s": round(gen_secs, 2),
        "combined_path": str(combined_path),
        "samples_path": str(samples_path),
        "probs_path": str(probs_path),
    }
