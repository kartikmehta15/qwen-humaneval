# Lightweight self-consistency & one-pass self-repair.

from dataclasses import dataclass
from typing import List, Tuple, Callable, Optional
import re

@dataclass
class Candidate:
    body: str
    raw_text: str
    compiled: bool

def _normalize_for_vote(body: str) -> str:
    # Normalize whitespace & trailing spaces for majority voting
    lines = [ln.rstrip() for ln in body.strip().splitlines() if ln.strip()]
    return "\n".join(lines)

def reduce_candidates(cands: List[Candidate]) -> Candidate:
    """
    Preference order:
      1) Any compiled candidates present → majority by normalized string; tie → shortest
      2) Else: majority across all; tie → shortest
    """
    if not cands:
        raise ValueError("No candidates")

    compiled = [c for c in cands if c.compiled]
    pool = compiled if compiled else cands

    buckets = {}
    for c in pool:
        k = _normalize_for_vote(c.body)
        buckets.setdefault(k, []).append(c)

    # majority bucket
    best_key = max(buckets.keys(), key=lambda k: len(buckets[k]))
    best_bucket = buckets[best_key]

    # choose shortest body within bucket
    best = min(best_bucket, key=lambda c: len(c.body))
    return best

# Optional: one-pass self-critique/repair
def self_repair(prompt: str, broken_body: str, send_fn: Callable[[str], str]) -> Optional[str]:
    """
    send_fn: a function that takes a user message and returns model text.
    Returns a repaired body or None if still invalid.
    """
    instruction = (
        "Fix the following Python function BODY so it compiles and satisfies the docstring.\n"
        "Rules: return ONLY the body between <sol> and </sol>, exactly 4-space indents, no imports/def/tests.\n\n"
        f"{prompt}\n\nBroken body:\n<sol>\n{broken_body}\n</sol>\n"
        "Reply with only the corrected body inside <sol> and </sol>."
    )
    text = send_fn(instruction)
    # caller should re-run their usual post-processor on this text to get the body
    return text or None
