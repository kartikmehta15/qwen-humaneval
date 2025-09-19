# Throughput / evaluation profiles (perf vs quality)

BASELINE = {
    "name": "baseline",
    "concurrency": 8,
    "eval_workers": 8,
    "stop": None,
    "request_timeout": 180,
}

OPTIMIZED = {
    "name": "optimized",
    "concurrency": 64,      # tune to GPU capacity
    "eval_workers": 32,
    "stop": ["</sol>"],     # early stop on tag
    "request_timeout": 180,
}

PROFILES = {
    "baseline": BASELINE,
    "optimized": OPTIMIZED,
}

def get_speed(speed_id: str) -> dict:
    if speed_id not in PROFILES:
        raise ValueError(f"Unknown speed_id: {speed_id}. Choices: {list(PROFILES)}")
    return PROFILES[speed_id]
