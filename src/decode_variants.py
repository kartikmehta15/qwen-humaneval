# Decoding grids (max_tokens is here for convenience; override via CLI if needed)

# DECODE_VARIANTS = {
#     "t02_p095_len512": {"temperature": 0.2, "top_p": 0.95, "max_tokens": 512},
#     "t02_p100_len512": {"temperature": 0.2, "top_p": 1.00, "max_tokens": 512},
#     "t01_p095_len512": {"temperature": 0.1, "top_p": 0.95, "max_tokens": 512},
#     "t03_p090_len512": {"temperature": 0.3, "top_p": 0.90, "max_tokens": 512},
#     "t00_p100_len512": {"temperature": 0.0, "top_p": 1.00, "max_tokens": 512},  # greedy
#     "t05_p090_len512": {"temperature": 0.5, "top_p": 0.90, "max_tokens": 512},
# }

DECODE_VARIANTS = [
    {"name": "t0.2_p0.95_len512", "temperature": 0.2, "top_p": 0.95, "max_tokens": 512},
    {"name": "t0.2_p1.0_len512",  "temperature": 0.2, "top_p": 1.0,  "max_tokens": 512},
    {"name": "t0.1_p0.95_len512", "temperature": 0.1, "top_p": 0.95, "max_tokens": 512},
    {"name": "t0.3_p0.9_len512",  "temperature": 0.3, "top_p": 0.9,  "max_tokens": 512},
    {"name": "t0.0_p1.0_len512",  "temperature": 0.0, "top_p": 1.00,  "max_tokens": 512},
    {"name": "t0.7_p0.9_len512",  "temperature": 0.7, "top_p": 0.9,  "max_tokens": 512},
    {"name": "t1.0_p1.0_len512",  "temperature": 1.0, "top_p": 1.0,  "max_tokens": 512},
]

def get_decode(decode_id: str) -> dict:
    if decode_id not in DECODE_VARIANTS:
        raise ValueError(f"Unknown decode_id: {decode_id}. Choices: {list(DECODE_VARIANTS)}")
    return DECODE_VARIANTS[decode_id]
