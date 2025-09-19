# Prompt headers + in-context examples (exact <sol>…</sol> formatting)

RAW = """# Python 3
# Return ONLY the function BODY. No 'def', imports, tests, prints.
# Indent with EXACTLY 4 spaces. Put final body between <sol> and </sol>.
"""

HARDENED_V1 = """# Python 3
# Return ONLY the function BODY (no 'def', imports, prints, tests, classes).
# Use EXACT 4 spaces per indent. Follow the docstring literally and handle edge cases.
# Put ONLY the final, executable body between <sol> and </sol>.
"""

HARDENED_V2 = """# Python 3
# Write ONLY the function BODY (no 'def', no imports, no tests, no prints).
# EXACT 4 spaces indentation. Robust to empty inputs, duplicates, ties, negatives.
# Put ONLY the final, executable body between <sol> and </sol>. Keep it concise.
"""

_FEWSHOTS = """
# Examples (format is exactly how your code must look):

def _example_sum(a, b):
    \"\"\"Return the sum of two numbers.\"\"\"
<sol>
    return a + b
</sol>

def _example_is_palindrome(s):
    \"\"\"Check if string s is a palindrome, ignoring case and non-alphanumerics.\"\"\"
<sol>
    cleaned = ''.join(ch.lower() for ch in s if ch.isalnum())
    n = len(cleaned)
    for i in range(n // 2):
        if cleaned[i] != cleaned[-(i + 1)]:
            return False
    return True
</sol>

def _example_group_by_parity(nums):
    \"\"\"Return [evens_sorted, odds_sorted] from the input list.\"\"\"
<sol>
    evens = [x for x in nums if x % 2 == 0]
    odds = [x for x in nums if x % 2 != 0]
    evens.sort()
    odds.sort()
    return [evens, odds]
</sol>
""".strip()

ICL_V2 = HARDENED_V2.rstrip() + "\n\n" + _FEWSHOTS + "\n"

HEADERS = {
    "raw": RAW,
    "hardened_v1": HARDENED_V1,
    "hardened_v2": HARDENED_V2,
    # "icl_v2": ICL_V2,
}

def get_header(prompt_id_or_text: str) -> str:
    """
    Return a header by id, or if the input isn't a known id,
    treat it as a raw header string and return it verbatim.
    """
    return HEADERS.get(prompt_id_or_text, prompt_id_or_text)

# def get_header(prompt_id: str) -> str:
#     if prompt_id not in HEADERS:
#         raise ValueError(f"Unknown prompt_id: {prompt_id}. Choices: {list(HEADERS)}")
#     return HEADERS[prompt_id]
