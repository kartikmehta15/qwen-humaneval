#!/usr/bin/env python3
import re, ast

class PostProcessor:
    """
    Handles model output cleanup (post-processing).
    Supports multiple versions (v1, v2, v3) for ablation experiments.
    """
    _version = "v2"  # default version
    DEFAULT_VERSIONS = ["v1", "v2", "v3"]

    @classmethod
    def set_version(cls, version: str):
        if version not in cls.DEFAULT_VERSIONS:
            raise ValueError(f"Unsupported PostProcessor version: {version}")
        cls._version = version

    @classmethod
    def normalize_with_version(cls, text: str) -> str:
        """Dispatch normalize_body depending on version setting."""
        if cls._version == "v1":
            return cls._normalize_v1(text)
        elif cls._version == "v2":
            return cls._normalize_v2(text)
        elif cls._version == "v3":
            return cls._normalize_v3(text)
        else:
            return cls._normalize_v2(text)

    # ------------------------------------------------------------
    # Core normalization helpers
    # ------------------------------------------------------------
    @staticmethod
    def strip_fences(text: str) -> str:
        """Remove markdown-style code fences if present."""
        if text.strip().startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", text.strip())
            text = text.strip("`")
        return text.strip()

    @staticmethod
    def between_tags(text: str, start_tag="<sol>", end_tag="</sol>") -> str:
        """Extract text between <sol> and </sol> tags if present."""
        m = re.search(rf"{re.escape(start_tag)}(.*?){re.escape(end_tag)}", text, flags=re.DOTALL)
        return m.group(1).strip("\n") if m else ""

    @staticmethod
    def normalize_body(text: str) -> str:
        """
        Default body normalization:
          - prefer <sol> ... </sol>
          - else last code block
          - else strip fences
        """
        candidate = PostProcessor.between_tags(text)
        if not candidate:
            blocks = re.findall(r"```[a-zA-Z0-9_+-]*\s*\n(.*?)```", text, flags=re.DOTALL)
            candidate = blocks[-1] if blocks else PostProcessor.strip_fences(text)
        # ensure at least one return statement to avoid empty bodies
        return candidate if candidate.strip() else "    return None\n"

    # ------------------------------------------------------------
    # Version-specific normalization
    # ------------------------------------------------------------
    @staticmethod
    def _normalize_v1(text: str) -> str:
        # Minimal: just strip code fences
        body = PostProcessor.strip_fences(text)
        return body if body.strip() else "    return None\n"

    @staticmethod
    def _normalize_v2(text: str) -> str:
        # Standard: current default normalize_body
        return PostProcessor.normalize_body(text)

    @staticmethod
    def _normalize_v3(text: str) -> str:
        # Aggressive: prefer <sol> tags, fallback to last code block
        candidate = PostProcessor.between_tags(text)
        if not candidate:
            blocks = re.findall(r"```[a-zA-Z0-9_+-]*\s*\n(.*?)```", text, flags=re.DOTALL)
            candidate = blocks[-1] if blocks else text
        return PostProcessor._normalize_v2(candidate)



def extract_def_from_prompt(prompt: str, entry_point: str = None) -> str:
    """Extract the `def ...` stub (with optional docstring) from a HumanEval prompt.
    If entry_point is given, prefer that function name; otherwise take the first one.
    """
    # Strip code fences if present
    src = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", prompt, flags=re.MULTILINE).strip("\n")
    try:
        tree = ast.parse(src)
        lines = src.splitlines()
    except SyntaxError:
        m = re.search(r'(?m)^\s*def\s+\w+\s*\(', src)
        if not m:
            raise ValueError("No function signature found in prompt.")
        src = src[m.start():]
        tree = ast.parse(src)
        lines = src.splitlines()

    target = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            entry_point is None or node.name == entry_point
        ):
            target = node
            if entry_point is None:
                break
    if target is None:
        raise ValueError("No FunctionDef found in prompt.")

    start = target.lineno - 1
    end = start + 1
    # Include docstring if present
    if target.body and isinstance(target.body[0], ast.Expr) and \
       isinstance(getattr(target.body[0], "value", None), ast.Constant) and \
       isinstance(target.body[0].value.value, str):
        end = getattr(target.body[0], "end_lineno", end)

    snippet = "\n".join(lines[start:end])
    if not snippet.endswith("\n"):
        snippet += "\n"
    # Remove `pass` or `...`
    snippet = re.sub(r"(?m)^[ \t]+pass\s*\n", "", snippet)
    snippet = re.sub(r"(?m)^[ \t]+\.\.\.\s*\n", "", snippet)
    return snippet

# Convenience: quick access from other scripts
def normalize_output(text: str, version: str = "v2") -> str:
    PostProcessor.set_version(version)
    return PostProcessor.normalize_with_version(text)
