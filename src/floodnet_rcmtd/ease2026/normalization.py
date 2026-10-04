"""Normalize concise model answers into a constrained label set."""

import re

def _answer_candidates(answer_space: object, answer: object | None = None) -> list[str]:
    candidates = [str(candidate) for candidate in answer_space or []]
    if answer is not None and str(answer) not in candidates:
        candidates.append(str(answer))
    return sorted(candidates, key=len, reverse=True)

def _contains_label(text: str, label: str) -> bool:
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(label.lower())}(?![A-Za-z0-9_])"
    return re.search(pattern, text.lower()) is not None

def normalize_prediction(generated_text: str, answer_space: list[str]) -> str:
    """Map model output to one label from the target answer space."""

    candidates = _answer_candidates(answer_space)
    if not candidates:
        raise ValueError("answer_space must not be empty")

    answer_lines = re.findall(r"answer\s*[:：]\s*([^\n\r]+)", generated_text, flags=re.I)
    regions = [answer_lines[-1]] if answer_lines else []
    regions.append(generated_text)
    for region in regions:
        normalized_region = region.strip().strip("`'\".。,:：;； \t\r\n").lower()
        for candidate in candidates:
            if normalized_region == candidate.lower():
                return candidate
        for candidate in candidates:
            if _contains_label(region, candidate):
                return candidate
    raise ValueError(f"could not normalize prediction from generated text: {generated_text!r}")

