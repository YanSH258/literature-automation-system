import re
from dataclasses import dataclass

CITATION_PATTERN = re.compile(r"\[(\d+)\]")
FORBIDDEN_PATTERNS = [
    re.compile(r"\[\d+,\s*\d+\]"),   # [1,2]
    re.compile(r"【\d+】"),            # 全角括号
    re.compile(r"\(\d+\)"),           # (1)
]

@dataclass
class StructuralCheckResult:
    passed: bool
    issues: list[str]
    uncited_sentences: list[str]
    cited_indices: list[int]

def structural_check(answer: str, max_index: int) -> StructuralCheckResult:
    """
    检查 LLM 输出的引用格式是否合法。
    answer: LLM 输出的答案文本
    max_index: 本轮 CitationMap 的最大 display_index
    """
    issues = []

    # 1. 检查引用编号是否越界
    indices = [int(m) for m in CITATION_PATTERN.findall(answer)]
    out_of_range = [i for i in indices if i < 1 or i > max_index]
    if out_of_range:
        issues.append(f"out_of_range_indices: {out_of_range}")

    # 2. 检查禁用格式
    for pat in FORBIDDEN_PATTERNS:
        if pat.search(answer):
            issues.append(f"forbidden_pattern: {pat.pattern}")

    # 3. 检查无引用句子（按句号/问号/感叹号分割）
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    uncited = [s for s in sentences if not CITATION_PATTERN.search(s)]

    return StructuralCheckResult(
        passed=len(issues) == 0 and len(uncited) == 0,
        issues=issues,
        uncited_sentences=uncited,
        cited_indices=list(set(indices)),
    )
