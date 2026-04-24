def extract_json_from_llm_output(content: str) -> str:
    """Strip markdown code fences from LLM JSON output."""
    if "```json" in content:
        return content.split("```json")[1].split("```")[0].strip()
    if "```" in content:
        return content.split("```")[1].split("```")[0].strip()
    return content
