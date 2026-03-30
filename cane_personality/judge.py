"""judge.py -- Lightweight LLM-as-judge for personality profiling.

Scores model responses on accuracy, completeness, and hallucination.
Supports Anthropic (default), OpenAI, and Ollama providers.
"""

import json
import logging
import os

log = logging.getLogger("cane_personality")


def _extract_first_json(text: str) -> dict | None:
    """Extract the first valid JSON object from a string.

    Handles cases where the judge returns JSON followed by explanation text,
    JSON wrapped in markdown fences, or JSON embedded in prose. Uses brace
    depth tracking with string awareness to find balanced objects.

    Returns None if no valid JSON object is found.
    """
    # Strip markdown fences first
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    # Try direct parse first (fastest path)
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    # Walk the string looking for balanced braces
    i = text.find("{")
    while i != -1 and i < len(text):
        depth = 0
        in_string = False
        escape_next = False
        for j in range(i, len(text)):
            ch = text[j]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[i:j + 1]
                    try:
                        return json.loads(candidate)
                    except (json.JSONDecodeError, ValueError):
                        break
        i = text.find("{", i + 1)

    return None


def _clamp_scores(result: dict) -> dict:
    """Clamp all numeric score values to 0-100 range."""
    for key in ("accuracy", "completeness", "hallucination", "overall_score"):
        if key in result:
            try:
                result[key] = max(0, min(100, float(result[key])))
            except (TypeError, ValueError):
                result[key] = 50
    return result

JUDGE_PROMPT = """You are an expert evaluator. Score the following AI response against the expected answer.

Question: {question}
Expected Answer: {expected_answer}
AI Response: {agent_answer}

Score each criterion from 0 to 100:
- accuracy: How factually correct is the response compared to the expected answer?
- completeness: Does it cover all key points from the expected answer?
- hallucination: How free is the response from fabricated information? (100 = no hallucination, 0 = completely fabricated)

Respond with ONLY a JSON object, no other text:
{{"accuracy": <score>, "completeness": <score>, "hallucination": <score>, "status": "<pass|warn|fail>", "overall_score": <score>}}

Scoring guide for status:
- "pass": all scores >= 70 and overall >= 70
- "warn": any score between 40-70
- "fail": any score < 40 or overall < 40"""


def _call_anthropic(prompt: str, model: str, api_key: str) -> str:
    """Call Anthropic API."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic is required. Install: pip install cane-personality[anthropic]")

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _call_openai(prompt: str, model: str, api_key: str, base_url: str = None) -> str:
    """Call OpenAI-compatible API (also works for Ollama)."""
    try:
        import openai
    except ImportError:
        raise ImportError("openai is required. Install: pip install cane-personality[openai]")

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    client = openai.OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


class Judge:
    """LLM-as-judge for scoring model responses."""

    DEFAULT_MODELS = {
        "anthropic": "claude-haiku-4-5-20241022",
        "openai": "gpt-4o-mini",
        "ollama": "llama3",
    }

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        prompt_template: str = None,
    ):
        self.provider = provider.lower()
        self.model = model or self.DEFAULT_MODELS.get(self.provider, "claude-haiku-4-5-20241022")
        self.base_url = base_url
        self.prompt_template = prompt_template or JUDGE_PROMPT

        # Resolve API key
        if api_key:
            self.api_key = api_key
        elif self.provider == "anthropic":
            self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        elif self.provider in ("openai", "ollama"):
            self.api_key = os.environ.get("OPENAI_API_KEY", "")
        else:
            self.api_key = ""

        # Ollama defaults
        if self.provider == "ollama":
            self.base_url = base_url or "http://localhost:11434/v1"
            if not self.api_key:
                self.api_key = "ollama"  # Ollama doesn't need a real key

    def score(self, question: str, expected_answer: str, agent_answer: str) -> dict:
        """
        Score a single response.

        Returns:
            Dict with keys: accuracy, completeness, hallucination, status, overall_score
        """
        prompt = self.prompt_template.format(
            question=question,
            expected_answer=expected_answer,
            agent_answer=agent_answer,
        )

        if self.provider == "anthropic":
            raw = _call_anthropic(prompt, self.model, self.api_key)
        else:
            raw = _call_openai(prompt, self.model, self.api_key, self.base_url)

        # Parse JSON from response
        result = _extract_first_json(raw)
        if result is None:
            log.warning("Could not parse judge response, using defaults: %s", raw[:120])
            result = {
                "accuracy": 50, "completeness": 50, "hallucination": 50,
                "status": "warn", "overall_score": 50,
            }

        # Ensure all keys exist with defaults
        for key in ("accuracy", "completeness", "hallucination"):
            result.setdefault(key, 50)

        # Clamp scores to valid range
        result = _clamp_scores(result)

        if "overall_score" not in result:
            result["overall_score"] = round(
                (result["accuracy"] * 0.4 + result["completeness"] * 0.3 + result["hallucination"] * 0.3), 1
            )

        if "status" not in result:
            if result["overall_score"] >= 70:
                result["status"] = "pass"
            elif result["overall_score"] >= 40:
                result["status"] = "warn"
            else:
                result["status"] = "fail"

        return result
