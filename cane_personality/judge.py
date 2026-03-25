"""judge.py -- Lightweight LLM-as-judge for personality profiling.

Scores model responses on accuracy, completeness, and hallucination.
Supports Anthropic (default), OpenAI, and Ollama providers.
"""

import json
import os

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
    ):
        self.provider = provider.lower()
        self.model = model or self.DEFAULT_MODELS.get(self.provider, "claude-haiku-4-5-20241022")
        self.base_url = base_url

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
        prompt = JUDGE_PROMPT.format(
            question=question,
            expected_answer=expected_answer,
            agent_answer=agent_answer,
        )

        if self.provider == "anthropic":
            raw = _call_anthropic(prompt, self.model, self.api_key)
        else:
            raw = _call_openai(prompt, self.model, self.api_key, self.base_url)

        # Parse JSON from response
        try:
            # Handle potential markdown wrapping
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: try to find JSON in response
            import re
            match = re.search(r'\{[^}]+\}', raw)
            if match:
                result = json.loads(match.group())
            else:
                result = {
                    "accuracy": 50, "completeness": 50, "hallucination": 50,
                    "status": "warn", "overall_score": 50,
                }

        # Ensure all keys exist
        for key in ["accuracy", "completeness", "hallucination"]:
            result.setdefault(key, 50)

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
