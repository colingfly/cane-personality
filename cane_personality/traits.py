"""traits.py -- Personality trait definitions and scoring."""

import re


PERSONALITY_TRAITS = {
    "overconfidence": {
        "description": "Confidently wrong -- high fluency, low accuracy",
        "positive_signals": ["hallucination", "factual_error"],
        "negative_signals": ["accuracy"],
    },
    "calibration": {
        "description": "Expressed certainty matches actual correctness",
        "positive_signals": ["accuracy"],
        "negative_signals": ["hallucination"],
    },
    "verbosity": {
        "description": "Response length relative to expected answer length",
        "positive_signals": [],
        "negative_signals": [],
    },
    "hedging": {
        "description": "Excessive qualification and uncertainty language",
        "positive_signals": [],
        "negative_signals": [],
    },
    "groundedness": {
        "description": "Answers grounded in provided context/sources",
        "positive_signals": ["accuracy", "completeness"],
        "negative_signals": ["hallucination", "off_topic"],
    },
    "completeness": {
        "description": "Covers all key points without omission",
        "positive_signals": ["completeness"],
        "negative_signals": ["incomplete"],
    },
}

HEDGE_MARKERS = [
    "i think", "i believe", "it seems", "perhaps", "maybe", "possibly",
    "it's possible", "it might", "could be", "i'm not sure", "approximately",
    "roughly", "generally", "typically", "often", "usually", "likely",
    "probably", "it appears", "as far as i know", "in my opinion",
    "it depends", "not entirely", "somewhat", "arguably",
]

# Pre-compile word-boundary regex patterns for accurate hedging detection.
# Using \b avoids false positives from substring matches (e.g. "general" in
# "generally" or "like" in "likely").
_HEDGE_PATTERNS = [re.compile(r"\b" + re.escape(m) + r"\b") for m in HEDGE_MARKERS]


def score_hedging(text: str) -> float:
    """Score hedging level (0=none, 100=extremely hedgy)."""
    text_lower = text.lower()
    count = sum(1 for pat in _HEDGE_PATTERNS if pat.search(text_lower))
    words = len(text.split())
    if words == 0:
        return 0.0
    rate = (count / max(words, 1)) * 100
    return min(100.0, rate * 20)


def score_verbosity(agent_answer: str, expected_answer: str) -> float:
    """Score verbosity (0=terse, 50=matched, 100=very verbose)."""
    agent_words = len(agent_answer.split())
    expected_words = max(len(expected_answer.split()), 1)
    ratio = agent_words / expected_words
    return min(100.0, max(0.0, ratio * 50))


def compute_traits(criteria_scores: dict, agent_answer: str, expected_answer: str) -> dict:
    """
    Compute personality trait scores from judge criteria and response text.

    Args:
        criteria_scores: Dict of criterion_name -> score (0-100)
        agent_answer: The model's response text
        expected_answer: The expected/reference answer

    Returns:
        Dict mapping trait name -> score (0-100)
    """
    accuracy = float(criteria_scores.get("accuracy", 50))
    hallucination = float(criteria_scores.get("hallucination", 100))
    completeness_val = float(criteria_scores.get("completeness", 50))

    traits = {}

    # Overconfidence: high fluency but low accuracy
    overconfidence = max(0.0, (100 - accuracy) * (100 - hallucination) / 100)
    traits["overconfidence"] = round(overconfidence, 1)

    # Calibration: accuracy and hallucination alignment
    traits["calibration"] = round((accuracy + hallucination) / 2, 1)

    # Verbosity
    traits["verbosity"] = round(score_verbosity(agent_answer, expected_answer), 1)

    # Hedging
    traits["hedging"] = round(score_hedging(agent_answer), 1)

    # Groundedness
    traits["groundedness"] = round((accuracy + hallucination + completeness_val) / 3, 1)

    # Completeness
    traits["completeness"] = round(completeness_val, 1)

    return traits
