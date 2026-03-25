"""export.py -- Export training data and steering vectors.

Generates DPO training pairs, SFT examples, and steering vector JSON
from personality profiling results.
"""

import json
from cane_personality.types import ProfileResult, ContrastivePair, SteeringVector


def export_dpo_pairs(profile: ProfileResult, path: str):
    """
    Export contrastive pairs as DPO training data (JSONL).

    Format compatible with TRL, OpenRLHF, and PRIME-RL:
    {"prompt": "...", "chosen": "...", "rejected": "...", "trait": "..."}
    """
    with open(path, "w", encoding="utf-8") as f:
        for pair in profile.contrastive_pairs:
            entry = {
                "prompt": pair.question,
                "chosen": pair.confident_right,
                "rejected": pair.confident_wrong,
                "chosen_score": pair.right_score,
                "rejected_score": pair.wrong_score,
                "trait": pair.trait_tag,
                "source": "cane-personality",
                "model": profile.model_name,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def export_sft_examples(profile: ProfileResult, path: str, min_score: float = 80):
    """
    Export high-scoring responses as SFT training examples (JSONL).

    Format: {"prompt": "...", "completion": "...", "score": ..., "traits": {...}}
    """
    with open(path, "w", encoding="utf-8") as f:
        for r in profile.embedded_results:
            if r.score >= min_score:
                entry = {
                    "prompt": r.question,
                    "completion": r.agent_answer,
                    "score": r.score,
                    "traits": r.traits,
                    "source": "cane-personality",
                    "model": profile.model_name,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def export_steering_vectors(profile: ProfileResult, path: str):
    """
    Export steering vectors as JSON.

    Includes the full direction vector, magnitude, and polarity labels
    for use in representation engineering workflows.
    """
    vectors = []
    for sv in profile.steering_vectors:
        vectors.append({
            "name": sv.name,
            "description": sv.description,
            "direction": sv.direction,
            "magnitude": round(sv.magnitude, 4),
            "positive_label": sv.positive_label,
            "negative_label": sv.negative_label,
            "n_positive": sv.n_positive,
            "n_negative": sv.n_negative,
            "model": profile.model_name,
            "embedding_model": profile.embedding_model,
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "model": profile.model_name,
            "suite": profile.suite_name,
            "embedding_model": profile.embedding_model,
            "vectors": vectors,
        }, f, indent=2)


def export_full_results(profile: ProfileResult, path: str):
    """Export complete profile as JSON (for baselines or comparison)."""
    profile.to_json(path)
