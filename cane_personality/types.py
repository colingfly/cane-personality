"""types.py -- Core data types for cane-personality."""

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EmbeddedResult:
    """A single eval result with its embedding and metadata."""
    index: int
    question: str
    agent_answer: str
    expected_answer: str
    score: float
    status: str  # pass/warn/fail
    embedding: Optional[list[float]] = None
    projection_2d: Optional[list[float]] = None
    projection_3d: Optional[list[float]] = None
    criteria_scores: dict = field(default_factory=dict)
    failure_type: Optional[str] = None
    traits: dict = field(default_factory=dict)
    cluster_id: int = -1
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "index": self.index,
            "question": self.question,
            "agent_answer": self.agent_answer[:200],
            "expected_answer": self.expected_answer[:200],
            "score": self.score,
            "status": self.status,
            "criteria_scores": self.criteria_scores,
            "traits": self.traits,
            "cluster_id": self.cluster_id,
            "tags": self.tags,
        }
        if self.projection_2d:
            d["x"] = self.projection_2d[0]
            d["y"] = self.projection_2d[1]
        if self.projection_3d:
            d["x3d"] = self.projection_3d[0]
            d["y3d"] = self.projection_3d[1]
            d["z3d"] = self.projection_3d[2]
        if self.failure_type:
            d["failure_type"] = self.failure_type
        return d


@dataclass
class PersonalityProfile:
    """Aggregate personality profile for a model/agent."""
    trait_scores: dict = field(default_factory=dict)
    trait_descriptions: dict = field(default_factory=dict)
    dominant_traits: list = field(default_factory=list)
    risk_traits: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "trait_scores": self.trait_scores,
            "trait_descriptions": self.trait_descriptions,
            "dominant_traits": self.dominant_traits,
            "risk_traits": self.risk_traits,
        }


@dataclass
class SteeringVector:
    """A direction in embedding space between behavioral clusters."""
    name: str
    description: str
    direction: list[float]
    magnitude: float
    positive_label: str
    negative_label: str
    n_positive: int = 0
    n_negative: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "direction": self.direction,
            "magnitude": round(self.magnitude, 4),
            "positive_label": self.positive_label,
            "negative_label": self.negative_label,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
        }


@dataclass
class ContrastivePair:
    """A pair of responses: one confidently right, one confidently wrong."""
    question: str
    confident_right: str
    confident_wrong: str
    right_score: float
    wrong_score: float
    right_embedding: Optional[list[float]] = None
    wrong_embedding: Optional[list[float]] = None
    trait_tag: str = ""  # which trait this pair demonstrates

    def to_dict(self) -> dict:
        d = {
            "question": self.question,
            "confident_right": self.confident_right,
            "confident_wrong": self.confident_wrong,
            "right_score": self.right_score,
            "wrong_score": self.wrong_score,
        }
        if self.trait_tag:
            d["trait_tag"] = self.trait_tag
        return d


@dataclass
class ProfileResult:
    """Complete profiling result for an eval run."""
    suite_name: str
    model_name: str = ""
    total_results: int = 0
    embedded_results: list[EmbeddedResult] = field(default_factory=list)
    personality: Optional[PersonalityProfile] = None
    clusters: dict = field(default_factory=dict)
    steering_vectors: list[SteeringVector] = field(default_factory=list)
    contrastive_pairs: list[ContrastivePair] = field(default_factory=list)
    embedding_model: str = ""
    projection_method: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "suite_name": self.suite_name,
            "model_name": self.model_name,
            "total_results": self.total_results,
            "embedding_model": self.embedding_model,
            "projection_method": self.projection_method,
            "metadata": self.metadata,
            "personality": self.personality.to_dict() if self.personality else None,
            "clusters": self.clusters,
            "steering_vectors": [sv.to_dict() for sv in self.steering_vectors],
            "contrastive_pairs": [cp.to_dict() for cp in self.contrastive_pairs],
            "results": [r.to_dict() for r in self.embedded_results],
        }

    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_html(self, path: str):
        from cane_personality.report import generate_html_report
        html = generate_html_report(self)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
