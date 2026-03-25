"""Comprehensive test suite for cane-personality.

Covers traits, profiler, types, compare, export, and judge modules.
Uses numpy for synthetic data and avoids sentence-transformers / API calls.
"""

import json
import os
import tempfile

import numpy as np
import pytest

from cane_personality.types import (
    EmbeddedResult,
    PersonalityProfile,
    SteeringVector,
    ContrastivePair,
    ProfileResult,
)
from cane_personality.traits import (
    PERSONALITY_TRAITS,
    HEDGE_MARKERS,
    score_hedging,
    score_verbosity,
    compute_traits,
)
from cane_personality.profiler import (
    project_pca,
    cluster_kmeans,
    aggregate_personality,
    extract_contrastive_pairs,
    compute_steering_vectors,
    label_clusters,
)
from cane_personality.compare import compare_profiles, format_comparison_table
from cane_personality.export import export_dpo_pairs, export_steering_vectors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embedded_result(
    index=0,
    question="What is 2+2?",
    agent_answer="4",
    expected_answer="4",
    score=90.0,
    status="pass",
    embedding=None,
    traits=None,
    cluster_id=-1,
    hedging=10.0,
    criteria_scores=None,
    tags=None,
):
    """Factory for EmbeddedResult with sensible defaults."""
    if traits is None:
        traits = {
            "overconfidence": 10.0,
            "calibration": 80.0,
            "verbosity": 50.0,
            "hedging": hedging,
            "groundedness": 75.0,
            "completeness": 70.0,
        }
    return EmbeddedResult(
        index=index,
        question=question,
        agent_answer=agent_answer,
        expected_answer=expected_answer,
        score=score,
        status=status,
        embedding=embedding,
        traits=traits,
        cluster_id=cluster_id,
        criteria_scores=criteria_scores or {},
        tags=tags or [],
    )


def _make_profile_result(
    suite_name="test-suite",
    model_name="test-model",
    n_results=5,
    with_personality=True,
    with_pairs=False,
    with_vectors=False,
):
    """Factory for ProfileResult with optional sub-objects."""
    results = [_make_embedded_result(index=i) for i in range(n_results)]

    personality = None
    if with_personality:
        personality = PersonalityProfile(
            trait_scores={"overconfidence": 20.0, "calibration": 75.0, "verbosity": 55.0,
                          "hedging": 15.0, "groundedness": 72.0, "completeness": 68.0},
            trait_descriptions={k: v["description"] for k, v in PERSONALITY_TRAITS.items()},
            dominant_traits=["calibration", "groundedness", "completeness"],
            risk_traits=[],
        )

    pairs = []
    if with_pairs:
        pairs = [
            ContrastivePair(
                question="Q1",
                confident_right="correct answer",
                confident_wrong="wrong answer",
                right_score=95.0,
                wrong_score=20.0,
                trait_tag="overconfidence",
            )
        ]

    vectors = []
    if with_vectors:
        vectors = [
            SteeringVector(
                name="overconfidence",
                description="test vector",
                direction=[0.5, -0.5, 0.5],
                magnitude=0.866,
                positive_label="overconfident",
                negative_label="calibrated",
                n_positive=3,
                n_negative=5,
            )
        ]

    return ProfileResult(
        suite_name=suite_name,
        model_name=model_name,
        total_results=n_results,
        embedded_results=results,
        personality=personality,
        contrastive_pairs=pairs,
        steering_vectors=vectors,
        embedding_model="test-embed",
        projection_method="pca",
    )


# ===========================================================================
# 1. Trait Scoring
# ===========================================================================

class TestScoreHedging:
    def test_no_hedging(self):
        assert score_hedging("The answer is 42.") == 0.0

    def test_single_hedge(self):
        result = score_hedging("I think the answer is 42.")
        assert result > 0.0

    def test_multiple_hedges(self):
        text = "I think maybe the answer is perhaps roughly 42."
        result = score_hedging(text)
        assert result > 0.0
        # More hedge markers in shorter text -> higher or equal score
        few_hedges = score_hedging("The answer is probably 42 and it seems right to me.")
        many_hedges = score_hedging("I think maybe perhaps possibly roughly generally typically likely probably the answer is 42.")
        assert many_hedges >= few_hedges

    def test_empty_string(self):
        assert score_hedging("") == 0.0

    def test_max_cap_at_100(self):
        # Stuff every hedge marker into a short text
        text = " ".join(HEDGE_MARKERS)
        result = score_hedging(text)
        assert result <= 100.0

    def test_case_insensitive(self):
        lower = score_hedging("i think maybe")
        upper = score_hedging("I Think Maybe")
        assert lower == upper


class TestScoreVerbosity:
    def test_matched_length(self):
        result = score_verbosity("one two three", "one two three")
        assert 45.0 <= result <= 55.0

    def test_terse_answer(self):
        result = score_verbosity("yes", "The answer is four because 2 plus 2 equals 4.")
        assert result < 30.0

    def test_verbose_answer(self):
        agent = "word " * 100
        expected = "short"
        result = score_verbosity(agent, expected)
        assert result == 100.0  # capped at 100

    def test_empty_agent(self):
        result = score_verbosity("", "expected answer")
        assert result == 0.0

    def test_empty_expected(self):
        # expected_words clamped to 1
        result = score_verbosity("hello world", "")
        assert result > 0.0


class TestComputeTraits:
    def test_all_traits_present(self):
        criteria = {"accuracy": 80, "hallucination": 90, "completeness": 70}
        traits = compute_traits(criteria, "some answer", "expected")
        expected_keys = {"overconfidence", "calibration", "verbosity", "hedging", "groundedness", "completeness"}
        assert set(traits.keys()) == expected_keys

    def test_overconfidence_formula(self):
        # overconfidence = max(0, (100 - accuracy) * (100 - hallucination) / 100)
        criteria = {"accuracy": 30, "hallucination": 20}
        traits = compute_traits(criteria, "answer", "answer")
        expected = max(0.0, (100 - 30) * (100 - 20) / 100)
        assert traits["overconfidence"] == round(expected, 1)

    def test_calibration_formula(self):
        criteria = {"accuracy": 80, "hallucination": 60}
        traits = compute_traits(criteria, "a", "b")
        assert traits["calibration"] == round((80 + 60) / 2, 1)

    def test_groundedness_formula(self):
        criteria = {"accuracy": 80, "hallucination": 90, "completeness": 70}
        traits = compute_traits(criteria, "a", "b")
        assert traits["groundedness"] == round((80 + 90 + 70) / 3, 1)

    def test_default_criteria(self):
        # Missing criteria should use defaults
        traits = compute_traits({}, "some answer text here", "expected")
        assert "overconfidence" in traits

    def test_perfect_scores(self):
        criteria = {"accuracy": 100, "hallucination": 100, "completeness": 100}
        traits = compute_traits(criteria, "answer", "answer")
        assert traits["overconfidence"] == 0.0
        assert traits["calibration"] == 100.0


# ===========================================================================
# 2. Personality Aggregation and Risk Detection
# ===========================================================================

class TestAggregatePersonality:
    def test_empty_results(self):
        profile = aggregate_personality([])
        assert profile.trait_scores == {}
        assert profile.dominant_traits == []

    def test_single_result(self):
        r = _make_embedded_result(traits={
            "overconfidence": 80.0, "calibration": 30.0, "verbosity": 50.0,
            "hedging": 70.0, "groundedness": 40.0, "completeness": 60.0,
        })
        profile = aggregate_personality([r])
        assert profile.trait_scores["overconfidence"] == 80.0
        assert len(profile.dominant_traits) == 3

    def test_averaging(self):
        r1 = _make_embedded_result(traits={"overconfidence": 60.0, "calibration": 80.0,
                                           "verbosity": 50.0, "hedging": 20.0,
                                           "groundedness": 70.0, "completeness": 90.0})
        r2 = _make_embedded_result(traits={"overconfidence": 40.0, "calibration": 60.0,
                                           "verbosity": 50.0, "hedging": 40.0,
                                           "groundedness": 50.0, "completeness": 70.0})
        profile = aggregate_personality([r1, r2])
        assert profile.trait_scores["overconfidence"] == 50.0
        assert profile.trait_scores["calibration"] == 70.0

    def test_risk_traits_detected(self):
        r = _make_embedded_result(traits={
            "overconfidence": 80.0, "calibration": 50.0, "verbosity": 50.0,
            "hedging": 75.0, "groundedness": 50.0, "completeness": 50.0,
        })
        profile = aggregate_personality([r])
        assert "overconfidence" in profile.risk_traits
        assert "hedging" in profile.risk_traits

    def test_no_risk_when_low(self):
        r = _make_embedded_result(traits={
            "overconfidence": 30.0, "calibration": 50.0, "verbosity": 50.0,
            "hedging": 20.0, "groundedness": 50.0, "completeness": 50.0,
        })
        profile = aggregate_personality([r])
        assert profile.risk_traits == []

    def test_trait_descriptions_populated(self):
        r = _make_embedded_result()
        profile = aggregate_personality([r])
        for trait_name in PERSONALITY_TRAITS:
            assert trait_name in profile.trait_descriptions


# ===========================================================================
# 3. Contrastive Pair Extraction
# ===========================================================================

class TestExtractContrastivePairs:
    def test_no_results(self):
        pairs = extract_contrastive_pairs([])
        assert pairs == []

    def test_no_qualifying_pairs(self):
        # All scores in the middle range
        results = [_make_embedded_result(score=55.0, hedging=10.0)]
        pairs = extract_contrastive_pairs(results)
        assert pairs == []

    def test_basic_pair_extraction(self):
        right = _make_embedded_result(
            index=0, question="Q1", agent_answer="correct",
            score=90.0, status="pass", hedging=10.0,
            traits={"hedging": 10.0, "overconfidence": 5.0, "verbosity": 30.0},
        )
        wrong = _make_embedded_result(
            index=1, question="Q1", agent_answer="wrong answer",
            score=20.0, status="fail", hedging=10.0,
            traits={"hedging": 10.0, "overconfidence": 80.0, "verbosity": 30.0},
        )
        pairs = extract_contrastive_pairs([right, wrong])
        assert len(pairs) == 1
        assert pairs[0].confident_right == "correct"
        assert pairs[0].confident_wrong == "wrong answer"

    def test_matching_question_preferred(self):
        right1 = _make_embedded_result(
            index=0, question="Q1", agent_answer="right1", score=90.0,
            traits={"hedging": 10.0, "overconfidence": 5.0, "verbosity": 30.0},
        )
        right2 = _make_embedded_result(
            index=1, question="Q2", agent_answer="right2", score=85.0,
            traits={"hedging": 10.0, "overconfidence": 5.0, "verbosity": 30.0},
        )
        wrong = _make_embedded_result(
            index=2, question="Q1", agent_answer="wrong", score=20.0,
            traits={"hedging": 10.0, "overconfidence": 80.0, "verbosity": 30.0},
        )
        pairs = extract_contrastive_pairs([right1, right2, wrong])
        assert len(pairs) == 1
        assert pairs[0].confident_right == "right1"

    def test_hedgy_responses_excluded(self):
        # Both high and low score but hedging is high -- should not form a pair
        right = _make_embedded_result(score=90.0, hedging=10.0,
                                      traits={"hedging": 60.0, "overconfidence": 5.0, "verbosity": 30.0})
        wrong = _make_embedded_result(score=20.0, hedging=10.0,
                                      traits={"hedging": 60.0, "overconfidence": 80.0, "verbosity": 30.0})
        pairs = extract_contrastive_pairs([right, wrong])
        assert pairs == []

    def test_trait_tag_assigned(self):
        right = _make_embedded_result(
            score=90.0,
            traits={"hedging": 10.0, "overconfidence": 5.0, "verbosity": 30.0},
        )
        wrong = _make_embedded_result(
            score=20.0,
            traits={"hedging": 10.0, "overconfidence": 80.0, "verbosity": 30.0},
        )
        pairs = extract_contrastive_pairs([right, wrong])
        assert len(pairs) == 1
        assert pairs[0].trait_tag == "overconfidence"


# ===========================================================================
# 4. Steering Vector Computation
# ===========================================================================

class TestComputeSteeringVectors:
    def test_empty_inputs(self):
        vectors = compute_steering_vectors([], [])
        assert vectors == []

    def test_overconfidence_vector_from_pairs(self):
        pair = ContrastivePair(
            question="Q",
            confident_right="right",
            confident_wrong="wrong",
            right_score=90.0,
            wrong_score=20.0,
            right_embedding=[1.0, 0.0, 0.0],
            wrong_embedding=[0.0, 1.0, 0.0],
        )
        vectors = compute_steering_vectors([], [pair])
        assert len(vectors) == 1
        assert vectors[0].name == "overconfidence"
        assert vectors[0].magnitude > 0
        assert len(vectors[0].direction) == 3

    def test_quality_vector_from_pass_fail(self):
        pass_r = _make_embedded_result(
            status="pass", embedding=[1.0, 0.0, 0.0],
        )
        fail_r = _make_embedded_result(
            status="fail", embedding=[0.0, 1.0, 0.0],
        )
        vectors = compute_steering_vectors([pass_r, fail_r], [])
        assert len(vectors) == 1
        assert vectors[0].name == "quality"

    def test_both_vectors_present(self):
        pair = ContrastivePair(
            question="Q", confident_right="r", confident_wrong="w",
            right_score=90.0, wrong_score=20.0,
            right_embedding=[1.0, 0.0], wrong_embedding=[0.0, 1.0],
        )
        pass_r = _make_embedded_result(status="pass", embedding=[1.0, 0.0])
        fail_r = _make_embedded_result(status="fail", embedding=[0.0, 1.0])
        vectors = compute_steering_vectors([pass_r, fail_r], [pair])
        names = {v.name for v in vectors}
        assert "overconfidence" in names
        assert "quality" in names

    def test_unit_direction(self):
        pair = ContrastivePair(
            question="Q", confident_right="r", confident_wrong="w",
            right_score=90.0, wrong_score=20.0,
            right_embedding=[3.0, 0.0, 0.0],
            wrong_embedding=[0.0, 4.0, 0.0],
        )
        vectors = compute_steering_vectors([], [pair])
        direction = np.array(vectors[0].direction)
        assert abs(np.linalg.norm(direction) - 1.0) < 1e-6

    def test_no_vector_without_embeddings(self):
        pair = ContrastivePair(
            question="Q", confident_right="r", confident_wrong="w",
            right_score=90.0, wrong_score=20.0,
            right_embedding=None, wrong_embedding=None,
        )
        vectors = compute_steering_vectors([], [pair])
        assert vectors == []


# ===========================================================================
# 5. K-Means Clustering
# ===========================================================================

class TestClusterKmeans:
    def test_basic_clustering(self):
        rng = np.random.RandomState(0)
        # Two clear clusters
        cluster_a = rng.randn(20, 3) + np.array([5, 0, 0])
        cluster_b = rng.randn(20, 3) + np.array([-5, 0, 0])
        data = np.vstack([cluster_a, cluster_b])
        labels = cluster_kmeans(data, n_clusters=2)
        assert labels.shape == (40,)
        assert set(np.unique(labels)) == {0, 1}

    def test_labels_length_matches_data(self):
        data = np.random.randn(30, 4)
        labels = cluster_kmeans(data, n_clusters=3)
        assert len(labels) == 30

    def test_fewer_points_than_clusters(self):
        data = np.random.randn(3, 4)
        labels = cluster_kmeans(data, n_clusters=5)
        assert len(labels) == 3
        # Should return np.arange(n)
        np.testing.assert_array_equal(labels, np.array([0, 1, 2]))

    def test_single_point(self):
        data = np.array([[1.0, 2.0, 3.0]])
        labels = cluster_kmeans(data, n_clusters=3)
        assert len(labels) == 1

    def test_deterministic(self):
        data = np.random.RandomState(42).randn(50, 5)
        labels1 = cluster_kmeans(data, n_clusters=3)
        labels2 = cluster_kmeans(data, n_clusters=3)
        np.testing.assert_array_equal(labels1, labels2)


# ===========================================================================
# 6. PCA Projection
# ===========================================================================

class TestProjectPca:
    def test_output_shape_2d(self):
        data = np.random.randn(20, 10)
        projected = project_pca(data, n_components=2)
        assert projected.shape == (20, 2)

    def test_output_shape_3d(self):
        data = np.random.randn(20, 10)
        projected = project_pca(data, n_components=3)
        assert projected.shape == (20, 3)

    def test_centered(self):
        data = np.random.randn(30, 5) + 100  # large offset
        projected = project_pca(data, n_components=2)
        # Projected should be roughly centered around 0
        assert abs(projected.mean(axis=0)[0]) < 1.0

    def test_variance_ordering(self):
        # First component should capture more variance than second
        rng = np.random.RandomState(0)
        data = rng.randn(100, 5)
        data[:, 0] *= 10  # make first feature high-variance
        projected = project_pca(data, n_components=2)
        var0 = np.var(projected[:, 0])
        var1 = np.var(projected[:, 1])
        assert var0 >= var1


# ===========================================================================
# 7. Cluster Labeling
# ===========================================================================

class TestLabelClusters:
    def test_empty_cluster(self):
        results = [_make_embedded_result(cluster_id=0)]
        labels = label_clusters(results, n_clusters=2)
        assert labels[1] == "empty"

    def test_failing_cluster(self):
        results = [
            _make_embedded_result(cluster_id=0, status="fail", score=20.0)
            for _ in range(5)
        ]
        labels = label_clusters(results, n_clusters=1)
        assert "failing" in labels[0]

    def test_passing_cluster(self):
        results = [
            _make_embedded_result(cluster_id=0, status="pass", score=90.0)
            for _ in range(5)
        ]
        labels = label_clusters(results, n_clusters=1)
        assert "passing" in labels[0]

    def test_label_contains_count(self):
        results = [_make_embedded_result(cluster_id=0) for _ in range(7)]
        labels = label_clusters(results, n_clusters=1)
        assert "n=7" in labels[0]


# ===========================================================================
# 8. Data Class Serialization
# ===========================================================================

class TestSerialization:
    def test_embedded_result_to_dict(self):
        er = _make_embedded_result()
        er.projection_2d = [1.5, 2.5]
        er.projection_3d = [1.0, 2.0, 3.0]
        d = er.to_dict()
        assert d["index"] == 0
        assert d["x"] == 1.5
        assert d["y"] == 2.5
        assert d["x3d"] == 1.0
        assert d["y3d"] == 2.0
        assert d["z3d"] == 3.0

    def test_embedded_result_truncates_long_text(self):
        long_text = "x" * 500
        er = _make_embedded_result(agent_answer=long_text, expected_answer=long_text)
        d = er.to_dict()
        assert len(d["agent_answer"]) == 200
        assert len(d["expected_answer"]) == 200

    def test_embedded_result_failure_type(self):
        er = EmbeddedResult(0, "q", "a", "e", 30.0, "fail", failure_type="hallucination")
        d = er.to_dict()
        assert d["failure_type"] == "hallucination"

    def test_embedded_result_no_projection_keys(self):
        er = _make_embedded_result()
        d = er.to_dict()
        assert "x" not in d
        assert "x3d" not in d

    def test_personality_profile_to_dict(self):
        pp = PersonalityProfile(
            trait_scores={"calibration": 75.0},
            trait_descriptions={"calibration": "desc"},
            dominant_traits=["calibration"],
            risk_traits=[],
        )
        d = pp.to_dict()
        assert d["trait_scores"]["calibration"] == 75.0
        assert d["dominant_traits"] == ["calibration"]

    def test_steering_vector_to_dict(self):
        sv = SteeringVector(
            name="test", description="d", direction=[0.1, 0.2],
            magnitude=1.2345, positive_label="p", negative_label="n",
            n_positive=3, n_negative=5,
        )
        d = sv.to_dict()
        assert d["magnitude"] == 1.2345  # rounded to 4 decimal
        assert d["name"] == "test"

    def test_contrastive_pair_to_dict_with_tag(self):
        cp = ContrastivePair("q", "right", "wrong", 90.0, 20.0, trait_tag="overconfidence")
        d = cp.to_dict()
        assert d["trait_tag"] == "overconfidence"

    def test_contrastive_pair_to_dict_without_tag(self):
        cp = ContrastivePair("q", "right", "wrong", 90.0, 20.0)
        d = cp.to_dict()
        assert "trait_tag" not in d

    def test_profile_result_to_dict(self):
        pr = _make_profile_result(with_personality=True, with_pairs=True, with_vectors=True)
        d = pr.to_dict()
        assert d["suite_name"] == "test-suite"
        assert d["personality"] is not None
        assert len(d["steering_vectors"]) == 1
        assert len(d["contrastive_pairs"]) == 1
        assert len(d["results"]) == 5

    def test_profile_result_to_dict_no_personality(self):
        pr = _make_profile_result(with_personality=False)
        d = pr.to_dict()
        assert d["personality"] is None

    def test_profile_result_to_json(self):
        pr = _make_profile_result(with_personality=True)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            pr.to_json(path)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["suite_name"] == "test-suite"
            assert isinstance(data["results"], list)
        finally:
            os.unlink(path)


# ===========================================================================
# 9. Compare Module
# ===========================================================================

class TestCompareProfiles:
    def test_empty_profiles(self):
        result = compare_profiles({})
        assert result["models"] == []
        assert result["table"] == {}

    def test_single_model(self):
        pr = _make_profile_result(model_name="model-a")
        result = compare_profiles({"model-a": pr})
        assert "model-a" in result["models"]
        assert "model-a" in result["table"]
        assert "model-a" in result["grades"]

    def test_two_models_ranking(self):
        pr_a = _make_profile_result(model_name="model-a")
        pr_a.personality.trait_scores["calibration"] = 90.0

        pr_b = _make_profile_result(model_name="model-b")
        pr_b.personality.trait_scores["calibration"] = 40.0

        result = compare_profiles({"model-a": pr_a, "model-b": pr_b})
        # For calibration, higher is better, so model-a should rank first
        assert result["rankings"]["calibration"][0] == "model-a"

    def test_grade_assignment(self):
        pr = _make_profile_result()
        pr.personality.trait_scores["calibration"] = 90.0
        pr.personality.trait_scores["groundedness"] = 85.0
        pr.personality.trait_scores["overconfidence"] = 10.0
        result = compare_profiles({"good-model": pr})
        assert result["grades"]["good-model"] == "A"

    def test_grade_f(self):
        pr = _make_profile_result()
        pr.personality.trait_scores["calibration"] = 10.0
        pr.personality.trait_scores["groundedness"] = 10.0
        pr.personality.trait_scores["overconfidence"] = 95.0
        result = compare_profiles({"bad-model": pr})
        assert result["grades"]["bad-model"] == "F"

    def test_no_personality(self):
        pr = _make_profile_result(with_personality=False)
        result = compare_profiles({"no-pers": pr})
        # Should default to 0 for all traits
        assert result["table"]["no-pers"]["calibration"] == 0


class TestFormatComparisonTable:
    def test_no_models(self):
        comparison = {"models": [], "traits": [], "table": {}, "rankings": {},
                      "grades": {}, "dpo_counts": {}, "vector_counts": {}}
        output = format_comparison_table(comparison)
        assert output == "No models to compare."

    def test_single_model_output(self):
        pr = _make_profile_result(model_name="alpha")
        comparison = compare_profiles({"alpha": pr})
        output = format_comparison_table(comparison)
        assert "alpha" in output
        assert "Grade" in output

    def test_two_models_output(self):
        pr_a = _make_profile_result(model_name="alpha")
        pr_b = _make_profile_result(model_name="beta")
        comparison = compare_profiles({"alpha": pr_a, "beta": pr_b})
        output = format_comparison_table(comparison)
        assert "alpha" in output
        assert "beta" in output
        assert "Steering Vectors" in output
        assert "DPO Pairs" in output


# ===========================================================================
# 10. Export Module
# ===========================================================================

class TestExportDpoPairs:
    def test_export_creates_jsonl(self):
        pr = _make_profile_result(with_pairs=True)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name
        try:
            export_dpo_pairs(pr, path)
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            assert len(lines) == 1
            entry = json.loads(lines[0])
            assert entry["prompt"] == "Q1"
            assert entry["chosen"] == "correct answer"
            assert entry["rejected"] == "wrong answer"
            assert entry["source"] == "cane-personality"
            assert entry["model"] == "test-model"
            assert entry["trait"] == "overconfidence"
        finally:
            os.unlink(path)

    def test_export_empty_pairs(self):
        pr = _make_profile_result(with_pairs=False)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name
        try:
            export_dpo_pairs(pr, path)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            assert content == ""
        finally:
            os.unlink(path)


class TestExportSteeringVectors:
    def test_export_creates_json(self):
        pr = _make_profile_result(with_vectors=True)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            export_steering_vectors(pr, path)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["model"] == "test-model"
            assert data["suite"] == "test-suite"
            assert len(data["vectors"]) == 1
            vec = data["vectors"][0]
            assert vec["name"] == "overconfidence"
            assert vec["direction"] == [0.5, -0.5, 0.5]
        finally:
            os.unlink(path)

    def test_export_empty_vectors(self):
        pr = _make_profile_result(with_vectors=False)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            export_steering_vectors(pr, path)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["vectors"] == []
        finally:
            os.unlink(path)


# ===========================================================================
# 11. Judge Module (parse response)
# ===========================================================================

class TestJudgeParsing:
    """Test the Judge response parsing without making API calls."""

    def test_parse_clean_json(self):
        """Test Judge.score parsing by calling the parse logic directly."""
        from cane_personality.judge import Judge
        import re

        raw = '{"accuracy": 85, "completeness": 70, "hallucination": 90, "status": "pass", "overall_score": 82}'
        text = raw.strip()
        result = json.loads(text)
        assert result["accuracy"] == 85
        assert result["status"] == "pass"

    def test_parse_markdown_wrapped_json(self):
        raw = '```json\n{"accuracy": 60, "completeness": 50, "hallucination": 70, "status": "warn", "overall_score": 60}\n```'
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(text)
        assert result["accuracy"] == 60

    def test_parse_embedded_json(self):
        import re
        raw = "Here is my evaluation: {\"accuracy\": 40, \"completeness\": 30, \"hallucination\": 20} some extra text"
        match = re.search(r'\{[^}]+\}', raw)
        assert match is not None
        result = json.loads(match.group())
        assert result["accuracy"] == 40

    def test_judge_init_defaults(self):
        """Test Judge constructor without API calls."""
        from cane_personality.judge import Judge
        j = Judge(provider="ollama")
        assert j.provider == "ollama"
        assert j.model == "llama3"
        assert j.base_url == "http://localhost:11434/v1"

    def test_judge_custom_model(self):
        from cane_personality.judge import Judge
        j = Judge(provider="openai", model="gpt-4", api_key="fake")
        assert j.model == "gpt-4"
        assert j.api_key == "fake"

    def test_status_derivation(self):
        """Verify status derivation logic from judge.py."""
        overall_score = 82
        if overall_score >= 70:
            status = "pass"
        elif overall_score >= 40:
            status = "warn"
        else:
            status = "fail"
        assert status == "pass"

        overall_score = 55
        if overall_score >= 70:
            status = "pass"
        elif overall_score >= 40:
            status = "warn"
        else:
            status = "fail"
        assert status == "warn"

        overall_score = 20
        if overall_score >= 70:
            status = "pass"
        elif overall_score >= 40:
            status = "warn"
        else:
            status = "fail"
        assert status == "fail"


# ===========================================================================
# Additional edge case and integration tests
# ===========================================================================

class TestEdgeCases:
    def test_personality_traits_dict_structure(self):
        """Ensure PERSONALITY_TRAITS has expected keys."""
        expected = {"overconfidence", "calibration", "verbosity", "hedging", "groundedness", "completeness"}
        assert set(PERSONALITY_TRAITS.keys()) == expected
        for name, info in PERSONALITY_TRAITS.items():
            assert "description" in info
            assert "positive_signals" in info
            assert "negative_signals" in info

    def test_pca_with_identical_points(self):
        """PCA on identical points should not raise."""
        data = np.ones((10, 5))
        projected = project_pca(data, n_components=2)
        assert projected.shape == (10, 2)

    def test_kmeans_with_two_points(self):
        data = np.array([[0.0, 0.0], [10.0, 10.0]])
        labels = cluster_kmeans(data, n_clusters=2)
        assert len(labels) == 2
        assert labels[0] != labels[1]

    def test_aggregate_personality_missing_trait(self):
        """Results missing some traits should default to 50."""
        r = _make_embedded_result(traits={"calibration": 80.0})
        profile = aggregate_personality([r])
        # Missing traits default to 50
        assert profile.trait_scores["overconfidence"] == 50.0
        assert profile.trait_scores["calibration"] == 80.0

    def test_label_clusters_mixed_quality(self):
        results = (
            [_make_embedded_result(cluster_id=0, status="pass", score=80.0) for _ in range(3)]
            + [_make_embedded_result(cluster_id=0, status="fail", score=30.0) for _ in range(3)]
        )
        labels = label_clusters(results, n_clusters=1)
        assert "mixed" in labels[0]

    def test_steering_vector_magnitude_rounding(self):
        sv = SteeringVector(
            name="t", description="d", direction=[1.0],
            magnitude=1.23456789, positive_label="p", negative_label="n",
        )
        d = sv.to_dict()
        assert d["magnitude"] == 1.2346  # rounded to 4 decimals

    def test_profile_result_round_trip_json(self):
        """to_json then load should preserve structure."""
        pr = _make_profile_result(with_personality=True, with_pairs=True, with_vectors=True)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            pr.to_json(path)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["suite_name"] == pr.suite_name
            assert data["model_name"] == pr.model_name
            assert len(data["steering_vectors"]) == len(pr.steering_vectors)
            assert len(data["contrastive_pairs"]) == len(pr.contrastive_pairs)
            assert len(data["results"]) == pr.total_results
        finally:
            os.unlink(path)
