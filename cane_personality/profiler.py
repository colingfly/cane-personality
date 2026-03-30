"""profiler.py -- Core personality profiling engine.

Embeds model outputs, projects to low-dimensional space, clusters behavioral
patterns, and extracts contrastive steering vectors.

Pipeline:
  Results -> Trait scoring -> Embed outputs -> UMAP/PCA projection
  -> Behavioral clustering -> Contrastive pair extraction -> Steering vectors
"""

import hashlib
from pathlib import Path

import numpy as np

from cane_personality.types import (
    EmbeddedResult,
    PersonalityProfile,
    SteeringVector,
    ContrastivePair,
    ProfileResult,
)
from cane_personality.traits import (
    PERSONALITY_TRAITS,
    compute_traits,
)


def _embedding_cache_dir() -> Path:
    """Return (and create if needed) the embedding cache directory."""
    cache_dir = Path.home() / ".cache" / "cane-personality"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _embedding_cache_key(texts: list[str], model_name: str) -> str:
    """Build a deterministic hash from the text list and model name."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")  # separator so ["ab","c"] != ["a","bc"]
    return f"embed_{h.hexdigest()}_{model_name}"


def embed_texts(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    use_cache: bool = True,
) -> np.ndarray:
    """Embed a list of texts, returning (n, dim) array.

    When use_cache is True (the default), embeddings are persisted to
    ~/.cache/cane-personality/ keyed by a SHA-256 hash of the input texts
    and model name. Subsequent calls with identical inputs skip the model
    entirely and load from disk.
    """
    # Try loading from cache
    if use_cache:
        cache_dir = _embedding_cache_dir()
        cache_file = cache_dir / (_embedding_cache_key(texts, model_name) + ".npy")
        try:
            if cache_file.exists():
                cached = np.load(str(cache_file))
                if cached.shape[0] == len(texts):
                    return cached
        except Exception:
            pass  # cache miss or corrupt file, fall through to compute

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for profiling. "
            "Install it: pip install cane-personality[embeddings]"
        )
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    result = np.array(embeddings)

    # Persist to cache
    if use_cache:
        try:
            np.save(str(cache_file), result)
        except Exception:
            pass  # non-fatal: caching is best-effort

    return result


def project_umap(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Project embeddings via UMAP. Falls back to PCA if unavailable."""
    try:
        from umap import UMAP
        reducer = UMAP(
            n_components=n_components,
            n_neighbors=min(15, len(embeddings) - 1),
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        return reducer.fit_transform(embeddings)
    except ImportError:
        return project_pca(embeddings, n_components)


def project_pca(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Project embeddings via PCA (numpy only, no sklearn)."""
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1][:n_components]
    components = eigenvectors[:, idx]
    projected = centered @ components
    return projected


def cluster_kmeans(embeddings: np.ndarray, n_clusters: int = 4, max_iter: int = 100) -> np.ndarray:
    """K-means++ clustering (no sklearn dependency)."""
    n = len(embeddings)
    if n <= n_clusters:
        return np.arange(n)

    rng = np.random.RandomState(42)
    centroids = [embeddings[rng.randint(n)]]

    for _ in range(1, n_clusters):
        dists = np.array([
            min(np.sum((e - c) ** 2) for c in centroids)
            for e in embeddings
        ])
        probs = dists / dists.sum()
        centroids.append(embeddings[rng.choice(n, p=probs)])

    centroids = np.array(centroids)
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        for i, e in enumerate(embeddings):
            dists = np.sum((centroids - e) ** 2, axis=1)
            labels[i] = np.argmin(dists)

        new_centroids = np.zeros_like(centroids)
        for k in range(n_clusters):
            members = embeddings[labels == k]
            if len(members) > 0:
                new_centroids[k] = members.mean(axis=0)
            else:
                new_centroids[k] = centroids[k]

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels


def aggregate_personality(embedded_results: list[EmbeddedResult]) -> PersonalityProfile:
    """Aggregate individual trait scores into a personality profile."""
    if not embedded_results:
        return PersonalityProfile()

    all_traits = {}
    for trait_name in PERSONALITY_TRAITS:
        values = [r.traits.get(trait_name, 50) for r in embedded_results]
        all_traits[trait_name] = round(sum(values) / len(values), 1)

    sorted_traits = sorted(
        all_traits.items(),
        key=lambda x: abs(x[1] - 50),
        reverse=True,
    )

    dominant = [t[0] for t in sorted_traits[:3]]
    risk = [
        t[0] for t in sorted_traits
        if t[0] in ("overconfidence", "hedging") and t[1] > 60
    ]

    return PersonalityProfile(
        trait_scores=all_traits,
        trait_descriptions={k: v["description"] for k, v in PERSONALITY_TRAITS.items()},
        dominant_traits=dominant,
        risk_traits=risk,
    )


def extract_contrastive_pairs(
    embedded_results: list[EmbeddedResult],
    high_threshold: float = 80,
    low_threshold: float = 40,
) -> list[ContrastivePair]:
    """Extract contrastive pairs: confidently right vs. confidently wrong."""
    confident_right = [
        r for r in embedded_results
        if r.score >= high_threshold and r.traits.get("hedging", 50) < 40
    ]
    confident_wrong = [
        r for r in embedded_results
        if r.score <= low_threshold and r.traits.get("hedging", 50) < 40
    ]

    all_trait_keys = [
        "overconfidence", "calibration", "hedging",
        "verbosity", "groundedness", "completeness",
    ]

    pairs = []
    for wrong in confident_wrong:
        best = None
        for right in confident_right:
            if right.question == wrong.question:
                best = right
                break
        if best is None:
            continue  # No valid match, skip rather than pairing unrelated questions

        # Determine trait tag from the wrong result's worst trait
        worst_trait = ""
        if wrong.traits:
            worst_trait = max(
                all_trait_keys,
                key=lambda t: wrong.traits.get(t, 0),
                default="",
            )

        pairs.append(ContrastivePair(
            question=wrong.question,
            confident_right=best.agent_answer,
            confident_wrong=wrong.agent_answer,
            right_score=best.score,
            wrong_score=wrong.score,
            right_embedding=best.embedding,
            wrong_embedding=wrong.embedding,
            trait_tag=worst_trait,
        ))

    return pairs


def compute_steering_vectors(
    embedded_results: list[EmbeddedResult],
    contrastive_pairs: list[ContrastivePair],
) -> list[SteeringVector]:
    """Compute steering vectors from contrastive pairs and cluster analysis."""
    vectors = []

    # 1. Overconfidence vector from contrastive pairs
    if contrastive_pairs:
        right_embeds = [
            np.array(p.right_embedding) for p in contrastive_pairs
            if p.right_embedding is not None
        ]
        wrong_embeds = [
            np.array(p.wrong_embedding) for p in contrastive_pairs
            if p.wrong_embedding is not None
        ]

        if right_embeds and wrong_embeds:
            right_mean = np.mean(right_embeds, axis=0)
            wrong_mean = np.mean(wrong_embeds, axis=0)
            direction = wrong_mean - right_mean
            magnitude = float(np.linalg.norm(direction))

            if magnitude > 0:
                unit_direction = (direction / magnitude).tolist()
                vectors.append(SteeringVector(
                    name="overconfidence",
                    description="Direction from calibrated confidence to overconfidence in embedding space",
                    direction=unit_direction,
                    magnitude=magnitude,
                    positive_label="overconfident",
                    negative_label="calibrated",
                    n_positive=len(wrong_embeds),
                    n_negative=len(right_embeds),
                ))

    # 2. Quality vector from pass/fail clusters
    pass_embeds = [
        np.array(r.embedding) for r in embedded_results
        if r.status == "pass" and r.embedding is not None
    ]
    fail_embeds = [
        np.array(r.embedding) for r in embedded_results
        if r.status == "fail" and r.embedding is not None
    ]

    if pass_embeds and fail_embeds:
        pass_mean = np.mean(pass_embeds, axis=0)
        fail_mean = np.mean(fail_embeds, axis=0)
        direction = fail_mean - pass_mean
        magnitude = float(np.linalg.norm(direction))

        if magnitude > 0:
            unit_direction = (direction / magnitude).tolist()
            vectors.append(SteeringVector(
                name="quality",
                description="Direction from high-quality to low-quality responses",
                direction=unit_direction,
                magnitude=magnitude,
                positive_label="low_quality",
                negative_label="high_quality",
                n_positive=len(fail_embeds),
                n_negative=len(pass_embeds),
            ))

    return vectors


def label_clusters(
    embedded_results: list[EmbeddedResult],
    n_clusters: int,
) -> dict:
    """Label clusters based on dominant traits and pass/fail composition."""
    labels = {}
    for k in range(n_clusters):
        members = [r for r in embedded_results if r.cluster_id == k]
        if not members:
            labels[k] = "empty"
            continue

        avg_score = sum(r.score for r in members) / len(members)
        fail_rate = sum(1 for r in members if r.status == "fail") / len(members)

        trait_avgs = {}
        for trait in PERSONALITY_TRAITS:
            vals = [r.traits.get(trait, 50) for r in members]
            trait_avgs[trait] = sum(vals) / len(vals)

        dominant_trait = max(trait_avgs, key=lambda t: abs(trait_avgs[t] - 50))
        dominant_val = trait_avgs[dominant_trait]

        if fail_rate > 0.6:
            quality = "failing"
        elif fail_rate > 0.3:
            quality = "mixed"
        else:
            quality = "passing"

        trait_level = "high" if dominant_val > 60 else "low" if dominant_val < 40 else "moderate"
        labels[k] = f"{quality} | {trait_level} {dominant_trait} (n={len(members)}, avg={avg_score:.0f})"

    return labels


class Profiler:
    """
    Behavioral profiler for LLM outputs.

    Embeds responses, projects to 2D/3D, clusters behavioral patterns,
    extracts personality traits, and computes contrastive steering vectors.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        n_clusters: int = 4,
        projection: str = "auto",
        verbose: bool = True,
    ):
        self.embedding_model = embedding_model
        self.n_clusters = n_clusters
        self.projection = projection
        self.verbose = verbose

    def profile(
        self,
        results: list[dict],
        suite_name: str = "profile",
        model_name: str = "",
        high_threshold: float = 80,
        low_threshold: float = 40,
    ) -> ProfileResult:
        """
        Profile a list of eval results.

        Args:
            results: List of dicts with keys: question, agent_answer, expected_answer,
                     score, status, criteria_scores (dict), tags (optional list)
            suite_name: Name for this profiling run.
            model_name: Name of the model being profiled.
            high_threshold: Score threshold for "confidently right".
            low_threshold: Score threshold for "confidently wrong".

        Returns:
            ProfileResult with embeddings, projections, personality, and steering vectors.
        """
        if not results:
            return ProfileResult(suite_name=suite_name, model_name=model_name)

        if self.verbose:
            print(f"  Profiling {len(results)} results...")

        # Step 1: Compute traits for each result
        if self.verbose:
            print(f"  Computing personality traits...")
        embedded_results = []
        for i, r in enumerate(results):
            criteria = r.get("criteria_scores", {})
            agent_answer = r.get("agent_answer", "")
            expected_answer = r.get("expected_answer", "")
            traits = compute_traits(criteria, agent_answer, expected_answer)

            er = EmbeddedResult(
                index=i,
                question=r.get("question", ""),
                agent_answer=agent_answer,
                expected_answer=expected_answer,
                score=r.get("score", 0),
                status=r.get("status", "warn"),
                criteria_scores=criteria,
                traits=traits,
                tags=r.get("tags", []),
            )
            embedded_results.append(er)

        # Step 2: Embed agent answers
        if self.verbose:
            print(f"  Embedding {len(embedded_results)} responses with {self.embedding_model}...")
        texts = [r.agent_answer or "(empty)" for r in embedded_results]
        embeddings = embed_texts(texts, self.embedding_model)

        for i, er in enumerate(embedded_results):
            er.embedding = embeddings[i].tolist()

        # Step 3: Project to 2D and 3D
        projection_method = self.projection
        if self.verbose:
            print(f"  Projecting to 2D/3D ({projection_method})...")

        if len(embeddings) >= 4:
            if projection_method == "auto":
                try:
                    proj_2d = project_umap(embeddings, 2)
                    proj_3d = project_umap(embeddings, 3)
                    projection_method = "umap"
                except ImportError:
                    proj_2d = project_pca(embeddings, 2)
                    proj_3d = project_pca(embeddings, 3)
                    projection_method = "pca"
            elif projection_method == "umap":
                proj_2d = project_umap(embeddings, 2)
                proj_3d = project_umap(embeddings, 3)
            else:
                proj_2d = project_pca(embeddings, 2)
                proj_3d = project_pca(embeddings, 3)

            for i, er in enumerate(embedded_results):
                er.projection_2d = proj_2d[i].tolist()
                er.projection_3d = proj_3d[i].tolist()
        else:
            projection_method = "none (too few results)"

        # Step 4: Cluster
        n_clusters = min(self.n_clusters, len(embeddings))
        if self.verbose:
            print(f"  Clustering into {n_clusters} groups...")

        if len(embeddings) >= 2:
            cluster_labels_arr = cluster_kmeans(embeddings, n_clusters)
            for i, er in enumerate(embedded_results):
                er.cluster_id = int(cluster_labels_arr[i])
        cluster_labs = label_clusters(embedded_results, n_clusters)

        # Step 5: Aggregate personality
        if self.verbose:
            print(f"  Computing personality profile...")
        personality = aggregate_personality(embedded_results)

        # Step 6: Extract contrastive pairs
        if self.verbose:
            print(f"  Extracting contrastive pairs...")
        contrastive_pairs = extract_contrastive_pairs(
            embedded_results, high_threshold, low_threshold
        )

        # Step 7: Compute steering vectors
        if self.verbose:
            print(f"  Computing steering vectors...")
        steering_vectors = compute_steering_vectors(embedded_results, contrastive_pairs)

        if self.verbose:
            print(f"  Done: {len(contrastive_pairs)} contrastive pairs, {len(steering_vectors)} steering vectors")

        return ProfileResult(
            suite_name=suite_name,
            model_name=model_name,
            total_results=len(embedded_results),
            embedded_results=embedded_results,
            personality=personality,
            clusters={str(k): v for k, v in cluster_labs.items()},
            steering_vectors=steering_vectors,
            contrastive_pairs=contrastive_pairs,
            embedding_model=self.embedding_model,
            projection_method=projection_method,
        )

    def profile_from_json(self, path: str, model_name: str = "") -> ProfileResult:
        """Profile from a saved results JSON file."""
        import json
        with open(path, "r") as f:
            data = json.load(f)

        results_list = data.get("results", data if isinstance(data, list) else [])
        return self.profile(
            results_list,
            suite_name=data.get("suite_name", "loaded"),
            model_name=model_name or data.get("model_name", ""),
        )
