"""compare.py -- Multi-model behavioral comparison engine.

Loads multiple ProfileResults and generates side-by-side comparisons
with trait deltas, ranking, and combined HTML reports.
"""

import json
from pathlib import Path
from typing import Optional

from cane_personality.types import ProfileResult, PersonalityProfile


def load_baseline(path: str) -> ProfileResult:
    """Load a pre-computed profile from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Reconstruct ProfileResult from saved JSON
    personality = None
    if data.get("personality"):
        p = data["personality"]
        personality = PersonalityProfile(
            trait_scores=p.get("trait_scores", {}),
            trait_descriptions=p.get("trait_descriptions", {}),
            dominant_traits=p.get("dominant_traits", []),
            risk_traits=p.get("risk_traits", []),
        )

    return ProfileResult(
        suite_name=data.get("suite_name", ""),
        model_name=data.get("model_name", ""),
        total_results=data.get("total_results", 0),
        personality=personality,
        clusters=data.get("clusters", {}),
        embedding_model=data.get("embedding_model", ""),
        projection_method=data.get("projection_method", ""),
    )


def load_baselines_dir(baselines_dir: str = None) -> dict[str, ProfileResult]:
    """Load all baseline profiles from the baselines directory."""
    if baselines_dir is None:
        baselines_dir = str(Path(__file__).parent.parent / "baselines")

    baselines = {}
    base_path = Path(baselines_dir)
    if not base_path.exists():
        return baselines

    for json_file in base_path.glob("*.json"):
        try:
            profile = load_baseline(str(json_file))
            name = profile.model_name or json_file.stem
            baselines[name] = profile
        except Exception:
            continue

    return baselines


def compare_profiles(profiles: dict[str, ProfileResult]) -> dict:
    """
    Compare multiple model profiles.

    Args:
        profiles: Dict mapping model_name -> ProfileResult

    Returns:
        Dict with comparison data: table, rankings, deltas
    """
    if not profiles:
        return {"models": [], "traits": [], "table": {}, "rankings": {}, "grades": {}}

    trait_names = ["overconfidence", "calibration", "verbosity", "hedging", "groundedness", "completeness"]
    model_names = list(profiles.keys())

    # Build comparison table
    table = {}
    for model_name, profile in profiles.items():
        if profile.personality:
            table[model_name] = profile.personality.trait_scores
        else:
            table[model_name] = {t: 0 for t in trait_names}

    # Compute grade for each model
    grades = {}
    dpo_counts = {}
    vector_counts = {}
    for model_name, profile in profiles.items():
        scores = table.get(model_name, {})
        # Grade based on calibration (higher = better) and overconfidence (lower = better)
        cal = scores.get("calibration", 50)
        oc = scores.get("overconfidence", 50)
        ground = scores.get("groundedness", 50)

        # Composite: high calibration + high groundedness + low overconfidence = good
        composite = (cal + ground + (100 - oc)) / 3

        if composite >= 75:
            grades[model_name] = "A"
        elif composite >= 62:
            grades[model_name] = "B"
        elif composite >= 50:
            grades[model_name] = "C"
        elif composite >= 38:
            grades[model_name] = "D"
        else:
            grades[model_name] = "F"

        dpo_counts[model_name] = len(profile.contrastive_pairs)
        vector_counts[model_name] = len(profile.steering_vectors)

    # Rankings per trait
    rankings = {}
    for trait in trait_names:
        # For overconfidence, hedging, verbosity: lower is better
        # For calibration, groundedness, completeness: higher is better
        reverse = trait in ("calibration", "groundedness", "completeness")
        ranked = sorted(
            model_names,
            key=lambda m: table.get(m, {}).get(trait, 50),
            reverse=reverse,
        )
        rankings[trait] = ranked

    return {
        "models": model_names,
        "traits": trait_names,
        "table": table,
        "rankings": rankings,
        "grades": grades,
        "dpo_counts": dpo_counts,
        "vector_counts": vector_counts,
    }


def format_comparison_table(comparison: dict) -> str:
    """Format comparison as a CLI-friendly ASCII table."""
    models = comparison["models"]
    traits = comparison["traits"]
    table = comparison["table"]
    grades = comparison["grades"]
    dpo_counts = comparison.get("dpo_counts", {})
    vector_counts = comparison.get("vector_counts", {})

    if not models:
        return "No models to compare."

    # Column widths
    label_width = 20
    col_width = max(14, max(len(m) for m in models) + 2)

    lines = []

    # Header
    header = " " * label_width
    for m in models:
        header += f"{m:>{col_width}}"
    lines.append(header)

    # Separator
    lines.append("-" * (label_width + col_width * len(models)))

    # Trait rows
    for trait in traits:
        row = f"{trait:<{label_width}}"
        for m in models:
            val = table.get(m, {}).get(trait, 0)
            row += f"{val:>{col_width}.1f}"
        lines.append(row)

    # Separator
    lines.append("-" * (label_width + col_width * len(models)))

    # Grade row
    row = f"{'Grade':<{label_width}}"
    for m in models:
        row += f"{grades.get(m, '?'):>{col_width}}"
    lines.append(row)

    # Steering vectors row
    row = f"{'Steering Vectors':<{label_width}}"
    for m in models:
        row += f"{vector_counts.get(m, 0):>{col_width}}"
    lines.append(row)

    # DPO pairs row
    row = f"{'DPO Pairs':<{label_width}}"
    for m in models:
        row += f"{dpo_counts.get(m, 0):>{col_width}}"
    lines.append(row)

    return "\n".join(lines)


def generate_comparison_html(comparison: dict, profiles: dict[str, ProfileResult]) -> str:
    """Generate side-by-side comparison HTML report."""
    models = comparison["models"]
    traits = comparison["traits"]
    table = comparison["table"]
    grades = comparison["grades"]
    dpo_counts = comparison.get("dpo_counts", {})
    vector_counts = comparison.get("vector_counts", {})

    trait_colors = {
        "overconfidence": "#ff6b6b",
        "calibration": "#51cf66",
        "verbosity": "#ffd43b",
        "hedging": "#ff922b",
        "groundedness": "#748ffc",
        "completeness": "#20c997",
    }

    grade_colors = {
        "A": "#51cf66", "B": "#748ffc", "C": "#ffd43b", "D": "#ff922b", "F": "#ff6b6b",
    }

    # Build table rows
    table_rows = ""
    for trait in traits:
        cells = ""
        values = [table.get(m, {}).get(trait, 0) for m in models]
        best_idx = values.index(min(values)) if trait in ("overconfidence", "hedging", "verbosity") else values.index(max(values))
        for i, m in enumerate(models):
            val = table.get(m, {}).get(trait, 0)
            color = trait_colors.get(trait, "#748ffc")
            bold = "font-weight:700;" if i == best_idx else ""
            cells += f'<td style="color:{color};{bold}">{val:.1f}</td>'
        table_rows += f'<tr><td style="color:#aaa;text-align:right;padding-right:16px">{trait}</td>{cells}</tr>'

    # Grade row
    grade_cells = ""
    for m in models:
        g = grades.get(m, "?")
        gc = grade_colors.get(g, "#888")
        grade_cells += f'<td style="color:{gc};font-weight:700;font-size:24px">{g}</td>'

    # Summary rows
    vector_cells = "".join(f'<td style="color:#aaa">{vector_counts.get(m, 0)}</td>' for m in models)
    dpo_cells = "".join(f'<td style="color:#aaa">{dpo_counts.get(m, 0)}</td>' for m in models)

    # Radar traces (one per model)
    model_colors = ["#748ffc", "#51cf66", "#ff6b6b", "#ffd43b", "#20c997", "#ff922b"]
    radar_traces = []
    for i, m in enumerate(models):
        scores = table.get(m, {})
        values = [scores.get(t, 0) for t in traits]
        color = model_colors[i % len(model_colors)]
        radar_traces.append(f"""{{
            type: 'scatterpolar',
            r: [{','.join(str(v) for v in values)},{values[0]}],
            theta: [{','.join(f'"{t}"' for t in traits)},"{traits[0]}"],
            fill: 'toself',
            fillcolor: '{color}22',
            line: {{color: '{color}'}},
            name: '{m}',
        }}""")

    header_cells = "".join(f'<th style="color:#fff;font-size:16px;padding:8px 16px">{m}</th>' for m in models)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Cane Personality Comparison</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0a; color: #e0e0e0; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 24px; font-weight: 600; color: #fff; margin-bottom: 4px; }}
  .subtitle {{ color: #888; font-size: 14px; margin-bottom: 24px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 24px 0; }}
  th, td {{ padding: 10px 16px; text-align: center; }}
  tr {{ border-bottom: 1px solid #222; }}
  .card {{ background: #141414; border: 1px solid #222; border-radius: 12px; padding: 20px; margin: 24px 0; }}
  #radar {{ width: 100%; height: 500px; }}
</style>
</head>
<body>
<div class="container">
  <h1>Model Behavioral Comparison</h1>
  <div class="subtitle">{len(models)} models | {len(traits)} traits | cane-personality</div>

  <div class="card">
    <table>
      <thead><tr><th></th>{header_cells}</tr></thead>
      <tbody>
        {table_rows}
        <tr style="border-top:2px solid #333"><td style="color:#aaa;text-align:right;padding-right:16px">Grade</td>{grade_cells}</tr>
        <tr><td style="color:#aaa;text-align:right;padding-right:16px">Steering Vectors</td>{vector_cells}</tr>
        <tr><td style="color:#aaa;text-align:right;padding-right:16px">DPO Pairs</td>{dpo_cells}</tr>
      </tbody>
    </table>
  </div>

  <div class="card">
    <div id="radar"></div>
  </div>
</div>

<script>
Plotly.newPlot('radar', [{','.join(radar_traces)}], {{
  polar: {{
    bgcolor: '#141414',
    radialaxis: {{ visible: true, range: [0, 100], color: '#444', gridcolor: '#222' }},
    angularaxis: {{ color: '#aaa', gridcolor: '#222' }},
  }},
  paper_bgcolor: '#141414',
  font: {{ color: '#e0e0e0', size: 12 }},
  margin: {{ l: 60, r: 60, t: 40, b: 40 }},
  legend: {{ font: {{ color: '#e0e0e0' }} }},
}});
</script>
</body>
</html>"""
