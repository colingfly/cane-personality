"""report.py -- HTML report generator for personality profiles.

Generates a self-contained HTML report with Plotly scatter + radar charts,
trait bars, steering vector details, and cluster labels.
"""

import json

from cane_personality.types import ProfileResult


def generate_html_report(profile: ProfileResult) -> str:
    """Generate self-contained HTML report with Plotly scatter + radar charts."""
    results_json = json.dumps([r.to_dict() for r in profile.embedded_results])
    personality_json = json.dumps(profile.personality.to_dict() if profile.personality else {})
    steering_json = json.dumps([sv.to_dict() for sv in profile.steering_vectors])
    clusters_json = json.dumps(profile.clusters)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cane Personality Profile: {profile.suite_name}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0a; color: #e0e0e0; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 24px; font-weight: 600; margin-bottom: 8px; color: #fff; }}
  h2 {{ font-size: 18px; font-weight: 500; margin: 24px 0 12px; color: #fff; }}
  .subtitle {{ color: #888; font-size: 14px; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  .card {{ background: #141414; border: 1px solid #222; border-radius: 12px; padding: 20px; }}
  .card-full {{ grid-column: 1 / -1; }}
  .trait-bar {{ display: flex; align-items: center; margin: 8px 0; gap: 12px; }}
  .trait-name {{ width: 140px; font-size: 13px; color: #aaa; text-align: right; }}
  .trait-track {{ flex: 1; height: 8px; background: #222; border-radius: 4px; overflow: hidden; }}
  .trait-fill {{ height: 100%; border-radius: 4px; transition: width 0.5s; }}
  .trait-value {{ width: 40px; font-size: 13px; font-weight: 600; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }}
  .badge-risk {{ background: #3d1111; color: #ff6b6b; }}
  .badge-good {{ background: #0d3311; color: #51cf66; }}
  .badge-neutral {{ background: #1a1a2e; color: #748ffc; }}
  .steering {{ margin: 8px 0; padding: 12px; background: #1a1a1a; border-radius: 8px; }}
  .steering-name {{ font-weight: 600; font-size: 14px; color: #fff; }}
  .steering-desc {{ font-size: 12px; color: #888; margin-top: 4px; }}
  .steering-stat {{ font-size: 13px; color: #aaa; margin-top: 4px; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px; }}
  .stat {{ text-align: center; }}
  .stat-value {{ font-size: 28px; font-weight: 700; color: #fff; }}
  .stat-label {{ font-size: 12px; color: #888; margin-top: 4px; }}
  #scatter {{ width: 100%; height: 500px; }}
  #radar {{ width: 100%; height: 400px; }}
</style>
</head>
<body>
<div class="container">
  <h1>Agent Personality Profile</h1>
  <div class="subtitle">{profile.suite_name} | {profile.total_results} responses | {profile.embedding_model} | {profile.projection_method}</div>

  <div class="stat-grid">
    <div class="stat"><div class="stat-value" id="stat-total">{profile.total_results}</div><div class="stat-label">Responses</div></div>
    <div class="stat"><div class="stat-value" id="stat-clusters">{len(profile.clusters)}</div><div class="stat-label">Clusters</div></div>
    <div class="stat"><div class="stat-value" id="stat-vectors">{len(profile.steering_vectors)}</div><div class="stat-label">Steering Vectors</div></div>
    <div class="stat"><div class="stat-value" id="stat-pairs">{len(profile.contrastive_pairs)}</div><div class="stat-label">Contrastive Pairs</div></div>
  </div>

  <div class="grid">
    <div class="card card-full">
      <h2>Embedding Space</h2>
      <div id="scatter"></div>
    </div>

    <div class="card">
      <h2>Personality Radar</h2>
      <div id="radar"></div>
    </div>

    <div class="card">
      <h2>Trait Scores</h2>
      <div id="traits"></div>
    </div>

    <div class="card">
      <h2>Steering Vectors</h2>
      <div id="vectors"></div>
    </div>

    <div class="card">
      <h2>Cluster Labels</h2>
      <div id="cluster-labels"></div>
    </div>
  </div>
</div>

<script>
const results = {results_json};
const personality = {personality_json};
const steeringVectors = {steering_json};
const clusters = {clusters_json};

// Color map for status
const statusColors = {{ pass: '#51cf66', warn: '#ffd43b', fail: '#ff6b6b' }};

// Scatter plot
if (results.length > 0 && results[0].x !== undefined) {{
  const traces = ['pass', 'warn', 'fail'].map(status => {{
    const filtered = results.filter(r => r.status === status);
    return {{
      x: filtered.map(r => r.x),
      y: filtered.map(r => r.y),
      mode: 'markers',
      type: 'scatter',
      name: status,
      marker: {{
        color: statusColors[status],
        size: filtered.map(r => 6 + (r.score / 20)),
        opacity: 0.7,
      }},
      text: filtered.map(r =>
        `Score: ${{r.score}}<br>` +
        `Q: ${{r.question.substring(0, 60)}}...<br>` +
        `Cluster: ${{r.cluster_id}}<br>` +
        `Overconfidence: ${{r.traits.overconfidence || 'N/A'}}`
      ),
      hoverinfo: 'text',
    }};
  }});

  Plotly.newPlot('scatter', traces, {{
    paper_bgcolor: '#141414',
    plot_bgcolor: '#141414',
    font: {{ color: '#e0e0e0' }},
    xaxis: {{ showgrid: false, zeroline: false, title: 'Dimension 1' }},
    yaxis: {{ showgrid: false, zeroline: false, title: 'Dimension 2' }},
    legend: {{ x: 0, y: 1 }},
    margin: {{ l: 40, r: 20, t: 20, b: 40 }},
  }});
}}

// Radar chart
if (personality && personality.trait_scores) {{
  const traits = Object.keys(personality.trait_scores);
  const values = traits.map(t => personality.trait_scores[t]);

  Plotly.newPlot('radar', [{{
    type: 'scatterpolar',
    r: [...values, values[0]],
    theta: [...traits, traits[0]],
    fill: 'toself',
    fillcolor: 'rgba(116, 143, 252, 0.2)',
    line: {{ color: '#748ffc' }},
    marker: {{ color: '#748ffc', size: 6 }},
  }}], {{
    polar: {{
      bgcolor: '#141414',
      radialaxis: {{ visible: true, range: [0, 100], color: '#444', gridcolor: '#222' }},
      angularaxis: {{ color: '#aaa', gridcolor: '#222' }},
    }},
    paper_bgcolor: '#141414',
    font: {{ color: '#e0e0e0', size: 12 }},
    margin: {{ l: 60, r: 60, t: 40, b: 40 }},
    showlegend: false,
  }});
}}

// Trait bars
if (personality && personality.trait_scores) {{
  const container = document.getElementById('traits');
  const traitColors = {{
    overconfidence: '#ff6b6b',
    calibration: '#51cf66',
    verbosity: '#ffd43b',
    hedging: '#ff922b',
    groundedness: '#748ffc',
    completeness: '#20c997',
  }};
  for (const [trait, score] of Object.entries(personality.trait_scores)) {{
    const color = traitColors[trait] || '#748ffc';
    const badge = personality.risk_traits.includes(trait)
      ? '<span class="badge badge-risk">RISK</span>'
      : score > 70 ? '<span class="badge badge-good">STRONG</span>'
      : '<span class="badge badge-neutral">OK</span>';
    container.innerHTML += `
      <div class="trait-bar">
        <div class="trait-name">${{trait}} ${{badge}}</div>
        <div class="trait-track"><div class="trait-fill" style="width:${{score}}%;background:${{color}}"></div></div>
        <div class="trait-value" style="color:${{color}}">${{score}}</div>
      </div>
    `;
  }}
}}

// Steering vectors
const vecContainer = document.getElementById('vectors');
for (const sv of steeringVectors) {{
  vecContainer.innerHTML += `
    <div class="steering">
      <div class="steering-name">${{sv.name}}</div>
      <div class="steering-desc">${{sv.description}}</div>
      <div class="steering-stat">
        Magnitude: ${{sv.magnitude}} | ${{sv.negative_label}} (${{sv.n_negative}}) ←→ ${{sv.positive_label}} (${{sv.n_positive}})
      </div>
    </div>
  `;
}}
if (steeringVectors.length === 0) {{
  vecContainer.innerHTML = '<div class="steering-desc">Not enough contrastive data to compute steering vectors.</div>';
}}

// Cluster labels
const clusterContainer = document.getElementById('cluster-labels');
for (const [id, label] of Object.entries(clusters)) {{
  clusterContainer.innerHTML += `
    <div class="steering">
      <div class="steering-name">Cluster ${{id}}</div>
      <div class="steering-desc">${{label}}</div>
    </div>
  `;
}}
</script>
</body>
</html>"""
