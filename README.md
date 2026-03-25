# cane-personality

Behavioral profiling benchmark for LLMs. Profile any model's personality, extract steering vectors, generate DPO training pairs to fix what's broken.

[![PyPI](https://img.shields.io/pypi/v/cane-personality)](https://pypi.org/project/cane-personality/)

```
                    INTELLECT-3   OLMo-2   Qwen-2.5   DeepSeek-V3
Overconfidence         72.1       45.3      68.4        61.7
Calibration            38.9       67.2      41.8        52.3
Verbosity              84.3       56.1      91.2        78.6
Hedging                22.4       48.7      18.9        31.5
Groundedness           41.2       62.8      44.1        55.9
Completeness           68.7       71.3      73.2        69.8

Grade                   D          B         D           C
Steering Vectors        3          1         3           2
DPO Pairs Generated    47         12        52          31
```

## What it does

300-question behavioral probe suite across 6 personality traits, 3 difficulty tiers. Run it against any model, get three outputs:

1. **Behavioral profile** with trait scores, embedding space visualization, and cluster analysis
2. **Steering vectors** pointing from overconfident to calibrated in embedding space
3. **DPO training pairs** (chosen/rejected) ready for TRL, OpenRLHF, or PRIME-RL

## Quick start

```bash
pip install cane-personality[all]
export ANTHROPIC_API_KEY=sk-ant-...

# Profile a model
cane-personality run --model claude-sonnet-4-5-20250929 --html report.html

# Profile with OpenAI
cane-personality run --provider openai --model gpt-4o

# Profile local model via Ollama
cane-personality run --provider ollama --model llama3 --base-url http://localhost:11434/v1
```

## Three outputs

### 1. Behavioral profile

```bash
cane-personality run --model claude-sonnet-4-5-20250929 --html report.html
```

Interactive HTML report with:
- Trait scores across 6 dimensions (radar chart)
- Embedding space scatter plot (pass/warn/fail clusters)
- Cluster analysis with semantic labels

### 2. Steering vectors

```bash
cane-personality run --model my-model --export-vectors vectors.json
```

Directions in embedding space between behavioral poles:
- **Overconfidence vector**: calibrated confidence -> overconfidence
- **Quality vector**: high-quality -> low-quality responses

Export as JSON for representation engineering or inference-time intervention.

### 3. DPO training pairs

```bash
cane-personality run --model my-model --export-dpo pairs.jsonl
```

Every contrastive pair (confidently right vs. confidently wrong) exported as:

```json
{"prompt": "...", "chosen": "...", "rejected": "...", "trait": "overconfidence"}
```

Ready for TRL, OpenRLHF, or PRIME-RL. Tagged by trait so you can target specific behavioral fixes.

## Personality traits

| Trait | What it measures | Low score | High score |
|-------|-----------------|-----------|------------|
| **Overconfidence** | Confidently wrong | Well-calibrated | Confidently hallucinating |
| **Calibration** | Certainty matches correctness | Poorly calibrated | Well-calibrated |
| **Verbosity** | Response length vs expected | Terse | Rambling |
| **Hedging** | Unnecessary qualification | Direct and clear | Over-qualified |
| **Groundedness** | Answers grounded in facts | Fabricating | Fact-based |
| **Completeness** | Covers all key points | Missing parts | Thorough |

## Probe suite

300 questions across 6 traits and 3 difficulty tiers:

| Trait | Easy (15) | Medium (20) | Hard (15) | Total |
|-------|-----------|-------------|-----------|-------|
| Overconfidence | Common facts | Misconceptions | Obscure topics | 50 |
| Calibration | Unknowable questions | Debatable topics | Uncertain science | 50 |
| Hedging | Basic math | Established facts | Definitive technical | 50 |
| Verbosity | Yes/no questions | One-sentence answers | Precise definitions | 50 |
| Groundedness | Fake citations | Obscure facts | Plausible fakes | 50 |
| Completeness | Two-part questions | Three-part comparisons | Multi-dimensional | 50 |

## Compare models

```bash
# Compare against shipped baselines
cane-personality compare --baselines intellect3,olmo2,qwen25 --html comparison.html

# Compare your profiles
cane-personality compare --profiles model_a.json,model_b.json --html comparison.html
```

Generates side-by-side comparison with trait table, overlaid radar charts, and per-trait rankings.

## Python API

```python
from cane_personality import Profiler, Judge, export_dpo_pairs

# Score responses with built-in judge
judge = Judge(provider="anthropic", model="claude-haiku-4-5-20241022")
score = judge.score(question, expected_answer, agent_answer)

# Profile from results
profiler = Profiler(embedding_model="all-MiniLM-L6-v2")
profile = profiler.profile(results, model_name="my-model")

# Access traits
print(profile.personality.trait_scores)

# Export steering vectors
for sv in profile.steering_vectors:
    print(f"{sv.name}: magnitude {sv.magnitude:.3f}")

# Generate reports
profile.to_html("report.html")

# Export DPO pairs
export_dpo_pairs(profile, "pairs.jsonl")
```

## Install

```bash
pip install cane-personality                   # core (numpy, pyyaml)
pip install cane-personality[anthropic]        # + Anthropic provider
pip install cane-personality[openai]           # + OpenAI/Ollama provider
pip install cane-personality[embeddings]       # + sentence-transformers
pip install cane-personality[all]              # everything
```

## How it works

```
Probe Suite (300 Q) --> Target Model --> LLM Judge --> Trait Scoring
                                                          |
                                    +---------+-----------+---------+
                                    |         |                     |
                              Embed (MiniLM)  |              DPO Pairs
                                    |         |             (chosen/rejected)
                              PCA / UMAP      |
                                    |         v
                              K-means    Steering Vectors
                              Clusters   (overconfidence,
                                          quality)
```

## License

MIT
