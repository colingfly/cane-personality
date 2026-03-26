# cane-personality

Behavioral profiling benchmark for LLMs. Profile any model's personality, extract steering vectors, generate DPO training pairs to fix what's broken.

[![PyPI](https://img.shields.io/pypi/v/cane-personality)](https://pypi.org/project/cane-personality/)

```
                   Qwen-2.5-72B  OLMo-2-32B  DeepSeek-V3  INTELLECT-3  Qwen-2.5-7B
Overall Score           90.7        90.5        90.0         88.2         87.5
Overconfidence           3.8         3.3         6.2          6.8          6.0
Calibration             92.8        92.4        90.9         89.3         89.3
Verbosity               93.9        95.9        99.0         97.3         94.8
Hedging                  9.4         8.5         7.4          7.5         11.1
Groundedness            90.8        90.6        90.2         88.4         87.5
Completeness            86.9        86.9        88.9         86.8         83.9

Fails (out of 300)        10          11          19           22           16
DPO Pairs Generated       11          11          17           21           22
```

**DPO Training Results (Qwen-2.5-7B):**
22 auto-generated DPO pairs, one round of QLoRA training on an RTX 4070 laptop GPU (2h 11m):
- Fabrication fails: 16 to 7 (down 56%)
- 9 out of 16 groundedness failures fixed on unseen questions
- Model learned epistemic humility from 22 examples

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
