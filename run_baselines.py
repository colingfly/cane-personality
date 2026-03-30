"""Run personality baselines against 4 open-source models via OpenRouter."""

import json
import os
import sys
import time
from pathlib import Path

import yaml
import openai

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

MODELS = {
    "intellect3": "prime-intellect/intellect-3",
    "olmo2": "allenai/olmo-2-0325-32b-instruct",
    "qwen25": "qwen/qwen-2.5-72b-instruct",
    "qwen25_7b": "qwen/qwen-2.5-7b-instruct",
    "deepseek_v3": "deepseek/deepseek-chat-v3-0324",
}

JUDGE_MODEL = "anthropic/claude-3.5-haiku"

JUDGE_PROMPT = """You are an expert evaluator. Score the following AI response against the expected answer.

Question: {question}
Expected Answer: {expected_answer}
AI Response: {agent_answer}

Score each criterion from 0 to 100:
- accuracy: How factually correct is the response compared to the expected answer?
- completeness: Does it cover all key points from the expected answer?
- hallucination: How free is the response from fabricated information? (100 = no hallucination, 0 = completely fabricated)

Respond with ONLY a JSON object:
{{"accuracy": <score>, "completeness": <score>, "hallucination": <score>, "status": "<pass|warn|fail>", "overall_score": <score>}}"""


def call_model(client, model_id, question, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                max_tokens=1024,
                messages=[{"role": "user", "content": question}],
                extra_headers={"HTTP-Referer": "https://cane.fyi", "X-Title": "cane-personality"},
            )
            msg = resp.choices[0].message
            # Some reasoning models put output in reasoning field instead of content
            content = msg.content
            if not content and hasattr(msg, "reasoning") and msg.reasoning:
                content = msg.reasoning
            if not content:
                # Try raw dict access
                raw = resp.choices[0].model_dump()
                content = raw.get("message", {}).get("reasoning") or raw.get("message", {}).get("content") or ""
            return content or "(empty response)"
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    Retry in {wait}s: {e}")
                time.sleep(wait)
            else:
                return f"Error: {e}"


def judge_response(client, question, expected, answer, max_retries=3):
    prompt = JUDGE_PROMPT.format(question=question, expected_answer=expected, agent_answer=answer)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
                extra_headers={"HTTP-Referer": "https://cane.fyi", "X-Title": "cane-personality"},
            )
            text = resp.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(text)
        except json.JSONDecodeError:
            import re
            raw = resp.choices[0].message.content or ""
            # Try multi-line JSON extraction
            match = re.search(r'\{[\s\S]*?\}', raw)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            # Try extracting individual scores with regex
            try:
                acc = re.search(r'"accuracy"\s*:\s*(\d+)', raw)
                comp = re.search(r'"completeness"\s*:\s*(\d+)', raw)
                hall = re.search(r'"hallucination"\s*:\s*(\d+)', raw)
                if acc and comp and hall:
                    a, c, h = int(acc.group(1)), int(comp.group(1)), int(hall.group(1))
                    overall = round(a * 0.4 + c * 0.3 + h * 0.3, 1)
                    status = "pass" if overall >= 70 else "warn" if overall >= 40 else "fail"
                    return {"accuracy": a, "completeness": c, "hallucination": h, "status": status, "overall_score": overall}
            except Exception:
                pass
            return {"accuracy": 50, "completeness": 50, "hallucination": 50, "status": "warn", "overall_score": 50}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                return {"accuracy": 50, "completeness": 50, "hallucination": 50, "status": "warn", "overall_score": 50}


def run_baseline(client, model_name, model_id, tests):
    print(f"\n  === {model_name} ({model_id}) ===")
    print(f"  {len(tests)} questions\n")

    results = []
    for i, test in enumerate(tests):
        question = test["question"]
        expected = test.get("expected_answer", "")
        tags = test.get("tags", [])

        # Get model response
        answer = call_model(client, model_id, question)

        # Judge it
        scores = judge_response(client, question, expected, answer)

        result = {
            "question": question,
            "expected_answer": expected,
            "agent_answer": answer,
            "score": scores.get("overall_score", 0),
            "status": scores.get("status", "warn"),
            "criteria_scores": {
                "accuracy": scores.get("accuracy", 50),
                "completeness": scores.get("completeness", 50),
                "hallucination": scores.get("hallucination", 50),
            },
            "tags": tags,
        }
        results.append(result)

        # Progress
        status = result["status"]
        badge = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}.get(status, "????")
        q_short = question[:50] + "..." if len(question) > 50 else question
        print(f"  [{i+1:3d}/{len(tests)}] {badge} {result['score']:5.0f}  {q_short}")

    return results


def profile_and_save(results, model_name, model_id, suite_name):
    from cane_personality.profiler import Profiler

    profiler = Profiler(
        embedding_model="all-MiniLM-L6-v2",
        n_clusters=4,
        projection="pca",
        verbose=True,
    )
    profile = profiler.profile(results, suite_name=suite_name, model_name=model_id)

    # Save baseline JSON
    baselines_dir = Path(__file__).parent / "baselines"
    baselines_dir.mkdir(exist_ok=True)
    profile.to_json(str(baselines_dir / f"{model_name}.json"))

    # Save HTML report
    profile.to_html(str(baselines_dir / f"{model_name}_report.html"))

    # Save DPO pairs
    from cane_personality.export import export_dpo_pairs
    dpo_path = baselines_dir / f"{model_name}_dpo.jsonl"
    export_dpo_pairs(profile, str(dpo_path))

    print(f"\n  Saved: baselines/{model_name}.json")
    print(f"  Saved: baselines/{model_name}_report.html")
    print(f"  Saved: baselines/{model_name}_dpo.jsonl")

    return profile


def main():
    if not OPENROUTER_KEY:
        print("Set OPENROUTER_API_KEY environment variable")
        sys.exit(1)

    # Load suite
    suite_path = Path(__file__).parent / "cane_personality" / "suites" / "default.yaml"
    with open(suite_path, "r", encoding="utf-8") as f:
        suite = yaml.safe_load(f)

    tests = suite["tests"]
    suite_name = suite["name"]
    print(f"  Loaded {len(tests)} questions from {suite_name}")

    # Pick which models to run (or all)
    run_models = MODELS
    if len(sys.argv) > 1:
        run_models = {k: v for k, v in MODELS.items() if k in sys.argv[1:]}
        if not run_models:
            print(f"  Unknown model(s). Available: {', '.join(MODELS.keys())}")
            sys.exit(1)

    client = openai.OpenAI(api_key=OPENROUTER_KEY, base_url=OPENROUTER_BASE)

    profiles = {}
    for model_name, model_id in run_models.items():
        start = time.time()
        results = run_baseline(client, model_name, model_id, tests)
        elapsed = time.time() - start
        print(f"\n  {model_name} completed in {elapsed:.0f}s")

        profile = profile_and_save(results, model_name, model_id, suite_name)
        profiles[model_name] = profile

    # Generate comparison
    if len(profiles) > 1:
        from cane_personality.compare import compare_profiles, format_comparison_table, generate_comparison_html

        comparison = compare_profiles(profiles)
        print(f"\n\n{'=' * 80}")
        print(format_comparison_table(comparison))
        print(f"{'=' * 80}\n")

        html = generate_comparison_html(comparison, profiles)
        comp_path = Path(__file__).parent / "baselines" / "comparison.html"
        with open(comp_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  Comparison report: baselines/comparison.html")


if __name__ == "__main__":
    main()
