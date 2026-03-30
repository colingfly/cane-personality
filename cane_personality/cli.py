"""cli.py -- Command-line interface for cane-personality.

Usage:
    cane-personality run --model claude-sonnet-4-5-20250929 --html report.html
    cane-personality run --provider openai --model gpt-4o
    cane-personality run --suite custom.yaml --export-dpo pairs.jsonl
    cane-personality compare --baselines intellect3,olmo2 --html comparison.html
"""

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import yaml


COLORS = {
    "red": "\033[91m", "green": "\033[92m", "yellow": "\033[93m",
    "blue": "\033[94m", "cyan": "\033[96m", "bold": "\033[1m",
    "dim": "\033[2m", "reset": "\033[0m",
}

def _supports_color():
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

USE_COLOR = _supports_color()

def c(text, color):
    if not USE_COLOR:
        return text
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


def _load_suite(suite_path: str = None) -> dict:
    """Load test suite from YAML. Uses built-in default if no path given."""
    if suite_path is None:
        suite_path = str(Path(__file__).parent / "suites" / "default.yaml")
    with open(suite_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _checkpoint_path(args):
    """Determine the checkpoint file path based on output json path or default."""
    if args.output_json:
        return Path(args.output_json).with_suffix(".checkpoint.jsonl")
    return Path(".cane_checkpoint.jsonl")


def _load_checkpoint(ckpt_path):
    """Load completed results from a checkpoint file. Returns a dict keyed by question text."""
    completed = {}
    if not ckpt_path.exists():
        return completed
    with open(ckpt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                completed[entry["question"]] = entry
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def _append_checkpoint(ckpt_path, result):
    """Append a single result as a JSON line to the checkpoint file."""
    with open(ckpt_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def cmd_run(args):
    """Run personality profiling on a model."""
    from cane_personality.judge import Judge
    from cane_personality.profiler import Profiler
    from cane_personality.export import export_dpo_pairs, export_sft_examples, export_steering_vectors

    # Try to import tqdm for progress bars (optional dependency)
    try:
        from tqdm import tqdm
        _has_tqdm = True
    except ImportError:
        _has_tqdm = False

    # Load suite
    suite = _load_suite(args.suite)
    tests = suite.get("tests", [])
    suite_name = suite.get("name", "Personality Probe")

    model = args.model or "claude-haiku-4-5-20241022"
    provider = args.provider or "anthropic"

    print()
    print(f"  {c('cane-personality', 'cyan')} {c(suite_name, 'bold')}")
    print(f"  {len(tests)} probes | model: {model} | provider: {provider}")
    print()

    # Load custom judge prompt if provided
    judge_prompt_template = None
    if args.judge_prompt_file:
        prompt_path = Path(args.judge_prompt_file)
        if not prompt_path.exists():
            print(f"  {c('Error:', 'red')} judge prompt file not found: {args.judge_prompt_file}")
            sys.exit(1)
        judge_prompt_template = prompt_path.read_text(encoding="utf-8")

    # Initialize judge
    judge = Judge(
        provider=provider,
        model=model,
        api_key=args.api_key,
        base_url=args.base_url,
        prompt_template=judge_prompt_template,
    )

    # Initialize target model
    target_provider = args.target_provider or provider
    target_model = args.target_model or model
    target_base_url = args.target_base_url or args.base_url

    if not args.target_model and target_model == model:
        print(f"  {c('Note:', 'yellow')} Profiling {model} using itself as judge.")
        print(f"  Use --target-model to profile a different model.")
        print()

    fail_fast = getattr(args, "fail_fast", False)

    # Checkpoint / resume support
    ckpt_path = _checkpoint_path(args)
    no_resume = getattr(args, "no_resume", False)

    if no_resume and ckpt_path.exists():
        ckpt_path.unlink()

    resumed = {}
    if not no_resume:
        resumed = _load_checkpoint(ckpt_path)

    if resumed:
        print(f"  Resumed {len(resumed)} completed question(s) from checkpoint.")
        print()

    # Collect results: keep resumed entries in suite order, gather remaining tests
    results = []
    remaining_tests = []
    for test in tests:
        q = test["question"]
        if q in resumed:
            results.append(resumed[q])
        else:
            remaining_tests.append(test)

    # Build the iterable for the main loop
    pbar = None
    if _has_tqdm and remaining_tests:
        pbar = tqdm(
            remaining_tests,
            desc="Probing",
            total=len(remaining_tests),
            bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        )

    iterable = pbar if pbar is not None else remaining_tests

    # Run probes
    for i, test in enumerate(iterable):
        question = test["question"]
        expected = test.get("expected_answer", "")
        tags = test.get("tags", [])

        # Get model response
        agent_answer = None
        try:
            if target_provider == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=args.target_api_key or args.api_key or os.environ.get("ANTHROPIC_API_KEY", ""))
                resp = client.messages.create(
                    model=target_model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": question}],
                )
                agent_answer = resp.content[0].text
            else:
                import openai
                kwargs = {"api_key": args.target_api_key or args.api_key or os.environ.get("OPENAI_API_KEY", "")}
                if target_base_url:
                    kwargs["base_url"] = target_base_url
                client = openai.OpenAI(**kwargs)
                resp = client.chat.completions.create(
                    model=target_model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": question}],
                )
                agent_answer = resp.choices[0].message.content
        except KeyboardInterrupt:
            if pbar is not None:
                pbar.close()
            print(f"\n  Interrupted at question {len(results) + 1}/{len(tests)}.")
            break
        except Exception as e:
            if pbar is None:
                print(f"  {c('API error:', 'red')} {e}")
            if fail_fast:
                if pbar is not None:
                    pbar.close()
                print("  Stopping (--fail-fast).")
                sys.exit(1)
            # Mark as error, do not judge an error string
            result = {
                "question": question,
                "expected_answer": expected,
                "agent_answer": "",
                "score": 0,
                "status": "error",
                "criteria_scores": {"accuracy": 0, "completeness": 0, "hallucination": 0},
                "tags": tags,
            }
            results.append(result)
            _append_checkpoint(ckpt_path, result)

            q_short = question[:55] + "..." if len(question) > 55 else question
            if pbar is not None:
                pbar.set_postfix_str(f"ERR | {q_short}")
            else:
                print(f"  {c(' ERR  ', 'red')}     0  {q_short}")
            continue

        if agent_answer is None:
            continue

        # Judge the response
        scores = judge.score(question, expected, agent_answer)

        result = {
            "question": question,
            "expected_answer": expected,
            "agent_answer": agent_answer,
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
        _append_checkpoint(ckpt_path, result)

        # Print progress
        status = result["status"]
        score_val = result["score"]
        q_short = question[:55] + "..." if len(question) > 55 else question

        if pbar is not None:
            status_label = status.upper()
            pbar.set_postfix_str(f"{status_label} | {q_short}")
        else:
            if status == "pass":
                badge = c(" PASS ", "green")
            elif status == "warn":
                badge = c(" WARN ", "yellow")
            else:
                badge = c(" FAIL ", "red")
            print(f"  {badge} {score_val:>5.0f}  {q_short}")

    if pbar is not None:
        pbar.close()

    # If all questions were completed, remove the checkpoint file
    completed_count = sum(1 for r in results if r.get("question"))
    if completed_count >= len(tests) and ckpt_path.exists():
        ckpt_path.unlink()

    print()

    # Profile
    print(f"  {c('Running personality profiler...', 'cyan')}")
    profiler = Profiler(
        embedding_model=args.embedding_model or "all-MiniLM-L6-v2",
        n_clusters=args.clusters or 4,
        projection=args.projection or "auto",
        verbose=True,
    )
    profile = profiler.profile(results, suite_name=suite_name, model_name=target_model)

    profile.metadata = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "judge_model": model,
        "judge_provider": provider,
        "target_model": target_model,
        "suite_name": suite_name,
        "question_count": len(results),
    }

    # Print personality summary
    print()
    if profile.personality:
        print(f"  {c('Personality Profile', 'bold')}  {c(target_model, 'cyan')}")
        print()
        trait_colors_map = {
            "overconfidence": "red", "calibration": "green", "verbosity": "yellow",
            "hedging": "yellow", "groundedness": "blue", "completeness": "cyan",
        }
        for trait, score in profile.personality.trait_scores.items():
            bar_len = int(score / 5)
            bar = "=" * bar_len + "-" * (20 - bar_len)
            color = trait_colors_map.get(trait, "dim")
            risk = " RISK" if trait in profile.personality.risk_traits else ""
            dominant = " DOMINANT" if trait in profile.personality.dominant_traits and not risk else ""
            print(f"    {trait:<18} [{bar}] {c(f'{score:.1f}', color)}{c(risk, 'red')}{c(dominant, 'blue')}")

    if profile.steering_vectors:
        print()
        print(f"  {c('Steering Vectors', 'bold')}")
        for sv in profile.steering_vectors:
            print(f"    {sv.name:<18} magnitude: {sv.magnitude:.3f}  {sv.negative_label} ({sv.n_negative}) <-> {sv.positive_label} ({sv.n_positive})")

    if profile.contrastive_pairs:
        print()
        print(f"  {c(f'{len(profile.contrastive_pairs)} DPO training pairs extracted', 'green')}")

    # Exports
    if args.html:
        profile.to_html(args.html)
        print(f"  HTML report: {args.html}")

    if args.output_json:
        profile.to_json(args.output_json)
        print(f"  Profile JSON: {args.output_json}")

    if args.export_dpo:
        export_dpo_pairs(profile, args.export_dpo)
        print(f"  DPO pairs: {args.export_dpo}")

    if args.export_vectors:
        export_steering_vectors(profile, args.export_vectors)
        print(f"  Steering vectors: {args.export_vectors}")

    print()


def cmd_compare(args):
    """Compare multiple model profiles."""
    from cane_personality.compare import (
        load_baseline, load_baselines_dir, compare_profiles,
        format_comparison_table, generate_comparison_html,
    )

    profiles = {}

    # Load baselines
    if args.baselines:
        baseline_names = [b.strip() for b in args.baselines.split(",")]
        all_baselines = load_baselines_dir()

        for name in baseline_names:
            if name in all_baselines:
                profiles[name] = all_baselines[name]
            else:
                # Try loading as a file path
                path = Path(name)
                if path.exists():
                    profiles[path.stem] = load_baseline(str(path))
                else:
                    print(c(f"  Warning: baseline '{name}' not found", "yellow"))

    # Load additional profiles from --profiles flag
    if args.profiles:
        for p in args.profiles.split(","):
            p = p.strip()
            path = Path(p)
            if path.exists():
                profile = load_baseline(str(path))
                name = profile.model_name or path.stem
                profiles[name] = profile

    if not profiles:
        print(c("  No profiles to compare. Use --baselines or --profiles.", "red"))
        sys.exit(1)

    print()
    print(f"  {c('cane-personality compare', 'cyan')} {c(f'{len(profiles)} models', 'bold')}")
    print()

    comparison = compare_profiles(profiles)
    table_str = format_comparison_table(comparison)
    print(table_str)
    print()

    if args.html:
        html = generate_comparison_html(comparison, profiles)
        with open(args.html, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  HTML comparison: {args.html}")
        print()


def main():
    parser = argparse.ArgumentParser(
        prog="cane-personality",
        description="Behavioral profiling benchmark for LLMs",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---- run ----
    run_parser = subparsers.add_parser("run", help="Profile a model's personality")
    run_parser.add_argument("--suite", help="Path to YAML probe suite (default: built-in)")
    run_parser.add_argument("--model", help="Target model to profile (also used as judge if no --judge-model)")
    run_parser.add_argument("--provider", default="anthropic", help="Target model provider (anthropic/openai/ollama)")
    run_parser.add_argument("--api-key", help="API key for provider")
    run_parser.add_argument("--base-url", help="Base URL for OpenAI-compatible endpoints")
    run_parser.add_argument("--target-model", help="Model to profile (if different from judge)")
    run_parser.add_argument("--target-provider", help="Provider for target model")
    run_parser.add_argument("--target-api-key", help="API key for target model")
    run_parser.add_argument("--target-base-url", help="Base URL for target model")
    run_parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Embedding model")
    run_parser.add_argument("--clusters", type=int, default=4, help="Number of behavioral clusters")
    run_parser.add_argument("--projection", choices=["auto", "umap", "pca"], default="auto")
    run_parser.add_argument("--html", help="Output HTML report path")
    run_parser.add_argument("--output-json", help="Output profile JSON path")
    run_parser.add_argument("--export-dpo", help="Export DPO training pairs (JSONL)")
    run_parser.add_argument("--export-vectors", help="Export steering vectors (JSON)")
    run_parser.add_argument("--fail-fast", action="store_true", help="Stop on first API error")
    run_parser.add_argument("--no-resume", action="store_true", help="Force a fresh run, deleting any existing checkpoint")
    run_parser.add_argument("--judge-prompt-file", help="Path to a text file containing a custom judge prompt template (uses {question}, {expected_answer}, {agent_answer} placeholders)")
    run_parser.set_defaults(func=cmd_run)

    # ---- compare ----
    compare_parser = subparsers.add_parser("compare", help="Compare model profiles")
    compare_parser.add_argument("--baselines", help="Comma-separated baseline names (e.g., intellect3,olmo2)")
    compare_parser.add_argument("--profiles", help="Comma-separated profile JSON paths")
    compare_parser.add_argument("--html", help="Output comparison HTML path")
    compare_parser.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
