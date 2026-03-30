"""Run personality baseline using a local model with optional LoRA adapter."""

import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
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


def load_local_model(base_model, adapter_path=None):
    """Load base model in 4-bit, optionally with LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try GPU first, fall back to CPU float16
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        print("Loaded on GPU (4-bit)")
    except Exception as e:
        print(f"GPU load failed ({e}), loading on CPU float16...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="cpu",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        print("Loaded on CPU (float16) — inference will be slower")

    if adapter_path:
        print(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def generate_local(model, tokenizer, question, max_new_tokens=256):
    """Generate a response from the local model."""
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def judge_response(client, question, expected, answer, max_retries=3):
    """Use OpenRouter judge to score the response."""
    prompt = JUDGE_PROMPT.format(question=question, expected_answer=expected, agent_answer=answer)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
                extra_headers={"HTTP-Referer": "https://cane.fyi", "X-Title": "cane-personality"},
            )
            content = resp.choices[0].message.content.strip()
            # Extract JSON
            if "```" in content:
                content = content.split("```")[1].strip()
                if content.startswith("json"):
                    content = content[4:].strip()
            # Extract just the JSON object, ignore trailing text
            import re
            match = re.search(r'\{[^{}]*\}', content)
            if match:
                parsed = json.loads(match.group())
            else:
                parsed = json.loads(content)
            if "overall_score" not in parsed:
                parsed["overall_score"] = int((parsed.get("accuracy", 50) + parsed.get("completeness", 50) + parsed.get("hallucination", 50)) / 3)
            return parsed
        except Exception as e:
            print(f"    Judge error (attempt {attempt+1}): {e}", flush=True)
            print(f"    Raw content: {content[:200]}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {"accuracy": 50, "completeness": 50, "hallucination": 50, "status": "warn", "overall_score": 50}


def main():
    base_model = "Qwen/Qwen2.5-7B-Instruct"
    adapter_path = Path(__file__).parent / "trained" / "qwen25-7b-personality"
    output_name = "qwen25_7b_dpo"
    baselines_dir = Path(__file__).parent / "baselines"

    print(f"Loading {base_model} + LoRA adapter...")
    model, tokenizer = load_local_model(base_model, str(adapter_path))

    # Load questions
    suite_file = Path(__file__).parent / "cane_personality" / "suites" / "default.yaml"
    suite = yaml.safe_load(open(suite_file, encoding="utf-8"))
    all_questions = suite["tests"]
    # Filter to groundedness only if --groundedness flag
    if "--groundedness" in sys.argv:
        questions = [q for q in all_questions if "groundedness" in q.get("tags", [])]
        print(f"Filtered to {len(questions)} groundedness questions\n")
    else:
        questions = all_questions
        print(f"Loaded {len(questions)} questions\n")

    # Judge client
    client = openai.OpenAI(api_key=OPENROUTER_KEY, base_url=OPENROUTER_BASE)

    results = []
    for i, q in enumerate(questions):
        question = q["question"]
        expected = q["expected_answer"]

        # Generate locally
        answer = generate_local(model, tokenizer, question)

        # Judge via OpenRouter
        scores = judge_response(client, question, expected, answer)
        overall = scores.get("overall_score", 50)
        status = scores.get("status", "warn").upper()

        results.append({
            "index": i,
            "question": question,
            "agent_answer": answer,
            "expected_answer": expected,
            "score": overall,
            "status": status.lower(),
            "criteria_scores": {
                "accuracy": scores.get("accuracy", 50),
                "completeness": scores.get("completeness", 50),
                "hallucination": scores.get("hallucination", 50),
            },
        })

        status_str = f"{'PASS' if overall >= 85 else 'WARN' if overall >= 50 else 'FAIL'}"
        print(f"  [{i+1:3d}/{len(questions)}] {status_str:4s} {overall:5.0f}  {question[:55]}...")
        sys.stdout.flush()

    # Save results
    avg = sum(r["score"] for r in results) / len(results)
    fails = sum(1 for r in results if r["score"] < 50)

    output = {
        "suite_name": suite.get("name", "Personality Probe"),
        "model_name": f"{base_model} + DPO",
        "total_results": len(results),
        "results": results,
    }

    out_file = baselines_dir / f"{output_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Model: {base_model} + DPO adapter")
    print(f"  Average: {avg:.1f}")
    print(f"  Fails: {fails}")
    print(f"  Saved: {out_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
