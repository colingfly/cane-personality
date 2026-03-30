"""
train_dpo.py -- DPO fine-tuning using cane-personality generated pairs.

Trains a small model (Qwen-2.5-7B) using QLoRA to fit in 8GB VRAM.
Uses DPO pairs from cane-personality baselines to fix behavioral flaws.
"""
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig


def load_dpo_pairs(jsonl_path):
    """Load DPO pairs from cane-personality export."""
    pairs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            pairs.append({
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
            })
    return pairs


def main():
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    dpo_path = Path(__file__).parent / "baselines" / "qwen25_7b_dpo.jsonl"
    output_dir = Path(__file__).parent / "trained" / "qwen25-7b-personality"

    if not dpo_path.exists():
        print(f"DPO pairs not found: {dpo_path}")
        sys.exit(1)

    pairs = load_dpo_pairs(dpo_path)
    print(f"Loaded {len(pairs)} DPO pairs from {dpo_path}")

    if len(pairs) < 5:
        print("Not enough pairs for meaningful training. Need at least 5.")
        sys.exit(1)

    # Duplicate pairs to get more training signal from limited data
    # 21 pairs x 8 = 168 training examples
    pairs = pairs * 8
    print(f"Expanded to {len(pairs)} training examples (8x duplication)")

    dataset = Dataset.from_list(pairs)

    print(f"\nLoading {model_name} in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config - small rank to fit in VRAM
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    print("Applying LoRA...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # DPO training config
    training_args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        beta=0.1,  # DPO temperature
        max_length=512,
        logging_steps=1,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
    )

    print("\nStarting DPO training...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    print(f"\nSaving to {output_dir}...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("Done! Model saved. Run cane-personality against it to see the improvement.")


if __name__ == "__main__":
    main()
