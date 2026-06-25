"""CPT Phase 0 — Continued Pre-training on workflow traces.

Trains a small language model (Qwen 0.5B) on formatted workflow traces
to learn plan generation from user requests.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)

logger = logging.getLogger("phase0_cpt.train")

HERE = Path(__file__).parent


@dataclass
class Config:
    model_base: str = "Qwen/Qwen2.5-0.5B"
    output_dir: str = "./training/phase0_cpt/checkpoints"
    torch_dtype: str = "bfloat16"

    train_file: str = "./training/phase0_cpt/data/train_sequences.jsonl"
    max_length: int = 2048
    dataset_repo: Optional[str] = None

    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01

    lora_enabled: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    seed: int = 42


def load_config(path: Optional[str] = None) -> Config:
    if path is None:
        path = HERE / "config.yaml"
    if not os.path.exists(path):
        logger.warning("Config not found at %s, using defaults", path)
        return Config()

    with open(path) as f:
        data = yaml.safe_load(f)

    return Config(
        model_base=data.get("model", {}).get("base", Config.model_base),
        output_dir=data.get("model", {}).get("output_dir", Config.output_dir),
        torch_dtype=data.get("model", {}).get("torch_dtype", Config.torch_dtype),
        train_file=data.get("data", {}).get("train_file", Config.train_file),
        max_length=data.get("data", {}).get("max_length", Config.max_length),
        dataset_repo=data.get("data", {}).get("dataset_repo"),
        batch_size=data.get("training", {}).get("batch_size", Config.batch_size),
        gradient_accumulation_steps=data.get("training", {}).get(
            "gradient_accumulation_steps", Config.gradient_accumulation_steps
        ),
        learning_rate=data.get("training", {}).get("learning_rate", Config.learning_rate),
        num_epochs=data.get("training", {}).get("num_epochs", Config.num_epochs),
        warmup_steps=data.get("training", {}).get("warmup_steps", Config.warmup_steps),
        logging_steps=data.get("training", {}).get("logging_steps", Config.logging_steps),
        save_steps=data.get("training", {}).get("save_steps", Config.save_steps),
        eval_steps=data.get("training", {}).get("eval_steps", Config.eval_steps),
        max_grad_norm=data.get("training", {}).get("max_grad_norm", Config.max_grad_norm),
        weight_decay=data.get("training", {}).get("weight_decay", Config.weight_decay),
        lora_enabled=data.get("lora", {}).get("enabled", Config.lora_enabled),
        lora_r=data.get("lora", {}).get("r", Config.lora_r),
        lora_alpha=data.get("lora", {}).get("alpha", Config.lora_alpha),
        lora_dropout=data.get("lora", {}).get("dropout", Config.lora_dropout),
    )


def tokenize_function(examples, tokenizer, max_length: int):
    texts = examples["text"]
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None,
    )
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings


def train(cfg: Config):
    set_seed(cfg.seed)

    dtype = torch.bfloat16 if cfg.torch_dtype == "bfloat16" else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training on %s with %s", device, cfg.torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_base,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if cfg.lora_enabled:
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        except ImportError:
            logger.warning("PEFT not installed, training full model")

    if not os.path.exists(cfg.train_file):
        logger.error("Training file not found: %s", cfg.train_file)
        sys.exit(1)

    dataset = load_dataset("json", data_files=cfg.train_file, split="train")
    tokenized = dataset.map(
        lambda x: tokenize_function(x, tokenizer, cfg.max_length),
        batched=True,
        remove_columns=dataset.column_names,
    )

    split = tokenized.train_test_split(test_size=0.05, seed=cfg.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        warmup_steps=cfg.warmup_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        max_grad_norm=cfg.max_grad_norm,
        weight_decay=cfg.weight_decay,
        fp16=dtype == torch.float16,
        bf16=dtype == torch.bfloat16,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()

    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info("Training complete. Model saved to %s", final_path)

    if cfg.dataset_repo:
        try:
            tokenized.push_to_hub(cfg.dataset_repo, split="train")
            logger.info("Dataset pushed to %s", cfg.dataset_repo)
        except Exception:
            logger.exception("Failed to push dataset to hub")


def main():
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    train(cfg)


if __name__ == "__main__":
    main()
