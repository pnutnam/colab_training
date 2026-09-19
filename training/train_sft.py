#!/usr/bin/env python3
"""QLoRA/SFT fine-tuning of a Hugging Face model — runs ON a Colab GPU VM.

Launched from your local machine via `make train` (scripts/launch.sh), which
prepends an env-injection header (HF_TOKEN + config from local .env) and runs
this file with `colab run --gpu <accelerator>`. The script is self-contained:

  1. pip-installs its own deps on the VM (skip with SKIP_DEPS=1)
  2. loads BASE_MODEL (4-bit QLoRA by default; --full for full-finetune)
  3. trains on DATASET_REPO (a HF dataset with a `text` or `messages` column)
  4. saves the LoRA adapter + pushes it to OUTPUT_REPO on the Hub

Artifacts live on the Hub, not the VM — the VM is torn down right after.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

# Keep this list as the single source of VM-side deps (there is no separate
# requirements file — this script is what gets sent to the VM).
DEPS = [
    "transformers>=4.55,<5",
    "datasets>=3.0,<4",
    "accelerate>=1.0",
    "peft>=0.14",
    "bitsandbytes>=0.45",
    "sentencepiece>=0.2",
    "hf_transfer>=0.1.8",
]


def ensure_deps() -> None:
    if os.environ.get("SKIP_DEPS") == "1":
        return
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # ~2-3x faster model pulls
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *DEPS],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    env = os.environ.get
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default=env("BASE_MODEL", "Qwen/Qwen3-1.7B"))
    p.add_argument("--dataset", default=env("DATASET_REPO", ""), help="HF dataset repo id")
    p.add_argument(
        "--dataset-config",
        default=env("DATASET_CONFIG", ""),
        help="Optional dataset config/subset name",
    )
    p.add_argument(
        "--output-repo",
        default=env("OUTPUT_REPO", ""),
        help="HF model repo to push the adapter to (created if missing)",
    )
    p.add_argument("--epochs", type=float, default=float(env("EPOCHS", "1")))
    p.add_argument("--lr", type=float, default=float(env("LEARNING_RATE", "2e-4")))
    p.add_argument("--max-length", type=int, default=int(env("MAX_LENGTH", "2048")))
    p.add_argument("--batch", type=int, default=int(env("PER_DEVICE_BATCH", "2")))
    p.add_argument("--grad-accum", type=int, default=int(env("GRAD_ACCUM", "8")))
    p.add_argument("--max-steps", type=int, default=int(env("MAX_STEPS", "-1")))
    p.add_argument("--full", action="store_true", help="Full finetune instead of QLoRA")
    p.add_argument(
        "--public-repo", action="store_true", help="Create OUTPUT_REPO as public (default: private)"
    )
    p.add_argument(
        "--save-steps",
        type=int,
        default=int(env("SAVE_STEPS", "250")),
        help="Checkpoint every N steps; each checkpoint is pushed to OUTPUT_REPO "
        "so a preempted run can be resumed",
    )
    p.add_argument(
        "--resume-from",
        default=env("RESUME_FROM", ""),
        help="Checkpoint name in OUTPUT_REPO to resume from, e.g. checkpoint-500",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset:
        sys.exit("No dataset given — set DATASET_REPO in .env or pass --dataset")
    if args.output_repo and not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN not set — cannot push to the Hub (run `make auth-hf`)")

    ensure_deps()

    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    class PushCheckpoints(TrainerCallback):
        """Upload each checkpoint to OUTPUT_REPO so preemptions are resumable."""

        def __init__(self, repo_id: str, out_dir: str):
            self.repo_id, self.out_dir = repo_id, out_dir

        def on_save(self, args, state, control, **kwargs):
            ckpt = f"{self.out_dir}/checkpoint-{state.global_step}"
            if os.path.isdir(ckpt):
                HfApi().upload_folder(
                    repo_id=self.repo_id,
                    folder_path=ckpt,
                    path_in_repo=f"checkpoints/checkpoint-{state.global_step}",
                    commit_message=f"checkpoint {state.global_step}",
                )
                print(f"[train] pushed checkpoint-{state.global_step} to {self.repo_id}")

    started = time.time()
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print(f"[train] device={device_name} bf16={bf16}")
    print(f"[train] base={args.base_model} dataset={args.dataset} out={args.output_repo}")

    # Create the output repo up front — checkpoints push to it mid-run.
    if args.output_repo:
        HfApi().create_repo(args.output_repo, exist_ok=True, private=not args.public_repo)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": torch.bfloat16 if bf16 else torch.float16,
        "attn_implementation": "sdpa",
    }
    if not args.full:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        )
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model.config.use_cache = False

    if not args.full:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        lora = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

    # ---- data -------------------------------------------------------------
    ds_kwargs = {"path": args.dataset}
    if args.dataset_config:
        ds_kwargs["name"] = args.dataset_config
    dataset = load_dataset(**ds_kwargs)
    split = "train" if "train" in dataset else next(iter(dataset.keys()))
    dataset = dataset[split]

    def to_text(example):
        if example.get("messages"):
            if getattr(tokenizer, "chat_template", None):
                return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
            return {"text": "\n".join(m["content"] for m in example["messages"])}
        return {"text": example["text"]}

    dataset = dataset.map(to_text, desc="format")

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=args.max_length,
            add_special_tokens=False,
        )

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names, desc="tokenize")
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    print(f"[train] {len(tokenized)} examples, max_length={args.max_length}")

    # ---- train ------------------------------------------------------------
    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    out_dir = "/content/outputs"
    callbacks = []
    if args.output_repo:
        callbacks.append(PushCheckpoints(args.output_repo, out_dir))
    trainer = Trainer(
        model=model,
        train_dataset=tokenized,
        data_collator=collator,
        callbacks=callbacks,
        args=TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=10,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=2,
            bf16=bf16,
            gradient_checkpointing=not args.full,
            report_to="wandb" if use_wandb else "none",
            run_name=args.output_repo or os.environ.get("WANDB_PROJECT", "colab-sft"),
            seed=42,
        ),
    )
    if args.resume_from:
        from huggingface_hub import snapshot_download

        local = snapshot_download(
            args.output_repo,
            allow_patterns=f"checkpoints/{args.resume_from}/**",
            local_dir="/content/resume",
        )
        ckpt_dir = f"{local}/checkpoints/{args.resume_from}"
        print(f"[train] resuming from {ckpt_dir}")
        trainer.train(resume_from_checkpoint=ckpt_dir)
    else:
        trainer.train()
    metrics = trainer.state.log_history[-1]

    # ---- save + push ------------------------------------------------------
    final_dir = f"{out_dir}/final_adapter"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    method = "Full fine-tune" if args.full else "QLoRA adapter"
    card = (
        "---\n"
        f"base_model: {args.base_model}\n"
        "library_name: peft\n"
        "---\n\n"
        f"# {args.output_repo or 'adapter'}\n\n"
        f"{method} trained on Colab ({device_name}).\n\n"
        f"- Base model: `{args.base_model}`\n"
        f"- Dataset: `{args.dataset}`\n"
        f"- Epochs: {args.epochs} | LR: {args.lr} | max_length: {args.max_length}\n"
        f"- Final train loss: {metrics.get('train_loss', 'n/a')}\n"
    )
    with open(f"{final_dir}/README.md", "w") as f:
        f.write(card)

    result = {
        "adapter_dir": final_dir,
        "base_model": args.base_model,
        "dataset": args.dataset,
        "examples": len(tokenized),
        "train_loss": metrics.get("train_loss"),
        "wall_minutes": round((time.time() - started) / 60, 1),
    }

    if args.output_repo:
        api = HfApi()
        api.create_repo(args.output_repo, exist_ok=True, private=not args.public_repo)
        api.upload_folder(
            repo_id=args.output_repo,
            folder_path=final_dir,
            commit_message=f"QLoRA adapter: {args.base_model} on {args.dataset}",
        )
        result["adapter_repo"] = f"https://huggingface.co/{args.output_repo}"
        print(f"[train] pushed adapter to {result['adapter_repo']}")

    print(f"[train] RESULT {json.dumps(result)}")


if __name__ == "__main__":
    main()
