#!/usr/bin/env python3
"""QLoRA fine-tuning CLI for Gemma 4 on lecture datasets.

Usage::

    # Fine-tune on SlideVQA with defaults
    python tools/finetune.py --dataset slidevqa

    # Use a YAML config
    python tools/finetune.py --config configs/qlora_slidevqa.yaml

    # Override specific settings
    python tools/finetune.py --dataset slidevqa --lr 1e-4 --epochs 5 --gpu 1
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger("slide_gemma.finetune")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune Gemma 4 with QLoRA on lecture datasets.",
    )
    p.add_argument("--config", default=None,
                   help="Path to a YAML config file (overrides defaults)")

    g_model = p.add_argument_group("model")
    g_model.add_argument("--model", default="e2b", choices=["e2b", "e4b"],
                         help="Gemma 4 variant (default: e2b)")
    g_model.add_argument("--local-model-dir", default=None,
                         help="Path to a pre-downloaded model directory")
    g_model.add_argument("--no-4bit", action="store_true",
                         help="Disable 4-bit quantization (full bf16)")

    g_lora = p.add_argument_group("lora")
    g_lora.add_argument("--lora-r", type=int, default=None, help="LoRA rank (default: 16)")
    g_lora.add_argument("--lora-alpha", type=int, default=None, help="LoRA alpha (default: 32)")

    g_data = p.add_argument_group("dataset")
    g_data.add_argument("--dataset", default=None,
                        choices=["slidevqa", "m3av", "lpm"],
                        help="Dataset name")
    g_data.add_argument("--max-samples", type=int, default=None,
                        help="Cap training samples (useful for debugging)")
    g_data.add_argument("--max-seq-length", type=int, default=None,
                        help="Max sequence length (default: 2048)")

    g_train = p.add_argument_group("training")
    g_train.add_argument("--output-dir", default=None, help="Checkpoint output directory")
    g_train.add_argument("--epochs", type=int, default=None, help="Number of epochs (default: 3)")
    g_train.add_argument("--batch-size", type=int, default=None,
                         help="Per-device batch size (default: 2)")
    g_train.add_argument("--grad-accum", type=int, default=None,
                         help="Gradient accumulation steps (default: 4)")
    g_train.add_argument("--lr", type=float, default=None, help="Learning rate (default: 2e-4)")
    g_train.add_argument("--gpu", type=int, default=None, help="CUDA device id (default: 0)")
    g_train.add_argument("--seed", type=int, default=None, help="Random seed (default: 42)")

    g_out = p.add_argument_group("output")
    g_out.add_argument("-v", "--verbose", action="store_true", help="Debug logging")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    from slide_gemma.training import TrainingConfig, QLoRATrainer

    if args.config:
        config = TrainingConfig.from_yaml(args.config)
        logger.info("Loaded config from %s", args.config)
    else:
        config = TrainingConfig()

    if args.model:
        config.model_name = args.model
    if args.local_model_dir:
        config.local_model_dir = args.local_model_dir
    if args.no_4bit:
        config.load_in_4bit = False
    if args.lora_r is not None:
        config.lora_r = args.lora_r
    if args.lora_alpha is not None:
        config.lora_alpha = args.lora_alpha
    if args.dataset:
        config.dataset_name = args.dataset
    if args.max_samples is not None:
        config.max_samples = args.max_samples
    if args.max_seq_length is not None:
        config.max_seq_length = args.max_seq_length
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.epochs is not None:
        config.num_train_epochs = args.epochs
    if args.batch_size is not None:
        config.per_device_train_batch_size = args.batch_size
    if args.grad_accum is not None:
        config.gradient_accumulation_steps = args.grad_accum
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.gpu is not None:
        config.gpu = args.gpu
    if args.seed is not None:
        config.seed = args.seed

    if not config.dataset_name:
        sys.exit("Please specify a dataset with --dataset or in the config file.")

    banner = "=" * 60
    print(f"\n{banner}")
    print("  Lecture-Lens QLoRA Fine-Tuning")
    print(f"{banner}\n")
    print(f"  Model     : {config.model_name}")
    print(f"  4-bit     : {config.load_in_4bit}")
    print(f"  LoRA r    : {config.lora_r}")
    print(f"  Dataset   : {config.dataset_name}")
    print(f"  Epochs    : {config.num_train_epochs}")
    print(f"  Batch     : {config.per_device_train_batch_size} x {config.gradient_accumulation_steps}")
    print(f"  LR        : {config.learning_rate}")
    print(f"  Output    : {config.output_dir}")
    print(f"  GPU       : {config.gpu}")
    print()

    trainer = QLoRATrainer(config)
    trainer.setup()
    trainer.train()
    save_dir = trainer.save()

    print(f"\n{banner}")
    print(f"  Training complete!")
    print(f"  Adapter saved to: {save_dir}")
    print(f"{banner}\n")


if __name__ == "__main__":
    main()
