"""Training configuration dataclass with YAML loading support."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainingConfig:
    # ── Model ─────────────────────────────────────────────────────────────
    model_name: str = "e2b"
    local_model_dir: str | None = None
    load_in_4bit: bool = True

    # ── LoRA ──────────────────────────────────────────────────────────────
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset_name: str = "slidevqa"
    dataset_split: str = "train"
    max_samples: int | None = None
    max_seq_length: int = 2048

    # ── Training ──────────────────────────────────────────────────────────
    output_dir: str = "./checkpoints/qlora_run"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    bf16: bool = True
    logging_steps: int = 10
    save_strategy: str = "epoch"
    eval_strategy: str = "no"
    eval_steps: int | None = None
    seed: int = 42
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 2
    report_to: str = "none"

    # ── GPU ────────────────────────────────────────────────────────────────
    gpu: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)
