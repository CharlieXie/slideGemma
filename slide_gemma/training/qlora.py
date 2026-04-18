"""QLoRA fine-tuning trainer for Gemma 4 multimodal models."""

from __future__ import annotations

import logging
import os

from .config import TrainingConfig

logger = logging.getLogger(__name__)


class QLoRATrainer:
    """Wraps PEFT + TRL to fine-tune Gemma 4 with QLoRA on lecture datasets."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.processor = None
        self.peft_model = None
        self.trainer = None

    def setup(self) -> None:
        """Load model, apply QLoRA adapters, and prepare the dataset."""
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu)

        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        from ..models.loader import load_model_for_training
        from ..data import DATASET_REGISTRY

        logger.info("Loading base model with %s quantization ...",
                     "4-bit" if self.config.load_in_4bit else "bf16")
        self.model, self.processor = load_model_for_training(
            model_name=self.config.model_name,
            local_dir=self.config.local_model_dir,
            load_in_4bit=self.config.load_in_4bit,
        )

        if self.config.load_in_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.peft_model = get_peft_model(self.model, lora_config)
        trainable, total = self._count_parameters()
        logger.info("LoRA applied: %d / %d trainable params (%.2f%%)",
                     trainable, total, 100.0 * trainable / total)

        logger.info("Loading dataset: %s ...", self.config.dataset_name)
        dataset_cls = DATASET_REGISTRY[self.config.dataset_name]
        ds_wrapper = dataset_cls()
        self.train_dataset = ds_wrapper.load(
            split=self.config.dataset_split,
            max_samples=self.config.max_samples,
        )
        logger.info("Dataset loaded: %d examples", len(self.train_dataset))

    def train(self) -> None:
        """Run the QLoRA fine-tuning loop."""
        from trl import SFTTrainer, SFTConfig

        assert self.peft_model is not None, "Call setup() first"

        sft_config = SFTConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            save_strategy=self.config.save_strategy,
            eval_strategy=self.config.eval_strategy,
            seed=self.config.seed,
            gradient_checkpointing=self.config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=self.config.dataloader_num_workers,
            report_to=self.config.report_to,
            remove_unused_columns=False,
            max_seq_length=self.config.max_seq_length,
            dataset_kwargs={"skip_prepare_dataset": True},
        )

        self.trainer = SFTTrainer(
            model=self.peft_model,
            processing_class=self.processor,
            train_dataset=self.train_dataset,
            args=sft_config,
            data_collator=self._collate_fn,
        )

        logger.info("Starting training for %d epochs ...", self.config.num_train_epochs)
        self.trainer.train()
        logger.info("Training complete.")

    def save(self, output_dir: str | None = None) -> str:
        """Save the LoRA adapter weights."""
        save_dir = output_dir or os.path.join(self.config.output_dir, "adapter")
        assert self.peft_model is not None, "Call setup() and train() first"
        self.peft_model.save_pretrained(save_dir)
        self.processor.save_pretrained(save_dir)
        logger.info("Adapter saved to %s", save_dir)
        return save_dir

    def _collate_fn(self, examples: list[dict]) -> dict:
        """Custom data collator for multimodal chat data.

        Each example has ``messages_json`` (JSON-serialized chat turns) and
        optionally ``images`` (list of PIL images).  We use the processor's
        chat template to convert these into model inputs.
        """
        import json
        import torch

        texts = []
        all_images = []

        for ex in examples:
            messages = json.loads(ex["messages_json"])
            images = ex.get("images", [])

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
            all_images.append(images)

        has_images = any(len(imgs) > 0 for imgs in all_images)

        if has_images:
            batch = self.processor(
                text=texts,
                images=[img for imgs in all_images for img in imgs] if has_images else None,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
            )
        else:
            batch = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
            )

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch

    def _count_parameters(self) -> tuple[int, int]:
        trainable = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.peft_model.parameters())
        return trainable, total
