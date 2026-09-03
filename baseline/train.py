from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    get_linear_schedule_with_warmup,
)

from .common import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL,
    DEFAULT_STRIDE,
    TAGS,
    TokenizedNerDataset,
    load_fast_tokenizer,
    read_records,
    resolve_device,
    set_seed,
    validate_window,
)


def parse_args() -> argparse.Namespace:
    """Разбирает параметры обучения baseline."""

    parser = argparse.ArgumentParser(description="Train a minimal Transformer NER baseline.")
    parser.add_argument("--train", type=Path, default=Path("train.jsonl"))
    parser.add_argument("--dev", type=Path, default=Path("dev.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="use bfloat16 autocast for forward and loss computation (default: FP32)",
    )
    parser.add_argument(
        "--flash-attention",
        action="store_true",
        help="use FlashAttention 2 with variable-length unpadding (requires CUDA and --bf16)",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--fused-adamw", action="store_true", help="use fused CUDA AdamW")
    parser.add_argument("--max-train-records", type=int)
    parser.add_argument("--max-dev-records", type=int)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    """Отсекает некорректные числовые параметры до загрузки модели."""

    positive = {
        "epochs": args.epochs,
        "batch-size": args.batch_size,
        "gradient-accumulation-steps": args.gradient_accumulation_steps,
        "max-length": args.max_length,
    }
    for name, value in positive.items():
        if value < 1:
            raise ValueError(f"{name} must be positive")
    optional_positive = {
        "max-train-records": args.max_train_records,
        "max-dev-records": args.max_dev_records,
    }
    for name, value in optional_positive.items():
        if value is not None and value < 1:
            raise ValueError(f"{name} must be positive")
    if args.prefetch_factor < 1:
        raise ValueError("prefetch-factor must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if args.weight_decay < 0:
        raise ValueError("weight-decay must be non-negative")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("warmup-ratio must be in [0, 1)")
    if args.max_grad_norm <= 0:
        raise ValueError("max-grad-norm must be positive")
    if args.flash_attention and not args.bf16:
        raise ValueError("flash-attention requires --bf16")


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    """Создаёт пустой каталог запуска или требует явное разрешение перезаписи."""

    if path.exists() and any(path.iterdir()) and not overwrite:
        raise ValueError(
            f"output directory is not empty: {path}; use --overwrite-output-dir to reuse it"
        )
    path.mkdir(parents=True, exist_ok=True)


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Переносит все тензоры batch на выбранное устройство."""

    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _loss_weight(batch: dict[str, torch.Tensor]) -> int:
    """Возвращает число непустых token labels для усреднения loss."""

    return int((batch["labels"] != -100).sum().item())


def _autocast_context(device: torch.device, bf16: bool):
    """Включает опциональный bfloat16 autocast, сохраняя FP32 по умолчанию."""

    if not bf16:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


_H100_DENSE_BF16_FLOPS = 989e12


def _estimate_hfu(model: torch.nn.Module, input_tokens: int, elapsed_seconds: float, device: torch.device) -> float:
    """Estimates hardware FLOPs utilization for transformer training."""

    if device.type != "cuda" or elapsed_seconds <= 0 or input_tokens <= 0:
        return 0.0
    peak_flops = _H100_DENSE_BF16_FLOPS if "H100" in torch.cuda.get_device_name(device) else 1e12
    # Standard training estimate: forward + backward ~= 6 FLOPs per parameter/token.
    estimated_flops = 6.0 * sum(parameter.numel() for parameter in model.parameters()) * input_tokens
    return estimated_flops / (peak_flops * elapsed_seconds)


@torch.inference_mode()
def evaluate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    bf16: bool,
) -> float:
    """Считает средний dev loss с весом по числу размеченных токенов."""

    model.eval()
    weighted_loss = 0.0
    token_count = 0
    for batch in tqdm(loader, desc="Dev loss", unit="batch", leave=False):
        batch = _move_batch(batch, device)
        with _autocast_context(device, bf16):
            output = model(**batch)
        weight = _loss_weight(batch)
        weighted_loss += float(output.loss.item()) * weight
        token_count += weight
    if not token_count:
        raise RuntimeError("dev dataset contains no labeled tokens")
    return weighted_loss / token_count


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler: Any,
    device: torch.device,
    *,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    bf16: bool,
    writer: Any = None,
    global_step: int = 0,
) -> tuple[float, int, int]:
    """Выполняет эпоху и возвращает средний token loss и номер шага оптимизатора."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    weighted_loss = 0.0
    token_count = 0
    input_tokens = 0

    progress = tqdm(loader, desc="Train", unit="batch", leave=False)
    for batch_index, batch in enumerate(progress, start=1):
        input_tokens += int(batch["attention_mask"].sum())
        batch = _move_batch(batch, device)
        with _autocast_context(device, bf16):
            output = model(**batch)
        loss = output.loss
        group_start = ((batch_index - 1) // gradient_accumulation_steps) * (
            gradient_accumulation_steps
        )
        group_size = min(gradient_accumulation_steps, len(loader) - group_start)
        (loss / group_size).backward()

        weight = int((batch["labels"] != -100).sum())
        weighted_loss += loss.detach() * weight
        token_count += weight
        should_step = batch_index % gradient_accumulation_steps == 0 or batch_index == len(loader)
        if should_step:
            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if writer is not None:
                writer.add_scalar("loss/train_step", loss.detach().item(), global_step)
                writer.add_scalar("learning_rate/step", scheduler.get_last_lr()[0], global_step)
        if batch_index % 10 == 0 or batch_index == len(loader):
            progress.set_postfix(loss=f"{loss.detach().item():.4f}")

    if not token_count:
        raise RuntimeError("train dataset contains no labeled tokens")
    return float((weighted_loss / token_count).item()), global_step, input_tokens


def _save_model(
    model: torch.nn.Module,
    tokenizer: Any,
    model_dir: Path,
    config: dict[str, Any],
) -> None:
    """Сохраняет веса, tokenizer и параметры оконного инференса."""

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    (model_dir / "baseline_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
def run(args: argparse.Namespace) -> Path:
    """Обучает baseline и сохраняет checkpoint с минимальным dev loss."""

    _validate_args(args)
    output_dir = args.output_dir.expanduser().resolve()
    _prepare_output_dir(output_dir, args.overwrite_output_dir)
    device = resolve_device(args.device)
    set_seed(args.seed, seed_cuda=device.type == "cuda")

    train_records = read_records(
        args.train.expanduser().resolve(),
        require_entities=True,
        limit=args.max_train_records,
    )
    dev_records = read_records(
        args.dev.expanduser().resolve(),
        require_entities=True,
        limit=args.max_dev_records,
    )
    tokenizer = load_fast_tokenizer(args.model_name)
    validate_window(tokenizer, args.max_length, args.stride)

    train_dataset = TokenizedNerDataset(
        train_records,
        tokenizer,
        max_length=args.max_length,
        stride=args.stride,
        description="Tokenize train",
    )
    dev_dataset = TokenizedNerDataset(
        dev_records,
        tokenizer,
        max_length=args.max_length,
        stride=args.stride,
        description="Tokenize dev",
    )
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader_options = {
        "collate_fn": collator,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.persistent_workers and args.num_workers > 0,
        **({"prefetch_factor": args.prefetch_factor} if args.num_workers > 0 else {}),
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        **({"prefetch_factor": args.prefetch_factor} if args.num_workers > 0 else {}),
    )

    id2label = dict(enumerate(TAGS))
    label2id = {tag: index for index, tag in id2label.items()}
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(TAGS),
        id2label=id2label,
        label2id=label2id,
        **({"attn_implementation": "flash_attention_2"} if args.flash_attention else {}),
        **({"dtype": torch.bfloat16} if args.flash_attention and args.bf16 else {}),
    ).to(device)
    optimizer_kwargs = {"lr": args.learning_rate, "weight_decay": args.weight_decay}
    if args.fused_adamw:
        optimizer_kwargs["fused"] = True
    optimizer = AdamW(model.parameters(), **optimizer_kwargs)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    print(f"Device: {device}")
    print(f"Train: {len(train_records)} documents, {len(train_dataset)} windows")
    print(f"Dev: {len(dev_records)} documents, {len(dev_dataset)} windows")

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as error:
        raise ValueError(
            "TensorBoard logging requires the 'tensorboard' package; "
            "install project requirements first"
        ) from error
    writer = SummaryWriter(log_dir=output_dir)

    model_dir = output_dir / "model"
    history: list[dict[str, float | int]] = []
    best_dev_loss = float("inf")
    baseline_config = {
        "schema_version": 1,
        "base_model": args.model_name,
        "tags": list(TAGS),
        "max_length": args.max_length,
        "stride": args.stride,
        "seed": args.seed,
        "bf16": args.bf16,
        "fused_adamw": args.fused_adamw,
        "flash_attention": args.flash_attention,
        "attention_implementation": "flash_attention_2" if args.flash_attention else "default",
    }
    training_started_at = time.perf_counter()
    global_step = 0
    total_input_tokens = 0
    train_compute_seconds = 0.0
    for epoch in range(1, args.epochs + 1):
        epoch_started_at = time.perf_counter()
        train_loss, global_step, epoch_input_tokens = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            bf16=args.bf16,
            writer=writer,
            global_step=global_step,
        )
        train_compute_seconds += time.perf_counter() - epoch_started_at
        total_input_tokens += epoch_input_tokens
        dev_loss = evaluate_loss(model, dev_loader, device, bf16=args.bf16)
        history.append({"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss})
        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, dev_loss={dev_loss:.6f}")
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            _save_model(model, tokenizer, model_dir, baseline_config)
        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/dev", dev_loss, epoch)
            writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], epoch)
    training_time_seconds = time.perf_counter() - training_started_at
    hfu = _estimate_hfu(model, total_input_tokens, train_compute_seconds, device)
    if writer is not None:
        writer.add_scalar("training/time_seconds", training_time_seconds, args.epochs)
        writer.add_scalar("training/hfu", hfu, args.epochs)
        writer.flush()
        writer.close()

    run_summary = {
        **baseline_config,
        "train_records": len(train_records),
        "dev_records": len(dev_records),
        "train_windows": len(train_dataset),
        "dev_windows": len(dev_dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "best_dev_loss": best_dev_loss,
        "training_time_seconds": training_time_seconds,
        "training_compute_seconds": train_compute_seconds,
        "training_input_tokens": total_input_tokens,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "hfu": hfu,
        "tensorboard_log_dir": str(output_dir),
        "history": history,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"HFU: {hfu:.4f}")
    print(f"Best model: {model_dir}")
    return model_dir


def main() -> int:
    """Запускает CLI обучения с компактным сообщением об ошибке."""

    try:
        run(parse_args())
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
