"""Trainable text-only next-user emotion baselines; no JEPA or privileged inputs.

screen: seed 42, validation only. confirm: validation-selected challengers plus
the frozen control, three seeds, then test. all: both stages. check/benchmark do
not perform official training. Original source and completed runs stay read-only.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

import run_controlled_experiments as common

ROOT = Path(__file__).resolve().parent
LABELS = common.LABELS
SEEDS = [42, 43, 44]
REPOS = {"minilm": common.MODEL, "roberta": "FacebookAI/roberta-base"}
TRAIN_ARTIFACTS = ["best.pt", "history.json", "validation_predictions.npz", "result.json"]
TEST_ARTIFACTS = ["test_result.json", "test_predictions.jsonl", "classification_report.json",
                  "confusion_matrix.json", "stratified_errors.json"]


def plan(suite):
    specs = [{"name": "minilm_frozen", "backbone": "minilm", "unfreeze": 0},
             {"name": "minilm_last2", "backbone": "minilm", "unfreeze": 2},
             {"name": "minilm_full", "backbone": "minilm", "unfreeze": -1}]
    if suite in ["all", "roberta"]:
        specs += [{"name": "roberta_frozen", "backbone": "roberta", "unfreeze": 0},
                  {"name": "roberta_last4", "backbone": "roberta", "unfreeze": 4}]
    return specs if suite != "roberta" else specs[3:]


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def validate_rows(rows, stats):
    dialogue_sets = {}
    for split, records in rows.items():
        expected = stats["splits"][split]
        if len(records) != expected["samples"] or not records:
            raise ValueError(f"Sample count differs: {split}")
        keys = [(r["dialogue_id"], r["target_turn"]) for r in records]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate sample keys")
        for r in records:
            if not isinstance(r["label"], int) or r["label"] not in range(7):
                raise ValueError("Invalid seven-class label")
            if r["target_turn"] < 2 or r["target_turn"] % 2 or not r["history"].startswith("User: "):
                raise ValueError("Unexpected temporal boundary/role layout")
            if "\nSystem: " not in r["history"]:
                raise ValueError("History has no system reply")
        counts = np.bincount([r["label"] for r in records], minlength=7).tolist()
        if counts != [expected["class_counts"][label] for label in LABELS]:
            raise ValueError(f"Class counts differ: {split}")
        dialogue_sets[split] = {r["dialogue_id"] for r in records}
        # Source statistics count every officially assigned dialogue. Four real
        # training dialogues produce no usable next-user sample and are absent
        # from prepared rows; do not mistake that original filtering for leakage.
        if len(dialogue_sets[split]) > expected["dialogues"]:
            raise ValueError("Prepared rows contain more dialogues than the official split")
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if dialogue_sets[a] & dialogue_sets[b]:
            raise ValueError("Dialogue split leakage")


def load_source(source):
    stats = common.read(source / "data_stats.json")
    meta = common.read(source / "embedding_meta.json")
    if stats["labels"] != LABELS or stats["history_turns"] != 6:
        raise ValueError("This experiment requires original seven labels and six utterance turns")
    if meta["signature"]["max_tokens"] != 256 or meta["signature"]["truncation_side"] != "left":
        raise ValueError("Original source tokenization differs")
    hashes = {name: common.digest(source / name) for name in
              ["train.json", "validation.json", "test.json", "data_stats.json", "embedding_meta.json", "embeddings.npz"]}
    for split in ["train", "validation", "test"]:
        if hashes[f"{split}.json"] != meta["signature"]["sha256"][f"{split}.json"]:
            raise ValueError(f"Source was modified since original encoding: {split}")
    rows = {s: common.read(source / f"{s}.json") for s in ["train", "validation", "test"]}
    validate_rows(rows, stats)
    return rows, stats, hashes


def resolve_assets(backbone, allow_download):
    from huggingface_hub import snapshot_download
    repo = REPOS[backbone]
    patterns = ["config.json", "*.safetensors", "*.safetensors.index.json", "tokenizer*",
                "special_tokens_map.json", "vocab.txt", "vocab.json", "merges.txt", "1_Pooling/config.json"]
    try:
        folder = Path(snapshot_download(repo, cache_dir=str(ROOT / "data" / "models"),
                                        local_files_only=True, allow_patterns=patterns))
        if not any(folder.glob("*.safetensors")):
            raise FileNotFoundError("Missing safetensors weights")
    except Exception as error:
        if not allow_download:
            raise RuntimeError(f"{repo} is not fully cached. Use --allow-download (PowerShell: -AllowDownload).") from error
        print(f"Downloading model assets only: {repo}; no dataset download", flush=True)
        folder = Path(snapshot_download(repo, cache_dir=str(ROOT / "data" / "models"), allow_patterns=patterns))
    files = sorted(p for p in folder.rglob("*") if p.is_file() and
                   (p.suffix in [".json", ".txt", ".safetensors"]))
    hashes = {p.relative_to(folder).as_posix(): common.digest(p) for p in files}
    if backbone == "minilm":
        pooling = common.read(folder / "1_Pooling" / "config.json")
        if not pooling.get("pooling_mode_mean_tokens") or any(pooling.get(k) for k in
                ["pooling_mode_cls_token", "pooling_mode_max_tokens", "pooling_mode_mean_sqrt_len_tokens"]):
            raise ValueError("Cached MiniLM pooling differs from original")
    return {"repo": repo, "path": str(folder.resolve()), "revision": folder.name, "sha256": hashes}


def runtime(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Use kg_project and check nvidia-smi.")
    if args.precision == "auto":
        args.precision = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if args.device == "cuda" else "fp32"
    if args.device == "cpu" and args.precision != "fp32":
        raise ValueError("CPU runs require fp32")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 unsupported; use fp16 or fp32")
    torch.set_num_threads(args.threads)
    # Strict opt-out of TF32; repeatability settings recorded in provenance.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def amp_context(args):
    if args.precision == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16 if args.precision == "bf16" else torch.float16)


def ensure_manifest(args, hashes, assets):
    for source in [args.source, ROOT / "runs" / "controlled_v2", ROOT / "runs" / "posthoc_diagnostic",
                   ROOT / "runs" / "error_analysis", ROOT / "data", ROOT / "任务梳理与实验结果_20261014"]:
        common.protect_paths(args.out, source)
    value = {"schema_version": 1, "purpose": "text-only next-user emotion baseline",
             "code_sha256": common.digest(Path(__file__)), "helper_sha256": common.digest(Path(common.__file__)),
             "source": str(args.source), "source_sha256": hashes, "models": assets,
             "plan": plan(args.suite), "seeds": SEEDS,
             "training": {k: getattr(args, k) for k in ["device", "precision", "micro_batch", "accumulation",
                         "eval_batch", "epochs", "patience", "encoder_lr", "head_lr", "weight_decay",
                         "warmup_ratio", "max_tokens", "threads", "gradient_checkpointing", "top_k"]},
             "input_contract": {"text_field": "history", "history_turns": 6, "truncation": "left",
                                "future_text_input": False, "previous_label_input": False,
                                "target_turn_input": False, "pooling": "masked_mean_then_L2_normalize"},
             "loss": {"type": "CE", "weights": "sqrt(N/(7*n_c)).clip(0.25,5), train only",
                      "reduction": "sum(weighted_loss)/sum(target_weights) per optimizer window",
                      "auxiliary_loss": None, "sampler": "uniform_without_replacement"},
             "schedule": "linear warmup/decay; identical head schedule across all configurations",
             "selection": "seed42 validation screen -> top_k trainable challengers + each family frozen control -> three seeds",
             "test_context": "exploratory followup; test was already used for error diagnosis; never used for model selection",
             "environment": {"python": sys.version, **{p: importlib.metadata.version(p) for p in
                             ["torch", "transformers", "huggingface-hub", "numpy", "scikit-learn"]}}}
    path = args.out / "provenance.json"
    if path.exists():
        if common.read(path) != value:
            raise ValueError("Code/config/assets/source changed. Use a NEW output directory; no old artifacts overwritten.")
    elif args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Nonempty output has no matching manifest; choose a new output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    common.dump(path, value)
    args.fingerprint = common.fingerprint(value)
    return value


def history_texts(rows):
    # Sole encoder text source. NEVER concatenate last_user/future_text/labels/turn metadata.
    return [r["history"] for r in rows]


def token_cache(args, asset, backbone, split, rows):
    folder = args.out / "cache"
    folder.mkdir(exist_ok=True)
    path = folder / f"{backbone}_{split}_tokens.pt"
    identity = {"run": args.fingerprint, "backbone": backbone, "split": split, "field": "history"}
    metadata = path.with_suffix(".meta.json")
    if metadata.exists():
        saved = common.read(metadata)
        if saved["identity"] != identity or common.digest(path) != saved["sha256"]:
            raise ValueError("Token cache mismatch/corruption")
        return torch.load(path, map_location="cpu", weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(asset["path"], local_files_only=True, trust_remote_code=False)
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "right"
    texts = history_texts(rows)
    all_ids, lengths = [], []
    for offset in range(0, len(texts), 1024):
        batch = texts[offset:offset + 1024]
        encoded = tokenizer(batch, max_length=args.max_tokens, truncation=True, padding="max_length",
                            return_attention_mask=True, return_tensors="pt")
        all_ids.append(encoded["input_ids"].to(torch.int32))
        lengths.append(encoded["attention_mask"].sum(1).to(torch.int32))
    data = {"input_ids": torch.cat(all_ids), "lengths": torch.cat(lengths),
            "pad_id": tokenizer.pad_token_id, "labels": torch.tensor([r["label"] for r in rows])}
    atomic_save(data, path)
    common.dump(metadata, {"identity": identity, "sha256": common.digest(path), "samples": len(rows),
                           "max_length_fraction": float((data["lengths"] == args.max_tokens).float().mean()),
                           "note": "max-length fraction is not necessarily the truncated fraction"})
    print(f"Tokenized {backbone}/{split}: {len(rows)} history-only samples", flush=True)
    return data


def token_batch(data, indices, device):
    lengths = data["lengths"][indices]
    width = int(lengths.max())
    ids = data["input_ids"][indices, :width].long().to(device)
    mask = torch.arange(width, device=device)[None, :] < lengths.to(device)[:, None]
    # Only explicit encoder arguments are returned; labels never forwarded into encoder.
    return {"input_ids": ids, "attention_mask": mask.long()}


def pooled_embedding(encoder, batch):
    hidden = encoder(**batch, return_dict=True).last_hidden_state.float()
    mask = batch["attention_mask"].unsqueeze(-1).float()
    mean = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
    return F.normalize(mean, dim=-1)


class Head(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(dimension, 256), nn.GELU(), nn.LayerNorm(256), nn.Dropout(0.1))
        self.classifier = nn.Linear(256, 7)

    def forward(self, features):
        return self.classifier(self.trunk(features))


class TextPredictor(nn.Module):
    def __init__(self, encoder, unfreeze, seed, gradient_checkpointing=False):
        super().__init__()
        self.encoder = encoder
        self.unfreeze = unfreeze
        layers = self.encoder.encoder.layer
        if unfreeze > len(layers):
            raise ValueError("Requested more trainable layers than encoder has")
        for p in encoder.parameters():
            p.requires_grad_(unfreeze == -1)
        if unfreeze > 0:
            for layer in layers[-unfreeze:]:
                for p in layer.parameters():
                    p.requires_grad_(True)
        if gradient_checkpointing and unfreeze:
            encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # Re-seed AFTER loading the backbone: identical MiniLM heads for paired seeds.
        common.seed_all(seed)
        self.head = Head(encoder.config.hidden_size)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.unfreeze != -1:
            # Frozen encoder blocks keep eval behavior; partial tuning activates only top layers.
            self.encoder.eval()
            if self.unfreeze > 0:
                for layer in self.encoder.encoder.layer[-self.unfreeze:]:
                    layer.train()
        return self

    def forward(self, batch):
        if "features" in batch:
            return self.head(batch["features"])
        return self.head(pooled_embedding(self.encoder, batch))


def build_model(args, asset, spec, seed):
    common.seed_all(seed)
    encoder = AutoModel.from_pretrained(asset["path"], local_files_only=True,
                                       trust_remote_code=False, add_pooling_layer=False)
    return TextPredictor(encoder, spec["unfreeze"], seed, args.gradient_checkpointing).to(args.device)


@torch.no_grad()
def frozen_cache(args, asset, spec, split, tokens):
    path = args.out / "cache" / f"{spec['backbone']}_{split}_frozen.npz"
    metadata = path.with_suffix(".meta.json")
    identity = {"run": args.fingerprint, "backbone": spec["backbone"], "split": split, "precision": args.precision}
    if metadata.exists():
        saved = common.read(metadata)
        if saved["identity"] != identity or saved["sha256"] != common.digest(path):
            raise ValueError("Frozen feature cache mismatch")
        with np.load(path, allow_pickle=False) as data:
            return torch.from_numpy(data["features"].copy())
    model = build_model(args, asset, {**spec, "unfreeze": 0}, 42)
    model.eval()
    features = []
    started = last_print = time.perf_counter()
    for offset in range(0, len(tokens["labels"]), args.eval_batch):
        indices = torch.arange(offset, min(offset + args.eval_batch, len(tokens["labels"])))
        with amp_context(args):
            z = pooled_embedding(model.encoder, token_batch(tokens, indices, args.device))
        features.append(z.cpu())
        if time.perf_counter() - last_print > 20:
            print(f"Frozen encoding {spec['backbone']}/{split}: {offset}/{len(tokens['labels'])}", flush=True)
            last_print = time.perf_counter()
    data = torch.cat(features)
    if not torch.isfinite(data).all():
        raise ValueError("Non-finite frozen embedding")
    atomic_npz(path, features=data.numpy())
    common.dump(metadata, {"identity": identity, "sha256": common.digest(path),
                           "seconds": time.perf_counter() - started})
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()
    return data


def input_batch(data, indices, args, features=None):
    return {"features": features[indices].to(args.device)} if features is not None else token_batch(data, indices, args.device)


def optimizer_for(model, args):
    # Same weight decay for matched frozen/partial/full runs; no dormant semantic head.
    groups = [{"params": list(model.head.parameters()), "lr": args.head_lr}]
    parameters = [p for p in model.encoder.parameters() if p.requires_grad]
    if parameters:
        groups.append({"params": parameters, "lr": args.encoder_lr})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def training_windows(order, micro_batch, accumulation):
    width = micro_batch * accumulation
    for start in range(0, len(order), width):
        yield list(order[start:start + width].split(micro_batch))


def ce_numerator(logits, labels, weights):
    return F.cross_entropy(logits.float(), labels, weight=weights, reduction="sum")


def update_window(model, data, parts, weights, optimizer, scheduler, scaler, args, features=None):
    optimizer.zero_grad(set_to_none=True)
    denominator = sum(float(weights[data["labels"][idx].to(args.device)].sum()) for idx in parts)
    if denominator <= 0:
        raise ValueError("Invalid class-weight denominator")
    total = 0.0
    for indices in parts:
        labels = data["labels"][indices].to(args.device)
        with amp_context(args):
            logits = model(input_batch(data, indices, args, features))
            numerator = ce_numerator(logits, labels, weights)
            loss = numerator / denominator
        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss; try fp32 or smaller LR in a new output directory")
        scaler.scale(loss).backward()
        total += float(numerator.detach())
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    if not torch.isfinite(norm) and not scaler.is_enabled():
        raise ValueError("Non-finite gradient")
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    if scaler.get_scale() >= old_scale:
        scheduler.step()
    return total, denominator


@torch.no_grad()
def predict(model, data, args, features=None):
    model.eval()
    result = []
    last_print = time.perf_counter()
    for offset in range(0, len(data["labels"]), args.eval_batch):
        indices = torch.arange(offset, min(offset + args.eval_batch, len(data["labels"])))
        with amp_context(args):
            logits = model(input_batch(data, indices, args, features)).float()
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite evaluation logits")
        result.append(logits.cpu().numpy())
        if time.perf_counter() - last_print > 20:
            print(f"Evaluation: {offset}/{len(data['labels'])}", flush=True)
            last_print = time.perf_counter()
    return np.concatenate(result)


def scores(labels, predictions):
    y = np.asarray(labels)
    actual_negative, predicted_negative = np.isin(y, [1, 2, 4]), np.isin(predictions, [1, 2, 4])
    tp = int(np.sum(actual_negative & predicted_negative))
    precision = tp / max(int(predicted_negative.sum()), 1)
    recall = tp / max(int(actual_negative.sum()), 1)
    return {**common.metrics(y, predictions), "per_class": common.detailed_class_diagnostics(y, predictions),
            "negative": {"precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12),
                         "true_count": int(actual_negative.sum()), "pred_count": int(predicted_negative.sum()), "tp_count": tp}}


def identity(args, spec, seed):
    return {"run": args.fingerprint, "spec": spec, "seed": seed}


def verify_completion(folder, name, expected, artifacts):
    path = folder / name
    if not path.exists():
        return False
    saved = common.read(path)
    if saved["identity"] != expected or saved["status"] != "completed" or set(saved["artifacts"]) != set(artifacts):
        raise ValueError(f"Completion record mismatch: {folder}")
    for artifact, sha in saved["artifacts"].items():
        if common.digest(folder / artifact) != sha:
            raise ValueError(f"Completed artifact was modified: {folder / artifact}")
    return True


def seal_completion(folder, name, expected, artifacts):
    common.dump(folder / name, {"identity": expected, "status": "completed",
                               "artifacts": {f: common.digest(folder / f) for f in artifacts}})


def folder_for(args, spec, seed):
    return args.out / "models" / f"{spec['name']}_seed{seed}"


def checkpoint_payload(model, spec, asset, expected):
    return {"identity": expected, "spec": spec, "asset": asset, "labels": LABELS,
            "input_field": "history", "uses_future_text": False, "uses_previous_label": False,
            "head_state": {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()},
            "encoder_state": ({k: v.detach().cpu().clone() for k, v in model.encoder.state_dict().items()}
                              if spec["unfreeze"] else None)}


def restore_payload(model, payload, expected):
    if payload["identity"] != expected or payload["uses_future_text"] or payload["uses_previous_label"]:
        raise ValueError("Checkpoint identity/input contract mismatch")
    model.head.load_state_dict(payload["head_state"], strict=True)
    if payload["encoder_state"] is not None:
        model.encoder.load_state_dict(payload["encoder_state"], strict=True)


def train_one(args, rows, assets, spec, seed):
    folder = folder_for(args, spec, seed)
    expected = identity(args, spec, seed)
    if verify_completion(folder, "completion.json", expected, TRAIN_ARTIFACTS):
        print(f"Verified completed training: {spec['name']} seed={seed}", flush=True)
        return common.read(folder / "result.json")
    config = folder / "experiment_config.json"
    if config.exists() and common.read(config) != expected:
        raise ValueError("Partial experiment configuration mismatch")
    if folder.exists() and any(folder.iterdir()) and not config.exists():
        raise ValueError("Untracked partial experiment directory")
    folder.mkdir(parents=True, exist_ok=True)
    common.dump(config, expected)
    asset = assets[spec["backbone"]]
    # Test tokens/features are deliberately not constructed during training or screening.
    data = {s: token_cache(args, asset, spec["backbone"], s, rows[s]) for s in ["train", "validation"]}
    features = {s: (frozen_cache(args, asset, spec, s, data[s]) if spec["unfreeze"] == 0 else None) for s in data}
    model = build_model(args, asset, spec, seed)
    optimizer = optimizer_for(model, args)
    steps = math.ceil(len(rows["train"]) / (args.micro_batch * args.accumulation))
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(args.epochs * steps * args.warmup_ratio), args.epochs * steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
    counts = np.bincount(data["train"]["labels"].numpy(), minlength=7)
    weights = torch.from_numpy(common.compute_class_weights(counts, counts.sum(), "baseline_sqrt_clip5")).to(args.device)
    print(f"TRAIN {spec['name']} seed={seed}; trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}; "
          f"effective batch={args.micro_batch * args.accumulation}; {args.precision}", flush=True)
    best_score, best_epoch, history, start_epoch, best = -1.0, 0, [], 1, None
    last_path = folder / "last.pt"
    previous_seconds = 0.0
    if last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if saved["identity"] != expected:
            raise ValueError("Resume state differs from this experiment")
        restore_payload(model, saved["current"], expected)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        history, best, best_score, best_epoch = saved["history"], saved["best"], saved["best_score"], saved["best_epoch"]
        start_epoch, previous_seconds = saved["epoch"] + 1, saved["training_seconds"]
        print(f"Resume from completed epoch {start_epoch - 1}; unfinished epoch is replayed", flush=True)
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        if best_epoch and epoch - 1 - best_epoch >= args.patience:
            break
        # Epoch-specific RNG makes boundary resume reproduce dropout/order without stale global RNG.
        common.seed_all(seed + 1000 * epoch)
        model.train()
        order = torch.randperm(len(rows["train"]), generator=torch.Generator().manual_seed(seed + epoch))
        numerator_sum = denominator_sum = 0.0
        epoch_started = last_print = time.perf_counter()
        for step, parts in enumerate(training_windows(order, args.micro_batch, args.accumulation), 1):
            numerator, denominator = update_window(model, data["train"], parts, weights, optimizer, scheduler, scaler, args, features["train"])
            numerator_sum += numerator
            denominator_sum += denominator
            now = time.perf_counter()
            if now - last_print > 20 or step == steps:
                elapsed = now - epoch_started
                remaining = elapsed / step * (steps - step)
                print(f"{spec['name']} seed={seed} epoch={epoch}/{args.epochs} step={step}/{steps}; "
                      f"loss={numerator_sum/denominator_sum:.4f}; epoch ETA={remaining/60:.1f} min", flush=True)
                last_print = now
        logits = predict(model, data["validation"], args, features["validation"])
        val = scores(data["validation"]["labels"], logits.argmax(1))
        score = val["macro_f1"]
        history.append({"epoch": epoch, "loss": numerator_sum / denominator_sum,
                        "validation": val, "seconds": time.perf_counter() - epoch_started})
        print(f"VALIDATION {spec['name']} seed={seed} epoch={epoch} Macro-F1={score:.4%}", flush=True)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best = checkpoint_payload(model, spec, asset, expected)
        current = checkpoint_payload(model, spec, asset, expected)
        # Best state is inside the same atomic epoch checkpoint; crash cannot mismatch best/last.
        atomic_save({"identity": expected, "epoch": epoch, "current": current, "best": best,
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "scaler": scaler.state_dict(), "history": history, "best_score": best_score,
                     "best_epoch": best_epoch, "training_seconds": previous_seconds + time.perf_counter() - started}, last_path)
        common.dump(folder / "history.json", history)
        del current
        if epoch - best_epoch >= args.patience:
            break
    if best is None:
        raise RuntimeError("No finite validation checkpoint")
    restore_payload(model, best, expected)
    val_logits = predict(model, data["validation"], args, features["validation"])
    val_scores = scores(data["validation"]["labels"], val_logits.argmax(1))
    if not np.isclose(val_scores["macro_f1"], best_score, atol=1e-10):
        raise ValueError("Best checkpoint failed to reproduce validation score")
    atomic_save(best, folder / "best.pt")
    atomic_npz(folder / "validation_predictions.npz", logits=val_logits, labels=data["validation"]["labels"].numpy())
    result = {"model": spec["name"], "seed": seed, "spec": spec, "validation": val_scores,
              "best_epoch": best_epoch, "epochs_completed": len(history), "status": "completed",
              "uses_future_text": False, "uses_previous_label": False,
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "training_seconds": previous_seconds + time.perf_counter() - started,
              "test_evaluated": False, "run": args.fingerprint}
    common.dump(folder / "result.json", result)
    common.dump(folder / "history.json", history)
    seal_completion(folder, "completion.json", expected, TRAIN_ARTIFACTS)
    del model, optimizer, best
    if args.device == "cuda":
        torch.cuda.empty_cache()
    return result


def choose_confirmation(screen, specs, top_k):
    challengers = sorted([r for r in screen if r["spec"]["unfreeze"] != 0],
                         key=lambda r: (-r["validation"]["macro_f1"], r["model"]))[:top_k]
    names = {r["model"] for r in challengers} | {s["name"] for s in specs if s["unfreeze"] == 0}
    return [s for s in specs if s["name"] in names]


def run_screen(args, rows, assets):
    specs = plan(args.suite)
    results = [train_one(args, rows, assets, spec, 42) for spec in specs]
    selected = choose_confirmation(results, specs, args.top_k)
    record = {"run": args.fingerprint, "selection_basis": "seed42 validation_macro_f1 ONLY",
              "screen_results": results, "confirmation_specs": selected,
              "test_evaluated": False, "note": "controls retained even if not top-ranked"}
    path = args.out / "screen_selection.json"
    if path.exists() and common.read(path) != record:
        raise ValueError("Screen selection changed unexpectedly")
    common.dump(path, record)
    print("SCREEN complete; no test evaluation. Confirm: " + ", ".join(s["name"] for s in selected), flush=True)
    return selected


def validation_summary(results, specs):
    summary = []
    for spec in specs:
        group = [r for r in results if r["model"] == spec["name"]]
        values = [r["validation"]["macro_f1"] for r in group]
        summary.append({"model": spec["name"], "seeds": [r["seed"] for r in group],
                        "validation_macro_f1_mean": float(np.mean(values)),
                        "validation_macro_f1_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0})
    chosen = sorted(summary, key=lambda r: (-r["validation_macro_f1_mean"], r["model"]))[0]["model"]
    return {"selection_basis": "three-seed validation_macro_f1 mean ONLY", "selected_model": chosen, "summary": summary}


def evaluate_test(args, rows, assets, spec, seed):
    folder = folder_for(args, spec, seed)
    expected = identity(args, spec, seed)
    if not verify_completion(folder, "completion.json", expected, TRAIN_ARTIFACTS):
        raise ValueError("Refusing test evaluation before completed training")
    if verify_completion(folder, "test_completion.json", expected, TEST_ARTIFACTS):
        return common.read(folder / "test_result.json")
    asset = assets[spec["backbone"]]
    data = token_cache(args, asset, spec["backbone"], "test", rows["test"])
    features = frozen_cache(args, asset, spec, "test", data) if spec["unfreeze"] == 0 else None
    model = build_model(args, asset, spec, seed)
    restore_payload(model, torch.load(folder / "best.pt", map_location="cpu", weights_only=False), expected)
    logits = predict(model, data, args, features)
    pred = logits.argmax(1)
    common.save_evaluation(folder, data["labels"].numpy(), pred, rows["test"])
    result = {"model": spec["name"], "seed": seed, "test": scores(data["labels"], pred),
              "run": args.fingerprint, "selected_by": "validation_only", "test_evaluated": True}
    common.dump(folder / "test_result.json", result)
    seal_completion(folder, "test_completion.json", expected, TEST_ARTIFACTS)
    print(f"TEST {spec['name']} seed={seed} Macro-F1={result['test']['macro_f1']:.4%}", flush=True)
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()
    return result


def run_confirm(args, rows, assets):
    selected_path = args.out / "screen_selection.json"
    if not selected_path.exists():
        raise ValueError("Run screen first (or use stage all)")
    selection = common.read(selected_path)
    if selection["run"] != args.fingerprint:
        raise ValueError("Screen signature mismatch")
    # Recompute from verified training artifacts; never trust a manually edited shortlist.
    all_specs = plan(args.suite)
    screen = []
    for spec in all_specs:
        folder = folder_for(args, spec, 42)
        if not verify_completion(folder, "completion.json", identity(args, spec, 42), TRAIN_ARTIFACTS):
            raise ValueError("Incomplete screen")
        screen.append(common.read(folder / "result.json"))
    selected = choose_confirmation(screen, all_specs, args.top_k)
    if selected != selection["confirmation_specs"] or screen != selection["screen_results"]:
        raise ValueError("Screen selection was modified")
    results = [train_one(args, rows, assets, spec, seed) for spec in selected for seed in SEEDS]
    validation = {"run": args.fingerprint, **validation_summary(results, selected)}
    common.dump(args.out / "validation_selection.json", validation)
    # Test happens only after the confirmation set and three-seed selection are fixed.
    test = [evaluate_test(args, rows, assets, spec, seed) for spec in selected for seed in SEEDS]
    summary = []
    for item in validation["summary"]:
        group = [r for r in test if r["model"] == item["model"]]
        values = [r["test"]["macro_f1"] for r in group]
        summary.append({**item, "test_macro_f1_mean": float(np.mean(values)),
                        "test_macro_f1_std": float(np.std(values, ddof=1)),
                        "test_accuracy_mean": float(np.mean([r["test"]["accuracy"] for r in group])),
                        "test_per_class": {label: {k + "_mean": float(np.mean([r["test"]["per_class"][label][k] for r in group]))
                            for k in ["precision", "recall", "f1", "true_count", "pred_count", "tp_count"]} for label in LABELS}})
    common.dump(args.out / "results.json", {"training": results, "test": test})
    common.dump(args.out / "summary.json", summary)
    lines = ["可训练编码器强基线", "", "输入：最近6个话语轮次，止于当前系统回复；无未来文本/真实历史情绪输入。",
             "统一加权CE、分类头、有效批量和训练日程；冻结底层保持eval。未加入JEPA/蒸馏/重采样。",
             "新冻结对照重新编码，匹配本次精度与训练设置；不是原冻结实验的逐位复刻。",
             "只按验证集筛选；测试已用于历史诊断，因此本次是探索性后续实验。",
             f"验证集选定模型：{validation['selected_model']}", "",
             "模型 | 验证Macro-F1均值±标准差 | 测试Macro-F1均值±标准差 | 测试Accuracy"]
    for r in summary:
        lines.append(f"{r['model']} | {r['validation_macro_f1_mean']:.2%} ± {100*r['validation_macro_f1_std']:.2f}pp | "
                     f"{r['test_macro_f1_mean']:.2%} ± {100*r['test_macro_f1_std']:.2f}pp | {r['test_accuracy_mean']:.2%}")
    lines += ["", "标准差为三个种子的样本标准差，不代表显著性。逐类Precision/Recall/F1及预测数见summary.json。",
              "不同骨干的差异包含预训练/词表/容量差异；微调作用应优先看同一骨干冻结与解冻对照。"]
    (args.out / "encoder_baseline_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    common.dump(args.out / "run_status.json", {"run": args.fingerprint, "status": "completed",
                "training_runs": len(results), "screen_runs": len(screen), "selected_model": validation["selected_model"]})


@torch.no_grad()
def check_only(args, rows, assets):
    info = {"python": sys.executable, "device": args.device, "precision": args.precision,
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "samples": {s: len(r) for s, r in rows.items()},
            "dialogues_with_prediction_samples": {s: len({r['dialogue_id'] for r in records}) for s, records in rows.items()},
            "official_training_started": False}
    for backbone, asset in assets.items():
        spec = next(s for s in plan(args.suite) if s["backbone"] == backbone)
        tokens = token_cache(args, asset, backbone, "check", rows["train"][:16])
        model = build_model(args, asset, spec, 42).eval()
        with amp_context(args):
            z = pooled_embedding(model.encoder, token_batch(tokens, torch.arange(16), args.device))
        if not torch.isfinite(z).all() or not torch.allclose(z.norm(dim=1), torch.ones(16, device=args.device), atol=1e-5):
            raise ValueError("Encoder/mean pooling check failed")
        if backbone == "minilm":
            with np.load(args.source / "embeddings.npz", allow_pickle=False) as cache:
                reference = torch.from_numpy(cache["train_x"][:16]).to(args.device)
            cosine = F.cosine_similarity(reference, z).min().item()
            info["minilm_original_embedding_min_cosine"] = cosine
            if cosine < 0.999:
                raise ValueError("MiniLM tokenizer/pooling differs from original cached features")
        del model
    common.dump(args.out / "check.json", info)
    print(info, flush=True)
    print("CHECK PASS: source hashes, dialogue splits, cached model, temporal contract, pooling. No official training.", flush=True)


def benchmark(args, rows, assets):
    measurements = []
    for spec in plan(args.suite):
        if spec["unfreeze"] == 0:
            continue
        model = build_model(args, assets[spec["backbone"]], spec, 42).train()
        n = args.micro_batch * args.accumulation
        common.seed_all(42)
        data = {"input_ids": torch.randint(5, model.encoder.config.vocab_size, (n, args.max_tokens), dtype=torch.int32),
                "lengths": torch.full((n,), args.max_tokens, dtype=torch.int32), "labels": torch.arange(n) % 7}
        optimizer = optimizer_for(model, args)
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, 20)
        scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
        parts = list(torch.arange(n).split(args.micro_batch))
        timings = []
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for step in range(5):
            if args.device == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            update_window(model, data, parts, torch.ones(7, device=args.device), optimizer, scheduler, scaler, args)
            if args.device == "cuda":
                torch.cuda.synchronize()
            if step >= 2:
                timings.append(time.perf_counter() - started)
        seconds = float(np.median(timings))
        item = {"model": spec["name"], "synthetic_only": True, "full_length_tokens": args.max_tokens,
                "optimizer_step_seconds": seconds, "estimated_training_only_minutes_per_epoch":
                math.ceil(len(rows["train"]) / n) * seconds / 60,
                "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30 if args.device == "cuda" else None,
                "exclusions": "tokenization, validation, checkpoint IO; dynamic real lengths may run faster"}
        measurements.append(item)
        print(item, flush=True)
        del model, optimizer
        if args.device == "cuda":
            torch.cuda.empty_cache()
    common.dump(args.out / "benchmark.json", measurements)
    print("Synthetic throughput measurement only. No real experiment metrics/checkpoints were created.", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["check", "benchmark", "screen", "confirm", "all"], default="all")
    parser.add_argument("--suite", choices=["minilm", "roberta", "all"], default="minilm")
    parser.add_argument("--source", type=Path, default=ROOT / "runs" / "full")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "encoder_minilm_v1")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--micro-batch", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--eval-batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, choices=[256], default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args(argv)
    if min(args.micro_batch, args.accumulation, args.eval_batch, args.epochs, args.patience, args.threads, args.top_k) < 1:
        parser.error("Positive batch/epoch/patience/thread/top-k settings required")
    if min(args.encoder_lr, args.head_lr) <= 0 or args.weight_decay < 0 or not 0 <= args.warmup_ratio < 1:
        parser.error("Invalid learning rate/decay/warmup")
    args.source, args.out = args.source.resolve(), args.out.resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    runtime(args)
    rows, stats, hashes = load_source(args.source)
    assets = {b: resolve_assets(b, args.allow_download) for b in sorted({s["backbone"] for s in plan(args.suite)})}
    ensure_manifest(args, hashes, assets)
    common.dump(args.out / "data_stats.json", stats)
    if args.stage == "check":
        check_only(args, rows, assets)
    elif args.stage == "benchmark":
        benchmark(args, rows, assets)
    else:
        if args.stage in ["screen", "all"]:
            run_screen(args, rows, assets)
            common.dump(args.out / "run_status.json", {"run": args.fingerprint, "status": "screen_completed",
                                                       "test_evaluated": False})
        if args.stage in ["confirm", "all"]:
            run_confirm(args, rows, assets)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted. Completed epoch checkpoints retained; rerun EXACT same command to resume.", flush=True)
        sys.exit(130)
