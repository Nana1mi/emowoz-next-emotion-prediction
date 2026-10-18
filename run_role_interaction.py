"""Matched-capacity role pooling / FiLM ablation on trainable MiniLM.

All variants encode exactly the original history once (six utterances, 256
tokens, left truncation). Offset masks select latest user/system CONTENT tokens
from the same contextual hidden states. Global history still includes the latest
system reply: this is conditional prediction, NOT a causal/pre-action state model.
No future text, gold emotion, extra tokens, teacher, JEPA loss, or sampling change.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer

import run_controlled_experiments as common
import run_encoder_baselines as base
import run_emotion_distillation as reference_tools

ROOT = Path(__file__).resolve().parent
SEEDS = [42, 43, 44]
RAW_FILES = ["emowoz-multiwoz.json", "emowoz-dialmage.json"]


def plan():
    return [{"name": "global_additive", "view": "global_duplicate", "mixing": "additive", "unfreeze": -1},
            {"name": "role_additive", "view": "role_spans", "mixing": "additive", "unfreeze": -1},
            {"name": "role_film", "view": "role_spans", "mixing": "film", "unfreeze": -1}]


def raw_hashes(args, stats):
    result = {}
    for name in RAW_FILES:
        sha = common.digest(args.raw_dir / name)
        if sha != stats["source_sha256"][name]:
            raise ValueError(f"Raw source no longer matches original preparation: {name}")
        result[name] = sha
    return result


def ensure_manifest(args, hashes, raw_sha, asset, reference, references):
    for protected in [args.source, args.reference, args.raw_dir, ROOT / "data", ROOT / "runs" / "controlled_v2",
                      ROOT / "runs" / "posthoc_diagnostic", ROOT / "runs" / "error_analysis",
                      ROOT / "runs" / "emotion_kd_v1", ROOT / "任务梳理与实验结果_20261014"]:
        common.protect_paths(args.out, protected)
    value = {"schema_version": 1, "purpose": "matched capacity role pooling and bounded FiLM ablation",
             "code_sha256": common.digest(Path(__file__)), "base_sha256": common.digest(Path(base.__file__)),
             "helper_sha256": common.digest(Path(common.__file__)),
             "reference_loader_sha256": common.digest(Path(reference_tools.__file__)),
             "source": str(args.source), "source_sha256": hashes, "raw_dir": str(args.raw_dir), "raw_sha256": raw_sha,
             "asset": asset, "reference": str(args.reference),
             "reference_provenance_sha256": common.digest(args.reference / "provenance.json"),
             "reference_training": [{"seed": r["seed"], "completion_sha256": r["completion_sha256"]} for r in references],
             "training": reference["training"], "seeds": SEEDS, "plan": plan(),
             "input": {"text": "unchanged history", "history_utterances": 6, "max_tokens": 256,
                       "truncation_side": "left", "encoder_passes": 1, "extra_tokens": False,
                       "views": ["normalized global mean H", "normalized latest user content mean U",
                                 "normalized latest system content mean A"],
                       "span_source": "reconstruct raw turns strictly before target; verify exact cached history equality",
                       "empty_or_truncated_span": "zero vector; no fallback to future or earlier turn",
                       "future_text_input": False, "previous_label_input": False, "target_turn_feature": False},
             "head": {"hidden": 256, "state_projection": "GELU(Linear([H,U]/sqrt(2),256))",
                      "action_projection": "Linear(A,512) split g,d", "normalization": "LayerNorm256 then Dropout0.1",
                      "additive": "s + 0.5*tanh(g) + 0.5*tanh(d)",
                      "film": "s*(1+0.5*tanh(g)) + 0.5*tanh(d)",
                      "capacity_control": "global_additive replaces U and A with H; all three have identical trainable parameter counts and paired initialization"},
             "loss": "same train-weighted CE; no auxiliary loss, sampling, label input, or encoder change",
             "initialization": "same pretrained encoder; fresh paired head, NOT reference fine-tuned checkpoints",
             "selection": "all three settings get all three seeds; choose among CE reference and settings using validation mean only, then test",
             "limitations": ["contextual span tokens already attend to the system reply; not strictly isolated pre-action state",
                             "observational prediction does not identify causal response effects",
                             "test previously used for diagnosis; exploratory followup"],
             "environment": reference["environment"]}
    path = args.out / "provenance.json"
    if path.exists():
        if common.read(path) != value:
            raise ValueError("Code/config/assets/reference/source changed. Use a NEW output directory.")
    elif args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Nonempty output has no matching manifest")
    args.out.mkdir(parents=True, exist_ok=True)
    common.dump(path, value)
    args.fingerprint = common.fingerprint(value)


def reconstruct_spans(row, turns, history_turns=6):
    target = row["target_turn"]
    if target < 2 or target % 2 or target >= len(turns):
        raise ValueError("Invalid next-user target position")
    pieces = []
    spans = {}
    position = 0
    for index in range(max(0, target - history_turns), target):
        prefix = "User: " if index % 2 == 0 else "System: "
        text = turns[index]["text"].strip()
        if pieces:
            position += 1  # Original history joins utterances with a single newline.
        start = position + len(prefix)
        end = start + len(text)
        if index == target - 2:
            spans["user"] = [start, end]
            if text != row["last_user"]:
                raise ValueError("Current user differs from prepared data")
        if index == target - 1:
            spans["system"] = [start, end]
        pieces.append(prefix + text)
        position = end
    history = "\n".join(pieces)
    if history != row["history"] or set(spans) != {"user", "system"}:
        raise ValueError("Raw role boundaries do not reconstruct the EXACT original history")
    return spans


def span_boundaries(args, rows):
    folder = args.out / "cache"
    folder.mkdir(exist_ok=True)
    path, metadata = folder / "role_boundaries.npz", folder / "role_boundaries.meta.json"
    if metadata.exists():
        saved = common.read(metadata)
        if saved["run"] != args.fingerprint or saved["sha256"] != common.digest(path):
            raise ValueError("Role boundary cache mismatch")
        with np.load(path, allow_pickle=False) as cache:
            return {name: cache[name].copy() for name in cache.files}
    dialogues = {}
    for name in RAW_FILES:
        data = common.read(args.raw_dir / name)
        if dialogues.keys() & data.keys():
            raise ValueError("Duplicate raw dialogue identifiers")
        dialogues.update(data)
    arrays = {}
    for split, records in rows.items():
        user, system = [], []
        for row in records:
            spans = reconstruct_spans(row, dialogues[row["dialogue_id"]]["log"])
            user.append(spans["user"])
            system.append(spans["system"])
        arrays[split + "_user"] = np.asarray(user, dtype=np.int32)
        arrays[split + "_system"] = np.asarray(system, dtype=np.int32)
    del dialogues, data
    base.atomic_npz(path, **arrays)
    common.dump(metadata, {"run": args.fingerprint, "sha256": common.digest(path),
                          "note": "character offsets only; no labels, future text, or additional input content"})
    print("Verified raw-turn alignment for ALL prepared history strings; saved role boundaries", flush=True)
    return arrays


def offset_masks(offsets, spans, attention):
    spans = torch.as_tensor(spans, dtype=torch.long)
    start, end = spans[:, 0, None], spans[:, 1, None]
    return (offsets[:, :, 1] > start) & (offsets[:, :, 0] < end) & (end > start) & attention.bool()


def token_cache(args, asset, split, rows, user_bounds, system_bounds):
    folder = args.out / "cache"
    folder.mkdir(exist_ok=True)
    path = folder / f"minilm_{split}_role_tokens.pt"
    metadata = path.with_suffix(".meta.json")
    if metadata.exists():
        saved = common.read(metadata)
        if saved["run"] != args.fingerprint or saved["sha256"] != common.digest(path):
            raise ValueError("Role token cache mismatch")
        return torch.load(path, map_location="cpu", weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(asset["path"], local_files_only=True, trust_remote_code=False, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("Fast tokenizer required for original-text offsets")
    tokenizer.truncation_side, tokenizer.padding_side = "left", "right"
    ids, lengths, user_masks, system_masks = [], [], [], []
    texts = base.history_texts(rows)
    for start in range(0, len(rows), 512):
        stop = start + 512
        encoded = tokenizer(texts[start:stop], padding="max_length", truncation=True, max_length=args.max_tokens,
                            return_offsets_mapping=True, return_tensors="pt")
        mask = encoded["attention_mask"]
        offsets = encoded["offset_mapping"]
        ids.append(encoded["input_ids"].to(torch.int32))
        lengths.append(mask.sum(1).to(torch.int32))
        user_masks.append(offset_masks(offsets, user_bounds[start:stop], mask))
        system_masks.append(offset_masks(offsets, system_bounds[start:stop], mask))
    data = {"input_ids": torch.cat(ids), "lengths": torch.cat(lengths), "user_mask": torch.cat(user_masks),
            "system_mask": torch.cat(system_masks), "labels": torch.tensor([r["label"] for r in rows])}
    # Independent equality check against the original, read-only baseline token cache.
    # Token budgets/content must NOT silently grow when role views are introduced.
    reference_path = args.reference / "cache" / f"minilm_{split}_tokens.pt"
    if reference_path.exists():
        record = common.read(reference_path.with_suffix(".meta.json"))
        if record["sha256"] != common.digest(reference_path):
            raise ValueError("Reference token cache was modified")
        original = torch.load(reference_path, map_location="cpu", weights_only=False)
        for name in ["input_ids", "lengths", "labels"]:
            if not torch.equal(original[name], data[name]):
                raise ValueError(f"Role experiment changed original {split} token inputs/labels")
    base.atomic_save(data, path)
    statistics = {"samples": len(rows), "missing_user_span": int((~data["user_mask"].any(1)).sum()),
                  "missing_system_span": int((~data["system_mask"].any(1)).sum())}
    common.dump(metadata, {"run": args.fingerprint, "sha256": common.digest(path), **statistics,
                           "missing_note": "empty content OR fully truncated; pooled vector is zero"})
    print(f"Role tokenization {split}: {statistics}", flush=True)
    return data


def batch_input(data, indices, device):
    batch = base.token_batch(data, indices, device)
    width = batch["input_ids"].shape[1]
    return {**batch, "user_mask": data["user_mask"][indices, :width].to(device),
            "system_mask": data["system_mask"][indices, :width].to(device)}


def masked_pool(hidden, mask):
    mask = mask.float().unsqueeze(-1)
    mean = (hidden.float() * mask).sum(1) / mask.sum(1).clamp_min(1)
    return F.normalize(mean, dim=-1)  # Empty span stays exactly zero.


class InteractionHead(nn.Module):
    def __init__(self, dimension, mixing):
        super().__init__()
        if mixing not in ["additive", "film"]:
            raise ValueError("Unknown interaction")
        self.mixing = mixing
        self.state = nn.Linear(2 * dimension, 256)
        self.action = nn.Linear(dimension, 512)
        self.norm = nn.LayerNorm(256)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(256, 7)

    def forward(self, history, user, action):
        state = F.gelu(self.state(torch.cat([history, user], dim=-1) / math.sqrt(2)))
        gamma, delta = self.action(action).chunk(2, dim=-1)
        gamma, delta = 0.5 * torch.tanh(gamma), 0.5 * torch.tanh(delta)
        hidden = state * (1 + gamma) + delta if self.mixing == "film" else state + gamma + delta
        return self.classifier(self.dropout(self.norm(hidden)))


class RolePredictor(nn.Module):
    def __init__(self, encoder, spec, seed, checkpointing=False):
        super().__init__()
        self.encoder, self.spec = encoder, spec
        for parameter in encoder.parameters():
            parameter.requires_grad_(True)
        if checkpointing:
            encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        common.seed_all(seed)
        self.head = InteractionHead(encoder.config.hidden_size, spec["mixing"])

    def forward(self, batch):
        if set(batch) != {"input_ids", "attention_mask", "user_mask", "system_mask"}:
            raise ValueError("Only original token IDs and history role masks are accepted")
        hidden = self.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], return_dict=True).last_hidden_state.float()
        history = masked_pool(hidden, batch["attention_mask"])
        if self.spec["view"] == "global_duplicate":
            user = action = history
        elif self.spec["view"] == "role_spans":
            user, action = masked_pool(hidden, batch["user_mask"]), masked_pool(hidden, batch["system_mask"])
        else:
            raise ValueError("Unknown input view")
        return self.head(history, user, action)


def build_model(args, asset, spec, seed):
    common.seed_all(seed)
    encoder = AutoModel.from_pretrained(asset["path"], local_files_only=True, trust_remote_code=False, add_pooling_layer=False)
    return RolePredictor(encoder, spec, seed, args.gradient_checkpointing).to(args.device)


def update_window(model, data, parts, weights, optimizer, scheduler, scaler, args):
    optimizer.zero_grad(set_to_none=True)
    denominator = sum(float(weights[data["labels"][idx].to(args.device)].sum()) for idx in parts)
    total = 0.0
    for indices in parts:
        labels = data["labels"][indices].to(args.device)
        with base.amp_context(args):
            logits = model(batch_input(data, indices, args.device))
            numerator = base.ce_numerator(logits, labels, weights)
            loss = numerator / denominator
        if not torch.isfinite(loss):
            raise ValueError("Non-finite role-model loss")
        scaler.scale(loss).backward()
        total += float(numerator.detach())
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    if not torch.isfinite(norm) and not scaler.is_enabled():
        raise ValueError("Non-finite gradient")
    scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    if scaler.get_scale() >= scale:
        scheduler.step()
    return total, denominator


@torch.no_grad()
def predict(model, data, args):
    model.eval()
    logits = []
    printed = time.perf_counter()
    for start in range(0, len(data["labels"]), args.eval_batch):
        indices = torch.arange(start, min(start + args.eval_batch, len(data["labels"])))
        with base.amp_context(args):
            output = model(batch_input(data, indices, args.device)).float()
        if not torch.isfinite(output).all():
            raise ValueError("Non-finite role-model logits")
        logits.append(output.cpu().numpy())
        if time.perf_counter() - printed > 20:
            print(f"Role evaluation {start}/{len(data['labels'])}", flush=True)
            printed = time.perf_counter()
    return np.concatenate(logits)


def split_data(args, rows, bounds, asset, splits):
    return {s: token_cache(args, asset, s, rows[s], bounds[s + "_user"], bounds[s + "_system"]) for s in splits}


def train_one(args, rows, bounds, asset, spec, seed):
    folder = base.folder_for(args, spec, seed)
    identity = base.identity(args, spec, seed)
    if base.verify_completion(folder, "completion.json", identity, base.TRAIN_ARTIFACTS):
        print(f"Verified completed role model {spec['name']} seed={seed}", flush=True)
        return common.read(folder / "result.json")
    config = folder / "experiment_config.json"
    if config.exists() and common.read(config) != identity:
        raise ValueError("Partial role-model configuration mismatch")
    if folder.exists() and any(folder.iterdir()) and not config.exists():
        raise ValueError("Untracked partial model directory")
    folder.mkdir(parents=True, exist_ok=True)
    common.dump(config, identity)
    data = split_data(args, rows, bounds, asset, ["train", "validation"])
    model = build_model(args, asset, spec, seed)
    optimizer = base.optimizer_for(model, args)
    steps = math.ceil(len(rows["train"]) / (args.micro_batch * args.accumulation))
    scheduler = base.get_linear_schedule_with_warmup(optimizer, math.ceil(args.epochs * steps * args.warmup_ratio), args.epochs * steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
    counts = np.bincount(data["train"]["labels"].numpy(), minlength=7)
    weights = torch.from_numpy(common.compute_class_weights(counts, counts.sum(), "baseline_sqrt_clip5")).to(args.device)
    best, best_score, best_epoch, history, first, previous_seconds = None, -1.0, 0, [], 1, 0.0
    last = folder / "last.pt"
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=False)
        base.restore_payload(model, saved["current"], identity)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        best, best_score, best_epoch, history = saved["best"], saved["best_score"], saved["best_epoch"], saved["history"]
        first, previous_seconds = saved["epoch"] + 1, saved["training_seconds"]
        del saved
        print(f"Resume {spec['name']} seed={seed} from completed epoch {first-1}", flush=True)
    print(f"TRAIN {spec['name']} seed={seed}; head params={sum(p.numel() for p in model.head.parameters()):,}; one encoder pass", flush=True)
    started = time.perf_counter()
    for epoch in range(first, args.epochs + 1):
        if best_epoch and epoch - 1 - best_epoch >= args.patience:
            break
        common.seed_all(seed + 1000 * epoch)
        model.train()
        order = torch.randperm(len(rows["train"]), generator=torch.Generator().manual_seed(seed + epoch))
        numerator_sum = denominator_sum = 0.0
        epoch_start = last_print = time.perf_counter()
        for step, parts in enumerate(base.training_windows(order, args.micro_batch, args.accumulation), 1):
            numerator, denominator = update_window(model, data["train"], parts, weights, optimizer, scheduler, scaler, args)
            numerator_sum += numerator
            denominator_sum += denominator
            now = time.perf_counter()
            if now - last_print > 20 or step == steps:
                print(f"{spec['name']} seed={seed} epoch={epoch}/{args.epochs} step={step}/{steps}; "
                      f"loss={numerator_sum/denominator_sum:.4f}; epoch ETA={(now-epoch_start)/step*(steps-step)/60:.1f} min", flush=True)
                last_print = now
        logits = predict(model, data["validation"], args)
        val = base.scores(data["validation"]["labels"], logits.argmax(1))
        history.append({"epoch": epoch, "loss": numerator_sum / denominator_sum, "validation": val,
                        "seconds": time.perf_counter() - epoch_start})
        if val["macro_f1"] > best_score:
            best_score, best_epoch = val["macro_f1"], epoch
            best = base.checkpoint_payload(model, spec, asset, identity)
        base.atomic_save({"identity": identity, "epoch": epoch, "current": base.checkpoint_payload(model, spec, asset, identity),
                          "best": best, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                          "scaler": scaler.state_dict(), "history": history, "best_score": best_score,
                          "best_epoch": best_epoch, "training_seconds": previous_seconds + time.perf_counter() - started}, last)
        common.dump(folder / "history.json", history)
        print(f"VALIDATION {spec['name']} seed={seed} epoch={epoch} Macro-F1={val['macro_f1']:.4%}", flush=True)
        if epoch - best_epoch >= args.patience:
            break
    if best is None:
        raise ValueError("No finite validation checkpoint")
    base.restore_payload(model, best, identity)
    logits = predict(model, data["validation"], args)
    validation = base.scores(data["validation"]["labels"], logits.argmax(1))
    if not np.isclose(validation["macro_f1"], best_score, atol=1e-10):
        raise ValueError("Role checkpoint failed validation reproduction")
    base.atomic_save(best, folder / "best.pt")
    base.atomic_npz(folder / "validation_predictions.npz", logits=logits, labels=data["validation"]["labels"].numpy())
    result = {"model": spec["name"], "seed": seed, "spec": spec, "validation": validation, "best_epoch": best_epoch,
              "epochs_completed": len(history), "status": "completed", "uses_future_text": False, "uses_previous_label": False,
              "head_parameter_count": sum(p.numel() for p in model.head.parameters()),
              "trained_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "test_evaluated": False, "run": args.fingerprint, "training_seconds": previous_seconds + time.perf_counter() - started}
    common.dump(folder / "result.json", result)
    common.dump(folder / "history.json", history)
    base.seal_completion(folder, "completion.json", identity, base.TRAIN_ARTIFACTS)
    del model, optimizer, best
    if args.device == "cuda":
        torch.cuda.empty_cache()
    return result


def evaluate_test(args, rows, bounds, asset, spec, seed):
    folder = base.folder_for(args, spec, seed)
    identity = base.identity(args, spec, seed)
    if not base.verify_completion(folder, "completion.json", identity, base.TRAIN_ARTIFACTS):
        raise ValueError("Test requires completed training")
    if base.verify_completion(folder, "test_completion.json", identity, base.TEST_ARTIFACTS):
        return common.read(folder / "test_result.json")
    data = split_data(args, rows, bounds, asset, ["test"])["test"]
    model = build_model(args, asset, spec, seed)
    base.restore_payload(model, torch.load(folder / "best.pt", map_location="cpu", weights_only=False), identity)
    logits = predict(model, data, args)
    prediction = logits.argmax(1)
    common.save_evaluation(folder, data["labels"].numpy(), prediction, rows["test"])
    result = {"model": spec["name"], "seed": seed, "test": base.scores(data["labels"], prediction),
              "run": args.fingerprint, "selected_by": "validation_only", "test_evaluated": True}
    common.dump(folder / "test_result.json", result)
    base.seal_completion(folder, "test_completion.json", identity, base.TEST_ARTIFACTS)
    print(f"TEST {spec['name']} seed={seed} Macro-F1={result['test']['macro_f1']:.4%}", flush=True)
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()
    return result


def screen(args, rows, bounds, asset):
    results = [train_one(args, rows, bounds, asset, spec, 42) for spec in plan()]
    value = {"run": args.fingerprint, "results": results, "test_evaluated": False,
             "note": "all three predeclared variants receive full three-seed confirmation; screen is an inspection point, not pruning"}
    path = args.out / "screen_results.json"
    if path.exists() and common.read(path) != value:
        raise ValueError("Completed screen changed")
    common.dump(path, value)
    print("SCREEN complete: all three seed42 variants; no test evaluation", flush=True)


def confirm(args, rows, bounds, asset, references):
    path = args.out / "screen_results.json"
    if not path.exists() or common.read(path)["run"] != args.fingerprint:
        raise ValueError("Run screen first, or use stage all")
    results = [train_one(args, rows, bounds, asset, spec, seed) for spec in plan() for seed in SEEDS]
    groups = {"minilm_full_ce_reference": [r["result"] for r in references]}
    groups.update({spec["name"]: [r for r in results if r["model"] == spec["name"]] for spec in plan()})
    means = {name: float(np.mean([r["validation"]["macro_f1"] for r in group])) for name, group in groups.items()}
    # Reference wins exact ties; otherwise follow predeclared validation-only criterion.
    selected = max(means, key=means.get)
    common.dump(args.out / "validation_selection.json", {"run": args.fingerprint, "selection_basis": "three-seed validation mean ONLY",
                "selected_model": selected, "validation_means": means})
    tests = [evaluate_test(args, rows, bounds, asset, spec, seed) for spec in plan() for seed in SEEDS]
    reference_tests = []
    for reference in references:
        folder = Path(reference["folder"])
        if not base.verify_completion(folder, "test_completion.json", reference["identity"], base.TEST_ARTIFACTS):
            raise ValueError("Reference test incomplete")
        reference_tests.append(common.read(folder / "test_result.json"))
    test_groups = {"minilm_full_ce_reference": reference_tests}
    test_groups.update({spec["name"]: [r for r in tests if r["model"] == spec["name"]] for spec in plan()})
    summary = []
    for name, group in groups.items():
        test = test_groups[name]
        val_values, test_values = [[r[key]["macro_f1"] for r in records] for key, records in [("validation", group), ("test", test)]]
        summary.append({"model": name, "validation_macro_f1_mean": float(np.mean(val_values)),
                        "validation_macro_f1_std": float(np.std(val_values, ddof=1)),
                        "test_macro_f1_mean": float(np.mean(test_values)), "test_macro_f1_std": float(np.std(test_values, ddof=1)),
                        "test_accuracy_mean": float(np.mean([r["test"]["accuracy"] for r in test])),
                        "test_negative_f1_mean": float(np.mean([r["test"]["negative"]["f1"] for r in test])),
                        "test_per_class": {label: {key + "_mean": float(np.mean([r["test"]["per_class"][label][key] for r in test]))
                            for key in ["precision", "recall", "f1", "true_count", "pred_count", "tp_count"]} for label in common.LABELS}})
    paired = []
    for a, b, meaning in [("global_additive", "minilm_full_ce_reference", "capacity/architecture control"),
                           ("role_additive", "global_additive", "role-specific pooling at matched head capacity"),
                           ("role_film", "role_additive", "multiplicative modulation at matched head capacity")]:
        differences = [{"seed": seed, "test_macro_f1_delta": next(r for r in test_groups[a] if r["seed"] == seed)["test"]["macro_f1"] -
                        next(r for r in test_groups[b] if r["seed"] == seed)["test"]["macro_f1"]} for seed in SEEDS]
        paired.append({"a": a, "b": b, "meaning": meaning, "paired_differences": differences,
                       "mean_delta": float(np.mean([d["test_macro_f1_delta"] for d in differences]))})
    common.dump(args.out / "results.json", {"reference_training": groups["minilm_full_ce_reference"], "training": results,
                "reference_test": reference_tests, "test": tests})
    common.dump(args.out / "summary.json", summary)
    common.dump(args.out / "paired_comparisons.json", paired)
    lines = ["角色分池化与FiLM交互消融", "", "原CE全量微调三个种子只读复用；三个新设置各跑42/43/44，共9次新训练。",
             "同一历史字符串、最近6个话语、左截断256 tokens、同一次MiniLM编码；没有增加额外文本或重复编码。",
             "global_additive：全局向量H重复到三个槽位，控制更大分类头的容量/结构影响。",
             "role_additive：H、当前用户U、最新系统A分别池化，使用加性融合。",
             "role_film：相同参数量、同种子相同初始权重，仅将加性项改为动作条件乘法调制。",
             "所有新设置分类头参数完全相同，编码器全量微调；无蒸馏、JEPA辅助损失、重采样或真实历史标签输入。",
             f"按验证集均值选定模型：{selected}", "", "模型 | 验证Macro-F1 | 测试Macro-F1 | 测试Accuracy | 负面F1"]
    for r in summary:
        lines.append(f"{r['model']} | {r['validation_macro_f1_mean']:.2%} ± {100*r['validation_macro_f1_std']:.2f}pp | "
                     f"{r['test_macro_f1_mean']:.2%} ± {100*r['test_macro_f1_std']:.2f}pp | {r['test_accuracy_mean']:.2%} | {r['test_negative_f1_mean']:.2%}")
    lines += ["", "配对测试差值（描述，不作显著性断言）："]
    for r in paired:
        lines.append(f"{r['a']} - {r['b']}: {100*r['mean_delta']:+.3f}pp；" +
                     "; ".join(f"seed{d['seed']} {100*d['test_macro_f1_delta']:+.3f}pp" for d in r["paired_differences"]))
    lines += ["", "表示来自双向上下文化编码，H本身包含最新系统回复，U也可能注意到该回复；不是严格分离的动作前心理状态。",
              "因此，本实验是角色条件预测消融，不证明客服行为的因果效果，也不进行反事实策略优劣断言。",
              "测试集曾用于错例诊断，本轮仍为探索性改进；不以测试集选模型。标准差不是显著性检验。",
              "不预设提升；完整逐类精确率、召回率、F1与预测/命中数量见summary.json。"]
    (args.out / "role_interaction_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    common.dump(args.out / "run_status.json", {"run": args.fingerprint, "status": "completed", "new_training_runs": 9,
                "reference_runs_reused": 3, "selected_model": selected})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["check", "screen", "confirm", "all"], default="all")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--reference", type=Path, default=ROOT / "runs" / "encoder_minilm_v1")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "role_interaction_v1")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    args = parser.parse_args(argv)
    for name in ["source", "reference", "out", "raw_dir"]:
        value = getattr(args, name)
        setattr(args, name, value.resolve() if value else None)
    return args


def main(argv=None):
    args = parse_args(argv)
    rows, stats, hashes, asset, reference, references = reference_tools.load_reference(args)
    raw_sha = raw_hashes(args, stats)
    ensure_manifest(args, hashes, raw_sha, asset, reference, references)
    common.dump(args.out / "data_stats.json", stats)
    bounds = span_boundaries(args, rows)
    if args.stage == "check":
        # CPU tokenization only; no encoder forward or real optimization.
        data = split_data(args, rows, bounds, asset, ["train", "validation"])
        common.dump(args.out / "check.json", {"run": args.fingerprint, "status": "passed", "official_training_started": False,
                    "samples": {s: len(v["labels"]) for s, v in data.items()}, "encoder_passes": 1,
                    "note": "raw alignment and same-token-budget checks; no test tokenization"})
        print("CHECK PASS: raw history reconstruction, latest user/system offsets, unchanged original tokens; no training/GPU workload", flush=True)
        return
    if args.stage in ["screen", "all"]:
        screen(args, rows, bounds, asset)
        common.dump(args.out / "run_status.json", {"run": args.fingerprint, "status": "screen_completed", "test_evaluated": False})
    if args.stage in ["confirm", "all"]:
        confirm(args, rows, bounds, asset, references)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted. Completed epochs retained; rerun identical command to resume.", flush=True)
        sys.exit(130)
