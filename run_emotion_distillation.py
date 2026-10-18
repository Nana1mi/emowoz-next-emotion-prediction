"""OOF privileged-emotion distillation on the completed trainable MiniLM baseline.

Teacher sees cached future utterance embeddings, TRAIN ONLY. Outer folds and
inner early-stopping splits are dialogue-disjoint and entirely inside train.
Student sees history only, starts from the same pretrained weights as the CE
reference, and adds weighted T^2 KL. No hidden-state alignment, resampling,
architecture change, validation/test teacher targets, or pretrained downloads.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold

import run_controlled_experiments as common
import run_encoder_baselines as base

ROOT = Path(__file__).resolve().parent
SEEDS = [42, 43, 44]
TEACHER_ARTIFACTS = ["best.pt", "history.json", "heldout_logits.npz", "result.json"]
OOF_ARTIFACTS = ["oof_logits.npz", "oof_diagnostics.json"]


def plan(betas):
    return [{"name": "minilm_full_kd_" + format(beta, ".6g").replace(".", "p"),
             "backbone": "minilm", "unfreeze": -1, "beta": beta} for beta in betas]


def load_reference(args):
    provenance = common.read(args.reference / "provenance.json")
    if provenance["code_sha256"] != common.digest(Path(base.__file__)) or provenance["helper_sha256"] != common.digest(Path(common.__file__)):
        raise ValueError("Baseline/helper code changed since the completed reference; recover it before reuse")
    environment = {"python": sys.version, **{p: importlib.metadata.version(p) for p in
                   ["torch", "transformers", "huggingface-hub", "numpy", "scikit-learn"]}}
    if environment != provenance["environment"]:
        raise ValueError("Runtime versions differ from completed CE reference; restore kg_project versions before reuse")
    if args.source is None:
        args.source = Path(provenance["source"]).resolve()
    rows, stats, hashes = base.load_source(args.source)
    if hashes != provenance["source_sha256"]:
        raise ValueError("Source differs from completed encoder baseline")
    for key, value in provenance["training"].items():
        setattr(args, key, value)
    base.runtime(args)
    if args.precision != provenance["training"]["precision"]:
        raise ValueError("Student precision differs from completed reference")
    asset = provenance["models"]["minilm"]
    for name, sha in asset["sha256"].items():
        if common.digest(Path(asset["path"]) / name) != sha:
            raise ValueError("Cached pretrained MiniLM changed; no download will be attempted")
    reference_fp = common.fingerprint(provenance)
    spec = {"name": "minilm_full", "backbone": "minilm", "unfreeze": -1}
    checker = SimpleNamespace(fingerprint=reference_fp)
    records = []
    for seed in SEEDS:
        folder = args.reference / "models" / f"minilm_full_seed{seed}"
        identity = base.identity(checker, spec, seed)
        if not base.verify_completion(folder, "completion.json", identity, base.TRAIN_ARTIFACTS):
            raise ValueError(f"Incomplete full-finetuning reference: seed {seed}")
        # Only the training/validation record is read during screening.
        record = common.read(folder / "result.json")
        if record["uses_future_text"] or record["uses_previous_label"]:
            raise ValueError("Reference is not a deployable history-only predictor")
        records.append({"seed": seed, "result": record, "folder": str(folder), "identity": identity,
                        "completion_sha256": common.digest(folder / "completion.json")})
    return rows, stats, hashes, asset, provenance, records


def make_folds(rows, folds=3, seed=2026):
    y = np.array([r["label"] for r in rows], dtype=np.int64)
    groups = np.array([r["dialogue_id"] for r in rows])
    outer = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    assignments = np.full(len(rows), -1, dtype=np.int64)
    plans = []
    for k, (fitting, heldout) in enumerate(outer.split(np.zeros(len(y)), y, groups)):
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 100 + k)
        inner_fit, inner_stop = next(inner.split(np.zeros(len(fitting)), y[fitting], groups[fitting]))
        fit, stop = fitting[inner_fit], fitting[inner_stop]
        gfit, gstop, gheld = [set(groups[idx]) for idx in [fit, stop, heldout]]
        if gfit & gstop or gfit & gheld or gstop & gheld:
            raise ValueError("Teacher dialogue leakage")
        if (assignments[heldout] != -1).any() or not (np.bincount(y[fit], minlength=7) > 0).all():
            raise ValueError("Invalid teacher fold coverage or missing fitting class")
        assignments[heldout] = k
        plans.append({"fold": k, "fit": fit.tolist(), "stop": stop.tolist(), "heldout": heldout.tolist(),
                      "counts": {name: np.bincount(y[idx], minlength=7).tolist() for name, idx in
                                 [("fit", fit), ("stop", stop), ("heldout", heldout)]},
                      "dialogues": {"fit": len(gfit), "stop": len(gstop), "heldout": len(gheld)}})
    if (assignments < 0).any():
        raise ValueError("Some training samples have no held-out teacher")
    return {"seed": seed, "folds": folds, "train_samples": len(rows), "plans": plans,
            "assignment": assignments.tolist(), "inner_selection": "first fold of five-fold SGKF inside outer fitting partition"}


def ensure_manifest(args, hashes, asset, reference, records, folds):
    for protected in [args.source, args.reference, ROOT / "runs" / "controlled_v2", ROOT / "runs" / "posthoc_diagnostic",
                      ROOT / "runs" / "error_analysis", ROOT / "data", ROOT / "任务梳理与实验结果_20261014"]:
        common.protect_paths(args.out, protected)
    value = {"schema_version": 1, "code_sha256": common.digest(Path(__file__)),
             "base_sha256": common.digest(Path(base.__file__)), "helper_sha256": common.digest(Path(common.__file__)),
             "source": str(args.source), "source_sha256": hashes, "asset": asset,
             "reference": str(args.reference), "reference_provenance_sha256": common.digest(args.reference / "provenance.json"),
             "reference_training": [{"seed": r["seed"], "completion_sha256": r["completion_sha256"]} for r in records],
             "student_training": reference["training"], "seeds": SEEDS, "plan": plan(args.betas),
             "teacher": {"input": "train_target (frozen MiniLM future-utterance embedding), no validation/test future",
                         "architecture": "384 -> 256 GELU LayerNorm dropout0.1 -> 7", "device": args.teacher_device,
                         "epochs": args.teacher_epochs, "patience": args.teacher_patience, "batch": args.teacher_batch,
                         "lr": args.teacher_lr, "weight_decay": 1e-3, "weights": "sqrt clip5 from inner fitting labels ONLY",
                         "outer_folds": 3, "fold_plan_sha256": common.fingerprint(folds), "seed": 2026,
                         "checkpoint_selection": "inner train-only stop partition Macro-F1", "precision": "fp32"},
             "student_input": "history only; six utterances; left truncation 256; no teacher/labels/metadata in forward",
             "initialization": "original pretrained MiniLM and paired random head; NOT a warmed-up reference checkpoint",
             "loss": {"formula": "weighted_CE + beta * T^2 * weighted_KL(teacher || student)",
                      "temperature": args.temperature, "denominator": "sum of train class weights in full optimizer window",
                      "teacher_gradient": False, "latent_loss": False},
             "selection": "seed42 validation selects best KD challenger; three-seed validation mean selects CE or KD; test only afterwards",
             "limitations": ["teacher information is training-only and may not be predictable from history",
                             "CE versus KD alone does not isolate instance-level future knowledge from generic regularization",
                             "test previously used for diagnosis; exploratory followup, not untouched evaluation"],
             "environment": reference["environment"]}
    path = args.out / "provenance.json"
    if path.exists():
        if common.read(path) != value:
            raise ValueError("Code/config/reference/source changed; use a NEW --out directory")
    elif args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Nonempty output has no matching manifest")
    args.out.mkdir(parents=True, exist_ok=True)
    common.dump(path, value)
    args.fingerprint = common.fingerprint(value)
    common.dump(args.out / "teacher_folds.json", folds)
    return value


@torch.no_grad()
def teacher_predict(model, x, indices, device, batch=1024):
    model.eval()
    result = []
    for offset in range(0, len(indices), batch):
        idx = indices[offset:offset + batch]
        result.append(model(x[idx].to(device)).cpu().numpy())
    logits = np.concatenate(result)
    if not np.isfinite(logits).all():
        raise ValueError("Non-finite teacher logits")
    return logits


def train_teacher_fold(args, x, y, fold):
    k = fold["fold"]
    folder = args.out / "teachers" / f"fold{k}"
    expected = {"run": args.fingerprint, "fold": k, "teacher_seed": 2026 + k}
    if base.verify_completion(folder, "completion.json", expected, TEACHER_ARTIFACTS):
        print(f"Verified completed OOF teacher fold {k}", flush=True)
        with np.load(folder / "heldout_logits.npz", allow_pickle=False) as data:
            if not np.array_equal(data["indices"], fold["heldout"]):
                raise ValueError("Teacher held-out row order differs")
            return data["logits"].copy()
    folder.mkdir(parents=True, exist_ok=True)
    config = folder / "config.json"
    if config.exists() and common.read(config) != expected:
        raise ValueError("Teacher identity mismatch")
    common.dump(config, expected)
    device = args.teacher_device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Teacher CUDA unavailable")
    common.seed_all(2026 + k)
    model = base.Head(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.teacher_lr, weight_decay=1e-3)
    fit, stop, heldout = [torch.tensor(fold[name], dtype=torch.long) for name in ["fit", "stop", "heldout"]]
    counts = np.bincount(y[fit].numpy(), minlength=7)
    weights = torch.from_numpy(common.compute_class_weights(counts, len(fit), "baseline_sqrt_clip5")).to(device)
    best, best_score, best_epoch, history, start = None, -1.0, 0, [], 1
    if (folder / "last.pt").exists():
        saved = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
        if saved["identity"] != expected:
            raise ValueError("Teacher resume identity mismatch")
        model.load_state_dict(saved["state"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        best, best_score, best_epoch, history, start = saved["best"], saved["best_score"], saved["best_epoch"], saved["history"], saved["epoch"] + 1
        print(f"Teacher fold {k}: resume completed epoch {start-1}", flush=True)
    for epoch in range(start, args.teacher_epochs + 1):
        if best_epoch and epoch - 1 - best_epoch >= args.teacher_patience:
            break
        common.seed_all(2026 + k + 1000 * epoch)
        model.train()
        order = fit[torch.randperm(len(fit), generator=torch.Generator().manual_seed(2026 + k + epoch))]
        loss_sum = count = 0.0
        for indices in order.split(args.teacher_batch):
            optimizer.zero_grad(set_to_none=True)
            target = y[indices].to(device)
            loss = F.cross_entropy(model(x[indices].to(device)), target, weight=weights)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite teacher loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
            count += len(indices)
        logits = teacher_predict(model, x, stop, device)
        score = common.metrics(y[stop].numpy(), logits.argmax(1))["macro_f1"]
        history.append({"epoch": epoch, "loss": loss_sum / count, "inner_stop_macro_f1": score})
        if score > best_score:
            best_score, best_epoch = score, epoch
            best = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
        base.atomic_save({"identity": expected, "state": {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()},
                          "optimizer": optimizer.state_dict(), "best": best, "best_score": best_score,
                          "best_epoch": best_epoch, "history": history, "epoch": epoch}, folder / "last.pt")
        print(f"Teacher fold={k} epoch={epoch} inner-stop Macro-F1={score:.4%}", flush=True)
        if epoch - best_epoch >= args.teacher_patience:
            break
    model.load_state_dict(best, strict=True)
    logits = teacher_predict(model, x, heldout, device)
    base.atomic_save({"identity": expected, "state_dict": best, "labels": common.LABELS,
                      "input_field": "future_utterance_embedding", "diagnostic_only": True,
                      "uses_future_text": True, "dimension": x.shape[1]}, folder / "best.pt")
    base.atomic_npz(folder / "heldout_logits.npz", indices=heldout.numpy(), logits=logits)
    common.dump(folder / "history.json", history)
    common.dump(folder / "result.json", {"fold": k, "best_epoch": best_epoch, "inner_stop_macro_f1": best_score,
                "oof": base.scores(y[heldout].numpy(), logits.argmax(1)), "selection_basis": "inner_stop_only",
                "uses_future_text": True, "diagnostic_only": True, "partitions": fold["counts"]})
    base.seal_completion(folder, "completion.json", expected, TEACHER_ARTIFACTS)
    del model, optimizer, best
    if device == "cuda":
        torch.cuda.empty_cache()
    return logits


def prepare_teacher(args, rows, folds):
    expected = {"run": args.fingerprint, "folds": common.fingerprint(folds)}
    folder = args.out / "teachers"
    if base.verify_completion(folder, "oof_completion.json", expected, OOF_ARTIFACTS):
        # Also verify individual folds; do not silently accept damaged teacher artifacts.
        for k in range(3):
            identity = {"run": args.fingerprint, "fold": k, "teacher_seed": 2026 + k}
            if not base.verify_completion(folder / f"fold{k}", "completion.json", identity, TEACHER_ARTIFACTS):
                raise ValueError("OOF completion exists but teacher fold is incomplete")
        print("Verified complete training-only OOF teacher cache", flush=True)
        with np.load(folder / "oof_logits.npz", allow_pickle=False) as cache:
            if not np.array_equal(cache["fold"], folds["assignment"]):
                raise ValueError("OOF ordering differs")
            return torch.from_numpy(cache["logits"].copy())
    # Deliberately never access validation_target or test_target.
    with np.load(args.source / "embeddings.npz", allow_pickle=False) as cache:
        x = torch.from_numpy(cache["train_target"].copy())
        cached_labels = cache["train_y"].copy()
    y = torch.tensor([r["label"] for r in rows["train"]])
    if not np.array_equal(cached_labels, y.numpy()) or x.shape != (len(y), 384) or not torch.isfinite(x).all():
        raise ValueError("Teacher cached inputs/row alignment differ")
    logits = np.full((len(y), 7), np.nan, dtype=np.float32)
    for fold in folds["plans"]:
        logits[fold["heldout"]] = train_teacher_fold(args, x, y, fold)
    if not np.isfinite(logits).all():
        raise ValueError("Incomplete/non-finite OOF coverage")
    base.atomic_npz(folder / "oof_logits.npz", logits=logits, fold=np.array(folds["assignment"]), labels=y.numpy())
    common.dump(folder / "oof_diagnostics.json", {"run": args.fingerprint, "samples": len(y),
                "scores": base.scores(y.numpy(), logits.argmax(1)), "uses_future_text": True, "diagnostic_only": True,
                "note": "OOF recognition on original training set; not predictive validation/test performance or an upper bound",
                "teacher_test_evaluated": False})
    base.seal_completion(folder, "oof_completion.json", expected, OOF_ARTIFACTS)
    return torch.from_numpy(logits)


def weighted_kl_numerator(student_logits, teacher_logits, y, weights, temperature):
    log_student = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher = F.softmax(teacher_logits.detach().float() / temperature, dim=-1)
    per_sample = F.kl_div(log_student, teacher, reduction="none").sum(-1) * temperature**2
    return (per_sample * weights[y]).sum()


def update_window(model, data, parts, weights, optimizer, scheduler, scaler, args, teacher, beta):
    if beta == 0:
        numerator, denominator = base.update_window(model, data, parts, weights, optimizer, scheduler, scaler, args)
        return numerator, 0.0, denominator
    if teacher.shape != (len(data["labels"]), 7) or teacher.requires_grad:
        raise ValueError("Teacher cache must be detached N x 7 logits aligned with train")
    optimizer.zero_grad(set_to_none=True)
    denominator = sum(float(weights[data["labels"][idx].to(args.device)].sum()) for idx in parts)
    ce_sum = kd_sum = 0.0
    for indices in parts:
        y = data["labels"][indices].to(args.device)
        with base.amp_context(args):
            logits = model(base.token_batch(data, indices, args.device))
            ce = base.ce_numerator(logits, y, weights)
            kd = weighted_kl_numerator(logits, teacher[indices].to(args.device), y, weights, args.temperature)
            loss = (ce + beta * kd) / denominator
        if not torch.isfinite(loss):
            raise ValueError("Non-finite distillation loss")
        scaler.scale(loss).backward()
        ce_sum += float(ce.detach())
        kd_sum += float(kd.detach())
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    if not torch.isfinite(norm) and not scaler.is_enabled():
        raise ValueError("Non-finite student gradient")
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    if scaler.get_scale() >= previous_scale:
        scheduler.step()
    return ce_sum, kd_sum, denominator


def train_student(args, rows, asset, spec, seed, teacher):
    folder = base.folder_for(args, spec, seed)
    expected = base.identity(args, spec, seed)
    if base.verify_completion(folder, "completion.json", expected, base.TRAIN_ARTIFACTS):
        print(f"Verified completed KD student {spec['name']} seed={seed}", flush=True)
        return common.read(folder / "result.json")
    config = folder / "experiment_config.json"
    if config.exists() and common.read(config) != expected:
        raise ValueError("Student configuration differs")
    if folder.exists() and any(folder.iterdir()) and not config.exists():
        raise ValueError("Untracked partial student folder")
    folder.mkdir(parents=True, exist_ok=True)
    common.dump(config, expected)
    data = {s: base.token_cache(args, asset, "minilm", s, rows[s]) for s in ["train", "validation"]}
    model = base.build_model(args, asset, spec, seed)
    optimizer = base.optimizer_for(model, args)
    steps = math.ceil(len(rows["train"]) / (args.micro_batch * args.accumulation))
    scheduler = base.get_linear_schedule_with_warmup(optimizer, math.ceil(args.epochs * steps * args.warmup_ratio), args.epochs * steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
    counts = np.bincount(data["train"]["labels"].numpy(), minlength=7)
    weights = torch.from_numpy(common.compute_class_weights(counts, counts.sum(), "baseline_sqrt_clip5")).to(args.device)
    best, best_score, best_epoch, history, start, previous_seconds = None, -1.0, 0, [], 1, 0.0
    last = folder / "last.pt"
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=False)
        if saved["identity"] != expected:
            raise ValueError("Student resume signature differs")
        base.restore_payload(model, saved["current"], expected)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        best, best_score, best_epoch = saved["best"], saved["best_score"], saved["best_epoch"]
        history, start, previous_seconds = saved["history"], saved["epoch"] + 1, saved["training_seconds"]
        del saved
        print(f"KD {spec['name']} seed={seed}: resume completed epoch {start-1}", flush=True)
    started = time.perf_counter()
    print(f"TRAIN {spec['name']} seed={seed}; beta={spec['beta']}; teacher NOT loaded on student GPU", flush=True)
    for epoch in range(start, args.epochs + 1):
        if best_epoch and epoch - 1 - best_epoch >= args.patience:
            break
        common.seed_all(seed + 1000 * epoch)
        model.train()
        order = torch.randperm(len(rows["train"]), generator=torch.Generator().manual_seed(seed + epoch))
        ce_sum = kd_sum = denominator_sum = 0.0
        epoch_started = last_print = time.perf_counter()
        for step, parts in enumerate(base.training_windows(order, args.micro_batch, args.accumulation), 1):
            ce, kd, denominator = update_window(model, data["train"], parts, weights, optimizer, scheduler, scaler, args, teacher, spec["beta"])
            ce_sum += ce
            kd_sum += kd
            denominator_sum += denominator
            now = time.perf_counter()
            if now - last_print > 20 or step == steps:
                print(f"{spec['name']} seed={seed} epoch={epoch}/{args.epochs} step={step}/{steps}; "
                      f"CE={ce_sum/denominator_sum:.4f}; KD={kd_sum/denominator_sum:.4f}; "
                      f"epoch ETA={(now-epoch_started)/step*(steps-step)/60:.1f} min", flush=True)
                last_print = now
        logits = base.predict(model, data["validation"], args)
        val = base.scores(data["validation"]["labels"], logits.argmax(1))
        history.append({"epoch": epoch, "loss": (ce_sum + spec["beta"] * kd_sum) / denominator_sum,
                        "ce": ce_sum / denominator_sum, "kd": kd_sum / denominator_sum,
                        "validation": val, "seconds": time.perf_counter() - epoch_started})
        if val["macro_f1"] > best_score:
            best_score, best_epoch = val["macro_f1"], epoch
            best = base.checkpoint_payload(model, spec, asset, expected)
        base.atomic_save({"identity": expected, "epoch": epoch, "current": base.checkpoint_payload(model, spec, asset, expected),
                          "best": best, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                          "scaler": scaler.state_dict(), "history": history, "best_score": best_score,
                          "best_epoch": best_epoch, "training_seconds": previous_seconds + time.perf_counter() - started}, last)
        common.dump(folder / "history.json", history)
        print(f"VALIDATION {spec['name']} seed={seed} epoch={epoch} Macro-F1={val['macro_f1']:.4%}", flush=True)
        if epoch - best_epoch >= args.patience:
            break
    if best is None:
        raise ValueError("No student validation checkpoint")
    base.restore_payload(model, best, expected)
    logits = base.predict(model, data["validation"], args)
    val = base.scores(data["validation"]["labels"], logits.argmax(1))
    if not np.isclose(val["macro_f1"], best_score, atol=1e-10):
        raise ValueError("Best student checkpoint did not reproduce validation score")
    base.atomic_save(best, folder / "best.pt")
    base.atomic_npz(folder / "validation_predictions.npz", logits=logits, labels=data["validation"]["labels"].numpy())
    result = {"model": spec["name"], "seed": seed, "spec": spec, "validation": val, "best_epoch": best_epoch,
              "epochs_completed": len(history), "status": "completed", "uses_future_text": False,
              "uses_previous_label": False, "training_supervision": "ground truth + TRAIN-ONLY OOF future teacher probabilities",
              "test_evaluated": False, "run": args.fingerprint, "training_seconds": previous_seconds + time.perf_counter() - started}
    common.dump(folder / "result.json", result)
    common.dump(folder / "history.json", history)
    base.seal_completion(folder, "completion.json", expected, base.TRAIN_ARTIFACTS)
    del model, optimizer, best
    if args.device == "cuda":
        torch.cuda.empty_cache()
    return result


def screen(args, rows, asset, teacher, references):
    results = [train_student(args, rows, asset, spec, 42, teacher) for spec in plan(args.betas)]
    selected = sorted(results, key=lambda r: (-r["validation"]["macro_f1"], r["model"]))[0]["spec"]
    value = {"run": args.fingerprint, "selection_basis": "seed42 validation Macro-F1 ONLY",
             "screen_results": results, "confirmation_spec": selected,
             "reference_validation_macro_f1": references[0]["result"]["validation"]["macro_f1"],
             "note": "best KD challenger retained even if CE wins screen; final choice may still be CE", "test_evaluated": False}
    path = args.out / "screen_selection.json"
    if path.exists() and common.read(path) != value:
        raise ValueError("KD screen selection changed")
    common.dump(path, value)
    print(f"SCREEN complete: {selected['name']}; no test evaluation", flush=True)
    return value


def summarize(reference_training, training, reference_tests, tests):
    summaries = []
    for name, tr, te in [("minilm_full_ce_reference", reference_training, reference_tests), (training[0]["model"], training, tests)]:
        val = [r["validation"]["macro_f1"] for r in tr]
        test = [r["test"]["macro_f1"] for r in te]
        summaries.append({"model": name, "validation_macro_f1_mean": float(np.mean(val)),
                          "validation_macro_f1_std": float(np.std(val, ddof=1)), "test_macro_f1_mean": float(np.mean(test)),
                          "test_macro_f1_std": float(np.std(test, ddof=1)),
                          "test_accuracy_mean": float(np.mean([r["test"]["accuracy"] for r in te])),
                          "test_negative_f1_mean": float(np.mean([r["test"]["negative"]["f1"] for r in te])),
                          "test_per_class": {label: {k + "_mean": float(np.mean([r["test"]["per_class"][label][k] for r in te]))
                                for k in ["precision", "recall", "f1", "true_count", "pred_count", "tp_count"]} for label in common.LABELS}})
    paired = [{"seed": seed, "test_macro_f1_delta": next(r for r in tests if r["seed"] == seed)["test"]["macro_f1"] -
               next(r for r in reference_tests if r["seed"] == seed)["test"]["macro_f1"]} for seed in SEEDS]
    return summaries, paired


def confirm(args, rows, asset, teacher, references):
    path = args.out / "screen_selection.json"
    if not path.exists():
        raise ValueError("Run screen first, or use stage all")
    saved = common.read(path)
    screen_results = []
    for spec in plan(args.betas):
        folder = base.folder_for(args, spec, 42)
        if not base.verify_completion(folder, "completion.json", base.identity(args, spec, 42), base.TRAIN_ARTIFACTS):
            raise ValueError("KD screen incomplete")
        screen_results.append(common.read(folder / "result.json"))
    selected = sorted(screen_results, key=lambda r: (-r["validation"]["macro_f1"], r["model"]))[0]["spec"]
    if saved["run"] != args.fingerprint or saved["screen_results"] != screen_results or saved["confirmation_spec"] != selected:
        raise ValueError("Screen selection was modified")
    training = [train_student(args, rows, asset, selected, seed, teacher) for seed in SEEDS]
    reference_training = [r["result"] for r in references]
    ce_mean = float(np.mean([r["validation"]["macro_f1"] for r in reference_training]))
    kd_mean = float(np.mean([r["validation"]["macro_f1"] for r in training]))
    selected_name = selected["name"] if kd_mean > ce_mean else "minilm_full_ce_reference"
    common.dump(args.out / "validation_selection.json", {"run": args.fingerprint, "selection_basis": "three-seed validation mean ONLY",
                "selected_model": selected_name, "ce_validation_mean": ce_mean, "kd_validation_mean": kd_mean,
                "confirmed_kd_spec": selected})
    # No teacher future embeddings or probabilities are supplied to validation/test forward.
    tests = [base.evaluate_test(args, rows, {"minilm": asset}, selected, seed) for seed in SEEDS]
    reference_tests = []
    for record in references:
        folder = Path(record["folder"])
        if not base.verify_completion(folder, "test_completion.json", record["identity"], base.TEST_ARTIFACTS):
            raise ValueError("Incomplete reference test evaluation")
        reference_tests.append(common.read(folder / "test_result.json"))
    summary, paired = summarize(reference_training, training, reference_tests, tests)
    common.dump(args.out / "results.json", {"reference_training": reference_training, "training": training,
                "reference_test": reference_tests, "test": tests})
    common.dump(args.out / "summary.json", summary)
    common.dump(args.out / "paired_comparisons.json", paired)
    lines = ["情绪导向软标签蒸馏实验", "", "对照：上一轮已完成的MiniLM全量微调CE，三个种子；校验哈希后只读复用。",
             "教师：仅原训练集，按对话三折交叉拟合；外层留出对话既不参与教师训练，也不参与教师早停。",
             "教师看到下一句话的冻结向量；学生训练/推理输入始终只有最近六个历史话语轮次。",
             "教师只在训练期给出七类软分布，不拟合256维隐层；温度预先固定，beta仅按验证集筛选。",
             f"实际温度T={args.temperature}; beta候选={args.betas}; 验证集选定模型={selected_name}", "",
             "模型 | 验证Macro-F1 | 测试Macro-F1 | 测试Accuracy | 负面二分类F1"]
    for r in summary:
        lines.append(f"{r['model']} | {r['validation_macro_f1_mean']:.2%} ± {100*r['validation_macro_f1_std']:.2f}pp | "
                     f"{r['test_macro_f1_mean']:.2%} ± {100*r['test_macro_f1_std']:.2f}pp | "
                     f"{r['test_accuracy_mean']:.2%} | {r['test_negative_f1_mean']:.2%}")
    lines += ["", "配对测试差值（KD-CE）：" + "; ".join(f"seed{r['seed']}: {100*r['test_macro_f1_delta']:+.3f}pp" for r in paired),
              "", "标准差不是显著性检验。教师OOF分数是事后识别诊断，不是预测成绩或理论上限。",
              "测试集此前用于过错例诊断，本轮为探索性改进；不以测试成绩挑beta/检查点/模型。",
              "仅CE与KD对照尚不能把收益归因于逐样本未来知识，而排除一般软标签正则化效应。",
              "不预设提分；KD不如CE时保留原模型。逐类精确率、召回率、F1及预测数量见summary.json。"]
    (args.out / "emotion_distillation_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    common.dump(args.out / "run_status.json", {"run": args.fingerprint, "status": "completed",
                "teacher_folds": 3, "screen_students": len(args.betas), "confirmed_students": 3,
                "selected_model": selected_name})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["check", "teacher", "screen", "confirm", "all"], default="all")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--reference", type=Path, default=ROOT / "runs" / "encoder_minilm_v1")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "emotion_kd_v1")
    parser.add_argument("--betas", nargs="+", type=float, default=[0.1, 0.3])
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--teacher-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--teacher-epochs", type=int, default=15)
    parser.add_argument("--teacher-patience", type=int, default=4)
    parser.add_argument("--teacher-batch", type=int, default=512)
    parser.add_argument("--teacher-lr", type=float, default=0.001)
    args = parser.parse_args(argv)
    if not args.betas or min(args.betas) <= 0 or not all(math.isfinite(v) for v in args.betas) or len(set(args.betas)) != len(args.betas):
        parser.error("Finite positive distinct beta candidates required; zero is the read-only CE reference")
    if len({s["name"] for s in plan(args.betas)}) != len(args.betas):
        parser.error("Beta names collide")
    if not math.isfinite(args.temperature) or args.temperature <= 0 or min(args.teacher_epochs, args.teacher_patience, args.teacher_batch) < 1 or not math.isfinite(args.teacher_lr) or args.teacher_lr <= 0:
        parser.error("Invalid temperature or teacher settings")
    args.reference, args.out = args.reference.resolve(), args.out.resolve()
    args.source = args.source.resolve() if args.source else None
    return args


def main(argv=None):
    args = parse_args(argv)
    rows, stats, hashes, asset, reference, records = load_reference(args)
    folds = make_folds(rows["train"])
    ensure_manifest(args, hashes, asset, reference, records, folds)
    common.dump(args.out / "data_stats.json", stats)
    if args.stage == "check":
        print(f"CHECK PASS: three completed CE references, unchanged code/source/assets, three dialogue-disjoint TRAIN-only teacher folds; "
              f"student={args.device}/{args.precision}; teacher={args.teacher_device}; no training.", flush=True)
        return
    teacher = prepare_teacher(args, rows, folds)
    if args.stage == "teacher":
        common.dump(args.out / "run_status.json", {"run": args.fingerprint, "status": "teacher_completed", "student_training_started": False})
        return
    if args.stage in ["screen", "all"]:
        screen(args, rows, asset, teacher, records)
        common.dump(args.out / "run_status.json", {"run": args.fingerprint, "status": "screen_completed", "test_evaluated": False})
    if args.stage in ["confirm", "all"]:
        confirm(args, rows, asset, teacher, records)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted. Completed teacher/student epochs retained; rerun identical command to resume.", flush=True)
        sys.exit(130)
