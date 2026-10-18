"""Controlled ablation and improvement experiments for EmoWOZ future-emotion prediction.

Designed for strict scientific attribution on top of the `runs/full` baseline:
- Route 1 (Minority Class Route): Isolates class weighting, Focal Loss, and 1-parameter
  Prior Logit Adjustment (tau) one factor at a time on the exact same architecture and input.
  Logs per-class predicted counts (`pred_count`) and true positives (`tp_count`) to distinguish
  never-predicting from false-positive misclassification.
- Route 2 (Semantic Prediction / JEPA Route): Uses the exact same input (`x_hist`) and
  `Predictor` trunk as `history_classifier`. Compares non-learned vector blends, residual
  semantic prediction, and duplicate-target-masked Multi-Positive InfoNCE. Also logs vector
  norm/std and candidate retrieval hubness (repeat-retrieval frequency).
- Route 3 (Offline Utility Diagnostics): Evaluates Next-Turn Negative Emotion Prediction and
  Observational Utility Ranking with accurate, non-overclaiming terminology.
- Report generation is 100% data-driven (never pre-writes success claims).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import importlib.metadata
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn

ROOT = Path(__file__).resolve().parent
LABELS = ["neutral", "fearful", "dissatisfied", "apologetic", "abusive", "excited", "satisfied"]
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SCHEMA_VERSION = 2
REQUIRED_DATA = ["train.json", "validation.json", "test.json", "data_stats.json", "embedding_meta.json", "embeddings.npz"]
EXPERIMENT_ARTIFACTS = ["best.pt", "history.json", "result.json", "classification_report.json",
                        "confusion_matrix.json", "test_predictions.jsonl", "stratified_errors.json"]


def dump(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path: Path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def protect_paths(out, source):
    out, source = Path(out).resolve(), Path(source).resolve()
    protected = [source, (ROOT / "runs" / "full").resolve()]
    for reference in protected:
        if out == reference or out in reference.parents or reference in out.parents:
            raise ValueError(f"Output must be disjoint from protected baseline: {reference}")


def inference_configuration(source):
    meta = read(source / "embedding_meta.json")
    signature = meta["signature"]
    stats = read(source / "data_stats.json")
    config = read(source / "run_config.json") if (source / "run_config.json").exists() else {}
    precision = meta.get("precision", "float16" if config.get("device") == "cuda" else "float32")
    return {"model": signature["model"], "max_tokens": int(signature["max_tokens"]),
            "truncation_side": signature["truncation_side"], "history_turns": int(stats["history_turns"]),
            "normalize_embeddings": True, "encoder_precision": precision,
            "precision_provenance": "recorded" if "precision" in meta else "inferred_from_source_run_config",
            "encoder_revision": signature.get("revision"), "labels": LABELS}


def build_run_manifest(args):
    protect_paths(args.out, args.reuse_from)
    source = args.reuse_from
    hashes = {name: digest(source / name) for name in REQUIRED_DATA}
    meta = read(source / "embedding_meta.json")
    for split in ["train", "validation", "test"]:
        if hashes[f"{split}.json"] != meta["signature"]["sha256"][f"{split}.json"]:
            raise ValueError(f"Source split changed since encoding: {split}")
    for name in ["run_config.json", "environment.json"]:
        if (source / name).exists():
            hashes[name] = digest(source / name)
    return {"schema_version": SCHEMA_VERSION, "code_sha256": digest(Path(__file__)), "source_sha256": hashes,
            "training": {"device": args.device, "seeds": args.seeds, "train_batch": args.train_batch,
                         "epochs": args.epochs, "patience": args.patience, "lr": args.lr},
            "experiment_plan": experiment_plan(), "inference": inference_configuration(source),
            "environment": {"python": sys.version, **{name: importlib.metadata.version(name)
                             for name in ["torch", "numpy", "scikit-learn"]}},
            "selection_policy": "validation_macro_f1_only",
            "test_context": "exploratory_followup; baseline test results have already been inspected"}


def check_manifest(args):
    expected = build_run_manifest(args)
    path = args.out / "provenance.json"
    if path.exists():
        if read(path) != expected:
            raise ValueError("Code, configuration, environment or source changed. Use a NEW --out directory; no files overwritten.")
    elif args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Existing output has no provenance manifest. Use a new empty output directory.")
    args.out.mkdir(parents=True, exist_ok=True)
    dump(path, expected)
    args.run_fingerprint = fingerprint(expected)
    args.inference_config = expected["inference"]
    return expected


def validate_arrays(data, source):
    stats = read(source / "data_stats.json")
    if stats["labels"] != LABELS:
        raise ValueError("Label mapping differs from expected seven classes")
    dimension = None
    dialogue_sets = {}
    for split in ["train", "validation", "test"]:
        rows = read(source / f"{split}.json")
        n = len(rows)
        if not n or n != stats["splits"][split]["samples"]:
            raise ValueError(f"Sample count mismatch: {split}")
        for key, field in [("y", "label"), ("prev", "previous_label")]:
            labels = data[f"{split}_{key}"]
            if labels.shape != (n,) or not np.array_equal(labels, [r[field] for r in rows]):
                raise ValueError(f"Rows and cached labels differ: {split}/{key}")
            if labels.dtype.kind not in "iu" or np.any((labels < 0) | (labels > 6)):
                raise ValueError("Invalid class indices")
        for key in ["x", "last", "target"]:
            vectors = data[f"{split}_{key}"]
            if vectors.ndim != 2 or len(vectors) != n or vectors.dtype != np.float32 or not np.isfinite(vectors).all():
                raise ValueError(f"Invalid vectors: {split}/{key}")
            dimension = vectors.shape[1] if dimension is None else dimension
            if vectors.shape[1] != dimension or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=0.01):
                raise ValueError("Embedding dimension or normalization mismatch")
        counts = np.bincount(data[split + "_y"], minlength=7)
        if counts.tolist() != [stats["splits"][split]["class_counts"][label] for label in LABELS]:
            raise ValueError("Recorded class counts differ from cached labels")
        dialogue_sets[split] = {r["dialogue_id"] for r in rows}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if dialogue_sets[a] & dialogue_sets[b]:
            raise ValueError("Dialogue-level split leakage")
    if dimension != int(read(source / "embedding_meta.json")["dimension"]):
        raise ValueError("Encoder metadata dimension differs from vectors")


def experiment_identity(args, spec, seed):
    return {"schema_version": SCHEMA_VERSION, "run_fingerprint": args.run_fingerprint,
            "specification": spec, "seed": seed}


def load_completed(args, spec, seed):
    folder = args.out / "models" / f"{spec['exp_name']}_seed{seed}"
    config_path, completion = folder / "experiment_config.json", folder / "completion.json"
    identity = experiment_identity(args, spec, seed)
    if config_path.exists() and read(config_path) != identity:
        raise ValueError(f"Experiment signature mismatch: {folder}")
    if completion.exists():
        record = read(completion)
        if record["identity"] != identity or record.get("status") != "completed":
            raise ValueError("Invalid completion record")
        if set(record["artifacts"]) != set(EXPERIMENT_ARTIFACTS):
            raise ValueError("Completion artifact list is incomplete")
        for name, sha in record["artifacts"].items():
            if digest(folder / name) != sha:
                raise ValueError(f"Completed artifact changed: {folder / name}")
        return read(folder / "result.json")
    if folder.exists() and any(folder.iterdir()) and not config_path.exists():
        raise ValueError(f"Untracked old experiment folder: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    dump(config_path, identity)
    return None


class Predictor(nn.Module):
    """Identical trunk and head parameter count to runs/full baseline for fair attribution.

    When `residual_anchor=True`, the semantic head predicts a residual update added to the
    input anchor vector `anchor` (defaults to `x`) before L2 normalization.
    """

    def __init__(self, dimension: int, hidden: int = 256, residual_anchor: bool = False):
        super().__init__()
        self.dimension = dimension
        self.hidden = hidden
        self.residual_anchor = residual_anchor
        self.trunk = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Linear(hidden, 7)
        self.future = nn.Linear(hidden, dimension)
        if residual_anchor:
            # Zero residual: initial prediction equals the unit anchor exactly.
            # No extra random draws, so trunk/classifier and RNG state are paired.
            nn.init.zeros_(self.future.weight)
            nn.init.zeros_(self.future.bias)

    def forward(self, x: torch.Tensor, anchor: torch.Tensor | None = None):
        z = self.trunk(x)
        logits = self.classifier(z)
        raw_future = self.future(z)
        if self.residual_anchor:
            base = x if anchor is None else anchor
            future_vec = F.normalize(base + raw_future, dim=-1)
        else:
            future_vec = F.normalize(raw_future, dim=-1)
        return logits, future_vec


def compute_class_weights(counts: np.ndarray, total: int, mode: str) -> np.ndarray:
    """Compute class weights according to the specified ablation mode."""
    safe_counts = np.maximum(counts.astype(np.float64), 1.0)
    if mode == "none":
        return np.ones(7, dtype=np.float32)
    if mode == "baseline_sqrt_clip5":
        # Exact formula from runs/full
        w = np.sqrt(total / (7.0 * safe_counts)).clip(0.25, 5.0)
        return w.astype(np.float32)
    if mode == "sqrt_clip15":
        # Removes the tight 5.0 ceiling to test stronger minority compensation
        w = np.sqrt(total / (7.0 * safe_counts)).clip(0.25, 15.0)
        return w.astype(np.float32)
    if mode == "effective_num":
        # Cui et al. (CVPR 2019) Class-Balanced Loss based on effective number of samples
        beta = 0.999
        eff_num = 1.0 - np.power(beta, safe_counts)
        w = (1.0 - beta) / np.maximum(eff_num, 1e-8)
        w = w / np.mean(w)
        return np.clip(w, 0.25, 20.0).astype(np.float32)
    raise ValueError(f"Unknown weight_mode: {mode}")


def classification_loss_fn(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    loss_type: str = "ce",
    focal_gamma: float = 1.5,
) -> torch.Tensor:
    if loss_type == "ce":
        return F.cross_entropy(logits, targets, weight=weights)
    if loss_type == "focal":
        ce_per_sample = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_per_sample)
        focal_factor = (1.0 - pt).clamp(min=1e-6) ** focal_gamma
        weighted = focal_factor * ce_per_sample * weights[targets]
        return weighted.sum() / weights[targets].sum().clamp_min(1e-12)
    raise ValueError(f"Unknown loss_type: {loss_type}")


def multi_positive_infonce(
    predicted: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 0.10,
    dup_threshold: float = 0.995,
) -> torch.Tensor:
    """Contrastive loss that groups duplicate/near-identical future targets in a batch as positives.

    Prevents identical responses (e.g. 'thank you', 'yes') in the same batch from being
    penalized as false negatives.
    """
    sim = (predicted @ targets.T) / temperature
    with torch.no_grad():
        target_sim = targets @ targets.T
        pos_mask = target_sim >= dup_threshold
        pos_mask.fill_diagonal_(True)
        pos_weights = pos_mask.float()
        pos_weights = pos_weights / pos_weights.sum(dim=1, keepdim=True).clamp(min=1.0)
    log_prob = F.log_softmax(sim, dim=1)
    loss_i2t = -(pos_weights * log_prob).sum(dim=1).mean()
    log_prob_t2i = F.log_softmax(sim.T, dim=1)
    loss_t2i = -(pos_weights * log_prob_t2i).sum(dim=1).mean()
    return 0.5 * (loss_i2t + loss_t2i)


def fit_single_param_logit_adjustment(
    val_logits: np.ndarray, val_y: np.ndarray, train_counts: np.ndarray
) -> tuple[float, np.ndarray]:
    """Menon et al. (ICLR 2021) 1-parameter post-hoc logit adjustment:

        adjusted_logits = logits - tau * log(pi)

    Tunes only a SINGLE scalar `tau` on the validation set to maximize 7-class Macro-F1.
    Avoids multi-parameter overfitting on small validation minority classes.
    """
    priors = np.maximum(train_counts.astype(np.float64), 1.0)
    priors = priors / priors.sum()
    log_prior = np.log(priors + 1e-8)

    best_tau = 0.0
    best_f1 = float(f1_score(val_y, val_logits.argmax(axis=1), labels=list(range(7)), average="macro", zero_division=0))
    for tau in np.linspace(-0.20, 0.80, 101):
        adj = val_logits - tau * log_prior[None, :]
        score = float(f1_score(val_y, adj.argmax(axis=1), labels=list(range(7)), average="macro", zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_tau = float(tau)
    offset = (-best_tau * log_prior).astype(np.float32)
    return best_tau, offset


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, labels=list(range(7)), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, prediction, labels=list(range(7)), average="weighted", zero_division=0)),
    }


def detailed_class_diagnostics(y: np.ndarray, prediction: np.ndarray) -> dict:
    """Record per-class F1, predicted count, true count, and true positive count."""
    f1s = f1_score(y, prediction, labels=list(range(7)), average=None, zero_division=0)
    true_counts = np.bincount(y, minlength=7)
    pred_counts = np.bincount(prediction, minlength=7)
    out = {}
    for c in range(7):
        tp = int(np.sum((y == c) & (prediction == c)))
        out[LABELS[c]] = {
            "f1": float(f1s[c]),
            "true_count": int(true_counts[c]),
            "pred_count": int(pred_counts[c]),
            "tp_count": tp,
            "precision": tp / max(int(pred_counts[c]), 1),
            "recall": tp / max(int(true_counts[c]), 1),
        }
    return out


def save_evaluation(folder, y, predictions, rows):
    dump(folder / "classification_report.json", classification_report(
        y, predictions, labels=list(range(7)), target_names=LABELS, zero_division=0, output_dict=True))
    dump(folder / "confusion_matrix.json", confusion_matrix(y, predictions, labels=list(range(7))).tolist())
    records = [{**row, "true_emotion": LABELS[int(actual)], "predicted_emotion": LABELS[int(pred)]}
               for row, actual, pred in zip(rows, y, predictions)]
    path = folder / "test_predictions.jsonl"
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)
    rng = random.Random(2026)
    errors = {}
    pairs = Counter()
    for label in LABELS:
        group = [r for r in records if r["true_emotion"] == label and r["true_emotion"] != r["predicted_emotion"]]
        pairs.update((r["true_emotion"], r["predicted_emotion"]) for r in group)
        errors[label] = {"total_errors": len(group), "examples": rng.sample(group, min(5, len(group)))}
    dump(folder / "stratified_errors.json", {"sampling_seed": 2026, "by_true_emotion": errors,
         "top_confusions": [{"true": a, "predicted": b, "count": count} for (a, b), count in pairs.most_common(12)]})


def retrieval_with_diagnostics(predicted: np.ndarray, targets: np.ndarray, rows: list[dict], seed: int = 2026) -> dict:
    """Evaluate sampled retrieval and compute hubness / repeat-retrieval diagnostics."""
    first = {}
    for i, row in enumerate(rows):
        first.setdefault(row["future_text"].strip().lower(), i)
    candidates = list(first.values())
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    candidates = np.array(candidates[:1000])
    queries = rng.permutation(len(candidates))[: min(500, len(candidates))]

    cand_targets = targets[candidates]
    query_preds = predicted[candidates[queries]]
    sim = query_preds @ cand_targets.T
    ranked = np.argsort(-sim, axis=1)
    correct = queries[:, None]

    # Hubness diagnostics: how many unique candidates appear in top-1 / top-10, and max repeat frequency
    top1_ids = ranked[:, 0]
    top10_ids = ranked[:, :10].reshape(-1)
    top1_counts = Counter(top1_ids.tolist())
    top10_counts = Counter(top10_ids.tolist())
    direct_cos = float(np.mean(np.sum(query_preds * cand_targets[queries], axis=1)))

    return {
        "candidate_count": int(len(candidates)),
        "query_count": int(len(queries)),
        "recall_at_1": float(np.mean(np.any(ranked[:, :1] == correct, axis=1))),
        "recall_at_5": float(np.mean(np.any(ranked[:, :5] == correct, axis=1))),
        "recall_at_10": float(np.mean(np.any(ranked[:, :10] == correct, axis=1))),
        "mean_target_cosine": direct_cos,
        "embedding_std_mean": float(predicted.std(axis=0).mean()),
        "unique_top1_candidates": int(len(top1_counts)),
        "max_top1_repeat_count": int(max(top1_counts.values()) if top1_counts else 0),
        "unique_top10_candidates": int(len(top10_counts)),
        "max_top10_repeat_count": int(max(top10_counts.values()) if top10_counts else 0),
    }


@torch.no_grad()
def infer(model: Predictor, x: torch.Tensor, anchor: torch.Tensor | None = None, batch: int = 1024):
    model.eval()
    logits, vectors = [], []
    for i in range(0, len(x), batch):
        anc = anchor[i : i + batch] if anchor is not None else None
        a, b = model(x[i : i + batch], anc)
        logits.append(a.cpu())
        vectors.append(b.cpu())
    return torch.cat(logits).numpy(), torch.cat(vectors).numpy()


def run_single_experiment(
    args,
    data: dict,
    rows: list[dict],
    exp_name: str,
    seed: int,
    *,
    input_field: str = "x",
    weight_mode: str = "baseline_sqrt_clip5",
    loss_type: str = "ce",
    use_logit_adjust: bool = False,
    semantic_weight: float = 0.0,
    residual_anchor: str | None = None,
    use_mp_infonce: bool = False,
    infonce_weight: float = 0.15,
) -> dict:
    """Run a single strictly controlled experiment configuration."""
    seed_all(seed)
    folder = args.out / "models" / f"{exp_name}_seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    x = {s: torch.from_numpy(data[s + "_" + input_field]).to(args.device) for s in ["train", "validation", "test"]}
    anchor = (
        {s: torch.from_numpy(data[s + "_" + residual_anchor]).to(args.device) for s in ["train", "validation", "test"]}
        if residual_anchor is not None
        else {s: None for s in ["train", "validation", "test"]}
    )
    y = torch.from_numpy(data["train_y"]).to(args.device)
    target = torch.from_numpy(data["train_target"]).to(args.device)

    counts = np.bincount(data["train_y"], minlength=7)
    weights_np = compute_class_weights(counts, len(y), weight_mode)
    weights = torch.tensor(weights_np, dtype=torch.float32, device=args.device)

    model = Predictor(x["train"].shape[1], residual_anchor=(residual_anchor is not None)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

    best_score = -1.0
    best_epoch = 0
    history = []
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(y), device=args.device)
        sums = np.zeros(3)
        for ids in order.split(args.train_batch):
            optimizer.zero_grad(set_to_none=True)
            anc_b = anchor["train"][ids] if residual_anchor is not None else None
            logits, prediction = model(x["train"][ids], anc_b)

            classification = classification_loss_fn(logits, y[ids], weights, loss_type=loss_type)
            cosine = (1.0 - (prediction * target[ids]).sum(dim=1)).mean()
            variance = F.mse_loss(prediction.std(dim=0, unbiased=False), target[ids].std(dim=0, unbiased=False))
            semantic = cosine + 0.1 * variance
            if use_mp_infonce and semantic_weight > 0:
                semantic = semantic + infonce_weight * multi_positive_infonce(prediction, target[ids])

            loss = classification + semantic_weight * semantic
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sums += np.array([loss.item(), classification.item(), semantic.item()]) * len(ids)

        val_logits, _ = infer(model, x["validation"], anchor["validation"])
        score = metrics(data["validation_y"], val_logits.argmax(axis=1))["macro_f1"]
        item = {
            "epoch": epoch,
            "loss": float(sums[0] / len(y)),
            "classification_loss": float(sums[1] / len(y)),
            "semantic_loss": float(sums[2] / len(y)),
            "validation_macro_f1": score,
        }
        history.append(item)
        print(f"{exp_name} seed={seed} epoch={epoch} loss={item['loss']:.4f} val_macro_f1={score:.4f}", flush=True)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= args.patience:
            break

    return finish_experiment(args, data, rows, exp_name, seed, model, best_state, best_epoch, best_score, history,
                             weights_np, input_field, weight_mode, loss_type, use_logit_adjust, semantic_weight,
                             residual_anchor, use_mp_infonce, infonce_weight, started)


def finish_experiment(args, data, rows, exp_name, seed, model, best_state, best_epoch, best_score, history,
                      weights_np, input_field, weight_mode, loss_type, use_logit_adjust, semantic_weight,
                      residual_anchor, use_mp_infonce, infonce_weight, started, reused_checkpoint=None):
    folder = args.out / "models" / f"{exp_name}_seed{seed}"
    x = {s: torch.from_numpy(data[s + "_" + input_field]).to(args.device) for s in ["validation", "test"]}
    anchor = {s: torch.from_numpy(data[s + "_" + residual_anchor]).to(args.device) if residual_anchor else None
              for s in ["validation", "test"]}
    counts = np.bincount(data["train_y"], minlength=7)
    model.load_state_dict(best_state, strict=True)
    val_logits, _ = infer(model, x["validation"], anchor["validation"])

    tau = 0.0
    logit_offset = np.zeros(7, dtype=np.float32)
    if use_logit_adjust:
        tau, logit_offset = fit_single_param_logit_adjustment(val_logits, data["validation_y"], counts)

    # Selection and tau fitting are complete before test inference.
    test_logits, predicted_vectors = infer(model, x["test"], anchor["test"])
    val_preds = (val_logits + logit_offset[None, :]).argmax(axis=1)
    test_preds = (test_logits + logit_offset[None, :]).argmax(axis=1)
    raw_test_preds = test_logits.argmax(axis=1)

    result = {
        "model": exp_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_raw_macro_f1": best_score,
        "validation_macro_f1": float(metrics(data["validation_y"], val_preds)["macro_f1"]),
        **metrics(data["test_y"], test_preds),
        "raw_argmax_metrics": metrics(data["test_y"], raw_test_preds),
        "class_diagnostics": detailed_class_diagnostics(data["test_y"], test_preds),
        "raw_class_diagnostics": detailed_class_diagnostics(data["test_y"], raw_test_preds),
        "logit_adjust_tau": tau,
        "logit_offset": logit_offset.tolist(),
        "weight_mode": weight_mode,
        "loss_type": loss_type,
        "semantic_weight": semantic_weight,
        "residual_anchor": residual_anchor,
        "use_mp_infonce": use_mp_infonce,
        "infonce_weight": infonce_weight,
        "input_field": input_field,
        "inference_config": args.inference_config,
        "reused_training_checkpoint": reused_checkpoint,
        "run_fingerprint": args.run_fingerprint,
        "status": "completed",
        "class_weights": weights_np.tolist(),
        "test_selection_used": False,
        "seconds": time.perf_counter() - started,
    }
    if semantic_weight > 0 or residual_anchor is not None:
        result["retrieval"] = retrieval_with_diagnostics(predicted_vectors, data["test_target"], rows)

    torch.save(
        {
            "state_dict": best_state,
            "dimension": x["validation"].shape[1],
            "input_field": input_field,
            "hidden": 256,
            "inference_config": args.inference_config,
            "run_fingerprint": args.run_fingerprint,
            "residual_anchor": residual_anchor,
            "logit_adjust_tau": tau,
            "logit_offset": logit_offset.tolist(),
            "labels": LABELS,
        },
        folder / "best.pt",
    )
    dump(folder / "history.json", history)
    dump(folder / "result.json", result)
    save_evaluation(folder, data["test_y"], test_preds, rows)
    identity = read(folder / "experiment_config.json")
    dump(folder / "completion.json", {"identity": identity, "status": "completed",
         "artifacts": {name: digest(folder / name) for name in EXPERIMENT_ARTIFACTS}})
    print(
        f"[{exp_name} seed={seed}] test_macro_f1={result['macro_f1']*100:.2f}% "
        f"(raw={result['raw_argmax_metrics']['macro_f1']*100:.2f}%) acc={result['accuracy']*100:.2f}%",
        flush=True,
    )
    return result


def adjust_existing_classifier(args, data, rows, seed):
    """Post-hoc adjustment of the EXACT history checkpoint, without retraining."""
    folder = args.out / "models" / f"history_classifier_seed{seed}"
    checkpoint_path = folder / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    baseline = read(folder / "result.json")
    model = Predictor(checkpoint["dimension"]).to(args.device)
    return finish_experiment(args, data, rows, "route1_logit_adjust_tau", seed, model,
         checkpoint["state_dict"], baseline["best_epoch"], baseline["validation_raw_macro_f1"],
         read(folder / "history.json"), np.array(baseline["class_weights"], dtype=np.float32),
         "x", "baseline_sqrt_clip5", "ce", True, 0.0, None, False, 0.15, time.perf_counter(),
         reused_checkpoint={"path": str(checkpoint_path), "sha256": digest(checkpoint_path)})


def evaluate_offline_diagnostics(args, data: dict, all_results: list[dict]) -> dict:
    """Route 3: Offline diagnostic metrics with exact, non-overclaiming definitions:
    1. Next-Turn Negative Emotion Prediction (predicting whether next emotion is in {fearful, dissatisfied, abusive}).
    2. Observational Utility Discrimination on held-out transitions (explicitly NOT causal counterfactual proof).
    """
    neg_classes = {1, 2, 4}
    y_true_neg = np.array([1 if y in neg_classes else 0 for y in data["test_y"]], dtype=np.int64)
    model_names = list(dict.fromkeys(r["model"] for r in all_results if r.get("seed") is not None))

    diagnostics = {
        "next_turn_negative_emotion_prediction": {},
        "observational_utility_discrimination_note": (
            "Compares predicted utility scores across held-out observed transitions from dissatisfied users "
            "(prev_label=dissatisfied -> next_label=satisfied vs next_label in {dissatisfied, abusive}). "
            "Because dialogue histories differ across pairs, this measures observational utility discrimination, "
            "not causal counterfactual intervention."
        ),
        "observational_utility_discrimination": {},
    }

    pos_idx = np.where((data["test_prev"] == 2) & (data["test_y"] == 6))[0]
    neg_idx = np.where((data["test_prev"] == 2) & np.isin(data["test_y"], [2, 4]))[0]
    rng = np.random.default_rng(2026)
    n_pairs = min(200, len(pos_idx), len(neg_idx))
    sampled_pos = rng.choice(pos_idx, size=n_pairs, replace=False) if n_pairs > 0 else np.array([], dtype=int)
    sampled_neg = rng.choice(neg_idx, size=n_pairs, replace=False) if n_pairs > 0 else np.array([], dtype=int)

    for name in model_names:
        f1s, precs, recs, rank_accs = [], [], [], []
        for seed in args.seeds:
            ckpt_path = args.out / "models" / f"{name}_seed{seed}" / "best.pt"
            res_path = args.out / "models" / f"{name}_seed{seed}" / "result.json"
            if not ckpt_path.exists() or not res_path.exists():
                continue
            ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
            res = read(res_path)
            field = "last" if name == "last_user_classifier" else "x"
            anc_field = ckpt.get("residual_anchor")
            model = Predictor(ckpt["dimension"], residual_anchor=(anc_field is not None)).to(args.device)
            model.load_state_dict(ckpt["state_dict"], strict=True)

            x_t = torch.from_numpy(data["test_" + field]).to(args.device)
            anc_t = torch.from_numpy(data["test_" + anc_field]).to(args.device) if anc_field else None
            logits, _ = infer(model, x_t, anc_t)
            offset = np.array(ckpt.get("logit_offset", np.zeros(7)), dtype=np.float32)
            preds = (logits + offset[None, :]).argmax(axis=1)

            y_pred_neg = np.array([1 if p in neg_classes else 0 for p in preds], dtype=np.int64)
            tp = int(np.sum((y_true_neg == 1) & (y_pred_neg == 1)))
            fp = int(np.sum((y_true_neg == 0) & (y_pred_neg == 1)))
            fn = int(np.sum((y_true_neg == 1) & (y_pred_neg == 0)))
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            f1s.append(f1)
            precs.append(prec)
            recs.append(rec)

            if n_pairs > 0:
                shifted = logits - logits.max(axis=1, keepdims=True)
                probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
                utility = probs[:, 6] + 0.8 * probs[:, 5] - probs[:, 2] - 1.5 * probs[:, 4] - 0.8 * probs[:, 1]
                rank_accs.append(float(np.mean(utility[sampled_pos] > utility[sampled_neg])))

        if f1s:
            diagnostics["next_turn_negative_emotion_prediction"][name] = {
                "f1_mean": float(np.mean(f1s)),
                "f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
                "precision_mean": float(np.mean(precs)),
                "recall_mean": float(np.mean(recs)),
            }
        if rank_accs:
            diagnostics["observational_utility_discrimination"][name] = {
                "pair_count": int(n_pairs),
                "accuracy_mean": float(np.mean(rank_accs)),
                "accuracy_std": float(np.std(rank_accs, ddof=1)) if len(rank_accs) > 1 else 0.0,
            }

    dump(args.out / "offline_application_diagnostics.json", diagnostics)
    return diagnostics


def ensure_data_from_baseline(args):
    """Link/copy verified splits and embeddings from runs/full without modifying runs/full."""
    args.out.mkdir(parents=True, exist_ok=True)
    src = args.reuse_from
    expected_hashes = read(args.out / "provenance.json")["source_sha256"]
    for name in REQUIRED_DATA:
        dst = args.out / name
        if not dst.exists():
            if not (src / name).exists():
                raise FileNotFoundError(f"Missing {src / name}. Run baseline `run_experiments.py prepare/encode` first.")
            shutil.copy2(src / name, dst)
        if digest(dst) != expected_hashes[name]:
            raise ValueError(f"Output data differs from the verified source: {dst}. Use a new output directory.")


def summarize_group(results: list[dict], model_name: str) -> dict:
    group = [r for r in results if r["model"] == model_name]
    row = {"model": model_name, "runs": len(group)}
    for k in ["accuracy", "macro_f1", "weighted_f1", "validation_macro_f1"]:
        vals = [r[k] for r in group]
        row[k + "_mean"] = float(np.mean(vals))
        row[k + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    if "class_diagnostics" in group[0]:
        per_cls = {}
        for lbl in LABELS:
            per_cls[lbl] = {
                "f1_mean": float(np.mean([r["class_diagnostics"][lbl]["f1"] for r in group])),
                "pred_count_mean": float(np.mean([r["class_diagnostics"][lbl]["pred_count"] for r in group])),
                "tp_count_mean": float(np.mean([r["class_diagnostics"][lbl]["tp_count"] for r in group])),
                "true_count": int(group[0]["class_diagnostics"][lbl]["true_count"]),
                "precision_mean": float(np.mean([r["class_diagnostics"][lbl]["precision"] for r in group])),
                "recall_mean": float(np.mean([r["class_diagnostics"][lbl]["recall"] for r in group])),
            }
        row["per_class"] = per_cls
    if "retrieval" in group[0]:
        ret = {}
        for rk in [
            "recall_at_1",
            "recall_at_5",
            "recall_at_10",
            "mean_target_cosine",
            "embedding_std_mean",
            "unique_top1_candidates",
            "max_top1_repeat_count",
        ]:
            vals = [r["retrieval"][rk] for r in group]
            ret[rk + "_mean"] = float(np.mean(vals))
            ret[rk + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        row["retrieval"] = ret
    return row


def generate_data_driven_report(args, summaries: list[dict], ret_baselines: dict, offline_diag: dict, all_results):
    """Generate a strictly data-driven markdown summary where every claim is conditional on actual numbers."""
    by_name = {r["model"]: r for r in summaries}
    lines = [
        "# 控制变量改进实验与归因分析汇总（由实测结果自动生成）",
        "",
        "说明：本文件所有结论均由脚本根据实测指标动态判定生成，不预设任何模块必然有效。",
        "基线测试结果已在此前分析中查看。本轮属于预先固定配置的探索性跟进；检查点、tau与推荐配置仅按验证集选择。",
        "均值和标准差仅描述三个随机种子的波动，不是置信区间；本报告不自动作显著性或因果结论。",
        "",
        "## 1. 主线与三条改进路线汇总表",
        "",
        "| 实验配置 | 说明 | 测试集 Macro-F1 | Accuracy | Weighted-F1 | 非零F1类别数 (共7类) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    descriptions = {
        "last_user_classifier": "基线：仅末轮用户话语",
        "history_classifier": "基线：多轮历史分类 (原权重clip5 + CE + argmax)",
        "route1_weight_clip15": "路线1A：仅放宽少数类权重上限 (clip15 + CE + argmax)",
        "route1_effective_num": "路线1B：仅改用有效样本数权重 (Class-Balanced CE + argmax)",
        "route1_focal_only": "路线1C：仅改用 Focal Loss (原权重clip5 + Focal + argmax)",
        "route1_logit_adjust_tau": "路线1D：仅加单参数验证集先验调整 (原权重clip5 + CE + 单参数tau)",
        "jepa_baseline": "路线2A：原版 JEPA (直接预测向量 + 余弦方差损失, w=0.5)",
        "jepa_residual": "路线2B：残差 JEPA (以last_user为残差锚点 + 余弦方差损失, w=0.5)",
        "jepa_residual_mp_infonce": "路线2C：残差 JEPA + 屏蔽重复目标的多正例 InfoNCE (w=0.5)",
        "jepa_w0_1": "路线2消融：残差多正例 JEPA (辅助权重 w=0.1)",
        "jepa_w1_0": "路线2消融：残差多正例 JEPA (辅助权重 w=1.0)",
        "jepa_residual_w0": "路线2匹配对照：同一零初始化残差架构，辅助权重 w=0",
    }
    for r in summaries:
        name = r["model"]
        nonzero_cls = sum(1 for lbl in LABELS if r.get("per_class", {}).get(lbl, {}).get("f1_mean", 0.0) > 1e-4)
        desc = descriptions.get(name, name)
        lines.append(
            f"| `{name}` | {desc} | {r['macro_f1_mean']*100:.2f}% ± {r['macro_f1_std']*100:.2f}% | "
            f"{r['accuracy_mean']*100:.2f}% | {r['weighted_f1_mean']*100:.2f}% | {nonzero_cls}/7 |"
        )

    lines.extend([
        "",
        "## 2. 路线一诊断：少数类预测数量 (`pred_count`) 与真正例 (`tp_count`) 对照",
        "",
        "| 模型配置 | fearful 预测/命中/F1 | apologetic 预测/命中/F1 | abusive 预测/命中/F1 | excited 预测/命中/F1 |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in [
        "history_classifier",
        "route1_weight_clip15",
        "route1_effective_num",
        "route1_focal_only",
        "route1_logit_adjust_tau",
        "jepa_residual_mp_infonce",
    ]:
        if name not in by_name or "per_class" not in by_name[name]:
            continue
        pc = by_name[name]["per_class"]
        cells = []
        for lbl in ["fearful", "apologetic", "abusive", "excited"]:
            d = pc[lbl]
            cells.append(f"{d['pred_count_mean']:.1f} / {d['tp_count_mean']:.1f} / {d['f1_mean']*100:.1f}%")
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## 3. 路线二诊断：未来语义检索与无学习混合参照、Hubness（重复检索频率）对比",
        "",
        "| 检索方式 | Recall@1 | Recall@5 | Recall@10 | 目标余弦相似度 | 向量平均标准差 | Top-1唯一候选数 | 单候选最高重复Top-1次数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for b_name, b_label in [
        ("copy_last_user_embedding", "参照：直接复制最近用户向量"),
        ("copy_history_embedding", "参照：直接复制多轮历史向量"),
        ("static_blend_last_and_history", "参照：无学习静态等权混合 (0.5*last + 0.5*hist)"),
        ("train_mean_target_embedding", "参照：训练集平均目标向量"),
    ]:
        b = ret_baselines[b_name]
        lines.append(
            f"| {b_label} | {b['recall_at_1']*100:.2f}% | {b['recall_at_5']*100:.2f}% | {b['recall_at_10']*100:.2f}% | "
            f"{b['mean_target_cosine']:.4f} | {b['embedding_std_mean']:.4f} | "
            f"{b['unique_top1_candidates']} | {b['max_top1_repeat_count']} |"
        )
    for m_name in ["jepa_baseline", "jepa_residual_w0", "jepa_residual", "jepa_residual_mp_infonce", "jepa_w0_1", "jepa_w1_0"]:
        if m_name in by_name and "retrieval" in by_name[m_name]:
            rt = by_name[m_name]["retrieval"]
            lines.append(
                f"| `{m_name}` | {rt['recall_at_1_mean']*100:.2f}% | {rt['recall_at_5_mean']*100:.2f}% | "
                f"{rt['recall_at_10_mean']*100:.2f}% ± {rt['recall_at_10_std']*100:.2f}% | "
                f"{rt['mean_target_cosine_mean']:.4f} | {rt['embedding_std_mean_mean']:.4f} | "
                f"{rt['unique_top1_candidates_mean']:.1f} | {rt['max_top1_repeat_count_mean']:.1f} |"
            )

    lines.extend([
        "",
        "## 4. 验证集选择与同种子配对差值（描述性，未作显著性检验）",
        "",
        "不能从检索未超过简单参照推断其失败原因，也不能从小幅均值提升直接推断模块有效。",
    ])
    minority = [name for name in ["history_classifier", "route1_weight_clip15", "route1_effective_num",
                "route1_focal_only", "route1_logit_adjust_tau"] if name in by_name]
    best_r1 = max(minority, key=lambda name: by_name[name]["validation_macro_f1_mean"])
    dump(args.out / "validation_selection.json", {"criterion": "mean_validation_macro_f1",
         "selected_minority_configuration": best_r1,
         "validation_macro_f1": by_name[best_r1]["validation_macro_f1_mean"], "test_used_for_selection": False})
    lines.append(f"- 路线一按验证集均值选择：`{best_r1}`，验证Macro-F1={by_name[best_r1]['validation_macro_f1_mean']*100:.2f}%。")
    lines.extend(["", "| 对比（A减B） | 各种子测试Macro-F1差值（百分点） | 差值均值±样本标准差 | 正/零/负种子数 |",
                  "|---|---|---:|---|"])
    pairs = [(name, "history_classifier") for name in minority if name != "history_classifier"]
    pairs += [("history_classifier", "last_user_classifier"), ("jepa_baseline", "history_classifier"),
              ("jepa_residual", "jepa_residual_w0"), ("jepa_residual_mp_infonce", "jepa_residual"),
              ("jepa_residual_mp_infonce", "jepa_residual_w0"), ("jepa_w0_1", "jepa_residual_w0"),
              ("jepa_w1_0", "jepa_residual_w0")]
    paired = []
    for a, b in pairs:
        ra = {r["seed"]: r for r in all_results if r["model"] == a}
        rb = {r["seed"]: r for r in all_results if r["model"] == b}
        seeds = sorted(ra.keys() & rb.keys())
        if not seeds:
            continue
        deltas = np.array([(ra[s]["macro_f1"] - rb[s]["macro_f1"]) * 100 for s in seeds])
        item = {"a": a, "b": b, "seeds": seeds, "delta_pp": deltas.tolist(), "mean_pp": float(deltas.mean()),
                "sample_std_pp": float(deltas.std(ddof=1)) if len(seeds) > 1 else 0.0,
                "positive": int((deltas > 0).sum()), "zero": int((deltas == 0).sum()), "negative": int((deltas < 0).sum())}
        paired.append(item)
        values = "; ".join(f"{s}: {d:+.3f}" for s, d in zip(seeds, deltas))
        lines.append(f"| {a} − {b} | {values} | {item['mean_pp']:+.3f} ± {item['sample_std_pp']:.3f} | {item['positive']}/{item['zero']}/{item['negative']} |")
    dump(args.out / "paired_comparisons.json", paired)
    if "jepa_residual_mp_infonce" in by_name:
        measured = by_name["jepa_residual_mp_infonce"]["retrieval"]["recall_at_10_mean"]
        reference = max(ret_baselines["copy_last_user_embedding"]["recall_at_10"],
                        ret_baselines["static_blend_last_and_history"]["recall_at_10"])
        lines.append(f"\n- 残差多正例模型Recall@10与复制/静态混合中较高者的均值差为{(measured-reference)*100:+.3f}个百分点；仅为实测差值，不自动判断因果或显著性。")
    lines.append("\n观测效用排序与下一轮负向情绪诊断见offline_application_diagnostics.json，不是反事实因果评测。")
    (args.out / "controlled_ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def experiment_plan():
    return [
        {"exp_name": "last_user_classifier", "route": "baseline", "input_field": "last"},
        {"exp_name": "history_classifier", "route": "baseline", "input_field": "x"},
        {"exp_name": "route1_weight_clip15", "route": "minority", "input_field": "x", "weight_mode": "sqrt_clip15"},
        {"exp_name": "route1_effective_num", "route": "minority", "input_field": "x", "weight_mode": "effective_num"},
        {"exp_name": "route1_focal_only", "route": "minority", "input_field": "x", "loss_type": "focal"},
        {"exp_name": "route1_logit_adjust_tau", "route": "minority", "input_field": "x", "use_logit_adjust": True},
        {"exp_name": "jepa_baseline", "route": "semantic", "input_field": "x", "semantic_weight": 0.5},
        {"exp_name": "jepa_residual_w0", "route": "semantic", "input_field": "x", "semantic_weight": 0.0,
         "residual_anchor": "last", "use_mp_infonce": True},
        {"exp_name": "jepa_residual", "route": "semantic", "input_field": "x", "semantic_weight": 0.5, "residual_anchor": "last"},
        {"exp_name": "jepa_residual_mp_infonce", "route": "semantic", "input_field": "x", "semantic_weight": 0.5,
         "residual_anchor": "last", "use_mp_infonce": True},
        {"exp_name": "jepa_w0_1", "route": "semantic", "input_field": "x", "semantic_weight": 0.1,
         "residual_anchor": "last", "use_mp_infonce": True},
        {"exp_name": "jepa_w1_0", "route": "semantic", "input_field": "x", "semantic_weight": 1.0,
         "residual_anchor": "last", "use_mp_infonce": True},
    ]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "runs" / "controlled_v2")
    p.add_argument("--reuse-from", type=Path, default=ROOT / "runs" / "full")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--train-batch", type=int, default=512)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--route", choices=["all", "minority", "semantic"], default="all")
    args = p.parse_args(argv)
    args.out = args.out.resolve()
    args.reuse_from = args.reuse_from.resolve()
    if min(args.train_batch, args.epochs, args.patience) < 1 or args.lr <= 0:
        p.error("Batch size, epoch limit, patience and learning rate must be positive")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        p.error("Seeds must be unique nonnegative integers")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use kg_project or explicitly choose --device cpu")
    provenance = check_manifest(args)
    ensure_data_from_baseline(args)
    data = dict(np.load(args.out / "embeddings.npz", allow_pickle=False))
    validate_arrays(data, args.out)
    rows = read(args.out / "test.json")

    # Compute all 4 non-learned retrieval baselines (including static blend and hubness diagnostics)
    static_blend = data["test_last"] * 0.5 + data["test_x"] * 0.5
    static_blend = static_blend / np.maximum(np.linalg.norm(static_blend, axis=-1, keepdims=True), 1e-8)
    mean_target = np.broadcast_to(data["train_target"].mean(axis=0), data["test_target"].shape)
    mean_target = mean_target / np.maximum(np.linalg.norm(mean_target, axis=-1, keepdims=True), 1e-8)

    ret_baselines = {
        "copy_last_user_embedding": retrieval_with_diagnostics(data["test_last"], data["test_target"], rows),
        "copy_history_embedding": retrieval_with_diagnostics(data["test_x"], data["test_target"], rows),
        "static_blend_last_and_history": retrieval_with_diagnostics(static_blend.astype(np.float32), data["test_target"], rows),
        "train_mean_target_embedding": retrieval_with_diagnostics(mean_target.astype(np.float32), data["test_target"], rows),
    }
    dump(args.out / "retrieval_baselines.json", ret_baselines)

    requested = [spec for spec in experiment_plan() if args.route == "all" or spec["route"] in ["baseline", args.route]]
    # Previously completed routes can be included only after signature/hash checks.
    experiment_specs = [spec for spec in experiment_plan() if spec in requested or all(
        (args.out / "models" / f"{spec['exp_name']}_seed{seed}" / "completion.json").exists() for seed in args.seeds)]
    dump(args.out / "run_config.json", {"run_fingerprint": args.run_fingerprint, "status": "running",
         "requested_route": args.route, "planned_models": [s["exp_name"] for s in experiment_plan()],
         "requested_models": [s["exp_name"] for s in requested], "training": provenance["training"]})

    all_results = []
    for spec in experiment_specs:
        exp_name = spec["exp_name"]
        kwargs = {k: v for k, v in spec.items() if k not in ["exp_name", "route"]}
        for seed in args.seeds:
            completed = load_completed(args, spec, seed)
            if completed is not None:
                print(f"Verified cached result: {exp_name} seed={seed}", flush=True)
                all_results.append(completed)
            else:
                if spec not in requested:
                    continue
                if kwargs.get("use_logit_adjust"):
                    result = adjust_existing_classifier(args, data, rows, seed)
                else:
                    result = run_single_experiment(args, data, rows, exp_name, seed, **kwargs)
                all_results.append(result)
            dump(args.out / "results.json", all_results)

    summaries = [summarize_group(all_results, spec["exp_name"]) for spec in experiment_specs]
    dump(args.out / "summary.json", summaries)
    with (args.out / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["model", "runs", "macro_f1_mean", "macro_f1_std", "accuracy_mean", "accuracy_std", "weighted_f1_mean", "weighted_f1_std",
                  "validation_macro_f1_mean", "validation_macro_f1_std"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(summaries)

    offline_diag = evaluate_offline_diagnostics(args, data, all_results)
    generate_data_driven_report(args, summaries, ret_baselines, offline_diag, all_results)
    done = {(r["model"], r["seed"]) for r in all_results}
    full_plan = {(spec["exp_name"], seed) for spec in experiment_plan() for seed in args.seeds}
    dump(args.out / "run_config.json", {"run_fingerprint": args.run_fingerprint, "status": "completed",
         "requested_route": args.route, "completed_experiments": len(done), "full_plan_experiments": len(full_plan),
         "full_plan_completed": done == full_plan, "training": provenance["training"]})
    print(f"Completed controlled ablations. Summary written to: {args.out / 'controlled_ablation_report.md'}", flush=True)


if __name__ == "__main__":
    main()
