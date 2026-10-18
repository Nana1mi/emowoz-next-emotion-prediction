"""EmoWOZ future-emotion prediction and a frozen-encoder JEPA prototype.

Real data are only used by prepare/encode/train/all. `smoke` is synthetic and
does not write research metrics. Official dialogue splits are never resampled.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn

ROOT = Path(__file__).resolve().parent
LABELS = ["neutral", "fearful", "dissatisfied", "apologetic", "abusive", "excited", "satisfied"]
FILES = ["emowoz-multiwoz.json", "emowoz-dialmage.json", "data-split.json"]
SOURCE = "https://zenodo.org/records/6506504/files/"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check(args):
    packages = {}
    for p in ["torch", "transformers", "sentence-transformers", "scikit-learn", "numpy", "matplotlib"]:
        packages[p] = importlib.metadata.version(p)
    info = {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "packages": packages,
            "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda}
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Activate kg_project and run nvidia-smi.")
        info["gpu"] = torch.cuda.get_device_name(0)
        info["vram_bytes"] = torch.cuda.get_device_properties(0).total_memory
        x = torch.randn(128, 128, device="cuda")
        (x @ x).sum().item()
    print(json.dumps(info, indent=2, ensure_ascii=False), flush=True)
    return info


def download(path):
    if path.exists():
        read(path)  # Reject corrupted cache before proceeding.
        return
    url = SOURCE + path.name + "?download=1"
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".download")
    headers = {"User-Agent": "EmoWOZ-course-project/1.0", "Accept-Encoding": "identity"}
    # Range probe provides total length, even if a proxy silently truncates a
    # full-file response. Downloads use small verified ranges and retain bytes.
    total = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers=headers | {"Range": "bytes=0-0"})
            with urllib.request.urlopen(req, timeout=45) as response:
                match = re.fullmatch(r"bytes 0-0/(\d+)", response.headers.get("Content-Range", ""))
                if response.status != 206 or not match or len(response.read(1)) != 1:
                    raise RuntimeError("Server did not return the requested byte range; use a browser download.")
                total = int(match.group(1))
            break
        except Exception as error:
            print(f"Download probe {attempt + 1}/5 failed: {error}", flush=True)
            if attempt == 4:
                raise
            time.sleep(2)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > total:
        raise RuntimeError(f"Partial file is larger than source: {partial}; inspect before retrying.")
    print(f"Downloading {path.name}: {offset / 1048576:.2f}/{total / 1048576:.2f} MiB (resumable)", flush=True)
    chunk_size = 4 * 1024 * 1024
    failures = 0
    started = last_print = time.monotonic()
    starting_offset = offset
    while offset < total:
        start = offset
        end = min(total - 1, start + chunk_size - 1)
        try:
            req = urllib.request.Request(url, headers=headers | {"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=45) as response:
                expected_range = f"bytes {start}-{end}/{total}"
                if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                    raise RuntimeError(f"Unexpected range response; expected {expected_range}")
                if int(response.headers.get("Content-Length", end - start + 1)) != end - start + 1:
                    raise RuntimeError("Range Content-Length mismatch")
                with partial.open("ab") as out:
                    remaining = end - start + 1
                    while remaining:
                        block = response.read(min(65536, remaining))
                        if not block:
                            raise ConnectionError("Response ended early; keeping downloaded bytes for resume")
                        out.write(block)
                        offset += len(block)
                        remaining -= len(block)
                        now = time.monotonic()
                        if now - last_print >= 2:
                            rate = (offset - starting_offset) / max(now - started, 0.001) / 1048576
                            print(f"  {100 * offset / total:5.1f}%  {offset / 1048576:.2f}/{total / 1048576:.2f} MiB  {rate:.2f} MiB/s", flush=True)
                            last_print = now
            failures = 0
        except Exception as error:
            offset = partial.stat().st_size if partial.exists() else 0
            failures = 0 if offset > start else failures + 1
            print(f"Transfer interrupted at {offset / 1048576:.2f} MiB: {error}; retrying from this position", flush=True)
            if failures >= 8:
                raise RuntimeError(f"8 retries without progress. Rerun later; {partial} is retained.") from error
            time.sleep(2)
    if partial.stat().st_size != total:
        raise RuntimeError("Downloaded size does not match source")
    read(partial)
    partial.replace(path)
    print(f"Download complete and JSON validated: {path.name} ({total / 1048576:.2f} MiB)", flush=True)


def emotion(turn):
    """The author's loader uses adjudicated annotation at emotion[3]."""
    value = turn["emotion"][3]["emotion"]
    value = int(value)
    if value not in range(7):
        raise ValueError(f"Unexpected emotion label {value}")
    return value


def make_rows(dialogue_id, dialogue, domain, history_turns):
    turns = dialogue["log"]
    if len(turns) % 2:
        raise ValueError(f"Unexpected odd turn count: {dialogue_id}")
    rows = []
    # Input ends at system t-1; target is user t. First user has no prior state.
    for target in range(2, len(turns), 2):
        history = turns[max(0, target - history_turns):target]
        start = max(0, target - history_turns)
        text = "\n".join(("User: " if (start + j) % 2 == 0 else "System: ") + t["text"].strip()
                         for j, t in enumerate(history))
        if not text or not turns[target]["text"].strip():
            continue
        rows.append({"dialogue_id": dialogue_id, "domain": domain, "target_turn": target,
                     "history": text, "last_user": turns[target - 2]["text"].strip(),
                     "future_text": turns[target]["text"].strip(),
                     "previous_label": emotion(turns[target - 2]), "label": emotion(turns[target])})
    return rows


def prepare(args):
    raw = ROOT / "data" / "raw"
    for name in FILES:
        download(raw / name)
    dialogues = {}
    domain_for = {}
    for name, domain in zip(FILES[:2], ["multiwoz", "dialmage"]):
        data = read(raw / name)
        if set(dialogues).intersection(data):
            raise ValueError("Duplicate dialogue IDs between source domains")
        dialogues.update(data)
        domain_for.update({k: domain for k in data})
    split = read(raw / FILES[2])
    ids_by_split = {}
    for name, key in [("train", "train"), ("validation", "dev"), ("test", "test")]:
        ids = list(dict.fromkeys(split[key]["multiwoz"] + split[key]["dialmage"]))
        if args.max_dialogues:
            rng = random.Random(args.data_seed + len(name))
            rng.shuffle(ids)
            ids = ids[:args.max_dialogues]
        ids_by_split[name] = ids
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if set(ids_by_split[a]) & set(ids_by_split[b]):
            raise ValueError(f"Dialogue leakage: {a}/{b}")
    args.out.mkdir(parents=True, exist_ok=True)
    stats = {"labels": LABELS, "history_turns": args.history_turns,
             "max_dialogues_per_split": args.max_dialogues, "data_seed": args.data_seed,
             "source": SOURCE, "source_sha256": {n: digest(raw / n) for n in FILES}, "splits": {}}
    for name, ids in ids_by_split.items():
        rows = []
        for key in ids:
            rows.extend(make_rows(key, dialogues[key], domain_for[key], args.history_turns))
        if not rows:
            raise ValueError(f"No valid rows in {name}")
        dump(args.out / f"{name}.json", rows)
        counts = Counter(r["label"] for r in rows)
        stats["splits"][name] = {"dialogues": len(ids), "samples": len(rows),
                                 "class_counts": {LABELS[i]: counts[i] for i in range(7)},
                                 "domain_counts": dict(Counter(r["domain"] for r in rows))}
        print(f"{name}: {len(ids)} dialogues, {len(rows)} prediction samples", flush=True)
    dump(args.out / "data_stats.json", stats)


def encode(args):
    from sentence_transformers import SentenceTransformer
    started = time.perf_counter()
    paths = [args.out / f"{name}.json" for name in ["train", "validation", "test"]]
    signature = {"model": args.model, "max_tokens": args.max_tokens,
                 "truncation_side": "left", "sha256": {p.name: digest(p) for p in paths}}
    cache = args.out / "embeddings.npz"
    meta_path = args.out / "embedding_meta.json"
    if cache.exists() and meta_path.exists() and read(meta_path)["signature"] == signature:
        print("Reusing matching embedding cache", flush=True)
        return
    encoder = SentenceTransformer(args.model, device=args.device, cache_folder=str(ROOT / "data" / "models"))
    encoder.max_seq_length = args.max_tokens
    encoder.tokenizer.truncation_side = "left"  # Preserve the newest turns.
    encoder.eval()
    if args.device == "cuda":
        encoder.half()
    arrays = {}
    for name in ["train", "validation", "test"]:
        split_cache = args.out / f"encoded_{name}.npz"
        split_meta = args.out / f"encoded_{name}.meta.json"
        if split_cache.exists() and split_meta.exists() and read(split_meta) == signature:
            arrays.update(dict(np.load(split_cache, allow_pickle=False)))
            print(f"Reusing completed {name} embedding cache", flush=True)
            continue
        rows = read(args.out / f"{name}.json")
        arrays[name + "_y"] = np.array([r["label"] for r in rows], dtype=np.int64)
        arrays[name + "_prev"] = np.array([r["previous_label"] for r in rows], dtype=np.int64)
        for field, suffix in [("history", "x"), ("last_user", "last"), ("future_text", "target")]:
            # Exact-text deduplication reduces repeated encoding; not cross-split learning.
            texts = [r[field] for r in rows]
            unique = list(dict.fromkeys(texts))
            mapping = {t: i for i, t in enumerate(unique)}
            print(f"Encoding {name}/{field}: {len(unique)} unique texts", flush=True)
            vectors = encoder.encode(unique, batch_size=args.encode_batch,
                                     convert_to_numpy=True, normalize_embeddings=True,
                                     show_progress_bar=True).astype(np.float32)
            if not np.isfinite(vectors).all():
                raise RuntimeError("Non-finite embeddings; try --device cpu or disable encoder.half()")
            arrays[name + "_" + suffix] = vectors[[mapping[t] for t in texts]]
        # Resume after an interrupted run with already encoded splits.
        np.savez(split_cache, **{k: v for k, v in arrays.items() if k.startswith(name + "_")})
        dump(split_meta, signature)
    np.savez(cache, **arrays)
    del encoder
    if args.device == "cuda":
        torch.cuda.empty_cache()
    dump(meta_path, {"signature": signature, "seconds": time.perf_counter() - started,
                     "dimension": int(arrays["train_x"].shape[1]), "frozen_encoder": True})


class Predictor(nn.Module):
    def __init__(self, dimension, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(0.1))
        self.classifier = nn.Linear(hidden, 7)
        self.future = nn.Linear(hidden, dimension)

    def forward(self, x):
        z = self.trunk(x)
        return self.classifier(z), F.normalize(self.future(z), dim=-1)


def metrics(y, prediction):
    return {"accuracy": float(accuracy_score(y, prediction)),
            "macro_f1": float(f1_score(y, prediction, labels=list(range(7)), average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, prediction, labels=list(range(7)), average="weighted", zero_division=0))}


@torch.no_grad()
def infer(model, x, batch=1024):
    model.eval()
    logits, vectors = [], []
    for i in range(0, len(x), batch):
        a, b = model(x[i:i + batch])
        logits.append(a.cpu())
        vectors.append(b.cpu())
    return torch.cat(logits).numpy(), torch.cat(vectors).numpy()


def retrieval(predicted, targets, rows, seed=2026):
    # Deduplicate identical future utterances. This is a sampled held-out retrieval
    # diagnostic (not Recall over the full corpus and not a generation metric).
    first = {}
    for i, row in enumerate(rows):
        first.setdefault(row["future_text"].strip().lower(), i)
    candidates = list(first.values())
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    candidates = np.array(candidates[:1000])
    queries = rng.permutation(len(candidates))[:min(500, len(candidates))]
    sim = predicted[candidates[queries]] @ targets[candidates].T
    ranked = np.argsort(-sim, axis=1)
    correct = queries[:, None]
    return {"candidate_count": len(candidates), "query_count": len(queries),
            "recall_at_1": float(np.mean(np.any(ranked[:, :1] == correct, axis=1))),
            "recall_at_5": float(np.mean(np.any(ranked[:, :5] == correct, axis=1))),
            "recall_at_10": float(np.mean(np.any(ranked[:, :10] == correct, axis=1)))}


def save_evaluation(folder, y, predictions, rows):
    dump(folder / "classification_report.json", classification_report(
        y, predictions, labels=list(range(7)), target_names=LABELS, zero_division=0, output_dict=True))
    dump(folder / "confusion_matrix.json", confusion_matrix(y, predictions, labels=list(range(7))).tolist())
    records = []
    for r, predicted in zip(rows, predictions):
        records.append({**r, "true_emotion": LABELS[r["label"]], "predicted_emotion": LABELS[int(predicted)]})
    with (folder / "test_predictions.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    dump(folder / "error_examples.json", [r for r in records if r["true_emotion"] != r["predicted_emotion"]][:30])


def train_one(args, data, rows, kind, seed):
    seed_all(seed)
    folder = args.out / "models" / f"{kind}_seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    field = "last" if kind == "last_user_classifier" else "x"
    x = {s: torch.from_numpy(data[s + "_" + field]).to(args.device) for s in ["train", "validation", "test"]}
    y = torch.from_numpy(data["train_y"]).to(args.device)
    target = torch.from_numpy(data["train_target"]).to(args.device)
    counts = np.bincount(data["train_y"], minlength=7)
    if np.any(counts == 0):
        print("WARNING: training split has missing classes; all 7 still enter Macro-F1", flush=True)
    weights = np.sqrt(len(y) / (7 * np.maximum(counts, 1))).clip(0.25, 5)
    weights = torch.tensor(weights, dtype=torch.float32, device=args.device)
    model = Predictor(x["train"].shape[1]).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    best_score = -1.0
    best_epoch = 0
    history = []
    best_state = None
    semantic_weight = args.semantic_weight if kind == "jepa" else 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(y), device=args.device)
        sums = np.zeros(3)
        for ids in order.split(args.train_batch):
            optimizer.zero_grad(set_to_none=True)
            logits, prediction = model(x["train"][ids])
            classification = F.cross_entropy(logits, y[ids], weight=weights)
            cosine = (1 - (prediction * target[ids]).sum(dim=1)).mean()
            # Frozen, nonconstant targets anchor the representation. Match batch
            # variation to the target as an additional anti-collapse diagnostic.
            variance = F.mse_loss(prediction.std(dim=0, unbiased=False), target[ids].std(dim=0, unbiased=False))
            semantic = cosine + 0.1 * variance
            loss = classification + semantic_weight * semantic
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sums += np.array([loss.item(), classification.item(), semantic.item()]) * len(ids)
        val_logits, _ = infer(model, x["validation"])
        score = metrics(data["validation_y"], val_logits.argmax(axis=1))["macro_f1"]
        item = {"epoch": epoch, "loss": sums[0] / len(y), "classification_loss": sums[1] / len(y),
                "semantic_loss": sums[2] / len(y), "validation_macro_f1": score}
        history.append(item)
        print(f"{kind} seed={seed} epoch={epoch} loss={item['loss']:.4f} val_macro_f1={score:.4f}", flush=True)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= args.patience:
            print("Early stopping (validation only)", flush=True)
            break
    model.load_state_dict(best_state)
    test_logits, predicted_vectors = infer(model, x["test"])
    predictions = test_logits.argmax(axis=1)
    result = {"model": kind, "seed": seed, "best_epoch": best_epoch,
              "validation_macro_f1": best_score, **metrics(data["test_y"], predictions),
              "seconds": time.perf_counter() - started,
              "test_selection_used": False, "semantic_weight": semantic_weight,
              "class_weights": weights.cpu().tolist()}
    if kind == "jepa":
        result["retrieval"] = retrieval(predicted_vectors, data["test_target"], rows)
        result["predicted_embedding_std_mean"] = float(predicted_vectors.std(axis=0).mean())
        result["target_embedding_std_mean"] = float(data["test_target"].std(axis=0).mean())
    torch.save({"state_dict": best_state, "dimension": x["train"].shape[1],
                "config": vars(args) | {"out": str(args.out)}, "labels": LABELS}, folder / "best.pt")
    dump(folder / "history.json", history)
    dump(folder / "result.json", result)
    save_evaluation(folder, data["test_y"], predictions, rows)
    print(f"TEST {kind} seed={seed}: {result}", flush=True)
    return result


def train(args):
    info = check(args)
    dump(args.out / "environment.json", info)
    signature = read(args.out / "embedding_meta.json")["signature"]
    for name in ["train", "validation", "test"]:
        path = args.out / f"{name}.json"
        if signature["sha256"][path.name] != digest(path):
            raise RuntimeError("Data changed after encoding. Rerun encode before train.")
    data = dict(np.load(args.out / "embeddings.npz", allow_pickle=False))
    rows = read(args.out / "test.json")
    majority = int(np.bincount(data["train_y"], minlength=7).argmax())
    baseline = [{"model": "majority", "seed": None,
                 **metrics(data["test_y"], np.full(len(rows), majority))},
                {"model": "persistence_oracle", "seed": None,
                 "note": "Uses gold current-user emotion; diagnostic only, not deployable.",
                 **metrics(data["test_y"], data["test_prev"])}]
    retrieval_baselines = {
        "copy_last_user_embedding": retrieval(data["test_last"], data["test_target"], rows),
        "train_mean_target_embedding": retrieval(
            np.broadcast_to(data["train_target"].mean(axis=0), data["test_target"].shape),
            data["test_target"], rows)}
    dump(args.out / "retrieval_baselines.json", retrieval_baselines)
    results = list(baseline)
    dump(args.out / "results.json", results)
    dump(args.out / "run_config.json", vars(args) | {"out": str(args.out), "status": "running"})
    for seed in args.seeds:
        for kind in ["last_user_classifier", "history_classifier", "jepa"]:
            results.append(train_one(args, data, rows, kind, seed))
            dump(args.out / "results.json", results)
    dump(args.out / "run_config.json", vars(args) | {"out": str(args.out), "status": "completed"})


def summarize(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    results = read(args.out / "results.json")
    stats = read(args.out / "data_stats.json")
    config = read(args.out / "run_config.json")
    if config["status"] != "completed":
        raise RuntimeError("Training incomplete. Finish train before creating a report.")
    summaries = []
    for name in ["majority", "persistence_oracle", "last_user_classifier", "history_classifier", "jepa"]:
        group = [r for r in results if r["model"] == name]
        if not group:
            raise RuntimeError(f"Missing results for {name}")
        row = {"model": name, "runs": len(group)}
        for key in ["accuracy", "macro_f1", "weighted_f1"]:
            values = [r[key] for r in group]
            row[key + "_mean"] = float(np.mean(values))
            row[key + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summaries.append(row)
    dump(args.out / "summary.json", summaries)
    with (args.out / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0]))
        w.writeheader()
        w.writerows(summaries)
    figures = args.out / "figures"
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    counts = stats["splits"]["train"]["class_counts"]
    ax.bar(LABELS, [counts[x] for x in LABELS])
    ax.set_ylabel("Training prediction samples")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(figures / "class_distribution.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([r["model"] for r in summaries], [r["macro_f1_mean"] for r in summaries],
           yerr=[r["macro_f1_std"] for r in summaries], capsize=4)
    ax.set_ylabel("Test Macro-F1 (all 7 classes)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figures / "macro_f1.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in ["last_user_classifier", "history_classifier", "jepa"]:
        hist = read(args.out / "models" / f"{name}_seed{config['seeds'][0]}" / "history.json")
        ax.plot([x["epoch"] for x in hist], [x["validation_macro_f1"] for x in hist], label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Macro-F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "validation_curve.png", dpi=180)
    plt.close(fig)
    text_rows = ["| 模型 | 测试Macro-F1 | 测试Accuracy |", "|---|---:|---:|"]
    for r in summaries:
        text_rows.append(f"| {r['model']} | {r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f} | {r['accuracy_mean']:.4f} |")
    table = "\n".join(text_rows)
    (args.out / "results_for_report.md").write_text(table + "\n\n情绪保持基线使用当前真实标签，仅作诊断参照。\n", encoding="utf-8")
    history = next(r for r in summaries if r["model"] == "history_classifier")
    jepa = next(r for r in summaries if r["model"] == "jepa")
    last = next(r for r in summaries if r["model"] == "last_user_classifier")
    delta = (jepa["macro_f1_mean"] - history["macro_f1_mean"]) * 100
    finding = (f"JEPA比历史分类器高{delta:.2f}个百分点，但还需更多种子和消融确认稳定性。"
               if delta > 0 else f"JEPA比历史分类器低{abs(delta):.2f}个百分点，当前实验未显示辅助目标的收益。")
    n = stats["splits"]
    scope = "完整官方划分" if not stats["max_dialogues_per_split"] else f"每个划分最多{stats['max_dialogues_per_split']}段对话的抽样实验"
    report = (
        "# 自然语言处理综合实践第二次项目进展报告\n\n"
        "项目题目：基于联合嵌入预测架构的多轮对话情感世界模型\n\n"
        "姓名：________　学号：________　日期：2026年10月9日\n\n"
        "## 一 本阶段目标\n\n"
        "本阶段将第一次报告的技术方案推进到可运行实验，重点验证历史对话能否预测下一轮用户情绪，以及预测未来话语语义表示是否有助于分类。"
        "JEPA通过表征空间预测学习高层语义[1]，EmoWOZ提供任务型对话情绪标注[2]，对话世界模型研究则为未来用户状态预测提供任务依据[3]。\n\n"
        "## 二 已完成的数据处理与实现\n\n"
        f"已读取作者发布的EmoWOZ原始JSON，沿用官方对话级训练、验证和测试划分。本次采用{scope}，分别构造{n['train']['samples']}、"
        f"{n['validation']['samples']}和{n['test']['samples']}个未来情绪预测样本。输入为最近{stats['history_turns']}个话语轮次，包含当前系统回复，"
        "目标为下一轮用户情绪和话语语义向量。首轮用户话语因缺少历史而不作预测目标；检查了划分之间的对话ID互斥，未来话语只作为训练监督。"
        "针对计算资源，采用冻结MiniLM句向量编码器的轻量原型，训练共享预测网络和情绪分类头。分类损失使用训练集频率计算类别权重，"
        "JEPA辅助损失使用未来向量的余弦距离和方差匹配。该实现尚未端到端更新编码器。\n\n"
        "## 三 阶段实验结果\n\n"
        f"已完成多数类、情绪保持、最后用户话语分类、历史分类及JEPA联合模型的对照，训练模型使用{len(config['seeds'])}个随机种子。"
        "只用验证集宏平均F1选取检查点，测试集用于最终评估；宏平均覆盖全部七类。"
        f"最后话语、历史分类及JEPA的测试宏平均F1分别为{last['macro_f1_mean']:.4f}、{history['macro_f1_mean']:.4f}和{jepa['macro_f1_mean']:.4f}。"
        f"{finding}情绪保持基线使用当前真实情绪标签，属于诊断参照，不能视为直接部署结果。"
        "另外导出了类别混淆矩阵、错误案例和训练曲线，并在固定抽样候选池上评估未来语义检索。\n\n"
        "## 四 问题与后续安排\n\n"
        "当前主要问题是情绪类别不平衡，以及相同历史可能对应多种未来表达。冻结编码器能降低训练成本，但限制了语义表示对预测任务的适配。"
        "本次结果属于单一数据集上的阶段验证，不能据此推断实际交互效果。下一阶段将分析少数类混淆，比较不同历史长度与语义损失权重，"
        "开展跨域测试及部分编码器微调，并完善演示系统、错误分析和最终报告，计划于10月18日前完成。\n\n"
        "## 参考文献\n\n"
        "[1] Huang H, LeCun Y, Balestriero R. LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures[EB/OL]. arXiv:2509.14252, 2025.\n\n"
        "[2] Feng S, Lubis N, Geishauser C, et al. EmoWOZ: A Large-Scale Corpus and Labelling Scheme for Emotion Recognition in Task-Oriented Dialogue Systems[C]//LREC. 2022: 4096-4113.\n\n"
        "[3] Zhao Y, Wang X, Wang D, et al. Dream to Chat: Model-based Reinforcement Learning on Dialogues with User Belief Modeling[C]//Findings of EMNLP. 2025: 4764-4781.\n"
    )
    (args.out / "第二次报告草稿.md").write_text(report, encoding="utf-8")
    print(table, flush=True)
    print(f"\nResults and report draft: {args.out}", flush=True)


def smoke(args):
    seed_all(42)
    def turn(text, label=None):
        return {"text": text, "emotion": [{}, {}, {}, {"emotion": label}]} if label is not None else {"text": text}
    dialogue = {"log": [turn("I need a hotel", 0), turn("Which area?"),
                        turn("In the west", 2), turn("There are two hotels"),
                        turn("Thanks", 6), turn("Welcome")]}
    rows = make_rows("synthetic", dialogue, "synthetic", 6)
    assert len(rows) == 2 and rows[0]["label"] == 2 and rows[0]["previous_label"] == 0
    assert "In the west" not in rows[0]["history"]
    assert rows[0]["history"].endswith("System: Which area?")
    assert rows[1]["history"].endswith("System: There are two hotels")
    model = Predictor(24).to(args.device)
    x = torch.randn(28, 24, device=args.device)
    targets = F.normalize(torch.randn_like(x), dim=-1)
    logits, future = model(x)
    loss = F.cross_entropy(logits, torch.arange(28, device=args.device) % 7) + (1 - (future * targets).sum(-1)).mean()
    loss.backward()
    assert model.trunk[0].weight.grad is not None and torch.isfinite(loss)
    torch.optim.AdamW(model.parameters(), lr=1e-3).step()
    result = metrics(np.arange(7), np.arange(7))
    assert result["macro_f1"] == 1
    r = retrieval(np.eye(7), np.eye(7), [{"future_text": str(i)} for i in range(7)])
    assert r["recall_at_1"] == 1.0
    print("PASS: temporal alignment, future-input exclusion, seven-class metrics, retrieval, CUDA backward.")
    print("Synthetic diagnostic only. No real experiment metrics created.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["check", "smoke", "prepare", "encode", "train", "summarize", "all"])
    p.add_argument("--out", type=Path, default=ROOT / "runs" / "full")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--history-turns", type=int, default=6)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--max-dialogues", type=int, default=0, help="0 uses all official dialogues; otherwise cap each split")
    p.add_argument("--data-seed", type=int, default=2026)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--encode-batch", type=int, default=128)
    p.add_argument("--train-batch", type=int, default=512)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--semantic-weight", type=float, default=0.5)
    args = p.parse_args()
    args.out = args.out.resolve()
    if min(args.history_turns, args.max_tokens, args.encode_batch, args.train_batch, args.epochs, args.patience) < 1:
        p.error("Lengths, batch sizes, epochs and patience must be positive")
    if args.max_dialogues < 0 or len(set(args.seeds)) != len(args.seeds):
        p.error("Invalid max-dialogues or duplicate seeds")
    started = time.perf_counter()
    if args.stage == "all":
        check(args)
        prepare(args)
        encode(args)
        train(args)
        summarize(args)
    else:
        globals()[args.stage](args)
    print(f"Finished {args.stage} in {(time.perf_counter() - started) / 60:.2f} minutes", flush=True)


if __name__ == "__main__":
    main()
