"""CPU-only audit of existing predictions; no training, selection, or GPU access.

Bootstrap/permutation units are whole dialogues. A permutation exchanges all
turns and all matched-seed predictions of a dialogue together. The inferential
target is the mean metric of three fixed trained runs, not a seed ensemble.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
LABELS = ["neutral", "fearful", "dissatisfied", "apologetic", "abusive", "excited", "satisfied"]
SEEDS = [42, 43, 44]
STAGES = ["full", "controlled_v2", "posthoc_diagnostic", "encoder_minilm_v1", "emotion_kd_v1", "role_interaction_v1"]
FIELDS = ["dialogue_id", "target_turn", "domain", "history", "last_user", "future_text", "label", "previous_label"]
METRICS = ["macro_f1", "accuracy", "negative_f1"] + ["f1_" + label for label in LABELS]
C = "controlled_v2/"
E = "encoder_minilm_v1/"
P = "posthoc_diagnostic/"
R = "role_interaction_v1/"
CONTRASTS = [
    ("全量微调减冻结", E + "minilm_full", E + "minilm_frozen", "同轮任务适配"),
    ("最后两层微调减冻结", E + "minilm_last2", E + "minilm_frozen", "同轮任务适配"),
    ("全量微调减最后两层", E + "minilm_full", E + "minilm_last2", "解冻范围"),
    ("原JEPA减历史CE", C + "jepa_baseline", C + "history_classifier", "冻结辅助目标"),
    ("多正例JEPA减历史CE", C + "jepa_w1_0", C + "history_classifier", "冻结辅助目标"),
    ("Focal减历史CE", C + "route1_focal_only", C + "history_classifier", "分类损失"),
    ("多正例JEPA减原JEPA", C + "jepa_w1_0", C + "jepa_baseline", "语义路线"),
    ("目标单句识别减历史预测", P + "posthoc_target_only_ce", C + "history_classifier", "信息条件和输入结构均不同的诊断"),
    ("历史加未来减历史重复", P + "posthoc_history_target_ce", P + "control_history_duplicate_ce", "同维度信息条件诊断"),
    ("KD减全量CE", "emotion_kd_v1/minilm_full_kd_0p3", E + "minilm_full", "蒸馏"),
    ("同容量全局头减原CE", R + "global_additive", E + "minilm_full", "分类头结构"),
    ("角色加性减同容量全局", R + "role_additive", R + "global_additive", "角色池化"),
    ("FiLM减同容量全局", R + "role_film", R + "global_additive", "角色与交互联合变化"),
    ("FiLM减角色加性", R + "role_film", R + "role_additive", "同输入交互形式"),
    ("角色加性减原CE", R + "role_additive", E + "minilm_full", "已按验证选择的角色方案"),
    ("FiLM减原CE", R + "role_film", E + "minilm_full", "相对原基线"),
]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def divide(a, b):
    a, b = np.broadcast_arrays(a, b)
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=b != 0)


def metric(s):
    """24 sufficient statistics: true7, predicted7, TP7, negative true/pred/TP."""
    t, p, tp = s[..., :7], s[..., 7:14], s[..., 14:21]
    f1 = divide(2 * tp, t + p)
    accuracy = divide(tp.sum(axis=-1), t.sum(axis=-1))
    negative = divide(2 * s[..., 23], s[..., 21] + s[..., 22])
    return np.concatenate([f1.mean(axis=-1)[..., None], accuracy[..., None], negative[..., None], f1], axis=-1)


def sufficient(truth, pred, groups, n_groups):
    stats = np.zeros((n_groups, 24), dtype=np.float64)
    for label in range(7):
        stats[:, label] = np.bincount(groups, weights=truth == label, minlength=n_groups)
        stats[:, 7 + label] = np.bincount(groups, weights=pred == label, minlength=n_groups)
        stats[:, 14 + label] = np.bincount(groups, weights=(truth == label) & (pred == label), minlength=n_groups)
    tn, pn = np.isin(truth, [1, 2, 4]), np.isin(pred, [1, 2, 4])
    for k, values in enumerate([tn, pn, tn & pn]):
        stats[:, 21 + k] = np.bincount(groups, weights=values, minlength=n_groups)
    return stats


def interval(values):
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def holm(p):
    order = np.argsort(p)
    result = np.empty(len(p))
    adjusted = np.maximum.accumulate(np.array(p)[order] * (len(p) - np.arange(len(p))))
    result[order] = np.minimum(adjusted, 1)
    return result.tolist()


class Inputs:
    def __init__(self):
        self.files = {}

    def track(self, path):
        path = Path(path).resolve()
        info = {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns}
        self.files[str(path)] = info
        return info["sha256"]

    def verify_unchanged(self):
        for record in self.files.values():
            path = Path(record["path"])
            if path.stat().st_mtime_ns != record["mtime_ns"] or sha(path) != record["sha256"]:
                raise ValueError("Input modified: " + str(path))


def data_audit(inputs):
    sources = {}
    raw_ids = {}
    for domain in ["multiwoz", "dialmage"]:
        path = ROOT / "data/raw" / f"emowoz-{domain}.json"
        inputs.track(path)
        raw_ids[domain] = set(read(path))
        if set(sources) & raw_ids[domain]:
            raise ValueError("Duplicate IDs across raw data sources")
        sources.update({key: domain for key in raw_ids[domain]})
    split_path = ROOT / "data/raw/data-split.json"
    inputs.track(split_path)
    official = read(split_path)
    all_rows, output, rare_dialogues, split_ids = {}, {}, [], {}
    for split, official_name in [("train", "train"), ("validation", "dev"), ("test", "test")]:
        path = ROOT / "runs/full" / f"{split}.json"
        inputs.track(path)
        rows = read(path)
        expected = set()
        for domain in ["multiwoz", "dialmage"]:
            ids = official[official_name][domain]
            if not set(ids) <= raw_ids[domain]:
                raise ValueError("Official split source mismatch")
            expected.update(ids)
        split_ids[split] = expected
        keys = [(r["dialogue_id"], r["target_turn"]) for r in rows]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate sample IDs")
        if any(r["dialogue_id"] not in expected or sources[r["dialogue_id"]] != r["domain"] for r in rows):
            raise ValueError("Prepared row source/split mismatch")
        all_rows[split] = rows
        output[split] = support(rows, expected)
        for label in [1, 3, 4, 5]:
            counts = Counter(r["dialogue_id"] for r in rows if r["label"] == label)
            for key, count in sorted(counts.items()):
                rare_dialogues.append({"split": split, "label": LABELS[label], "domain": sources[key],
                                       "dialogue_id": key, "positive_turns": count})
    for a, b in itertools.combinations(split_ids, 2):
        if split_ids[a] & split_ids[b]:
            raise ValueError("Official dialogue split leakage")
    merged = [row for rows in all_rows.values() for row in rows]
    output["all"] = support(merged, set(sources))
    output["split_dialogue_overlap"] = False
    output["source_verified_against_raw_files"] = True
    return all_rows, output, rare_dialogues


def support(rows, allocated):
    out = {"samples": len(rows), "allocated_dialogues": len(allocated),
           "effective_dialogues": len({r["dialogue_id"] for r in rows}), "by_source": {}}
    for domain in ["all", "multiwoz", "dialmage"]:
        subset = rows if domain == "all" else [r for r in rows if r["domain"] == domain]
        per_class = {}
        for label, name in enumerate(LABELS):
            counts = Counter(r["dialogue_id"] for r in subset if r["label"] == label)
            n = sum(counts.values())
            per_class[name] = {"turns": n, "positive_dialogues": len(counts),
                               "max_positive_turns_one_dialogue": max(counts.values(), default=0),
                               "top5_dialogue_share": sum(sorted(counts.values(), reverse=True)[:5]) / n if n else None,
                               "concentration_equivalent_dialogues": n * n / sum(x * x for x in counts.values()) if n else 0}
        out["by_source"][domain] = {"samples": len(subset), "dialogues": len({r["dialogue_id"] for r in subset}),
                                      "class_counts": per_class}
    return out


def load_models(rows, inputs):
    truth = np.array([row["label"] for row in rows])
    ids = sorted({r["dialogue_id"] for r in rows})
    index = {key: i for i, key in enumerate(ids)}
    groups = np.array([index[r["dialogue_id"]] for r in rows])
    domains = {r["dialogue_id"]: r["domain"] for r in rows}
    domain_masks = {domain: np.array([domains[key] == domain for key in ids]) for domain in ["multiwoz", "dialmage"]}
    models, stats, metadata, verification = [], [], [], []
    prediction_signatures = defaultdict(list)
    for stage in STAGES:
        stage_path = ROOT / "runs" / stage
        stage_config = stage_path / "run_config.json"
        if not stage_config.exists():
            stage_config = stage_path / "provenance.json"
        inputs.track(stage_config)
        status_path = stage_path / "run_status.json"
        if status_path.exists():
            inputs.track(status_path)
            if read(status_path).get("status") != "completed":
                raise ValueError("Stage is not completed: " + stage)
        candidates = defaultdict(dict)
        for path in sorted((stage_path / "models").glob("*/test_predictions.jsonl")):
            match = re.fullmatch(r"(.+)_seed(\d+)", path.parent.name)
            if not match:
                raise ValueError("Unexpected prediction path: " + str(path))
            name, seed = match.group(1), int(match.group(2))
            candidates[name][seed] = path
        for name, seed_paths in sorted(candidates.items()):
            if sorted(seed_paths) != SEEDS:
                raise ValueError("Expected complete three-seed test group: " + name)
            model_name = stage + "/" + name
            model_stats = []
            for seed in SEEDS:
                path = seed_paths[seed]
                checksum = inputs.track(path)
                marker = path.parent / ("test_completion.json" if (path.parent / "test_completion.json").exists() else "completion.json")
                if marker.exists():
                    inputs.track(marker)
                    completed = read(marker)
                    if completed["status"] != "completed" or completed["artifacts"][path.name] != checksum:
                        raise ValueError("Prediction completion hash mismatch")
                elif stage != "full":
                    raise ValueError("Missing completion marker")
                else:
                    if read(stage_path / "run_config.json").get("status") != "completed":
                        raise ValueError("Original run not completed")
                result_path = path.parent / ("test_result.json" if (path.parent / "test_result.json").exists() else "result.json")
                result_hash = inputs.track(result_path)
                if marker.exists() and read(marker)["artifacts"].get(result_path.name) != result_hash:
                    raise ValueError("Result completion hash mismatch")
                result = read(result_path)
                expected = result["test"] if "test" in result else result["scores"]["test"] if "scores" in result else result
                predictions = []
                with path.open(encoding="utf-8") as stream:
                    for i, line in enumerate(stream):
                        record = json.loads(line)
                        if i >= len(rows) or any(record[k] != rows[i][k] for k in FIELDS):
                            raise ValueError("Misaligned test sample: " + str(path))
                        if record["true_emotion"] != LABELS[rows[i]["label"]]:
                            raise ValueError("Label vocabulary mismatch")
                        predictions.append(LABELS.index(record["predicted_emotion"]))
                if len(predictions) != len(rows):
                    raise ValueError("Prediction length mismatch")
                pred = np.array(predictions)
                counts = sufficient(truth, pred, groups, len(ids))
                scores = metric(counts.sum(axis=0))
                for k, value in [("macro_f1", scores[0]), ("accuracy", scores[1])]:
                    if abs(value - expected[k]) > 1e-12:
                        raise ValueError("Recomputed result mismatch: " + str(path))
                model_stats.append(counts)
                verification.append({"model": model_name, "seed": seed, "samples": len(rows),
                                     "macro_f1": float(scores[0]), "accuracy": float(scores[1]),
                                     "completion_verified": marker.exists()})
                prediction_signatures[hashlib.sha256(pred.astype(np.int8).tobytes()).hexdigest()].append(model_name + f"/seed{seed}")
            models.append(model_name)
            stats.append(np.stack(model_stats, axis=1))
            metadata.append({"model": model_name, "seeds": SEEDS,
                             "information_condition": "posthoc_future_visible" if name.startswith("posthoc_") else "predictive"})
    # Group, model, seed, sufficient-statistic. Rows are never resampled alone.
    return models, np.stack(stats, axis=1), domain_masks, metadata, verification, [v for v in prediction_signatures.values() if len(v) > 1]


def resample(stats, repetitions, rng, batch):
    g, m, s, _ = stats.shape
    fixed = np.empty((repetitions, m, len(METRICS)))
    crossed = np.empty_like(fixed)
    absent = np.zeros(7, dtype=np.int64)
    flat = stats.reshape(g, -1)
    for start in range(0, repetitions, batch):
        n = min(batch, repetitions - start)
        weights = rng.multinomial(g, np.full(g, 1 / g), size=n).astype(np.float64)
        sampled = (weights @ flat).reshape(n, m, s, 24)
        values = metric(sampled)
        fixed[start:start + n] = values.mean(axis=2)
        seed_weights = rng.multinomial(s, np.full(s, 1 / s), size=n) / s
        crossed[start:start + n] = np.einsum("bmsk,bs->bmk", values, seed_weights)
        absent += (sampled[:, 0, 0, :7] == 0).sum(axis=0)
    return fixed, crossed, absent.tolist()


def permutation(stats, pairs, repetitions, rng, batch):
    g = len(stats)
    a = np.stack([stats[:, ia] for ia, _ in pairs], axis=1)
    b = np.stack([stats[:, ib] for _, ib in pairs], axis=1)
    middle = ((a + b) / 2).sum(axis=0)
    delta = (a - b) / 2
    observed = metric(a.sum(axis=0)).mean(axis=1)[:, 0] - metric(b.sum(axis=0)).mean(axis=1)[:, 0]
    extreme = np.zeros(len(pairs), dtype=np.int64)
    for start in range(0, repetitions, batch):
        n = min(batch, repetitions - start)
        signs = rng.integers(0, 2, size=(n, g)) * 2.0 - 1.0
        shift = (signs @ delta.reshape(g, -1)).reshape((n,) + middle.shape)
        shuffled = metric(middle + shift).mean(axis=2)[:, :, 0] - metric(middle - shift).mean(axis=2)[:, :, 0]
        extreme += (np.abs(shuffled) >= np.abs(observed)[None, :] - 1e-12).sum(axis=0)
    return ((extreme + 1) / (repetitions + 1)).tolist()


def model_results(models, stats, fixed, crossed, metadata):
    totals = stats.sum(axis=0)
    seed_scores = metric(totals)
    results = []
    for i, model in enumerate(models):
        classes = {}
        for j, label in enumerate(LABELS):
            classes[label] = {"true_turns": int(totals[i, 0, j]),
                "positive_dialogues": int((stats[:, i, 0, j] > 0).sum()),
                "predicted_turns_by_seed": totals[i, :, 7 + j].astype(int).tolist(),
                "correct_turns_by_seed": totals[i, :, 14 + j].astype(int).tolist(),
                "precision_by_seed": divide(totals[i, :, 14 + j], totals[i, :, 7 + j]).tolist(),
                "recall_by_seed": divide(totals[i, :, 14 + j], totals[i, :, j]).tolist()}
        results.append({**metadata[i], "metrics": {name: {
            "mean": float(seed_scores[i, :, j].mean()), "seed_sample_sd": float(seed_scores[i, :, j].std(ddof=1)),
            "by_seed": seed_scores[i, :, j].tolist(), "dialogue_percentile_ci95": interval(fixed[:, i, j]),
            "dialogue_seed_resampling_sensitivity_ci95": interval(crossed[:, i, j])}
            for j, name in enumerate(METRICS)}, "classes": classes})
    return results


def contrast_results(models, stats, fixed, crossed, pvalues):
    seedscores = metric(stats.sum(axis=0))
    adjusted = holm(pvalues)
    results = []
    for k, (name, a, b, reason) in enumerate(CONTRASTS):
        ia, ib = models.index(a), models.index(b)
        difference = seedscores[ia, :, 0] - seedscores[ib, :, 0]
        signs = np.array(list(itertools.product([-1, 1], repeat=len(SEEDS))))
        seed_p = float(np.mean(np.abs((signs * difference).mean(axis=1)) >= abs(difference.mean()) - 1e-12))
        results.append({"name": name, "a": a, "b": b, "comparison": reason,
            "macro_f1_difference": float(difference.mean()), "paired_seed_differences": difference.tolist(),
            "paired_seed_difference_sample_sd": float(difference.std(ddof=1)),
            "dialogue_percentile_ci95": interval(fixed[:, ia, 0] - fixed[:, ib, 0]),
            "dialogue_seed_resampling_sensitivity_ci95": interval(crossed[:, ia, 0] - crossed[:, ib, 0]),
            "dialogue_permutation_p_two_sided": pvalues[k], "holm_p_within_scope_16": adjusted[k],
            "seed_sign_flip_p_two_sided_descriptive_only": seed_p,
            "other_metric_differences": {name: {"difference": float(seedscores[ia, :, j].mean() - seedscores[ib, :, j].mean()),
                "dialogue_percentile_ci95": interval(fixed[:, ia, j] - fixed[:, ib, j])}
                for j, name in enumerate(METRICS) if j != 0}})
    return results


def pct(x):
    return f"{100 * x:.2f}"


def ci(values):
    return f"[{pct(values[0])}, {pct(values[1])}]"


def report(out, audit, scopes, args, duplicates):
    lines = ["# 现有预测的对话级统计审计", "",
        "本次只分析已有预测，不训练、不重选模型。主分析为官方测试集上三个固定训练种子的平均七类Macro-F1；不是三模型投票或概率集成。", "",
        f"使用{args.bootstrap:,}次对话整组bootstrap与{args.permutations:,}次对话配对随机交换。所有模型共享重采样对话，同一对话所有轮次和三个种子的预测一起交换。", "",
        "## 关键比较", "",
        "差值为前者减后者，单位均为百分点。95%区间仅量化给定三个已训练模型的测试对话抽样不确定性；p值也是此条件下的探索性检验，不是对训练算法整体的显著性证明。", "",
        "| 比较 | 差值 | 对话95%区间 | 对话和种子重采样敏感性区间 | 原始p | Holm校正p |", "|---|---:|---:|---:|---:|---:|"]
    for r in scopes["all"]["contrasts"]:
        lines.append(f"| {r['name']} | {pct(r['macro_f1_difference'])} | {ci(r['dialogue_percentile_ci95'])} | {ci(r['dialogue_seed_resampling_sensitivity_ci95'])} | {r['dialogue_permutation_p_two_sided']:.4f} | {r['holm_p_within_scope_16']:.4f} |")
    lines += ["", "主表Holm校正覆盖上述16个Macro-F1比较。比较名单在本次计算前固定，但既有点估计已被查看，因此不是预注册确认实验；校正不能消除历史反复查看测试集造成的选择偏差。", "",
        "只有三个训练种子，双侧种子符号翻转检验的最小p值为0.25，无法据此在0.05水平建立跨训练随机性的显著性。种子重采样区间只是敏感性分析，不能弥补缺少新种子和训练集重采样。", "",
        "## 少数类对话覆盖", "", "| 划分 | 类别 | 样本轮次 | 含该类的对话 | 单对话最多轮次 | 前5对话占比 |", "|---|---|---:|---:|---:|---:|"]
    for split in ["train", "validation", "test", "all"]:
        for label in ["fearful", "apologetic", "abusive", "excited"]:
            s = audit[split]["by_source"]["all"]["class_counts"][label]
            share = f"{100 * s['top5_dialogue_share']:.1f}%" if s["top5_dialogue_share"] is not None else "无样本"
            lines.append(f"| {split} | {label} | {s['turns']} | {s['positive_dialogues']} | {s['max_positive_turns_one_dialogue']} | {share} |")
    lines += ["", "浓度等效对话数仅为正例轮次集中程度的描述量，不是推导出的有效样本量。五折评估能扩大被评估样本覆盖，但不会创造新对话。", "",
              "## 来源覆盖与成绩", "", "来源由原始文件成员关系核验，不按文件名前缀猜测。以下为同一混合来源训练模型的测试子集分析，不是来源迁移实验。", "",
              "| 来源 | 对话 | 样本 | fearful轮次及对话 | abusive轮次及对话 |", "|---|---:|---:|---:|---:|"]
    for source in ["multiwoz", "dialmage"]:
        s = audit["test"]["by_source"][source]
        f, a = s["class_counts"]["fearful"], s["class_counts"]["abusive"]
        lines.append(f"| {source} | {s['dialogues']} | {s['samples']} | {f['turns']} / {f['positive_dialogues']} | {a['turns']} / {a['positive_dialogues']} |")
    lines += ["", "| 模型 | 全体Macro-F1 | MultiWOZ Macro-F1 | DialMAGE Macro-F1 |", "|---|---:|---:|---:|"]
    for i, model in enumerate(scopes["all"]["models"]):
        values = [scopes[s]["models"][i]["metrics"]["macro_f1"] for s in ["all", "multiwoz", "dialmage"]]
        lines.append(f"| `{model['model']}` | " + " | ".join(pct(v["mean"]) + "% " + ci(v["dialogue_percentile_ci95"]) for v in values) + " |")
    lines += ["", "三栏均固定七类，某来源没有真实样本的类别不能据零F1解释为模型失败；类别缺失、基率和对话难度也使来源分数不能直接用于归因。逐类支持数、命中数及区间见JSON。", "",
        "## 来源内配对比较", "", "| 来源 | 比较 | 差值 | 对话95%区间 | 原始p | 来源32项联合Holm p |", "|---|---|---:|---:|---:|---:|"]
    for source in ["multiwoz", "dialmage"]:
        for r in scopes[source]["contrasts"]:
            lines.append(f"| {source} | {r['name']} | {pct(r['macro_f1_difference'])} | {ci(r['dialogue_percentile_ci95'])} | {r['dialogue_permutation_p_two_sided']:.4f} | {r['holm_p_source_family_32']:.4f} |")
    lines += ["", "来源内区间是否跨零，不等价于来源间差异显著；正式的模型乘来源交互检验需另行设计。", "",
        "## 零命中和区间退化", "",
        "对所有固定模型均未命中的类别，经验bootstrap的F1区间可能退化为[0,0]。这只表示现有预测和样本的重采样无法产生新命中，不能证明总体召回必为零、少数类不可预测或置信度极高。稀有类别未被抽中的比例保存在各来源JSON中。", "",
        "## 可复核文件", "",
        "- `方法说明.md`：统计对象、公式、抽样单位、假设与边界。",
        "- `数据覆盖审计.json`与`少数类对话清单.json`：全划分及来源的支持数和对话分布。",
        "- `统计结果_all.json`、`统计结果_multiwoz.json`、`统计结果_dialmage.json`：全部配置、逐类指标、配对区间与p值。",
        "- `bootstrap分布_*.npz`：固定种子与种子重采样的指标分布，仅包含统计量而非训练权重。",
        "- `输入核验.json`：原始输入哈希、逐条对齐、指标复算及相同预测记录。",
        "- `provenance.json`与`完成核验.json`：脚本、环境、参数及输出校验。", "",
        f"检测到{len(duplicates)}组完全相同的逐种子预测；原始轮次与复用对照均保留，但不视作独立重复实验。", "",
        "本次不生成新种子、交叉验证、功效分析或最小可检测差异，也不根据这些结果重新选择模型。"]
    (out / "统计审计报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "统计稳健性审计_20261015")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=20261015)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if min(args.bootstrap, args.permutations) < 1000 or min(args.batch, args.threads) < 1:
        parser.error("Need at least 1000 repetitions and positive batch/threads")
    out = args.out.resolve()
    # A new direct child of the project only; never write inside existing runs.
    if out.parent != ROOT or out.name in ["runs", "data"]:
        raise ValueError("Output must be a new direct project subfolder")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output is nonempty; use a new directory")
    start = time.perf_counter()
    inputs = Inputs()
    rows, audit, rare_dialogues = data_audit(inputs)
    models, stats, domain_masks, metadata, verified, duplicates = load_models(rows["test"], inputs)
    if len(models) != 28 or len(verified) != 84:
        raise ValueError("Historical audit scope changed; review the 28 configuration plan")
    pairs = [(models.index(a), models.index(b)) for _, a, b, _ in CONTRASTS]
    signature = {"created_at": datetime.now().astimezone().isoformat(), "script_sha256": sha(Path(__file__)),
                 "parameters": {**vars(args), "out": str(out)}, "python": sys.version,
                 "platform": platform.platform(), "numpy": np.__version__, "models": models,
                 "metrics": METRICS, "contrasts": CONTRASTS, "training_started": False}
    # First finish all input checks; then create only the new output folder.
    out.mkdir(exist_ok=True)
    write_json(out / "provenance.json", signature)
    write_json(out / "数据覆盖审计.json", audit)
    write_json(out / "少数类对话清单.json", rare_dialogues)
    scopes = {}
    with threadpool_limits(limits=args.threads):
        for scope_id, name in enumerate(["all", "multiwoz", "dialmage"]):
            subset = stats if name == "all" else stats[domain_masks[name]]
            print(f"Analyzing {name}: {len(subset)} dialogues, {len(models)} models x 3 seeds", flush=True)
            rng_boot = np.random.default_rng(np.random.SeedSequence([args.random_seed, scope_id, 1]))
            rng_perm = np.random.default_rng(np.random.SeedSequence([args.random_seed, scope_id, 2]))
            fixed, crossed, absent = resample(subset, args.bootstrap, rng_boot, args.batch)
            pvalues = permutation(subset, pairs, args.permutations, rng_perm, args.batch)
            scopes[name] = {"scope": name, "dialogues": len(subset), "bootstrap_repetitions": args.bootstrap,
                           "permutation_repetitions": args.permutations,
                           "bootstrap_no_true_class_counts": dict(zip(LABELS, absent)),
                           "models": model_results(models, subset, fixed, crossed, metadata),
                           "contrasts": contrast_results(models, subset, fixed, crossed, pvalues)}
            np.savez_compressed(out / f"bootstrap分布_{name}.npz", fixed_seed_mean=fixed.astype(np.float32),
                                dialogue_seed_sensitivity=crossed.astype(np.float32),
                                models=np.array(models), metrics=np.array(METRICS))
            print(f"Finished {name}: {args.bootstrap} bootstrap and {args.permutations} permutations", flush=True)
    source_p = [r["dialogue_permutation_p_two_sided"] for name in ["multiwoz", "dialmage"] for r in scopes[name]["contrasts"]]
    source_adjusted = holm(source_p)
    for i, name in enumerate(["multiwoz", "dialmage"]):
        for j, result in enumerate(scopes[name]["contrasts"]):
            result["holm_p_source_family_32"] = source_adjusted[i * len(CONTRASTS) + j]
    for name, value in scopes.items():
        write_json(out / f"统计结果_{name}.json", value)
    report(out, audit, scopes, args, duplicates)
    (out / "方法说明.md").write_text(METHODS, encoding="utf-8")
    inputs.verify_unchanged()
    write_json(out / "输入核验.json", {"status": "verified", "original_files_modified": False,
        "samples_per_prediction_file": len(rows["test"]), "prediction_files": len(verified),
        "aligned_sample_predictions": len(verified) * len(rows["test"]),
        "checked_results": verified, "identical_prediction_groups": duplicates,
        "source_files": list(inputs.files.values())})
    artifacts = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    write_json(out / "完成核验.json", {"status": "completed", "training_started": False,
        "original_files_modified": False, "seconds": time.perf_counter() - start,
        "artifacts": artifacts, "self_hash_excluded": True})
    print(json.dumps({"out": str(out), "models": len(models), "prediction_files": len(verified),
                      "seconds": round(time.perf_counter() - start, 2)}, ensure_ascii=False), flush=True)


METHODS = """# 对话级统计审计方法

## 评价对象

官方测试集1100个对话、7534个下一轮用户情绪预测样本。覆盖六个正式结果目录下28种配置，各含42、43、44三个种子的完整测试预测，包括明确标记的事后识别诊断。平均分先在每个种子上计算固定七类Macro-F1，再取三个分数的算术平均；不是合并三个种子的混淆矩阵，也不是集成模型成绩。全量数据覆盖审计读取训练、验证和测试，但模型评价只使用官方测试。

## 对话整组bootstrap

从当前评价范围的G个对话有放回抽取G个对话，保留被抽中对话的全部轮次及重复次数。所有模型和种子使用完全相同的对话权重。对每个重采样副本重新累计逐类TP、预测数与真实数，以2TP/(预测数+真实数)计算F1；零分母记0，固定七类求均值。Accuracy及负面二分类F1也从加权计数重算。区间是10000次副本的2.5%与97.5%分位数，未做BCa偏差修正；配对区间来自每个副本内的模型差，而不是两个区间端点相减。

主区间以已训练的三个固定模型为条件，覆盖测试对话抽样而不覆盖训练数据变化。补充敏感性区间在同一对话副本内，再从三个匹配种子索引有放回抽三个，对所有模型使用同一索引。它反映有限的训练随机性敏感度；只有三个已观察种子，不能当作可靠估计全部训练随机性的保证。

## 配对随机交换和多重比较

在两个模型之间，对每个对话以0.5概率交换其所有轮次、所有匹配种子的预测。每次重新计算三个种子平均Macro-F1之差；以绝对差不小于观察绝对差的次数计算双侧p值：(次数+1)/(10000+1)。这是对话级近似随机化检验，依赖对话之间可作为独立抽样单元及零假设下模型输出可交换的假设；不是自动适用于任意种群的算法级因果检验。来源内也按对话交换。

主分析16个Macro-F1比较使用Holm校正；两个来源的32项分析另作为探索性家族联合校正。其他指标只提供差值和点态区间，不声称经多重性校正的显著性。所有配置和点估计先前已经被查看，本次名单不是预注册，也没有新的未见测试集。校正不能消除此前探索性选型带来的偏差。

三个匹配种子另枚举2^3种整种子符号翻转，报告描述性p值。双侧最小p值0.25；不把大量测试轮次当成新增训练种子。

## 来源分析和稀有类别

来源标记用原始emowoz-multiwoz.json和emowoz-dialmage.json的对话ID成员关系核验，同时检查官方划分与预处理样本、各划分对话不相交。分来源结果是在混合来源训练模型上的子集成绩，不是跨来源迁移，不足以单独解释为模型在某种对话来源上泛化更好。来源内仍固定七类，缺失类别的支持数必须一同阅读。

逐类统计同时报告轮次数、含该类的对话数、单对话最大正例轮次、前5个对话占比，以及(sum n)^2/sum(n^2)描述集中程度。最后一项不是形式化的有效样本量。稀有类完整对话清单不包含原始对话文本。bootstrap缺少某真实类别的次数单独报告。

零正确命中可能使经验F1区间退化为[0,0]，不意味着总体效果确定为零。经验bootstrap无法创造观察中没有出现的正确命中。全数据交叉验证、增加新种子、训练集重采样、功效分析、等效性检验都不属于本轮已完成内容；不以区间宽度冒充最小可检测差异，不以p大于0.05宣称两种方法等效。

## 方法参考

- [Dror等 ACL 2018 NLP统计检验指南](https://aclanthology.org/P18-1128/)提供检验选择背景；对话整组以及匹配种子的安排是针对本项目依赖结构的具体设计。
- [Lakens 2017 等效性检验](https://doi.org/10.1177/1948550617697177)区分未发现差异与支持实际等效；本轮没有执行等效性检验。
"""


if __name__ == "__main__":
    main()
