"""Post-hoc emotion recognition probes. Future-text probes are NOT deployable predictors.

Fixed views: target-only, normalized [history,target], normalized [history,history].
CE/Focal x three reference seeds. Existing predictive checkpoints are evaluated but never retrained.
Only validation Macro-F1 selects checkpoints. This is not a theoretical upper-bound measurement.
"""
import argparse
import importlib.metadata
import json
import math
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

import run_controlled_experiments as common

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ["best.pt", "history.json", "result.json", "test_classification_report.json",
             "test_confusion_matrix.json", "test_predictions.jsonl", "stratified_errors.json"]


def plan():
    return [{"model": f"{view}_{loss}", "view": view, "loss": loss,
             "uses_future_text": view != "control_history_duplicate"}
            for view in ["posthoc_target_only", "posthoc_history_target", "control_history_duplicate"]
            for loss in ["ce", "focal"]]


def features(data, split, view):
    if view == "posthoc_target_only":
        return data[split + "_target"]
    history = data[split + "_x"]
    second = data[split + "_target"] if view == "posthoc_history_target" else history
    if view not in ["posthoc_history_target", "control_history_duplicate"]:
        raise ValueError(f"Unknown input view: {view}")
    # Equal norm and equal dimension for the matched pair; no mean blending that erases information.
    return np.concatenate([history, second], axis=1) / math.sqrt(2)


@torch.no_grad()
def classify(model, inputs):
    model.eval()
    predictions = []
    for batch in inputs.split(1024):
        logits = model.classifier(model.trunk(batch))
        predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions)


def split_scores(data, predictions):
    return {split: {**common.metrics(data[split + "_y"], pred),
                    "per_class": common.detailed_class_diagnostics(data[split + "_y"], pred)}
            for split, pred in predictions.items()}


def reference_records(args):
    provenance = common.read(args.reference / "provenance.json")
    if provenance["code_sha256"] != common.digest(Path(common.__file__)):
        raise ValueError("Controlled helper code differs from the completed reference. Preserve/recover that version first.")
    for name in common.REQUIRED_DATA:
        if common.digest(args.source / name) != provenance["source_sha256"][name]:
            raise ValueError(f"Reference and source data differ: {name}")
    train_config = provenance["training"]
    for key in ["device", "seeds", "train_batch", "epochs", "patience", "lr"]:
        if getattr(args, key) is None:
            setattr(args, key, train_config[key])
        if getattr(args, key) != train_config[key]:
            raise ValueError(f"Diagnostic training setting must match the reference: {key}")
    records = []
    checker_args = Namespace(out=args.reference, run_fingerprint=common.fingerprint(provenance))
    for name in ["history_classifier", "route1_focal_only"]:
        spec = next(s for s in common.experiment_plan() if s["exp_name"] == name)
        for seed in args.seeds:
            folder = args.reference / "models" / f"{name}_seed{seed}"
            # Never call the cache helper on a missing reference: its incomplete-run
            # branch writes a config. Reference paths must remain strictly read-only.
            if not (folder / "completion.json").exists():
                raise ValueError(f"Reference checkpoint incomplete: {folder}")
            result = common.load_completed(checker_args, spec, seed)
            records.append({"model": name, "seed": seed, "result": result,
                            "checkpoint": str(folder / "best.pt"),
                            "checkpoint_sha256": common.digest(folder / "best.pt")})
    return records


def ensure_provenance(args, references):
    common.protect_paths(args.out, args.source)
    common.protect_paths(args.out, args.reference)
    value = {"schema_version": 1, "purpose": "posthoc_diagnostic_only; not theoretical upper bound",
             "probe_code_sha256": common.digest(Path(__file__)),
             "helper_code_sha256": common.digest(Path(common.__file__)),
             "source_sha256": {name: common.digest(args.source / name) for name in common.REQUIRED_DATA},
             "reference_provenance_sha256": common.digest(args.reference / "provenance.json"),
             "reference_checkpoints": [{k: r[k] for k in ["model", "seed", "checkpoint_sha256"]} for r in references],
             "training": {key: getattr(args, key) for key in ["device", "seeds", "train_batch", "epochs", "patience", "lr"]},
             "classification_loss": {"weight_mode": "baseline_sqrt_clip5", "focal_gamma": 1.5,
                                     "denominator": "sum_of_target_class_weights"},
             "plan": plan(), "checkpoint_selection": "validation_macro_f1_only",
             "environment": {"python": sys.version, **{p: importlib.metadata.version(p) for p in ["torch", "numpy", "scikit-learn"]}},
             "test_context": "exploratory followup; original prediction test metrics already inspected"}
    path = args.out / "provenance.json"
    if path.exists():
        if common.read(path) != value:
            raise ValueError("Diagnostic provenance/config changed. Use a new output directory.")
    elif args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Nonempty diagnostic output has no manifest; use a new directory.")
    args.out.mkdir(parents=True, exist_ok=True)
    common.dump(path, value)
    args.fingerprint = common.fingerprint(value)


def save_predictions(folder, rows, labels, predictions):
    common.save_evaluation(folder, labels, predictions, rows)
    # Retain the common error export, but name the two score files explicitly as test results.
    (folder / "classification_report.json").replace(folder / "test_classification_report.json")
    (folder / "confusion_matrix.json").replace(folder / "test_confusion_matrix.json")


def run_probe(args, data, rows, spec, seed):
    folder = args.out / "models" / f"{spec['model']}_seed{seed}"
    identity = {"fingerprint": args.fingerprint, "spec": spec, "seed": seed}
    if (folder / "completion.json").exists():
        done = common.read(folder / "completion.json")
        if done["identity"] != identity or done["status"] != "completed" or set(done["artifacts"]) != set(ARTIFACTS):
            raise ValueError("Invalid probe completion signature")
        for name, sha in done["artifacts"].items():
            if common.digest(folder / name) != sha:
                raise ValueError(f"Diagnostic artifact changed: {folder / name}")
        print(f"Verified diagnostic cache: {spec['model']} seed={seed}", flush=True)
        return common.read(folder / "result.json")
    config_path = folder / "config.json"
    if config_path.exists() and common.read(config_path) != identity:
        raise ValueError("Probe configuration mismatch")
    if folder.exists() and any(folder.iterdir()) and not config_path.exists():
        raise ValueError("Untracked partial diagnostic folder")
    folder.mkdir(parents=True, exist_ok=True)
    common.dump(config_path, identity)
    common.seed_all(seed)
    started = time.perf_counter()
    inputs = {s: torch.from_numpy(features(data, s, spec["view"])).to(args.device)
              for s in ["train", "validation", "test"]}
    y = torch.from_numpy(data["train_y"]).to(args.device)
    counts = np.bincount(data["train_y"], minlength=7)
    weights = torch.from_numpy(common.compute_class_weights(counts, len(y), "baseline_sqrt_clip5")).to(args.device)
    # Original module preserves original initialization/RNG. The dormant future head
    # is never used in forward, loss or optimization; only trunk/classifier are trained.
    model = common.Predictor(inputs["train"].shape[1]).to(args.device)
    parameters = list(model.trunk.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-3)
    best_score, best_epoch, best_state = -1.0, 0, None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(y), device=args.device)
        total_loss = 0.0
        for indices in order.split(args.train_batch):
            optimizer.zero_grad(set_to_none=True)
            logits = model.classifier(model.trunk(inputs["train"][indices]))
            loss = common.classification_loss_fn(logits, y[indices], weights, loss_type=spec["loss"])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite diagnostic loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            total_loss += loss.item() * len(indices)
        val_score = common.metrics(data["validation_y"], classify(model, inputs["validation"]))["macro_f1"]
        history.append({"epoch": epoch, "loss": total_loss / len(y), "validation_macro_f1": val_score})
        print(f"{spec['model']} seed={seed} epoch={epoch} loss={total_loss/len(y):.4f} val_macro_f1={val_score:.4f}", flush=True)
        if val_score > best_score:
            best_score, best_epoch = val_score, epoch
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
        if epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(best_state, strict=True)
    predictions = {s: classify(model, x) for s, x in inputs.items()}
    result = {"model": spec["model"], "seed": seed, "view": spec["view"], "loss": spec["loss"],
              "uses_future_text": spec["uses_future_text"], "diagnostic_only": True,
              "best_epoch": best_epoch, "checkpoint_selection": "validation_macro_f1_only",
              "scores": split_scores(data, predictions), "input_dimension": inputs["train"].shape[1],
              "trained_parameter_count": sum(p.numel() for p in parameters),
              "seconds": time.perf_counter() - started, "status": "completed"}
    torch.save({"state_dict": best_state, "dimension": inputs["train"].shape[1], "hidden": 256,
                "labels": common.LABELS, "input_field": spec["view"], "diagnostic_only": True,
                "uses_future_text": spec["uses_future_text"], "fingerprint": args.fingerprint}, folder / "best.pt")
    common.dump(folder / "history.json", history)
    common.dump(folder / "result.json", result)
    save_predictions(folder, rows, data["test_y"], predictions["test"])
    common.dump(folder / "completion.json", {"identity": identity, "status": "completed",
                "artifacts": {name: common.digest(folder / name) for name in ARTIFACTS}})
    print(f"DONE {spec['model']} seed={seed}: train={result['scores']['train']['macro_f1']:.4f}, "
          f"validation={result['scores']['validation']['macro_f1']:.4f}, test={result['scores']['test']['macro_f1']:.4f}", flush=True)
    return result


def evaluate_reference(args, data, reference):
    ckpt = torch.load(reference["checkpoint"], map_location="cpu", weights_only=False)
    model = common.Predictor(ckpt["dimension"]).to(args.device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    predictions = {s: classify(model, torch.from_numpy(data[s + "_x"]).to(args.device))
                   for s in ["train", "validation", "test"]}
    scores = split_scores(data, predictions)
    recorded = reference["result"]
    if not np.isclose(scores["test"]["macro_f1"], recorded["macro_f1"], atol=1e-8):
        raise ValueError("Reference checkpoint inference did not reproduce saved test results")
    return {"model": "predictive_history_" + ("focal" if reference["model"] == "route1_focal_only" else "ce"),
            "seed": reference["seed"], "view": "history", "loss": "focal" if reference["model"] == "route1_focal_only" else "ce",
            "uses_future_text": False, "diagnostic_only": False, "reference_only": True,
            "scores": scores, "checkpoint_sha256": reference["checkpoint_sha256"]}


def summarize(results):
    summaries = []
    for name in dict.fromkeys(r["model"] for r in results):
        group = [r for r in results if r["model"] == name]
        item = {"model": name, "runs": len(group), "uses_future_text": group[0]["uses_future_text"], "scores": {}}
        for split in ["train", "validation", "test"]:
            scores = {}
            for metric in ["accuracy", "macro_f1", "weighted_f1"]:
                values = [r["scores"][split][metric] for r in group]
                scores[metric + "_mean"] = float(np.mean(values))
                scores[metric + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            scores["per_class"] = {label: {key + "_mean": float(np.mean([r["scores"][split]["per_class"][label][key] for r in group]))
                                          for key in ["f1", "precision", "recall", "pred_count", "tp_count", "true_count"]}
                                   for label in common.LABELS}
            item["scores"][split] = scores
        summaries.append(item)
    return summaries


def export_report(args, results, summaries):
    lines = ["事后分类诊断：真实话语信息与表示/泛化瓶颈", "",
             "target-only与history+target读取真实下一句话，只是事后识别参照，不可部署，也不是理论上限。",
             "history+history不读取未来，提供同维度/同参数量参照。拼接除以sqrt(2)，保持输入范数一致。",
             "CE/Focal均沿用原clip5权重、同一分母、15轮上限与验证集早停（具体运行配置见provenance.json）。",
             "不按测试结果选择配置，不自动推断不可预测性、情感盲区或显著性。", "",
             "模型 | 读取未来 | 训练Macro-F1 | 验证Macro-F1 | 测试Macro-F1（均值±样本标准差） | 测试Accuracy"]
    for row in summaries:
        s = row["scores"]
        lines.append(f"{row['model']} | {row['uses_future_text']} | {s['train']['macro_f1_mean']*100:.2f}% | "
                     f"{s['validation']['macro_f1_mean']*100:.2f}% | {s['test']['macro_f1_mean']*100:.2f}% ± "
                     f"{s['test']['macro_f1_std']*100:.2f}个百分点 | {s['test']['accuracy_mean']*100:.2f}%")
    lines.extend(["", "测试集七类F1（逐类Precision/Recall与三划分明细见summary.json）",
                  "模型 | " + " | ".join(common.LABELS)])
    for row in summaries:
        lines.append(row["model"] + " | " + " | ".join(f"{row['scores']['test']['per_class'][label]['f1_mean']*100:.2f}%" for label in common.LABELS))
    pairs = []
    for loss in ["ce", "focal"]:
        for a, b in [(f"posthoc_target_only_{loss}", f"predictive_history_{loss}"),
                     (f"posthoc_history_target_{loss}", f"control_history_duplicate_{loss}"),
                     (f"control_history_duplicate_{loss}", f"predictive_history_{loss}")]:
            ra = {r["seed"]: r for r in results if r["model"] == a}
            rb = {r["seed"]: r for r in results if r["model"] == b}
            seeds = sorted(ra.keys() & rb.keys())
            values = [(ra[seed]["scores"]["test"]["macro_f1"] - rb[seed]["scores"]["test"]["macro_f1"]) * 100 for seed in seeds]
            pairs.append({"a": a, "b": b, "seeds": seeds, "delta_pp": values,
                          "mean_pp": float(np.mean(values)), "sample_std_pp": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0})
    common.dump(args.out / "paired_differences.json", pairs)
    lines.extend(["", "同种子配对测试差值，仅作描述："])
    for pair in pairs:
        values = "; ".join(f"{seed}: {delta:+.3f}" for seed, delta in zip(pair["seeds"], pair["delta_pp"]))
        lines.append(f"{pair['a']} 减 {pair['b']}: {values}；均值{pair['mean_pp']:+.3f} ± {pair['sample_std_pp']:.3f}个百分点")
    lines.extend(["", "解释边界：事后更好提示真实话语提供额外可识别信息，不证明历史预测存在某个固定理论天花板。",
                  "事后仍差应结合训练/验证差距、少数类样本数、标签与错例判断，不能单独断言编码器存在情感盲区。"])
    (args.out / "diagnostic_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "runs/full")
    parser.add_argument("--reference", type=Path, default=ROOT / "runs/controlled_v2")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/posthoc_diagnostic")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--train-batch", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args(argv)
    for key in ["source", "reference", "out"]:
        setattr(args, key, getattr(args, key).resolve())
    common.protect_paths(args.out, args.source)
    common.protect_paths(args.out, args.reference)
    references = reference_records(args)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Reference uses CUDA, but CUDA is unavailable")
    ensure_provenance(args, references)
    with np.load(args.source / "embeddings.npz", allow_pickle=False) as cache:
        data = dict(cache)
    common.validate_arrays(data, args.source)
    rows = common.read(args.source / "test.json")
    common.dump(args.out / "run_config.json", {"status": "running", "fingerprint": args.fingerprint,
                "diagnostic_only": True, "expected_training_runs": len(plan()) * len(args.seeds)})
    results = []
    for reference in references:
        results.append(evaluate_reference(args, data, reference))
    common.dump(args.out / "reference_results.json", results)
    for spec in plan():
        for seed in args.seeds:
            results.append(run_probe(args, data, rows, spec, seed))
            common.dump(args.out / "results.json", results)
    summaries = summarize(results)
    common.dump(args.out / "summary.json", summaries)
    export_report(args, results, summaries)
    common.dump(args.out / "run_config.json", {"status": "completed", "fingerprint": args.fingerprint,
                "diagnostic_only": True, "training_runs_completed": len(plan()) * len(args.seeds),
                "reused_reference_runs": len(references), "seeds": args.seeds})
    print(f"Post-hoc diagnostics completed: {args.out / 'diagnostic_report.txt'}", flush=True)


if __name__ == "__main__":
    main()
