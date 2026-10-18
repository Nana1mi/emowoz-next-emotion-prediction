"""CPU-only synthetic checks: no real source/model access, downloads, or GPU workload."""
from __future__ import annotations

import contextlib
import copy
import io
import math
import tempfile
from pathlib import Path

import numpy as np
import torch
from unittest.mock import patch

import run_controlled_experiments as common
import run_encoder_baselines as base
import run_emotion_distillation as kd
from verify_encoder_baselines import assets, tiny_encoder, reject


def make_source(root):
    source = root / "source"
    source.mkdir()
    rng = np.random.default_rng(2026)
    stats = {"labels": common.LABELS, "history_turns": 6, "splits": {}}
    arrays = {}
    for split, dialogues in [("train", 15), ("validation", 2), ("test", 2)]:
        n = dialogues * 7
        rows = [{"dialogue_id": f"synthetic_{split}_{i//7}", "target_turn": 2*(i % 7 + 1), "domain": "synthetic",
                 "history": f"User: hello {i % 22}\nSystem: reply", "last_user": "hello",
                 "future_text": "FUTURE_MUST_NOT_ENTER_STUDENT", "previous_label": 6, "label": i % 7} for i in range(n)]
        common.dump(source / f"{split}.json", rows)
        y = np.array([r["label"] for r in rows], dtype=np.int64)
        arrays[split + "_y"] = y
        for field in ["x", "target"]:
            vectors = rng.normal(size=(n, 384)).astype(np.float32)
            arrays[split + "_" + field] = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        stats["splits"][split] = {"samples": n, "dialogues": dialogues,
                    "class_counts": {label: dialogues for label in common.LABELS}}
    common.dump(source / "data_stats.json", stats)
    common.dump(source / "embedding_meta.json", {"dimension": 384, "signature": {"max_tokens": 256,
                "truncation_side": "left", "sha256": {s + ".json": common.digest(source / f"{s}.json") for s in stats["splits"]}}})
    base.atomic_npz(source / "embeddings.npz", **arrays)
    return source


def make_reference(root, source, asset):
    reference = root / "reference"
    args = base.parse_args(["--source", str(source), "--out", str(reference), "--device", "cpu", "--precision", "fp32",
                           "--micro-batch", "4", "--accumulation", "2", "--eval-batch", "16", "--epochs", "2",
                           "--patience", "2", "--threads", "2"])
    base.runtime(args)
    rows, _, hashes = base.load_source(source)
    base.ensure_manifest(args, hashes, {"minilm": asset})
    spec = {"name": "minilm_full", "backbone": "minilm", "unfreeze": -1}
    records = [base.train_one(args, rows, {"minilm": asset}, spec, seed) for seed in kd.SEEDS]
    common.dump(reference / "validation_selection.json", base.validation_summary(records, [spec]))
    for seed in kd.SEEDS:
        base.evaluate_test(args, rows, {"minilm": asset}, spec, seed)
    return reference


def unit_checks():
    args = base.parse_args(["--device", "cpu", "--precision", "fp32", "--threads", "2"])
    base.runtime(args)
    args.temperature = 2.0
    weights = torch.tensor([0.5, 5, 1, 2, 5, 4, 0.8])
    y = torch.tensor([0, 4, 6, 1, 2])
    common.seed_all(11)
    logits = torch.randn(5, 7, requires_grad=True)
    split_logits = logits.detach().clone().requires_grad_(True)
    teacher = torch.randn(5, 7, requires_grad=True)
    denominator = weights[y].sum()
    loss = (base.ce_numerator(logits, y, weights) + 0.3 * kd.weighted_kl_numerator(logits, teacher, y, weights, 2)) / denominator
    loss.backward()
    for indices in [torch.arange(3), torch.arange(3, 5)]:
        micro_loss = (base.ce_numerator(split_logits[indices], y[indices], weights) +
                      0.3 * kd.weighted_kl_numerator(split_logits[indices], teacher[indices], y[indices], weights, 2)) / denominator
        micro_loss.backward()
    assert torch.allclose(logits.grad, split_logits.grad, atol=1e-7)
    assert teacher.grad is None
    identical = kd.weighted_kl_numerator(logits.detach(), logits.detach(), y, weights, 2)
    assert abs(float(identical)) < 1e-5
    data = {"input_ids": torch.tensor([[2, 8, 9, 3, 0]] * 5, dtype=torch.int32),
            "lengths": torch.full((5,), 4), "labels": y}
    a = base.TextPredictor(tiny_encoder("minilm"), -1, 42).train()
    b = copy.deepcopy(a)
    optim_a, optim_b = [base.optimizer_for(m, args) for m in [a, b]]
    schedule_a, schedule_b = [base.get_linear_schedule_with_warmup(opt, 0, 10) for opt in [optim_a, optim_b]]
    scaler_a = torch.amp.GradScaler("cuda", enabled=False)
    scaler_b = torch.amp.GradScaler("cuda", enabled=False)
    parts = [torch.arange(3), torch.arange(3, 5)]
    common.seed_all(901)
    base.update_window(a, data, parts, weights, optim_a, schedule_a, scaler_a, args)
    common.seed_all(901)
    kd.update_window(b, data, parts, weights, optim_b, schedule_b, scaler_b, args, None, 0)
    assert all(torch.equal(a.state_dict()[key], b.state_dict()[key]) for key in a.state_dict())
    assert set(base.token_batch(data, torch.arange(5), "cpu")) == {"input_ids", "attention_mask"}


def pipeline_checks():
    with tempfile.TemporaryDirectory(prefix="emotion_kd_synthetic_", dir=ROOT) as temporary:
        root = Path(temporary)
        source = make_source(root)
        asset = assets(root)["minilm"]
        with contextlib.redirect_stdout(io.StringIO()):
            reference = make_reference(root, source, asset)
            before = {str(p): common.digest(p) for directory in [source, reference] for p in directory.rglob("*") if p.is_file()}
            out = root / "kd"
            argv = ["--reference", str(reference), "--out", str(out), "--teacher-epochs", "2", "--teacher-patience", "2"]
            # Check must not train either teachers or students.
            with patch.object(kd, "train_teacher_fold", side_effect=AssertionError("check trained a teacher")), \
                 patch.object(kd, "train_student", side_effect=AssertionError("check trained a student")):
                kd.main(argv + ["--stage", "check"])
            args = kd.parse_args(argv)
            rows, stats, hashes, checked_asset, provenance, references = kd.load_reference(args)
            folds = kd.make_folds(rows["train"])
            kd.ensure_manifest(args, hashes, checked_asset, provenance, references, folds)
            groups = np.array([r["dialogue_id"] for r in rows["train"]])
            coverage = []
            for fold in folds["plans"]:
                fit, stop, heldout = [set(groups[fold[name]]) for name in ["fit", "stop", "heldout"]]
                assert not fit & heldout and not stop & heldout and not fit & stop
                coverage += fold["heldout"]
            assert sorted(coverage) == list(range(len(rows["train"])))
            teacher = kd.prepare_teacher(args, rows, folds)
            assert teacher.shape == (len(rows["train"]), 7) and not teacher.requires_grad
            # Zero-KD full trainer must reproduce the matched CE reference exactly.
            zero_spec = {"name": "synthetic_beta_zero", "backbone": "minilm", "unfreeze": -1, "beta": 0.0}
            zero = kd.train_student(args, rows, checked_asset, zero_spec, 42, None)
            ce_state = torch.load(reference / "models" / "minilm_full_seed42" / "best.pt", map_location="cpu", weights_only=False)
            zero_state = torch.load(base.folder_for(args, zero_spec, 42) / "best.pt", map_location="cpu", weights_only=False)
            assert zero["best_epoch"] == references[0]["result"]["best_epoch"]
            for group in ["head_state", "encoder_state"]:
                assert all(torch.equal(zero_state[group][key], ce_state[group][key]) for key in zero_state[group])
            kd.main(argv + ["--stage", "screen"])
            assert not list((out / "cache").glob("*test*"))
            assert not list((out / "models").glob("*/test_result.json"))
            with patch.object(kd, "update_window", side_effect=AssertionError("completed screen retrained")), \
                 patch.object(kd, "train_teacher_fold", side_effect=AssertionError("completed teacher retrained")):
                kd.main(argv + ["--stage", "screen"])
            spec = common.read(out / "screen_selection.json")["confirmation_spec"]
            calls = 0
            original_update = kd.update_window
            steps = math.ceil(len(rows["train"]) / (args.micro_batch * args.accumulation))
            def interrupt(*a, **kw):
                nonlocal calls
                calls += 1
                if calls == steps + 1:
                    raise KeyboardInterrupt
                return original_update(*a, **kw)
            try:
                with patch.object(kd, "update_window", side_effect=interrupt):
                    kd.train_student(args, rows, checked_asset, spec, 43, teacher)
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError("student interruption did not occur")
            resumed = kd.train_student(args, rows, checked_asset, spec, 43, teacher)
            uninterrupted_args = copy.copy(args)
            uninterrupted_args.out = root / "uninterrupted"
            uninterrupted_args.out.mkdir()
            uninterrupted = kd.train_student(uninterrupted_args, rows, checked_asset, spec, 43, teacher)
            assert resumed["best_epoch"] == uninterrupted["best_epoch"]
            resumed_state = torch.load(base.folder_for(args, spec, 43) / "best.pt", map_location="cpu", weights_only=False)
            reference_state = torch.load(base.folder_for(uninterrupted_args, spec, 43) / "best.pt", map_location="cpu", weights_only=False)
            for group in ["head_state", "encoder_state"]:
                assert all(torch.equal(resumed_state[group][k], reference_state[group][k]) for k in resumed_state[group])
            kd.main(argv + ["--stage", "confirm"])
            assert common.read(out / "run_status.json")["status"] == "completed"
            result = common.read(out / "results.json")
            assert [r["seed"] for r in result["training"]] == [42, 43, 44]
            assert len(result["test"]) == 3
            assert all(not r["uses_future_text"] and not r["uses_previous_label"] for r in result["training"])
            report = (out / "emotion_distillation_report.txt").read_text(encoding="utf-8")
            assert "不预设提分" in report and "不是预测成绩或理论上限" in report
            with patch.object(kd, "update_window", side_effect=AssertionError("completed confirm retrained")):
                kd.main(argv + ["--stage", "confirm"])
            reject(lambda: kd.main(argv + ["--stage", "check", "--temperature", "3"]), "changed temperature")
            reject(lambda: kd.main(["--reference", str(reference), "--out", str(reference)]), "reference overwrite")
            after = {str(p): common.digest(p) for directory in [source, reference] for p in directory.rglob("*") if p.is_file()}
            assert before == after
            logits_path = out / "teachers" / "oof_logits.npz"
            with logits_path.open("ab") as stream:
                stream.write(b"corruption")
            reject(lambda: kd.prepare_teacher(args, rows, folds), "OOF cache corruption")


ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2)
    unit_checks()
    pipeline_checks()
    print("PASS: dialogue-disjoint outer OOF + inner stopping folds, complete aligned training-only teacher coverage.")
    print("PASS: detached T^2 KL, exact weighted accumulation, beta-zero trainer exactly reproduces CE checkpoint.")
    print("PASS: validation-only screen, three-seed confirmation, test after selection, checkpoint resume equals uninterrupted run.")
    print("PASS: source/reference unchanged, corruption/config/overwrite rejection, verified completed cache reuse.")
    print("CPU synthetic outputs cleaned. No real data/model access, downloads, or GPU workload.")


if __name__ == "__main__":
    main()
