"""Offline synthetic verification. No real dataset is opened and no model is downloaded."""
import argparse
import contextlib
import io
import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import run_controlled_experiments as workflow
import demo_world_model as demo


def expect_rejection(action, description, exception_type=ValueError):
    try:
        action()
    except exception_type:
        return
    raise AssertionError(f"Expected rejection: {description}")


def unit_checks(device):
    logits = torch.randn(14, 7, device=device, requires_grad=True)
    labels = torch.arange(14, device=device) % 7
    weights = torch.tensor([0.5, 5, 1.5, 3, 5, 4, 0.8], device=device)
    ce = workflow.classification_loss_fn(logits, labels, weights)
    focal_zero = workflow.classification_loss_fn(logits, labels, weights, "focal", focal_gamma=0)
    assert torch.allclose(ce, focal_zero, atol=1e-6)
    focal = workflow.classification_loss_fn(logits, labels, weights, "focal")
    assert torch.allclose(focal, workflow.classification_loss_fn(logits, labels, 10 * weights, "focal"), atol=1e-6)
    workflow.seed_all(42)
    ordinary = workflow.Predictor(24).to(device)
    rng_a = torch.get_rng_state().clone()
    workflow.seed_all(42)
    residual = workflow.Predictor(24, residual_anchor=True).to(device)
    rng_b = torch.get_rng_state().clone()
    assert torch.equal(rng_a, rng_b)
    for key in ordinary.state_dict():
        if key.startswith(("trunk.", "classifier.")):
            assert torch.equal(ordinary.state_dict()[key], residual.state_dict()[key])
    x = F.normalize(torch.randn(12, 24, device=device), dim=-1)
    anchor = F.normalize(torch.randn_like(x), dim=-1)
    _, future = residual(x, anchor)
    assert torch.allclose(future, anchor, atol=1e-6)
    target = F.normalize(torch.randn(12, 24, device=device), dim=-1)
    target[1] = target[0]
    prediction = F.normalize(torch.randn_like(target), dim=-1).detach().requires_grad_(True)
    loss = workflow.multi_positive_infonce(prediction, target)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(prediction.grad).all()
    logits.detach().sum()  # ensure all probes are diagnostic, not persistent training outputs


def make_synthetic_source(source):
    source.mkdir()
    rng = np.random.default_rng(2026)
    arrays = {}
    stats = {"labels": workflow.LABELS, "history_turns": 4, "max_dialogues_per_split": 0, "splits": {}}
    for split, size in [("train", 70), ("validation", 21), ("test", 21)]:
        labels = np.arange(size, dtype=np.int64) % 7
        rows = [{"dialogue_id": f"synthetic_{split}_{i}", "domain": "synthetic", "target_turn": 2,
                 "history": f"User: synthetic {i}\nSystem: synthetic reply", "last_user": f"synthetic {i}",
                 "future_text": f"synthetic future {i}", "label": int(labels[i]), "previous_label": 2}
                for i in range(size)]
        workflow.dump(source / f"{split}.json", rows)
        arrays[split + "_y"] = labels
        arrays[split + "_prev"] = np.full(size, 2, dtype=np.int64)
        for field in ["x", "last", "target"]:
            vectors = rng.normal(size=(size, 24)).astype(np.float32)
            arrays[split + "_" + field] = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        stats["splits"][split] = {"samples": size, "class_counts": {
            label: int((labels == i).sum()) for i, label in enumerate(workflow.LABELS)}}
    workflow.dump(source / "data_stats.json", stats)
    workflow.dump(source / "embedding_meta.json", {"dimension": 24, "frozen_encoder": True,
        "precision": "float32", "signature": {"model": "synthetic-encoder", "max_tokens": 33,
        "truncation_side": "left", "sha256": {f"{s}.json": workflow.digest(source / f"{s}.json")
        for s in ["train", "validation", "test"]}}})
    np.savez(source / "embeddings.npz", **arrays)


class FakeEncoder:
    """No network or real encoder. Deterministic synthetic inputs for demo contract checks."""
    def __init__(self, name, **kwargs):
        assert name == "synthetic-encoder"
        self.tokenizer = types.SimpleNamespace(truncation_side=None)
        self.max_seq_length = None

    def eval(self):
        return self

    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            seed = int(workflow.fingerprint(text)[:8], 16)
            vec = np.random.default_rng(seed).normal(size=24).astype(np.float32)
            vectors.append(vec / np.linalg.norm(vec))
        return np.stack(vectors)


def demo_checks(out, device):
    saved_module = sys.modules.get("sentence_transformers")
    sys.modules["sentence_transformers"] = types.SimpleNamespace(SentenceTransformer=FakeEncoder)
    try:
        path = out / "models" / "jepa_residual_mp_infonce_seed42" / "best.pt"
        runner = demo.CheckpointRunner(path, device=device)
        assert runner.encoder.max_seq_length == 33 and runner.inference_config["history_turns"] == 4
        history = ["User: old", "System: old reply", "User: now"]
        candidates = [{"id": "A", "text": "option A"}, {"id": "B", "text": "option B"}]
        result = runner.score_candidates(history, "now", candidates)
        assert len(result["results"]) == 2
        assert result["input_field"] == "x" and "training_examples" in result["retrieval_pool"]
        rows = ["\n".join(history + ["System: " + c["text"]]) for c in candidates]
        x = torch.from_numpy(runner.encoder.encode(rows)).to(device)
        anchor = torch.from_numpy(runner.encoder.encode(["now", "now"])).to(device)
        with torch.no_grad():
            logits, _ = runner.model(x, anchor)
        observed = np.array([[r["raw_logits"][label] for label in workflow.LABELS] for r in result["results"]])
        assert np.allclose(observed, logits.cpu().numpy(), atol=1e-4)
        last_runner = demo.CheckpointRunner(out / "models" / "last_user_classifier_seed42" / "best.pt", device=device)
        last_result = last_runner.score_candidates(history, "now", candidates)
        assert last_result["input_field"] == "last"
        assert last_result["results"][0]["raw_logits"] == last_result["results"][1]["raw_logits"]
        expect_rejection(lambda: runner.score_candidates(history, "different user", candidates), "mismatched last user")
        expect_rejection(lambda: runner.score_candidates(history, "now", []), "empty candidates")
        expect_rejection(lambda: demo.CheckpointRunner._resolve_ckpt(out / "nonexistent"),
                         "silent checkpoint fallback", FileNotFoundError)
    finally:
        if saved_module is None:
            del sys.modules["sentence_transformers"]
        else:
            sys.modules["sentence_transformers"] = saved_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(4)
    unit_checks(args.device)
    with tempfile.TemporaryDirectory(prefix="controlled_synthetic_", dir=workflow.ROOT) as temporary:
        root = Path(temporary)
        source, out = root / "source", root / "output"
        make_synthetic_source(source)
        source_hashes = {name: workflow.digest(source / name) for name in workflow.REQUIRED_DATA}
        arguments = ["--out", str(out), "--reuse-from", str(source), "--device", args.device,
                     "--seeds", "42", "--epochs", "2", "--patience", "2", "--train-batch", "16"]
        # Suppress synthetic scores: they are not official experimental results.
        with contextlib.redirect_stdout(io.StringIO()):
            workflow.main(arguments + ["--route", "minority"])
            workflow.main(arguments + ["--route", "semantic"])
            workflow.main(arguments + ["--route", "all"])
            demo_checks(out, args.device)
        results = workflow.read(out / "results.json")
        assert len(results) == 12
        assert workflow.read(out / "run_config.json")["full_plan_completed"]
        baseline = torch.load(out / "models/history_classifier_seed42/best.pt", map_location="cpu", weights_only=False)
        zero = torch.load(out / "models/jepa_residual_w0_seed42/best.pt", map_location="cpu", weights_only=False)
        adjusted = torch.load(out / "models/route1_logit_adjust_tau_seed42/best.pt", map_location="cpu", weights_only=False)
        for key in baseline["state_dict"]:
            assert torch.equal(baseline["state_dict"][key], adjusted["state_dict"][key])
            if key.startswith(("trunk.", "classifier.")):
                assert torch.allclose(baseline["state_dict"][key], zero["state_dict"][key], atol=1e-5)
        for spec in workflow.experiment_plan():
            folder = out / "models" / f"{spec['exp_name']}_seed42"
            assert np.array(workflow.read(folder / "confusion_matrix.json")).shape == (7, 7)
            assert len((folder / "test_predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 21
            assert workflow.read(folder / "completion.json")["status"] == "completed"
        text = (out / "controlled_ablation_report.md").read_text(encoding="utf-8")
        assert "差异在波动范围内" not in text and "收益主要来自" not in text
        assert workflow.read(out / "validation_selection.json")["test_used_for_selection"] is False
        assert {name: workflow.digest(source / name) for name in workflow.REQUIRED_DATA} == source_hashes
        expect_rejection(lambda: workflow.protect_paths(source, source), "baseline output protection")
        expect_rejection(lambda: workflow.main(arguments + ["--lr", "0.002"]), "changed configuration")
        tampered = out / "models/history_classifier_seed42/result.json"
        changed = workflow.read(tampered)
        changed["macro_f1"] = -1
        workflow.dump(tampered, changed)
        with contextlib.redirect_stdout(io.StringIO()):
            expect_rejection(lambda: workflow.main(arguments), "tampered completed artifact")
    print("PASS: CE/Focal normalization; zero residual; paired initialization; duplicate-target backward.")
    print("PASS: 12 synthetic configs; staged resume; exact post-hoc checkpoint reuse; reports and predictions.")
    print("PASS: cache/config/hash guards; baseline unchanged; demo metadata and input restoration.")
    print("Synthetic diagnostic only. Temporary outputs cleaned; no official experiment or download started.")


if __name__ == "__main__":
    main()
