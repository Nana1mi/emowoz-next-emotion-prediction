"""Offline tiny-random-Transformer checks. Never downloads or opens real experiment data."""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast, RobertaConfig, RobertaModel

import run_encoder_baselines as workflow
import run_controlled_experiments as common
import run_experiments as original


def reject(action, description):
    try:
        action()
    except ValueError:
        return
    raise AssertionError(f"Expected rejection: {description}")


def tiny_encoder(backbone):
    common.seed_all(2026)
    config = dict(vocab_size=32, hidden_size=24, num_hidden_layers=4, num_attention_heads=4,
                  intermediate_size=32, max_position_embeddings=512, pad_token_id=0,
                  hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
    return BertModel(BertConfig(**config), add_pooling_layer=False) if backbone == "minilm" else \
        RobertaModel(RobertaConfig(**config, bos_token_id=2, eos_token_id=3), add_pooling_layer=False)


def make_source(root):
    source = root / "source"
    source.mkdir()
    stats = {"labels": common.LABELS, "history_turns": 6, "splits": {}}
    arrays = {}
    for split, n in [("train", 21), ("validation", 14), ("test", 14)]:
        rows = [{"dialogue_id": f"synthetic_{split}_{i}", "domain": "synthetic", "target_turn": 2,
                 "history": f"User: hello {i}\nSystem: reply", "last_user": "hello",
                 "future_text": "NEVER_ENCODE_THIS_FUTURE_SENTINEL", "previous_label": 6,
                 "label": i % 7} for i in range(n)]
        common.dump(source / f"{split}.json", rows)
        stats["splits"][split] = {"samples": n, "dialogues": n,
                    "class_counts": {label: n // 7 for label in common.LABELS}}
        arrays[split + "_x"] = np.ones((n, 24), dtype=np.float32) / np.sqrt(24)
    common.dump(source / "data_stats.json", stats)
    common.dump(source / "embedding_meta.json", {"signature": {"max_tokens": 256, "truncation_side": "left",
                "sha256": {s + ".json": common.digest(source / f"{s}.json") for s in stats["splits"]}}})
    workflow.atomic_npz(source / "embeddings.npz", **arrays)
    return source


def assets(root):
    result = {}
    for backbone in ["minilm", "roberta"]:
        folder = root / ("tiny_" + backbone)
        folder.mkdir()
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "User", "System", ":", "hello", "reply"] + [str(i) for i in range(22)]
        (folder / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
        tokenizer = BertTokenizerFast(vocab_file=str(folder / "vocab.txt"), do_lower_case=False)
        tokenizer.save_pretrained(folder)
        tiny_encoder(backbone).save_pretrained(folder)
        result[backbone] = {"repo": "synthetic-only/" + backbone, "path": str(folder), "revision": "synthetic",
                           "sha256": {p.name: common.digest(p) for p in folder.iterdir() if p.is_file()}}
    return result


def temporal_checks():
    def turn(text, label=0):
        return {"text": text, "emotion": [{}, {}, {}, {"emotion": label}]}
    dialogue = {"log": [turn("request"), turn("system reply"), turn("FUTURE_SENTINEL", 4),
                        turn("later system"), turn("LATER_FUTURE", 3), turn("end")]}
    rows = original.make_rows("synthetic", dialogue, "synthetic", 6)
    assert rows[0]["history"] == "User: request\nSystem: system reply"
    assert "FUTURE_SENTINEL" not in workflow.history_texts(rows[:1])[0]
    altered = copy.deepcopy(rows)
    altered[0]["future_text"], altered[0]["previous_label"], altered[0]["target_turn"] = "CHANGED", 6, 100
    assert workflow.history_texts(rows) == workflow.history_texts(altered)


def gradient_checks(device):
    args = workflow.parse_args(["--device", device, "--precision", "fp32", "--micro-batch", "3", "--accumulation", "2"])
    workflow.runtime(args)
    data = {"input_ids": torch.tensor([[2, 8, 9, 3, 0], [2, 9, 3, 0, 0], [2, 7, 3, 0, 0]], dtype=torch.int32),
            "lengths": torch.tensor([4, 3, 3]), "labels": torch.tensor([0, 4, 6])}
    for backbone in ["minilm", "roberta"]:
        for unfreeze in [0, 2, -1]:
            model = workflow.TextPredictor(tiny_encoder(backbone), unfreeze, 42, gradient_checkpointing=unfreeze > 0).to(device)
            model.train()
            before = {k: p.detach().clone() for k, p in model.encoder.named_parameters()}
            assert model.encoder.embeddings.training == (unfreeze == -1)
            assert model.encoder.encoder.layer[-1].training == bool(unfreeze)
            assert model.encoder.encoder.layer[0].training == (unfreeze == -1)
            loss = torch.nn.functional.cross_entropy(model(workflow.token_batch(data, torch.arange(3), device)), data["labels"].to(device))
            loss.backward()
            grads = {k: p.grad for k, p in model.encoder.named_parameters()}
            if unfreeze == 0:
                assert all(g is None for g in grads.values())
            else:
                assert any(g is not None and g.abs().sum() > 0 for g in grads.values())
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
            optimizer.step()
            for k, p in model.encoder.named_parameters():
                if not p.requires_grad:
                    assert torch.equal(before[k], p)
            if unfreeze == 2:
                assert all(grads[k] is None for k in grads if k.startswith("embeddings.") or k.startswith("encoder.layer.0."))
            del model
    # Weighted accumulation must equal full-window weighted CE, including partial windows.
    weights = torch.tensor([0.5, 5, 1, 2, 5, 4, 0.8], device=device)
    y = torch.tensor([0, 0, 4, 6, 1], device=device)
    common.seed_all(9)
    a = torch.randn(5, 7, device=device, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    torch.nn.functional.cross_entropy(a, y, weight=weights).backward()
    for idx in [torch.arange(3, device=device), torch.arange(3, 5, device=device)]:
        (workflow.ce_numerator(b[idx], y[idx], weights) / weights[y].sum()).backward()
    assert torch.allclose(a.grad, b.grad, atol=1e-7)
    windows = list(workflow.training_windows(torch.arange(11), 3, 2))
    assert [sum(len(p) for p in w) for w in windows] == [6, 5]
    assert torch.equal(torch.cat([p for w in windows for p in w]), torch.arange(11))
    if device == "cuda":
        # Exercise the actual AMP optimizer/scaler path, not only fp32 CUDA gradients.
        for precision in ["bf16", "fp16"]:
            if precision == "bf16" and not torch.cuda.is_bf16_supported():
                continue
            amp_args = workflow.parse_args(["--device", "cuda", "--precision", precision,
                                           "--micro-batch", "2", "--accumulation", "2"])
            model = workflow.TextPredictor(tiny_encoder("minilm"), 2, 42).to(device).train()
            optimizer = workflow.optimizer_for(model, amp_args)
            scheduler = workflow.get_linear_schedule_with_warmup(optimizer, 0, 3)
            scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
            initial = model.encoder.encoder.layer[-1].output.dense.weight.detach().clone()
            # FP16's initial scale can legitimately overflow in a tiny random
            # LayerNorm network. Ensure skipped updates also skip the scheduler,
            # then verify a finite update after automatic scale backoff.
            for _ in range(12):
                previous_scale, previous_step = scaler.get_scale(), scheduler.last_epoch
                numerator, denominator = workflow.update_window(model, data, [torch.arange(2), torch.arange(2, 3)],
                                    weights, optimizer, scheduler, scaler, amp_args)
                assert np.isfinite(numerator / denominator)
                if scaler.get_scale() < previous_scale:
                    assert scheduler.last_epoch == previous_step
                if not torch.equal(initial, model.encoder.encoder.layer[-1].output.dense.weight):
                    break
            assert not torch.equal(initial, model.encoder.encoder.layer[-1].output.dense.weight)
            del model, optimizer
    unit = workflow.scores(np.array([0, 4, 6]), np.array([0, 2, 6]))
    assert unit["negative"]["f1"] == 1.0 and len(unit["per_class"]) == 7
    # Screening decisions cannot use test scores.
    specs = workflow.plan("minilm")
    records = [{"model": s["name"], "spec": s, "validation": {"macro_f1": v}, "test": {"macro_f1": t}}
               for s, v, t in zip(specs, [.2, .4, .3], [.9, .1, .8])]
    assert [s["name"] for s in workflow.choose_confirmation(records, specs, 1)] == ["minilm_frozen", "minilm_last2"]


def pipeline_checks(device):
    with tempfile.TemporaryDirectory(prefix="encoder_synthetic_", dir=workflow.ROOT) as temporary:
        root = Path(temporary)
        source = make_source(root)
        tiny_assets = assets(root)
        out = root / "output"
        arguments = ["--source", str(source), "--out", str(out), "--suite", "all", "--device", device,
                     "--precision", "fp32", "--micro-batch", "3", "--accumulation", "2", "--eval-batch", "7",
                     "--epochs", "2", "--patience", "2", "--top-k", "1", "--threads", "2"]
        before = {str(p): common.digest(p) for p in source.iterdir()}
        with patch.object(workflow, "resolve_assets", side_effect=lambda b, _: tiny_assets[b]), \
                contextlib.redirect_stdout(io.StringIO()):
            workflow.main(arguments + ["--stage", "screen"])
            assert not list((out / "cache").glob("*test*"))
            assert not list((out / "models").glob("*/test_result.json"))
            original_trainer = workflow.update_window
            with patch.object(workflow, "update_window", side_effect=AssertionError("Completed screen was retrained")):
                workflow.main(arguments + ["--stage", "screen"])
            # Simulate an interruption after epoch 1, then resume with identical run identity.
            specs = workflow.plan("all")
            spec = specs[1]
            prepared = workflow.parse_args(arguments)
            workflow.runtime(prepared)
            prepared.fingerprint = common.fingerprint(common.read(out / "provenance.json"))
            rows, _, _ = workflow.load_source(source)
            steps_per_epoch = 4
            calls = 0
            def interrupted(*a, **kw):
                nonlocal calls
                calls += 1
                if calls == steps_per_epoch + 1:
                    raise KeyboardInterrupt
                return original_trainer(*a, **kw)
            try:
                with patch.object(workflow, "update_window", side_effect=interrupted):
                    workflow.train_one(prepared, rows, tiny_assets, spec, 43)
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError("Interruption did not happen")
            resumed = workflow.train_one(prepared, rows, tiny_assets, spec, 43)
            assert resumed["epochs_completed"] == 2
            resumed_state = torch.load(workflow.folder_for(prepared, spec, 43) / "best.pt", map_location="cpu", weights_only=False)
            # Same seed/config with no interruption must produce matching best weights.
            uninterrupted_args = copy.copy(prepared)
            uninterrupted_args.out = root / "uninterrupted"
            uninterrupted_args.out.mkdir()
            uninterrupted = workflow.train_one(uninterrupted_args, rows, tiny_assets, spec, 43)
            reference_state = torch.load(workflow.folder_for(uninterrupted_args, spec, 43) / "best.pt", map_location="cpu", weights_only=False)
            assert resumed["best_epoch"] == uninterrupted["best_epoch"]
            for group in ["head_state", "encoder_state"]:
                assert all(torch.allclose(resumed_state[group][k], reference_state[group][k], atol=1e-6, rtol=1e-6)
                           for k in resumed_state[group])
            workflow.main(arguments + ["--stage", "confirm"])
            selected = common.read(out / "validation_selection.json")
            assert all(r["seeds"] == [42, 43, 44] for r in selected["summary"])
            assert len(selected["summary"]) == 3  # two family controls plus one challenger
            assert common.read(out / "run_status.json")["status"] == "completed"
            with patch.object(workflow, "update_window", side_effect=AssertionError("Completed run was retrained")):
                workflow.main(arguments + ["--stage", "confirm"])
            reject(lambda: workflow.main(arguments + ["--stage", "screen", "--head-lr", "0.002"]), "changed training config")
            reject(lambda: workflow.main(["--source", str(source), "--out", str(source), "--device", device]), "source overwrite")
            cache_path = out / "cache" / "minilm_train_tokens.pt"
            with cache_path.open("ab") as stream:
                stream.write(b"corruption")
            reject(lambda: workflow.token_cache(prepared, tiny_assets["minilm"], "minilm", "train", rows["train"]), "corrupt token cache")
        after = {str(p): common.digest(p) for p in source.iterdir()}
        assert before == after
        for result in common.read(out / "results.json")["training"]:
            assert not result["uses_future_text"] and not result["uses_previous_label"]
        assert len(common.read(out / "summary.json")[0]["test_per_class"]) == 7


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(2)
    temporal_checks()
    gradient_checks(args.device)
    pipeline_checks(args.device)
    print("PASS: history-only cutoff, BERT/RoBERTa frozen/partial/full gradients, frozen-block eval, checkpointing.")
    print("PASS: exact weighted accumulation, partial windows, validation-only screen, three-seed confirmation.")
    print("PASS: epoch-boundary resume, checksum-verified completed caches, corruption/config rejection, read-only source.")
    print("Tiny synthetic outputs cleaned. No real training, dataset access or model download.")


if __name__ == "__main__":
    main()
