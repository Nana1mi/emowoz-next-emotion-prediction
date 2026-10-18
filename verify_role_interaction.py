"""CPU synthetic role-span/interaction verification; no real data/model/GPU workload."""
from __future__ import annotations

import contextlib
import copy
import io
import math
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

import run_controlled_experiments as common
import run_encoder_baselines as base
import run_role_interaction as role
from verify_encoder_baselines import assets, tiny_encoder, reject
from verify_emotion_distillation import make_source as make_initial_source, make_reference

ROOT = Path(__file__).resolve().parent


def make_source(root):
    source = make_initial_source(root)
    raw_dir = root / "raw"
    raw_dir.mkdir()
    raw = {}
    for split in ["train", "validation", "test"]:
        rows = common.read(source / f"{split}.json")
        for row in rows:
            key = row["dialogue_id"]
            if key not in raw:
                turns = []
                for i in range(8):
                    user = "hello " + str(i)
                    if i == 1:
                        user += "\nSystem: quoted text"  # Role-looking text inside an utterance.
                    system = "reply " + str(i) if i != 1 else ""  # Actual empty latest reply.
                    turns.extend([{"text": user}, {"text": system}])
                raw[key] = {"log": turns}
            target = row["target_turn"]
            turns = raw[key]["log"]
            row["history"] = "\n".join(("User: " if i % 2 == 0 else "System: ") + turns[i]["text"].strip()
                                       for i in range(max(0, target - 6), target))
            row["last_user"] = turns[target-2]["text"].strip()
            row["future_text"] = turns[target]["text"].strip()
        common.dump(source / f"{split}.json", rows)
    common.dump(raw_dir / role.RAW_FILES[0], raw)
    common.dump(raw_dir / role.RAW_FILES[1], {})
    stats = common.read(source / "data_stats.json")
    stats["source_sha256"] = {name: common.digest(raw_dir / name) for name in role.RAW_FILES}
    common.dump(source / "data_stats.json", stats)
    meta = common.read(source / "embedding_meta.json")
    meta["signature"]["sha256"] = {s + ".json": common.digest(source / f"{s}.json") for s in ["train", "validation", "test"]}
    common.dump(source / "embedding_meta.json", meta)
    return source, raw_dir


def unit_checks():
    common.seed_all(10)
    hidden = torch.randn(2, 5, 24)
    zero = role.masked_pool(hidden, torch.zeros(2, 5, dtype=torch.bool))
    assert torch.equal(zero, torch.zeros_like(zero))
    offsets = torch.tensor([[[0, 0], [0, 4], [5, 6], [7, 11], [0, 0]]])
    attention = torch.ones(1, 5)
    assert role.offset_masks(offsets, [[7, 11]], attention).tolist() == [[False, False, False, True, False]]
    assert not role.offset_masks(offsets, [[7, 7]], attention).any()
    data = {"input_ids": torch.tensor([[2, 8, 9, 3, 0], [2, 7, 9, 3, 0], [2, 8, 7, 3, 0]], dtype=torch.int32),
            "lengths": torch.tensor([4, 4, 4]), "labels": torch.tensor([0, 4, 6]),
            "user_mask": torch.tensor([[0, 1, 0, 0, 0]] * 3, dtype=torch.bool),
            "system_mask": torch.tensor([[0, 0, 1, 0, 0]] * 3, dtype=torch.bool)}
    models = [role.RolePredictor(tiny_encoder("minilm"), spec, 42).eval() for spec in role.plan()]
    assert len({sum(p.numel() for p in m.parameters()) for m in models}) == 1
    assert all(torch.equal(models[0].head.state_dict()[k], m.head.state_dict()[k]) for m in models for k in m.head.state_dict())
    batch = role.batch_input(data, torch.arange(3), "cpu")
    altered = {**batch, "user_mask": torch.zeros_like(batch["user_mask"]), "system_mask": torch.zeros_like(batch["system_mask"])}
    assert torch.equal(models[0](batch), models[0](altered))  # Global control cannot exploit role masks.
    assert not torch.allclose(models[1](batch), models[1](altered))
    for model in models:
        calls = []
        handle = model.encoder.register_forward_hook(lambda *args: calls.append(1))
        logits = model(batch)
        handle.remove()
        assert calls == [1]
        F.cross_entropy(logits, data["labels"]).backward()
        assert all(p.grad is not None for p in model.head.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
        reject(lambda: model({**batch, "future_text": "forbidden"}), "privileged model input")
    history, user, action = [F.normalize(torch.randn(3, 24), dim=-1) for _ in range(3)]
    for model in models:
        head = model.head
        s = F.gelu(head.state(torch.cat([history, user], -1) / math.sqrt(2)))
        g, d = head.action(action).chunk(2, -1)
        expected = s * (1 + 0.5 * torch.tanh(g)) + 0.5 * torch.tanh(d) if head.mixing == "film" else \
            s + 0.5 * torch.tanh(g) + 0.5 * torch.tanh(d)
        assert torch.allclose(head(history, user, action), head.classifier(head.norm(expected)))
    # With gamma=0, additive and FiLM become the same function: checks the intended isolated term.
    additive, film = models[1].head, models[2].head
    for head in [additive, film]:
        with torch.no_grad():
            head.action.weight[:256].zero_()
            head.action.bias[:256].zero_()
    assert torch.allclose(additive(history, user, action), film(history, user, action))
    # Weighted accumulation matches a single full window (including uneven micro-batches).
    args = base.parse_args(["--device", "cpu", "--precision", "fp32", "--threads", "2"])
    weights = torch.tensor([0.5, 5, 1, 2, 5, 4, 0.8])
    a, b = copy.deepcopy(models[2]), copy.deepcopy(models[2])
    optim_a, optim_b = [torch.optim.SGD(m.parameters(), lr=0.01) for m in [a, b]]
    sch_a, sch_b = [base.get_linear_schedule_with_warmup(o, 0, 10) for o in [optim_a, optim_b]]
    role.update_window(a, data, [torch.arange(3)], weights, optim_a, sch_a, torch.amp.GradScaler("cuda", enabled=False), args)
    role.update_window(b, data, [torch.arange(2), torch.arange(2, 3)], weights, optim_b, sch_b, torch.amp.GradScaler("cuda", enabled=False), args)
    assert all(torch.allclose(a.state_dict()[k], b.state_dict()[k], atol=1e-6) for k in a.state_dict())


def pipeline_checks():
    with tempfile.TemporaryDirectory(prefix="role_synthetic_", dir=ROOT) as temporary:
        root = Path(temporary)
        source, raw_dir = make_source(root)
        asset = assets(root)["minilm"]
        with contextlib.redirect_stdout(io.StringIO()):
            reference = make_reference(root, source, asset)
            before = {str(p): common.digest(p) for directory in [source, raw_dir, reference] for p in directory.rglob("*") if p.is_file()}
            out = root / "roles"
            argv = ["--reference", str(reference), "--raw-dir", str(raw_dir), "--out", str(out)]
            with patch.object(role, "train_one", side_effect=AssertionError("check started training")):
                role.main(argv + ["--stage", "check"])
            args = role.parse_args(argv)
            rows, stats, hashes, checked_asset, provenance, references = role.reference_tools.load_reference(args)
            role.ensure_manifest(args, hashes, role.raw_hashes(args, stats), checked_asset, provenance, references)
            bounds = role.span_boundaries(args, rows)
            data = role.split_data(args, rows, bounds, asset, ["train", "validation"])
            for split in ["train", "validation"]:
                original = torch.load(reference / "cache" / f"minilm_{split}_tokens.pt", map_location="cpu", weights_only=False)
                assert torch.equal(data[split]["input_ids"], original["input_ids"])
            assert (data["train"]["system_mask"].sum(1) == 0).any()
            # Future content changes do not affect any reconstructed input boundary.
            raw = common.read(raw_dir / role.RAW_FILES[0])
            row = rows["train"][0]
            turns = copy.deepcopy(raw[row["dialogue_id"]]["log"])
            original_spans = role.reconstruct_spans(row, turns)
            turns[row["target_turn"]]["text"] = "CHANGED FUTURE MUST NOT MATTER"
            assert original_spans == role.reconstruct_spans(row, turns)
            altered_row = {**row, "history": row["history"] + "\nUser: injected future"}
            reject(lambda: role.reconstruct_spans(altered_row, turns), "history boundary injection")
            # A very long current system reply may remove all user tokens; no fabricated fallback.
            long_row = {"history": "User: hello\nSystem: " + "reply " * 300, "label": 0}
            start = len("User: hello\nSystem: ")
            long_data = role.token_cache(args, asset, "synthetic_long", [long_row], np.array([[6, 11]]),
                                         np.array([[start, len(long_row["history"])]]))
            assert not long_data["user_mask"].any() and long_data["system_mask"].any()
            role.main(argv + ["--stage", "screen"])
            assert not list((out / "cache").glob("*test*"))
            assert not list((out / "models").glob("*/test_result.json"))
            with patch.object(role, "update_window", side_effect=AssertionError("completed screen retrained")):
                role.main(argv + ["--stage", "screen"])
            # Epoch-boundary recovery must reproduce an uninterrupted model.
            spec = role.plan()[2]
            steps = math.ceil(len(rows["train"]) / (args.micro_batch * args.accumulation))
            calls = 0
            original_update = role.update_window
            def interrupted(*a, **kw):
                nonlocal calls
                calls += 1
                if calls == steps + 1:
                    raise KeyboardInterrupt
                return original_update(*a, **kw)
            try:
                with patch.object(role, "update_window", side_effect=interrupted):
                    role.train_one(args, rows, bounds, asset, spec, 43)
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError("interruption did not happen")
            resumed = role.train_one(args, rows, bounds, asset, spec, 43)
            reference_args = copy.copy(args)
            reference_args.out = root / "uninterrupted"
            reference_args.out.mkdir()
            uninterrupted = role.train_one(reference_args, rows, bounds, asset, spec, 43)
            assert resumed["best_epoch"] == uninterrupted["best_epoch"]
            a = torch.load(base.folder_for(args, spec, 43) / "best.pt", map_location="cpu", weights_only=False)
            b = torch.load(base.folder_for(reference_args, spec, 43) / "best.pt", map_location="cpu", weights_only=False)
            for group in ["head_state", "encoder_state"]:
                assert all(torch.equal(a[group][k], b[group][k]) for k in a[group])
            role.main(argv + ["--stage", "confirm"])
            result = common.read(out / "results.json")
            assert len(result["training"]) == len(result["test"]) == 9
            assert len({r["head_parameter_count"] for r in result["training"]}) == 1
            assert len(common.read(out / "summary.json")) == 4
            assert len(common.read(out / "paired_comparisons.json")) == 3
            assert common.read(out / "run_status.json")["status"] == "completed"
            with patch.object(role, "update_window", side_effect=AssertionError("completed confirm retrained")):
                role.main(argv + ["--stage", "confirm"])
            reject(lambda: role.main(["--reference", str(reference), "--raw-dir", str(raw_dir), "--out", str(reference)]), "reference overwrite")
            after = {str(p): common.digest(p) for directory in [source, raw_dir, reference] for p in directory.rglob("*") if p.is_file()}
            assert before == after
            path = out / "cache" / "minilm_train_role_tokens.pt"
            with path.open("ab") as stream:
                stream.write(b"corrupted")
            reject(lambda: role.split_data(args, rows, bounds, asset, ["train"]), "role cache corruption")


def main():
    torch.set_num_threads(2)
    unit_checks()
    pipeline_checks()
    print("PASS: raw-turn offsets, quoted role text, empty/truncated replies, future exclusion, unchanged 256-token inputs.")
    print("PASS: equal parameter counts and initialization, single encoder pass, active encoder/head gradients, isolated FiLM term.")
    print("PASS: weighted accumulation, 9 training/test runs, validation-only selection, resume equals uninterrupted weights.")
    print("PASS: source/reference/raw unchanged, completed cache reuse, corruption and overwrite rejection.")
    print("CPU synthetic outputs cleaned. No real data/model access, downloads, or GPU workload.")


if __name__ == "__main__":
    main()
