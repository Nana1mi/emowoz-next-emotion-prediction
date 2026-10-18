"""Synthetic post-hoc diagnostic verification; does not open real dataset files."""
import argparse
import contextlib
import io
import tempfile
from pathlib import Path

import numpy as np
import torch

import run_controlled_experiments as common
import run_posthoc_diagnostic as probe
from verify_controlled import expect_rejection, make_synthetic_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    with tempfile.TemporaryDirectory(prefix="posthoc_synthetic_", dir=probe.ROOT) as temporary:
        root = Path(temporary)
        source, reference, out = root / "source", root / "reference", root / "probes"
        make_synthetic_source(source)
        with np.load(source / "embeddings.npz", allow_pickle=False) as cache:
            data = dict(cache)
        joint = probe.features(data, "train", "posthoc_history_target")
        duplicated = probe.features(data, "train", "control_history_duplicate")
        assert joint.shape == duplicated.shape == (70, 48)
        assert np.allclose(np.linalg.norm(joint, axis=1), 1, atol=1e-6)
        assert np.array_equal(joint[:, :24], duplicated[:, :24])
        assert np.allclose(joint[:, 24:], data["train_target"] / np.sqrt(2))
        assert np.array_equal(probe.features(data, "train", "posthoc_target_only"), data["train_target"])
        assert len(probe.plan()) == 6
        with contextlib.redirect_stdout(io.StringIO()):
            common.main(["--out", str(reference), "--reuse-from", str(source), "--device", args.device,
                         "--seeds", "42", "--epochs", "2", "--patience", "2", "--train-batch", "16",
                         "--route", "minority"])
            before = {str(path): common.digest(path) for folder in [source, reference]
                      for path in folder.rglob("*") if path.is_file()}
            arguments = ["--source", str(source), "--reference", str(reference), "--out", str(out)]
            probe.main(arguments)
            original_trainer = probe.run_probe
            def cached_only(*function_args, **function_kwargs):
                checkpoint = function_args[0].out / "models" / f"{function_args[3]['model']}_seed{function_args[4]}" / "best.pt"
                assert checkpoint.exists()
                return original_trainer(*function_args, **function_kwargs)
            probe.run_probe = cached_only
            try:
                probe.main(arguments)
            finally:
                probe.run_probe = original_trainer
        after = {str(path): common.digest(path) for folder in [source, reference]
                 for path in folder.rglob("*") if path.is_file()}
        assert before == after
        assert common.read(out / "run_config.json")["training_runs_completed"] == 6
        results = common.read(out / "results.json")
        assert len(results) == 8 and len(common.read(out / "summary.json")) == 8
        for result in results:
            assert set(result["scores"]) == {"train", "validation", "test"}
            for scores in result["scores"].values():
                assert set(scores["per_class"]) == set(common.LABELS)
        for spec in probe.plan():
            folder = out / "models" / f"{spec['model']}_seed42"
            checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
            assert checkpoint["diagnostic_only"]
            assert checkpoint["uses_future_text"] == spec["uses_future_text"]
            assert np.array(common.read(folder / "test_confusion_matrix.json")).shape == (7, 7)
            assert set(common.read(folder / "completion.json")["artifacts"]) == set(probe.ARTIFACTS)
        text = (out / "diagnostic_report.txt").read_text(encoding="utf-8")
        assert "不是理论上限" in text and "不按测试结果选择配置" in text
        expect_rejection(lambda: probe.main(arguments + ["--lr", "0.002"]), "different reference hyperparameters")
        expect_rejection(lambda: probe.main(["--source", str(source), "--reference", str(reference),
                         "--out", str(reference)]), "write into reference directory")
    print("PASS: input views, equal-dimensional control, six CE/Focal probes, all three split/class scores.")
    print("PASS: strict reference reproduction, hash-checked resume, read-only source/reference, diagnostic labeling.")
    print("Synthetic outputs cleaned; no formal training or download started.")


if __name__ == "__main__":
    main()
