"""Offline end-to-end verification. Synthetic outputs live in a temp directory."""
import argparse
import contextlib
import io
import tempfile
from pathlib import Path

import numpy as np

import run_experiments as workflow


def main():
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory(prefix="synthetic_verify_", dir=workflow.ROOT) as temp:
        output = Path(temp)
        args = argparse.Namespace(out=output, device="cuda", seeds=[42], epochs=2,
                                  patience=2, lr=0.001, train_batch=32,
                                  semantic_weight=0.5, stage="synthetic_verify")
        data = {}
        stats = {"max_dialogues_per_split": 1, "history_turns": 6, "splits": {}}
        for split, size in [("train", 70), ("validation", 21), ("test", 21)]:
            labels = np.arange(size) % 7
            data[split + "_y"] = labels.astype(np.int64)
            data[split + "_prev"] = labels.astype(np.int64)
            rows = [{"dialogue_id": "synthetic_" + split + str(i), "domain": "synthetic",
                     "target_turn": 2, "history": "synthetic history",
                     "last_user": "synthetic user", "future_text": str(i),
                     "label": int(labels[i]), "previous_label": int(labels[i])} for i in range(size)]
            workflow.dump(output / f"{split}.json", rows)
            for name in ["x", "last", "target"]:
                x = rng.normal(size=(size, 24)).astype(np.float32)
                data[split + "_" + name] = x / np.linalg.norm(x, axis=1, keepdims=True)
            stats["splits"][split] = {"samples": size, "class_counts": {
                label: int(np.sum(labels == i)) for i, label in enumerate(workflow.LABELS)}}
        workflow.dump(output / "data_stats.json", stats)
        workflow.dump(output / "embedding_meta.json", {"signature": {"sha256": {
            f"{split}.json": workflow.digest(output / f"{split}.json") for split in ["train", "validation", "test"]}}})
        np.savez(output / "embeddings.npz", **data)
        # Suppress synthetic metrics so they cannot be mistaken for research results.
        with contextlib.redirect_stdout(io.StringIO()):
            workflow.train(args)
            workflow.summarize(args)
        assert len(workflow.read(output / "summary.json")) == 5
        for kind in ["last_user_classifier", "history_classifier", "jepa"]:
            folder = output / "models" / f"{kind}_seed42"
            assert (folder / "best.pt").exists()
            assert np.array(workflow.read(folder / "confusion_matrix.json")).shape == (7, 7)
            assert (folder / "test_predictions.jsonl").exists()
        assert len(list((output / "figures").glob("*.png"))) == 3
        draft = (output / "第二次报告草稿.md").read_text(encoding="utf-8")
        assert "[1]" in draft and "21个未来情绪预测样本" in draft
        body = draft.split("## 参考文献")[0]
        print("PASS: three training models, validation selection, metrics aggregation, plots, report generation.")
        print(f"Report draft length in synthetic check (including headings): {len(body)} characters.")
    print("Temporary synthetic files cleaned. No official dataset downloaded or experiment started.")


if __name__ == "__main__":
    main()
