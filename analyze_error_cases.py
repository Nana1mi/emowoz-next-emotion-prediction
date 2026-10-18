"""Read-only alignment and error-case analysis; never trains or selects model configurations."""
import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

import run_controlled_experiments as common

ROOT = Path(__file__).resolve().parent
FIELDS = ["dialogue_id", "target_turn", "domain", "history", "last_user", "future_text", "label", "previous_label"]
MODEL_PATHS = {
    "history_ce": ("controlled_v2", "history_classifier"),
    "history_focal": ("controlled_v2", "route1_focal_only"),
    "jepa_w1": ("controlled_v2", "jepa_w1_0"),
    "posthoc_target": ("posthoc_diagnostic", "posthoc_target_only_ce"),
    "posthoc_context_target": ("posthoc_diagnostic", "posthoc_history_target_ce"),
}


def read_predictions(path):
    import json
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def normalize(text):
    return " ".join(text.casefold().split())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/error_analysis")
    args = parser.parse_args(argv)
    args.out = args.out.resolve()
    for name in ["full", "controlled_v2", "posthoc_diagnostic"]:
        common.protect_paths(args.out, ROOT / "runs" / name)
    rows = common.read(ROOT / "runs/full/test.json")
    keys = [(r["dialogue_id"], r["target_turn"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate test sample identity")
    seeds = common.read(ROOT / "runs/controlled_v2/run_config.json")["training"]["seeds"]
    if seeds != common.read(ROOT / "runs/posthoc_diagnostic/run_config.json")["seeds"]:
        raise ValueError("Diagnostic and predictive seeds differ")
    predictions, source_hashes, per_seed = {}, {}, {}
    source_hashes["test.json"] = common.digest(ROOT / "runs/full/test.json")
    source_hashes["train.json"] = common.digest(ROOT / "runs/full/train.json")
    for model, (directory, prefix) in MODEL_PATHS.items():
        predictions[model], per_seed[model] = [], []
        for seed in seeds:
            folder = ROOT / "runs" / directory / "models" / f"{prefix}_seed{seed}"
            path = folder / "test_predictions.jsonl"
            checksum = common.digest(path)
            completion = common.read(folder / "completion.json")
            if completion["status"] != "completed" or completion["artifacts"][path.name] != checksum:
                raise ValueError(f"Unverified prediction file: {path}")
            source_hashes[str(path.relative_to(ROOT))] = checksum
            records = read_predictions(path)
            if len(records) != len(rows):
                raise ValueError("Prediction sample counts differ")
            labels = []
            for expected, record in zip(rows, records):
                if any(record[field] != expected[field] for field in FIELDS):
                    raise ValueError(f"Misaligned or altered test sample: {path}")
                if record["true_emotion"] != common.LABELS[expected["label"]]:
                    raise ValueError("True-label mapping mismatch")
                labels.append(common.LABELS.index(record["predicted_emotion"]))
            predictions[model].append(labels)
            diagnostics = {}
            for i, label in enumerate(common.LABELS):
                actual_indices = [j for j, row in enumerate(rows) if row["label"] == i]
                diagnoses = Counter(common.LABELS[labels[j]] for j in actual_indices)
                diagnostics[label] = {"support": len(actual_indices), "predicted_as": dict(diagnoses),
                                      "correct": diagnoses[label]}
            per_seed[model].append({"seed": seed, "by_true_class": diagnostics})
    train_text_labels = defaultdict(Counter)
    for row in common.read(ROOT / "runs/full/train.json"):
        train_text_labels[normalize(row["future_text"])][common.LABELS[row["label"]]] += 1

    def all_correct(model, index):
        return all(values[index] == rows[index]["label"] for values in predictions[model])

    def all_wrong(model, index):
        return all(values[index] != rows[index]["label"] for values in predictions[model])

    groups = defaultdict(lambda: defaultdict(list))
    group_rules = {
        "history_wrong_target_correct": lambda i: all_wrong("history_ce", i) and all_correct("posthoc_target", i),
        "history_and_target_both_wrong": lambda i: all_wrong("history_ce", i) and all_wrong("posthoc_target", i),
        "history_and_target_both_correct": lambda i: all_correct("history_ce", i) and all_correct("posthoc_target", i),
        "context_rescues_target": lambda i: all_wrong("history_ce", i) and all_wrong("posthoc_target", i) and all_correct("posthoc_context_target", i),
        "history_correct_target_wrong": lambda i: all_correct("history_ce", i) and all_wrong("posthoc_target", i),
        "focal_gain": lambda i: all_wrong("history_ce", i) and all_correct("history_focal", i),
        "focal_loss": lambda i: all_correct("history_ce", i) and all_wrong("history_focal", i),
        "jepa_gain": lambda i: all_wrong("history_ce", i) and all_correct("jepa_w1", i),
        "jepa_loss": lambda i: all_correct("history_ce", i) and all_wrong("jepa_w1", i),
    }
    for index, row in enumerate(rows):
        for name, rule in group_rules.items():
            if rule(index):
                groups[name][common.LABELS[row["label"]]].append(index)
    class_summary = {}
    for c, label in enumerate(common.LABELS):
        indices = [i for i, row in enumerate(rows) if row["label"] == c]
        class_summary[label] = {"support": len(indices),
            "all_seed_correct": {model: sum(all_correct(model, i) for i in indices) for model in predictions},
            "all_seed_wrong": {model: sum(all_wrong(model, i) for i in indices) for model in predictions},
            "groups": {name: len(groups[name][label]) for name in group_rules}}

    def case(index):
        row = rows[index]
        return {**row, "true_emotion": common.LABELS[row["label"]], "previous_emotion": common.LABELS[row["previous_label"]],
                "sample_key": f"{row['dialogue_id']}:{row['target_turn']}",
                "predictions": {model: {str(seed): common.LABELS[values[index]] for seed, values in zip(seeds, runs)}
                                for model, runs in predictions.items()},
                "same_future_text_training_labels": dict(train_text_labels[normalize(row["future_text"])])}

    rng = random.Random(2026)
    selected = {name: {label: [case(i) for i in rng.sample(indices, min(2, len(indices)))]
                      for label, indices in by_class.items()}
                for name, by_class in groups.items()}
    bundle = {"purpose": "descriptive test error analysis; not evidence of causal or inherent unpredictability",
              "sampling_seed": 2026, "seed_rule": "all configured seeds must satisfy each correctness condition",
              "seeds": seeds, "samples": len(rows), "source_sha256": source_hashes,
              "by_true_class": class_summary,
              "group_counts": {name: {label: len(indices) for label, indices in group.items()} for name, group in groups.items()},
              "per_seed_confusions": per_seed}
    signature = {"source_sha256": source_hashes, "script_sha256": common.digest(Path(__file__)), "sampling_seed": 2026}
    manifest = args.out / "provenance.json"
    if manifest.exists() and common.read(manifest) != signature:
        raise ValueError("Analysis sources/code changed; use a new output directory")
    if args.out.exists() and any(args.out.iterdir()) and not manifest.exists():
        raise ValueError("Nonempty analysis output lacks provenance")
    args.out.mkdir(parents=True, exist_ok=True)
    common.dump(manifest, signature)
    common.dump(args.out / "statistics.json", bundle)
    common.dump(args.out / "candidate_cases.json", selected)
    # Full rare-class inspection prevents relying only on convenient sampled cases.
    rare = [case(i) for i, row in enumerate(rows) if row["label"] in [1, 3, 4, 5]]
    common.dump(args.out / "all_rare_cases.json", rare)
    print(f"PASS: {len(rows)} samples aligned across {len(predictions)} models x {len(seeds)} seeds.")
    print(f"Saved descriptive statistics, deterministic candidate samples and all {len(rare)} rare-class cases to {args.out}")
    print("No model training, inference or source-file modification performed.")


if __name__ == "__main__":
    main()
