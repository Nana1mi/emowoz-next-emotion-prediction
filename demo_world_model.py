"""Model-based conditional response scoring demonstration for EmoWOZ checkpoints.

Fixes identified in review:
1. Exact checkpoint restoration (`strict=True`) using the exact `Predictor` architecture
   from `runs/full` (or `runs/controlled_ablation`) without any uninitialized skip projections.
2. Strict alignment between evaluation and demonstration inference: uses the exact model
   logits (and optional single-parameter `logit_offset` if present in the checkpoint) and
   standard `softmax(logits)` without ad-hoc probability mixing.
3. Accurate terminology: presents candidate response comparison as "同一历史下候选系统回复的模型条件预测对比"
   (Model-Based Conditional Response Scoring), explicitly noting that uncalibrated softmax scores
   are not causal risk probabilities.
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parent
LABELS = ["neutral", "fearful", "dissatisfied", "apologetic", "abusive", "excited", "satisfied"]
LABELS_ZH = ["中性", "恐惧/失望", "不满意", "道歉", "辱骂", "兴奋", "满意"]
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Predictor(nn.Module):
    """Exact Predictor architecture matching runs/full and run_controlled_experiments.py."""

    def __init__(self, dimension: int, hidden: int = 256, residual_anchor: bool = False):
        super().__init__()
        self.residual_anchor = residual_anchor
        self.trunk = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Linear(hidden, 7)
        self.future = nn.Linear(hidden, dimension)

    def forward(self, x: torch.Tensor, anchor: torch.Tensor | None = None):
        z = self.trunk(x)
        logits = self.classifier(z)
        raw_future = self.future(z)
        if self.residual_anchor:
            base = x if anchor is None else anchor
            future_vec = F.normalize(base + raw_future, dim=-1)
        else:
            future_vec = F.normalize(raw_future, dim=-1)
        return logits, future_vec


PRESET_SCENARIOS = [
    {
        "title": "示例一：酒店价格超出预算（测试不同系统回复下的模型预测变化）",
        "history_before_system": [
            "User: I am looking for a place to stay in the center of town.",
            "System: I have the University Arms Hotel in the center. It is a 4-star hotel in the expensive price range.",
            "User: That is way too expensive for my budget! Do you not have anything cheaper?",
        ],
        "last_user": "That is way too expensive for my budget! Do you not have anything cheaper?",
        "candidates": [
            {"id": "A", "strategy": "候选回复 A（直接拒绝）", "text": "No, there are no cheap hotels in the center."},
            {"id": "B", "strategy": "候选回复 B（反问是否更换区域）", "text": "Would you like to try another area of town?"},
            {
                "id": "C",
                "strategy": "候选回复 C（致歉并提供平价替代酒店）",
                "text": "I am sorry about that. The Alexander Bed and Breakfast in the center is cheap and has free parking. Would you like me to book it?",
            },
        ],
    },
    {
        "title": "示例二：餐厅预订时段已满（测试不同补救话术下的模型输出）",
        "history_before_system": [
            "User: Please book a table for 4 people at The Missing Sock at 19:00 on Friday.",
            "System: I am sorry, the booking could not be completed because that time slot is full.",
            "User: Why did it fail again? I really need a table for Friday evening!",
        ],
        "last_user": "Why did it fail again? I really need a table for Friday evening!",
        "candidates": [
            {"id": "A", "strategy": "候选回复 A（仅报错）", "text": "The booking failed. Please pick another restaurant."},
            {"id": "B", "strategy": "候选回复 B（仅道歉）", "text": "I apologize that the restaurant is full at 19:00."},
            {
                "id": "C",
                "strategy": "候选回复 C（致歉并提供相邻时段备选）",
                "text": "I am very sorry for the inconvenience. 18:30 at The Missing Sock is available for 4 people, or I can book Finders Keepers at 19:00. Which do you prefer?",
            },
        ],
    },
]


class CheckpointRunner:
    def __init__(self, ckpt_path: Path, device: str = "cuda"):
        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.ckpt_path = self._resolve_ckpt(ckpt_path)
        print(f"Loading checkpoint (strict=True): {self.ckpt_path}", flush=True)
        ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        if ckpt.get("labels", LABELS) != LABELS:
            raise ValueError("Checkpoint label mapping differs from the expected seven classes")
        run_root = self.ckpt_path.parent.parent.parent
        metadata = ckpt.get("inference_config")
        if metadata is None:
            # Original checkpoints lack inference metadata; recover it from their
            # saved config and colocated encoder record, and identify the legacy model.
            config = ckpt.get("config", {})
            meta_path = run_root / "embedding_meta.json"
            signature = json.loads(meta_path.read_text(encoding="utf-8"))["signature"] if meta_path.exists() else {}
            metadata = {"model": signature.get("model", config.get("model", MODEL)),
                        "max_tokens": signature.get("max_tokens", config.get("max_tokens", 256)),
                        "truncation_side": signature.get("truncation_side", "left"),
                        "history_turns": config.get("history_turns", 6), "normalize_embeddings": True,
                        "encoder_precision": "float16" if config.get("device") == "cuda" else "float32",
                        "precision_provenance": "legacy_inferred_from_saved_config",
                        "encoder_revision": signature.get("revision")}
        self.inference_config = metadata
        if ckpt.get("input_field") is not None:
            self.input_field = ckpt["input_field"]
        elif self.ckpt_path.parent.name.startswith("last_user_classifier_"):
            self.input_field = "last"
        elif self.ckpt_path.parent.name.startswith(("history_classifier_", "jepa_")):
            self.input_field = "x"
        else:
            raise ValueError("Legacy checkpoint input type is unknown; use its original named model folder")
        if self.input_field not in ["x", "last"]:
            raise ValueError("Unsupported checkpoint input_field")

        self.residual_anchor = ckpt.get("residual_anchor", None)
        self.logit_offset = np.array(ckpt.get("logit_offset", np.zeros(7)), dtype=np.float32)
        self.logit_adjust_tau = float(ckpt.get("logit_adjust_tau", 0.0))

        self.model = Predictor(
            int(ckpt["dimension"]),
            hidden=int(ckpt.get("hidden", 256)),
            residual_anchor=(self.residual_anchor is not None),
        ).to(self.device)
        # Strict loading verifies weights; preprocessing is restored separately.
        self.model.load_state_dict(ckpt["state_dict"], strict=True)
        self.model.eval()

        from sentence_transformers import SentenceTransformer

        encoder_kwargs = {"device": self.device, "cache_folder": str(ROOT / "data" / "models"), "local_files_only": True}
        if metadata.get("encoder_revision"):
            encoder_kwargs["revision"] = metadata["encoder_revision"]
        self.encoder = SentenceTransformer(metadata["model"], **encoder_kwargs)
        self.encoder.max_seq_length = int(metadata["max_tokens"])
        self.encoder.tokenizer.truncation_side = metadata["truncation_side"]
        if metadata["encoder_precision"] == "float16":
            if self.device != "cuda":
                raise ValueError("This encoder cache used CUDA FP16; choose --device cuda to preserve preprocessing precision")
            self.encoder.half()
        self.encoder.eval()

        # Training-set examples only: this is an illustrative pool, not the formal
        # held-out retrieval evaluation and not newly generated user responses.
        self.candidate_texts = []
        self.candidate_vecs = None
        test_json = run_root / "train.json"
        emb_npz = run_root / "embeddings.npz"
        if test_json.exists() and emb_npz.exists():
            rows = json.loads(test_json.read_text(encoding="utf-8"))
            npz = np.load(emb_npz, allow_pickle=False)
            targets = npz["train_target"]
            first = {}
            for idx, r in enumerate(rows):
                key = r["future_text"].strip().lower()
                if key and key not in first:
                    first[key] = (r["future_text"].strip(), LABELS[r["label"]], targets[idx])
                if len(first) >= 1000:
                    break
            self.candidate_texts = [(v[0], v[1]) for v in first.values()]
            self.candidate_vecs = np.stack([v[2] for v in first.values()], axis=0)

    @staticmethod
    def _resolve_ckpt(path: Path) -> Path:
        if path.is_file():
            return path
        for candidate in [
            path / "models" / "jepa_residual_mp_infonce_seed42" / "best.pt",
            path / "models" / "jepa_seed42" / "best.pt",
            path / "models" / "history_classifier_seed42" / "best.pt",
        ]:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not find best.pt under {path}")

    @torch.no_grad()
    def score_candidates(self, history_lines: list[str], last_user: str, candidates: list[dict]) -> dict:
        if not history_lines or not candidates:
            raise ValueError("Provide history and at least one nonempty system reply")
        canonical = []
        for i, line in enumerate(history_lines):
            role, separator, text = line.partition(":")
            expected = "User" if i % 2 == 0 else "System"
            if not separator or role.strip().lower() != expected.lower() or not text.strip():
                raise ValueError("History must alternate User:/System: and end with User:")
            canonical.append(f"{expected}: {text.strip()}")
        if not canonical[-1].startswith("User:"):
            raise ValueError("Input history must end at the current user turn")
        derived_user = canonical[-1].partition(":")[2].strip()
        if last_user.strip() and last_user.strip() != derived_user:
            raise ValueError("last_user differs from the final user turn")
        last_user = derived_user

        histories = []
        for c in candidates:
            if not c["text"].strip():
                raise ValueError("Empty system reply")
            turns = (canonical + [f"System: {c['text'].strip()}"])[-int(self.inference_config["history_turns"]):]
            histories.append("\n".join(turns))

        x_np = self.encoder.encode(
            histories, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)
        last_np = self.encoder.encode(
            [last_user.strip()], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)
        last_np = np.repeat(last_np, len(candidates), axis=0)

        x_t = torch.from_numpy(x_np).to(self.device)
        anc_t = torch.from_numpy(last_np).to(self.device) if self.residual_anchor == "last" else None
        model_input = torch.from_numpy(last_np).to(self.device) if self.input_field == "last" else x_t
        if self.residual_anchor == "x":
            anc_t = x_t
        logits_t, future_t = self.model(model_input, anc_t)

        raw_logits = logits_t.cpu().numpy()
        adjusted_logits = raw_logits + self.logit_offset[None, :]
        future_vecs = future_t.cpu().numpy()

        # Standard softmax on raw logits (matches training objective)
        shifted = raw_logits - raw_logits.max(axis=1, keepdims=True)
        raw_softmax = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)

        outputs = []
        for i, c in enumerate(candidates):
            raw_pred = int(raw_logits[i].argmax())
            adj_pred = int(adjusted_logits[i].argmax())
            p = raw_softmax[i]

            retrieved = []
            if self.candidate_vecs is not None:
                sims = future_vecs[i] @ self.candidate_vecs.T
                for k_idx in np.argsort(-sims)[:3]:
                    txt, emo = self.candidate_texts[int(k_idx)]
                    retrieved.append({"text": txt, "label": emo, "cosine": round(float(sims[int(k_idx)]), 4)})

            outputs.append(
                {
                    "id": c.get("id", str(i + 1)),
                    "strategy": c.get("strategy", f"候选 {i + 1}"),
                    "system_reply": c["text"],
                    "raw_argmax_emotion": LABELS[raw_pred],
                    "raw_argmax_emotion_zh": LABELS_ZH[raw_pred],
                    "adjusted_argmax_emotion": LABELS[adj_pred],
                    "adjusted_argmax_emotion_zh": LABELS_ZH[adj_pred],
                    "raw_softmax_pct": {LABELS[k]: round(float(p[k]) * 100, 2) for k in range(7)},
                    "raw_logits": {LABELS[k]: round(float(raw_logits[i, k]), 4) for k in range(7)},
                    "retrieved_candidates": retrieved,
                }
            )

        return {
            "checkpoint": str(self.ckpt_path),
            "residual_anchor": self.residual_anchor,
            "logit_adjust_tau": self.logit_adjust_tau,
            "input_field": self.input_field,
            "inference_config": self.inference_config,
            "retrieval_pool": "training_examples; not held-out benchmark or generated responses",
            "encoder_revision_note": "Original encoder revision was not recorded; use the existing local model cache. Strict loading alone does not certify encoder revision.",
            "note": "raw_softmax_pct reflects uncalibrated model softmax outputs; adjusted_argmax_emotion applies validation-selected logit offset.",
            "results": outputs,
        }


def run_cli(runner: CheckpointRunner):
    print("=" * 78)
    print(f"Checkpoint: {runner.ckpt_path}")
    print(f"Settings: residual_anchor={runner.residual_anchor}, logit_adjust_tau={runner.logit_adjust_tau:.3f}")
    print("说明：以下展示同一对话历史拼接不同候选系统回复时的模型条件预测结果（非因果效应证明，未校准Softmax不等同于真实风险概率）。")
    print("=" * 78)
    for sc in PRESET_SCENARIOS:
        print(f"\n>>> {sc['title']}")
        for ln in sc["history_before_system"]:
            print(f"  {ln}")
        res = runner.score_candidates(sc["history_before_system"], sc["last_user"], sc["candidates"])
        for item in res["results"]:
            top_probs = sorted(item["raw_softmax_pct"].items(), key=lambda kv: -kv[1])[:3]
            prob_str = ", ".join(f"{k}={v:.1f}%" for k, v in top_probs)
            print(
                f"  [{item['id']}] {item['strategy']}: \"{item['system_reply']}\"\n"
                f"      原始argmax: {item['raw_argmax_emotion']} ({item['raw_argmax_emotion_zh']}) | "
                f"校准后argmax: {item['adjusted_argmax_emotion']} ({item['adjusted_argmax_emotion_zh']}) | "
                f"Top-3 Softmax: {prob_str}"
            )
            if item["retrieved_candidates"]:
                r0 = item["retrieved_candidates"][0]
                print(f"      Top-1检索候选: \"{r0['text']}\" (标签={r0['label']}, cos={r0['cosine']:.4f})")


HTML_UI = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>多轮对话下一轮情绪与语义预测 — 真实检查点推理演示</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif; margin: 0; padding: 24px; background: #f8fafc; color: #0f172a; }
  .wrap { max-width: 1160px; margin: 0 auto; }
  .banner { background: #fffbeb; border: 1px solid #f59e0b; color: #92400e; padding: 10px 14px; border-radius: 8px; font-size: 13px; margin-bottom: 16px; }
  .grid { display: grid; grid-template-columns: 400px 1fr; gap: 20px; }
  .card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
  label { display: block; font-size: 12.5px; font-weight: 600; color: #475569; margin-bottom: 4px; }
  select, textarea { width: 100%; box-sizing: border-box; padding: 8px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 13px; margin-bottom: 10px; }
  button { width: 100%; background: #2563eb; color: #fff; border: none; padding: 10px; border-radius: 6px; font-weight: 600; cursor: pointer; }
  .item { border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin-bottom: 12px; }
  .bar-wrap { display: flex; align-items: center; gap: 8px; font-size: 12px; margin: 3px 0; }
  .bar-bg { flex: 1; height: 7px; background: #e2e8f0; border-radius: 4px; overflow: hidden; }
  .bar-fg { height: 100%; background: #2563eb; }
</style>
</head>
<body>
<div class="wrap">
  <h2 style="margin:0 0 8px 0;">基于联合嵌入预测架构的多轮对话情感世界模型 — 真实模型推理演示</h2>
  <div class="banner">
    <strong>口径说明：</strong>本页面直接调用本地 PyTorch 检查点（<code>strict=True</code>）与 <code>all-MiniLM-L6-v2</code> 编码器。展示的 Softmax 分布为模型原始条件输出，不代表经过校准的真实概率；候选回复对比反映模型对输入文本变化的条件敏感度，不构成因果效应证明。
  </div>
  <div class="grid">
    <div class="card">
      <label>预设对话示例</label>
      <select id="scSel" onchange="loadSc()"></select>
      <label>多轮对话历史（不含当前系统回复）</label>
      <textarea id="hist" rows="6"></textarea>
      <label>候选系统回复 A</label>
      <textarea id="cA" rows="2"></textarea>
      <label>候选系统回复 B</label>
      <textarea id="cB" rows="2"></textarea>
      <label>候选系统回复 C</label>
      <textarea id="cC" rows="3"></textarea>
      <button onclick="runInfer()">调用 PyTorch 模型推理</button>
    </div>
    <div class="card" id="outBox">正在加载...</div>
  </div>
</div>
<script>
let presets = [];
function esc(value) {
  return String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}
async function init() {
  presets = await (await fetch('/api/presets')).json();
  const sel = document.getElementById('scSel');
  presets.forEach((p, i) => {
    const o = document.createElement('option');
    o.value = i; o.textContent = p.title; sel.appendChild(o);
  });
  loadSc();
  runInfer();
}
function loadSc() {
  const p = presets[document.getElementById('scSel').value];
  document.getElementById('hist').value = p.history_before_system.join('\\n');
  document.getElementById('cA').value = p.candidates[0]?.text || '';
  document.getElementById('cB').value = p.candidates[1]?.text || '';
  document.getElementById('cC').value = p.candidates[2]?.text || '';
}
async function runInfer() {
  const h = document.getElementById('hist').value.split('\\n').filter(x => x.trim());
  const cands = [
    {id: 'A', strategy: '候选回复 A', text: document.getElementById('cA').value},
    {id: 'B', strategy: '候选回复 B', text: document.getElementById('cB').value},
    {id: 'C', strategy: '候选回复 C', text: document.getElementById('cC').value}
  ].filter(c => c.text.trim());
  const res = await (await fetch('/api/score', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({history_before_system: h, last_user: '', candidates: cands})
  })).json();
  if (res.error) { document.getElementById('outBox').textContent = res.error; return; }
  let html = `<div style="font-size:12px;color:#64748b;margin-bottom:10px;">当前加载检查点: <code>${esc(res.checkpoint)}</code> (tau=${res.logit_adjust_tau})。检索候选来自训练集，不是新生成回复。</div>`;
  res.results.forEach(r => {
    const bars = Object.entries(r.raw_softmax_pct).map(([k, v]) => `
      <div class="bar-wrap">
        <span style="width:90px;">${k}</span>
        <div class="bar-bg"><div class="bar-fg" style="width:${Math.min(100, v)}%;"></div></div>
        <span style="width:48px;text-align:right;">${v.toFixed(1)}%</span>
      </div>`).join('');
    const ret = r.retrieved_candidates.map(u => `<div style="font-size:12px;color:#475569;margin-top:3px;">• 检索候选: "${esc(u.text)}" (标签: ${u.label}, cos=${u.cosine})</div>`).join('');
    html += `<div class="item">
      <div><strong>[${esc(r.id)}] ${esc(r.strategy)}</strong> — System: "${esc(r.system_reply)}"</div>
      <div style="font-size:13px;margin:6px 0;">
        原始 argmax: <strong>${r.raw_argmax_emotion} (${r.raw_argmax_emotion_zh})</strong> |
        先验调整后 argmax: <strong>${r.adjusted_argmax_emotion} (${r.adjusted_argmax_emotion_zh})</strong>
      </div>
      ${bars}
      <div style="margin-top:6px;border-top:1px dashed #cbd5e1;padding-top:6px;">${ret}</div>
    </div>`;
  });
  document.getElementById('outBox').innerHTML = html;
}
init();
</script>
</body>
</html>
"""


def serve(runner: CheckpointRunner, port: int):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_GET(self):
            if self.path == "/api/presets":
                body = json.dumps(PRESET_SCENARIOS, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)
                return
            body = HTML_UI.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path == "/api/score":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length < 1 or length > 65536:
                        raise ValueError("Invalid input size")
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                    out = runner.score_candidates(payload.get("history_before_system", []),
                          payload.get("last_user", ""), payload.get("candidates", []))
                except (ValueError, KeyError, TypeError) as error:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(error)}, ensure_ascii=False).encode("utf-8"))
                    return
                body = json.dumps(out, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)

    url = f"http://127.0.0.1:{port}"
    print(f"\nLocal inference web UI running at {url} (Ctrl+C to stop)", flush=True)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=ROOT / "runs" / "full", help="Path to best.pt or run directory")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--cli-only", action="store_true")
    args = p.parse_args()

    runner = CheckpointRunner(args.ckpt.resolve(), device=args.device)
    run_cli(runner)
    if not args.cli_only:
        serve(runner, args.port)


if __name__ == "__main__":
    main()
