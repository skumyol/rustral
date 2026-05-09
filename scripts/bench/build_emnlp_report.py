#!/usr/bin/env python3
"""
Assemble benchmark JSON into a single Markdown report (tables + EMNLP-oriented draft text).

Usage (from repo root):
  python3 scripts/bench/build_emnlp_report.py \\
    --micro benchmarks/results/emnlp_micro_*.json \\
    [--rust-cuda benchmarks/results/emnlp_rust_cuda_*.json] \\
    [--nlp-dir benchmarks/runs/v0.1.0/nlp] \\
    --out docs/emnlp_experiments_report.md
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _iter_suites(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "suites" in doc and isinstance(doc["suites"], list):
        return [b for b in doc["suites"] if isinstance(b, dict)]
    if "samples" in doc:
        return [{"suite": doc.get("suite", "?"), "samples": doc.get("samples", [])}]
    return []


def table_micro(paths: List[Path]) -> str:
    lines: List[str] = [
        "### Table A. Micro-benchmarks (wall-clock ms)",
        "",
        "Rows share workload `name` and `params` across stacks. Mean and 95% CI on `runs_ms` where present.",
        "",
        "| Workload | Params | Suite | Backend | Device | Mean (ms) | 95% CI | Std |",
        "|----------|--------|-------|---------|--------|-----------|--------|-----|",
    ]
    for path in paths:
        if not path.exists():
            continue
        doc = _read_json(path)
        try:
            rel = path.relative_to(REPO_ROOT)
        except ValueError:
            rel = path
        for block in _iter_suites(doc):
            sname = block.get("suite", "?")
            for s in block.get("samples") or []:
                if not isinstance(s, dict):
                    continue
                name = s.get("name", "?")
                params = s.get("params") or {}
                pstr = " ".join(f"{k}={v}" for k, v in sorted(params.items())) if params else "—"
                mean = float(s.get("mean_ms", 0))
                std = float(s.get("std_ms", 0))
                ci = "—"
                lo, hi = s.get("ci95_low_ms"), s.get("ci95_high_ms")
                if lo is not None and hi is not None:
                    ci = f"[{float(lo):.3f}, {float(hi):.3f}]"
                lines.append(
                    f"| `{name}` | {pstr} | {sname} | {s.get('backend', '?')} | "
                    f"{s.get('device', '?')} | {mean:.3f} | {ci} | {std:.3f} |"
                )
        lines.append("")
        lines.append(f"*Source file: `{rel}`*")
        lines.append("")
    return "\n".join(lines)


def _nlp_aggregate_table(
    title: str,
    rust_p: Path,
    pt_p: Optional[Path],
    metric_rust: str,
    metric_pt: str,
) -> str:
    lines = [f"### {title}", "", "| Stack | Mean ± Std | n | Curated file |", "|-------|------------|---|--------------|"]
    if not rust_p.exists():
        lines.append(f"| Rustral | — | — | *missing `{rust_p}`* |")
    else:
        rj = _read_json(rust_p)
        agg = rj.get("aggregate", {})
        m, sd = float(agg.get("mean", 0)), float(agg.get("std", 0))
        n = agg.get("n", "?")
        lines.append(
            f"| **Rustral** | {m:.4f} ± {sd:.4f} | {n} | `{rust_p.relative_to(REPO_ROOT)}` |"
        )
    if pt_p and pt_p.exists():
        pj = _read_json(pt_p)
        agg = pj.get("aggregate", {})
        m, sd = float(agg.get("mean", 0)), float(agg.get("std", 0))
        n = agg.get("n", "?")
        lines.append(
            f"| **PyTorch** | {m:.4f} ± {sd:.4f} | {n} | `{pt_p.relative_to(REPO_ROOT)}` |"
        )
    elif pt_p:
        lines.append(f"| **PyTorch** | — | — | *missing `{pt_p}`* |")
    lines.append("")
    return "\n".join(lines)


def _hardware_table(paths: List[Path]) -> str:
    lines = [
        "### Table 0. Harness environment (from JSON `machine` blocks)",
        "",
        "| Source JSON | Suite | OS | Arch | Host | Extras |",
        "|-------------|-------|----|------|------|--------|",
    ]
    for path in paths:
        if not path.exists():
            continue
        doc = _read_json(path)
        try:
            rel = str(path.relative_to(REPO_ROOT))
        except ValueError:
            rel = str(path)
        for block in _iter_suites(doc):
            m = block.get("machine") or {}
            if not isinstance(m, dict):
                continue
            extras: List[str] = []
            if m.get("torch_version"):
                extras.append(f"torch={m['torch_version']}")
            if m.get("python"):
                extras.append(f"py={m['python']}")
            if m.get("cuda_device_name"):
                extras.append(str(m["cuda_device_name"])[:40])
            if m.get("rustc") and m["rustc"] != "n/a":
                extras.append(f"rustc={m['rustc'][:20]}")
            lines.append(
                f"| `{rel}` | {block.get('suite', '?')} | {m.get('os', '?')} | "
                f"{m.get('arch', '?')} | {m.get('hostname', '?')} | {'; '.join(extras) or '—'} |"
            )
    lines.append("")
    return "\n".join(lines)


def _nlp_per_seed_table(rust_p: Path, pt_p: Optional[Path], key_r: str, key_p: str) -> str:
    lines = ["| Seed | Rustral | PyTorch | Δ |", "|------|---------|---------|---|"]
    if not rust_p.exists():
        return ""
    rj = _read_json(rust_p)
    pj = _read_json(pt_p) if pt_p and pt_p.exists() else {}
    r_by = {int(x["seed"]): x["manifest"] for x in rj.get("runs", []) if "seed" in x}
    p_by = {int(x["seed"]): x["manifest"] for x in pj.get("runs", []) if "seed" in x} if pj else {}
    for seed in sorted(set(r_by) | set(p_by)):
        rv = r_by.get(seed, {}).get(key_r)
        pv = p_by.get(seed, {}).get(key_p)
        d = "—"
        if rv is not None and pv is not None:
            try:
                d = f"{float(rv) - float(pv):+.4f}"
            except (TypeError, ValueError):
                pass
        lines.append(f"| {seed} | {rv} | {pv} | {d} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--micro", type=Path, action="append", default=[], help="Combined harness JSON (repeatable)")
    ap.add_argument("--rust-cuda", type=Path, default=None, help="Optional rustral-cuda JSON")
    ap.add_argument("--nlp-dir", type=Path, default=None, help="Directory with sst2.json, wikitext2.json, *_pytorch.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    micro_paths = [p.resolve() for p in args.micro if p]
    if args.rust_cuda:
        micro_paths.append(args.rust_cuda.resolve())

    nlp_dir = args.nlp_dir
    if nlp_dir:
        nlp_dir = nlp_dir.resolve()

    git_sha = "unknown"
    try:
        import subprocess

        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        pass

    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    parts: List[str] = [
        "# Experiments summary for EMNLP-style reporting",
        "",
        f"*Generated {stamp} UTC. Repository HEAD: `{git_sha}`.*",
        "",
        "This document aggregates **micro-benchmarks** (operator timing, schema v2) and **NLP task** "
        "results (SST-2, WikiText-2) where available. Numbers are suitable as a **system paper** or "
        "**reproducibility appendix** draft; tighten claims after internal review.",
        "",
        "---",
        "",
        "## 1. Abstract (draft)",
        "",
        "We report end-to-end and operator-level evaluation of Rustral, a Rust-first differentiable "
        "stack, against Candle (Rust) and PyTorch baselines on shared micro-workloads (matmul, "
        "attention, convolution) and on word-level SST-2 classification and WikiText-2 language "
        "modelling with matched tokenization and vocabulary. Timings include repeated runs with "
        "95% confidence intervals where applicable; NLP aggregates report mean ± std over seeds. "
        "All artifacts are emitted as versioned JSON for independent verification.",
        "",
        "## 2. Experimental setup",
        "",
        "- **Micro-benchmarks**: `scripts/bench/run_all.py`, warmup + repeats as in each JSON; "
        "PyTorch CUDA uses device synchronization around timed regions.",
        "- **NLP**: `scripts/eval/run_nlp_real.py`; `--benchmark` = small model for fast parity; "
        "curated manifests under `benchmarks/runs/<version>/nlp/` record hyperparameters and data caps.",
        "- **Reproducibility**: See `EVALUATION.md`, `BENCHMARKS.md`, and per-run `manifest.json` / schema-v2 `machine` blocks.",
        "",
        "## 3. Results",
        "",
    ]

    parts.append(_hardware_table(micro_paths))
    parts.append(table_micro(micro_paths))

    if nlp_dir and nlp_dir.is_dir():
        parts.append("## 4. NLP task results (curated aggregates)")
        parts.append("")
        parts.append(
            f"*NLP directory: `{nlp_dir.relative_to(REPO_ROOT)}` — see manifests inside each JSON for "
            "exact model width, data caps, and tokenizer settings.*"
        )
        parts.append("")
        sst_r = nlp_dir / "sst2.json"
        sst_p = nlp_dir / "sst2_pytorch.json"
        wt_r = nlp_dir / "wikitext2.json"
        wt_p = nlp_dir / "wikitext2_pytorch.json"
        parts.append(_nlp_aggregate_table("Table B. SST-2 (development accuracy)", sst_r, sst_p, "dev_accuracy", "dev_accuracy"))
        parts.append("#### Per-seed SST-2")
        parts.append("")
        parts.append(_nlp_per_seed_table(sst_r, sst_p, "dev_accuracy", "dev_accuracy"))
        parts.append(_nlp_aggregate_table("Table C. WikiText-2 (development perplexity)", wt_r, wt_p, "dev_perplexity", "dev_perplexity"))
        parts.append("#### Per-seed WikiText-2")
        parts.append("")
        parts.append(_nlp_per_seed_table(wt_r, wt_p, "dev_perplexity", "dev_perplexity"))
    else:
        parts.append("## 4. NLP task results")
        parts.append("")
        parts.append("*No `--nlp-dir` provided or directory missing.*")
        parts.append("")

    parts.extend(
        [
            "## 5. Discussion and limitations (draft)",
            "",
            "- **Scope**: Micro-benchmarks isolate operators; they do not replace end-to-end training efficiency studies.",
            "- **Fairness**: Compare across rows with identical `name` and `params`; CPU vs GPU rows are not directly comparable without stating device.",
            "- **NLP**: `--benchmark` uses small models and caps; paper-scale runs use `--paper` and larger caps (see `EVALUATION.md`).",
            "- **Variance**: CI width reflects run-to-run noise; for publication, report hardware, library versions, and fixed seeds from JSON.",
            "",
            "## 6. Checklist before submission",
            "",
            "- [ ] Attach or cite exact JSON paths and git SHA.",
            "- [ ] Confirm shared vocabulary paths for Rustral vs PyTorch NLP rows.",
            "- [ ] State train/eval caps for WikiText-2.",
            "- [ ] Run `python3 scripts/bench/validate_schema.py` on all harness JSON used in tables.",
            "",
            "---",
            "",
            "## 7. EMNLP-oriented paper draft (system / reproducibility track)",
            "",
            "### Suggested title",
            "",
            "*Rustral: A Rust-Centric Differentiable Stack with Reproducible NLP and Operator Benchmarks*",
            "",
            "### Authors",
            "",
            "*[Author list and affiliations — placeholder]*",
            "",
            "### Abstract (submission length, ~200 words)",
            "",
            "We present Rustral, an open-source differentiable programming stack implemented primarily in "
            "Rust, with explicit attention to reproducible evaluation. We release a unified JSON schema for "
            "micro-benchmarks that compares Rustral's ndarray CPU backend, a Candle-based Rust baseline, "
            "and PyTorch on CPU and CUDA over matched operator workloads (matrix multiply, multi-head "
            "attention-style computation, and convolution). Timings report repeated wall-clock measurements "
            "with 95% confidence intervals. Complementing operator data, we evaluate word-level SST-2 "
            "sentiment classification and WikiText-2 language modeling with shared vocabularies and "
            "documented data caps, including PyTorch mirrors for stack parity. All experiments are driven "
            "by scripts that pin dataset checksums, emit per-run manifests, and integrate with continuous "
            "integration for schema validation. Our goal is not leaderboard placement on these small "
            "baselines, but transparent, citeable numbers suitable for systems-oriented NLP venues such "
            "as EMNLP. We discuss limitations—including word-level tokenization, modest model scale in "
            "default presets, and the gap between forward-only and full tape training for some layers—and "
            "outline a roadmap toward larger models and additional frameworks.",
            "",
            "### 1. Introduction",
            "",
            "Systems papers at EMNLP increasingly require evidence that a new implementation is not only "
            "correct but also competitive and reproducible. Rustral targets researchers who want native "
            "Rust performance and safety while retaining familiar autodiff ergonomics. This draft "
            "grounds claims in **paired experiments**: the same workload names and tensor shapes across "
            "backends, and matched NLP tasks against PyTorch with shared vocabulary files.",
            "",
            "### 2. Method: tasks and metrics",
            "",
            "**Micro-benchmarks.** We follow schema v2 (`benchmarks/SCHEMA.md`): each sample records "
            "`runs_ms`, aggregates, optional 95% CIs, and machine metadata. GPU timings synchronize the "
            "device before and after each timed region.",
            "",
            "**NLP.** SST-2 accuracy is measured on the development split; WikiText-2 perplexity is "
            "`exp(mean cross-entropy in nats)` over capped evaluation windows, as documented in "
            "`EVALUATION.md`.",
            "",
            "### 3. Experiments (summary)",
            "",
            "Table 0 summarizes hardware and software captured in JSON. Table A lists operator timings "
            "across stacks. Tables B–C summarize NLP aggregates; per-seed rows support variance reporting.",
            "",
            "### 4. Analysis (draft bullets)",
            "",
            "- Compare **within device class** (CPU vs CPU, GPU vs GPU) when discussing speedups.",
            "- Use CIs to avoid over-interpreting single runs.",
            "- For NLP, verify that `vocab.txt` paths match between Rustral and PyTorch rows before claiming parity.",
            "",
            "### 5. Ethics and reproducibility",
            "",
            "Public datasets (SST-2, WikiText-2) are accessed via pinned URLs and hashes in `rustral-data`. "
            "No additional annotation or crowd-sourcing is introduced. Full commands to regenerate this "
            "document are given in `BENCHMARKS.md`.",
            "",
            "### 6. References (placeholder)",
            "",
            "- Socher et al., SST / sentiment (cite standard SST-2 reference used by the community).",
            "- Merity et al., WikiText-2.",
            "- Paszke et al., PyTorch.",
            "",
            "---",
            "",
            "## 8. Input artifacts (this run)",
            "",
        ]
    )

    for p in micro_paths:
        try:
            parts.append(f"- `{p.relative_to(REPO_ROOT)}`")
        except ValueError:
            parts.append(f"- `{p}`")
    if nlp_dir and nlp_dir.is_dir():
        parts.append(f"- `{nlp_dir.relative_to(REPO_ROOT)}/` (NLP curated JSONs)")

    parts.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
