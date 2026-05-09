#!/usr/bin/env python3
"""
Build a comparative analysis Markdown report: Rustral vs PyTorch (NLP) and
micro-benchmark harness JSON (rustral / candle / pytorch / pytorch-cuda / GPU Rust suites).

Usage:
  python3 scripts/bench/comparative_report.py
  python3 scripts/bench/comparative_report.py --runs-version 0.2.0
  python3 scripts/bench/comparative_report.py --harness benchmarks/results/queue-foo.json \\
      --harness-extra benchmarks/results/queue-foo-pytorch.json

Output: benchmarks/reports/comparative_<runs_version>_<date>.md (or --out path)
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return (float("nan"), float("nan"))
    if len(xs) == 1:
        return (float(xs[0]), 0.0)
    return (float(statistics.mean(xs)), float(statistics.stdev(xs)))


def _fmt_pm(mean: float, std: float, nd: int = 4) -> str:
    if mean != mean:  # nan
        return "n/a"
    if std == 0 or std != std:
        return f"{mean:.{nd}f}"
    return f"{mean:.{nd}f} ± {std:.{nd}f}"


def _nlp_section(
    title: str,
    rust_path: Optional[Path],
    pt_path: Optional[Path],
    metric_rust_key: str,
    metric_pt_key: str,
) -> str:
    lines: List[str] = [f"## {title}", ""]
    if not rust_path or not rust_path.exists():
        lines.append(f"*Rustral aggregate not found: `{rust_path}`*")
        lines.append("")
        return "\n".join(lines)
    if not pt_path or not pt_path.exists():
        lines.append(f"*PyTorch aggregate not found: `{pt_path}`*")
        lines.append("")
        rust = _read_json(rust_path)
        agg = rust.get("aggregate", {})
        lines.append(f"- **Rustral** ({rust_path.name}): {agg.get('metric', '?')} "
                     f"mean={agg.get('mean')}, std={agg.get('std')}, n={agg.get('n')}")
        lines.append("")
        return "\n".join(lines)

    rust = _read_json(rust_path)
    pt = _read_json(pt_path)
    r_agg = rust.get("aggregate", {})
    p_agg = pt.get("aggregate", {})

    lines.append("| Stack | Metric | Mean ± Std | n | Notes |")
    lines.append("|-------|--------|------------|---|-------|")
    lines.append(
        f"| **Rustral** | {r_agg.get('metric', '?')} | "
        f"{_fmt_pm(float(r_agg.get('mean', 0)), float(r_agg.get('std', 0)), 4)} | "
        f"{r_agg.get('n', '?')} | curated `{rust_path.relative_to(REPO_ROOT)}` |"
    )
    lines.append(
        f"| **PyTorch** | {p_agg.get('metric', '?')} | "
        f"{_fmt_pm(float(p_agg.get('mean', 0)), float(p_agg.get('std', 0)), 4)} | "
        f"{p_agg.get('n', '?')} | curated `{pt_path.relative_to(REPO_ROOT)}` |"
    )
    lines.append("")

    # Per-seed detail
    lines.append("### Per-seed (headline metric)")
    lines.append("")
    lines.append("| Seed | Rustral | PyTorch | Δ |")
    lines.append("|------|---------|---------|---|")
    r_by_seed = {int(r["seed"]): r["manifest"] for r in rust.get("runs", [])}
    p_by_seed = {int(r["seed"]): r["manifest"] for r in pt.get("runs", [])}
    for seed in sorted(set(r_by_seed) | set(p_by_seed)):
        rm = r_by_seed.get(seed, {})
        pm = p_by_seed.get(seed, {})
        rv = rm.get(metric_rust_key)
        pv = pm.get(metric_pt_key)
        if rv is None:
            rv = rm.get("dev_accuracy", rm.get("dev_perplexity"))
        if pv is None:
            pv = pm.get("dev_accuracy", pm.get("dev_perplexity"))
        try:
            delta = float(rv) - float(pv) if rv is not None and pv is not None else None
        except (TypeError, ValueError):
            delta = None
        d_str = f"{delta:+.4f}" if delta is not None else "—"
        lines.append(f"| {seed} | {rv} | {pv} | {d_str} |")
    lines.append("")

    # Provenance from first Rustral run
    r0 = rust.get("runs", [{}])[0].get("manifest", {})
    lines.append("### Rustral run provenance (seed 0 excerpt)")
    lines.append("")
    for k in (
        "task",
        "model_type",
        "seq_len",
        "block_size",
        "d_model",
        "num_heads",
        "ffn_dim",
        "num_layers",
        "vocab_size",
        "total_params",
        "epochs",
        "batch_size",
        "learning_rate",
        "train_examples",
        "train_tokens_used",
        "paper_mode",
        "quick_mode",
    ):
        if k in r0:
            lines.append(f"- `{k}`: {r0[k]}")
    lines.append("")
    return "\n".join(lines)


def _resolve_harness_paths(harness: Optional[Path], extras: List[Path]) -> List[Path]:
    raw: List[Optional[Path]] = []
    if harness is not None:
        raw.append(harness)
    raw.extend(extras)
    out: List[Path] = []
    for p in raw:
        if p is None:
            continue
        path = p if p.is_absolute() else (REPO_ROOT / p).resolve()
        if path.exists():
            out.append(path)
    return out


def _harness_section(paths: List[Path]) -> str:
    lines: List[str] = ["## Micro-benchmarks (schema v2 harness)", ""]
    if not paths:
        lines.append("*No harness JSON provided or files missing.*")
        lines.append("")
        lines.append(
            "Generate with: `python3 scripts/bench/run_all.py --suite rustral --suite candle "
            "[--suite pytorch] [--suite pytorch-cuda] [--suite rustral-cuda] --out …`"
        )
        lines.append("")
        return "\n".join(lines)

    lines.append("### Methodology (operator timing)")
    lines.append("")
    lines.append(
        "- **Wall-clock ms** per repeat; **CPU** timings use `time.perf_counter()` around the op."
    )
    lines.append(
        "- **PyTorch CUDA**: `torch.cuda.synchronize()` before/after each timed region so "
        "async launch does not skew means (standard practice for academic GPU micro-benchmarks)."
    )
    lines.append(
        "- **Rustral `rustral-cuda` / `rustral-metal`**: same schema; compare only within "
        "the same device class (CPU vs CPU, GPU vs GPU)."
    )
    lines.append(
        "- **Frameworks**: ndarray CPU and Candle CPU vs PyTorch CPU/CUDA — apples-to-apples on "
        "`name` + `params`, not across different op fusion or autograd paths."
    )
    lines.append("")

    lines.append("| Workload | Suite | Backend | Device | Mean (ms) | 95% CI | Std (ms) | JSON |")
    lines.append("|----------|-------|---------|--------|-----------|--------|----------|------|")
    for path in paths:
        doc = _read_json(path)
        rel = path.relative_to(REPO_ROOT)
        raw_suites = doc.get("suites")
        if isinstance(raw_suites, list) and raw_suites:
            blocks = raw_suites
        elif isinstance(doc.get("samples"), list):
            blocks = [{"suite": doc.get("suite", "?"), "samples": doc["samples"]}]
        else:
            blocks = []
        for block in blocks:
            sname = block.get("suite", "?")
            for sample in block.get("samples", []):
                name = sample.get("name", "?")
                backend = sample.get("backend", "?")
                dev = sample.get("device", "?")
                mean = float(sample.get("mean_ms", 0))
                std = float(sample.get("std_ms", 0))
                lo = sample.get("ci95_low_ms")
                hi = sample.get("ci95_high_ms")
                if lo is not None and hi is not None:
                    ci_cell = f"[{float(lo):.3f}, {float(hi):.3f}]"
                else:
                    ci_cell = "—"
                lines.append(
                    f"| {name} | {sname} | {backend} | {dev} | {mean:.3f} | {ci_cell} | {std:.3f} | `{rel}` |"
                )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-version", default="0.2.0", help="NLP curated folder v<version> (default 0.2.0)")
    ap.add_argument("--fallback-version", default="0.1.0", help="if v0.2.0 nlp missing, try this")
    ap.add_argument("--harness", type=Path, default=None, help="primary run_all.py output JSON (e.g. rustral+candle)")
    ap.add_argument(
        "--harness-extra",
        type=Path,
        action="append",
        default=[],
        help="additional harness JSON files (e.g. pytorch stack, rustral-cuda); repeatable",
    )
    ap.add_argument("--out", type=Path, default=None, help="output Markdown path")
    args = ap.parse_args()

    nlp_root = REPO_ROOT / "benchmarks" / "runs" / f"v{args.runs_version}" / "nlp"
    if not nlp_root.is_dir() or not (nlp_root / "sst2.json").exists():
        nlp_root = REPO_ROOT / "benchmarks" / "runs" / f"v{args.fallback_version}" / "nlp"
        ver_note = args.fallback_version
    else:
        ver_note = args.runs_version

    sst2_r = nlp_root / "sst2.json"
    sst2_p = nlp_root / "sst2_pytorch.json"
    wt_r = nlp_root / "wikitext2.json"
    wt_p = nlp_root / "wikitext2_pytorch.json"

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = args.out
    if out is None:
        REPO_ROOT.joinpath("benchmarks/reports").mkdir(parents=True, exist_ok=True)
        out = REPO_ROOT / "benchmarks/reports" / f"comparative_v{ver_note}_{stamp}.md"

    parts: List[str] = [
        "# Rustral comparative analysis",
        "",
        f"Generated `{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}` (UTC).",
        f"NLP aggregates read from `benchmarks/runs/v{ver_note}/nlp/` "
        f"(requested `{args.runs_version}`, fallback `{args.fallback_version}` if needed).",
        "",
        "> Use this as a starting draft for the paper: add statistical tests, confidence intervals,",
        "> and ablations as needed. Verify that Rustral and PyTorch runs share vocabulary and",
        "> hyperparameters before claiming parity.",
        "",
    ]

    parts.append(_nlp_section("SST-2 (dev accuracy)", sst2_r, sst2_p, "dev_accuracy", "dev_accuracy"))
    parts.append(_nlp_section("WikiText-2 (dev perplexity)", wt_r, wt_p, "dev_perplexity", "dev_perplexity"))

    harness_path = args.harness
    if harness_path and not harness_path.is_absolute():
        harness_path = (REPO_ROOT / harness_path).resolve()
    extra_paths: List[Path] = []
    for h in args.harness_extra or []:
        hp = h if h.is_absolute() else (REPO_ROOT / h).resolve()
        extra_paths.append(hp)
    h_paths = _resolve_harness_paths(harness_path, extra_paths)
    parts.append(_harness_section(h_paths))

    parts.append("## Checklist for the paper")
    parts.append("")
    parts.append("- [ ] Same tokenizer vocabulary file for Rustral and PyTorch rows being compared.")
    parts.append("- [ ] Same train token cap / eval window cap documented (WikiText-2).")
    parts.append("- [ ] Hardware (CPU/GPU), batch size, and wall-clock noted.")
    parts.append(
        "- [ ] Micro-benchmark table: compare same `name`+`params` only; GPU rows use synchronized timing."
    )
    parts.append("- [ ] Learning-rate schedule: document if warmup differs between stacks.")
    parts.append("")

    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
