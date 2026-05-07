#!/usr/bin/env python3
"""
Render a static HTML benchmark dashboard from `benchmarks/runs/`.

Produces:
    docs/site/index.html          — landing page with per-version dropdown
    docs/site/<version>.html      — one page per snapshot, rendered from
                                    benchmarks/runs/<version>/*.json
    docs/site/data/<version>.json — copy of the per-suite JSONs flattened so the page can
                                    fetch them at load time

This is a deliberately lightweight, dependency-free renderer: pure stdlib + a single
inline `<table>` per page. The output is committed when releases happen and deployed via
[`.github/workflows/pages.yml`](../../.github/workflows/pages.yml).

If `benchmarks/runs/` is empty (no snapshots yet) the script still emits a valid
landing page that says "no snapshots yet" so the Pages deploy never fails on a fresh
checkout.

Usage:
    python3 scripts/bench/render_site.py
    python3 scripts/bench/render_site.py --out docs/site
"""
from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "benchmarks" / "runs"
DEFAULT_OUT = REPO_ROOT / "docs" / "site"


def discover_versions() -> List[Tuple[str, Path]]:
    """Return [(version, dir)] sorted by version (lexicographic; release tags sort fine)."""
    if not RUNS_DIR.exists():
        return []
    out: List[Tuple[str, Path]] = []
    for entry in sorted(RUNS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "manifest.json").exists():
            out.append((entry.name, entry))
    return out


def load_suite_files(version_dir: Path) -> List[Dict]:
    """Load every per-suite JSON in the version directory."""
    out: List[Dict] = []
    for f in sorted(version_dir.glob("*.json")):
        if f.name == "manifest.json":
            continue
        try:
            doc = json.loads(f.read_text())
        except json.JSONDecodeError as exc:  # pragma: no cover - bad data
            print(f"warn: skipping {f}: {exc}", file=sys.stderr)
            continue
        out.append(doc)
    return out


def render_landing(versions: List[Tuple[str, Path]]) -> str:
    """Render the index.html landing page with a version dropdown."""
    options = "\n".join(
        f'      <li><a href="{html.escape(v)}.html">{html.escape(v)}</a></li>'
        for v, _ in versions
    )
    if not versions:
        body_versions = "<p><em>No snapshots have been captured yet.</em></p>"
    else:
        body_versions = f"""
    <h2>Snapshots</h2>
    <ul>
{options}
    </ul>
"""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Rustral benchmarks</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body {{ font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 980px; margin: 2em auto; padding: 0 1em; color: #1a1a1a; }}
    h1 {{ font-size: 1.6em; }}
    h2 {{ font-size: 1.2em; margin-top: 2em; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 0.5em; }}
    th, td {{ border-bottom: 1px solid #e0e0e0; padding: 6px 10px; text-align: left; }}
    th {{ background: #f7f7f7; }}
    code {{ background: #f3f3f3; padding: 0 4px; border-radius: 3px; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    a {{ color: #0a58c2; }}
  </style>
</head>
<body>
  <h1>Rustral benchmark snapshots</h1>
  <p>This dashboard renders <a href="https://github.com/skumyol/neural_engine/tree/main/benchmarks/runs"><code>benchmarks/runs/</code></a>
  per-release JSON snapshots. For continuous trend tracking across every commit, see
  <a href="https://bencher.dev/console/projects/rustral">bencher.dev/rustral</a> (only populated when
  <code>BENCHER_API_TOKEN</code> is configured at the repo level).</p>
  {body_versions}
  <h2>Schema</h2>
  <p>All snapshots conform to <a href="https://github.com/skumyol/neural_engine/blob/main/benchmarks/schema_v2.json">schema v2.0.0</a>.
  See <a href="https://github.com/skumyol/neural_engine/blob/main/BENCHMARKS.md">BENCHMARKS.md</a> for backend coverage and run instructions.</p>
</body>
</html>
"""


def render_version(version: str, manifest: Dict, suites: List[Dict]) -> str:
    """Render a single per-version page."""
    rows: List[str] = []
    for suite in suites:
        suite_name = html.escape(str(suite.get("suite", "?")))
        machine = suite.get("machine", {})
        host = html.escape(str(machine.get("hostname", "?")))
        for sample in suite.get("samples", []):
            rows.append(
                f"<tr>"
                f"<td>{html.escape(str(sample.get('name', '?')))}</td>"
                f"<td>{suite_name}</td>"
                f"<td>{html.escape(str(sample.get('backend', '?')))}</td>"
                f"<td>{html.escape(str(sample.get('device', '?')))}</td>"
                f'<td class="num">{sample.get("mean_ms", 0):.3f}</td>'
                f'<td class="num">{sample.get("p50_ms", 0):.3f}</td>'
                f'<td class="num">{sample.get("min_ms", 0):.3f}</td>'
                f'<td class="num">{sample.get("max_ms", 0):.3f}</td>'
                f"<td>{host}</td>"
                f"</tr>"
            )

    body_rows = "\n".join(rows) or "<tr><td colspan='9'><em>No samples in this snapshot.</em></td></tr>"

    manifest_pretty = html.escape(json.dumps(manifest, indent=2))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Rustral benchmarks — {html.escape(version)}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body {{ font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #1a1a1a; }}
    h1 {{ font-size: 1.6em; }}
    h2 {{ font-size: 1.2em; margin-top: 2em; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 0.5em; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e0e0e0; padding: 4px 8px; text-align: left; }}
    th {{ background: #f7f7f7; position: sticky; top: 0; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    pre {{ background: #f3f3f3; padding: 1em; overflow-x: auto; font-size: 12px; }}
    a {{ color: #0a58c2; }}
  </style>
</head>
<body>
  <p><a href="index.html">← all snapshots</a></p>
  <h1>Snapshot {html.escape(version)}</h1>
  <h2>Samples</h2>
  <table>
    <thead>
      <tr>
        <th>Workload</th>
        <th>Suite</th>
        <th>Backend</th>
        <th>Device</th>
        <th class="num">Mean (ms)</th>
        <th class="num">p50 (ms)</th>
        <th class="num">Min (ms)</th>
        <th class="num">Max (ms)</th>
        <th>Host</th>
      </tr>
    </thead>
    <tbody>
{body_rows}
    </tbody>
  </table>
  <h2>Manifest</h2>
  <pre>{manifest_pretty}</pre>
</body>
</html>
"""


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory (default: docs/site)")
    args = ap.parse_args(argv)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    versions = discover_versions()

    landing = render_landing(versions)
    (out_dir / "index.html").write_text(landing)
    print(f"wrote {out_dir / 'index.html'}")

    for version, vdir in versions:
        manifest_path = vdir / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            print(f"warn: skipping {version}: bad manifest.json: {exc}", file=sys.stderr)
            continue
        suites = load_suite_files(vdir)
        page = render_version(version, manifest, suites)
        (out_dir / f"{version}.html").write_text(page)
        # Copy raw JSONs into data/<version>/ so the page can be inspected without
        # navigating the repo.
        target = data_dir / version
        target.mkdir(parents=True, exist_ok=True)
        for src in vdir.glob("*.json"):
            shutil.copy2(src, target / src.name)
        print(f"wrote {out_dir / f'{version}.html'} (+ {len(suites)} suite files)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
