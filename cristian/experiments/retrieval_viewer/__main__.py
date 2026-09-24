"""CLI del visor de retrieval (benchmark v2 entre compañeros).

Uso (desde la raíz del repo)::

    python -m cristian.experiments.retrieval_viewer
    python -m cristian.experiments.retrieval_viewer --no-open
"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

from .data import COMPARISONS_DIR, payload_retrieval_dashboard
from .html_app import render_html


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Dashboard HTML comparando JSON de retrieval en "
            "results/comparisons/ (Dani, Marco, Cristian)."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=COMPARISONS_DIR,
        help="Directorio con JSON de comparación (default: results/comparisons).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="HTML de salida (default: retrieval_viewer/dashboard.html).",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="No abrir el navegador automáticamente.",
    )
    args = parser.parse_args(argv)

    out = args.out or (Path(__file__).resolve().parent / "dashboard.html")
    payload = payload_retrieval_dashboard(args.results_dir)
    html = render_html(payload)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    print(
        f"Escritos {payload['n_files']} ficheros → {payload['n_runs']} runs → {out}"
    )
    for run in payload["runs"]:
        r5 = run["primary_r5"]
        r5_s = "—" if r5 is None else f"{100 * r5:.1f}%"
        print(f"  - {run['label']}: NON-7A R@5={r5_s}")
    if payload.get("errors"):
        print("Avisos:")
        for err in payload["errors"]:
            print(f"  ! {err}")

    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
