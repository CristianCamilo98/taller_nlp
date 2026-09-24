"""CLI del visor de resultados.

Uso (desde la raíz del repo)::

    python -m cristian.experiments.results_viewer
    python -m cristian.experiments.results_viewer --no-open
    python -m cristian.experiments.results_viewer --out /tmp/dashboard.html
"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

from .data import RESULTS_DIR, payload_dashboard
from .html_app import render_html


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Genera un dashboard HTML comparando *.jsonl de results/.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directorio con ficheros JSONL (default: cristian/experiments/results).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Ruta del HTML de salida (default: results_viewer/dashboard.html).",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="No abrir el navegador automáticamente.",
    )
    args = parser.parse_args(argv)

    out = args.out or (Path(__file__).resolve().parent / "dashboard.html")
    payload = payload_dashboard(args.results_dir)
    html = render_html(payload)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    print(f"Escritos {payload['n_rows']} filas / {payload['n_files']} ficheros → {out}")
    print(f"Runs: {len(payload['runs'])}")
    for run in payload["runs"]:
        print(
            f"  - {run['label']}: n={run['n']} "
            f"cifra={_pct(run['pct_cifra'])} "
            f"cita={_pct(run['pct_cita'])} "
            f"tray={_pct(run['pct_trayectoria'])} "
            f"err={run['n_errores']}"
        )

    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())
    return 0


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}%"


if __name__ == "__main__":
    raise SystemExit(main())
