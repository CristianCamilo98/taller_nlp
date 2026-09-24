"""HTML autocontenido: comparación de runs de retrieval (benchmark v2)."""

from __future__ import annotations

import json
from typing import Any


def render_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=False, default=str)
    data_json = data_json.replace("</", "<\\/")

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Comparador retrieval · benchmark v2</title>
<style>
  :root {{
    --bg: #f7f6f3;
    --panel: #ffffff;
    --ink: #1c1b19;
    --muted: #6b6560;
    --stroke: #e4e0d8;
    --accent: #1f4b6e;
    --ok: #1f6b45;
    --ok-bg: #e6f3eb;
    --fail: #9b2c2c;
    --fail-bg: #f8e8e8;
    --na: #8a847c;
    --na-bg: #efeee9;
    --best-bg: #e8f0e4;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    color: var(--ink);
    background: var(--bg);
    line-height: 1.45;
  }}
  header {{
    padding: 1.25rem 1.5rem 1rem;
    border-bottom: 1px solid var(--stroke);
    background: var(--panel);
  }}
  header h1 {{
    margin: 0 0 0.25rem;
    font-size: 1.35rem;
    font-weight: 650;
    letter-spacing: -0.02em;
  }}
  header p {{ margin: 0; color: var(--muted); font-size: 0.92rem; }}
  main {{
    display: grid;
    grid-template-columns: 300px 1fr;
    min-height: calc(100vh - 88px);
  }}
  aside {{
    padding: 1rem 1rem 2rem;
    border-right: 1px solid var(--stroke);
    background: var(--panel);
  }}
  aside h2 {{
    margin: 0 0 0.6rem;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
  }}
  .run-list {{
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    max-height: 62vh;
    overflow: auto;
  }}
  .run-item {{
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.5rem;
    align-items: start;
    padding: 0.55rem 0.6rem;
    border: 1px solid var(--stroke);
    border-radius: 6px;
    cursor: pointer;
    background: var(--bg);
  }}
  .run-item.selected {{
    border-color: var(--accent);
    background: #eef4f8;
  }}
  .run-item strong {{ display: block; font-size: 0.84rem; }}
  .run-item span {{ display: block; font-size: 0.72rem; color: var(--muted); }}
  .aside-actions {{ display: flex; gap: 0.4rem; margin: 0.75rem 0 1rem; }}
  button, select {{
    font: inherit;
    border-radius: 6px;
    border: 1px solid var(--stroke);
    background: var(--panel);
    color: var(--ink);
    padding: 0.35rem 0.65rem;
    cursor: pointer;
  }}
  button.primary {{
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
  }}
  .content {{ padding: 1rem 1.25rem 2.5rem; overflow: auto; }}
  section {{ margin-bottom: 1.5rem; }}
  section h2 {{ margin: 0 0 0.65rem; font-size: 1.05rem; }}
  .caption {{
    margin: -0.35rem 0 0.75rem;
    color: var(--muted);
    font-size: 0.82rem;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--panel);
    font-size: 0.84rem;
  }}
  th, td {{
    border: 1px solid var(--stroke);
    padding: 0.4rem 0.5rem;
    text-align: left;
    vertical-align: top;
  }}
  th {{ background: #f0eee8; font-weight: 600; white-space: nowrap; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td.best {{ background: var(--best-bg); font-weight: 650; }}
  .heatmap-wrap {{ overflow: auto; }}
  .heatmap td.cell {{
    text-align: center;
    font-weight: 700;
    width: 3rem;
    font-variant-numeric: tabular-nums;
  }}
  .heatmap td.hit {{ background: var(--ok-bg); color: var(--ok); }}
  .heatmap td.miss {{ background: var(--fail-bg); color: var(--fail); }}
  .heatmap td.na {{ background: var(--na-bg); color: var(--na); }}
  .errors {{
    color: var(--fail);
    font-size: 0.82rem;
    margin-top: 0.75rem;
  }}
  @media (max-width: 960px) {{
    main {{ grid-template-columns: 1fr; }}
    aside {{ border-right: none; border-bottom: 1px solid var(--stroke); }}
  }}
</style>
</head>
<body>
<header>
  <h1>Comparador de retrieval · benchmark v2</h1>
  <p id="subtitle">Cargando…</p>
</header>
<main>
  <aside>
    <h2>Runs</h2>
    <div class="aside-actions">
      <button type="button" id="btn-all">Todos</button>
      <button type="button" id="btn-none">Ninguno</button>
    </div>
    <div class="run-list" id="run-list"></div>
    <div class="errors" id="errors"></div>
  </aside>
  <div class="content">
    <section>
      <h2>Resumen (métrica primaria: NON-7A-36 Recall@5)</h2>
      <p class="caption">Mejor valor de cada columna numérica remarcado. ALL-48 R@5 es secundaria de contexto.</p>
      <div id="summary-table"></div>
    </section>
    <section>
      <h2>Por vista</h2>
      <p class="caption">
        Vista
        <select id="view-select"></select>
      </p>
      <div id="view-table"></div>
    </section>
    <section>
      <h2>Heatmap first_relevant_rank</h2>
      <p class="caption">Verde = rank ≤ 5 (hit a retrieval_k del agente). Rojo = &gt;5 o ausente. — = sin dato.</p>
      <div class="heatmap-wrap" id="heatmap"></div>
    </section>
  </div>
</main>
<script id="payload" type="application/json">{data_json}</script>
<script>
const DATA = JSON.parse(document.getElementById("payload").textContent);
const selected = new Set(DATA.runs.map(r => r.uid));
let activeView = "NON-7A-36";

function fmtPct(v) {{
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return (100 * v).toFixed(1) + "%";
}}
function fmtNum(v, digits=3) {{
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Number(v).toFixed(digits);
}}
function selectedRuns() {{
  return DATA.runs.filter(r => selected.has(r.uid));
}}
function bestOf(values) {{
  const nums = values.filter(v => typeof v === "number" && !Number.isNaN(v));
  return nums.length ? Math.max(...nums) : null;
}}

function renderSidebar() {{
  const root = document.getElementById("run-list");
  root.innerHTML = "";
  DATA.runs.forEach(run => {{
    const div = document.createElement("label");
    div.className = "run-item" + (selected.has(run.uid) ? " selected" : "");
    div.innerHTML =
      '<input type="checkbox"' + (selected.has(run.uid) ? " checked" : "") + '/>' +
      "<div><strong>" + escapeHtml(run.label) + "</strong>" +
      "<span>" + escapeHtml(run.model || "—") +
      (run.n_questions ? " · n=" + run.n_questions : "") +
      (run.primary_r5 != null ? " · R@5=" + fmtPct(run.primary_r5) : "") +
      "</span></div>";
    const cb = div.querySelector("input");
    cb.addEventListener("change", () => {{
      if (cb.checked) selected.add(run.uid); else selected.delete(run.uid);
      renderAll();
    }});
    root.appendChild(div);
  }});
  const err = document.getElementById("errors");
  err.innerHTML = (DATA.errors || []).map(e => escapeHtml(e)).join("<br/>");
}}

function renderSummary() {{
  const runs = selectedRuns();
  const host = document.getElementById("summary-table");
  if (!runs.length) {{
    host.innerHTML = '<div class="caption">Selecciona al menos un run.</div>';
    return;
  }}
  const bestR5 = bestOf(runs.map(r => r.primary_r5));
  const bestMrr = bestOf(runs.map(r => r.primary_mrr));
  const bestAll = bestOf(runs.map(r => r.all48_r5));
  let html = "<table><thead><tr>" +
    "<th>Run</th><th>Autor</th><th>Modelo</th><th>Variante</th><th>n</th>" +
    "<th>NON-7A R@5</th><th>NON-7A MRR@10</th><th>ALL-48 R@5</th>" +
    "</tr></thead><tbody>";
  runs.forEach(r => {{
    html += "<tr>" +
      "<td><strong>" + escapeHtml(r.label) + "</strong><br/>" +
      "<span style='color:var(--muted);font-size:0.75rem'>" + escapeHtml(r.source_file) + "</span></td>" +
      "<td>" + escapeHtml(r.author) + "</td>" +
      "<td>" + escapeHtml(r.model || "—") + "</td>" +
      "<td>" + escapeHtml(r.variant || "—") + "</td>" +
      "<td class='num'>" + (r.n_questions ?? "—") + "</td>" +
      "<td class='num" + (r.primary_r5 === bestR5 && bestR5 != null ? " best" : "") + "'>" + fmtPct(r.primary_r5) + "</td>" +
      "<td class='num" + (r.primary_mrr === bestMrr && bestMrr != null ? " best" : "") + "'>" + fmtNum(r.primary_mrr, 4) + "</td>" +
      "<td class='num" + (r.all48_r5 === bestAll && bestAll != null ? " best" : "") + "'>" + fmtPct(r.all48_r5) + "</td>" +
      "</tr>";
  }});
  html += "</tbody></table>";
  host.innerHTML = html;
}}

function renderViewSelect() {{
  const sel = document.getElementById("view-select");
  if (sel.options.length) return;
  (DATA.views || []).forEach(v => {{
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    if (v === activeView) opt.selected = true;
    sel.appendChild(opt);
  }});
  sel.addEventListener("change", () => {{
    activeView = sel.value;
    renderViewTable();
  }});
}}

function renderViewTable() {{
  const runs = selectedRuns();
  const host = document.getElementById("view-table");
  if (!runs.length) {{ host.innerHTML = ""; return; }}
  const keys = ["recall@1","recall@3","recall@5","recall@10","mrr@10"];
  const bests = {{}};
  keys.forEach(k => {{
    bests[k] = bestOf(runs.map(r => (r.metrics[activeView] || {{}})[k]));
  }});
  let html = "<table><thead><tr><th>Run</th>";
  keys.forEach(k => {{ html += "<th>" + k + "</th>"; }});
  html += "<th>n</th></tr></thead><tbody>";
  runs.forEach(r => {{
    const m = r.metrics[activeView] || {{}};
    html += "<tr><td>" + escapeHtml(r.label) + "</td>";
    keys.forEach(k => {{
      const v = m[k];
      const cls = (v === bests[k] && bests[k] != null) ? "num best" : "num";
      html += "<td class='" + cls + "'>" +
        (k.startsWith("recall") ? fmtPct(v) : fmtNum(v, 4)) + "</td>";
    }});
    html += "<td class='num'>" + (m.n_questions ?? "—") + "</td></tr>";
  }});
  html += "</tbody></table>";
  host.innerHTML = html;
}}

function renderHeatmap() {{
  const runs = selectedRuns();
  const host = document.getElementById("heatmap");
  if (!runs.length) {{ host.innerHTML = ""; return; }}
  const qids = DATA.preguntas || [];
  let html = "<table class='heatmap'><thead><tr><th>id</th><th>item</th>";
  runs.forEach(r => {{ html += "<th title='" + escapeAttr(r.uid) + "'>" + escapeHtml(r.label) + "</th>"; }});
  html += "<th>pregunta</th></tr></thead><tbody>";
  qids.forEach(q => {{
    html += "<tr><td>" + escapeHtml(q.id) + "</td><td>" + escapeHtml(q.item || "") + "</td>";
    runs.forEach(r => {{
      const rank = (r.ranks || {{}})[q.id];
      let cls = "na", text = "—";
      if (typeof rank === "number") {{
        text = String(rank);
        cls = rank <= 5 ? "hit" : "miss";
      }}
      html += "<td class='cell " + cls + "'>" + text + "</td>";
    }});
    html += "<td>" + escapeHtml(q.pregunta || "") + "</td></tr>";
  }});
  html += "</tbody></table>";
  host.innerHTML = html;
}}

function escapeHtml(s) {{
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}}
function escapeAttr(s) {{ return escapeHtml(s).replace(/'/g, "&#39;"); }}

function renderAll() {{
  document.getElementById("subtitle").textContent =
    DATA.n_files + " fichero(s) · " + DATA.n_runs + " run(s) · " + DATA.results_dir;
  renderSidebar();
  renderViewSelect();
  renderSummary();
  renderViewTable();
  renderHeatmap();
}}

document.getElementById("btn-all").addEventListener("click", () => {{
  DATA.runs.forEach(r => selected.add(r.uid));
  renderAll();
}});
document.getElementById("btn-none").addEventListener("click", () => {{
  selected.clear();
  renderAll();
}});
renderAll();
</script>
</body>
</html>
"""
