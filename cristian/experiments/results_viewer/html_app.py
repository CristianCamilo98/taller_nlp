"""HTML autocontenido del dashboard de comparación de runs."""

from __future__ import annotations

import json
from typing import Any


def render_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=False, default=str)
    # Evitar romper el script si aparece </script> en respuestas.
    data_json = data_json.replace("</", "<\\/")

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Comparador de resultados · experimentos</title>
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
    --warn: #8a5a00;
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
  header p {{
    margin: 0;
    color: var(--muted);
    font-size: 0.92rem;
  }}
  main {{
    display: grid;
    grid-template-columns: 280px 1fr;
    gap: 0;
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
    max-height: 55vh;
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
  .run-item:hover {{ border-color: #cfc8bc; }}
  .run-item.selected {{
    border-color: var(--accent);
    background: #eef4f8;
  }}
  .run-item strong {{
    display: block;
    font-size: 0.86rem;
  }}
  .run-item span {{
    display: block;
    font-size: 0.75rem;
    color: var(--muted);
  }}
  .aside-actions {{
    display: flex;
    gap: 0.4rem;
    margin: 0.75rem 0 1rem;
  }}
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
  button:hover {{ filter: brightness(0.97); }}
  .content {{
    padding: 1rem 1.25rem 2.5rem;
    overflow: auto;
  }}
  section {{
    margin-bottom: 1.5rem;
  }}
  section h2 {{
    margin: 0 0 0.65rem;
    font-size: 1.05rem;
  }}
  .caption {{
    margin: -0.35rem 0 0.75rem;
    color: var(--muted);
    font-size: 0.82rem;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--panel);
    font-size: 0.86rem;
  }}
  th, td {{
    border: 1px solid var(--stroke);
    padding: 0.45rem 0.55rem;
    text-align: left;
    vertical-align: top;
  }}
  th {{
    background: #f0eee8;
    font-weight: 600;
    white-space: nowrap;
  }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .pill {{
    display: inline-block;
    padding: 0.1rem 0.45rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
  }}
  .pill.ok {{ background: var(--ok-bg); color: var(--ok); }}
  .pill.fail {{ background: var(--fail-bg); color: var(--fail); }}
  .pill.na {{ background: var(--na-bg); color: var(--na); }}
  .heatmap-wrap {{ overflow: auto; }}
  .heatmap td.cell {{
    text-align: center;
    font-weight: 700;
    width: 3.2rem;
    cursor: pointer;
  }}
  .heatmap td.cell.ok {{ background: var(--ok-bg); color: var(--ok); }}
  .heatmap td.cell.fail {{ background: var(--fail-bg); color: var(--fail); }}
  .heatmap td.cell.na {{ background: var(--na-bg); color: var(--na); }}
  .heatmap td.qid {{
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.78rem;
    white-space: nowrap;
    cursor: pointer;
  }}
  .heatmap tr.active td {{ outline: 2px solid var(--accent); outline-offset: -2px; }}
  .toolbar {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    align-items: center;
    margin-bottom: 0.75rem;
  }}
  .toolbar label {{
    font-size: 0.85rem;
    color: var(--muted);
  }}
  .drill {{
    display: none;
    border: 1px solid var(--stroke);
    background: var(--panel);
    border-radius: 8px;
    padding: 1rem;
  }}
  .drill.visible {{ display: block; }}
  .drill h3 {{
    margin: 0 0 0.35rem;
    font-size: 1rem;
  }}
  .drill .meta {{
    color: var(--muted);
    font-size: 0.82rem;
    margin-bottom: 0.85rem;
  }}
  .compare-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 0.75rem;
  }}
  .card {{
    border: 1px solid var(--stroke);
    border-radius: 8px;
    padding: 0.75rem;
    background: var(--bg);
  }}
  .card h4 {{
    margin: 0 0 0.5rem;
    font-size: 0.88rem;
  }}
  .card pre {{
    white-space: pre-wrap;
    word-break: break-word;
    margin: 0.25rem 0 0.7rem;
    font-size: 0.8rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    background: var(--panel);
    border: 1px solid var(--stroke);
    border-radius: 6px;
    padding: 0.55rem;
    max-height: 220px;
    overflow: auto;
  }}
  .kv {{
    display: grid;
    grid-template-columns: 110px 1fr;
    gap: 0.2rem 0.5rem;
    font-size: 0.8rem;
    margin-bottom: 0.55rem;
  }}
  .kv dt {{ color: var(--muted); }}
  .kv dd {{ margin: 0; }}
  .empty {{
    padding: 1.5rem;
    color: var(--muted);
    background: var(--panel);
    border: 1px dashed var(--stroke);
    border-radius: 8px;
  }}
  @media (max-width: 900px) {{
    main {{ grid-template-columns: 1fr; }}
    aside {{ border-right: 0; border-bottom: 1px solid var(--stroke); }}
  }}
</style>
</head>
<body>
<header>
  <h1>Comparador de resultados</h1>
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
  </aside>
  <div class="content">
    <section>
      <h2>Resumen por run</h2>
      <p class="caption">% acierto solo donde la métrica aplica · fuente: JSONL en results/</p>
      <div id="summary-table"></div>
    </section>
    <section>
      <h2>Desglose por familia</h2>
      <div id="familia-table"></div>
    </section>
    <section>
      <h2>Heatmap pregunta × run</h2>
      <div class="toolbar">
        <label for="metric-select">Métrica</label>
        <select id="metric-select">
          <option value="cifra">cifra</option>
          <option value="cita">cita</option>
          <option value="trayectoria">trayectoria</option>
        </select>
      </div>
      <p class="caption">✓ acierto · ✗ fallo · — no aplica · clic en fila para drill-down</p>
      <div class="heatmap-wrap" id="heatmap"></div>
    </section>
    <section>
      <h2>Drill-down</h2>
      <div class="drill" id="drill">
        <p class="caption">Elige una pregunta en el heatmap.</p>
      </div>
    </section>
  </div>
</main>
<script id="payload" type="application/json">{data_json}</script>
<script>
const DATA = JSON.parse(document.getElementById("payload").textContent);
const selected = new Set(DATA.runs.map(r => r.uid));
let activeQid = null;
let metric = "cifra";

function fmtPct(v, n) {{
  if (v == null) return "—";
  return v.toFixed(1) + "% (" + n + ")";
}}
function fmtNum(v, digits=2) {{
  if (v == null || Number.isNaN(v)) return "—";
  return Number(v).toFixed(digits);
}}
function fmtCost(v) {{
  if (v == null) return "—";
  return "$" + Number(v).toFixed(4);
}}
function cellMark(state) {{
  if (state === "ok") return "✓";
  if (state === "fail") return "✗";
  return "—";
}}
function aciertoPill(v) {{
  if (v === true) return '<span class="pill ok">✓</span>';
  if (v === false) return '<span class="pill fail">✗</span>';
  return '<span class="pill na">—</span>';
}}

function selectedRuns() {{
  return DATA.runs.filter(r => selected.has(r.uid));
}}

function renderSidebar() {{
  const root = document.getElementById("run-list");
  root.innerHTML = "";
  if (!DATA.runs.length) {{
    root.innerHTML = '<div class="empty">No hay *.jsonl en results/</div>';
    return;
  }}
  DATA.runs.forEach(run => {{
    const div = document.createElement("label");
    div.className = "run-item" + (selected.has(run.uid) ? " selected" : "");
    div.innerHTML =
      '<input type="checkbox" ' + (selected.has(run.uid) ? "checked" : "") + '/>' +
      '<div><strong>' + escapeHtml(run.label) + '</strong>' +
      '<span>' + escapeHtml((run.model_solicitado || "").split(":").pop() || "—") +
      ' · n=' + run.n +
      (run.n_errores ? ' · err=' + run.n_errores : '') +
      '</span></div>';
    const cb = div.querySelector("input");
    cb.addEventListener("change", () => {{
      if (cb.checked) selected.add(run.uid); else selected.delete(run.uid);
      renderAll();
    }});
    root.appendChild(div);
  }});
}}

function renderSummary() {{
  const runs = selectedRuns();
  const host = document.getElementById("summary-table");
  if (!runs.length) {{
    host.innerHTML = '<div class="empty">Selecciona al menos un run.</div>';
    return;
  }}
  let html = "<table><thead><tr>" +
    "<th>Run</th><th>n</th><th>cifra</th><th>cita</th><th>trayectoria</th>" +
    "<th>latencia media (s)</th><th>coste total</th><th>coste medio</th>" +
    "<th>tools medios</th><th>errores</th><th>prompt</th><th>commit</th>" +
    "</tr></thead><tbody>";
  runs.forEach(r => {{
    html += "<tr>" +
      "<td><strong>" + escapeHtml(r.label) + "</strong><br/><span style='color:var(--muted);font-size:0.75rem'>" +
      escapeHtml(r.timestamp || "") + "</span></td>" +
      "<td class='num'>" + r.n + "</td>" +
      "<td class='num'>" + fmtPct(r.pct_cifra, r.n_cifra) + "</td>" +
      "<td class='num'>" + fmtPct(r.pct_cita, r.n_cita) + "</td>" +
      "<td class='num'>" + fmtPct(r.pct_trayectoria, r.n_trayectoria) + "</td>" +
      "<td class='num'>" + fmtNum(r.latencia_media_s, 2) + "</td>" +
      "<td class='num'>" + fmtCost(r.coste_total) + "</td>" +
      "<td class='num'>" + fmtCost(r.coste_medio) + "</td>" +
      "<td class='num'>" + fmtNum(r.tool_calls_medios, 2) + "</td>" +
      "<td class='num'>" + r.n_errores + "</td>" +
      "<td>" + escapeHtml(r.prompt_version || "—") + "</td>" +
      "<td><code>" + escapeHtml((r.commit_sha || "—").slice(0, 8)) + "</code></td>" +
      "</tr>";
  }});
  html += "</tbody></table>";
  host.innerHTML = html;
}}

function renderFamilia() {{
  const runs = selectedRuns();
  const host = document.getElementById("familia-table");
  if (!runs.length) {{
    host.innerHTML = "";
    return;
  }}
  const familias = new Set();
  runs.forEach(r => Object.keys(r.por_familia || {{}}).forEach(f => familias.add(f)));
  const fams = Array.from(familias).sort();
  let html = "<table><thead><tr><th>Familia</th><th>Run</th><th>n</th>" +
    "<th>cifra</th><th>cita</th><th>trayectoria</th><th>latencia media (s)</th><th>errores</th>" +
    "</tr></thead><tbody>";
  fams.forEach(fam => {{
    runs.forEach(r => {{
      const s = (r.por_familia || {{}})[fam];
      if (!s) return;
      html += "<tr>" +
        "<td>" + escapeHtml(fam) + "</td>" +
        "<td>" + escapeHtml(r.label) + "</td>" +
        "<td class='num'>" + s.n + "</td>" +
        "<td class='num'>" + fmtPct(s.pct_cifra, s.n_cifra) + "</td>" +
        "<td class='num'>" + fmtPct(s.pct_cita, s.n_cita) + "</td>" +
        "<td class='num'>" + fmtPct(s.pct_trayectoria, s.n_trayectoria) + "</td>" +
        "<td class='num'>" + fmtNum(s.latencia_media_s, 2) + "</td>" +
        "<td class='num'>" + s.n_errores + "</td>" +
        "</tr>";
    }});
  }});
  html += "</tbody></table>";
  host.innerHTML = html;
}}

function renderHeatmap() {{
  const runs = selectedRuns();
  const host = document.getElementById("heatmap");
  if (!runs.length) {{
    host.innerHTML = "";
    return;
  }}
  const hm = DATA.heatmaps[metric];
  let html = "<table class='heatmap'><thead><tr><th>id</th><th>familia</th>";
  runs.forEach(r => {{ html += "<th title='" + escapeAttr(r.uid) + "'>" + escapeHtml(r.label) + "</th>"; }});
  html += "<th>pregunta</th></tr></thead><tbody>";
  hm.preguntas.forEach(q => {{
    const active = activeQid === q.id ? " active" : "";
    html += "<tr class='" + active + "' data-qid='" + escapeAttr(q.id) + "'>" +
      "<td class='qid'>" + escapeHtml(q.id) + "</td>" +
      "<td>" + escapeHtml(q.familia) + "</td>";
    runs.forEach(r => {{
      const state = (hm.cells[q.id] || {{}})[r.uid] || "na";
      html += "<td class='cell " + state + "'>" + cellMark(state) + "</td>";
    }});
    html += "<td>" + escapeHtml(q.pregunta) + "</td></tr>";
  }});
  html += "</tbody></table>";
  host.innerHTML = html;
  host.querySelectorAll("tr[data-qid]").forEach(tr => {{
    tr.addEventListener("click", () => {{
      activeQid = tr.getAttribute("data-qid");
      renderHeatmap();
      renderDrill();
    }});
  }});
}}

function renderDrill() {{
  const host = document.getElementById("drill");
  if (!activeQid) {{
    host.className = "drill";
    host.innerHTML = '<p class="caption">Elige una pregunta en el heatmap.</p>';
    return;
  }}
  const byRun = DATA.drilldown[activeQid] || {{}};
  const runs = selectedRuns().filter(r => byRun[r.uid]);
  const sample = runs.length ? byRun[runs[0].uid] : Object.values(byRun)[0];
  host.className = "drill visible";
  let html = "<h3>" + escapeHtml(activeQid) + " · " + escapeHtml((sample && sample.familia) || "") + "</h3>" +
    "<div class='meta'>" + escapeHtml((sample && sample.pregunta) || "") + "</div>" +
    "<div class='compare-grid'>";
  if (!runs.length) {{
    html += '<div class="empty">Esta pregunta no aparece en los runs seleccionados.</div>';
  }} else {{
    runs.forEach(r => {{
      const d = byRun[r.uid];
      html += "<div class='card'><h4>" + escapeHtml(r.label) + "</h4>" +
        "<div class='kv'>" +
        "<dt>cifra</dt><dd>" + aciertoPill(d.acierto_cifra) + " " + escapeHtml(String(d.cifra_agente ?? "—")) +
        " / esp " + escapeHtml(String(d.cifra_esperada ?? "—")) + "</dd>" +
        "<dt>cita</dt><dd>" + aciertoPill(d.acierto_cita) + "</dd>" +
        "<dt>trayectoria</dt><dd>" + aciertoPill(d.acierto_trayectoria) + "</dd>" +
        "<dt>tools</dt><dd>" + escapeHtml(JSON.stringify(d.tool_calls_agente || [])) +
        " (n=" + escapeHtml(String(d.tool_call_count ?? "—")) + ")</dd>" +
        "<dt>latencia</dt><dd>" + fmtNum(d.latencia_s, 2) + " s</dd>" +
        "<dt>coste</dt><dd>" + fmtCost(d.coste) + "</dd>" +
        "<dt>error</dt><dd>" + escapeHtml(d.error || "—") + "</dd>" +
        "</div>" +
        "<div style='font-size:0.78rem;color:var(--muted)'>respuesta_agente</div>" +
        "<pre>" + escapeHtml(d.respuesta_agente || "—") + "</pre>" +
        "<div style='font-size:0.78rem;color:var(--muted)'>respuesta_esperada</div>" +
        "<pre>" + escapeHtml(d.respuesta_esperada || "—") + "</pre>";
      if (d.tool_calls_detallado && d.tool_calls_detallado.length) {{
        html += "<div style='font-size:0.78rem;color:var(--muted)'>tool_calls_detallado</div>" +
          "<pre>" + escapeHtml(JSON.stringify(d.tool_calls_detallado, null, 2)) + "</pre>";
      }}
      if (d.errores_metricas && Object.keys(d.errores_metricas).length) {{
        html += "<div style='font-size:0.78rem;color:var(--muted)'>errores / detalle métricas</div>" +
          "<pre>" + escapeHtml(JSON.stringify(d.errores_metricas, null, 2)) + "</pre>";
      }}
      html += "</div>";
    }});
  }}
  html += "</div>";
  host.innerHTML = html;
  host.scrollIntoView({{ behavior: "smooth", block: "nearest" }});
}}

function escapeHtml(s) {{
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}}
function escapeAttr(s) {{
  return escapeHtml(s).replace(/'/g, "&#39;");
}}

function renderAll() {{
  document.getElementById("subtitle").textContent =
    DATA.n_files + " fichero(s) · " + DATA.n_rows + " filas · " +
    DATA.runs.length + " run(s) · " + DATA.results_dir;
  renderSidebar();
  renderSummary();
  renderFamilia();
  renderHeatmap();
  renderDrill();
}}

document.getElementById("btn-all").addEventListener("click", () => {{
  DATA.runs.forEach(r => selected.add(r.uid));
  renderAll();
}});
document.getElementById("btn-none").addEventListener("click", () => {{
  selected.clear();
  renderAll();
}});
document.getElementById("metric-select").addEventListener("change", (e) => {{
  metric = e.target.value;
  renderHeatmap();
}});

renderAll();
</script>
</body>
</html>
"""
