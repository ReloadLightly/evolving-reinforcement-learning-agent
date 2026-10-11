#!/usr/bin/env python3
"""Build an offline descriptive pilot-results explorer from validated artifacts.

Uses only Python's standard library. Run analyze_pilot.py first; this exporter
requires its complete 24-run summary and independently checks evaluation grids.
The output embeds measurements and has no external dependencies or requests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import itertools
import json
import math
from pathlib import Path
import re


TASKS = ("Pendulum-v1", "Reacher-v5")
ARMS = ("td3", "erl_pooled", "erl_balanced", "erl_no_ea_replay")
GRID = tuple(range(0, 50_001, 10_000))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def build_data(root: Path) -> dict:
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text())
    require(summary.get("verified_runs") == 24 and len(summary.get("runs", [])) == 24,
            "Explorer requires the complete analyzed 24-run pilot")
    expected = set(itertools.product(TASKS, ARMS, (0, 1, 2)))
    seen, runs = set(), []
    for record in summary["runs"]:
        key = record["task"], record["arm"], record["seed"]
        require(key in expected and key not in seen, f"Unexpected or duplicate run: {key}")
        seen.add(key)
        run_id = record["run_id"]
        require(bool(re.fullmatch(r"[A-Za-z0-9_-]+", run_id)), "Unsafe run directory name")
        path = root / run_id / "evaluation.csv"
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        values = {}
        for row in rows:
            require((row["task"], row["arm"], int(row["seed"])) == key and int(row["delay"]) == 1,
                    f"Evaluation identity mismatch: {path}")
            position = int(row["step"]), int(row["evaluation_episode"])
            require(position not in values, f"Duplicate evaluation row: {path}")
            value = float(row["episode_return"])
            require(math.isfinite(value), f"Nonfinite return: {path}")
            require(int(row["episode_length"]) == (200 if key[0] == "Pendulum-v1" else 50),
                    f"Unexpected evaluation episode length: {path}")
            values[position] = value
        require(set(values) == set(itertools.product(GRID, range(5))),
                f"Incomplete evaluation grid: {path}")
        points = [dict(step=step, episodes=[values[step, ep] for ep in range(5)],
                       mean=math.fsum(values[step, ep] for ep in range(5)) / 5)
                  for step in GRID]
        auc = math.fsum((left["mean"] + right["mean"]) / 2 * (right["step"] - left["step"])
                        for left, right in zip(points, points[1:])) / 50_000
        for field, observed in (("initial_return", points[0]["mean"]),
                                ("final_return", points[-1]["mean"]), ("raw_auc", auc)):
            require(math.isfinite(record[field]) and math.isclose(record[field], observed, rel_tol=1e-10, abs_tol=1e-10),
                    f"Summary/evaluation mismatch for {field}: {run_id}")
        require((root / run_id / "journal.md").is_file(), f"Missing run journal: {run_id}")
        runs.append(dict(**record, points=points, csv_path=f"{run_id}/evaluation.csv",
                         journal_path=f"{run_id}/journal.md", csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    require(seen == expected, "Pilot run grid is incomplete")
    return dict(scope=summary["scope"], runs=runs,
                summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                summary_path="summary.json", task_order=TASKS, arm_order=ARMS)


PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ERL CPU pilot · offline results explorer</title>
<style>
:root{--ink:#172631;--muted:#45596a;--bg:#f3f6f8;--line:#cbd6df;--blue:#0072b2;--card:#fff;--focus:#7942b8}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1400px;margin:auto;padding:clamp(18px,4vw,48px)}h1,h2,h3{line-height:1.2;letter-spacing:-.025em}h1{font-size:clamp(2rem,4vw,3.2rem);margin:.3em 0}h2{font-size:1.4rem;margin:0 0 16px}h3{font-size:1rem;margin:0 0 8px}p{margin:.7em 0}a{color:var(--blue);text-underline-offset:3px}.eyebrow{font-size:.77rem;letter-spacing:.12em;text-transform:uppercase;font-weight:750;color:var(--muted)}.lede{max-width:940px;font-size:1.08rem;color:var(--muted)}.notice{border-left:4px solid var(--blue);background:#e8f2f8;padding:14px 20px;margin:24px 0}.toolbar,.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px}.toolbar{display:flex;gap:20px 30px;flex-wrap:wrap;align-items:flex-start;margin:20px 0}.toolbar>div{min-width:160px}label{font-weight:650}select{font:inherit;background:var(--card);color:var(--ink);border:1px solid #879baa;border-radius:6px;padding:8px;max-width:100%;display:block;margin:8px 0 0;min-height:42px}fieldset{border:0;padding:0;margin:0}legend{font-weight:650;margin-bottom:8px}.choices{display:flex;gap:8px 16px;flex-wrap:wrap}.choices label{font-size:.92rem;font-weight:450;display:flex;align-items:center;gap:6px;min-height:32px;cursor:pointer}input[type=checkbox]{width:18px;height:18px;accent-color:var(--blue)}.swatch{display:inline-block;width:20px;border-top:4px solid var(--blue);vertical-align:middle}.layout{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:20px}.plot{width:100%;height:auto;display:block}.plot text{font-family:inherit;fill:var(--muted)}.plot .dot{cursor:pointer;stroke:#fff;stroke-width:1.5}.plot .dot:focus{outline:none;stroke:var(--focus);stroke-width:5}.plot .dot:hover{stroke:var(--ink);stroke-width:3}.note{font-size:.88rem;color:var(--muted)}.stat{font-variant-numeric:tabular-nums;font-size:1.5rem;font-weight:750;margin:0;overflow-wrap:anywhere}.metrics{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:18px 0}.metrics dt{font-size:.8rem;color:var(--muted)}.metrics dd{margin:0;font-weight:700;font-variant-numeric:tabular-nums}.divider{border-top:1px solid var(--line);padding-top:16px;margin-top:16px}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;font-size:.88rem}th,td{text-align:right;border-bottom:1px solid var(--line);padding:8px;white-space:nowrap;font-variant-numeric:tabular-nums}th:first-child,td:first-child{text-align:left}caption{text-align:left;padding:0 0 10px;font-weight:650}button{font:inherit;background:white;color:var(--ink);border:1px solid #879baa;border-radius:6px;padding:5px 9px;cursor:pointer}button:hover{background:#e8f2f8}:focus-visible{outline:3px solid var(--focus);outline-offset:3px}.source{font-family:ui-monospace,monospace;font-size:.77rem;overflow-wrap:anywhere}.empty{padding:50px;text-align:center;color:var(--muted)}.below{margin-top:20px}.checkpoint-note{font-size:.82rem;overflow-wrap:anywhere;color:var(--muted)}details summary{cursor:pointer;font-weight:650}footer{padding:20px 0;color:var(--muted);font-size:.9rem}@media(max-width:1050px){.layout{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(3,1fr)}}@media(max-width:600px){.toolbar,.panel{padding:14px}.metrics{grid-template-columns:1fr 1fr}.toolbar{gap:16px}.plot text{font-size:14px}.choices{gap:5px 10px}}@media print{.toolbar{display:none}.layout{grid-template-columns:1fr}main{padding:0}.panel{break-inside:avoid}.plot{max-height:420px}}
</style>
</head>
<body><main>
<header><div class="eyebrow">Evolutionary reinforcement learning · measured CPU pilot</div>
<h1>Inspect how each agent learned.</h1>
<p class="lede">Explore 24 completed development runs across two tasks, four methods, and three training seeds. Each plotted checkpoint is the mean native return over five evaluation episodes from the learner policy.</p>
<div class="notice"><strong>Descriptive pilot evidence, n = 3 trained seeds per task and method.</strong> Rewards were immediate (delay = 1). These runs check implementation, baseline learning, and CPU cost; they do not test the delayed-reward hypothesis. No confidence intervals or significance claims are shown.</div></header>
<div class="toolbar" aria-label="Results filters">
<div><label for="task">Environment</label><select id="task"><option>Pendulum-v1</option><option>Reacher-v5</option></select></div>
<fieldset><legend>Methods</legend><div class="choices" id="methods"></div></fieldset>
<fieldset><legend>Training seeds</legend><div class="choices" id="seeds"></div></fieldset>
<div><label><input id="show-mean" type="checkbox" checked> Show selected-seed mean</label><p class="note">Thin traces = individual seeds.<br>Thick traces = selected-seed mean.</p></div>
</div>
<div class="layout">
<section class="panel" aria-labelledby="curves-title"><h2 id="curves-title">Native-return learning curves</h2>
<p class="note" id="selection-description" aria-live="polite"></p>
<div id="chart"></div>
<p class="note">Higher (less negative) is better. Shading marks 0–10,000 random-action warmup transitions. Points are measured checkpoints; connecting segments guide the eye and do not claim measured performance between checkpoints. No smoothing is applied. Task scales differ.</p>
<p class="note">Hover, click, or keyboard-focus an individual checkpoint to inspect its recorded mean. The run and checkpoint selectors provide the same information without using the chart.</p>
</section>
<aside class="panel" aria-labelledby="run-title"><h2 id="run-title">Follow one trained agent</h2>
<label for="selected-run">Visible run</label><select id="selected-run"></select>
<label for="checkpoint" style="display:block;margin-top:16px">Training checkpoint</label><select id="checkpoint"></select>
<div id="run-details" aria-live="polite" aria-atomic="true"></div>
</aside>
</div>
<section class="panel below" aria-labelledby="summary-title"><h2 id="summary-title">Compare the visible seed records</h2>
<div class="table-wrap"><table><caption id="summary-caption">Descriptive method summaries</caption><thead><tr><th scope="col">Method</th><th scope="col">Selected seeds</th><th scope="col">Mean final return</th><th scope="col">Mean raw AUC</th></tr></thead><tbody id="summary-table"></tbody></table></div>
<p class="note"><strong>Final return</strong> measures the learner at exactly 50,000 training transitions. <strong>Raw AUC</strong> summarizes its entire measured learning curve, including initialization and warmup: trapezoidal integral divided by 50,000 steps. Its units are native episode return. A method can finish better yet have lower AUC if it improved later. Neither metric is normalized here, and raw values should not be averaged across these two tasks.</p>
<p class="note">Changing a filter changes this display, not the experiment or analysis plan. Selecting a favorable seed does not establish a method advantage. The five evaluation episodes are repeated evaluations of one trained agent; they are not five independent training replications.</p>
</section>
<section class="panel below"><h2>Recorded data and interpretation</h2>
<p>This page embeds the completed evaluation CSVs and their validated <a href="summary.json">summary.json</a>. It makes no network requests and requires no server. Open the HTML file locally in a browser. A run's journal explains the collection, learning, and checks that occurred.</p>
<p>Read the <a href="report.md">pilot report</a>, <a href="run-guide.md">individual-run explanations</a>, <a href="../../docs/learning-guide.md">learning guide</a>, or <a href="../../docs/learning-lab.html">interactive worked calculations</a>.</p>
<details><summary>Source files and integrity hashes</summary><p class="source">Input directory: __ROOT__<br>Summary SHA-256: <span id="summary-hash"></span></p><div id="source-list"></div></details>
</section>
<footer>Local research artifact. All curves come from logged pilot observations; no synthetic measurements are included. Use the complete planned seed records when interpreting a comparison.</footer>
<noscript><div class="notice">JavaScript is needed for interactive curves. The <a href="report.md">static pilot report</a> and <a href="summary.json">summary data</a> remain available.</div></noscript>
</main>
<script id="pilot-data" type="application/json">__DATA__</script>
<script>
"use strict";
const DATA = JSON.parse(document.getElementById("pilot-data").textContent);
const METHODS = {
  td3: {label:"TD3", color:"#0072B2", dash:"", explain:"The learner collected all training transitions and sampled its own replay."},
  erl_pooled: {label:"ERL · pooled", color:"#D55E00", dash:"9 4", explain:"Four population slots and one learner slot collected data; sampling was uniform over their common replay buffer."},
  erl_balanced: {label:"ERL · balanced", color:"#00805B", dash:"3 3", explain:"The same population/learner collection schedule fed minibatches containing exactly 64 samples from each source."},
  erl_no_ea_replay: {label:"ERL · learner-only replay", color:"#9A4891", dash:"10 3 2 3", explain:"Population rollouts still consumed 80% of training interactions and occupied replay, but updates sampled only learner-origin transitions."}
};
const $ = id => document.getElementById(id);
const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
const format = value => Number(value).toLocaleString("en-US", {maximumFractionDigits:2});
const escapeHtml = text => String(text).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
function trapezoid(points) {
  if (points.length < 2) return null;
  let sum = 0;
  for(let i=1;i<points.length;i++) sum += (points[i-1].mean+points[i].mean)/2*(points[i].step-points[i-1].step);
  return sum / (points[points.length-1].step-points[0].step);
}
function visibleRuns() {
  const methods = [...document.querySelectorAll("[data-arm]:checked")].map(input=>input.dataset.arm);
  const seeds = [...document.querySelectorAll("[data-seed]:checked")].map(input=>Number(input.dataset.seed));
  return DATA.runs.filter(run=>run.task===$("task").value && methods.includes(run.arm) && seeds.includes(run.seed));
}
function svgElement(tag, attrs, text) {
  const element=document.createElementNS("http://www.w3.org/2000/svg",tag);
  for(const [key,value] of Object.entries(attrs || {})) element.setAttribute(key,String(value));
  if(text !== undefined) element.textContent=text;
  return element;
}
function inspect(runId, step) {
  $("selected-run").value=runId; $("checkpoint").value=String(step); updateRun();
}
function updateRun() {
  const run=DATA.runs.find(candidate=>candidate.run_id===$("selected-run").value);
  if(!run) { $("run-details").innerHTML='<p class="note">Select at least one method and training seed.</p>'; return; }
  const point=run.points.find(candidate=>candidate.step===Number($("checkpoint").value)) || run.points[0];
  const method=METHODS[run.arm];
  $("run-details").innerHTML=`<div class="divider"><h3>Mean native return at ${format(point.step)} steps</h3><p class="stat">${escapeHtml(String(point.mean))}</p><p class="checkpoint-note">Five evaluation episodes; the displayed number preserves the stored numerical precision.</p></div>
    <dl class="metrics"><div><dt>Initial return</dt><dd>${format(run.initial_return)}</dd></div><div><dt>Final return</dt><dd>${format(run.final_return)}</dd></div><div><dt>Raw AUC</dt><dd>${format(run.raw_auc)}</dd></div><div><dt>Run wall time</dt><dd>${format(run.elapsed_s)} s</dd></div></dl>
    <p class="note">${method.explain}</p><p><a href="${escapeHtml(run.journal_path)}">Read this run's journal</a> · <a href="${escapeHtml(run.csv_path)}">Evaluation CSV</a></p>
    <details><summary>Five recorded episode returns</summary><ol>${point.episodes.map(value=>`<li class="checkpoint-note">${escapeHtml(String(value))}</li>`).join("")}</ol></details><p class="source">${escapeHtml(run.run_id)}</p>`;
}
function renderChart(runs) {
  $("chart").replaceChildren();
  if(!runs.length) { const p=document.createElement("p");p.className="empty";p.textContent="Choose at least one method and training seed to display recorded checkpoints.";$("chart").appendChild(p);return; }
  const width=900,height=490,left=100,right=24,top=32,bottom=72;
  const values=runs.flatMap(run=>run.points.map(point=>point.mean));
  const low=Math.min(...values),high=Math.max(...values),margin=Math.max((high-low)*0.1,1);
  const minY=low-margin,maxY=high+margin;
  const x=step=>left+step/50000*(width-left-right);
  const y=value=>height-bottom-(value-minY)/(maxY-minY)*(height-top-bottom);
  const svg=svgElement("svg",{viewBox:`0 0 ${width} ${height}`,class:"plot",role:"group","aria-labelledby":"plot-title plot-description"});
  svg.append(svgElement("title",{id:"plot-title"},`${$("task").value} recorded learner returns`),svgElement("desc",{id:"plot-description"},"Horizontal axis: total training transitions. Vertical axis: mean native episode return, higher is better. Thin lines connect individual seed checkpoints; thick lines show the mean across selected seeds. Individual points are keyboard accessible."));
  svg.appendChild(svgElement("rect",{x:x(0),y:top,width:x(10000)-x(0),height:height-top-bottom,fill:"#e8edf1"}));
  svg.appendChild(svgElement("text",{x:x(5000),y:top+17,"text-anchor":"middle","font-size":13},"Warmup"));
  for(let i=0;i<=5;i++) {
    const value=minY+(maxY-minY)*i/5;
    svg.appendChild(svgElement("line",{x1:left,x2:width-right,y1:y(value),y2:y(value),stroke:"#dce3e8","stroke-width":1}));
    svg.appendChild(svgElement("text",{x:left-10,y:y(value)+5,"text-anchor":"end","font-size":14},format(value)));
    const step=i*10000;
    svg.appendChild(svgElement("text",{x:x(step),y:height-bottom+27,"text-anchor":"middle","font-size":14},format(step)));
  }
  svg.appendChild(svgElement("text",{x:(left+width-right)/2,y:height-17,"text-anchor":"middle","font-size":16},"Total training transitions (all collector sources)"));
  svg.appendChild(svgElement("text",{transform:`translate(22 ${(top+height-bottom)/2}) rotate(-90)`,"text-anchor":"middle","font-size":16},"Mean native episode return ↑"));
  for(const run of runs) {
    const method=METHODS[run.arm];
    svg.appendChild(svgElement("polyline",{points:run.points.map(point=>`${x(point.step)},${y(point.mean)}`).join(" "),fill:"none",stroke:method.color,"stroke-width":1.5,"stroke-opacity":0.4,"stroke-dasharray":method.dash}));
  }
  if($("show-mean").checked) for(const arm of DATA.arm_order) {
    const group=runs.filter(run=>run.arm===arm);if(!group.length)continue;
    const points=group[0].points.map((point,i)=>({step:point.step,mean:mean(group.map(run=>run.points[i].mean))}));
    svg.appendChild(svgElement("polyline",{points:points.map(point=>`${x(point.step)},${y(point.mean)}`).join(" "),fill:"none",stroke:METHODS[arm].color,"stroke-width":3.5,"stroke-dasharray":METHODS[arm].dash}));
  }
  for(const run of runs) for(const point of run.points) {
    const label=`${METHODS[run.arm].label}, seed ${run.seed}, ${point.step} training transitions, mean native return ${point.mean}`;
    const dot=svgElement("circle",{cx:x(point.step),cy:y(point.mean),r:4.5,fill:METHODS[run.arm].color,class:"dot",tabindex:0,role:"button","aria-label":label});
    dot.appendChild(svgElement("title",{},label));
    for(const event of ["mouseenter","focus","click"])dot.addEventListener(event,()=>inspect(run.run_id,point.step));
    dot.addEventListener("keydown",event=>{if(event.key==="Enter"||event.key===" "){event.preventDefault();inspect(run.run_id,point.step);}});
    svg.appendChild(dot);
  }
  $("chart").appendChild(svg);
}
function render() {
  const runs=visibleRuns(),selected=$("selected-run").value;
  const seedIds=[...new Set(runs.map(run=>run.seed))].sort();
  $("selection-description").textContent=`${$("task").value} · ${runs.length} visible runs · selected training seeds: ${seedIds.length ? seedIds.join(", ") : "none"}. Thick curves average these selected seeds only (n = ${seedIds.length} per visible method).`;
  $("selected-run").replaceChildren();
  for(const run of runs) {const option=document.createElement("option");option.value=run.run_id;option.textContent=`${METHODS[run.arm].label} · seed ${run.seed}`;$("selected-run").appendChild(option);}
  if(runs.some(run=>run.run_id===selected))$("selected-run").value=selected;
  $("selected-run").disabled=!runs.length;$("checkpoint").disabled=!runs.length;
  renderChart(runs);updateRun();
  $("summary-table").innerHTML=DATA.arm_order.map(arm=>{const group=runs.filter(run=>run.arm===arm);if(!group.length)return "";return `<tr><th scope="row">${METHODS[arm].label}</th><td>${group.length}</td><td>${format(mean(group.map(run=>run.final_return)))}</td><td>${format(mean(group.map(run=>run.raw_auc)))}</td></tr>`;}).join("");
  $("summary-caption").textContent=`${$("task").value}: means across the selected training seeds; no confidence intervals`;
}
for(const arm of DATA.arm_order) {
  const method=METHODS[arm],label=document.createElement("label"),input=document.createElement("input"),swatch=document.createElement("span");
  input.type="checkbox";input.checked=true;input.dataset.arm=arm;input.addEventListener("change",render);
  swatch.className="swatch";swatch.style.borderColor=method.color;if(method.dash)swatch.style.borderTopStyle="dashed";swatch.setAttribute("aria-hidden","true");
  label.append(input,swatch,document.createTextNode(method.label));$("methods").appendChild(label);
}
for(const seed of [0,1,2]) {const label=document.createElement("label"),input=document.createElement("input");input.type="checkbox";input.checked=true;input.dataset.seed=String(seed);input.addEventListener("change",render);label.append(input,document.createTextNode(String(seed)));$("seeds").appendChild(label);}
for(const step of [0,10000,20000,30000,40000,50000]) {const option=document.createElement("option");option.value=String(step);option.textContent=format(step)+" transitions";$("checkpoint").appendChild(option);}
$("checkpoint").value="50000";
$("task").addEventListener("change",render);$("show-mean").addEventListener("change",render);$("selected-run").addEventListener("change",updateRun);$("checkpoint").addEventListener("change",updateRun);
$("summary-hash").textContent=DATA.summary_sha256;
for(const run of DATA.runs) {const p=document.createElement("p");p.className="source";const a=document.createElement("a");a.href=run.csv_path;a.textContent=run.csv_path;p.append(a,document.createElement("br"),document.createTextNode("SHA-256: "+run.csv_sha256));$("source-list").appendChild(p);}
render();
</script>
</body></html>
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/pilot"))
    args = parser.parse_args()
    data = build_data(args.root)
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    encoded = encoded.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    document = PAGE.replace("__ROOT__", html.escape(str(args.root))).replace("__DATA__", encoded)
    path = args.root / "explorer.html"
    path.write_text(document, encoding="utf-8")
    print(f"Wrote {path}: {len(data['runs'])} verified runs, {path.stat().st_size:,} bytes; no external dependencies")


if __name__ == "__main__":
    main()
