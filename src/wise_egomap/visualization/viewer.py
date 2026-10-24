"""Self-contained HTML replay viewer with no network or external assets."""

# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path

from wise_egomap.visualization.replay import ReplayDocument

VIEWER_TITLE = "WISE-EgoMap replay"


def write_replay_json(replay: ReplayDocument, output_path: str | Path) -> Path:
    """Write stable replay JSON and return the output path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(replay.to_json() + "\n", encoding="utf-8")
    return path


def render_replay_html(replay: ReplayDocument) -> str:
    """Return a self-contained offline viewer page."""
    data = json.dumps(replay.to_dict(), sort_keys=True, separators=(",", ":"))
    data = data.replace("<", "\\u003c").replace(">", "\\u003e")
    title = _escape_html(replay.title)
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{title}</title>\n"
        "<style>\n"
        ":root{--bg:#101418;--panel:#182028;--line:#34414c;--text:#eef3f6;"
        "--muted:#9fb0bd;--cyan:#57d1c9;--amber:#ffc857;--red:#ff7373;"
        "--green:#75d787;--blue:#7fb3ff}\n"
        "*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);"
        "font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}"
        "header{padding:18px 22px;border-bottom:1px solid var(--line);display:flex;"
        "gap:18px;align-items:baseline;flex-wrap:wrap}h1{font-size:20px;margin:0}"
        ".sub{color:var(--muted)}main{max-width:1440px;margin:0 auto;padding:18px 22px 40px}"
        ".tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}"
        "button{background:var(--panel);color:var(--text);border:1px solid var(--line);"
        "border-radius:7px;padding:8px 12px;cursor:pointer;font:inherit}"
        "button:hover{border-color:var(--cyan)}button.active{border-color:var(--cyan);"
        "background:#173230}button:disabled{opacity:.45;cursor:default}"
        ".scenario{display:grid;grid-template-columns:minmax(320px,1.4fr) minmax(300px,1fr);"
        "gap:14px}.panel{background:var(--panel);border:1px solid var(--line);"
        "border-radius:10px;padding:14px}.panel h2{font-size:13px;letter-spacing:.08em;"
        "text-transform:uppercase;color:var(--muted);margin:0 0 10px}"
        ".maps{display:grid;grid-template-columns:1fr 1fr;gap:12px}"
        ".map-title{display:flex;justify-content:space-between;gap:8px;font-weight:600;"
        "margin-bottom:6px}.ego{color:var(--cyan)}.instant{color:var(--amber)}"
        "canvas{width:100%;height:auto;background:#0c1115;border:1px solid var(--line);"
        "border-radius:8px;display:block}"
        ".controls{display:flex;gap:10px;align-items:center;margin:12px 0;flex-wrap:wrap}"
        "input[type=range]{flex:1;min-width:160px;accent-color:var(--cyan)}"
        ".badge{display:inline-block;border-radius:999px;padding:2px 8px;font-size:11px;"
        "font-weight:700;letter-spacing:.04em;text-transform:uppercase}"
        ".stable,.keep{background:#173c29;color:var(--green)}"
        ".defer,.unknown{background:#423617;color:var(--amber)}"
        ".reject,.transient{background:#4a2020;color:var(--red)}"
        ".commit,.persistent_change{background:#1c3153;color:var(--blue)}"
        ".caption{color:var(--muted);min-height:40px}.query{border-left:3px solid var(--cyan);"
        "background:#111a20;padding:10px 12px;border-radius:4px;margin-top:10px}"
        ".decision{border-top:1px solid var(--line);padding:10px 0}"
        ".decision:first-child{border-top:0}.decision-head{display:flex;justify-content:"
        "space-between;gap:8px;align-items:center}.reason{color:var(--muted);font-size:13px;"
        "margin-top:3px}.compare{display:grid;grid-template-columns:1fr 1fr;gap:8px;"
        "margin-top:6px}.compare div{background:#111a20;padding:7px 9px;border-radius:5px}"
        ".evidence{display:grid;gap:7px}.bar{height:7px;background:#0c1115;border-radius:4px;"
        "overflow:hidden;margin-top:3px}.bar span{display:block;height:100%;background:var(--cyan)}"
        ".bar.warn span{background:var(--amber)}.bar.bad span{background:var(--red)}"
        ".timeline{position:relative;height:68px;margin-top:8px}"
        ".rail{position:absolute;left:4%;right:4%;top:31px;height:2px;background:var(--line)}"
        ".tick{position:absolute;top:23px;width:3px;height:18px;background:var(--muted);"
        "transform:translateX(-50%)}.tick.commit{background:var(--blue)}"
        ".tick.defer{background:var(--amber)}.tick.reject{background:var(--red)}"
        ".tick-label{position:absolute;top:43px;font-size:10px;color:var(--muted);"
        "transform:translateX(-50%);white-space:nowrap}.playhead{position:absolute;top:18px;"
        "width:2px;height:30px;background:var(--cyan);transform:translateX(-50%)}"
        ".meta{color:var(--muted);font-size:12px}.stack{display:grid;gap:14px}"
        "@media(max-width:850px){.scenario{grid-template-columns:1fr}"
        ".maps{grid-template-columns:1fr}.compare{grid-template-columns:1fr}}"
        "</style>\n"
        "</head>\n"
        "<body>\n"
        "<header><h1>WISE-EgoMap synthetic replay</h1>"
        '<span class="sub">Simple primitives. No external data. Deterministic CPU replay.</span>'
        "</header>\n"
        '<main><div id="tabs" class="tabs"></div><div id="app"></div></main>\n'
        '<script id="replay-data" type="application/json">'
        f"{data}</script>\n"
        "<script>\n" + _VIEWER_SCRIPT + "</script>\n"
        "</body>\n"
        "</html>\n"
    )


def write_replay_html(replay: ReplayDocument, output_path: str | Path) -> Path:
    """Write a self-contained replay viewer and return the output path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_replay_html(replay), encoding="utf-8")
    return path


def _escape_html(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


_VIEWER_SCRIPT = r"""
const data = JSON.parse(document.getElementById("replay-data").textContent);
const app = document.getElementById("app");
const tabs = document.getElementById("tabs");
let active = 0;
let frameIndex = 0;
let timer = null;
let playing = false;

function label(value) {
  return String(value || "").toLowerCase().replaceAll("_", " ");
}
function badge(value, action) {
  const cls = String(action || value || "").toLowerCase();
  return `<span class="badge ${cls}">${label(value)}</span>`;
}
function esc(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[char]);
}
function findDecision(frame, id) {
  return frame.decisions.find((item) => item.object_id === id);
}
function drawMap(canvas, entries, title, color, current) {
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#0c1115";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#26333d";
  ctx.lineWidth = 1;
  for (let x = 0; x <= width; x += width / 8) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
  }
  for (let y = 0; y <= height; y += height / 6) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
  }
  ctx.fillStyle = "#9fb0bd";
  ctx.font = "12px system-ui";
  ctx.fillText(title, 12, 20);
  entries.forEach((entry) => {
    const x = width * (0.14 + (entry.position_m[0] + 0.5) * 0.72);
    const y = height * (0.82 - entry.position_m[1] * 0.55);
    const decision = findDecision(current, entry.object_id);
    const isFocus = entry.object_id === "mug";
    ctx.save();
    ctx.translate(x, y);
    ctx.fillStyle = color;
    ctx.globalAlpha = isFocus ? 0.95 : 0.55;
    if (entry.kind === "person") {
      ctx.fillRect(-9, -18, 18, 36);
    } else if (entry.kind === "table") {
      ctx.fillRect(-35, -12, 70, 24);
    } else {
      ctx.beginPath();
      ctx.arc(0, 0, isFocus ? 12 : 8, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#eef3f6";
    ctx.font = isFocus ? "bold 12px system-ui" : "11px system-ui";
    ctx.fillText(entry.display_name, 14, -8);
    if (decision) {
      ctx.fillStyle = "#9fb0bd";
      ctx.font = "10px system-ui";
      ctx.fillText(label(decision.action), 14, 7);
    }
    ctx.restore();
  });
  if (!entries.length) {
    ctx.fillStyle = "#9fb0bd";
    ctx.font = "13px system-ui";
    ctx.fillText("No committed map entries.", 12, height / 2);
  }
}
function decisionHtml(frame) {
  if (!frame.decisions.length) {
    return '<div class="reason">No decision at this keyframe.</div>';
  }
  const focus = frame.decisions.filter((item) =>
    frame.observations.some((obs) => obs.object_id === item.object_id) ||
    frame.ego_map.some((entry) => entry.object_id === item.object_id)
  );
  const items = focus.length ? focus : frame.decisions;
  return items.map((item) => `
    <div class="decision">
      <div class="decision-head">
        <strong>${esc(item.display_name)}</strong>
        <span>${badge(item.label, item.action)} &nbsp; ${Math.round(item.confidence * 100)}%</span>
      </div>
      <div class="reason">${esc(item.reason)}</div>
      <div class="compare">
        <div><span class="ego">WISE-EgoMap</span><br>${esc(label(item.action))} · ${esc(label(item.label))}</div>
        <div><span class="instant">Instant</span><br>${esc(label(item.instant_action))} · ${esc(label(item.instant_label))}<br>
          <span class="meta">${esc(item.instant_reason)}</span></div>
      </div>
    </div>`).join("");
}
function evidenceHtml(frame) {
  const items = frame.evidence.filter((item) => item.object_id === frame.query.object_id);
  if (!items.length) return '<div class="reason">No evidence for the query object.</div>';
  return `<div class="evidence">${items.map((item) => {
    const ratio = Math.max(0, Math.min(1, item.value / item.max_value));
    const threshold = Math.max(0, Math.min(1, item.threshold / item.max_value));
    const cls = item.value + 1e-9 < item.threshold ? "warn" : "";
    return `<div><div>${esc(item.label)} <span class="meta">${item.value.toFixed(2)} ${esc(item.unit)}
      (gate ${item.threshold.toFixed(2)})</span></div>
      <div class="bar ${cls}"><span style="width:${ratio * 100}%;background:${ratio >= threshold ? "var(--cyan)" : "var(--amber)"}"></span></div></div>`;
  }).join("")}</div>`;
}
function timelineHtml(scenario, frame) {
  const duration = scenario.duration_s;
  const pct = (time) => `${Math.max(0, Math.min(100, (time / duration) * 100))}%`;
  const ticks = scenario.key_events.map((event) =>
    `<div class="tick ${esc(event.kind)}" style="left:${pct(event.timestamp_s)}" title="${esc(event.title)}"></div>
     <div class="tick-label" style="left:${pct(event.timestamp_s)}">${esc(event.title)}</div>`
  ).join("");
  return `<div class="timeline"><div class="rail"></div>${ticks}
    <div class="playhead" style="left:${pct(frame.timestamp_s)}"></div></div>`;
}
function render() {
  const scenario = data.scenarios[active];
  frameIndex = Math.max(0, Math.min(frameIndex, scenario.frames.length - 1));
  const frame = scenario.frames[frameIndex];
  tabs.innerHTML = data.scenarios.map((item, index) =>
    `<button class="${index === active ? "active" : ""}" data-tab="${index}">${esc(item.title)}</button>`
  ).join("");
  tabs.querySelectorAll("[data-tab]").forEach((button) => {
    button.addEventListener("click", () => {
      stop(); active = Number(button.dataset.tab); frameIndex = 0; render();
    });
  });
  app.innerHTML = `
    <div class="scenario">
      <div class="stack">
        <section class="panel">
          <h2>${esc(scenario.title)}</h2>
          <p class="sub">${esc(scenario.summary)}</p>
          <div class="maps">
            <div><div class="map-title"><span class="ego">WISE-EgoMap map</span><span class="meta">v${frame.ego_map.length ? Math.max(...frame.ego_map.map(x => x.map_version)) : 0}</span></div>
              <canvas id="ego-canvas" width="420" height="260"></canvas></div>
            <div><div class="map-title"><span class="instant">Instant map</span><span class="meta">v${frame.instant_map.length ? Math.max(...frame.instant_map.map(x => x.map_version)) : 0}</span></div>
              <canvas id="instant-canvas" width="420" height="260"></canvas></div>
          </div>
          <div class="controls">
            <button id="reset">Reset</button>
            <button id="play">${playing ? "Pause" : "Play"}</button>
            <input id="time" type="range" min="0" max="${scenario.frames.length - 1}" value="${frameIndex}">
            <span class="meta">${frame.timestamp_s.toFixed(1)} s</span>
          </div>
          <div class="caption">${esc(frame.caption)}</div>
          ${timelineHtml(scenario, frame)}
        </section>
        <section class="panel">
          <h2>Mug query</h2>
          <div class="query">${esc(frame.query.answer)}</div>
          <div class="meta" style="margin-top:7px">State ${badge(frame.query.state)} position ${frame.query.position_m.map(v => v.toFixed(2)).join(", ")} m · ${Math.round(frame.query.confidence * 100)}%</div>
        </section>
      </div>
      <div class="stack">
        <section class="panel"><h2>Decision and comparison</h2>${decisionHtml(frame)}</section>
        <section class="panel"><h2>Evidence for ${esc(frame.query.object_name)}</h2>${evidenceHtml(frame)}</section>
      </div>
    </div>`;
  drawMap(document.getElementById("ego-canvas"), frame.ego_map, "WISE-EgoMap", "#57d1c9", frame);
  drawMap(document.getElementById("instant-canvas"), frame.instant_map, "Instant update", "#ffc857", frame);
  document.getElementById("time").addEventListener("input", (event) => {
    stop(); frameIndex = Number(event.target.value); render();
  });
  document.getElementById("play").addEventListener("click", () => playing ? stop() : play());
  document.getElementById("reset").addEventListener("click", () => { stop(); frameIndex = 0; render(); });
}
function play() {
  playing = true;
  const scenario = data.scenarios[active];
  timer = setInterval(() => {
    frameIndex += 1;
    if (frameIndex >= scenario.frames.length) { stop(); frameIndex = scenario.frames.length - 1; }
    render();
  }, 900);
  render();
}
function stop() {
  playing = false;
  if (timer) clearInterval(timer);
  timer = null;
}
render();
"""
