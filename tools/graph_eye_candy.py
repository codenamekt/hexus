"""
graph_eye_candy.py — Animated force-directed graph of hexus entity co-occurrences.

Talks directly to your Postgres via MemoryStore. No new pip deps.
Emits a single self-contained HTML file with a D3.js force layout
loaded from CDN — just open it in a browser.

Usage:
    # Overview: every co-occurring pair (constellation mode)
    python tools/graph_eye_candy.py

    # Walk mode: recursive BFS from a seed entity, colored by hop depth
    python tools/graph_eye_candy.py --seed domain example.com --max-depth 3

    # Filter to one agent's memory
    python tools/graph_eye_candy.py --agent my-agent --min-strength 3

DSN is read from $HEXUS_DSN (same env var the MCP server uses).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make `hexus` importable when run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hexus.store import MemoryStore

# ---------- data layer ----------


def fetch_overview(
    store: MemoryStore, agent: str | None, min_strength: int, limit: int
) -> tuple[list[dict], list[dict]]:
    """Constellation: all heavy co-occurrences, no seed required."""
    pairs = store.common_topics(
        agent_identity=agent, min_strength=min_strength, limit=limit
    )
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    for p in pairs:
        a_id = f"{p['type_a']}:{p['value_a']}"
        b_id = f"{p['type_b']}:{p['value_b']}"
        nodes.setdefault(
            a_id, {"id": a_id, "type": p["type_a"], "value": p["value_a"], "weight": 0}
        )
        nodes.setdefault(
            b_id, {"id": b_id, "type": p["type_b"], "value": p["value_b"], "weight": 0}
        )
        nodes[a_id]["weight"] += p["strength"]
        nodes[b_id]["weight"] += p["strength"]
        edges.append(
            {
                "source": a_id,
                "target": b_id,
                "strength": p["strength"],
            }
        )

    return list(nodes.values()), edges


def fetch_walk(
    store: MemoryStore,
    seed_type: str,
    seed_value: str,
    agent: str | None,
    max_depth: int,
    limit: int,
) -> tuple[list[dict], list[dict]]:
    """Recursive walk: edges reconstructed from per-hop results."""
    hops = store.graph_walk(
        entity_type=seed_type,
        entity_value=seed_value,
        agent_identity=agent,
        max_depth=max_depth,
        limit=limit,
    )

    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    seed_id = f"{seed_type}:{seed_value}"
    nodes[seed_id] = {"id": seed_id, "type": seed_type, "value": seed_value, "depth": 0}

    for h in hops:
        nid = f"{h['type']}:{h['value']}"
        nodes.setdefault(
            nid,
            {
                "id": nid,
                "type": h["type"],
                "value": h["value"],
                "depth": h["min_depth"],
            },
        )
        nodes[nid]["depth"] = min(nodes[nid].get("depth", 99), h["min_depth"])
        edges.append(
            {
                "source": seed_id,
                "target": nid,
                "depth": h["min_depth"],
                "occurrences": h["occurrences"],
            }
        )

    return list(nodes.values()), edges


# ---------- HTML template ----------

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>hexus — entity graph</title>
<style>
  :root {{
    --bg-0:#07080d; --bg-1:#0f1219;
    --fg:#e6e8ee; --fg-dim:#8a90a2;
    --accent:#7dd3fc; --warn:#fbbf24;
  }}
  html,body{{margin:0;padding:0;height:100%;background:radial-gradient(ellipse at 30% 20%,#0f1219 0%,#07080d 60%);color:var(--fg);font:13px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif;overflow:hidden}}
  #graph{{width:100vw;height:100vh;display:block;cursor:grab}}
  #graph:active{{cursor:grabbing}}
  .legend{{position:fixed;top:14px;left:14px;background:rgba(15,18,25,.78);backdrop-filter:blur(8px);padding:10px 12px;border-radius:10px;border:1px solid #1f2532;max-width:280px}}
  .legend h1{{margin:0 0 6px;font-size:13px;font-weight:600;letter-spacing:.02em;color:var(--accent)}}
  .legend .meta{{color:var(--fg-dim);font-size:11px}}
  .legend ul{{list-style:none;margin:8px 0 0;padding:0;display:grid;grid-template-columns:1fr 1fr;gap:3px 10px}}
  .legend li{{display:flex;align-items:center;gap:6px;font-size:11px}}
  .swatch{{width:9px;height:9px;border-radius:50%;flex-shrink:0}}
  .tip{{position:fixed;pointer-events:none;background:rgba(7,8,13,.94);border:1px solid #2a3142;border-radius:6px;padding:6px 9px;font-size:11px;color:var(--fg);max-width:280px;opacity:0;transition:opacity .12s;z-index:10}}
  .tip b{{color:var(--accent)}}
  .hint{{position:fixed;bottom:14px;right:14px;color:var(--fg-dim);font-size:11px}}
  .controls{{position:fixed;top:14px;right:14px;background:rgba(15,18,25,.78);backdrop-filter:blur(8px);padding:8px 10px;border-radius:10px;border:1px solid #1f2532;display:flex;gap:6px}}
  .controls button{{background:#1a2030;color:var(--fg);border:1px solid #2a3142;border-radius:6px;padding:5px 9px;font-size:11px;cursor:pointer}}
  .controls button:hover{{border-color:var(--accent);color:var(--accent)}}
  .controls label{{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--fg-dim)}}
  node{{stroke:#0b0d13;stroke-width:1.5px;transition:opacity .2s}}
  node.muted{{opacity:.15}}
  link{{stroke-opacity:.35;transition:stroke-opacity .2s}}
  link.muted{{stroke-opacity:.04}}
</style>
</head>
<body>
<svg id="graph"></svg>
<div class="legend">
  <h1>hexus ◦ entity graph</h1>
  <div class="meta">{subtitle}</div>
  <ul id="legend-types"></ul>
</div>
<div class="controls">
  <button id="btn-reset">reset view</button>
  <button id="btn-freeze">pause physics</button>
  <label><input type="checkbox" id="toggle-labels" checked /> labels</label>
</div>
<div class="tip" id="tip"></div>
<div class="hint">drag nodes · scroll to zoom · click node to pin its neighborhood</div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const DATA = {data_json};
const TYPE_COLORS = {{
  url:"#7dd3fc", domain:"#a78bfa", email:"#f472b6", file_path:"#fbbf24",
  version:"#34d399", ip_address:"#fb7185", docker_image:"#60a5fa", hostname:"#facc15",
}};
const FALLBACK = "#94a3b8";

const svg = d3.select("#graph");
const width = window.innerWidth, height = window.innerHeight;
svg.attr("viewBox", [0, 0, width, height]);

const g = svg.append("g");

const zoom = d3.zoom().scaleExtent([0.2, 8]).on("zoom", (e) => g.attr("transform", e.transform));
svg.call(zoom);

const link = g.append("g").attr("stroke-linecap","round")
  .selectAll("line").data(DATA.edges).join("line")
    .attr("stroke","#5b6478")
    .attr("stroke-width", d => Math.min(1 + Math.log2((d.strength||d.occurrences||1)+1)*1.2, 6));

const node = g.append("g")
  .selectAll("circle").data(DATA.nodes).join("circle")
    .attr("r", d => 5 + Math.min(Math.sqrt(d.weight||1)*3, 22))
    .attr("fill", d => TYPE_COLORS[d.type] || FALLBACK)
    .attr("data-id", d => d.id)
    .call(drag(simulation));

const label = g.append("g")
  .selectAll("text").data(DATA.nodes).join("text")
    .text(d => d.value.length > 28 ? d.value.slice(0,25)+"…" : d.value)
    .attr("font-size", 10)
    .attr("fill", "#cbd5e1")
    .attr("dx", d => 6 + Math.min(Math.sqrt(d.weight||1)*3, 22))
    .attr("dy", 4)
    .attr("paint-order","stroke")
    .attr("stroke","#07080d")
    .attr("stroke-width",3);

const simulation = d3.forceSimulation(DATA.nodes)
  .force("link", d3.forceLink(DATA.edges).id(d => d.id).distance(d => 70 / Math.sqrt((d.strength||d.occurrences||1))).strength(0.6))
  .force("charge", d3.forceManyBody().strength(-180))
  .force("center", d3.forceCenter(width/2, height/2))
  .force("collide", d3.forceCollide().radius(d => 8 + Math.min(Math.sqrt(d.weight||1)*3, 22)))
  .on("tick", () => {{
    link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
    node.attr("cx", d => d.x).attr("cy", d => d.y);
    label.attr("x", d => d.x).attr("y", d => d.y);
  }});

// --- interactions ---
const tip = d3.select("#tip");
node.on("mouseover", (e,d) => {{
  const neighbors = DATA.edges.filter(l => l.source.id===d.id || l.target.id===d.id).length;
  tip.style("opacity",1)
     .html(`<b>${{escapeHtml(d.value)}}</b><br><span style="color:#8a90a2">${{d.type}}${{d.weight?` · weight ${{d.weight}}`:""}}${{d.depth!=null?` · hop ${{d.depth}}`:""}}<br>${{neighbors}} connection${{neighbors===1?"":"s"}}</span>`)
     .style("left",(e.clientX+12)+"px").style("top",(e.clientY+12)+"px");
}}).on("mousemove", (e) => tip.style("left",(e.clientX+12)+"px").style("top",(e.clientY+12)+"px"))
  .on("mouseout", () => tip.style("opacity",0))
  .on("click", (e,d) => {{
    const keep = new Set([d.id, ...DATA.edges.filter(l => l.source.id===d.id || l.target.id===d.id).flatMap(l => [l.source.id, l.target.id])]);
    node.classed("muted", n => !keep.has(n.id));
    link.classed("muted", l => l.source.id!==d.id && l.target.id!==d.id && !(keep.has(l.source.id) && keep.has(l.target.id)));
    label.style("display", n => keep.has(n.id) ? null : "none");
  }});

svg.on("click", (e) => {{
  if (e.target !== svg.node()) return;
  node.classed("muted", false); link.classed("muted", false);
  label.style("display", d3.select("#toggle-labels").property("checked") ? null : "none");
}});

function drag(sim) {{
  function dragstarted(e,d) {{ if (!e.active) sim.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; }}
  function dragged(e,d) {{ d.fx=e.x; d.fy=e.y; }}
  function dragended(e,d) {{ if (!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; }}
  return d3.drag().on("start",dragstarted).on("drag",dragged).on("end",dragended);
}}

document.getElementById("btn-reset").onclick = () => svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
let frozen = false;
document.getElementById("btn-freeze").onclick = (ev) => {{
  frozen = !frozen; ev.target.textContent = frozen ? "resume physics" : "pause physics";
  frozen ? simulation.stop() : simulation.alpha(0.5).restart();
}};
document.getElementById("toggle-labels").onchange = (ev) => label.style("display", ev.target.checked ? null : "none");

// --- legend ---
const typesSeen = [...new Set(DATA.nodes.map(n => n.type))].sort();
const legend = d3.select("#legend-types").selectAll("li").data(typesSeen).join("li")
  .html(t => `<span class="swatch" style="background:${{TYPE_COLORS[t]||FALLBACK}}"></span>${{t}}`);

function escapeHtml(s) {{ return s.replace(/[&<>"']/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}})[c]); }}
</script>
</body>
</html>
"""


def render_html(nodes: list[dict], edges: list[dict], subtitle: str) -> str:
    payload = {"nodes": nodes, "edges": edges}
    return HTML_TEMPLATE.format(subtitle=subtitle, data_json=json.dumps(payload))


# ---------- CLI ----------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dsn",
        default=os.environ.get("HEXUS_DSN"),
        help="Postgres DSN (default: $HEXUS_DSN)",
    )
    ap.add_argument("--agent", default=None, help="Filter to one agent_identity")
    ap.add_argument(
        "--seed",
        nargs=2,
        metavar=("TYPE", "VALUE"),
        help="Walk mode: graph_walk from <type>:<value>",
    )
    ap.add_argument("--max-depth", type=int, default=2, help="Walk depth (1-5)")
    ap.add_argument(
        "--min-strength",
        type=int,
        default=2,
        help="Min co-occurrence count (overview mode)",
    )
    ap.add_argument("--limit", type=int, default=80, help="Max edges/nodes")
    ap.add_argument("-o", "--out", default="hexus-graph.html", help="Output HTML path")
    args = ap.parse_args()

    if not args.dsn:
        print("error: set HEXUS_DSN or pass --dsn", file=sys.stderr)
        return 2

    store = MemoryStore(dsn=args.dsn)

    if args.seed:
        seed_type, seed_value = args.seed
        nodes, edges = fetch_walk(
            store, seed_type, seed_value, args.agent, args.max_depth, args.limit
        )
        subtitle = f"walk from <b>{seed_type}:{seed_value}</b> · depth ≤ {args.max_depth} · {len(nodes)} nodes / {len(edges)} edges"
        if args.agent:
            subtitle += f" · agent=<b>{args.agent}</b>"
    else:
        nodes, edges = fetch_overview(store, args.agent, args.min_strength, args.limit)
        subtitle = f"overview · min strength {args.min_strength} · {len(nodes)} nodes / {len(edges)} edges"
        if args.agent:
            subtitle += f" · agent=<b>{args.agent}</b>"

    if not nodes:
        print("no data — check DSN / agent filter / seed", file=sys.stderr)
        return 1

    html = render_html(nodes, edges, subtitle)
    out_path = Path(args.out).resolve()
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path}  ({len(nodes)} nodes, {len(edges)} edges)")
    print(f"open with:  xdg-open {out_path}    # or just drag into a browser")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
