/* =============================================================================
 * CD4+ Perturbation Explorer — core app
 * State + router + data loading + demo-badge + shared kit (icons, charts, a11y).
 * Panels self-register via App.registerPanel(). See DESIGN_SPEC.md §5/§6.
 * Vanilla JS, one global `App`. D3 v7 is the only external dep (window.d3).
 * ============================================================================= */
(function () {
  "use strict";

  var App = window.App = {};

  /* ---- constants ---------------------------------------------------------- */
  App.EASE = "cubic-bezier(0.16,1,0.3,1)";
  App.DUR = 260;
  App.reducedMotion = window.matchMedia
    ? window.matchMedia("(prefers-reduced-motion: reduce)").matches : false;
  var MINUS = "−"; // real minus glyph
  var DATA_FILES = ["manifest", "act1", "act2", "act3"];
  var OPTIONAL = {};
  var RESULT_FILES = ["act1", "act2", "act3"]; // decide the demo badge

  /* ---- tiny event bus ----------------------------------------------------- */
  var listeners = {};
  App.on = function (evt, fn) { (listeners[evt] || (listeners[evt] = [])).push(fn); return App; };
  App.emit = function (evt, payload) { (listeners[evt] || []).forEach(function (fn) { try { fn(payload); } catch (e) { console.error(e); } }); };

  /* ---- state -------------------------------------------------------------- */
  // Narrative dashboard state: which act is showing, and the eval axis (split) that
  // Act 1 / Act 2 respond to. Act 3 (the frontier map) is split-agnostic.
  App.state = { panel: "act1", split: "condition" };
  App.data = {};
  App.isReal = false;

  App.setState = function (patch) {
    var prev = App.state, next = {};
    for (var k in prev) next[k] = prev[k];
    for (var j in patch) next[j] = patch[j];
    var panelChanged = next.panel !== prev.panel;
    App.state = next;
    if (panelChanged) { renderPanel(next.panel); focusPanelHeading(); App.emit("panelchange", next.panel); }
    syncControls();
    App.emit("statechange", next);
    if (!panelChanged) updateActivePanel();
  };
  App.split = function () { return App.state.split; };

  /* ---- formatters (NEVER invent; null -> "—") ----------------------------- */
  function d3fmt(spec) { return (window.d3 && d3.format) ? d3.format(spec) : function (v) { return String(v); }; }
  App.fmt = {
    num: function (v, spec) { if (v == null || (typeof v === "number" && !isFinite(v))) return "—"; return d3fmt(spec || ".3f")(v).replace(/^-/, MINUS); },
    int: function (v) { if (v == null) return "—"; return d3fmt(",")(Math.round(v)); },
    signed: function (v, spec) { if (v == null || !isFinite(v)) return "—"; var s = d3fmt("+" + (spec || ".2f"))(v); return s.replace(/-/g, MINUS).replace(/^\+/, "+"); },
    pct: function (v, dp) { if (v == null) return "—"; return (v * 100).toFixed(dp == null ? 0 : dp) + "%"; },
    compact: function (v) { if (v == null) return "—"; return d3fmt(".2~s")(v).replace("G", "B"); }
  };


  /* ---- ICONS (inline SVG, Lucide-style) ---------------------------------- */
  var P = { // path/inner content keyed by name (24x24, stroke currentColor)
    activity: "<polyline points='22 12 18 12 15 21 9 3 6 12 2 12'/>",
    dna: "<path d='M4 3c0 6 16 6 16 12M20 3c0 6-16 6-16 12M5 8h14M5 16h14'/>",
    target: "<circle cx='12' cy='12' r='9'/><circle cx='12' cy='12' r='5'/><circle cx='12' cy='12' r='1.5'/>",
    sliders: "<line x1='4' y1='21' x2='4' y2='14'/><line x1='4' y1='10' x2='4' y2='3'/><line x1='12' y1='21' x2='12' y2='12'/><line x1='12' y1='8' x2='12' y2='3'/><line x1='20' y1='21' x2='20' y2='16'/><line x1='20' y1='12' x2='20' y2='3'/><line x1='1' y1='14' x2='7' y2='14'/><line x1='9' y1='8' x2='15' y2='8'/><line x1='17' y1='16' x2='23' y2='16'/>",
    "bar-chart": "<line x1='12' y1='20' x2='12' y2='10'/><line x1='18' y1='20' x2='18' y2='4'/><line x1='6' y1='20' x2='6' y2='16'/>",
    flask: "<path d='M9 3h6M10 3v6l-6 10a2 2 0 0 0 1.7 3h12.6A2 2 0 0 0 20 19l-6-10V3'/><line x1='7' y1='15' x2='17' y2='15'/>",
    cells: "<circle cx='7' cy='7' r='3.5'/><circle cx='17' cy='9' r='3'/><circle cx='11' cy='17' r='3.5'/>",
    "chevron-down": "<polyline points='6 9 12 15 18 9'/>",
    "chevron-right": "<polyline points='9 6 15 12 9 18'/>",
    search: "<circle cx='11' cy='11' r='7'/><line x1='21' y1='21' x2='16.65' y2='16.65'/>",
    check: "<polyline points='20 6 9 17 4 12'/>",
    x: "<line x1='18' y1='6' x2='6' y2='18'/><line x1='6' y1='6' x2='18' y2='18'/>",
    "circle-half": "<circle cx='12' cy='12' r='9'/><path d='M12 3a9 9 0 0 1 0 18z' fill='currentColor' stroke='none'/>",
    "arrow-up": "<line x1='12' y1='19' x2='12' y2='5'/><polyline points='5 12 12 5 19 12'/>",
    "arrow-down": "<line x1='12' y1='5' x2='12' y2='19'/><polyline points='19 12 12 19 5 12'/>",
    info: "<circle cx='12' cy='12' r='9'/><line x1='12' y1='16' x2='12' y2='11'/><circle cx='12' cy='8' r='.6' fill='currentColor'/>",
    layers: "<polygon points='12 2 2 7 12 12 22 7 12 2'/><polyline points='2 17 12 22 22 17'/><polyline points='2 12 12 17 22 12'/>",
    shuffle: "<polyline points='16 3 21 3 21 8'/><line x1='4' y1='20' x2='21' y2='3'/><polyline points='21 16 21 21 16 21'/><line x1='15' y1='15' x2='21' y2='21'/><line x1='4' y1='4' x2='9' y2='9'/>",
    zap: "<polygon points='13 2 3 14 12 14 11 22 21 10 12 10 13 2'/>",
    "alert-triangle": "<path d='M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z'/><line x1='12' y1='9' x2='12' y2='13'/><circle cx='12' cy='17' r='.6' fill='currentColor'/>",
    play: "<polygon points='6 3 20 12 6 21 6 3' fill='currentColor' stroke='none'/>",
    donor: "<circle cx='12' cy='8' r='4'/><path d='M4 21c0-4 4-6 8-6s8 2 8 6'/>",
    refresh: "<polyline points='23 4 23 10 17 10'/><path d='M20.5 15a9 9 0 1 1-2.1-9.4L23 10'/>",
    "help": "<circle cx='12' cy='12' r='9'/><path d='M9.5 9a2.5 2.5 0 1 1 3.5 2.3c-.8.4-1 .9-1 1.7'/><circle cx='12' cy='16.5' r='.6' fill='currentColor'/>"
  };
  App.icon = function (name, size) {
    var inner = P[name] || "<circle cx='12' cy='12' r='2' fill='currentColor' stroke='none'/>";
    var s = size || 20;
    return "<svg width='" + s + "' height='" + s + "' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' aria-hidden='true'>" + inner + "</svg>";
  };
  function hydrateIcons(root) {
    (root || document).querySelectorAll("[data-icon]").forEach(function (el) {
      if (el.__iconed) return; el.__iconed = 1; el.innerHTML = App.icon(el.getAttribute("data-icon"), +el.getAttribute("data-size") || 20);
    });
  }
  App.hydrateIcons = hydrateIcons;

  /* ---- a11y: visually-hidden data table under a chart -------------------- */
  App.a11y = {
    dataTable: function (container, cfg) {
      var old = container.querySelector(":scope > table.visually-hidden"); if (old) old.remove();
      var t = document.createElement("table"); t.className = "visually-hidden";
      var cap = document.createElement("caption"); cap.textContent = cfg.caption || ""; t.appendChild(cap);
      var thead = document.createElement("thead"), htr = document.createElement("tr");
      (cfg.columns || []).forEach(function (c) { var th = document.createElement("th"); th.textContent = c; htr.appendChild(th); });
      thead.appendChild(htr); t.appendChild(thead);
      var tb = document.createElement("tbody");
      (cfg.rows || []).forEach(function (r) { var tr = document.createElement("tr"); r.forEach(function (cell) { var td = document.createElement("td"); td.textContent = cell; tr.appendChild(td); }); tb.appendChild(tr); });
      t.appendChild(tb); container.appendChild(t);
    }
  };

  /* ---- data loading ------------------------------------------------------ */
  function loadOne(name) {
    if (window.__APP_DATA__ && (name in window.__APP_DATA__)) return Promise.resolve(window.__APP_DATA__[name]);
    return fetch("./data/" + name + ".json", { cache: "no-store" })
      .then(function (r) { if (!r.ok) throw new Error(name + " " + r.status); return r.json(); })
      .catch(function (e) { if (OPTIONAL[name]) return null; console.warn("data load failed:", name, e.message); return null; });
  }
  App.loadData = function () {
    return Promise.all(DATA_FILES.map(loadOne)).then(function (arr) {
      DATA_FILES.forEach(function (n, i) { if (arr[i]) App.data[n] = arr[i]; });
      computeBadge();
      App.emit("dataloaded", App.data);
      return App.data;
    });
  };
  function isRealFile(name) { var f = App.data[name]; return f && f._meta && f._meta.source === "real"; }
  function computeBadge() {
    var man = App.data.manifest;
    var allReal = RESULT_FILES.every(isRealFile) && !!(man && man.any_real === true);
    App.isReal = allReal;
    App.badgeVisible = function () { return !allReal; };
    var demo = document.getElementById("badge-demo"), real = document.getElementById("badge-real");
    if (demo) demo.hidden = allReal;
    if (real) real.hidden = !allReal;
  }
  App.badgeVisible = function () { return !App.isReal; };

  /* ---- panel registry + router ------------------------------------------- */
  var PANELS = {}; var PANEL_ORDER = ["act1", "act2", "act3"];
  var PANEL_META = {
    act1: { num: 1, icon: "zap", label: "The do-operator works" },
    act2: { num: 2, icon: "layers", label: "The predictability budget" },
    act3: { num: 3, icon: "target", label: "The frontier, six ways" }
  };
  App.registerPanel = function (id, def) { PANELS[id] = def; def.__rendered = false; };
  function renderPanel(id) {
    var main = document.getElementById("main"); if (!main) return;
    var def = PANELS[id]; main.innerHTML = "";
    if (!def) { main.innerHTML = "<div class='panel'><p class='muted'>Panel not available.</p></div>"; return; }
    var root = document.createElement("div"); root.className = "panel"; main.appendChild(root);
    try { def.render(root); def.__rendered = true; if (def.update) def.update(); } catch (e) { console.error("panel render error", id, e); root.innerHTML = "<p class='muted'>This panel hit an error. See console.</p>"; }
    hydrateIcons(main); main.scrollTop = 0;
    document.querySelectorAll(".nav__item").forEach(function (b) { b.setAttribute("aria-current", b.dataset.panel === id ? "page" : "false"); });
  }
  function updateActivePanel() { var def = PANELS[App.state.panel]; if (def && def.__rendered && def.update) { try { def.update(); } catch (e) { console.error(e); } hydrateIcons(document.getElementById("main")); } }
  // Move focus to the new panel's heading after a panel change (keyboard/SR landing spot).
  function focusPanelHeading() {
    var h = document.querySelector("#main .panel__title"); if (!h) return;
    h.setAttribute("tabindex", "-1");
    try { h.focus({ preventScroll: true }); } catch (e) { h.focus(); }
  }

  /* ---- nav ---------------------------------------------------------------- */
  function buildNav() {
    var nav = document.getElementById("nav"); if (!nav) return; nav.innerHTML = "";
    PANEL_ORDER.forEach(function (id) {
      var m = PANEL_META[id], def = PANELS[id];
      var b = document.createElement("button");
      b.className = "nav__item"; b.dataset.panel = id; b.type = "button";
      b.setAttribute("aria-current", App.state.panel === id ? "page" : "false");
      b.innerHTML = "<span class='nav__num'>" + m.num + "</span><span class='nav__label'>" + (def ? def.title || m.label : m.label) + "</span>";
      b.addEventListener("click", function () { App.setState({ panel: id }); });
      nav.appendChild(b);
    });
    var foot = document.createElement("div"); foot.className = "nav__foot";
    foot.innerHTML = "GSE278572 · primary CD4+ T cells<br/>CRISPRi Perturb-seq";
    nav.appendChild(foot);
  }

  /* ---- control bar: eval-axis (split) toggle + arc stepper --------------- */
  function buildControls() {
    var c = document.getElementById("control"); if (!c) return; c.innerHTML = "";
    var splits = (App.data.manifest && App.data.manifest.splits) || [
      { id: "condition", label: "Condition hold-out", sub: "zero-shot Stim48hr" },
      { id: "gene", label: "Gene hold-out", sub: "unseen silenced genes" }
    ];
    var grp = el("div", "control__group"); grp.appendChild(labelEl("Eval axis"));
    var seg = el("div", "seg"); seg.setAttribute("role", "group"); seg.setAttribute("aria-label", "Evaluation axis (hold-out)");
    splits.forEach(function (sp) {
      var b = document.createElement("button"); b.className = "seg__btn"; b.type = "button";
      b.textContent = sp.label; b.dataset.split = sp.id; if (sp.sub) b.title = sp.sub;
      b.setAttribute("aria-pressed", App.state.split === sp.id);
      b.addEventListener("click", function () { App.setState({ split: sp.id }); });
      seg.appendChild(b);
    });
    grp.appendChild(seg); c.appendChild(grp);
    var hint = el("span", "muted"); hint.style.fontSize = "12px"; hint.style.maxWidth = "30ch"; hint.style.lineHeight = "1.35";
    hint.textContent = "Acts 1–2 respond to the axis; Act 3 (the frontier) is axis-agnostic.";
    c.appendChild(hint);

    c.appendChild(el("span", "spacer"));

    // arc stepper — walk the three acts for the demo
    var step = el("div", "control__group");
    var prev = document.createElement("button"); prev.className = "btn"; prev.type = "button";
    prev.innerHTML = "<span aria-hidden='true'>‹</span><span>Prev</span>"; prev.setAttribute("aria-label", "Previous act");
    prev.addEventListener("click", function () { stepAct(-1); });
    var next = document.createElement("button"); next.className = "btn btn--primary"; next.type = "button";
    next.innerHTML = "<span>Next act</span><span aria-hidden='true'>›</span>"; next.setAttribute("aria-label", "Next act");
    next.addEventListener("click", function () { stepAct(1); });
    step.appendChild(prev); step.appendChild(next);
    c.appendChild(step);
    hydrateIcons(c); syncControls();
  }
  function stepAct(dir) {
    var i = PANEL_ORDER.indexOf(App.state.panel);
    var n = Math.max(0, Math.min(PANEL_ORDER.length - 1, i + dir));
    if (PANEL_ORDER[n] !== App.state.panel) App.setState({ panel: PANEL_ORDER[n] });
  }
  function syncControls() {
    var c = document.getElementById("control"); if (!c) return;
    c.querySelectorAll(".seg__btn[data-split]").forEach(function (b) { b.setAttribute("aria-pressed", b.dataset.split === App.state.split); });
  }
  App.rebuildControls = buildControls;

  /* ---- CHART KIT (D3, SVG, a11y-first) ----------------------------------- */
  function ensureChart(container) {
    var host = container.classList && container.classList.contains("chart") ? container : (function () { var d = document.createElement("div"); d.className = "chart"; container.appendChild(d); return d; })();
    host.querySelectorAll("svg,.chart__tooltip").forEach(function (n) { n.remove() });
    return host;
  }
  function makeTooltip(host) {
    var tt = document.createElement("div"); tt.className = "chart__tooltip"; host.appendChild(tt);
    return { el: tt, show: function (html, x, y) { tt.innerHTML = html; tt.style.left = x + "px"; tt.style.top = y + "px"; tt.style.opacity = 1; }, hide: function () { tt.style.opacity = 0; } };
  }
  // Render final state immediately (no draw-in) when reduced-motion is set OR the tab
  // is backgrounded (rAF is throttled there, which would otherwise freeze a draw-in
  // mid-way). A focused recording tab animates normally.
  App.staticRender = function () { return App.reducedMotion || (typeof document !== "undefined" && document.hidden); };
  App.chart = {};
  App.chart.frame = function (container, opts) {
    opts = opts || {}; var host = ensureChart(container);
    var W = opts.width || host.clientWidth || 640, H = opts.height || 260;
    var m = Object.assign({ top: 16, right: 18, bottom: 34, left: 44 }, opts.margin || {});
    var svg = d3.select(host).append("svg").attr("viewBox", "0 0 " + W + " " + H).attr("role", "img");
    if (opts.aria) svg.attr("aria-label", opts.aria);
    var g = svg.append("g").attr("transform", "translate(" + m.left + "," + m.top + ")");
    return { host: host, svg: svg, g: g, W: W, H: H, m: m, innerW: W - m.left - m.right, innerH: H - m.top - m.bottom, tooltip: makeTooltip(host) };
  };

  App.chart.barsDelta = function (container, rows, opts) {
    opts = opts || {}; rows = (rows || []).slice(0, opts.maxBars || 20);
    var host = ensureChart(container);
    var rowH = opts.rowH || 22, padTop = 8, padBot = 8, labelW = opts.labelW || 66, valW = 56;
    var W = opts.width || host.clientWidth || 640, H = padTop + padBot + rows.length * rowH;
    var innerW = W - labelW - valW - 14;
    var maxAbs = d3.max(rows, function (d) { return Math.max(Math.abs(d.delta || 0), Math.abs(d.ci_hi || 0), Math.abs(d.ci_lo || 0)); }) || 1;
    var x = d3.scaleLinear().domain([-maxAbs, maxAbs]).range([0, innerW]);
    var zero = labelW + x(0);
    var svg = d3.select(host).append("svg").attr("viewBox", "0 0 " + W + " " + H).attr("role", "img").attr("aria-label", opts.aria || "Predicted expression change bar chart");
    // zero line
    svg.append("line").attr("x1", zero).attr("x2", zero).attr("y1", padTop - 4).attr("y2", H - padBot).attr("stroke", "var(--ink-3)").attr("stroke-width", 1);
    var g = svg.append("g");
    var tt = makeTooltip(host);
    rows.forEach(function (d, i) {
      var y = padTop + i * rowH, up = (d.delta || 0) >= 0, color = up ? "var(--accent)" : "var(--negative)";
      var bw = Math.abs(x(d.delta || 0) - x(0)), bx = up ? zero : zero - bw;
      // label
      g.append("text").attr("x", labelW - 8).attr("y", y + rowH / 2 + 4).attr("text-anchor", "end").attr("fill", "var(--ink)").style("font-size", "12px").style("font-weight", "600").text(d.symbol || d.gene);
      // bar
      var bar = g.append("rect").attr("x", up ? zero : zero).attr("y", y + 3).attr("height", rowH - 8).attr("width", App.staticRender() ? bw : 0).attr("rx", 3).attr("fill", color);
      if (!App.staticRender()) bar.transition().duration(App.DUR + i * 12).ease(d3.easeCubicOut).attr("x", bx).attr("width", bw); else bar.attr("x", bx);
      // arrow glyph
      g.append("text").attr("x", up ? bx + bw + 6 : bx - 6).attr("y", y + rowH / 2 + 4).attr("text-anchor", up ? "start" : "end").attr("fill", color).style("font-size", "11px").style("font-weight", "700").text((up ? "▲ " : "▼ ") + App.fmt.signed(d.delta, ".2f"));
      // CI whisker (only if present) — shown on hover row
      var rowRect = g.append("rect").attr("x", 0).attr("y", y).attr("width", W).attr("height", rowH).attr("fill", "transparent").style("cursor", "pointer");
      rowRect.on("mousemove", function (ev) {
        var ci = (d.ci_lo != null && d.ci_hi != null) ? "<div class='t-row'><span>95% CI</span><b>" + App.fmt.signed(d.ci_lo, ".2f") + " … " + App.fmt.signed(d.ci_hi, ".2f") + "</b></div>" : "";
        var meas = (d.measured != null) ? "<div class='t-row'><span>measured</span><b>" + App.fmt.signed(d.measured, ".2f") + "</b></div>" : "";
        var r = host.getBoundingClientRect();
        tt.show("<div class='t-sym'>" + (d.symbol || d.gene) + "</div><div class='t-row'><span>predicted Δ</span><b>" + App.fmt.signed(d.delta, ".2f") + "</b></div>" + ci + meas, ev.clientX - r.left, y + 4);
      }).on("mouseleave", function () { tt.hide(); });
      // CI band drawn subtly
      if (d.ci_lo != null && d.ci_hi != null) {
        g.append("line").attr("x1", labelW + x(d.ci_lo)).attr("x2", labelW + x(d.ci_hi)).attr("y1", y + rowH / 2).attr("y2", y + rowH / 2).attr("stroke", color).attr("stroke-opacity", .35).attr("stroke-width", 1.5);
      }
      // measured overlay marker
      if (opts.showMeasured && d.measured != null) {
        g.append("path").attr("d", d3.symbol().type(d3.symbolDiamond).size(46)()).attr("transform", "translate(" + (labelW + x(d.measured)) + "," + (y + rowH / 2) + ")").attr("fill", "none").attr("stroke", "var(--ink)").attr("stroke-width", 1.6);
      }
    });
    App.a11y.dataTable(host, { caption: opts.caption || "Predicted expression change per gene", columns: ["Gene", "Predicted Δ", "CI low", "CI high", "Measured Δ"], rows: rows.map(function (d) { return [d.symbol || d.gene, App.fmt.signed(d.delta, ".2f"), App.fmt.num(d.ci_lo, ".2f"), App.fmt.num(d.ci_hi, ".2f"), d.measured != null ? App.fmt.signed(d.measured, ".2f") : "—"]; }) });
    return host;
  };

  App.chart.line = function (container, series, opts) {
    opts = opts || {}; var f = App.chart.frame(container, { height: opts.height || 280, margin: opts.margin || { top: 16, right: 20, bottom: 40, left: 56 }, aria: opts.aria || "Line chart" });
    var all = []; series.forEach(function (s) { s.points.forEach(function (p) { all.push(p); }); });
    var x = d3.scaleLinear().domain(opts.xDomain || d3.extent(all, function (p) { return p.x; })).nice().range([0, f.innerW]);
    var y = d3.scaleLinear().domain(opts.yDomain || [0, d3.max(all, function (p) { return Math.max(p.y, p.hi || 0); }) * 1.1]).nice().range([f.innerH, 0]);
    // gridlines
    y.ticks(5).forEach(function (t) { f.g.append("line").attr("class", "gridline").attr("x1", 0).attr("x2", f.innerW).attr("y1", y(t)).attr("y2", y(t)); });
    f.g.append("g").attr("class", "axis").attr("transform", "translate(0," + f.innerH + ")").call(d3.axisBottom(x).ticks(6).tickFormat(opts.xFmt || null));
    f.g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5).tickFormat(opts.yFmt || null));
    if (opts.xLabel) f.g.append("text").attr("x", f.innerW / 2).attr("y", f.innerH + 30).attr("text-anchor", "middle").attr("fill", "var(--ink-2)").style("font-size", "12px").text(opts.xLabel);
    if (opts.yLabel) f.g.append("text").attr("transform", "rotate(-90)").attr("x", -f.innerH / 2).attr("y", -f.m.left + 14).attr("text-anchor", "middle").attr("fill", "var(--ink-2)").style("font-size", "12px").text(opts.yLabel);
    var lineGen = d3.line().x(function (p) { return x(p.x); }).y(function (p) { return y(p.y); }).curve(d3.curveMonotoneX);
    series.forEach(function (s) {
      if (s.band) { var area = d3.area().x(function (p) { return x(p.x); }).y0(function (p) { return y(p.lo); }).y1(function (p) { return y(p.hi); }).curve(d3.curveMonotoneX); f.g.append("path").attr("d", area(s.band)).attr("fill", s.color || "var(--accent)").attr("opacity", .12); }
      var path = f.g.append("path").attr("d", lineGen(s.points)).attr("fill", "none").attr("stroke", s.color || "var(--accent)").attr("stroke-width", s.width || 2.5).attr("stroke-dasharray", s.style === "dashed" ? "6 5" : null).attr("stroke-linecap", "round");
      if (!App.staticRender() && s.style !== "dashed") { var L = path.node().getTotalLength(); path.attr("stroke-dasharray", L + " " + L).attr("stroke-dashoffset", L).transition().duration(700).ease(d3.easeCubicInOut).attr("stroke-dashoffset", 0); }
      s.points.forEach(function (p) { f.g.append("circle").attr("cx", x(p.x)).attr("cy", y(p.y)).attr("r", 3).attr("fill", s.color || "var(--accent)"); });
    });
    if (opts.annot) { var ax = x(opts.annot.x); f.g.append("line").attr("x1", ax).attr("x2", ax).attr("y1", 0).attr("y2", f.innerH).attr("stroke", "var(--warn)").attr("stroke-dasharray", "4 4").attr("stroke-width", 1.5); f.g.append("text").attr("x", ax + 6).attr("y", 12).attr("fill", "var(--warn)").style("font-size", "11px").style("font-weight", "700").text(opts.annot.label); }
    App.a11y.dataTable(f.host, { caption: opts.caption || opts.aria || "Line chart data", columns: [opts.xLabel || "x"].concat(series.map(function (s) { return s.name; })), rows: (function () { var xs = {}; series.forEach(function (s) { s.points.forEach(function (p) { (xs[p.x] || (xs[p.x] = {}))[s.name] = p.y; }); }); return Object.keys(xs).map(function (k) { return [k].concat(series.map(function (s) { return xs[k][s.name] != null ? App.fmt.num(xs[k][s.name], ".3f") : "—"; })); }); })() });
    return f;
  };

  App.chart.scatter = function (container, points, opts) {
    opts = opts || {}; var f = App.chart.frame(container, { height: opts.height || 340, margin: { top: 16, right: 20, bottom: 44, left: 48 }, aria: opts.aria || "Scatter plot" });
    var ext = [d3.min(points, function (p) { return Math.min(p.x, p.y); }), d3.max(points, function (p) { return Math.max(p.x, p.y); })];
    if (ext[0] == null) ext = [0, 1]; var pad = (ext[1] - ext[0]) * 0.06 || 0.1;
    var x = d3.scaleLinear().domain([ext[0] - pad, ext[1] + pad]).range([0, f.innerW]);
    var y = d3.scaleLinear().domain([ext[0] - pad, ext[1] + pad]).range([f.innerH, 0]);
    f.g.append("g").attr("class", "axis").attr("transform", "translate(0," + f.innerH + ")").call(d3.axisBottom(x).ticks(6));
    f.g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(6));
    if (opts.diagonal !== false) f.g.append("line").attr("x1", x(ext[0] - pad)).attr("y1", y(ext[0] - pad)).attr("x2", x(ext[1] + pad)).attr("y2", y(ext[1] + pad)).attr("stroke", "var(--ink-3)").attr("stroke-dasharray", "5 5").attr("stroke-width", 1);
    if (opts.xLabel) f.g.append("text").attr("x", f.innerW / 2).attr("y", f.innerH + 36).attr("text-anchor", "middle").attr("fill", "var(--ink-2)").style("font-size", "12px").text(opts.xLabel);
    if (opts.yLabel) f.g.append("text").attr("transform", "rotate(-90)").attr("x", -f.innerH / 2).attr("y", -36).attr("text-anchor", "middle").attr("fill", "var(--ink-2)").style("font-size", "12px").text(opts.yLabel);
    var dots = f.g.selectAll("circle.pt").data(points).enter().append("circle").attr("class", "pt")
      .attr("cx", function (p) { return x(p.x); }).attr("cy", function (p) { return y(p.y); }).attr("r", App.staticRender() ? 4 : 0)
      .attr("fill", function (p) { return p.y > p.x ? "var(--accent)" : "var(--negative)"; }).attr("fill-opacity", .62).attr("stroke", "var(--surface)").attr("stroke-width", .8).style("cursor", "pointer");
    if (!App.staticRender()) dots.transition().duration(App.DUR).delay(function (_, i) { return i * 4; }).attr("r", 4);
    dots.on("mousemove", function (ev, p) { var r = f.host.getBoundingClientRect(); f.tooltip.show("<div class='t-sym'>" + (p.symbol || p.gene) + "</div><div class='t-row'><span>" + (opts.xLabel || "x") + "</span><b>" + App.fmt.num(p.x, ".3f") + "</b></div><div class='t-row'><span>" + (opts.yLabel || "y") + "</span><b>" + App.fmt.num(p.y, ".3f") + "</b></div>", ev.clientX - r.left, ev.clientY - r.top); d3.select(this).attr("r", 6).attr("fill-opacity", 1); if (opts.onHover) opts.onHover(p); })
      .on("mouseleave", function () { f.tooltip.hide(); d3.select(this).attr("r", 4).attr("fill-opacity", .62); });
    App.a11y.dataTable(f.host, { caption: opts.caption || "Per-gene model comparison", columns: ["Gene", opts.xLabel || "x", opts.yLabel || "y"], rows: points.map(function (p) { return [p.symbol || p.gene, App.fmt.num(p.x, ".3f"), App.fmt.num(p.y, ".3f")]; }) });
    return f;
  };

  App.chart.heatmapMini = function (container, cfg, opts) {
    opts = opts || {}; var host = ensureChart(container);
    var rows = cfg.rows, cols = cfg.cols, values = cfg.values; // values[r][c]
    var cell = opts.cell || 46, labelW = opts.labelW || 70, headH = 22, W = labelW + cols.length * cell, H = headH + rows.length * cell;
    var maxAbs = 0; values.forEach(function (r) { r.forEach(function (v) { if (v != null) maxAbs = Math.max(maxAbs, Math.abs(v)); }); }); maxAbs = maxAbs || 1;
    var svg = d3.select(host).append("svg").attr("viewBox", "0 0 " + W + " " + H).attr("role", "img").attr("aria-label", opts.aria || "Expression change heatmap");
    cols.forEach(function (c, ci) { svg.append("text").attr("x", labelW + ci * cell + cell / 2).attr("y", 15).attr("text-anchor", "middle").attr("fill", "var(--ink-2)").style("font-size", "11.5px").style("font-weight", "600").text(shortCond(c)); });
    rows.forEach(function (rw, ri) {
      svg.append("text").attr("x", labelW - 8).attr("y", headH + ri * cell + cell / 2 + 4).attr("text-anchor", "end").attr("fill", "var(--ink)").style("font-size", "12px").style("font-weight", "600").text(rw);
      cols.forEach(function (c, ci) {
        var v = values[ri][ci], up = v >= 0;
        var col = v == null ? "var(--surface-2)" : (up ? "var(--accent)" : "var(--negative)"), op = v == null ? 1 : (0.18 + 0.72 * Math.abs(v) / maxAbs);
        svg.append("rect").attr("x", labelW + ci * cell + 3).attr("y", headH + ri * cell + 3).attr("width", cell - 6).attr("height", cell - 6).attr("rx", 6).attr("fill", col).attr("fill-opacity", op).attr("stroke", "var(--line)");
        if (v != null) svg.append("text").attr("x", labelW + ci * cell + cell / 2).attr("y", headH + ri * cell + cell / 2 + 4).attr("text-anchor", "middle").attr("fill", Math.abs(v) / maxAbs > .5 ? "#fff" : "var(--ink)").style("font-size", "11px").style("font-weight", "700").text((up ? "▲" : "▼") + Math.abs(v).toFixed(1));
      });
    });
    App.a11y.dataTable(host, { caption: opts.caption || "Expression change by state", columns: ["Gene"].concat(cols), rows: rows.map(function (rw, ri) { return [rw].concat(values[ri].map(function (v) { return v == null ? "—" : App.fmt.signed(v, ".1f"); })); }) });
    return host;
  };

  /* ---- small DOM helpers -------------------------------------------------- */
  function el(tag, cls) { var e = document.createElement(tag); if (cls) e.className = cls; return e; }
  function labelEl(t) { var s = el("span", "control__label"); s.textContent = t; return s; }
  function divider() { return el("span", "control__divider"); }
  function shortCond(c) { return c === "Stim8hr" ? "Stim 8h" : c === "Stim48hr" ? "Stim 48h" : c; }
  App.el = el; App.shortCond = shortCond;

  /* ---- boot --------------------------------------------------------------- */
  App.boot = function () {
    hydrateIcons(document);
    App.loadData().then(function () {
      buildNav(); buildControls(); renderPanel(App.state.panel);
      hydrateIcons(document);
    });
    window.addEventListener("resize", debounce(function () { if (App.state.panel) updateActivePanel(); }, 180));
  };
  function debounce(fn, ms) { var t; return function () { clearTimeout(t); t = setTimeout(fn, ms); }; }
})();
