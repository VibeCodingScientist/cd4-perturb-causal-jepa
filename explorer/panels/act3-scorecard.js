/* Act 3 — "The predictability scorecard" (v2 centerpiece).
 * Consumes App.data.act3: the committed scorecard SVG (hero), the seven probes + the C2
 * positive-control anchor (from predictability_audit_gate.csv), the residual identity, the
 * positive-control argument, and the Tier-2 novelty note. Split-agnostic. */
(function () {
  "use strict";
  var A;
  function d() { return App.data.act3 || {}; }
  function h(tag, cls, html) { var e = App.el(tag, cls); if (html != null) e.innerHTML = html; return e; }
  function esc(s) { return String(s == null ? "" : s); }
  var VERDICT_CLASS = { FAIL: "tag--verdict", NEGATIVE: "tag--verdict", "NO-GO": "tag--verdict",
    FLOOR: "tag--verdict", "IN-DIST": "tag--warn", POSITIVE: "tag--good" };

  App.registerPanel("act3", {
    title: "The scorecard",
    render: function (root) {
      A = d();
      root.appendChild(h("div", "panel__head",
        "<div class='panel__eyebrow'>Act 3 · the scorecard</div>" +
        "<h1 class='panel__title'>The predictability scorecard</h1>" +
        "<p class='panel__lede'>" + esc(A.finding) + "</p>"));
      if (!A.probes) { root.appendChild(h("div", "card", "<p class='muted'>Act 3 data unavailable.</p>")); return; }

      // HERO — the committed scorecard SVG, verbatim
      var hero = h("div", "card card--pad-lg");
      hero.appendChild(h("div", "card__title",
        App.icon("bar-chart") + "<span>Marson CD4 predictability scorecard</span>" +
        "<span class='tag tag--good' style='margin-left:auto'>" + App.icon("check", 12) +
        "G-PA.1: " + A.n_probes + "/" + A.n_probes + " reproduce committed verdicts</span>"));
      var fig = h("div", "scorecard-figure"); fig.innerHTML = A.scorecard_svg;
      fig.setAttribute("role", "img");
      fig.setAttribute("aria-label", "Predictability scorecard. Panel A: fraction of the reliability ceiling recovered on the gene hold-out — Ridge 0.02 collapses, the do-operator reaches 0.56. Panel B: seven probes each at the noise floor or in-distribution, plus the do-operator C2 positive control.");
      hero.appendChild(fig);
      root.appendChild(hero);

      // interactive probe list — 7 probes + the C2 anchor (Budget lives in Act 2)
      var listCard = h("div", "card"); listCard.style.marginTop = "18px";
      listCard.appendChild(h("div", "card__title", App.icon("layers") + "<span>Seven probes + the positive control — click any to expand</span>"));
      listCard.appendChild(h("p", "card__sub", "Each probe is scored against its own degree/label-preserving null and read relative to the measured reliability ceiling. Every number is read from that probe's committed gate CSV."));
      var grid = h("div", "gate-grid");
      A.probes.filter(function (p) { return p.code !== "Budget"; }).forEach(function (p) { grid.appendChild(buildProbe(p)); });
      listCard.appendChild(grid);
      root.appendChild(listCard);

      // the positive-control argument
      var pc = h("div", "data-note"); pc.style.marginTop = "18px";
      pc.innerHTML = "<span data-icon='zap'></span><div><b>Why this is a map, not a failure.</b> " + esc(A.positive_control_argument) + "</div>";
      root.appendChild(pc);

      // what the floor is — residual identity
      root.appendChild(buildResidual());

      // novelty — Tier-2, weak occupation (honest)
      var nov = (App.data.manifest && App.data.manifest.novelty) || {};
      root.appendChild(h("div", "data-note data-note--qual",
        "<span data-icon='info'></span><div><b>Novelty — " + esc(nov.tier) + " (" + esc(nov.occupation) + " occupation).</b> " + esc(nov.note) + "</div>"));

      // takeaway
      root.appendChild(h("div", "data-note",
        "<span data-icon='check'></span><div><b>The takeaway.</b> " + esc(A.summary) + "</div>"));
    },
    update: function () { /* split-agnostic */ }
  });

  function buildProbe(p) {
    var card = document.createElement("button");
    card.type = "button"; card.className = "gate" + (p.is_anchor ? " gate--anchor" : "");
    card.setAttribute("aria-expanded", "false");
    var vclass = VERDICT_CLASS[p.verdict] || "tag--axis";
    var tags = "<span class='tag " + vclass + "'>" + esc(p.verdict) + "</span>" +
      (p.family_tags || []).map(function (f) { return "<span class='tag tag--axis'>" + esc(f) + "</span>"; }).join("");
    card.innerHTML =
      "<div class='gate__top'><span class='gate__n" + (p.is_anchor ? " gate__n--pos" : "") + "'>" + esc(p.code) + "</span><span class='gate__name'>" + esc(p.name) + "</span></div>" +
      "<div class='gate__tags'>" + tags + "</div>" +
      "<div class='gate__key'>" + esc(p.key) + "</div>" +
      "<div class='gate__more' hidden><div class='gate__q'>" + esc(p.question) + "</div>" + esc(p.detail) +
      "<div class='gate__csv'>" + App.icon("info", 12) + " " + esc(p.source) + "</div></div>";
    var more = card.querySelector(".gate__more");
    card.addEventListener("click", function () {
      var open = more.hidden; more.hidden = !open;
      card.classList.toggle("is-open", open); card.setAttribute("aria-expanded", String(open));
    });
    return card;
  }

  function buildResidual() {
    var res = A.residual || {}; var card = h("div", "card"); card.style.marginTop = "18px";
    var chips = (res.top_genes || []).map(function (g) {
      return "<span class='gene-chip'><b>" + esc(g.symbol) + "</b> <span class='ann'>" + esc(g.annotation) + "</span></span>";
    }).join("");
    card.innerHTML =
      "<div class='card__title'>" + App.icon("dna") + "<span>What the floor <em>is</em></span></div>" +
      "<p class='card__sub'>The reproducible part of the residual localizes to the <b>" + esc(res.program) + "</b>, peaking at the <b>" + esc(res.peaks) + "</b> — a coherent biological program whose per-perturbation part is noise-limited at pseudobulk depth.</p>" +
      "<div class='gene-chips'>" + chips + "</div>";
    return card;
  }
})();
