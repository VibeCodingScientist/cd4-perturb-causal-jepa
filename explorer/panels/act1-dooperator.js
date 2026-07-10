/* Act 1 — "The do-operator works, and here's what it buys."
 * Consumes App.data.act1 (from results/benchmark_table.csv + do_operator_localization.csv).
 * Every number comes from the data via App.fmt.* — no literals. Responds to App.state.split. */
(function () {
  "use strict";
  var A, refHero, refLead, refLoc;
  function d() { return App.data.act1 || {}; }
  function sp() { return App.state.split; }
  function splitLabel(s) { return s === "condition" ? "condition hold-out (zero-shot Stim48hr)" : "gene hold-out (unseen silenced genes)"; }
  function h(tag, cls, html) { var e = App.el(tag, cls); if (html != null) e.innerHTML = html; return e; }

  App.registerPanel("act1", {
    title: "The do-operator works",
    render: function (root) {
      A = d();
      root.appendChild(h("div", "panel__head",
        "<div class='panel__eyebrow'>Act 1 · the positive result</div>" +
        "<h1 class='panel__title'>The do-operator works — and here's what it buys</h1>" +
        "<p class='panel__lede'>Standard models treat a knockdown as an <em>observation</em>. The corrected do-mask removes only the edges <em>into</em> the perturbed gene, so the intervention still propagates downstream — treating it as an <em>intervention</em>. The pre-registered <b>C2</b> test isolates exactly that: same architecture, causal mask on vs off.</p>"));

      if (!A.c2) { root.appendChild(h("div", "card", "<p class='muted'>Act 1 data unavailable.</p>")); return; }

      refHero = h("div", "card card--pad-lg"); root.appendChild(refHero);
      var grid = h("div", "grid grid--2"); grid.style.marginTop = "18px";
      refLead = h("div", "card"); refLoc = h("div", "card");
      grid.appendChild(refLead); grid.appendChild(refLoc); root.appendChild(grid);

      var cav = h("div", "data-note"); cav.style.marginTop = "18px";
      cav.innerHTML = "<span data-icon='info'></span><div><b>Honest caveats.</b> " + (A.caveats || []).join(" ") + "</div>";
      root.appendChild(cav);

      this.update();
    },

    update: function () {
      if (!A || !A.c2) return;
      var s = sp(), c2 = A.c2[s];

      refHero.innerHTML =
        "<div class='card__title'>" + App.icon("zap") + "<span>C2 · the do-operator effect — " + splitLabel(s) + "</span></div>" +
        "<div class='stat-hero'>" +
        "<div class='stat-hero__num'>" + App.fmt.signed(c2.delta, ".3f") + "</div>" +
        "<div class='stat-hero__cap'>Causal (do-mask on) <b>" + App.fmt.num(c2.causal, ".3f") + "</b> vs its non-causal twin <b>" + App.fmt.num(c2.noncausal, ".3f") + "</b> — a <b>+" + c2.pct + "%</b> relative gain in Pearson-δ over the top-50 DEGs. Reported regardless of leaderboard position.</div>" +
        "</div>";
      if (s === "gene" && A.zero_shot) {
        var zs = h("div", "stat-row",
          "<div class='stat'><div class='stat__num pos'>" + App.fmt.num(A.zero_shot.causal_gene, ".3f") + "</div><div class='stat__lab'>causal — generalizes to genes never seen silenced</div></div>" +
          "<div class='stat'><div class='stat__num neg'>" + App.fmt.num(A.zero_shot.ridge_gene, ".3f") + "</div><div class='stat__lab'>linear baseline (Ridge) — fully collapses</div></div>");
        zs.style.marginTop = "16px"; refHero.appendChild(zs);
      }

      buildLeaderboard(s);
      buildLocalization(s);
      App.hydrateIcons(refHero.parentNode.parentNode);
    }
  });

  function collapseChip(mc) {
    return mc
      ? "<span class='tag tag--warn'>" + App.icon("circle-half", 12) + "borderline</span>"
      : "<span class='tag tag--good'>" + App.icon("check", 12) + "clears bar</span>";
  }

  function buildLeaderboard(s) {
    var rows = (A.leaderboard && A.leaderboard[s]) || [];
    var html = "<div class='card__title'>" + App.icon("bar-chart") + "<span>Raw leaderboard — Pearson-δ (top-50)</span></div>" +
      "<p class='card__sub'>Raw Pearson-δ is baseline-dominated; Act 2 re-reads it as fraction-of-ceiling. The causal ▸ non-causal gap is the C2 effect.</p>" +
      "<table class='table'><thead><tr><th>Model</th><th class='num'>Pearson-δ</th><th>Mode-collapse</th></tr></thead><tbody>";
    rows.forEach(function (r) {
      var mark = (r.model === "causal" || r.model === "noncausal") ? " style='background:var(--accent-wash)'" : "";
      html += "<tr" + mark + "><td>" + r.label + "</td><td class='num'>" + App.fmt.num(r.pearson_delta, ".3f") + "</td><td>" + collapseChip(r.mode_collapse) + "</td></tr>";
    });
    html += "</tbody></table>";
    refLead.innerHTML = html;
  }

  function buildLocalization(s) {
    var L = (A.localization && A.localization[s]) || {};
    var localizes = (L.corr_gap_vs_reliability || 0) > 0;
    refLoc.innerHTML =
      "<div class='card__title'>" + App.icon("target") + "<span>The edge concentrates on <em>reliable</em> perturbations</span></div>" +
      "<p class='card__sub'>Per-perturbation, the do-operator's gain (causal − non-causal) is correlated with how reliably that perturbation reproduces — it is the only model whose edge tracks reliability, not noise.</p>" +
      "<div class='stat-row'>" +
      "<div class='stat'><div class='stat__num " + (localizes ? "pos" : "") + "'>" + App.fmt.signed(L.corr_gap_vs_reliability, ".3f") + "</div><div class='stat__lab'>corr( C2 gap , reliability )</div></div>" +
      "<div class='stat'><div class='stat__num'>" + App.fmt.signed(L.reliable, ".3f") + "</div><div class='stat__lab'>C2 gap on reliable perts</div></div>" +
      "<div class='stat'><div class='stat__num'>" + App.fmt.signed(L.unreliable, ".3f") + "</div><div class='stat__lab'>C2 gap on unreliable perts</div></div>" +
      "</div>" +
      "<p class='card__sub' style='margin-top:14px'>" +
      (s === "gene"
        ? "On unseen genes the gain is <b>" + App.fmt.signed(L.reliable, ".3f") + "</b> where the target reproduces vs <b>" + App.fmt.signed(L.unreliable, ".3f") + "</b> where it does not — the do-operator buys signal exactly where signal exists."
        : "On the condition shift the correlation is weakly positive (" + App.fmt.signed(L.corr_gap_vs_reliability, ".3f") + "); the localization is clearest on the gene axis.") +
      "</p>";
  }
})();
