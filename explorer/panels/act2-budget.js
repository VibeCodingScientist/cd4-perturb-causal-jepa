/* Act 2 — "What's actually learnable — the predictability budget."
 * Consumes App.data.act2 (fraction_of_ceiling.csv + budget_decomposition.csv + budget_cross_donor.csv).
 * Shows raw δ BESIDE fraction-of-ceiling so the dissociation is visible. Responds to App.state.split. */
(function () {
  "use strict";
  var A, refBudget, refFrac, refCross, refRecovery;
  function d() { return App.data.act2 || {}; }
  function sp() { return App.state.split; }
  function h(tag, cls, html) { var e = App.el(tag, cls); if (html != null) e.innerHTML = html; return e; }

  App.registerPanel("act2", {
    title: "The reframe",
    render: function (root) {
      A = d();
      root.appendChild(h("div", "panel__head",
        "<div class='panel__eyebrow'>Act 2 · the reframe</div>" +
        "<h1 class='panel__title'>Raw δ is uninterpretable — calibrate to the reliability ceiling</h1>" +
        "<p class='panel__lede'>A raw Pearson-δ score means nothing without knowing how much signal was reliably <em>there</em> to recover. So every score is read relative to the measured reliability ceiling, per axis. The response then splits into <b>A</b> linear-explainable, <b>C</b> structured-but-unmodelled, and <b>B</b> irreducible noise — and the two axes reach very different fractions of what's achievable, by completely different mechanisms, exactly what raw δ hides.</p>"));

      if (!A.budget) { root.appendChild(h("div", "card", "<p class='muted'>Act 2 data unavailable.</p>")); return; }

      refBudget = h("div", "card card--pad-lg"); root.appendChild(refBudget);
      var grid = h("div", "grid grid--2"); grid.style.marginTop = "18px";
      refFrac = h("div", "card"); refCross = h("div", "card");
      grid.appendChild(refFrac); grid.appendChild(refCross); root.appendChild(grid);
      refRecovery = h("div", "card"); refRecovery.style.marginTop = "18px"; root.appendChild(refRecovery);

      var note = h("div", "data-note"); note.style.marginTop = "18px";
      note.innerHTML = "<span data-icon='info'></span><div>" + (A.notes || []).join(" ") + "</div>";
      root.appendChild(note);

      this.update();
    },

    update: function () {
      if (!A || !A.budget) return;
      var s = sp();
      buildBudget(s);
      buildFrac(s);
      buildCross(s);
      buildRecovery(s);
      App.hydrateIcons(refBudget.parentNode);
    }
  });

  function seg(cls, frac, label) {
    var pct = Math.max(0, (frac || 0)) * 100;
    var txt = pct >= 12 ? label : "";
    return "<div class='budget__seg budget__seg--" + cls + "' style='width:" + pct.toFixed(1) + "%' title='" + label + "'>" + txt + "</div>";
  }

  function buildBudget(s) {
    var b = A.budget[s];
    var dominated = s === "gene" ? "structure-dominated" : "linear-dominated";
    refBudget.innerHTML =
      "<div class='card__title'>" + App.icon("layers") + "<span>Where the achievable signal lives — " + (s === "condition" ? "condition hold-out" : "gene hold-out") + "</span></div>" +
      "<p class='card__sub'>" + (s === "condition"
        ? "The condition shift is <b>linear-dominated</b>: a simple gene→δ map reaches most of the achievable signal."
        : "The gene shift is <b>structure-dominated</b>: the linear map collapses to the noise floor, yet the achievable ceiling is large. Almost the entire signal is <b>structured and unmodelled</b>.") + "</p>" +
      "<div class='budget'>" +
      seg("A", b.A, "A · linear " + App.fmt.num(b.A, ".2f")) +
      seg("C", b.C, "C · structured " + App.fmt.num(b.C, ".2f")) +
      seg("B", b.B, "B · noise " + App.fmt.num(b.B, ".2f")) +
      "</div>" +
      "<div class='budget-legend'>" +
      "<span><span class='sw' style='background:var(--accent)'></span><b>A</b> linear-explainable — " + App.fmt.num(b.A, ".2f") + "</span>" +
      "<span><span class='sw' style='background:var(--warn)'></span><b>C</b> structured residual — " + App.fmt.num(b.C, ".2f") + "</span>" +
      "<span><span class='sw' style='background:#B9B1A4'></span><b>B</b> irreducible noise — " + App.fmt.num(b.B, ".2f") + "</span>" +
      "<span class='muted'>(" + dominated + "; achievable ceiling r = " + App.fmt.num(b.r_ceiling, ".2f") + ")</span>" +
      "</div>";
  }

  function buildFrac(s) {
    var rows = (A.frac_of_ceiling && A.frac_of_ceiling[s]) || [];
    var html = "<div class='card__title'>" + App.icon("bar-chart") + "<span>Raw δ, beside fraction-of-ceiling</span></div>" +
      "<p class='card__sub'>Same models, two columns: the median per-perturbation raw Pearson (uncalibrated) and the fraction of each axis's <em>achievable</em> ceiling it captures. The dissociation between the two is the point.</p>" +
      "<table class='table'><thead><tr><th>Model</th><th class='num'>raw δ</th><th class='num'>fraction of ceiling</th></tr></thead><tbody>";
    rows.forEach(function (r) {
      var mark = (r.model === "ridge" || r.model === "causal") ? " style='background:var(--accent-wash)'" : "";
      html += "<tr" + mark + "><td>" + r.label + "</td><td class='num'>" + App.fmt.num(r.raw, ".3f") + "</td>" +
        "<td class='num'>" + App.fmt.num(r.frac, ".3f") + " <span class='muted' style='font-size:11px'>(" + App.fmt.num(r.frac_lo, ".2f") + "–" + App.fmt.num(r.frac_hi, ".2f") + ")</span></td></tr>";
    });
    html += "</tbody></table>";
    refFrac.innerHTML = html;
  }

  function buildCross(s) {
    var c = (A.cross_donor && A.cross_donor[s]) || {};
    var ratio = (c.specific_r && c.null_p95) ? Math.round(c.specific_r / c.null_p95) : null;
    refCross.innerHTML =
      "<div class='card__title'>" + App.icon("check") + "<span>Bucket C is <em>real</em>, not fit-noise</span></div>" +
      "<p class='card__sub'>The perturbation-specific structured residual reproduces across donors, far above a shuffled-label null — so it is signal we are failing to model, not overfitting.</p>" +
      "<div class='stat-row'>" +
      "<div class='stat'><div class='stat__num pos'>" + App.fmt.num(c.specific_r, ".3f") + "</div><div class='stat__lab'>cross-donor specific-r</div></div>" +
      "<div class='stat'><div class='stat__num'>" + App.fmt.num(c.null_p95, ".4f") + "</div><div class='stat__lab'>shuffled-null 95th pct</div></div>" +
      "<div class='stat'><div class='stat__num'>&lt; 0.001</div><div class='stat__lab'>permutation p</div></div>" +
      "</div>" +
      (ratio ? "<p class='card__sub' style='margin-top:14px'>The specific residual reproduces <b>~" + ratio + "×</b> above the shuffled-label null (perm p &lt; 0.001).</p>" : "");
  }

  function buildRecovery(s) {
    var rec = A.recovery && A.recovery.gene;
    if (!rec) { refRecovery.hidden = true; return; }
    refRecovery.hidden = false;
    refRecovery.innerHTML =
      "<div class='card__title'>" + App.icon("zap") + "<span>How much of bucket C the do-operator recovers (gene axis)</span></div>" +
      "<div class='stat-row'>" +
      "<div class='stat'><div class='stat__num pos'>" + rec.pct_recovered + "%</div><div class='stat__lab'>of the achievable gene signal, recovered by the do-operator</div></div>" +
      "<div class='stat'><div class='stat__num neg'>" + rec.gap_pct + "%</div><div class='stat__lab'>a located gap — reproducible, structured, unmodelled by every model here</div></div>" +
      "</div>" +
      "<p class='card__sub' style='margin-top:12px'>The scientific frontier is that remaining gene-generalization structured residual. The next act asks what it is — and bounds it six ways.</p>";
  }
})();
