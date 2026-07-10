/* Act 3 — "The frontier, mapped six ways" (centerpiece).
 * Consumes App.data.act3: the ~0.03 pointwise floor, the residual's biological identity,
 * and six pre-registered negatives, each with a verdict + key number pulled from its gate CSV.
 * Split-agnostic. Every number comes from the data via App.fmt.* / the JSON strings. */
(function () {
  "use strict";
  var A;
  function d() { return App.data.act3 || {}; }
  function h(tag, cls, html) { var e = App.el(tag, cls); if (html != null) e.innerHTML = html; return e; }
  function esc(s) { return String(s == null ? "" : s); }

  App.registerPanel("act3", {
    title: "The frontier, six ways",
    render: function (root) {
      A = d();
      root.appendChild(h("div", "panel__head",
        "<div class='panel__eyebrow'>Act 3 · the frontier</div>" +
        "<h1 class='panel__title'>The frontier, mapped six ways</h1>" +
        "<p class='panel__lede'>The remaining per-perturbation signal sits at a low pointwise floor. Six <b>pre-registered</b> analyses each asked whether a different structure could push past it — and each returns a clean negative. This is not \"we solved perturbation prediction\"; it is \"we bounded the frontier honestly, six ways, with zero GPU.\"</p>"));

      if (!A.gates) { root.appendChild(h("div", "card", "<p class='muted'>Act 3 data unavailable.</p>")); return; }

      // floor headline
      var floor = A.floor || {};
      root.appendChild(h("div", "card card--pad-lg",
        "<div class='card__title'>" + App.icon("target") + "<span>The pointwise floor</span></div>" +
        "<div class='stat-hero'>" +
        "<div class='stat-hero__num'>" + App.fmt.num(floor.value, ".3f") + "</div>" +
        "<div class='stat-hero__cap'>" + esc(floor.label) + ". " + esc(floor.triangulation) + "</div>" +
        "</div>"));

      // the six-gate map
      var mapWrap = h("div", "card"); mapWrap.style.marginTop = "18px";
      mapWrap.appendChild(h("div", "card__title", App.icon("layers") + "<span>Six pre-registered negatives — click any to expand</span>"));
      mapWrap.appendChild(h("p", "card__sub", "Spanning <b>pointwise</b> and <b>relational</b>, <b>raw</b> and <b>specific</b>. Every verdict's number is read from its committed gate CSV."));
      var grid = h("div", "gate-grid");
      (A.gates || []).forEach(function (g) { grid.appendChild(buildGate(g)); });
      mapWrap.appendChild(grid);
      root.appendChild(mapWrap);

      // residual identity — "what the floor is"
      root.appendChild(buildResidual());

      // closing summary
      var sum = h("div", "data-note"); sum.style.marginTop = "18px";
      sum.innerHTML = "<span data-icon='check'></span><div><b>What this adds up to.</b> " + esc(A.summary) + "</div>";
      root.appendChild(sum);
    },
    update: function () { /* split-agnostic */ }
  });

  function buildGate(g) {
    var card = document.createElement("button");
    card.type = "button"; card.className = "gate"; card.setAttribute("aria-expanded", "false");
    var tags = "<span class='tag tag--verdict'>" + esc(g.verdict) + "</span>" +
      (g.family || []).map(function (f) { return "<span class='tag tag--axis'>" + esc(f) + "</span>"; }).join("");
    var k = g.key_number || {};
    card.innerHTML =
      "<div class='gate__top'><span class='gate__n'>" + g.n + "</span><span class='gate__name'>" + esc(g.name) + "</span></div>" +
      "<div class='gate__tags'>" + tags + "</div>" +
      "<div class='gate__headline'>" + esc(g.headline) + "</div>" +
      "<div class='gate__key'>" + esc(k.label) + ": <b>" + esc(k.value) + "</b></div>" +
      "<div class='gate__detail'>" + esc(k.detail) + "</div>" +
      "<div class='gate__more' hidden>" + esc(g.aside) +
      "<div class='gate__csv'>" + App.icon("info", 12) + " " + esc(g.csv) + "</div></div>";
    var more = card.querySelector(".gate__more");
    card.addEventListener("click", function () {
      var open = more.hidden;
      more.hidden = !open; card.classList.toggle("is-open", open);
      card.setAttribute("aria-expanded", String(open));
    });
    return card;
  }

  function buildResidual() {
    var res = A.residual || {};
    var card = h("div", "card"); card.style.marginTop = "18px";
    var chips = (res.top_genes || []).map(function (g) {
      return "<span class='gene-chip'><b>" + esc(g.symbol) + "</b> <span class='ann'>" + esc(g.annotation) + "</span></span>";
    }).join("");
    card.innerHTML =
      "<div class='card__title'>" + App.icon("dna") + "<span>What the floor <em>is</em></span></div>" +
      "<p class='card__sub'>The reproducible part of the residual is not diffuse noise — it localizes to a specific, biologically coherent program: the <b>" + esc(res.program) + "</b>, peaking at the <b>" + esc(res.peaks) + "</b>. A far-from-equilibrium response; its <em>per-perturbation</em> part is noise-limited at pseudobulk depth.</p>" +
      "<div class='gene-chips'>" + chips + "</div>";
    return card;
  }
})();
