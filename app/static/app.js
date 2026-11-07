// Plateful front end. No framework: one page, three views, a small JSON API.
(() => {
  "use strict";

  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  let status = null;
  let last = null; // last request, for "write another version"

  // ---------------------------------------------------------------- routing
  function route() {
    const view = (location.hash || "#cook").slice(1);
    const known = ["cook", "taste-test", "method"];
    const v = known.includes(view) ? view : "cook";
    $$(".view").forEach((el) => (el.hidden = el.dataset.view !== v));
    $$(".nav a").forEach((a) => {
      if (a.dataset.view === v) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
    });
    if (v === "taste-test") loadTasteTest();
    if (v === "method") renderMethod();
  }
  window.addEventListener("hashchange", () => { route(); window.scrollTo({ top: 0 }); });

  // ---------------------------------------------------------------- status
  async function loadStatus() {
    try {
      status = await (await fetch("/api/status")).json();
    } catch { status = null; }
    const s = status;
    const bit = (ok, label) => `<span class="${ok ? "on" : "off"}">${label}</span>`;
    $("#status-line").innerHTML = s
      ? [bit(s.finetuned, s.finetuned ? "Fine-tuned model ready" : "Fine-tuned model not trained"),
         bit(s.ner === "bert", s.ner === "bert" ? "BERT tagger" : "Rule tagger"),
         bit(s.image, s.image ? "Camera on" : "No camera")].join(" · ")
      : "Kitchen offline";
    if (s && !s.finetuned) {
      const base = $('input[name="model"][value="base"]');
      base.checked = true;
      $$('input[name="model"]').forEach((i) => { if (i.value !== "base") i.disabled = true; });
    }
    if (s && !s.image) {
      $("#want-image").checked = false;
      $("#want-image").disabled = true;
      $("#want-image").parentElement.title = "Install diffusers + torch to enable image generation";
    }
    if (!$("#view-method").hidden) renderMethod();
  }

  // ---------------------------------------------------------------- helpers
  const AMOUNT_RE = /^((?:[\d½¼¾⅓⅔⅛.\/-]+\s*)+(?:\([^)]*\)\s*)?(?:(?:c|cups?|tsp|tbsp|Tbsp|T|t|oz|lbs?|pkg|pkgs|packages?|cans?|jars?|cloves?|g|kg|ml|l|sticks?|pints?|pt|qt|quarts?|pounds?|ounces?|teaspoons?|tablespoons?|slices?|bunch|heads?|sprigs?|pinch|dash|large|small|medium|whole)\.?\s+)?)(.*)$/;

  function splitAmount(line) {
    const m = line.match(AMOUNT_RE);
    if (!m || !m[1].trim()) return [null, line];
    return [m[1].trim(), m[2]];
  }

  function estimateMinutes(steps) {
    let total = 0;
    for (const s of steps) {
      for (const m of s.matchAll(/(\d+)(?:\s*(?:-|to)\s*(\d+))?\s*(hours?|hrs?|minutes?|mins?)\b/gi)) {
        const n = Number(m[2] || m[1]);
        total += /h/i.test(m[3]) ? n * 60 : n;
      }
    }
    return total;
  }
  function fmtTime(min) {
    if (!min) return "—";
    if (min < 60) return `${min} min`;
    const h = Math.floor(min / 60), r = min % 60;
    return r ? `${h} h ${r}` : `${h} h`;
  }

  function flaggedSet(r) {
    const flags = r.safety?.quantity_flags || [];
    return new Set(flags.map((f) => f.line));
  }

  function ingredientList(r, idPrefix) {
    const flagged = flaggedSet(r);
    return `<ul class="ing-list">${r.ingredients.map((line, i) => {
      const [amt, rest] = splitAmount(line);
      return `<li class="${flagged.has(line) ? "flagged" : ""}" style="--i:${i}"><label>
        <input type="checkbox" id="${idPrefix}-i${i}"><span>${amt ? `<span class="amt">${esc(amt)}</span> ` : ""}${esc(rest)}</span>
      </label></li>`;
    }).join("")}</ul>`;
  }

  function methodList(r) {
    const flagged = flaggedSet(r);
    return `<ol class="method-list">${r.directions.map((s, i) =>
      `<li class="${flagged.has(s) ? "flagged" : ""}" style="--i:${i + 3}">${esc(s)}</li>`).join("")}</ol>`;
  }

  function isStructured(r) { return r.ingredients.length > 0; }

  function proseBlock(r) {
    return `<div class="prose">${r.directions.slice(0, 8).map((p) => `<p>${esc(p)}</p>`).join("") || "<p><em>(nothing usable)</em></p>"}</div>
      <p class="unstructured-note">No ingredient list or numbered steps. This is what GPT-2 writes before it has learned the format.</p>`;
  }

  function containsLine(r) {
    const a = Object.keys(r.safety?.allergens || {});
    return `<p class="contains"><b>Contains</b>${a.length ? a.map((x) => `<span class="a">${esc(x)}</span>`).join("") : `<span class="none">none of the major allergens detected</span>`}</p>`;
  }

  function avoidAlert(r) {
    const hits = r.safety?.avoid_conflicts || [];
    if (!hits.length) return "";
    const lines = hits.flatMap((h) => r.safety.allergens[h].map((l) => `<q>${esc(l)}</q>`));
    return `<div class="avoid-alert" role="alert"><strong>You asked to avoid ${esc(hits.join(", "))}</strong>
      This recipe uses ${lines.join(", ")}. Swap or leave these out.</div>`;
  }

  function notesBlock(flags, heading) {
    if (!flags.length) return "";
    return `<section class="notes"><h3 class="col-head">${esc(heading)} <span>${flags.length}</span></h3>
      ${flags.map((f) => `<div class="note"><span class="lvl ${f.level}">${f.level}</span>
        <div>${esc(f.reason)}<span class="src">${esc(f.line)}</span></div></div>`).join("")}</section>`;
  }

  function readingBlock(input) {
    const tagged = input.tagged.map((t) => `<span class="line">${t.spans.map((s) =>
      s.tag === "O" ? esc(s.text) : `<span class="tag tag--${s.tag.toLowerCase()}">${esc(s.text)}</span>`).join(" ")}</span>`).join("");
    return `<section class="reading">
      <h3 class="col-head">How the model read your list <span>${input.ner_engine === "bert" ? "BERT tagger" : "rule tagger"}</span></h3>
      <div class="reading-lines">${tagged}</div>
      <div class="legend">
        <span><i class="tag--qty" style="box-shadow: inset 0 -5px 0 var(--tomato-soft)"></i>amount</span>
        <span><i style="box-shadow: inset 0 -5px 0 var(--olive-soft)"></i>unit</span>
        <span><i style="background: var(--mustard)"></i>ingredient → prompt</span>
      </div>
    </section>`;
  }

  const who = (m) => m === "base"
    ? `<span class="who base">Untrained GPT-2</span>`
    : `<span class="who">Fine-tuned GPT-2</span>`;

  // ---------------------------------------------------------------- recipe views
  function renderFull(data, opts) {
    const r = data.recipes[0];
    const minutes = estimateMinutes(r.directions);
    const structured = isStructured(r);
    const title = r.title || "Untitled";
    const flagsIn = data.input.quantity_flags || [];
    return `<article class="recipe wrap">
      <div class="recipe-hero ${opts.image ? "" : "no-photo"}">
        ${opts.image ? `<figure class="photo"><div class="photo-frame loading" id="photo">
            <svg class="plate" viewBox="0 0 160 120" aria-hidden="true">
              <g class="steam"><path d="M62 44c-5-7 5-12 0-21"/><path d="M80 40c-5-8 6-13 0-24"/><path d="M98 44c-5-7 5-12 0-21"/></g>
              <ellipse cx="80" cy="84" rx="62" ry="16"/><ellipse cx="80" cy="82" rx="40" ry="9"/>
              <path d="M10 70v30M6 70v9c0 3 2 5 4 5s4-2 4-5v-9M150 70c-4 3-5 10-1 14v16"/>
            </svg>
            <span>Plating up for the photographer</span></div>
          <figcaption>Generated by SD-Turbo from the recipe title and ingredients. Not a photograph of a real dish.</figcaption></figure>` : ""}
        <div>
          <p class="byline">${who(r.model)}<span>${r.seconds}s</span></p>
          <h2 class="recipe-title">${esc(title)}</h2>
          <p class="standfirst">Written around <b>${data.input.names.map(esc).join("</b>, <b>")}</b>.</p>
          <dl class="meta">
            <div><dt>Ingredients</dt><dd>${r.ingredients.length || "—"}</dd></div>
            <div><dt>Steps</dt><dd>${structured ? r.directions.length : "—"}</dd></div>
            <div><dt>Time</dt><dd>${fmtTime(minutes)}</dd></div>
          </dl>
          ${containsLine(r)}
          ${avoidAlert(r)}
        </div>
      </div>

      ${structured ? `<div class="recipe-body">
        <aside class="ingredients"><h3 class="col-head">Ingredients <span>${r.ingredients.length}</span></h3>${ingredientList(r, "full")}</aside>
        <div><h3 class="col-head">Method <span>${r.complete ? "" : "unfinished"}</span></h3>${methodList(r)}
          ${notesBlock(r.safety.quantity_flags, "Kitchen notes")}
          ${notesBlock(flagsIn, "About your list")}
        </div>
      </div>` : `${proseBlock(r)}${notesBlock(flagsIn, "About your list")}`}

      ${readingBlock(data.input)}

      <div class="actions">
        <button class="btn" type="button" data-act="again">Write another version</button>
        <button class="btn btn--ghost" type="button" data-act="print">Print recipe</button>
      </div>
    </article>`;
  }

  function renderCard(r, ingredientsIdPrefix) {
    const structured = isStructured(r);
    return `<div class="recipe-card">
      <p class="byline">${who(r.model)}${r.seconds != null ? `<span>${r.seconds}s</span>` : ""}</p>
      <h3 class="${r.title ? "" : "untitled"}">${esc(r.title || "No title")}</h3>
      ${structured ? `<p class="subhead">Ingredients</p>${ingredientList(r, ingredientsIdPrefix)}
        <p class="subhead">Method</p>${methodList(r)}` : proseBlock(r)}
      ${Object.keys(r.safety?.allergens || {}).length ? `<p class="mini-contains">Contains · ${Object.keys(r.safety.allergens).map(esc).join(" · ")}</p>` : ""}
      ${notesBlock(r.safety?.quantity_flags || [], "Kitchen notes")}
    </div>`;
  }

  function renderDuo(data) {
    const order = ["base", "finetuned"];
    const rs = order.map((m) => data.recipes.find((r) => r.model === m)).filter(Boolean);
    return `<div class="wrap"><section class="duo">${rs.map((r, i) => renderCard(r, "duo" + i)).join("")}</section>
      ${readingBlock(data.input)}
      <div class="actions"><button class="btn" type="button" data-act="again">Try again with a new seed</button></div></div>`;
  }

  // ---------------------------------------------------------------- generate flow
  function setStep(name, state) {
    const li = $(`.progress [data-step="${name}"]`);
    if (li) { li.classList.remove("active", "done", "skip"); if (state) li.classList.add(state); }
  }

  async function generate(req) {
    last = req;
    const btn = $("#submit-btn");
    const err = $("#form-error");
    err.hidden = true;
    btn.disabled = true;
    $("#progress").hidden = false;
    $$(".progress li").forEach((li) => li.classList.remove("active", "done", "skip"));
    setStep("read", "active");
    $("#result").innerHTML = "";
    $("#progress").scrollIntoView({ behavior: "smooth", block: "start" });

    const t = setTimeout(() => { setStep("read", "done"); setStep("write", "active"); }, 500);
    let data;
    try {
      const res = await fetch("/api/generate", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req),
      });
      data = await res.json();
      if (!res.ok) throw new Error(data.error || "Something went wrong.");
    } catch (e) {
      clearTimeout(t);
      $("#progress").hidden = true;
      err.textContent = e.message;
      err.hidden = false;
      btn.disabled = false;
      return;
    }
    clearTimeout(t);
    ["read", "write", "check"].forEach((s) => setStep(s, "done"));

    const wantImage = req.image && req.model !== "both" && isStructured(data.recipes[0]);
    $("#result").innerHTML = req.model === "both" ? renderDuo(data) : renderFull(data, { image: wantImage });
    btn.disabled = false;

    if (wantImage) {
      setStep("photo", "active");
      const r = data.recipes[0];
      try {
        const img = await (await fetch("/api/image", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title: r.title, ingredients: data.input.names, seed: req.seed || 0 }),
        })).json();
        const frame = $("#photo");
        if (frame) {
          if (img.image) {
            frame.className = "photo-frame";
            frame.innerHTML = `<img src="${img.image}" alt="Generated photo of ${esc(r.title)}">`;
            frame.parentElement.querySelector("figcaption").title = img.prompt;
          } else {
            frame.className = "photo-frame missing";
            frame.innerHTML = `<span>The photographer couldn’t make it. ${esc(img.error || "")}</span>`;
          }
        }
        setStep("photo", img.image ? "done" : "skip");
      } catch { setStep("photo", "skip"); }
    } else {
      setStep("photo", "skip");
    }
    setTimeout(() => { $("#progress").hidden = true; }, 1200);
  }

  $("#cook-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const fd = new FormData(e.target);
    generate({
      ingredients: fd.get("ingredients"),
      model: fd.get("model"),
      avoid: fd.getAll("avoid"),
      image: $("#want-image").checked,
      seed: null,
    });
  });

  $$(".starter").forEach((b) => b.addEventListener("click", () => {
    const ta = $("#ingredients");
    ta.value = b.dataset.ingredients.split(", ").join("\n");
    ta.focus();
  }));

  document.addEventListener("click", (e) => {
    const act = e.target.closest("[data-act]")?.dataset.act;
    if (act === "again" && last) generate({ ...last, seed: Math.floor(Math.random() * 1e6) });
    if (act === "print") window.print();
  });

  // ---------------------------------------------------------------- taste test
  let tasteLoaded = false;
  async function loadTasteTest() {
    if (tasteLoaded) return;
    const res = await fetch("/api/comparison");
    if (!res.ok) {
      $("#taste-sets").innerHTML = `<p class="empty">No taste test yet. Run <code>python scripts/compare.py</code> after training.</p>`;
      return;
    }
    tasteLoaded = true;
    const d = await res.json();
    const rows = [
      ["Recipes with an ingredient list", "structured_outputs", (v) => `${v}<small style="font-size:.5em;color:var(--muted)"> / 5</small>`, "max"],
      ["Stopped cleanly at the end marker", "finished_cleanly", (v) => `${v}<small style="font-size:.5em;color:var(--muted)"> / 5</small>`, "max"],
      ["Your ingredients actually used", "mean_input_coverage", (v) => `${Math.round(v * 100)}%`, "max"],
      ["Validation perplexity (lower is better)", "val_perplexity", (v) => (v == null ? "—" : v.toFixed(1)), "min"],
    ];
    const S = d.summary;
    const win = (k, m, side) => {
      const b = S.base[k], f = S.finetuned[k];
      if (b == null || f == null || b === f) return "";
      const best = m === "max" ? (f > b ? "finetuned" : "base") : (f < b ? "finetuned" : "base");
      return best === side ? "win" : "";
    };
    $("#taste-summary").innerHTML = `<table class="summary-table">
      <caption>Scorecard</caption>
      <thead><tr><th></th><th>Untrained</th><th>Fine-tuned</th></tr></thead>
      <tbody>${rows.map(([label, k, fmt, m]) => `<tr><td>${label}</td>
        <td class="${win(k, m, "base")}">${fmt(S.base[k])}</td>
        <td class="${win(k, m, "finetuned")}">${fmt(S.finetuned[k])}</td></tr>`).join("")}</tbody></table>`;

    $("#taste-sets").innerHTML = d.sets.map((set, i) => {
      const b = { ...set.outputs.base, model: "base", seconds: null };
      const f = { ...set.outputs.finetuned, model: "finetuned", seconds: null };
      return `<section class="tt-set">
        <div class="tt-head"><span class="tt-no">${i + 1}</span>
          <p class="tt-ings"><span class="kicker">Shopping list</span>${set.ingredients.map(esc).join(", ")}</p></div>
        <div class="duo">${renderCard(b, `tt${i}b`)}${renderCard(f, `tt${i}f`)}</div>
      </section>`;
    }).join("");
  }

  // ---------------------------------------------------------------- method
  function stat(k, v, cls = "") { return `<div class="stat"><div class="k">${k}</div><div class="v ${cls}">${v}</div></div>`; }

  function renderMethod() {
    const t = status?.training;
    if (t) {
      $("#training-stats").innerHTML = [
        stat("Base perplexity", t.base.val_perplexity.toFixed(1)),
        stat("Fine-tuned perplexity", t.finetuned.val_perplexity.toFixed(1), "good"),
        stat("Training recipes", t.train_recipes.toLocaleString()),
        stat("Steps", `${t.steps}<small> · ${t.train_minutes} min</small>`),
      ].join("");
      if (t.loss_curve?.length > 1) { $("#loss-figure").hidden = false; drawLoss(t.loss_curve); }
    } else {
      $("#training-stats").innerHTML = `<p class="caption" style="padding:14px 0;margin:0">Not trained yet. Run <code>python scripts/train_gpt2.py</code>.</p>`;
    }
    const n = status?.ner_meta;
    $("#ner-stats").innerHTML = n?.val_entity_metrics ? [
      stat("Ingredient F1", n.val_entity_metrics.ING.f1.toFixed(2), "good"),
      stat("Amount F1", n.val_entity_metrics.QTY.f1.toFixed(2)),
      stat("Unit F1", n.val_entity_metrics.UNIT.f1.toFixed(2)),
      stat("Lines seen", (n.train_lines || 0).toLocaleString()),
    ].join("") : `<p class="caption" style="padding:14px 0;margin:0">BERT tagger not trained, so the rule-based tagger is in use. Run <code>python scripts/train_ner.py</code>.</p>`;
  }

  // Single-series line: loss vs step. 2px line, recessive grid, crosshair + tooltip on hover.
  function drawLoss(points) {
    const el = $("#loss-chart");
    const W = 760, H = 260, m = { t: 12, r: 12, b: 28, l: 40 };
    const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    const yMax = Math.ceil(Math.max(...ys) * 2) / 2, yMin = Math.max(0, Math.floor(Math.min(...ys) * 2) / 2);
    const sx = (v) => m.l + ((v - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
    const sy = (v) => m.t + (1 - (v - yMin) / (yMax - yMin || 1)) * (H - m.t - m.b);
    const ticksY = []; for (let v = yMin; v <= yMax + 1e-9; v += (yMax - yMin) / 4) ticksY.push(v);
    const ticksX = [x0, Math.round((x0 + x1) / 2), x1];
    const d = points.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join("");
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Training loss falls from ${ys[0].toFixed(2)} to ${ys[ys.length - 1].toFixed(2)} over ${x1} steps">
      <g class="grid">${ticksY.map((v) => `<line x1="${m.l}" x2="${W - m.r}" y1="${sy(v)}" y2="${sy(v)}"/>`).join("")}</g>
      <g class="axis">${ticksY.map((v) => `<text x="${m.l - 8}" y="${sy(v) + 4}" text-anchor="end">${+v.toFixed(2)}</text>`).join("")}
        ${ticksX.map((v) => `<text x="${sx(v)}" y="${H - 8}" text-anchor="middle">${v === x1 ? `step ${v}` : v}</text>`).join("")}</g>
      <path class="line" d="${d}"/>
      <line class="cross" y1="${m.t}" y2="${H - m.b}" visibility="hidden"/>
      <circle class="dot" r="5" visibility="hidden"/>
      <rect x="${m.l}" y="0" width="${W - m.l - m.r}" height="${H}" fill="transparent" class="hit"/>
    </svg><div class="tooltip" hidden></div>`;
    const svg = $("svg", el), cross = $(".cross", el), dot = $(".dot", el), tip = $(".tooltip", el);
    const move = (evt) => {
      const r = svg.getBoundingClientRect();
      const vx = ((evt.clientX - r.left) / r.width) * W;
      let best = points[0];
      for (const p of points) if (Math.abs(sx(p[0]) - vx) < Math.abs(sx(best[0]) - vx)) best = p;
      const px = sx(best[0]), py = sy(best[1]);
      cross.setAttribute("x1", px); cross.setAttribute("x2", px); cross.setAttribute("visibility", "visible");
      dot.setAttribute("cx", px); dot.setAttribute("cy", py); dot.setAttribute("visibility", "visible");
      tip.hidden = false;
      tip.style.left = `${(px / W) * r.width}px`; tip.style.top = `${(py / H) * r.height}px`;
      tip.textContent = `step ${best[0]} · loss ${best[1].toFixed(3)}`;
    };
    const hit = $(".hit", el);
    hit.addEventListener("pointermove", move);
    hit.addEventListener("pointerleave", () => { tip.hidden = true; cross.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); });
  }

  // ---------------------------------------------------------------- theme
  const root = document.documentElement;
  const isDark = () => root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  const label = () => { $("#theme-toggle").textContent = isDark() ? "Lights on" : "Lights off"; };
  $("#theme-toggle").addEventListener("click", () => {
    root.dataset.theme = isDark() ? "light" : "dark";
    try { localStorage.setItem("plateful-theme", root.dataset.theme); } catch {}
    label();
  });
  label();

  route();
  loadStatus();
})();
