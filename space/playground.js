// The tinytowns playground. Type letters, see the model's five likeliest next symbols, tap one to add it, or let
// the model finish the name; ending a name checks it against the real list. The model runs in the visitor's browser
// (tinytowns.js); the weights come from Hugging Face, the list from the model's GitHub repo through jsDelivr.
// The same playground runs at quirkyagents.com/wizard/projects/tinytowns/.
(() => {
  "use strict";
  const el = (tag, cls, text) => Object.assign(document.createElement(tag), cls ? { className: cls } : {}, text != null ? { textContent: text } : {});
  const pct = (p) => p >= 0.995 ? "99%+" : p < 0.005 ? "<1%" : `${Math.round(p * 100)}%`;
  const MAX = 20;   // the model's context: one '.' plus at most 20 letters

  // Weights, kept on the device. Hugging Face's file list (1.3 KB) gives each weights file's SHA-256. A file is
  // downloaded only when that fingerprint is new to this device, checked against the bytes, and saved in the
  // browser's Cache Storage under it, replacing the older copy. Editing only the model card changes no fingerprint,
  // so nothing is downloaded again. If Hugging Face can't be reached (offline, or down), the saved copy is used.
  // (Hugging Face's own links can't be cached by the browser: each download is a new one-time signed address.)
  const STORE = "tinytowns-weights";
  const TREE = "https://huggingface.co/api/models/dave8172/tinytowns/tree/main";
  let fingerprints = null;   // one look at the file list per page
  const keyFor = (file, sha) => new URL(`/tinytowns-weights/${file}/${sha}`, location.origin).href;
  const hex = (buf) => Array.from(new Uint8Array(buf), (b) => b.toString(16).padStart(2, "0")).join("");

  async function weights(id) {
    const file = `${id}.safetensors`, mine = (r) => r.url.includes(`/tinytowns-weights/${file}/`);
    let store = null;
    try { store = await caches.open(STORE); } catch { /* no Cache Storage (some private windows): download every visit */ }
    fingerprints ??= fetch(TREE, { cache: "no-store", signal: AbortSignal.timeout(5000) })
      .then((r) => r.ok ? r.json() : Promise.reject(new Error(r.status)))
      .then((list) => Object.fromEntries(list.filter((f) => f.lfs).map((f) => [f.path, f.lfs.oid])))
      .catch(() => null);
    const sha = (await fingerprints)?.[file];
    if (!sha) {   // can't ask Hugging Face: use the copy on this device, if there is one
      const saved = store && (await store.keys()).find(mine);
      if (saved) return (await store.match(saved)).arrayBuffer();
      throw Object.assign(new Error("offline, not saved"), { offline: true });
    }
    const hit = store && await store.match(keyFor(file, sha));
    if (hit) return hit.arrayBuffer();
    const res = await fetch(TinyTowns.url(id));
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const buffer = await res.arrayBuffer();
    if (hex(await crypto.subtle.digest("SHA-256", buffer)) !== sha) {
      fingerprints = null;   // the file changed while we looked: ask again next time
      throw new Error("download does not match Hugging Face's fingerprint");
    }
    if (store) {
      for (const old of (await store.keys()).filter(mine)) await store.delete(old);
      await store.put(keyFor(file, sha), new Response(buffer)).catch(() => {});   // storage full: works, just not saved
    }
    return buffer;
  }

  const start = () => {
    if (!self.TinyTowns) return;   // tinytowns.js loads first
    for (const fig of document.querySelectorAll("figure.widget[data-widget='tinytowns']:not([data-ready])")) mount(fig);
  };

  function mount(fig) {
    fig.dataset.ready = "";
    const d = JSON.parse(fig.querySelector("script[type='application/json']").textContent);
    const A = TinyTowns.ALPHABET;
    const ui = el("div", "tt");
    const cache = new Map();   // size id -> model promise, so switching back in this visit costs nothing
    let size = d.start, model = null, temperature = 1, text = "", ended = false, run = 0, cities = null;

    // --- model chips -----------------------------------------------------------------------------
    const chips = el("div", "tt-models");
    chips.setAttribute("role", "group");
    chips.setAttribute("aria-label", "Model");
    const chipFor = new Map(d.models.map((m) => {
      const b = el("button", "tt-model");
      b.type = "button";
      b.append(el("strong", null, m.label), el("small", null, m.short));
      b.addEventListener("click", () => use(m.id));
      chips.append(b);
      return [m.id, b];
    }));

    // --- temperature -------------------------------------------------------------------------------
    const temp = el("label", "tt-temp");
    const tempOut = el("output", "tt-temp-v", "1.0");
    const slider = Object.assign(document.createElement("input"), { type: "range", min: "0.1", max: "2", step: "0.1", value: "1" });
    slider.setAttribute("aria-label", "Temperature");
    slider.addEventListener("input", () => { temperature = Number(slider.value); tempOut.textContent = temperature.toFixed(1); draw(); });
    temp.append(el("span", "tt-temp-k", "Temperature"), tempOut, el("span", "tt-temp-end safe", "safe"), slider, el("span", "tt-temp-end wild", "wild"));

    // --- the type bar ------------------------------------------------------------------------------
    const bar = el("div", "tt-bar");
    const input = Object.assign(document.createElement("input"), {
      type: "text", placeholder: "type a town…", autocomplete: "off", autocapitalize: "none", spellcheck: false, enterKeyHint: "done",
    });
    input.setAttribute("aria-label", "Letters so far. Type a full stop to end the name.");
    input.maxLength = MAX + 1;
    const clear = el("button", "tt-clear", "×");
    clear.type = "button";
    clear.setAttribute("aria-label", "Clear");
    clear.addEventListener("click", () => { set(""); input.focus(); });
    bar.append(input, clear);

    // Only a-z and space are letters tinytowns knows (its data turned hyphens into spaces and dropped accents and
    // apostrophes, so the same happens here); a full stop (or Enter) ends the name.
    input.addEventListener("input", () => {
      const raw = input.value.toLowerCase();
      const stop = raw.includes(".");
      set(raw.split(".")[0].normalize("NFD").replace(/-/g, " ").replace(/[^a-z ]/g, "").replace(/^ +/, "").replace(/ {2,}/g, " ").slice(0, MAX), stop);
    });
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); finishTyped(); } });

    // --- the five bars -----------------------------------------------------------------------------
    const hint = el("p", "tt-hint");
    const list = el("ol", "tt-next");
    const rows = Array.from({ length: 5 }, () => {
      const li = el("li"), b = el("button", "tt-opt"), sym = el("span", "tt-sym"), track = el("span", "tt-track"), fill = el("span"), num = el("span", "tt-p");
      b.type = "button";
      track.append(fill);
      b.append(sym, track, num);
      li.append(b);
      list.append(li);
      return { b, sym, fill, num };
    });

    // --- finish it, and the verdict ----------------------------------------------------------------
    const actions = el("div", "tt-actions");
    const finish = el("button", "tt-finish", "Finish it for me");
    finish.type = "button";
    finish.addEventListener("click", () => autoFinish());
    actions.append(finish);
    const verdict = el("div", "tt-verdict");
    verdict.setAttribute("aria-live", "polite");
    const status = el("p", "tt-status");
    status.setAttribute("aria-live", "polite");

    // the two symbols that are not letters, explained once, under the bars
    const legend = el("dl", "tt-legend");
    legend.append(el("dt", null, "end of name"), el("dd", null, "The name is complete. To end a name, tap this bar, type a full stop (.), or press Enter."),
      el("dt", null, "space"), el("dd", null, "A gap between two words, as in “san jose”."));

    ui.append(chips, temp, bar, hint, list, legend, actions, verdict, status);
    fig.querySelector(".widget-tables").hidden = true;
    fig.append(ui);

    function set(t, stop = false) {
      run++;   // a newer action cancels an older "finish it" mid-way
      text = t;
      ended = false;
      if (input.value !== t) input.value = t;
      verdict.replaceChildren();
      if (stop) finishTyped(); else draw();
    }

    function draw() {
      for (const [id, b] of chipFor) b.setAttribute("aria-pressed", String(id === size));
      finish.disabled = !model || ended;
      if (!model) { list.hidden = legend.hidden = true; hint.textContent = ""; return; }
      list.hidden = legend.hidden = ended;
      if (ended) { hint.textContent = ""; return; }
      if (text.length >= MAX) { hint.textContent = "That's 20 letters, the most it can read. Tap end of name."; }
      else hint.textContent = text ? `After “${text}”, the model would pick…` : "To start a name, the model would pick…";
      const p = model.next(text, temperature);
      const top = p.map((v, i) => [v, i]).filter(([, i]) => text.length < MAX || i === 0)
        .filter(([, i]) => i !== 1 || (text && !text.endsWith(" ")))   // never a space first or twice
        .sort((a, b) => b[0] - a[0]).slice(0, 5);
      rows.forEach((r, k) => {
        const item = top[k];
        r.b.parentElement.hidden = !item;
        if (!item) return;
        const [v, i] = item;
        const label = i === 0 ? "end of name" : i === 1 ? "space" : A[i];
        r.sym.textContent = label;
        r.sym.className = `tt-sym${i < 2 ? " word" : ""}`;
        r.fill.style.width = `${Math.max(v * 100, 0.5)}%`;
        r.num.textContent = pct(v);
        r.b.setAttribute("aria-label", `${label}, ${pct(v)}`);
        r.b.onclick = () => { if (i === 0) finishTyped(); else set(text + A[i]); };
      });
    }

    // The model writes the rest, one letter at a time, at the chosen temperature.
    async function autoFinish() {
      if (!model || ended) return;
      const mine = ++run;
      finish.disabled = true;
      while (text.length < MAX) {
        const p = model.next(text, temperature);
        let r = Math.random(), i = 0;
        while (i < p.length - 1 && r >= p[i]) r -= p[i++];
        if (text === "" || text.endsWith(" ")) while (i === 0 || i === 1) {   // a name can't start or end on a space
          r = Math.random(); i = 0;
          while (i < p.length - 1 && r >= p[i]) r -= p[i++];
        }
        if (i === 0) break;
        text += A[i];
        input.value = text;
        draw();
        await new Promise((ok) => setTimeout(ok, matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 110));
        if (mine !== run) return;
      }
      text = text.trim();
      input.value = text;
      finishTyped(mine);
    }

    // The name is done: real (in the list of 142,706 places) or invented.
    async function finishTyped(mine = ++run) {
      text = text.trim();
      if (!text) { draw(); return; }
      ended = true;
      input.value = text;
      draw();
      verdict.replaceChildren(el("p", "tt-checking", "Checking the map…"));
      const places = await loadCities().catch(() => null);
      if (mine !== run) return;
      if (!places) { verdict.replaceChildren(el("p", "tt-checking", "Couldn't load the list of real places. Try again in a moment.")); return; }
      const real = places.has(text);
      const badge = el("p", `tt-badge ${real ? "real" : "invented"}`, real ? "Real" : "Invented");
      const name = el("p", "tt-name", text);
      const line = el("p", "tt-line", real ? d.real : d.invented);
      const again = el("button", "tt-again", "Another one");
      again.type = "button";
      again.addEventListener("click", () => { set(""); autoFinish(); });
      verdict.replaceChildren(badge, name, line, again);
    }

    function loadCities() {
      cities ??= fetch(d.cities).then((r) => {
        if (!r.ok) throw new Error(r.status);
        return r.text();
      }).then((t) => new Set(t.split("\n").map((s) => s.trim()).filter(Boolean)))
        .catch((e) => { cities = null; throw e; });
      return cities;
    }

    async function use(id) {
      size = id;
      const m = d.models.find((x) => x.id === id);
      model = null;
      draw();
      status.textContent = `Summoning ${m.label} (${m.file})…`;
      if (!cache.has(id)) cache.set(id, weights(id).then((buffer) => TinyTowns.load(buffer)).catch((e) => { cache.delete(id); throw e; }));
      try {
        const loaded = await cache.get(id);
        if (size !== id) return;
        model = loaded;
        status.textContent = "";
        draw();
      } catch (e) {
        if (size === id) status.textContent = e.offline ? `You're offline, and ${m.label} isn't saved on this device yet. Try again when you're online.`
          : `Couldn't fetch ${m.label} from Hugging Face. Tap it to try again.`;
      }
    }

    use(size);
    // fetch the list while the visitor types, so the first verdict is quick
    const idle = self.requestIdleCallback ? (f) => requestIdleCallback(f, { timeout: 4000 }) : (f) => setTimeout(f, 2000);
    idle(() => loadCities().catch(() => {}));
  }

  start();
})();
