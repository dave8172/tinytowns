// tinytowns.js: the tinytowns transformer in plain JavaScript, for a browser or Node. No libraries.
//
// It does what model.py's forward pass and sample() do, without training: read a .safetensors
// weights file, turn letters into notes, run the layers, and give a probability for each of the
// 28 possible next symbols. Weights are only ever read; everything happens on your own machine.
//
//   const model = await TinyTowns.load(TinyTowns.url("tinytowns-small"));
//   model.next("ban")            // 28 probabilities: what comes after ".ban"
//   model.sample("ban", { temperature: 0.8 })   // "banora", say
//
// Shapes in comments: T = positions (at most 21), C = note size (16 / 64 / 128), V = 28 symbols.
(() => {
  "use strict";

  const ALPHABET = ". abcdefghijklmnopqrstuvwxyz";  // index 0 '.' = start of name going in, end of name coming out
  const CONTEXT = 21;                                 // one '.' plus at most 20 letters
  const REPO = "https://huggingface.co/dave8172/tinytowns/resolve/main/";

  // --- Reading a .safetensors file -------------------------------------------------------------
  // 8 bytes: header length n (little-endian). Then n bytes of JSON: each grid's name, number type,
  // shape and byte range. Then all the numbers, back to back, as 4-byte floats.
  function parse(buffer) {
    const view = new DataView(buffer);
    const n = Number(view.getBigUint64(0, true));
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, n)));
    const meta = header.__metadata__ || {};
    const config = JSON.parse(meta.config);
    const grids = {};
    for (const [name, t] of Object.entries(header)) {
      if (name === "__metadata__") continue;
      if (t.dtype !== "F32") throw new Error(`${name}: expected F32, got ${t.dtype}`);
      const [start, end] = t.data_offsets;
      // A copy, so the Float32Array starts on a 4-byte boundary whatever the header length is.
      grids[name] = new Float32Array(buffer.slice(8 + n + start, 8 + n + end));
    }
    return { config, grids, best: meta.best ? JSON.parse(meta.best) : null };
  }

  // --- The small pieces ------------------------------------------------------------------------
  // linear: out = grid · x + bias. PyTorch stores the grid as [out, in], one row per output number.
  function linear(x, w, b, nIn, nOut) {
    const out = new Float64Array(nOut);
    for (let o = 0; o < nOut; o++) {
      let s = b[o];
      const row = o * nIn;
      for (let i = 0; i < nIn; i++) s += w[row + i] * x[i];
      out[o] = s;
    }
    return out;
  }

  // LayerNorm: shift the note to average 0 and scale it to spread 1, then the learned scale and shift.
  function layerNorm(x, w, b) {
    const C = x.length;
    let mean = 0;
    for (let i = 0; i < C; i++) mean += x[i];
    mean /= C;
    let variance = 0;
    for (let i = 0; i < C; i++) variance += (x[i] - mean) ** 2;
    const scale = 1 / Math.sqrt(variance / C + 1e-5);  // 1e-5 = PyTorch's default, keeps it from dividing by 0
    const out = new Float64Array(C);
    for (let i = 0; i < C; i++) out[i] = (x[i] - mean) * scale * w[i] + b[i];
    return out;
  }

  // GELU, the bend in the knowledge step: x times the chance a bell-curve number is below x.
  // JavaScript has no erf(), so this one is a standard approximation, good to about 1e-7.
  function erf(x) {
    const s = Math.sign(x), a = Math.abs(x), t = 1 / (1 + 0.5 * a);
    const y = t * Math.exp(-a * a - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
      t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return s * (1 - y);
  }
  const gelu = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));

  function softmax(scores, temperature = 1) {
    let max = -Infinity;
    for (const s of scores) max = Math.max(max, s / temperature);
    const p = Array.from(scores, (s) => Math.exp(s / temperature - max));  // minus max: same result, no overflow
    const sum = p.reduce((a, b) => a + b, 0);
    return p.map((v) => v / sum);
  }

  // --- The model ---------------------------------------------------------------------------------
  function build({ config, grids, best }) {
    const { note: C, layers: L, heads: H, mlp: M, vocab: V } = config;
    const hs = C / H;  // numbers per head
    const g = (name) => {
      if (!grids[name]) throw new Error(`weights file has no ${name}`);
      return grids[name];
    };

    // ids: symbol numbers, starting with 0 ('.'). Returns the scores for every position: [T][V].
    // Each position sees only itself and earlier ones, so the scores at position t are the
    // prediction for symbol t+1, exactly as in training.
    function forward(ids) {
      const T = ids.length;
      if (T > CONTEXT) throw new Error(`at most ${CONTEXT} symbols`);
      // 1) Lookup table + position table: each symbol becomes a note.  notes: [T][C]
      const lookup = g("lookup.weight"), position = g("position.weight");
      let notes = ids.map((id, t) => {
        const n = new Float64Array(C);
        for (let i = 0; i < C; i++) n[i] = lookup[id * C + i] + position[t * C + i];
        return n;
      });

      for (let l = 0; l < L; l++) {
        const p = `layers.${l}.`;
        // 2) Attention. q, k, v for every position: [T][C], each split into H heads of hs numbers.
        const normed = notes.map((n) => layerNorm(n, g(p + "norm1.weight"), g(p + "norm1.bias")));
        const proj = (k) => normed.map((n) => linear(n, g(`${p}attention.${k}.weight`), g(`${p}attention.${k}.bias`), C, C));
        const q = proj("q"), k = proj("k"), v = proj("v");
        const mix = notes.map(() => new Float64Array(C));
        for (let h = 0; h < H; h++) {
          const o = h * hs;
          for (let i = 0; i < T; i++) {
            // score: my q against the k of every position up to me (the mask: never a later one)
            const score = new Float64Array(i + 1);
            for (let j = 0; j <= i; j++) {
              let s = 0;
              for (let d = 0; d < hs; d++) s += q[i][o + d] * k[j][o + d];
              score[j] = s / Math.sqrt(hs);
            }
            // shares: softmax of the scores. mix: the shares times each position's v.
            const shares = softmax(score);
            for (let j = 0; j <= i; j++) for (let d = 0; d < hs; d++) mix[i][o + d] += shares[j] * v[j][o + d];
          }
        }
        // the out grid, added onto the note (the residual connection)
        notes = notes.map((n, t) => {
          const upd = linear(mix[t], g(p + "attention.out.weight"), g(p + "attention.out.bias"), C, C);
          return n.map((x, i) => x + upd[i]);
        });

        // 3) Knowledge step, each note on its own: C -> M, bend, M -> C, added onto the note.
        notes = notes.map((n) => {
          const up = linear(layerNorm(n, g(p + "norm2.weight"), g(p + "norm2.bias")), g(p + "knowledge.up.weight"), g(p + "knowledge.up.bias"), C, M);
          const down = linear(up.map(gelu), g(p + "knowledge.down.weight"), g(p + "knowledge.down.bias"), M, C);
          return n.map((x, i) => x + down[i]);
        });
      }

      // 4) Last tidy-up, then the output table: note -> 28 scores.  [T][V]
      return notes.map((n) => linear(layerNorm(n, g("norm.weight"), g("norm.bias")), g("output.weight"), g("output.bias"), C, V));
    }

    const encode = (prefix) => [0, ...Array.from(prefix, (ch) => {
      const i = ALPHABET.indexOf(ch);
      if (i < 1) throw new Error(`"${ch}" is not a letter tinytowns knows (a-z and space)`);
      return i;
    })];

    return {
      config, best, forward,
      weights: Object.values(grids).reduce((s, a) => s + a.length, 0),
      // Probabilities for the symbol after '.' + prefix: an array of 28, in ALPHABET order.
      next(prefix, temperature = 1) {
        const scores = forward(encode(prefix));
        return softmax(scores[scores.length - 1], temperature);
      },
      // Keep picking letters until the model picks '.' (end of name) or the name is 20 letters long.
      // random: any function giving numbers in [0, 1), e.g. a seeded one for repeatable names.
      sample(prefix = "", { temperature = 1, random = Math.random } = {}) {
        let name = prefix;
        while (name.length < CONTEXT - 1) {
          const p = this.next(name, temperature);
          let r = random(), pick = 0;
          while (pick < p.length - 1 && r >= p[pick]) r -= p[pick++];
          if (pick === 0) break;
          name += ALPHABET[pick];
        }
        return name;
      },
    };
  }

  async function load(source) {
    let buffer = source;
    if (typeof source === "string") {
      const res = await fetch(source);
      if (!res.ok) throw new Error(`${source}: HTTP ${res.status}`);
      buffer = await res.arrayBuffer();
    }
    return build(parse(buffer));
  }

  const api = { ALPHABET, CONTEXT, load, parse, build, url: (size) => `${REPO}${size}.safetensors` };
  // A plain <script> tag gives you the global TinyTowns; in Node, require() it or import it for the global.
  globalThis.TinyTowns = api;
  if (typeof module === "object" && module.exports) module.exports = api;
})();
