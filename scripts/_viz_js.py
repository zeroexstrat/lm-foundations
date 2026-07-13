# ═══════════════════════════════════════════════════════════════════════
#  DATA-DRIVEN VISUALIZATIONS for the interactive textbook.
#
#  Every interactive runs on real traces from scripts/viz_data.py
#  (output/viz_data.json): the actual trained toy model's logits, attention
#  weights, loss curve, embedding table, and weight tensors. If the JSON is
#  missing, each panel shows a one-line note explaining how to generate it.
#
#  VIZ_CONTAINERS: HTML blocks injected per chapter by build_html().
#  VIZ_JS: plain-string JS appended after the main (f-string) script,
#          so no brace-doubling is needed here.
# ═══════════════════════════════════════════════════════════════════════

def _container(icon, title, hint, body):
    return (f'<div class="viz-container"><div class="viz-header">'
            f'<span class="viz-icon">{icon}</span><span class="viz-title">{title}</span>'
            f'<span class="viz-hint">{hint}</span></div>'
            f'<div class="viz-body">{body}</div></div>')


VIZ_CONTAINERS = {
    "softmax": _container(
        "🌡️", "Interactive: Temperature On Real Logits",
        "the model's actual next-token logits after ‘The model ’",
        '<canvas id="softmax-canvas" width="720" height="330" role="img" aria-label="Bar chart of real next-token probabilities at the selected temperature"></canvas>'
        '<div class="viz-controls"><label for="temp-slider">Temperature τ <span id="temp-val">1.0</span></label>'
        '<input type="range" id="temp-slider" min="0.1" max="5" step="0.1" value="1">'
        '<span class="viz-readout" id="softmax-readout"></span></div>'),

    "attention": _container(
        "🔍", "Interactive: Trained Attention, Layer By Head",
        "real weights on the probe sentence — try removing the mask or the √D",
        '<div class="attention-panels"><div class="attn-panel">'
        '<canvas id="attn-weights-canvas" width="460" height="460" role="img" aria-label="Attention-weight matrix for the selected layer and head" aria-describedby="attn-summary"></canvas></div>'
        '<div class="attn-side"><div class="viz-controls viz-stack">'
        '<label>Layer <select id="attn-layer"></select></label>'
        '<label>Head <select id="attn-head"></select></label>'
        '<label><input type="checkbox" id="attn-mask" checked> causal mask M</label>'
        '<label><input type="checkbox" id="attn-scale" checked> scale by 1/√D</label>'
        '</div><div class="attn-info" id="attn-info"><span class="attn-placeholder">Hover a cell to inspect w<sub>t,s</sub></span></div>'
        '<p class="sr-only" id="attn-summary" aria-live="polite"></p></div></div>'),

    "transformer": _container(
        "🏗️", "Interactive: The Residual Stream, Measured",
        "real per-stage activation norms on the probe — click a stage",
        '<canvas id="transformer-canvas" width="720" height="440" role="img" aria-label="Residual-stream activation norms by transformer stage"></canvas>'
        '<div class="viz-controls"><label for="transformer-stage">Stage</label>'
        '<select id="transformer-stage"></select><span class="viz-readout" id="transformer-readout" aria-live="polite"></span></div>'),

    "training": _container(
        "📉", "Interactive: A Real Training Run",
        "scrub 400 real AdamW steps — watch p(correct token) rise",
        '<canvas id="training-canvas" width="720" height="380" role="img" aria-label="Training and validation loss with next-token probability by checkpoint"></canvas>'
        '<div class="viz-controls"><label for="step-slider">Checkpoint <span id="step-val">0</span></label>'
        '<input type="range" id="step-slider" min="0" max="20" step="1" value="0">'
        '<span class="viz-readout" id="training-readout"></span></div>'),

    "generation": _container(
        "🎲", "Interactive: Decoding From Real Logits",
        "step the recorded sample; reshape its distribution with τ and top-k",
        '<canvas id="generation-canvas" width="720" height="330" role="img" aria-label="Recorded next-token distribution under temperature and top-k decoding"></canvas>'
        '<div class="viz-controls"><button id="gen-next-btn">Next token</button>'
        '<button id="gen-reset-btn">Reset</button>'
        '<label for="gen-temp">τ <span id="gen-temp-val">1.0</span></label>'
        '<input type="range" id="gen-temp" min="0.1" max="3" step="0.1" value="1" style="width:110px">'
        '<label>top-k <select id="gen-topk"><option>all</option><option>10</option><option>5</option><option selected>3</option><option>1</option></select></label>'
        '<span class="gen-output" id="gen-output"></span></div>'),

    "embedding": _container(
        "🗺️", "Interactive: The Learned Embedding Geometry",
        "PCA of the trained token-embedding table — hover the characters",
        '<canvas id="embedding-canvas" width="720" height="430" role="img" aria-label="Two-dimensional PCA projection of the learned token embeddings"></canvas>'
        '<div class="viz-controls"><label for="embedding-token">Token</label>'
        '<select id="embedding-token"></select><span class="viz-readout" id="embedding-readout" aria-live="polite"></span></div>'),

    "quant": _container(
        "🎚️", "Interactive: Quantization Grid On Real Weights",
        "96 real weights from blocks[0].attn.qkv — sweep the bit width",
        '<canvas id="quant-canvas" width="720" height="330" role="img" aria-label="Original and quantized model weights at the selected bit width"></canvas>'
        '<div class="viz-controls"><label for="quant-bits">Bits <span id="quant-bits-val">4</span></label>'
        '<input type="range" id="quant-bits" min="2" max="8" step="1" value="4">'
        '<label><input type="checkbox" id="quant-sym"> symmetric</label>'
        '<span class="viz-readout" id="quant-readout"></span></div>'),

    "kv": _container(
        "📦", "Interactive: KV-Cache Memory — MHA → GQA → MQA",
        "bytes = 2·L·G·D·T·B·(bytes/value): slide G between H and 1",
        '<canvas id="kv-canvas" width="720" height="300" role="img" aria-label="KV-cache memory use for the selected model dimensions and grouped-query setting"></canvas>'
        '<div class="viz-controls viz-wrap">'
        '<label for="kv-l">L <span id="kv-l-val">32</span></label><input type="range" id="kv-l" min="1" max="80" value="32" style="width:80px">'
        '<label for="kv-h">H <span id="kv-h-val">32</span></label><input type="range" id="kv-h" min="1" max="64" value="32" style="width:80px">'
        '<label for="kv-d">D <span id="kv-d-val">128</span></label><input type="range" id="kv-d" min="32" max="256" step="32" value="128" style="width:80px">'
        '<label for="kv-t">T = 2^<span id="kv-t-val">13</span></label><input type="range" id="kv-t" min="7" max="17" value="13" style="width:80px">'
        '<label for="kv-g">G <span id="kv-g-val">8</span></label><input type="range" id="kv-g" min="1" max="32" value="8" style="width:80px">'
        '</div>'),
}


VIZ_JS = r"""
// ═════════════════════════════════════════════════════════════════════
//  DATA-DRIVEN VISUALIZATIONS (real traces from scripts/viz_data.py)
// ═════════════════════════════════════════════════════════════════════
const VIZ = window.VIZ_DATA || null;
const VC = {
  bg: '#161b22', panel: '#1c2333', grid: '#21262d', text: '#c9d1d9',
  dim: '#8b949e', bright: '#f0f6fc', accent: '#f0883e', blue: '#58a6ff',
  green: '#3fb950', purple: '#d2a8ff', red: '#f97583', gold: '#e3b341',
};
function vizTok(t) { return t === ' ' ? '␣' : t === '\n' ? '⏎' : t; }
function vizMissing(canvasId) {
  const c = document.getElementById(canvasId);
  if (!c) return true;
  if (VIZ) return false;
  const ctx = c.getContext('2d');
  ctx.fillStyle = VC.dim; ctx.font = '13px -apple-system, sans-serif';
  ctx.fillText('Live data not generated yet — run:  python scripts/viz_data.py  then rebuild.', 30, 60);
  return true;
}
function softmaxArr(z) {
  const m = Math.max(...z.filter(v => v !== -Infinity));
  const e = z.map(v => v === -Infinity ? 0 : Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / (s || 1));
}
function entropyArr(p) { return -p.reduce((a, v) => a + (v > 0 ? v * Math.log(v) : 0), 0); }
function heatColor(v) {   // 0..1 → dark → blue → orange
  const stops = [[13,17,23],[24,50,90],[38,109,196],[240,136,62]];
  const x = Math.max(0, Math.min(1, v)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x)), f = x - i;
  const c = stops[i].map((a, j) => Math.round(a + f * (stops[i + 1][j] - a)));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

// ═══ 1. Softmax temperature on real logits (Theorem 1.6 / Prop 7.1) ═══
function initSoftmaxViz() {
  if (vizMissing('softmax-canvas')) return;
  const canvas = document.getElementById('softmax-canvas');
  const ctx = canvas.getContext('2d');
  const slider = document.getElementById('temp-slider');
  const valSpan = document.getElementById('temp-val');
  const readout = document.getElementById('softmax-readout');
  const W = canvas.width, H = canvas.height;

  const step0 = VIZ.generation.steps[0];
  const itos = VIZ.itos;
  const idx = step0.logits.map((_, i) => i).sort((a, b) => step0.logits[b] - step0.logits[a]).slice(0, 12);
  const labels = idx.map(i => vizTok(itos[String(i)]));
  const logits = idx.map(i => step0.logits[i]);
  const V = step0.logits.length;

  function draw(T) {
    ctx.clearRect(0, 0, W, H);
    const full = softmaxArr(step0.logits.map(v => v / T));
    const probs = idx.map(i => full[i]);
    const H_p = entropyArr(full);
    ctx.fillStyle = VC.dim; ctx.font = '12px -apple-system, sans-serif';
    ctx.fillText(`p( next token | "${VIZ.generation.prompt}" ) — trained model, top ${idx.length} of V=${V} shown`, 30, 24);
    const barW = 44, gap = 12, startX = 46, baseY = H - 56, maxH = H - 116;
    for (let i = 0; i < probs.length; i++) {
      const x = startX + i * (barW + gap), h = probs[i] * maxH / Math.max(...probs, 0.001) * 0.96;
      ctx.fillStyle = i === 0 ? VC.accent : VC.blue;
      ctx.fillRect(x, baseY - h, barW, h);
      ctx.fillStyle = VC.text; ctx.font = 'bold 15px monospace'; ctx.textAlign = 'center';
      ctx.fillText(labels[i], x + barW / 2, baseY + 20);
      ctx.fillStyle = VC.dim; ctx.font = '10px monospace';
      ctx.fillText(probs[i].toFixed(3), x + barW / 2, baseY - h - 6);
      ctx.textAlign = 'start';
    }
    readout.textContent = `H(p) = ${H_p.toFixed(3)} nats   e^H = ${Math.exp(H_p).toFixed(2)} effective tokens   (uniform: H = ln ${V} = ${Math.log(V).toFixed(3)})`;
  }
  slider.addEventListener('input', () => { const T = parseFloat(slider.value); valSpan.textContent = T.toFixed(1); draw(T); });
  draw(1.0);
}

// ═══ 2. Trained attention with live mask / √D toggles (Def 4.2) ═══
function initAttentionViz() {
  if (vizMissing('attn-weights-canvas')) return;
  const wc = document.getElementById('attn-weights-canvas');
  const wCtx = wc.getContext('2d');
  const info = document.getElementById('attn-info');
  const summary = document.getElementById('attn-summary');
  const selL = document.getElementById('attn-layer');
  const selH = document.getElementById('attn-head');
  const chkM = document.getElementById('attn-mask');
  const chkS = document.getElementById('attn-scale');

  const A = VIZ.attention;
  const T = A.tokens.length, D = A.head_dim;
  const L = A.raw_scores.length, NH = A.raw_scores[0].length;
  for (let l = 0; l < L; l++) selL.add(new Option(`${l + 1}`, l));
  for (let h = 0; h < NH; h++) selH.add(new Option(`${h + 1}`, h));

  const M = { left: 58, top: 46 };
  const cell = Math.floor((wc.width - M.left - 10) / T);
  let cur = null;

  function computeW() {
    const raw = A.raw_scores[selL.value][selH.value];
    const scale = chkS.checked ? 1 / Math.sqrt(D) : 1;
    return raw.map((row, t) =>
      softmaxArr(row.map((v, s) => (chkM.checked && s > t) ? -Infinity : v * scale)));
  }
  function draw() {
    cur = computeW();
    wCtx.clearRect(0, 0, wc.width, wc.height);
    wCtx.fillStyle = VC.dim; wCtx.font = '11px -apple-system, sans-serif';
    wCtx.fillText(`W — layer ${+selL.value + 1}, head ${+selH.value + 1} on "${VIZ.probe.text.slice(0, T)}"`, M.left, 18);
    wCtx.font = 'bold 12px monospace'; wCtx.textAlign = 'center';
    for (let s = 0; s < T; s++) {
      wCtx.fillStyle = VC.dim;
      wCtx.fillText(vizTok(A.tokens[s]), M.left + s * cell + cell / 2, M.top - 8);
    }
    for (let t = 0; t < T; t++) {
      wCtx.fillStyle = VC.dim;
      wCtx.fillText(vizTok(A.tokens[t]), M.left - 16, M.top + t * cell + cell / 2 + 4);
      for (let s = 0; s < T; s++) {
        wCtx.fillStyle = heatColor(cur[t][s]);
        wCtx.fillRect(M.left + s * cell, M.top + t * cell, cell - 1, cell - 1);
      }
    }
    wCtx.textAlign = 'start';
    if (!chkM.checked) {
      wCtx.fillStyle = VC.red; wCtx.font = '11px -apple-system, sans-serif';
      wCtx.fillText('⚠ mask off: upper triangle is nonzero — every row now reads the future', M.left, M.top + T * cell + 18);
    } else if (!chkS.checked) {
      wCtx.fillStyle = VC.gold; wCtx.font = '11px -apple-system, sans-serif';
      wCtx.fillText('⚠ unscaled scores: rows saturate toward argmax (lower temperature)', M.left, M.top + T * cell + 18);
    }
    const strongest = cur.map((row, t) => {
      const s = row.indexOf(Math.max(...row));
      return `${vizTok(A.tokens[t])}→${vizTok(A.tokens[s])} ${row[s].toFixed(3)}`;
    });
    summary.textContent = `Layer ${+selL.value + 1}, head ${+selH.value + 1}. Strongest attention by query token: ${strongest.join('; ')}.`;
  }
  wc.addEventListener('mousemove', (e) => {
    const r = wc.getBoundingClientRect();
    const sx = wc.width / r.width, sy = wc.height / r.height;
    const s = Math.floor(((e.clientX - r.left) * sx - M.left) / cell);
    const t = Math.floor(((e.clientY - r.top) * sy - M.top) / cell);
    if (t >= 0 && t < T && s >= 0 && s < T && cur) {
      const raw = VIZ.attention.raw_scores[selL.value][selH.value][t][s];
      info.innerHTML = `w<sub>${t + 1},${s + 1}</sub> = <b>${cur[t][s].toFixed(3)}</b><br>` +
        `‘${vizTok(A.tokens[t])}’ (pos ${t + 1}) reading ‘${vizTok(A.tokens[s])}’ (pos ${s + 1})<br>` +
        `q·k = ${raw.toFixed(3)}   q·k/√${D} = ${(raw / Math.sqrt(D)).toFixed(3)}` +
        (s > t ? '<br><span style="color:#f97583">future position — masked in training</span>' : '');
    }
  });
  [selL, selH, chkM, chkS].forEach(el => el.addEventListener('change', draw));
  draw();
}

// ═══ 3. Residual stream, measured (Ch 5 flow picture) ═══
function initTransformerViz() {
  if (vizMissing('transformer-canvas')) return;
  const canvas = document.getElementById('transformer-canvas');
  const ctx = canvas.getContext('2d');
  const stageSelect = document.getElementById('transformer-stage');
  const readout = document.getElementById('transformer-readout');
  const W = canvas.width, H = canvas.height;
  const flow = VIZ.flow;
  const tokens = VIZ.attention.tokens;
  let selected = flow.length - 1;
  flow.forEach((stage, i) => stageSelect.add(new Option(stage.name, i)));

  const colors = [VC.dim, VC.accent, VC.blue, VC.accent, VC.blue, VC.purple];
  const stageColor = (i) => i === 0 ? VC.dim : (i === flow.length - 1 ? VC.purple : (flow[i].name.includes('Attn') ? VC.accent : VC.blue));
  const boxX = 40, boxW = 250, boxH = 40, gapY = (H - 70) / flow.length;

  function draw() {
    ctx.clearRect(0, 0, W, H);
    const maxMean = Math.max(...flow.map(s => s.mean));
    ctx.fillStyle = VC.dim; ctx.font = '11px -apple-system, sans-serif';
    ctx.fillText('stages (click)', boxX, 22);
    ctx.fillText('mean ‖x_t‖ per stage', boxX + boxW + 60, 22);
    ctx.fillText(`per-token ‖x_t‖ at stage: ${flow[selected].name}`, boxX + boxW + 60, H - 118);
    flow.forEach((st, i) => {
      const y = 34 + i * gapY;
      // stage box
      ctx.fillStyle = i === selected ? VC.panel : VC.bg;
      ctx.strokeStyle = i === selected ? stageColor(i) : VC.grid;
      ctx.lineWidth = i === selected ? 2 : 1;
      ctx.fillRect(boxX, y, boxW, boxH); ctx.strokeRect(boxX, y, boxW, boxH);
      ctx.fillStyle = stageColor(i); ctx.font = '12px monospace';
      ctx.fillText(st.name, boxX + 10, y + 24);
      if (i < flow.length - 1) {   // residual arrow
        ctx.strokeStyle = VC.green; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(boxX + boxW / 2, y + boxH); ctx.lineTo(boxX + boxW / 2, y + gapY); ctx.stroke();
      }
      // mean-norm bar
      const bw = (st.mean / maxMean) * 220;
      ctx.fillStyle = stageColor(i);
      ctx.fillRect(boxX + boxW + 60, y + 10, bw, 18);
      ctx.fillStyle = VC.text; ctx.font = '10px monospace';
      ctx.fillText(st.mean.toFixed(2), boxX + boxW + 66 + bw, y + 23);
    });
    // per-token norms of selected stage
    const pt = flow[selected].per_token, maxPt = Math.max(...pt);
    stageSelect.value = String(selected);
    readout.textContent = `${flow[selected].name}: mean norm ${flow[selected].mean.toFixed(3)}; per-token norms ${pt.map(v => v.toFixed(2)).join(', ')}`;
    const px0 = boxX + boxW + 60, pw = 24;
    pt.forEach((v, t) => {
      const h = (v / maxPt) * 66;
      ctx.fillStyle = stageColor(selected);
      ctx.fillRect(px0 + t * pw, H - 40 - h, pw - 3, h);
      ctx.fillStyle = VC.dim; ctx.font = 'bold 11px monospace'; ctx.textAlign = 'center';
      ctx.fillText(vizTok(tokens[t]), px0 + t * pw + pw / 2 - 1, H - 24);
      ctx.textAlign = 'start';
    });
  }
  canvas.addEventListener('click', (e) => {
    const r = canvas.getBoundingClientRect();
    const y = (e.clientY - r.top) * (canvas.height / r.height);
    const i = Math.floor((y - 34) / gapY);
    if (i >= 0 && i < flow.length) { selected = i; draw(); }
  });
  stageSelect.addEventListener('change', () => { selected = Number(stageSelect.value); draw(); });
  draw();
}

// ═══ 4. A real training run (Ch 6) ═══
function initTrainingViz() {
  if (vizMissing('training-canvas')) return;
  const canvas = document.getElementById('training-canvas');
  const ctx = canvas.getContext('2d');
  const slider = document.getElementById('step-slider');
  const valSpan = document.getElementById('step-val');
  const readout = document.getElementById('training-readout');
  const W = canvas.width, H = canvas.height;

  const L = VIZ.loss, P = VIZ.probe;
  slider.max = L.eval_steps.length - 1;
  const maxStep = L.eval_steps[L.eval_steps.length - 1];
  const allLoss = L.step_loss.concat(L.eval_train, L.eval_val);
  const minL = 0, maxL = Math.max(...allLoss) * 1.05;

  const pane = { x: 46, y: 34, w: 380, h: H - 96 };
  const bars = { x: 470, y: 34, w: W - 490, h: H - 96 };
  const sx = (s) => pane.x + pane.w * (s / maxStep);
  const sy = (v) => pane.y + pane.h * (1 - (v - minL) / (maxL - minL));

  function drawCurve(arr, xs, color, lw) {
    ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.beginPath();
    arr.forEach((v, i) => { const X = sx(xs(i)), Y = sy(v); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
    ctx.stroke();
  }
  function draw(ei) {
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = VC.dim; ctx.font = '11px -apple-system, sans-serif';
    ctx.fillText('cross-entropy (nats/token) — real AdamW run', pane.x, 20);
    ctx.fillText(`p( correct next token | prefix ) on "${P.text.slice(0, 12)}…" at step ${L.eval_steps[ei]}`, bars.x, 20);
    // axes
    ctx.strokeStyle = VC.grid; ctx.lineWidth = 1;
    for (let v = 0; v <= Math.floor(maxL); v++) {
      ctx.beginPath(); ctx.moveTo(pane.x, sy(v)); ctx.lineTo(pane.x + pane.w, sy(v)); ctx.stroke();
      ctx.fillStyle = VC.dim; ctx.font = '9px monospace'; ctx.fillText(String(v), pane.x - 14, sy(v) + 3);
    }
    ctx.globalAlpha = 0.35;
    drawCurve(L.step_loss, (i) => i, VC.dim, 1);          // per-batch loss, faint
    ctx.globalAlpha = 1;
    drawCurve(L.eval_train, (i) => L.eval_steps[i], VC.blue, 2);
    drawCurve(L.eval_val, (i) => L.eval_steps[i], VC.accent, 2);
    ctx.fillStyle = VC.blue; ctx.fillText('train', pane.x + pane.w - 60, pane.y + 12);
    ctx.fillStyle = VC.accent; ctx.fillText('val', pane.x + pane.w - 26, pane.y + 12);
    // scrub line
    const X = sx(L.eval_steps[ei]);
    ctx.strokeStyle = VC.green; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(X, pane.y); ctx.lineTo(X, pane.y + pane.h); ctx.stroke();
    // per-position p(correct) bars
    const pc = P.p_correct[ei], T = pc.length, bw = bars.w / T;
    const tokens = P.target_ids.map(i => vizTok(VIZ.itos[String(i)]));
    pc.forEach((p, t) => {
      const h = p * bars.h;
      ctx.fillStyle = VC.green; ctx.globalAlpha = 0.85;
      ctx.fillRect(bars.x + t * bw, bars.y + bars.h - h, bw - 2, h);
      ctx.globalAlpha = 1;
      ctx.fillStyle = VC.dim; ctx.font = 'bold 10px monospace'; ctx.textAlign = 'center';
      ctx.fillText(tokens[t], bars.x + t * bw + bw / 2 - 1, bars.y + bars.h + 14);
      ctx.textAlign = 'start';
    });
    ctx.strokeStyle = VC.grid; ctx.strokeRect(bars.x, bars.y, bars.w, bars.h);
    const tr = L.eval_train[ei], va = L.eval_val[ei];
    readout.textContent = `train ${tr.toFixed(3)}  val ${va.toFixed(3)}  PPL(val) = e^${va.toFixed(2)} = ${Math.exp(va).toFixed(1)}`;
    valSpan.textContent = L.eval_steps[ei];
  }
  slider.addEventListener('input', () => draw(parseInt(slider.value)));
  slider.value = 0; draw(0);
}

// ═══ 5. Decoding from real logits (Ch 7) ═══
function initGenerationViz() {
  if (vizMissing('generation-canvas')) return;
  const canvas = document.getElementById('generation-canvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const nextBtn = document.getElementById('gen-next-btn');
  const resetBtn = document.getElementById('gen-reset-btn');
  const outputSpan = document.getElementById('gen-output');
  const tempSlider = document.getElementById('gen-temp');
  const tempVal = document.getElementById('gen-temp-val');
  const topkSel = document.getElementById('gen-topk');

  const G = VIZ.generation, itos = VIZ.itos;
  let step = 0;

  function policyProbs(logits) {
    const T = parseFloat(tempSlider.value);
    const kRaw = topkSel.value;
    let z = logits.map(v => v / T);
    if (kRaw !== 'all') {
      const k = parseInt(kRaw);
      const thr = [...z].sort((a, b) => b - a)[k - 1];
      z = z.map(v => v < thr ? -Infinity : v);
    }
    return softmaxArr(z);
  }
  function draw() {
    ctx.clearRect(0, 0, W, H);
    const st = G.steps[Math.min(step, G.steps.length - 1)];
    const p = policyProbs(st.logits);
    const idx = p.map((_, i) => i).sort((a, b) => p[b] - p[a]).slice(0, 10);
    ctx.fillStyle = VC.dim; ctx.font = '11px -apple-system, sans-serif';
    ctx.fillText(`policy distribution at step ${Math.min(step + 1, G.steps.length)} / ${G.steps.length} — real logits, reshaped by τ and top-k (Remark 7.2)`, 30, 20);
    const sy0 = 40, bh = 20, bg = 5, mw = W - 230;
    idx.forEach((i, row) => {
      const y = sy0 + row * (bh + bg), w = p[i] * mw;
      const isChosen = (i === st.chosen) && step < G.steps.length;
      ctx.fillStyle = VC.bg; ctx.fillRect(150, y, mw, bh);
      ctx.fillStyle = isChosen ? VC.green : VC.blue; ctx.fillRect(150, y, w, bh);
      ctx.fillStyle = VC.text; ctx.font = 'bold 12px monospace'; ctx.textAlign = 'right';
      ctx.fillText(vizTok(itos[String(i)]), 138, y + 14);
      ctx.textAlign = 'start'; ctx.fillStyle = VC.dim; ctx.font = '10px monospace';
      ctx.fillText(p[i].toFixed(3) + (isChosen ? '  ← sampled (τ=1 record)' : ''), 156 + w, y + 14);
    });
    const H_p = entropyArr(p);
    ctx.fillStyle = VC.dim; ctx.font = '11px -apple-system, sans-serif';
    ctx.fillText(`H(policy) = ${H_p.toFixed(3)} nats — zeroed rows are the deleted tail`, 30, H - 12);
    const shown = G.steps[0].context + G.steps.slice(0, step).map(s => itos[String(s.chosen)]).join('');
    outputSpan.textContent = JSON.stringify(shown).slice(1, -1) + '▍';
  }
  nextBtn.addEventListener('click', () => { if (step < G.steps.length) { step++; draw(); } });
  resetBtn.addEventListener('click', () => { step = 0; draw(); });
  tempSlider.addEventListener('input', () => { tempVal.textContent = parseFloat(tempSlider.value).toFixed(1); draw(); });
  topkSel.addEventListener('change', draw);
  draw();
}

// ═══ 6. Embedding geometry (Ch 3) ═══
function initEmbeddingViz() {
  if (vizMissing('embedding-canvas')) return;
  const canvas = document.getElementById('embedding-canvas');
  const ctx = canvas.getContext('2d');
  const tokenSelect = document.getElementById('embedding-token');
  const readout = document.getElementById('embedding-readout');
  const W = canvas.width, H = canvas.height;
  const E = VIZ.embedding;
  const xs = E.xy.map(p => p[0]), ys = E.xy.map(p => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const pad = 46;
  const px = (x) => pad + (x - x0) / (x1 - x0) * (W - 2 * pad);
  const py = (y) => pad + (y - y0) / (y1 - y0) * (H - 2 * pad);
  const cls = (ch) => /\s/.test(ch) ? VC.dim : /[aeiou]/i.test(ch) ? VC.accent : /[a-z]/i.test(ch) ? VC.blue : VC.purple;
  let hover = -1;
  E.labels.forEach((label, i) => tokenSelect.add(new Option(vizTok(label), i)));

  function draw() {
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = VC.dim; ctx.font = '11px -apple-system, sans-serif';
    ctx.fillText('W_E rows projected to first two principal components — trained toy model', 26, 20);
    ctx.fillText('vowels', W - 200, 20); ctx.fillStyle = VC.accent; ctx.fillRect(W - 216, 12, 10, 10);
    ctx.fillStyle = VC.dim; ctx.fillText('consonants', W - 130, 20); ctx.fillStyle = VC.blue; ctx.fillRect(W - 146, 12, 10, 10);
    ctx.fillStyle = VC.dim; ctx.fillText('other', W - 48, 20); ctx.fillStyle = VC.purple; ctx.fillRect(W - 64, 12, 10, 10);
    E.labels.forEach((ch, i) => {
      const X = px(E.xy[i][0]), Y = py(E.xy[i][1]);
      const big = i === hover;
      ctx.fillStyle = cls(ch);
      ctx.font = (big ? 'bold 22px' : 'bold 14px') + ' monospace';
      ctx.textAlign = 'center';
      ctx.fillText(vizTok(ch), X, Y + 5);
      ctx.textAlign = 'start';
    });
    const ve = E.var_explained;
    const selectedText = hover >= 0
      ? `token ${vizTok(E.labels[hover])}: PC1 ${E.xy[hover][0].toFixed(3)}, PC2 ${E.xy[hover][1].toFixed(3)}. `
      : '';
    readout.textContent = `${selectedText}variance explained: PC1 ${(ve[0] * 100).toFixed(0)}%  PC2 ${(ve[1] * 100).toFixed(0)}% — nearby characters behave alike under every later linear map`;
  }
  canvas.addEventListener('mousemove', (e) => {
    const r = canvas.getBoundingClientRect();
    const mx = (e.clientX - r.left) * (W / r.width), my = (e.clientY - r.top) * (H / r.height);
    hover = -1;
    E.labels.forEach((_, i) => {
      const dX = px(E.xy[i][0]) - mx, dY = py(E.xy[i][1]) - my;
      if (dX * dX + dY * dY < 140) hover = i;
    });
    tokenSelect.value = hover >= 0 ? String(hover) : '';
    draw();
  });
  tokenSelect.addEventListener('change', () => { hover = Number(tokenSelect.value); draw(); });
  draw();
}

// ═══ 7. Quantization grid on real weights (Lemma 10.2) ═══
function initQuantViz() {
  if (vizMissing('quant-canvas')) return;
  const canvas = document.getElementById('quant-canvas');
  const ctx = canvas.getContext('2d');
  const bitsSlider = document.getElementById('quant-bits');
  const bitsVal = document.getElementById('quant-bits-val');
  const symChk = document.getElementById('quant-sym');
  const readout = document.getElementById('quant-readout');
  const W = canvas.width, H = canvas.height;
  const x = VIZ.quant.weights;
  const xmin = Math.min(...x), xmax = Math.max(...x);

  function quantize(bits, sym) {
    let qmin, qmax, s, z;
    if (sym) {
      qmax = 2 ** (bits - 1) - 1; qmin = -qmax;
      s = Math.max(Math.max(Math.abs(xmin), Math.abs(xmax)) / qmax, 1e-12); z = 0;
    } else {
      qmin = 0; qmax = 2 ** bits - 1;
      s = Math.max((xmax - xmin) / (qmax - qmin), 1e-12);
      z = Math.max(qmin, Math.min(qmax, Math.round(qmin - xmin / s)));
    }
    const q = x.map(v => Math.max(qmin, Math.min(qmax, Math.round(v / s + z))));
    const xh = q.map(qi => (qi - z) * s);
    return { s, z, qmin, qmax, xh };
  }
  function draw() {
    const bits = parseInt(bitsSlider.value), sym = symChk.checked;
    const { s, z, qmin, qmax, xh } = quantize(bits, sym);
    ctx.clearRect(0, 0, W, H);
    const lo = Math.min(xmin, (qmin - z) * s), hi = Math.max(xmax, (qmax - z) * s);
    const nx = (v) => 40 + (v - lo) / (hi - lo) * (W - 80);
    ctx.fillStyle = VC.dim; ctx.font = '11px -apple-system, sans-serif';
    ctx.fillText(`${VIZ.quant.name}: real values (top row) snap to the ${qmax - qmin + 1}-point grid (bottom row)`, 30, 20);
    // grid points
    const yGrid = H - 70, yVals = 70;
    ctx.strokeStyle = VC.grid;
    for (let q = qmin; q <= qmax; q++) {
      const X = nx((q - z) * s);
      ctx.beginPath(); ctx.moveTo(X, yGrid - 8); ctx.lineTo(X, yGrid + 8); ctx.stroke();
      if (qmax - qmin <= 16) { ctx.fillStyle = VC.dim; ctx.font = '9px monospace'; ctx.textAlign = 'center'; ctx.fillText(String(q), X, yGrid + 22); ctx.textAlign = 'start'; }
    }
    ctx.strokeStyle = VC.grid; ctx.beginPath(); ctx.moveTo(40, yGrid); ctx.lineTo(W - 40, yGrid); ctx.stroke();
    // values, error segments
    let maxErr = 0;
    x.forEach((v, i) => {
      const err = Math.abs(v - xh[i]); maxErr = Math.max(maxErr, err);
      ctx.strokeStyle = 'rgba(240,136,62,0.35)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(nx(v), yVals + 6); ctx.lineTo(nx(xh[i]), yGrid - 6); ctx.stroke();
      ctx.fillStyle = VC.blue; ctx.beginPath(); ctx.arc(nx(v), yVals, 3.5, 0, 7); ctx.fill();
      ctx.fillStyle = VC.accent; ctx.beginPath(); ctx.arc(nx(xh[i]), yGrid, 3.5, 0, 7); ctx.fill();
    });
    ctx.fillStyle = VC.blue; ctx.font = '10px monospace'; ctx.fillText('x (real weights)', 40, yVals - 14);
    ctx.fillStyle = VC.accent; ctx.fillText('x̂ (dequantized)', 40, yGrid - 16);
    const bound = s / 2, tight = Math.abs(maxErr - bound) < 1e-9;
    readout.textContent = `s = ${s.toFixed(4)}  z = ${z}  max|x−x̂| = ${maxErr.toFixed(4)} ≤ s/2 = ${bound.toFixed(4)}${tight ? ' (achieved!)' : ''}`;
  }
  bitsSlider.addEventListener('input', () => { bitsVal.textContent = bitsSlider.value; draw(); });
  symChk.addEventListener('change', draw);
  draw();
}

// ═══ 8. KV-cache memory: MHA → GQA → MQA (pure formula) ═══
function initKVViz() {
  const canvas = document.getElementById('kv-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const get = (id) => document.getElementById(id);
  const ids = ['kv-l', 'kv-h', 'kv-d', 'kv-t', 'kv-g'];

  function fmt(bytes) {
    const u = ['B', 'KB', 'MB', 'GB', 'TB']; let i = 0;
    while (bytes >= 1024 && i < u.length - 1) { bytes /= 1024; i++; }
    return bytes.toFixed(bytes >= 100 ? 0 : 1) + ' ' + u[i];
  }
  function draw() {
    const L = +get('kv-l').value, Hh = +get('kv-h').value, D = +get('kv-d').value;
    const T = 2 ** (+get('kv-t').value), B = 1, BPV = 2;
    get('kv-g').max = Hh;
    const G = Math.min(+get('kv-g').value, Hh);
    ['l', 'h', 'd', 'g'].forEach(k => get(`kv-${k}-val`).textContent = ({ l: L, h: Hh, d: D, g: G })[k]);
    get('kv-t-val').textContent = get('kv-t').value;
    const bytes = (g) => 2 * L * g * D * T * B * BPV;
    const rows = [
      { label: `MHA  (G = H = ${Hh})`, v: bytes(Hh), c: VC.red },
      { label: `GQA  (G = ${G})`, v: bytes(G), c: VC.accent },
      { label: 'MQA  (G = 1)', v: bytes(1), c: VC.green },
    ];
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = VC.dim; ctx.font = '12px -apple-system, sans-serif';
    ctx.fillText(`bytes = 2 · L · G · D · T · B · bytes/value   with T = ${T.toLocaleString()}, B = 1, fp16`, 30, 24);
    const maxV = rows[0].v, bx = 190, bw = W - 260;
    rows.forEach((r, i) => {
      const y = 56 + i * 62, w = Math.max(2, r.v / maxV * bw);
      ctx.fillStyle = VC.text; ctx.font = '13px monospace'; ctx.textAlign = 'right';
      ctx.fillText(r.label, bx - 12, y + 22);
      ctx.textAlign = 'start';
      ctx.fillStyle = VC.bg; ctx.fillRect(bx, y, bw, 32);
      ctx.fillStyle = r.c; ctx.fillRect(bx, y, w, 32);
      ctx.fillStyle = VC.bright; ctx.font = 'bold 13px monospace';
      ctx.fillText(fmt(r.v), bx + w + 10, y + 21);
    });
    ctx.fillStyle = VC.dim; ctx.font = '11px -apple-system, sans-serif';
    ctx.fillText(`GQA saves ${(100 * (1 - G / Hh)).toFixed(0)}% of the MHA cache; the loss function never changes.`, 30, H - 14);
  }
  ids.forEach(id => get(id).addEventListener('input', draw));
  draw();
}
"""
