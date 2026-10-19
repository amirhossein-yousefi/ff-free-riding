// Component "theory" (SPEC_FINAL.md §5.2), ES module, no imports: Theorem 3.1 made tangible.
// (A) ILLUSTRATION: sliders evaluate the paper's Eq. 3 barrier (softplus, beta = 4), the Eq. 4 ratio
//     R = (1 + e^(b m)) / (1 + e^(b (m + g P))) (log space) and the Eq. 6 band; or the squared hinge of the Remark
//     (theta = 2). Outputs sit in [data-illus] ("about", one significant digit below 0.01) next to a MEASURED chip.
//     On phones (< 640 px) the sliders sit between the loss chart and the readouts (then the ratio chart), so that
//     dragging P shows both the moving tangent and R without scrolling.
// (B) MEASURED free-riding index per block (Fig. 1a); the block of the loaded Tab. 4 preset is shaded; one Tab
//     stop: arrow keys, Home and End move between points and show the tooltip, Esc hides it.
// (C) ILLUSTRATION of the Sec. 3.3 hardness gate with MEASURED Tab. 16 cards for the conditions the paper ran;
//     the shell's #eq-gate is adopted (the same node) at the top of (C).
// Contract (SPEC_FINAL §4.4): render into [data-live] with textContent, then data-state="live"; on any error log
// "[theory]", put #eq-gate back and set data-state="broken". Printed results come only from SITE_DATA.theory.facts
// and carry data-fact/data-sub. Colors come from CSS tokens only (theme changes need no JS). Ids start "theory-".

const NAME = 'theory';
const NS = 'http://www.w3.org/2000/svg';
const MINUS = '\u2212';
const NB = '\u00a0';
const THIN = '\u202f';          // narrow no-break space (math spacing that never wraps)

/* ------------------------------------------------------------------------------------------ small helpers */
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v === null || v === undefined || v === false) continue;
      if (k === 'class') el.className = v;
      else el.setAttribute(k, v === true ? '' : String(v));
    }
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}
function sv(tag, attrs, parent) {
  const el = document.createElementNS(NS, tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined && v !== false) el.setAttribute(k, String(v));
  if (parent) parent.appendChild(el);
  return el;
}
// SVG text from parts: a string, or [string, {class, dy, ...}] for a tspan (labels are data: textContent only)
function stext(parent, x, y, parts, attrs) {
  const t = sv('text', { x: r1(x), y: r1(y), ...attrs }, parent);
  for (const p of [].concat(parts)) {
    if (typeof p === 'string') t.appendChild(document.createTextNode(p));
    else { const ts = sv('tspan', p[1], t); ts.textContent = p[0]; }
  }
  return t;
}
// multi-line SVG label: rows of parts; returns the <text>
function slines(parent, x, y, rows, attrs, lh = 16) {
  const t = sv('text', { x: r1(x), y: r1(y), ...attrs }, parent);
  rows.forEach((row, i) => {
    const line = sv('tspan', { x: r1(x), dy: i === 0 ? 0 : lh }, t);
    for (const p of [].concat(row)) {
      if (typeof p === 'string') line.appendChild(document.createTextNode(p));
      else { const ts = sv('tspan', p[1], line); ts.textContent = p[0]; }
    }
  });
  return t;
}
const r1 = (v) => Math.round(v * 10) / 10;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const lin = (d0, d1, a, b) => (v) => a + ((v - d0) / (d1 - d0)) * (b - a);
const logScale = (d0, d1, a, b) => { const l0 = Math.log10(d0), l1 = Math.log10(d1); return (v) => a + ((Math.log10(v) - l0) / (l1 - l0)) * (b - a); };
const SUP = { '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '−': '⁻' };
const supText = (s) => s.split('').map((c) => SUP[c] || c).join('');
const signed = (v, d = 2) => (v < 0 ? MINUS : '') + Math.abs(v).toFixed(d);
const plainNum = (v) => String(v).replace('-', MINUS);
// stable softplus log(1 + e^x)
const softplus = (x) => (x > 0 ? x + Math.log1p(Math.exp(-x)) : Math.log1p(Math.exp(x)));
// one significant digit in powers of ten (illustration grammar below 0.01)
function sci1(x) {
  let e = Math.floor(Math.log10(x));
  let k = Math.round(x / 10 ** e);
  if (k >= 10) { k = 1; e += 1; }
  return { k, e };
}
// a printed string such as "7.2×10−3" or "≤ 10−6" with its exponent as a superscript (text content unchanged)
function appendPow(node, s) {
  const m = /^(.*10)([−-]\d+)$/.exec(s);
  if (!m) { node.append(document.createTextNode(s)); return node; }
  node.append(document.createTextNode(m[1]), h('sup', null, m[2]));
  return node;
}
function shapePath(kind, cx, cy, r) {
  switch (kind) {
    case 'square': return `M${r1(cx - r)},${r1(cy - r)}h${r1(2 * r)}v${r1(2 * r)}h${r1(-2 * r)}z`;
    case 'triangle': { const t = r * 1.3; return `M${r1(cx)},${r1(cy - t)}L${r1(cx + t)},${r1(cy + t * 0.8)}L${r1(cx - t)},${r1(cy + t * 0.8)}z`; }
    default: return `M${r1(cx - r)},${r1(cy)}a${r},${r} 0 1,0 ${2 * r},0a${r},${r} 0 1,0 ${-2 * r},0z`;
  }
}
// width-driven redraw: one ResizeObserver per chart container, throttled with requestAnimationFrame
function onSize(node, draw) {
  let lw = 0, lh = 0, raf = 0;
  const run = () => {
    raf = 0;
    const w = Math.round(node.clientWidth), hh = Math.round(node.clientHeight);
    if (w > 0 && hh > 0 && (w !== lw || hh !== lh)) { lw = w; lh = hh; draw(w, hh); }
  };
  if ('ResizeObserver' in window) new ResizeObserver(() => { if (!raf) raf = requestAnimationFrame(run); }).observe(node);
  return { now: () => { lw = 0; run(); }, redraw: () => { if (lw) draw(lw, lh); } };
}
function bbox(el) { try { const b = el.getBBox(); return b.width ? b : null; } catch (_) { return null; } }
// keep a (multi-line) SVG label horizontally inside [minX, maxX] by shifting its x attributes
function fitX(t, minX, maxX) {
  const b = bbox(t);
  if (!b) return;
  let dx = 0;
  if (b.x + b.width > maxX) dx = maxX - (b.x + b.width);
  if (b.x + dx < minX) dx = minX - b.x;
  if (!dx) return;
  for (const el of [t, ...t.querySelectorAll('tspan[x]')]) el.setAttribute('x', r1(+el.getAttribute('x') + dx));
}
// keep a focused chart mark fully in view: Firefox does not scroll a focused SVG <g> into view while its <svg> is
// partly visible, and g.scrollIntoView() does nothing there. The window scrolls by the mark's overflow, between the
// page's scroll padding at the top (site.css: under the sticky top bar) and 12 px above the bottom edge.
function reveal(node) {
  const r = node.getBoundingClientRect();
  if (!r.width && !r.height) return;
  const top = parseFloat(getComputedStyle(document.documentElement).scrollPaddingTop) || 72, bottom = window.innerHeight - 12;
  const dy = r.bottom > bottom && r.height <= bottom - top ? r.bottom - bottom : r.top < top ? r.top - top : 0;
  if (!dy) return;
  try { window.scrollBy({ top: dy, left: 0, behavior: 'instant' }); } catch (_) { window.scrollBy(0, dy); }
}
const overlap = (a, b, pad = 3) => a && b && a.x < b.x + b.width + pad && b.x < a.x + a.width + pad && a.y < b.y + b.height + pad && b.y < a.y + a.height + pad;

/* ------------------------------------------------------------------------------------------ boot */
function boot() {
  const root = document.getElementById(`viz-${NAME}`);
  if (!root) return;
  const undo = [];
  let failed = false;
  const fail = (err) => {
    if (failed) return;
    failed = true;
    console.error(`[${NAME}]`, err);
    for (const f of undo.splice(0).reverse()) { try { f(); } catch (_) { /* keep restoring */ } }
    root.dataset.state = 'broken';
  };
  try { mount(root, undo, fail); } catch (err) { fail(err); }
}

function mount(root, undo, fail) {
  const SD = window.SITE_DATA || {};
  const D = SD[NAME];
  if (!D || !D.facts) throw new Error(`window.SITE_DATA.${NAME} is missing (assets/data/${NAME}.js)`);
  const SITE = SD.site || {};
  const live = root.querySelector('[data-live]');
  if (!live) throw new Error('[data-live] is missing in the mount');
  const fact = (key) => { const f = D.facts[key]; if (!f || typeof f.s !== 'string') throw new Error(`fact ${key} is missing`); return f; };
  const val = (key) => { const f = fact(key); if (typeof f.v !== 'number') throw new Error(`fact ${key} has no single value`); return f.v; };
  // a printed fact as a node: data-fact (+ data-sub) so the checkers compare it with FACTS.json
  const F = (key, cls) => {
    const [id, sub] = key.split('#');
    return appendPow(h('span', { class: cls || null, 'data-fact': id, 'data-sub': sub === undefined ? null : sub }), fact(key).s);
  };
  const safe = (fn) => (...a) => { try { return fn(...a); } catch (err) { fail(err); return undefined; } };

  const BETA = val(D.constants.beta);           // Eq. 3: beta = 4
  const THETA = val(D.constants.theta);         // Remark: theta = 2
  const GAMMA0 = val(D.constants.gamma0);       // the gate's gamma0 = 0.7 (Tab. 23 caption)
  const TAU_REP = val(D.constants.tau_reported); // reported gate: tau = 1
  const N_TEST = fact(D.constants.n).s;         // "1024"
  const presets = D.presets;
  const gate = D.gate;
  if (!Array.isArray(presets) || presets.length !== 4 || !gate || !Array.isArray(gate.conditions)) throw new Error('theory.js: presets or gate conditions missing');
  const def = presets.find((p) => p.default) || presets[2];
  const arm = (k) => (SITE.arms && SITE.arms[k]) || { name: k, shape: 'square', glyph: `glyph--${k}` };
  const SYMBOL = (shape) => (SITE.shapes && SITE.shapes[shape]) || `sh-${shape}`;
  const glyph = (k) => {
    const a = arm(k);
    const g = sv('svg', { class: `glyph ${a.glyph || 'glyph--' + k}`, 'aria-hidden': 'true', focusable: 'false' });
    sv('use', { href: `#${SYMBOL(a.shape)}` }, g);
    return g;
  };
  const armChip = (k, text) => h('span', { class: 'arm' }, glyph(k), text || arm(k).name);
  const eqs = (a, b) => `${a}${NB}=${NB}${b}`;
  const gTxt = (g) => (g === 0 ? '0' : plainNum(g));

  /* ---------------------------------------------------------------------------------------- state */
  const st = {
    barrier: 'softplus',
    m: val(def.m), P: val(def.P), g: def.gamma,
    preset: def.id,
    kappa: 0, tau: TAU_REP,
  };
  const presetById = Object.fromEntries(presets.map((p) => [p.id, p]));

  /* ---------------------------------------------------------------------------------------- formulas */
  // |l'(u)| and l(u) for the chosen barrier; R by Eq. (4) (softplus, log space) or the Remark (squared hinge)
  const BAR = {
    softplus: { name: `softplus, β${NB}=${NB}${plainNum(BETA)}`, short: 'softplus', loss: (u) => softplus(-BETA * u), slope: (u) => BETA / (1 + Math.exp(BETA * u)), s0: BETA / 2, yMax: 1.25, yTicks: [0.5, 1] },
    hinge: { name: `squared hinge, θ${NB}=${NB}${plainNum(THETA)}`, short: 'squared hinge', loss: (u) => Math.max(0, THETA - u) ** 2, slope: (u) => 2 * Math.max(0, THETA - u), s0: 2 * THETA, yMax: 5, yTicks: [2, 4] },
  };
  function ratio(barrier, m, P, g) {
    if (barrier === 'softplus') return Math.exp(softplus(BETA * m) - softplus(BETA * (m + g * P)));
    if (m >= THETA) return NaN;
    return Math.max(0, THETA - m - g * P) / (THETA - m);
  }
  const noShift = () => Math.abs(st.g * st.P) < 1e-9;
  function slopeWords(barrier, u) {
    const s = BAR[barrier].slope(u);
    if (s === 0) return ['flat', 'no signal (hard zero)'];
    const rel = s / BAR[barrier].s0;
    if (rel >= 0.05) return ['sloped', 'learning signal'];
    if (rel >= 0.005) return ['shallow', 'weak signal'];
    return ['flat', 'almost no signal'];
  }
  // illustration readouts (SPEC_FINAL §5.2): "about", two significant digits, one below 0.01
  function rReadout(R) {
    if (Number.isNaN(R)) return { plain: 'undefined: no local slope at this margin', parts: ['undefined: no local slope at this margin'] };
    if (noShift()) return { plain: '1', parts: ['1'] };
    if (R > 1) return { plain: 'more than 1 (history amplifies)', parts: ['more than 1 (history amplifies)'] };
    if (R === 0) return { plain: '0 (hard zero)', parts: ['0 (hard zero)'] };
    if (R < 0.01) {
      const { k, e } = sci1(R);
      const ex = (e < 0 ? MINUS : '') + Math.abs(e);
      return { plain: `about ${k} × 10${supText(ex)}`, parts: [`about ${k}${NB}×${NB}10`, h('sup', null, ex)] };
    }
    const t = `about ${R.toPrecision(2)}`;
    return { plain: t, parts: [t] };
  }
  function lostReadout(R) {
    if (Number.isNaN(R)) return 'undefined';
    if (noShift() || R >= 1) return 'none';
    if (R === 0) return 'all of it';
    const lost = 1 - R;
    if (lost >= 0.99) return 'more than 99%';
    if (lost < 0.01) return 'less than 1%';
    return `about ${Math.round(lost * 100)}%`;
  }
  function regimeText() {
    if (st.barrier === 'hinge' && st.m >= THETA) return `Own margin m ≥ θ: the squared hinge is already zero here, so there is no local slope at this margin.`;
    if (st.g === 0) return 'γ = 0: nothing is inherited, so nothing is lost (history-free).';
    if (Math.abs(st.P) < 1e-9) return 'P = 0: nothing is inherited yet (as at block 0), so R = 1.';
    if (st.P < 0) return 'Outside the theorem’s regime: with P < 0 the ratio can exceed one, so history amplifies the slope here.';
    if (st.barrier === 'hinge') return 'Squared hinge: R falls linearly to a hard zero (Remark after Theorem 3.1).';
    if (st.m < 0) return 'Outside the theorem’s regime (it needs m ≥ 0): R still stays at most 1, but the band of Eq. (6) is not guaranteed.';
    return 'Inside the theorem’s regime (m ≥ 0, P ≥ 0): R lies in the band of Eq. (6).';
  }

  /* ---------------------------------------------------------------------------------------- (A) markup */
  const liveRegion = h('p', { class: 'vh', 'aria-live': 'polite', id: `${NAME}-live`, 'data-illus': '' });

  function rangeField({ id, label, min, max, step, value, hint, ticks }) {
    const input = h('input', { class: 'range', type: 'range', id, min, max, step, value, 'aria-describedby': hint ? `${id}-hint` : null });
    const out = h('output', { for: id, id: `${id}-out` });
    const field = h('div', { class: 'range-field th-range', 'data-illus': '' },
      h('div', { class: 'range-field__top' }, h('label', { for: id }, label), out), input);
    if (ticks) {
      const tk = h('div', { class: 'th-rticks', 'aria-hidden': 'true' });
      for (const [v, text, sub] of ticks) {
        const f = (v - min) / (max - min);
        const sp = h('span', { class: `th-rtick${f <= 0 ? ' th-rtick--start' : f >= 1 ? ' th-rtick--end' : ''}` }, h('span', null, text), sub ? h('span', { class: 'th-rtick__sub' }, sub) : null);
        sp.style.setProperty('--f', String(f));
        tk.append(sp);
      }
      field.append(tk);
    }
    if (hint) field.append(h('p', { class: 'range-field__hint', id: `${id}-hint` }, hint));
    return { field, input, out };
  }
  const fillPct = (input) => input.style.setProperty('--pct', `${((+input.value - +input.min) / (+input.max - +input.min)) * 100}%`);

  const rP = rangeField({ id: `${NAME}-p`, label: ['Inherited margin ', h('i', null, 'P')], min: -2, max: 8, step: 0.01, value: st.P, hint: 'earlier blocks’ total margin, detached' });
  const rM = rangeField({ id: `${NAME}-m`, label: ['Own margin ', h('i', null, 'm')], min: -1, max: 3, step: 0.01, value: st.m, hint: 'block d’s own margin' });
  const rG = rangeField({ id: `${NAME}-g`, label: 'History weight γ', min: 0, max: 1, step: 0.05, value: st.g, ticks: [[0, '0', 'history-free'], [0.7, '0.7', 'cumulative'], [1, '1', '']] });
  rM.field.querySelector('.range-field__hint').replaceChildren('block ', h('i', null, 'd'), '’s own margin');

  const presetBtns = presets.map((p) => h('button', { type: 'button', class: 'th-preset', 'aria-pressed': 'false', 'data-preset': p.id },
    `Block ${p.block} (${eqs('γ', gTxt(p.gamma))})`));
  const presetGroup = h('div', { class: 'th-presets', role: 'group', 'aria-labelledby': `${NAME}-presets-label` },
    h('span', { class: 'ctl__label', id: `${NAME}-presets-label` }, 'Load batch means (softplus run, App.', NB, 'Tab.', NB, '4)'),
    h('div', { class: 'th-presets__row' }, presetBtns));
  const seg = h('div', { class: 'seg' });
  for (const key of ['softplus', 'hinge']) {
    const id = `${NAME}-barrier-${key}`;
    seg.append(h('input', { type: 'radio', name: `${NAME}-barrier`, id, value: key, checked: key === st.barrier }),
      h('label', { for: id }, h('span', { class: 'seg__long' }, BAR[key].name), h('span', { class: 'seg__short', 'aria-hidden': 'true' }, BAR[key].short)));
  }
  const barrierCtl = h('fieldset', { class: 'ctl th-barrier' }, h('legend', { class: 'ctl__label' }, 'Barrier'), seg);
  const rangesA = h('div', { class: 'th-controls__ranges' }, rP.field, rM.field, rG.field);
  const controlsA = h('div', { class: 'controls th-controls' },
    h('div', { class: 'th-controls__top' }, presetGroup, barrierCtl), rangesA);

  const lossTitle = h('h4', { class: 'th-chart__title', id: `${NAME}-loss-title` }, 'Loss at block ', h('i', null, 'd'), h('span', { class: 'th-chart__sub' }, ' · the tangent’s slope is the learning signal'));
  const ratioSub = h('span', { class: 'th-chart__sub' });
  const ratioSubText = () => ratioSub.replaceChildren(...(st.barrier === 'hinge'
    ? [' · Remark after Theorem', NB, '3.1, log scale'] : [' · Eq.', NB, '(4), log scale']));
  const ratioTitle = h('h4', { class: 'th-chart__title', id: `${NAME}-ratio-title` }, 'Surviving signal ', h('i', null, 'R'), ' ',
    h('span', { class: 'nw' }, '= |ℓ′(', h('i', null, 'm'), THIN, '+', THIN, 'γ', h('i', null, 'P'), ')|', THIN, '/', THIN, '|ℓ′(', h('i', null, 'm'), ')|'),
    ratioSub);
  const lossPlot = h('div', { class: 'th-plot th-plot--a', id: `${NAME}-loss-plot` });
  const ratioPlot = h('div', { class: 'th-plot th-plot--a', id: `${NAME}-ratio-plot` });
  const lossSvg = sv('svg', { class: 'th-svg', role: 'img' });
  const ratioSvg = sv('svg', { class: 'th-svg', role: 'img' });
  lossPlot.append(lossSvg);
  ratioPlot.append(ratioSvg);
  const lossChart = h('div', { class: 'th-chart' }, lossTitle, lossPlot);
  const ratioChart = h('div', { class: 'th-chart' }, ratioTitle, ratioPlot);
  const charts = h('div', { class: 'th-charts', 'data-illus': '' }, lossChart, ratioChart);

  const rVal = h('span', { class: 'tray__val', id: `${NAME}-r` });
  const lostVal = h('span', { class: 'tray__val', id: `${NAME}-lost` });
  const meterFill = h('span', { class: 'th-meter__fill' });
  const regime = h('p', { class: 'th-regime', id: `${NAME}-regime`, 'data-illus': '' });
  const measuredA = h('div', { class: 'measured th-measured', id: `${NAME}-measured` });
  const tray = h('div', { class: 'tray th-tray' },
    h('div', { class: 'th-tray__r', 'data-illus': '' }, h('span', { class: 'tray__key' }, 'Illustration: surviving signal ', h('i', null, 'R')), rVal),
    h('div', { class: 'th-tray__lost', 'data-illus': '' }, h('span', { class: 'tray__key' }, 'Illustration: share of the loss slope lost to history, 1', THIN, MINUS, THIN, 'min(1,', NB, h('i', null, 'R'), ')'), lostVal,
      h('span', { class: 'th-meter', 'aria-hidden': 'true' }, meterFill)),
    regime, measuredA);

  const capA = h('figcaption', { class: 'fig__caption' },
    h('i', { class: 'fig__lead' }, 'Illustration:'), ' the sliders evaluate Theorem', NB, '3.1’s formulas (Eqs.', NB, '4 and 6) for the values you choose; the outputs are not measurements. Defaults are block', NB, '3’s batch-mean margins in the trained cumulative model (Tab.', NB, '4: ',
    h('i', null, 'm'), NB, '=', NB, F(def.m), ', ', h('i', null, 'P'), NB, '=', NB, F(def.P), '). ',
    h('a', { href: '#theorem', class: 'th-more' }, 'Theorem', NB, '3.1 as stated in the paper', NB, '↓'));
  const figA = h('figure', { class: 'fig fig--illus th-fig th-a', id: `${NAME}-a`, 'aria-labelledby': `${NAME}-a-title` },
    h('div', { class: 'fig__head' }, h('h3', { class: 'fig__title', id: `${NAME}-a-title` }, 'Theorem', NB, '3.1, live: slide the inherited margin'), h('span', { class: 'badge badge--illus' }, 'Illustration · Eqs.', NB, '(3)–(6)')),
    controlsA, charts, tray, capA);

  /* ---------------------------------------------------------------------------------------- (B) markup */
  const fdPlot = h('div', { class: 'th-plot th-plot--fd', id: `${NAME}-fd-plot` });
  const fdSvg = sv('svg', { class: 'th-svg', role: 'group', 'aria-labelledby': `${NAME}-b-title`, 'aria-describedby': `${NAME}-fd-help` });
  fdPlot.append(fdSvg);
  const fdHelp = h('p', { class: 'vh', id: `${NAME}-fd-help` }, 'Chart: use the arrow keys to move between the points; Home and End jump to block 0 and block 3; Escape hides the tooltip.');
  const fdNote = h('p', { class: 'th-note', id: `${NAME}-fd-note` });
  const capB = h('figcaption', { class: 'fig__caption' },
    h('span', { class: 'th-def' }, h('i', null, 'F'), h('sub', null, h('i', null, 'd')), `${NB}=${NB}E[1${THIN}${MINUS}${THIN}min(1,${NB}`, h('i', null, 'R'), h('sub', null, h('i', null, 'd')), ')]', h('span', { class: 'th-def__src' }, ` (App.${NB}A)`)),
    'Share of block ', h('i', null, 'd'), '’s loss slope that history removes (the readout 1', THIN, MINUS, THIN, 'min(1,', NB, h('i', null, 'R'), ') above), averaged over test images (0', NB, '=', NB, 'none). The theorem’s assumptions hold pointwise for ', F('s3.coverage_96'), ' of test examples (', F('s3.coverage_ema'), ' on EMA weights).',
    h('span', { class: 'src' }, `CIFAR-10 · 4 blocks, width 128 · seed 42 · first ${N_TEST} test images · ${eqs('β', plainNum(BETA))} · Fig.${NB}1a`));
  const figB = h('figure', { class: 'fig th-fig th-b', id: `${NAME}-b`, 'aria-labelledby': `${NAME}-b-title` },
    h('div', { class: 'fig__head' }, h('h3', { class: 'fig__title', id: `${NAME}-b-title` }, 'The free-riding index, block by block'), h('span', { class: 'badge badge--measured' }, 'Measured')),
    h('ul', { class: 'legend th-legend' }, h('li', null, glyph('cum'), arm('cum').name), h('li', null, glyph('free'), arm('free').name)),
    fdPlot, fdHelp, fdNote, capB);
  const tip = h('div', { class: 'tip', id: `${NAME}-tip`, hidden: true, 'aria-hidden': 'true' });

  /* ---------------------------------------------------------------------------------------- (C) markup */
  const ranCond = (k, t) => gate.conditions.find((c) => Math.abs(c.kappa - k) < 1e-9 && Math.abs(c.tau - t) < 1e-9) || null;
  const kTicks = [[-2, plainNum(-2), ''], [0, '0', 'run'], [2, '2', 'run'], [4, '4', 'run'], [6, '6', '']];
  const tTicks = [[0.5, '0.5', ''], [1, '1', 'run'], [3, '3', 'run'], [4, '4', '']];
  const rK = rangeField({ id: `${NAME}-kappa`, label: 'Threshold κ', min: -2, max: 6, step: 0.5, value: st.kappa, ticks: kTicks, hint: 'where the gate switches' });
  const rT = rangeField({ id: `${NAME}-tau`, label: 'Sharpness τ', min: 0.5, max: 4, step: 0.5, value: st.tau, ticks: tTicks, hint: 'the paper’s temperature: higher τ, sharper switch' });
  const controlsC = h('div', { class: 'controls th-controls th-controls--gate' }, rK.field, rT.field);
  const gatePlot = h('div', { class: 'th-plot th-plot--gate', id: `${NAME}-gate-plot`, 'data-illus': '' });
  const gateSvg = sv('svg', { class: 'th-svg', role: 'img' });
  gatePlot.append(gateSvg);
  const runsCard = h('div', { class: 'measured th-runs', id: `${NAME}-runs` });
  const legendC = h('ul', { class: 'legend th-legend' },
    h('li', null, glyph('cum'), arm('cum').name), h('li', null, glyph('free'), arm('free').name), h('li', null, glyph('gated'), 'hardness gate (your κ, τ)'));
  const capC = h('figcaption', { class: 'fig__caption' },
    'In the paper (Sec.', NB, '3.3), this repair replaces the constant γ by a per-example gate on the goodness that earlier blocks gave the same input, so block ', h('i', null, 'd'),
    ' inherits little history where that goodness is already high (easy examples) and up to γ', h('sub', null, '0'), ' where it is low (hard ones); the paper reports ', eqs('κ', '0'), '.');
  const figC = h('figure', { class: 'fig fig--illus th-fig th-c', id: `${NAME}-c`, 'aria-labelledby': `${NAME}-c-title` },
    h('div', { class: 'fig__head' }, h('h3', { class: 'fig__title', id: `${NAME}-c-title` }, 'The repair it suggests: gate the history per example'), h('span', { class: 'badge badge--illus' }, 'Illustration · gate of Sec.', NB, '3.3')),
    controlsC, legendC, h('div', { class: 'th-gate' }, h('div', { class: 'th-gate__chart' }, gatePlot), runsCard), capC);

  live.replaceChildren(figA, h('div', { class: 'th-pair' }, figB, figC), tip, liveRegion);

  // adopt the shell's gate equation (the same node) at the top of frame (C); put it back if anything fails later
  const eq = root.querySelector('#eq-gate[data-adopt="c"]');
  if (eq) {
    const home = { parent: eq.parentNode, next: eq.nextSibling };
    figC.insertBefore(eq, controlsC);
    undo.push(() => { if (home.parent) home.parent.insertBefore(eq, home.next && home.next.parentNode === home.parent ? home.next : null); });
  }
  undo.push(() => live.replaceChildren());

  /* ---------------------------------------------------------------------------------------- (A) drawing */
  const A_X = [-1, 7];
  const R_P = [-2, 8];
  const R_Y = [1e-8, 1e2];

  function drawLoss(W, H) {
    const svg = lossSvg;
    svg.replaceChildren();
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const B = BAR[st.barrier];
    const narrow = W < 420;
    const mg = { l: 42, r: 14, t: 30, b: 46 };
    const x = lin(A_X[0], A_X[1], mg.l, W - mg.r);
    const u0 = st.m, u1 = st.m + st.g * st.P;
    const shift = !noShift();
    const onX = (u) => u >= A_X[0] && u <= A_X[1];
    const xOf = (u) => x(clamp(u, A_X[0], A_X[1]));

    // 1. direct labels in the poster's words, in a band ABOVE the plot (they never sit on the curve, the marks or
    //    the arrow): one row per label, the label of the left point first; each is centered over its point and
    //    the second row keeps clear of the first row's leader
    const [w0, s0] = slopeWords(st.barrier, clamp(u0, A_X[0], A_X[1]));
    const ownSay = `${w0} → ${w0 === 'flat' && s0 === 'almost no signal' ? 'already well separated' : s0}`;
    let w1 = null, s1 = null;
    if (shift) {
      if (st.g * st.P < 0) { w1 = 'steeper'; s1 = 'history amplifies'; } else [w1, s1] = slopeWords(st.barrier, u1);
    }
    const lab = sv('g', { class: 'th-labels' }, svg);
    const label = (head, say, extra, cls) => {            // one line when it fits the chart, otherwise two
      let t = slines(lab, 0, 0, [[[head, { class: 'th-strong' }], ` ${say}`], ...extra], { class: cls }, 16);
      const b = bbox(t);
      if (b && b.width > W - 12) { t.remove(); t = slines(lab, 0, 0, [[[head, { class: 'th-strong' }]], [say], ...extra], { class: cls }, 16); }
      return t;
    };
    const ownT = label('own margin only:', ownSay, shift ? [] : [[[st.g === 0 ? 'nothing inherited (γ = 0)' : 'nothing inherited (P = 0)', { class: 'th-muted' }]]], 'th-lbl th-halo');
    const histT = shift ? label('+ inherited margin:', `${w1} → ${s1}`, onX(u1) ? [] : [[[`(off the chart ${u1 > A_X[1] ? '→' : '←'})`, { class: 'th-muted' }]]], 'th-lbl th-lbl--hist th-halo') : null;
    const rows = !shift ? [[ownT, xOf(u0), 'own']] : xOf(u0) <= xOf(u1) ? [[ownT, xOf(u0), 'own'], [histT, xOf(u1), 'hist']] : [[histT, xOf(u1), 'hist'], [ownT, xOf(u0), 'own']];
    let top = 4, prevX = null;
    for (const row of rows) {
      const [t, px] = row;
      const b = bbox(t) || { width: 0, height: 16 };
      let left = clamp(px - b.width / 2, 4, W - 4 - b.width);
      if (prevX !== null && left < prevX + 10) left = Math.max(4, Math.min(W - 4 - b.width, prevX + 10));
      for (const el of [t, ...t.querySelectorAll(':scope > tspan')]) el.setAttribute('x', r1(left));
      t.setAttribute('y', r1(top + 12));
      top += b.height + 4;
      prevX = px;
    }
    mg.t = Math.max(30, Math.round(top + 12));
    const y = lin(0, B.yMax, H - mg.b, mg.t);

    // 2. the plot, drawn underneath the label band
    const plot = sv('g', null);
    svg.insertBefore(plot, lab);
    const defs = sv('defs', null, plot);
    const clip = sv('clipPath', { id: `${NAME}-clip-loss` }, defs);
    sv('rect', { x: mg.l, y: mg.t - 4, width: W - mg.l - mg.r, height: H - mg.t - mg.b + 4 }, clip);
    for (const t of B.yTicks) sv('line', { class: 'th-grid', x1: mg.l, x2: W - mg.r, y1: r1(y(t)), y2: r1(y(t)) }, plot);
    sv('line', { class: 'th-axis', x1: mg.l, x2: W - mg.r, y1: H - mg.b, y2: H - mg.b }, plot);
    for (const t of [0, ...B.yTicks]) stext(plot, mg.l - 8, y(t) + 4.5, plainNum(t), { class: 'th-tick', 'text-anchor': 'end' });
    for (let u = A_X[0]; u <= A_X[1]; u += narrow ? 2 : 1) {
      sv('line', { class: 'th-axis', x1: r1(x(u)), x2: r1(x(u)), y1: H - mg.b, y2: H - mg.b + 4 }, plot);
      stext(plot, x(u), H - mg.b + 18, plainNum(u), { class: 'th-tick', 'text-anchor': 'middle' });
    }
    stext(plot, (mg.l + W - mg.r) / 2, H - 6, narrow ? ['margin ', ['u', { class: 'th-it' }], ': ', ['m', { class: 'th-it' }], ' or ', ['m', { class: 'th-it' }], ' + γ', ['P', { class: 'th-it' }]]
      : ['margin ', ['u', { class: 'th-it' }], ': own ', ['m', { class: 'th-it' }], ', or ', ['m', { class: 'th-it' }], ' + γ', ['P', { class: 'th-it' }], ' with history'], { class: 'th-axis-title', 'text-anchor': 'middle' });
    let d = '';
    const n = Math.max(120, Math.round(W / 2));
    for (let i = 0; i <= n; i++) {
      const u = A_X[0] + ((A_X[1] - A_X[0]) * i) / n;
      d += `${i ? 'L' : 'M'}${r1(x(u))},${r1(y(Math.min(B.loss(u), B.yMax * 1.3)))}`;
    }
    sv('path', { class: 'th-curve', d, 'clip-path': `url(#${NAME}-clip-loss)` }, plot);

    const kx = x(1) - x(0), ky = y(1) - y(0);
    const pt = (u) => {                       // screen position of a point on the curve, clamped to the plot
      const uc = clamp(u, A_X[0], A_X[1]);
      const l = B.loss(uc);
      return { x: x(uc), y: l > B.yMax ? mg.t : y(l), offTop: l > B.yMax, offX: !onX(u), side: u > A_X[1] ? 1 : u < A_X[0] ? -1 : 0 };
    };
    const p0 = pt(u0), p1 = pt(u1);
    const tangent = (u, p, cls) => {
      if (p.offX || p.offTop) return;
      const vx = kx, vy = ky * -B.slope(u), len = Math.hypot(vx, vy) || 1, L = narrow ? 22 : 30;
      sv('line', { class: `th-tan ${cls}`, x1: r1(p.x - (vx / len) * L), y1: r1(p.y - (vy / len) * L), x2: r1(p.x + (vx / len) * L), y2: r1(p.y + (vy / len) * L) }, plot);
    };
    // arrow from "own margin only" to "+ inherited margin" (+ gamma P; geometry only), kept inside the plot
    if (shift) {
      const dir = p1.x >= p0.x ? 1 : -1;
      const dx = Math.abs(p1.x - p0.x);
      if (dx > 40) {
        const xa = p0.x + dir * 9, ya = p0.y - 9, xb = p1.x - dir * 10, yb = p1.y - 11;
        const cx = (xa + xb) / 2, cy = Math.max(mg.t + 4, Math.min(ya, yb) - clamp(dx * 0.32, 20, 58));
        sv('path', { class: 'th-arrow', d: `M${r1(xa)},${r1(ya)}Q${r1(cx)},${r1(cy)} ${r1(xb)},${r1(yb)}` }, plot);
        const ang = Math.atan2(yb - cy, xb - cx);
        const hx = (a, r) => r1(xb + Math.cos(ang + a) * r), hy = (a, r) => r1(yb + Math.sin(ang + a) * r);
        sv('path', { class: 'th-arrowhead', d: `M${hx(0, 2)},${hy(0, 2)}L${hx(2.6, 10)},${hy(2.6, 10)}L${hx(-2.6, 10)},${hy(-2.6, 10)}z` }, plot);
        const mx = 0.25 * xa + 0.5 * cx + 0.25 * xb, my = 0.25 * ya + 0.5 * cy + 0.25 * yb;
        if (dx > 70 && my - 7 > mg.t + 12) stext(plot, mx, my - 7, ['+ γ', ['P', { class: 'th-it' }]], { class: 'th-hist-text th-halo', 'text-anchor': 'middle' });
      }
    }
    tangent(u0, p0, 'th-tan--own');
    if (shift) tangent(u1, p1, 'th-tan--hist');
    // markers (2 px surface ring); chevrons at the plot edge for points off the chart
    const chevron = (p, cls) => {
      if (p.offX) {
        const sd = p.side, cx = p.x - sd * 2;
        sv('path', { class: `th-chev ${cls}`, d: `M${r1(cx - sd * 9)},${r1(p.y - 8)}L${r1(cx)},${r1(p.y)}L${r1(cx - sd * 9)},${r1(p.y + 8)}` }, plot);
      } else {
        sv('path', { class: `th-chev ${cls}`, d: `M${r1(p.x - 8)},${r1(p.y + 7)}L${r1(p.x)},${r1(p.y - 2)}L${r1(p.x + 8)},${r1(p.y + 7)}` }, plot);
      }
    };
    if (shift) {
      if (p1.offX || p1.offTop) chevron(p1, 'th-chev--hist');
      else sv('circle', { class: 'th-pt th-pt--hist', cx: r1(p1.x), cy: r1(p1.y), r: 6.5 }, plot);
    }
    if (p0.offTop) chevron(p0, 'th-chev--own');
    else sv('circle', { class: 'th-pt th-pt--own', cx: r1(p0.x), cy: r1(p0.y), r: 6.5 }, plot);

    // 3. leaders from each label down to its point (straight down when the label is centered over it)
    for (const [t, , who] of rows) {
      const p = who === 'own' ? p0 : p1;
      const b = bbox(t);
      if (!b || p.y - 10 <= b.y + b.height + 4) continue;
      sv('line', { class: `th-leader th-leader--${who}`, x1: r1(clamp(p.x, b.x + 6, b.x + b.width - 6)), y1: r1(b.y + b.height + 3), x2: r1(p.x), y2: r1(p.y - 10) }, lab);
    }
    const desc = `${B.short} barrier. With its own margin only, block d sits on the ${w0} part of the loss (${s0}).` +
      (shift ? ` With the inherited margin it moves to the ${w1} part (${s1}).` : ' Nothing is inherited.');
    svg.setAttribute('aria-label', `Loss at block d, illustration. ${desc}`);
  }

  function drawRatio(W, H) {
    const svg = ratioSvg;
    svg.replaceChildren();
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const narrow = W < 420;
    const mg = { l: 50, r: 14, t: 30, b: 46 };
    const x = lin(R_P[0], R_P[1], mg.l, W - mg.r);
    const y = logScale(R_Y[0], R_Y[1], H - mg.b, mg.t);
    const defs = sv('defs', null, svg);
    const pat = sv('pattern', { id: `${NAME}-hatch`, width: 7, height: 7, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, defs);
    sv('line', { class: 'th-hatch', x1: 0, y1: 0, x2: 0, y2: 7 }, pat);
    const clip = sv('clipPath', { id: `${NAME}-clip-ratio` }, defs);
    sv('rect', { x: mg.l, y: mg.t, width: W - mg.l - mg.r, height: H - mg.t - mg.b }, clip);
    // P < 0: outside the theorem (hatched)
    sv('rect', { class: 'th-outside', x: r1(x(R_P[0])), y: mg.t, width: r1(x(0) - x(R_P[0])), height: H - mg.t - mg.b, fill: `url(#${NAME}-hatch)` }, svg);
    for (let k = 2; k >= -8; k -= 2) {
      const yy = r1(y(10 ** k));
      sv('line', { class: k === 0 ? 'th-grid th-grid--one' : 'th-grid', x1: mg.l, x2: W - mg.r, y1: yy, y2: yy }, svg);
      const lbl = k === 0 ? '1' : k === 1 ? '10' : null;
      if (lbl) stext(svg, mg.l - 8, yy + 4.5, lbl, { class: 'th-tick', 'text-anchor': 'end' });
      else stext(svg, mg.l - 8, yy + 4.5, ['10', [(k < 0 ? MINUS : '') + Math.abs(k), { class: 'th-sup', dy: -5 }]], { class: 'th-tick', 'text-anchor': 'end' });
    }
    sv('line', { class: 'th-axis', x1: mg.l, x2: W - mg.r, y1: H - mg.b, y2: H - mg.b }, svg);
    for (let p = R_P[0]; p <= R_P[1]; p += 2) {
      sv('line', { class: 'th-axis', x1: r1(x(p)), x2: r1(x(p)), y1: H - mg.b, y2: H - mg.b + 4 }, svg);
      stext(svg, x(p), H - mg.b + 18, plainNum(p), { class: 'th-tick', 'text-anchor': 'middle' });
    }
    stext(svg, mg.l - 42, 15, [['R', { class: 'th-it' }], ' (log scale)'], { class: 'th-axis-title' });
    stext(svg, (mg.l + W - mg.r) / 2, H - 6, narrow ? ['inherited margin ', ['P', { class: 'th-it' }]] : ['inherited margin ', ['P', { class: 'th-it' }], ' (earlier blocks’ total margin)'], { class: 'th-axis-title', 'text-anchor': 'middle' });
    const g = sv('g', { 'clip-path': `url(#${NAME}-clip-ratio)` }, svg);
    const floorY = H - mg.b;
    // Eq. (6) band, only where the theorem states it: softplus, m >= 0, gamma > 0, P >= 0
    const bandOn = st.barrier === 'softplus' && st.m >= 0 && st.g > 0;
    if (bandOn) {
      let top = '', bot = '';
      const nb = 120;
      for (let i = 0; i <= nb; i++) {
        const p = (R_P[1] * i) / nb, e = Math.exp(-BETA * st.g * p);
        top += `${i ? 'L' : 'M'}${r1(x(p))},${r1(y(Math.max(R_Y[0] / 100, Math.min(1, 2 * e))))}`;
        bot = `L${r1(x(p))},${r1(y(Math.max(R_Y[0] / 100, e)))}` + bot;
      }
      sv('path', { class: 'th-band', d: `${top}${bot}Z` }, g);
      sv('path', { class: 'th-band-edge', d: top }, g);
      sv('path', { class: 'th-band-edge', d: `M${bot.slice(1)}` }, g);
    }
    // R(P) for the reader's m and gamma
    let d = '', on = false, zeroAt = null;
    const n = Math.max(160, Math.round(W / 1.5));
    for (let i = 0; i <= n; i++) {
      const p = R_P[0] + ((R_P[1] - R_P[0]) * i) / n;
      const R = ratio(st.barrier, st.m, p, st.g);
      if (Number.isNaN(R)) continue;
      if (R <= 0) {
        if (zeroAt === null) zeroAt = st.g > 0 ? (THETA - st.m) / st.g : p;
        if (on) { d += `L${r1(x(zeroAt))},${floorY + 2}`; on = false; }
        continue;
      }
      d += `${on ? 'L' : 'M'}${r1(x(p))},${r1(y(clamp(R, R_Y[0] / 50, R_Y[1] * 50)))}`;
      on = true;
    }
    if (d) sv('path', { class: 'th-ratio', d }, g);
    // the reader's point with a drop line to the P axis (position only: the readout below prints the value)
    const R = ratio(st.barrier, st.m, st.P, st.g);
    let edgeNote = null;
    if (!Number.isNaN(R)) {
      const px = x(st.P);
      let py;
      let edge = 0;
      if (R <= 0 || R < R_Y[0]) { py = floorY; edge = -1; } else if (R > R_Y[1]) { py = mg.t; edge = 1; } else py = y(R);
      sv('line', { class: 'th-drop', x1: r1(px), x2: r1(px), y1: r1(py), y2: floorY }, svg);
      if (edge === 0) sv('circle', { class: 'th-pt th-pt--hist', cx: r1(px), cy: r1(py), r: 6.5 }, svg);
      else {
        sv('path', { class: 'th-chev th-chev--hist', d: edge < 0 ? `M${r1(px - 8)},${r1(py - 9)}L${r1(px)},${r1(py - 1)}L${r1(px + 8)},${r1(py - 9)}` : `M${r1(px - 8)},${r1(py + 9)}L${r1(px)},${r1(py + 1)}L${r1(px + 8)},${r1(py + 9)}` }, svg);
        const txt = edge < 0 ? (R <= 0 ? [['R', { class: 'th-it' }], ' = 0'] : ['below 10', [MINUS + '8', { class: 'th-sup', dy: -5 }]]) : ['above 10', ['2', { class: 'th-sup', dy: -5 }]];
        edgeNote = { x: px + (px > W - 110 ? -12 : 12), y: edge < 0 ? py - 14 : py + 22, txt, anchor: px > W - 110 ? 'end' : 'start' };
      }
    }
    // annotations last, on top of the curve, the point and its drop line (each with a surface halo)
    const notes = sv('g', { class: 'th-labels' }, svg);
    const outT = slines(notes, x(R_P[0]) + 6, H - mg.b - (narrow ? 38 : 24), narrow ? [[['P', { class: 'th-it' }], ' < 0:'], ['outside'], ['the theorem']] : [[['P', { class: 'th-it' }], ' < 0: outside'], ['the theorem']], { class: 'th-lbl th-muted th-halo' }, 15);
    const edgeT = edgeNote ? stext(notes, edgeNote.x, edgeNote.y, edgeNote.txt, { class: 'th-lbl th-hist-text th-halo', 'text-anchor': edgeNote.anchor }) : null;
    let bandLabel = null, zeroT = null, gNote = null;
    if (bandOn) {
      const lx = W - mg.r - 6, ly = mg.t + 16;
      bandLabel = slines(notes, lx, ly, [[['Theorem 3.1 band', { class: 'th-strong' }]], [`(Eq.${NB}6)`]], { class: 'th-lbl th-halo', 'text-anchor': 'end' }, 16);
      const pl = clamp(1.6 / Math.max(st.g, 0.05), 0.6, 4.5);
      const by = y(clamp(1.35 * Math.exp(-BETA * st.g * pl), R_Y[0], 1));
      const bb = bbox(bandLabel);
      if (bb) sv('line', { class: 'th-leader', x1: r1(bb.x - 4), y1: r1(bb.y + bb.height * 0.6), x2: r1(x(pl) + 5), y2: r1(by - 3) }, notes);
    }
    if (zeroAt !== null && zeroAt >= R_P[0] && zeroAt <= R_P[1]) {
      sv('line', { class: 'th-zero-tick', x1: r1(x(zeroAt)), x2: r1(x(zeroAt)), y1: floorY - 8, y2: floorY }, notes);
      zeroT = stext(notes, x(zeroAt) + (x(zeroAt) > W - 90 ? -6 : 6), floorY - 10, 'hard zero', { class: 'th-lbl th-muted th-halo', 'text-anchor': x(zeroAt) > W - 90 ? 'end' : 'start' });
      const ob = bbox(outT);
      if (ob && overlap(bbox(zeroT), ob, 4)) {                    // clear of "P < 0: outside the theorem"
        zeroT.setAttribute('x', r1(ob.x + ob.width + 8)); zeroT.setAttribute('text-anchor', 'start');
      }
    }
    if (st.barrier === 'hinge' && st.m >= THETA) {
      slines(notes, (mg.l + W - mg.r) / 2, (mg.t + floorY) / 2 - 8, [[['m', { class: 'th-it' }], ' ≥ θ: no local slope'], ['at this margin']], { class: 'th-lbl th-muted th-halo', 'text-anchor': 'middle' }, 16);
    } else if (st.g === 0) {
      gNote = stext(notes, x(0) + 10, y(1) + 20, ['γ = 0: ', ['R', { class: 'th-it' }], ' = 1 for every ', ['P', { class: 'th-it' }]], { class: 'th-lbl th-muted th-halo' });
      fitX(gNote, mg.l + 4, W - mg.r - 4);
    }
    // the edge note ("above 10²", "below 10⁻⁸", "R = 0") yields to every other label
    if (edgeT && [bandLabel, outT, zeroT, gNote].some((o) => o && overlap(bbox(edgeT), bbox(o), 3))) edgeT.remove();
    const rr = rReadout(R).plain;
    svg.setAttribute('aria-label', `Surviving signal R against the inherited margin P, log scale, illustration for your m and γ (${BAR[st.barrier].short} barrier). At your P, R is ${rr}.` +
      (bandOn ? ' The curve lies inside the Theorem 3.1 band of Eq. 6.' : ''));
  }

  function readoutsA(announce, prefix) {
    const R = ratio(st.barrier, st.m, st.P, st.g);
    const rr = rReadout(R);
    rVal.replaceChildren(...rr.parts);
    const lost = lostReadout(R);
    lostVal.textContent = lost;
    const frac = Number.isNaN(R) ? 0 : (noShift() ? 0 : clamp(1 - Math.min(1, R), 0, 1));
    meterFill.style.setProperty('--w', `${(frac * 100).toFixed(1)}%`);
    regime.textContent = regimeText();
    // measured chip: the printed value that belongs to the current state
    const p = st.preset && presetById[st.preset];
    let kids;
    if (st.barrier === 'hinge') {
      kids = [h('span', { class: 'measured__key' }, 'Measured · App.', NB, 'B.2'),
        h('span', { class: 'measured__val' }, F('s3.hinge_sep#1.67'), ` vs.${NB}`, F('s3.hinge_sep#4.44')),
        h('span', { class: 'measured__src' }, `Squared hinge (${eqs('θ', plainNum(THETA))}): deepest-block separation at ${eqs('γ', '0.7')} vs. ${eqs('γ', '0')}. CIFAR-10 · 4 blocks, width 128 · 3 seeds`)];
    } else if (p) {
      kids = [h('span', { class: 'measured__key' }, 'Measured · App.', NB, 'Tab.', NB, '4'),
        h('span', { class: 'measured__val' }, F(p.R)),
        h('span', { class: 'measured__src' }, `R at the batch means of block ${p.block}, ${eqs('γ', gTxt(p.gamma))} (`, h('i', null, 'm'), `${NB}=${NB}`, F(p.m), ', ', h('i', null, 'P'), `${NB}=${NB}`, F(p.P),
          `). CIFAR-10 · 4 blocks, width 128 · seed 42 · ${eqs('N', N_TEST)}`)];
    } else {
      kids = [h('span', { class: 'measured__key' }, 'Measured · Fig.', NB, '1a'),
        h('span', { class: 'measured__val' }, F('s3.fd_b3')),
        h('span', { class: 'measured__src' }, `Free-riding index at block 3 under ${eqs('γ', '0.7')}: the share of the loss slope lost to history, averaged over test images. CIFAR-10 · 4 blocks, width 128 · seed 42`)];
    }
    measuredA.replaceChildren(...kids);
    if (announce) {
      const rSay = /^(about|more)/.test(rr.plain) ? `R ${rr.plain}` : `R = ${rr.plain}`;
      const msg = Number.isNaN(R) ? `Illustration: ${regimeText()}` : `Illustration: ${rSay}; ${lost === 'none' ? 'none' : lost} of the loss slope lost to history.`;
      liveRegion.textContent = (prefix ? `${prefix} ` : '') + msg;
    }
  }

  function syncInputsA() {
    rP.out.textContent = signed(st.P); rM.out.textContent = signed(st.m); rG.out.textContent = st.g.toFixed(2);
    rP.input.setAttribute('aria-valuetext', `inherited margin P = ${signed(st.P)}`);
    rM.input.setAttribute('aria-valuetext', `own margin m = ${signed(st.m)}`);
    rG.input.setAttribute('aria-valuetext', `history weight γ = ${st.g.toFixed(2)}${Math.abs(st.g) < 1e-9 ? ', history-free' : Math.abs(st.g - 0.7) < 1e-9 ? ', cumulative' : ''}`);
    [rP.input, rM.input, rG.input].forEach(fillPct);
    for (const b of presetBtns) b.setAttribute('aria-pressed', String(b.dataset.preset === st.preset));
  }

  /* ---------------------------------------------------------------------------------------- (B) drawing */
  const FD = [
    { series: 'cum', block: 0, key: 's3.fd_zero#0', v: 0, zero: true },
    { series: 'cum', block: 1, key: 's3.fd_b1', v: val('s3.fd_b1') },
    { series: 'cum', block: 2, key: 's3.fd_b2', v: val('s3.fd_b2') },
    { series: 'cum', block: 3, key: 's3.fd_b3', v: val('s3.fd_b3') },
    ...[0, 1, 2, 3].map((b) => ({ series: 'free', block: b, key: 's3.fd_zero#0', v: 0, zero: true })),
  ];
  let fdFocus = 0;           // roving index into FD
  let fdNodes = [];          // per FD point: {g, label}
  let fdShade = null;
  const fdLabelOf = (p) => (p.series === 'cum'
    ? (p.zero ? `block 0, ${arm('cum').name}: 0, by construction (no earlier blocks)` : `block ${p.block}, ${arm('cum').name}: free-riding index ${fact(p.key).s}`)
    : `block ${p.block}, ${arm('free').name}: 0, by construction`);
  function linkedBlock() {
    const p = st.preset && presetById[st.preset];
    return p ? { block: p.block, series: p.gamma > 0 ? 'cum' : 'free' } : null;
  }

  function drawFd(W, H) {
    const svg = fdSvg;
    const hadFocus = fdSvg.contains(document.activeElement);
    svg.replaceChildren();
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const wide = W >= 470;
    const mg = { l: 40, r: wide ? 128 : 14, t: 34, b: 36 };
    const band = (W - mg.l - mg.r) / 4;
    const X = (b) => mg.l + band * (b + 0.5);
    const y = lin(0, 1, H - mg.b, mg.t);
    fdShade = sv('rect', { class: 'th-col', x: 0, y: 4, width: Math.max(0, r1(band - 8)), height: r1(H - mg.b - 4), rx: 8, visibility: 'hidden' }, svg);
    for (const t of [0.5, 1]) sv('line', { class: 'th-grid', x1: mg.l, x2: W - mg.r, y1: r1(y(t)), y2: r1(y(t)) }, svg);
    sv('line', { class: 'th-axis', x1: mg.l, x2: W - mg.r, y1: r1(y(0)), y2: r1(y(0)) }, svg);
    for (const t of [0, 0.5, 1]) stext(svg, mg.l - 8, y(t) + 4.5, plainNum(t), { class: 'th-tick', 'text-anchor': 'end' });
    for (let b = 0; b < 4; b++) stext(svg, X(b), H - mg.b + 20, `block ${b}`, { class: 'th-xlab', 'text-anchor': 'middle' });
    stext(svg, mg.l - 32, 15, [['F', { class: 'th-it' }], ['d', { class: 'th-it th-sub', dy: 4 }], ['', { dy: -4 }]], { class: 'th-axis-title' });
    // lines: history-free flat at 0 (by construction), cumulative 0 -> 0.930 -> 0.952 -> 0.955
    sv('path', { class: 'th-line th-line--free', d: `M${r1(X(0))},${r1(y(0))}H${r1(X(3))}` }, svg);
    sv('path', { class: 'th-line th-line--cum', d: FD.filter((p) => p.series === 'cum').map((p, i) => `${i ? 'L' : 'M'}${r1(X(p.block))},${r1(y(p.v))}`).join('') }, svg);
    // value labels above the cumulative points (printed strings), end labels on wide charts
    const vals = sv('g', null, svg);
    const valText = {};
    for (const p of FD) if (p.series === 'cum' && !p.zero) {
      const t = stext(vals, X(p.block), y(p.v) - 13, fact(p.key).s, { class: 'th-val', 'text-anchor': 'middle', 'data-fact': p.key });
      valText[p.block] = t;
    }
    if (wide) {
      const ex = X(3) + band / 2 + 2;
      stext(svg, ex, y(val('s3.fd_b3')) + 4.5, arm('cum').short || 'cumulative', { class: 'th-lbl' });
      slines(svg, ex, y(0) - 22, [[arm('free').short || 'history-free'], [['0 by construction', { class: 'th-muted' }]]], { class: 'th-lbl' }, 16);
    } else {
      stext(svg, (X(1) + X(2)) / 2, y(0) - 10, '0 by construction', { class: 'th-lbl th-muted th-halo', 'text-anchor': 'middle' });
    }
    // points: one roving Tab stop; each point is focusable with its own accessible label
    fdNodes = FD.map((p, i) => {
      const dx = p.block === 0 ? (p.series === 'cum' ? 6 : -6) : 0;
      const cx = X(p.block) + dx, cy = y(p.v);
      const gp = sv('g', { class: `th-fdpt th-fdpt--${p.series}`, tabindex: i === fdFocus ? 0 : -1, role: 'img', 'aria-label': fdLabelOf(p), 'data-i': i }, svg);
      const half = p.block === 0 ? (band - 4) / 2 : band - 4;          // block 0: the two points share the column
      const hx = X(p.block) - band / 2 + 2 + (p.block === 0 && p.series === 'cum' ? half : 0);
      sv('rect', { class: 'th-hit', x: r1(hx), y: r1(Math.min(cy, y(0)) - (p.series === 'cum' ? 26 : 14)), width: r1(half), height: p.series === 'cum' && !p.zero ? 40 : 28 }, gp);
      sv('circle', { class: 'th-focusring', cx: r1(cx), cy: r1(cy), r: 11 }, gp);
      const kind = p.series === 'cum' ? 'square' : 'circle', r = p.series === 'cum' ? 5 : 5.2;
      const mark = sv('path', { class: `th-mark m--${p.series}`, d: shapePath(kind, cx, cy, r) }, gp);
      return { g: gp, mark, kind, r, cx, cy, p, label: valText[p.block] && p.series === 'cum' ? valText[p.block] : null };
    });
    shadeFd(X, band);
    if (hadFocus && fdNodes[fdFocus]) fdNodes[fdFocus].g.focus({ preventScroll: true });
  }
  let fdGeom = null;
  function shadeFd(X, band) {
    if (X) fdGeom = { X, band };
    if (!fdShade || !fdGeom) return;
    const lb = linkedBlock();
    if (lb) {
      fdShade.setAttribute('x', r1(fdGeom.X(lb.block) - fdGeom.band / 2 + 4));
      fdShade.setAttribute('visibility', 'visible');
    } else fdShade.setAttribute('visibility', 'hidden');
    for (const n of fdNodes) {
      const on = !!lb && n.p.block === lb.block && n.p.series === lb.series;
      n.g.classList.toggle('is-linked', on);
      n.mark.setAttribute('d', shapePath(n.kind, n.cx, n.cy, on ? n.r * 1.4 : n.r));   // the linked point is drawn larger
      if (n.label) n.label.classList.toggle('th-val--on', !!lb && lb.series === 'cum' && n.p.block === lb.block);
    }
    const p = st.preset && presetById[st.preset];
    if (!p) fdNote.replaceChildren('Load a batch-mean preset above to mark its block here.');
    else if (p.gamma > 0) fdNote.replaceChildren(`Shaded: block ${p.block}, whose batch means are loaded above (${eqs('γ', gTxt(p.gamma))}).`);
    else fdNote.replaceChildren(`Shaded: block ${p.block} of the history-free run (${eqs('γ', '0')}), loaded above: nothing is inherited, so its index is 0 by construction.`);
  }
  function showTip(i) {
    const n = fdNodes[i];
    if (!n) return;
    const p = n.p;
    const rows = p.series === 'cum'
      ? (p.zero ? ['0', `block 0 · ${arm('cum').name}`, 'no earlier blocks: 0 by construction'] : [fact(p.key).s, `block ${p.block} · ${arm('cum').name}`, `recomputing on EMA weights changes it by ${fact('s3.fd_ema_change').s}`])
      : ['0', `block ${p.block} · ${arm('free').name}`, 'nothing is inherited: 0 by construction'];
    tip.replaceChildren(h('span', { class: 'tip__v' }, rows[0]), h('span', { class: 'tip__l' }, rows[1]), h('span', { class: 'tip__l' }, rows[2]));
    tip.hidden = false;
    const r = n.g.querySelector('.th-mark').getBoundingClientRect();
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    const left = clamp(r.left + r.width / 2 - tw / 2, 6, document.documentElement.clientWidth - tw - 6);
    const top = r.top - th - 12 < 6 ? r.bottom + 12 : r.top - th - 12;
    tip.style.left = `${Math.round(left)}px`;
    tip.style.top = `${Math.round(top)}px`;
    fdNodes.forEach((m, j) => m.g.classList.toggle('is-active', j === i));
  }
  function hideTip() {
    tip.hidden = true;
    fdNodes.forEach((m) => m.g.classList.remove('is-active'));
  }
  function moveFd(i) {
    fdFocus = clamp(i, 0, FD.length - 1);
    fdNodes.forEach((n, j) => n.g.setAttribute('tabindex', j === fdFocus ? 0 : -1));
    fdNodes[fdFocus].g.focus({ preventScroll: true });
    reveal(fdNodes[fdFocus].g);
    showTip(fdFocus);
  }
  fdSvg.addEventListener('keydown', safe((e) => {
    const cur = FD[fdFocus];
    const at = (series, block) => FD.findIndex((p) => p.series === series && p.block === block);
    let next = null;
    switch (e.key) {
      case 'ArrowRight': next = at(cur.series, Math.min(3, cur.block + 1)); break;
      case 'ArrowLeft': next = at(cur.series, Math.max(0, cur.block - 1)); break;
      case 'ArrowUp': next = at('cum', cur.block); break;
      case 'ArrowDown': next = at('free', cur.block); break;
      case 'Home': next = at(cur.series, 0); break;
      case 'End': next = at(cur.series, 3); break;
      case 'Escape': hideTip(); e.preventDefault(); return;
      default: return;
    }
    e.preventDefault();
    moveFd(next);
  }));
  fdSvg.addEventListener('focusin', safe((e) => {
    const g = e.target.closest && e.target.closest('.th-fdpt');
    if (g) { fdFocus = +g.dataset.i; fdNodes.forEach((n, j) => n.g.setAttribute('tabindex', j === fdFocus ? 0 : -1)); reveal(g); showTip(fdFocus); }
  }));
  fdSvg.addEventListener('focusout', safe((e) => { if (!fdSvg.contains(e.relatedTarget)) hideTip(); }));
  fdSvg.addEventListener('pointerover', safe((e) => {
    const g = e.target.closest && e.target.closest('.th-fdpt');
    if (g) showTip(+g.dataset.i); else if (!fdSvg.contains(document.activeElement)) hideTip();
  }));
  fdSvg.addEventListener('pointerleave', safe(() => { if (!fdSvg.contains(document.activeElement)) hideTip(); }));
  window.addEventListener('scroll', safe(() => { if (!tip.hidden && !fdSvg.contains(document.activeElement)) hideTip(); else if (!tip.hidden) showTip(fdFocus); }), { passive: true });

  /* ---------------------------------------------------------------------------------------- (C) drawing */
  const G_X = [-6, 10];
  const gateW = (S, k, t) => GAMMA0 / (1 + Math.exp(-t * (k - S)));
  function drawGate(W, H) {
    const svg = gateSvg;
    svg.replaceChildren();
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const mg = { l: 44, r: 14, t: 28, b: 46 };
    const x = lin(G_X[0], G_X[1], mg.l, W - mg.r);
    // the axis title and the "threshold κ" label share the top band; on narrow charts they would collide,
    // so the plot moves down one row to give the label its own line
    const kx = x(st.kappa);
    const titleT = stext(svg, mg.l - 36, 15, 'inherited weight', { class: 'th-axis-title' });
    const kT = stext(svg, kx, 0, ['threshold ', ['κ', { class: 'th-it' }]], { class: 'th-lbl th-strong th-halo', 'text-anchor': kx < mg.l + 50 ? 'start' : kx > W - mg.r - 50 ? 'end' : 'middle' });
    const tb = bbox(titleT), kb = bbox(kT);
    if (tb && kb && kb.x < tb.x + tb.width + 8) mg.t += 16;
    kT.setAttribute('y', r1(mg.t - 2));
    const y = lin(0, GAMMA0, H - mg.b, mg.t + 14);
    sv('line', { class: 'th-grid', x1: mg.l, x2: W - mg.r, y1: r1(y(GAMMA0 / 2)), y2: r1(y(GAMMA0 / 2)) }, svg);
    sv('line', { class: 'th-axis', x1: mg.l, x2: W - mg.r, y1: r1(y(0)), y2: r1(y(0)) }, svg);
    stext(svg, mg.l - 8, y(0) + 4.5, '0', { class: 'th-tick', 'text-anchor': 'end' });
    stext(svg, mg.l - 8, y(GAMMA0) + 4.5, plainNum(GAMMA0), { class: 'th-tick', 'text-anchor': 'end' });
    stext(svg, mg.l, H - mg.b + 18, '← hard', { class: 'th-axis-title' });
    stext(svg, W - mg.r, H - mg.b + 18, 'easy →', { class: 'th-axis-title', 'text-anchor': 'end' });
    stext(svg, (mg.l + W - mg.r) / 2, H - 6, 'earlier blocks’ goodness for this input', { class: 'th-axis-title', 'text-anchor': 'middle' });
    // threshold guide (its label is placed above, with the axis title)
    sv('line', { class: 'th-kappa', x1: r1(kx), x2: r1(kx), y1: r1(mg.t + 2), y2: r1(y(0)) }, svg);
    // constant rules, then the gate on top
    sv('path', { class: 'th-line th-line--cum', d: `M${mg.l},${r1(y(GAMMA0))}H${W - mg.r}` }, svg);
    sv('path', { class: 'th-line th-line--free th-line--under', d: `M${mg.l},${r1(y(0))}H${W - mg.r}` }, svg);
    let d = '';
    const n = Math.max(120, Math.round(W / 2));
    for (let i = 0; i <= n; i++) {
      const S = G_X[0] + ((G_X[1] - G_X[0]) * i) / n;
      d += `${i ? 'L' : 'M'}${r1(x(S))},${r1(y(gateW(S, st.kappa, st.tau)))}`;
    }
    sv('path', { class: 'th-line th-line--gated', d }, svg);
    sv('path', { class: 'th-mark m--cum', d: shapePath('square', W - mg.r, y(GAMMA0), 5) }, svg);
    sv('path', { class: 'th-mark m--free', d: shapePath('circle', W - mg.r, y(0), 5.2) }, svg);
    sv('path', { class: 'th-mark m--gated', d: shapePath('triangle', kx, y(GAMMA0 / 2), 5) }, svg);
    // annotations in the paper's words
    const notes = sv('g', { class: 'th-labels' }, svg);
    const gl = stext(notes, x(st.kappa) + 12, y(GAMMA0 / 2) - 8, 'hardness gate', { class: 'th-lbl th-strong th-halo' });
    const gb = bbox(gl);
    if (gb && gb.x + gb.width > W - mg.r - 2) {           // no room on the right: below-left of the triangle
      gl.setAttribute('x', r1(x(st.kappa) - 12)); gl.setAttribute('y', r1(y(GAMMA0 / 2) + 22)); gl.setAttribute('text-anchor', 'end');
    }
    // the two end annotations are drawn only where the gate curve stays clear of them (it can cross them for a
    // threshold near the ends of the axis or a soft gate)
    const sAt = (px) => G_X[0] + ((px - mg.l) / (W - mg.r - mg.l)) * (G_X[1] - G_X[0]);
    const clear = (t) => {
      const b = bbox(t);
      if (!b) return true;
      for (let px = b.x - 2; px <= b.x + b.width + 2; px += 2) {
        const cy = y(gateW(sAt(px), st.kappa, st.tau));
        if (cy >= b.y - 2 && cy <= b.y + b.height + 2) return false;
      }
      return true;
    };
    const leftRoom = x(st.kappa) - mg.l;
    if (leftRoom > 120) {
      const t = slines(notes, mg.l + 8, y(GAMMA0) + 26, [['hard: inherits'], [['up to γ', {}], ['0', { class: 'th-sub', dy: 4 }]]], { class: 'th-lbl th-halo' }, 16);
      if (!clear(t)) t.remove();
    }
    const rightRoom = W - mg.r - x(st.kappa);
    if (rightRoom > 130) {
      const t = slines(notes, W - mg.r - 8, y(0) - 30, [['easy: inherits'], ['little']], { class: 'th-lbl th-halo', 'text-anchor': 'end' }, 16);
      if (!clear(t)) t.remove();
    }
    const c = ranCond(st.kappa, st.tau);
    svg.setAttribute('aria-label', `Illustration: inherited weight against the earlier blocks’ goodness for an input. Cumulative keeps γ = 0.7 for every input; history-free keeps 0; the hardness gate with threshold κ = ${plainNum(st.kappa)} and sharpness τ = ${plainNum(st.tau)} gives hard inputs up to γ0 and easy inputs little${c ? ' (a condition the paper ran)' : ''}.`);
  }

  function runsCardUpdate(announce) {
    const c = ranCond(st.kappa, st.tau);
    const cond = `${eqs('κ', plainNum(st.kappa))}, ${eqs('τ', plainNum(st.tau))}`;
    const head = h('span', { class: 'measured__key' }, 'Measured · App.', NB, 'Tab.', NB, '16 (seed', NB, '42)');
    const rows = [];
    const row = (armKey, label, sepKey, accKey, current) => h('tr', { class: current ? 'is-current' : null },
      h('th', { scope: 'row' }, armChip(armKey, label)), h('td', { class: 'num', 'data-label': 'Block-3 separation' }, F(sepKey)), h('td', { class: 'num', 'data-label': 'Stage-1 accuracy' }, F(accKey)));
    if (c) rows.push(row('gated', `hardness-gated, ${cond}${c.reported ? ' (reported)' : ''}`, c.sep, c.acc, true));
    for (const r of gate.reference) rows.push(row(r.arm, r.label, r.sep, r.acc, false));
    const table = h('table', { class: 'tbl th-runs__tbl' },
      h('caption', { class: 'vh' }, 'Block-3 separation and Stage-1 accuracy (FF’s own sum, single crop) of the gate setting and the two reference rules'),
      h('thead', null, h('tr', null, h('th', { scope: 'col' }, 'Run'), h('th', { scope: 'col', class: 'num' }, 'Block-3 separation'), h('th', { scope: 'col', class: 'num' }, 'Stage-1 accuracy'))),
      h('tbody', null, rows));
    const state = c
      ? h('p', { class: 'th-runs__state' }, h('b', null, cond), c.reported ? ': the gate the paper reports.' : ': a condition the paper ran.')
      : h('p', { class: 'th-runs__state th-runs__state--none' }, h('b', null, cond), ': not a condition the paper ran: no measured outcome.');
    const kids = [head, state, table, h('p', { class: 'th-runs__note' }, 'Stage-1 accuracy: FF’s own sum, single crop. Higher block-3 separation is healthier.'),
      h('p', { class: 'th-runs__note' }, `The constant ${eqs('γ', '0.7')} run also carries the depth-scaled auxiliary and λ`, h('sub', null, 'depth'),
        `${NB}=${NB}0.2, which the ${eqs('γ', '0')} run lacks (Fig.${NB}1 caption), so the two differ in more than γ.`)];
    if (c && c.reported) {
      kids.push(h('p', { class: 'th-runs__quote' }, '“…this improved block health does not translate to higher accuracy” ', h('span', { class: 'th-runs__cite' }, '(App.', NB, 'C, against constant γ', NB, '=', NB, '0.7)')));
    }
    if (c && c.kappa === 4) {
      kids.push(h('p', { class: 'th-runs__quote' }, '“High-threshold gating (κ=4) recreates the constant-γ collapse, confirming that κ controls a sharp phase transition (Table 16).” ', h('span', { class: 'th-runs__cite' }, '(App.', NB, 'C)')));
    }
    if (!c) {
      kids.push(h('p', { class: 'th-runs__note' }, `The paper ran ${eqs('κ', '0')} (${eqs('τ', '1')} and 3), ${eqs('κ', '2')} and ${eqs('κ', '4')} (${eqs('τ', '1')}).`));
    }
    kids.push(h('span', { class: 'measured__src' }, 'CIFAR-10 · 4 blocks, width 128 · seed 42 · single seed, reduced scale · App.', NB, 'Tab.', NB, '16'));
    runsCard.replaceChildren(...kids);
    rK.out.textContent = plainNum(st.kappa);
    rT.out.textContent = plainNum(st.tau);
    const ran = (v, list) => list.includes(v);
    rK.input.setAttribute('aria-valuetext', `threshold κ = ${plainNum(st.kappa)}${ran(st.kappa, [0, 2, 4]) ? ' (the paper ran this threshold)' : ''}`);
    rT.input.setAttribute('aria-valuetext', `sharpness τ = ${plainNum(st.tau)}${ran(st.tau, [1, 3]) ? ' (the paper ran this sharpness)' : ''}`);
    [rK.input, rT.input].forEach(fillPct);
    if (announce) {
      liveRegion.textContent = c
        ? `Gate ${cond.replace(/\u00a0/g, ' ')}, a condition the paper ran: block-3 separation ${fact(c.sep).s}, Stage-1 accuracy ${fact(c.acc).s}.`
        : `Gate ${cond.replace(/\u00a0/g, ' ')}: not a condition the paper ran; no measured outcome.`;
    }
  }

  /* ---------------------------------------------------------------------------------------- wiring */
  root.dataset.state = 'live';    // [data-live] becomes visible, so the charts can be measured
  const lossSize = onSize(lossPlot, safe(drawLoss));
  const ratioSize = onSize(ratioPlot, safe(drawRatio));
  const fdSize = onSize(fdPlot, safe(drawFd));
  const gateSize = onSize(gatePlot, safe(drawGate));
  let raf = 0;
  const redrawA = () => { lossSize.redraw(); ratioSize.redraw(); };
  const scheduleA = () => { if (!raf) raf = requestAnimationFrame(safe(() => { raf = 0; redrawA(); })); };

  function applyPreset(id, announce) {
    const p = presetById[id];
    if (!p) return;
    st.preset = id; st.m = val(p.m); st.P = val(p.P); st.g = p.gamma;
    rP.input.value = st.P; rM.input.value = st.m; rG.input.value = st.g;
    syncInputsA();
    redrawA();
    readoutsA(announce, announce ? `Loaded the batch means of block ${p.block} (γ = ${gTxt(p.gamma)}).` : '');
    shadeFd();
  }
  const onRange = (announce) => safe(() => {
    st.P = +rP.input.value; st.m = +rM.input.value; st.g = +rG.input.value;
    st.preset = null;                       // moving any slider clears the pressed preset
    syncInputsA();
    readoutsA(announce);
    shadeFd();
    if (announce) { if (raf) { cancelAnimationFrame(raf); raf = 0; } redrawA(); } else scheduleA();
  });
  for (const r of [rP, rM, rG]) { r.input.addEventListener('input', onRange(false)); r.input.addEventListener('change', onRange(true)); }
  seg.addEventListener('change', safe((e) => {
    if (!e.target || e.target.name !== `${NAME}-barrier`) return;
    st.barrier = e.target.value;
    ratioSubText();
    redrawA();
    readoutsA(true, st.barrier === 'hinge' ? `Squared hinge, θ = ${plainNum(THETA)}.` : `Softplus, β = ${plainNum(BETA)}.`);
  }));
  presetBtns.forEach((b) => b.addEventListener('click', safe(() => applyPreset(b.dataset.preset, true))));

  let rafC = 0;
  const onGate = (announce) => safe(() => {
    st.kappa = +rK.input.value; st.tau = +rT.input.value;
    runsCardUpdate(announce);
    if (announce) { if (rafC) { cancelAnimationFrame(rafC); rafC = 0; } gateSize.redraw(); } else if (!rafC) rafC = requestAnimationFrame(safe(() => { rafC = 0; gateSize.redraw(); }));
  });
  for (const r of [rK, rT]) { r.input.addEventListener('input', onGate(false)); r.input.addEventListener('change', onGate(true)); }
  root.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !tip.hidden) hideTip(); });

  // phones (< 640 px): presets and barrier on top, then the loss chart, the sliders, the readouts and the ratio chart,
  // so that dragging a slider shows the tangent above it and R below it; wider screens keep the paper-like layout
  // (controls, both charts side by side, readouts). Focus is kept when nodes move.
  const phoneRanges = h('div', { class: 'controls th-controls th-controls--phone' });
  const ratioHolder = h('div', { class: 'th-charts th-charts--ratio', 'data-illus': '' });
  const phoneA = window.matchMedia('(max-width: 639px)');
  const placeA = safe(() => {
    const a = document.activeElement;
    const had = a && (rangesA.contains(a) || tray.contains(a));
    if (phoneA.matches) {
      if (rangesA.parentNode !== phoneRanges) {
        phoneRanges.append(rangesA);
        figA.insertBefore(phoneRanges, charts.nextSibling);
        figA.insertBefore(tray, phoneRanges.nextSibling);
        ratioHolder.append(ratioChart);
        figA.insertBefore(ratioHolder, tray.nextSibling);
      }
    } else if (rangesA.parentNode !== controlsA) {
      controlsA.append(rangesA);
      phoneRanges.remove();
      charts.append(ratioChart);
      ratioHolder.remove();
      figA.insertBefore(tray, capA);
    }
    if (had) a.focus({ preventScroll: true });
  });
  placeA();
  phoneA.addEventListener('change', placeA);

  /* automatic sweep (no button), like a short clip: the first time the loss chart is at least 60 % in view, and again
     each time it comes back after leaving the screen, the inherited margin P sweeps from 0 to the default batch mean
     (block 3, γ = 0.7: P = +5.41, Tab. 4) in about 3.5 s, so the tangent flattens and R falls on the log scale while
     the reader watches; it ends exactly on that preset (state, sliders, readouts, the shaded F_d block). It stops for
     good once the reader touches the figure (pointer or key), and it never runs with reduced motion, without
     IntersectionObserver, or in automated test browsers (navigator.webdriver: the component tests check the stable
     default state). The live region stays silent during the sweep. */
  if ('IntersectionObserver' in window && !navigator.webdriver) {
    const still = window.matchMedia('(prefers-reduced-motion: reduce)');
    let rafS = 0;
    let touched = false;
    let away = true;
    const stopSweep = (restore) => {
      if (!rafS) return;
      cancelAnimationFrame(rafS);
      rafS = 0;
      if (restore) applyPreset(def.id, false);
    };
    const sweep = safe(() => {
      if (touched || still.matches || rafS) return;
      const P1 = val(def.P);
      const T = 3500;
      let t0 = 0;
      applyPreset(def.id, false);
      const step = safe((now) => {
        if (!t0) t0 = now;
        const u = Math.min(1, (now - t0) / T);
        const e = u < 0.5 ? 2 * u * u : 1 - ((-2 * u + 2) ** 2) / 2;   // ease in and out
        st.P = Math.round(P1 * e * 100) / 100;
        st.preset = null;
        rP.input.value = st.P;
        syncInputsA();
        readoutsA(false);
        shadeFd();
        redrawA();
        if (u < 1) rafS = requestAnimationFrame(step);
        else { rafS = 0; applyPreset(def.id, false); }
      });
      rafS = requestAnimationFrame(step);
    });
    const touch = () => { touched = true; stopSweep(false); };
    figA.addEventListener('pointerdown', touch, true);
    figA.addEventListener('keydown', touch, true);
    new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting && e.intersectionRatio >= 0.6) { if (away) { away = false; sweep(); } }
        else if (!e.isIntersecting) { away = true; stopSweep(true); }
      }
    }, { threshold: [0, 0.6] }).observe(lossPlot);
  }

  // first render (synchronous, so a failure shows up here and the static version comes back)
  ratioSubText();
  syncInputsA();
  readoutsA(false);
  runsCardUpdate(false);
  lossSize.now(); ratioSize.now(); fdSize.now(); gateSize.now();
  shadeFd();
}

boot();
