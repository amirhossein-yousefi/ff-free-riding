// Component "cumulative" (SPEC_FINAL.md §5.1): how cumulative goodness works. ES module, no imports.
//   (A) Forward-Forward in one picture (toy numbers): backpropagation vs. FF strips; toy goodness per block for
//       the chosen candidate label; prediction = the label with the largest sum; the true label's per-block
//       margins (the toy margins that frame B feeds into Eq. (2)).
//   (B) A training score with history (illustration of Eq. (2)): the reader's γ splits each block's toy training
//       margin into own + inherited; presets 0 / 0.7 / 1 also select that run in (C); a measured chip shows
//       block 3's printed own share for the run the slider sits on, and says so when no trained run has that γ.
//       On phones (< 640 px) the presets move under the readouts, so the slider sits right above its chart.
//   (C) Free-riding, pictured (measured, App. Fig. 6, Tab. 16, App. C): own vs. inherited positive goodness per
//       block for the four history rules, a block-3 reading per block (hover, tap, or arrow keys), and block 3's
//       own share in all four runs as 100% bars.
// Contract (SPEC_FINAL §4.4): render into [data-live] of #viz-cumulative with textContent only, move the
// adoptable equation #eq-2 to the top of frame (B), then set data-state="live". On any error: console.error,
// put #eq-2 back, data-state="broken" (the static fallback returns). Nothing outside the root is touched;
// colors come from CSS classes bound to the site tokens, so theme changes need no work here.
// Numbers: measured values only from SITE_DATA.cumulative.facts[id].s; toy and illustration values only
// inside [data-illus]. Geometry (bar totals) positions marks and is never printed.
// Key states (tests, screenshots): default, label-dog, gamma-0, gamma-1-preset, rule-gated.
const NAME = 'cumulative';
const SVGNS = 'http://www.w3.org/2000/svg';
const BLOCKS = [0, 1, 2, 3];

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const toy = (x) => {             // toy values: at most two decimals, trailing zeros dropped, U+2212 minus
  const r = Math.round(x * 100) / 100;
  return r.toFixed(2).replace(/0+$/, '').replace(/\.$/, '').replace('-', '−');
};
const gam = (g100) => (g100 / 100).toFixed(2);
// typography (SPEC_FINAL §3.2): no line break inside "γ = 0.7", "Eq. (2)", "Tab. 16", "App. C", "block 3", "seed 42"
const NB = '\u00a0';
const nb = (s) => String(s)
  .replace(/ = /g, `${NB}=${NB}`)
  .replace(/\b(Eq\.|Tab\.|App\.|Fig\.|Sec\.|Stage|seed|blocks?|width) (?=[\d(A-Z])/g, `$1${NB}`);

function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    n.setAttribute(k, v === true ? '' : String(v));
  }
  for (const c of kids.flat(Infinity)) if (c !== null && c !== undefined && c !== false) n.append(typeof c === 'string' ? nb(c) : c);
  return n;
}
function sv(tag, attrs = {}, parent = null) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined && v !== false) n.setAttribute(k, String(v));
  if (parent) parent.appendChild(n);
  return n;
}
function tx(parent, x, y, str, attrs = {}) {
  const t = sv('text', { x: r1(x), y: r1(y), ...attrs }, parent);
  t.textContent = str;
  return t;
}
const r1 = (v) => Math.round(v * 10) / 10;
function clear(n) { while (n.firstChild) n.removeChild(n.firstChild); }
function textWidth(t, fallbackPerChar = 7.2) {
  let w = 0;
  try { w = t.getComputedTextLength(); } catch (_) { w = 0; }
  return w > 0 ? w : t.textContent.length * fallbackPerChar;
}
// rectangle path with rounded top corners (data end) and a square baseline (dataviz mark spec)
function barPath(x, yTop, w, yBase, r = 4) {
  const h = yBase - yTop;
  if (h <= 0.2 || w <= 0) return '';
  const rr = Math.min(r, h, w / 2);
  return `M${r1(x)},${r1(yBase)}V${r1(yTop + rr)}Q${r1(x)},${r1(yTop)} ${r1(x + rr)},${r1(yTop)}` +
    `H${r1(x + w - rr)}Q${r1(x + w)},${r1(yTop)} ${r1(x + w)},${r1(yTop + rr)}V${r1(yBase)}Z`;
}
function arrowHead(parent, x, y, dir, cls, size = 5) {      // small filled triangle pointing in dir
  const s = size;
  const pts = {
    right: [[x, y], [x - s * 1.4, y - s], [x - s * 1.4, y + s]],
    left: [[x, y], [x + s * 1.4, y - s], [x + s * 1.4, y + s]],
    up: [[x, y], [x - s, y + s * 1.4], [x + s, y + s * 1.4]],
    down: [[x, y], [x - s, y - s * 1.4], [x + s, y - s * 1.4]],
  }[dir];
  return sv('path', { d: `M${pts.map((p) => p.map(r1).join(',')).join('L')}Z`, class: cls }, parent);
}
// width-driven redraw: charts re-lay out at their container's width (ResizeObserver, rAF-throttled)
function onWidth(node, draw, guard) {
  let last = 0, raf = 0;
  const run = guard(() => {
    raf = 0;
    const w = Math.floor(node.clientWidth);
    if (w > 0 && w !== last) { last = w; draw(w); }
  });
  if ('ResizeObserver' in window) {
    new ResizeObserver(() => { if (!raf) raf = requestAnimationFrame(run); }).observe(node);
  } else {
    window.addEventListener('resize', () => { if (!raf) raf = requestAnimationFrame(run); });
  }
  run();
  return { redraw: () => { last = 0; run(); } };
}
// size-driven redraw for a box whose height the layout decides (frame A's drawing fills the row it shares with
// frame B): redraw when the width or the height changes; draw() may set the box's reserved height itself, and
// the size read after it is remembered, so that change does not trigger another redraw
function onBox(node, draw, guard) {
  let lw = 0, lh = 0, raf = 0;
  const run = guard(() => {
    raf = 0;
    const w = Math.floor(node.clientWidth), h = Math.floor(node.clientHeight);
    if (w > 0 && (w !== lw || h !== lh)) { lw = w; draw(w); lh = Math.floor(node.clientHeight); }
  });
  if ('ResizeObserver' in window) {
    new ResizeObserver(() => { if (!raf) raf = requestAnimationFrame(run); }).observe(node);
  } else {
    window.addEventListener('resize', () => { if (!raf) raf = requestAnimationFrame(run); });
  }
  run();
  return { redraw: () => { lw = 0; run(); } };
}
function glyph(site, arm) {
  const a = site.arms[arm];
  const g = sv('svg', { class: `glyph ${a.glyph}`, 'aria-hidden': 'true', focusable: 'false' });
  sv('use', { href: `#${site.shapes[a.shape]}` }, g);
  return g;
}
function swatch(cls) {                     // legend swatch mirroring the mark (rect for bars)
  const g = sv('svg', { class: 'cum-sw', viewBox: '0 0 14 14', 'aria-hidden': 'true', focusable: 'false' });
  sv('rect', { x: 1, y: 1, width: 12, height: 12, rx: 2, class: cls }, g);
  return g;
}

function boot() {
  const root = document.getElementById(`viz-${NAME}`);
  if (!root) return;
  const live = root.querySelector(':scope > [data-live]');
  const eq = root.querySelector(':scope > [data-adopt="b"]');
  const eqNext = eq ? eq.nextElementSibling : null;
  let failed = false;
  const fail = (err) => {
    console.error(`[${NAME}]`, err);
    if (failed) return;
    failed = true;
    try {
      if (eq && eq.parentNode !== root) root.insertBefore(eq, eqNext && eqNext.parentNode === root ? eqNext : null);
    } catch (_) { /* keep the static version */ }
    root.dataset.state = 'broken';
  };
  const guard = (fn) => function guarded(...args) {
    if (failed) return undefined;
    try { return fn.apply(this, args); } catch (err) { fail(err); return undefined; }
  };
  try {
    if (!live) throw new Error('[data-live] is missing in #viz-cumulative');
    mount(root, live, eq, guard);
  } catch (err) {
    fail(err);
  }
}

function mount(root, live, eq, guard) {
  const D = (window.SITE_DATA || {})[NAME];
  const SITE = (window.SITE_DATA || {}).site;
  if (!D) throw new Error(`window.SITE_DATA.${NAME} is missing (assets/data/${NAME}.js)`);
  if (!SITE || !SITE.toy || !SITE.arms) throw new Error('window.SITE_DATA.site is missing (assets/data/site_data.js)');
  const F = (k) => {
    const e = D.facts[k];
    if (!e || typeof e.s !== 'string') throw new Error(`fact ${k} is missing from SITE_DATA.${NAME}`);
    return e.s;
  };
  const T = SITE.toy;
  const BEST = T.labels.reduce((a, b) => (T.sums[b] > T.sums[a] ? b : a), T.labels[0]);   // FF's toy prediction
  const RULES = Object.fromEntries(D.rules.map((r) => [r.key, r]));
  const GEO = D.geometry;
  const G = D.toy.gamma;
  const state = { label: T.true_label, g100: Math.round(G.value * 100), rule: D.rules[0].key, active: -1 };

  // ------------------------------------------------------------------ copy (numbers only from facts)
  const ruleName = (k) => SITE.arms[RULES[k].arm].name;
  const captionC = (k) => {
    const r = RULES[k];
    if (k === 'cum') return `In the cumulative model, block 3 makes only ${F(r.share[3])} of its training score itself: own goodness ${F(r.b3_own)} of ${F(r.b3_total)}.`;
    if (k === 'gated') return `Under the hardness gate, block 3 makes ${F(r.share[3])} of it: ${F(r.b3_own)} of ${F(r.b3_total)}.`;
    if (k === 'free') return `Without history, every block’s score is its own (${F(r.share[3])}).`;
    return `With γ = 1 and no auxiliary, block 3 makes ${F(r.share[3])} of its score itself.`;
  };
  const marginsA = (lab) => {      // the bridge to frame (B): per-block margins of the true label (toy)
    const vs = lab === T.true_label ? T.margins_vs : lab;
    const m = T.goodness[T.true_label].map((g, d) => g - T.goodness[vs][d]);
    const lead = lab === T.true_label ? `Margins of “${T.true_label}” over the hardest wrong label, “${vs}”` : `Margins of “${T.true_label}” over “${vs}”`;
    return `${lead}: ${m.map(toy).join(', ')}${vs === T.margins_vs ? ', the toy margins of Eq. (2)' : ''}.`;
  };
  const sentenceA = (lab) => {
    const g = T.goodness[lab];
    return `Toy goodness of “${lab}” at blocks 0–3: ${g.map(toy).join(', ')}; sum ${toy(T.sums[lab])}. ` +
      `FF predicts “${BEST}”, the label with the largest sum.`;
  };
  const toyB = (g100) => {         // Eq. (2) on the toy margins, in hundredths (exact for the 0.05 grid)
    const own = T.margins.map((v) => Math.round(v * 100));
    const pre = own.map((_, d) => own.slice(0, d).reduce((a, b) => a + b, 0));
    const inh = pre.map((p) => Math.round((g100 * p) / 100));
    return { own: own.map((v) => v / 100), inh: inh.map((v) => v / 100) };
  };
  const presetFor = (g100) => D.toy.presets.find((p) => Math.round(p.gamma * 100) === g100) || null;
  const sentenceB = (g100) => {
    const b = toyB(g100);
    return `At γ = ${gam(g100)}, block 3’s toy training margin is ${toy(b.own[3])} of its own plus ${toy(b.inh[3])} inherited.`;
  };

  // ------------------------------------------------------------------ live region (one, polite)
  const liveRegion = el('p', { class: 'vh', 'aria-live': 'polite', id: `${NAME}-live` });
  const announce = (text, illus) => {
    if (illus) liveRegion.setAttribute('data-illus', ''); else liveRegion.removeAttribute('data-illus');
    liveRegion.textContent = nb(text);
  };

  // ================================================================== frame (A)
  const radiosA = T.labels.map((lab) => {
    const id = `${NAME}-label-${lab}`;
    return [el('input', { type: 'radio', id, name: `${NAME}-label`, value: lab, checked: lab === state.label }),
      el('label', { for: id }, lab === T.true_label ? `${lab} (true)` : lab)];
  });
  const svgA = sv('svg', { class: 'cum-svg cum-svg--a', 'aria-hidden': 'true', focusable: 'false' });
  const netA = el('div', { class: 'cum-net', 'data-illus': '' }, svgA);
  const readA = el('p', { class: 'cum-read', 'data-illus': '', id: `${NAME}-a-read` });
  const readA2 = el('p', { class: 'cum-read cum-read--margin', 'data-illus': '', id: `${NAME}-a-margins` });
  const stripBP = sv('svg', { class: 'cum-strip__svg', viewBox: '0 0 236 66', width: 236, height: 66, 'aria-hidden': 'true', focusable: 'false' });
  const stripFF = sv('svg', { class: 'cum-strip__svg', viewBox: '0 0 236 66', width: 236, height: 66, 'aria-hidden': 'true', focusable: 'false' });
  drawStrips(stripBP, stripFF);
  const figA = el('figure', { class: 'fig fig--toy cum-fig cum-a', 'aria-labelledby': `${NAME}-a-title` },
    el('div', { class: 'fig__head' },
      el('h3', { class: 'fig__title', id: `${NAME}-a-title` }, 'Forward-Forward in one picture'),
      el('span', { class: 'badge badge--toy' }, 'Toy numbers')),
    el('div', { class: 'cum-strips' },
      el('div', { class: 'cum-strip' }, stripBP,
        el('p', { class: 'cum-strip__text' }, el('b', {}, 'Backpropagation:'), ' errors flow back through all layers')),
      el('div', { class: 'cum-strip' }, stripFF,
        el('p', { class: 'cum-strip__text' }, el('b', {}, 'Forward-Forward (FF):'), ' each block learns from its own loss; no gradient between blocks'))),
    el('div', { class: 'controls cum-controls' },
      el('fieldset', { class: 'ctl' },
        el('legend', { class: 'ctl__label' }, 'Candidate label'),
        el('div', { class: 'seg' }, radiosA))),
    netA,
    el('div', { class: 'tray cum-tray cum-tray--a' }, readA, readA2),
    el('figcaption', { class: 'fig__caption' },
      el('b', {}, 'Goodness:'), ' each block’s score for (image, label), high for the true label, low for wrong ones. ',
      el('b', {}, 'Prediction:'), ' label with the largest summed goodness.',
      el('span', { class: 'src' }, `${T.note.replace(/\.$/, '')} · after the poster’s Background panel`)));

  // ================================================================== frame (B)
  const range = el('input', {
    class: 'range', type: 'range', id: `${NAME}-gamma`, min: G.min, max: G.max, step: G.step, value: G.value,
    'aria-describedby': `${NAME}-gamma-hint`,
  });
  const out = el('output', { for: `${NAME}-gamma`, id: `${NAME}-gamma-out`, 'data-illus': '' });
  const presetBtns = D.toy.presets.map((p) => {
    const b = el('button', { class: 'btn btn--sm cum-preset', type: 'button', 'data-gamma': p.gamma, 'data-rule': p.rule },
      glyph(SITE, RULES[p.rule].arm), `${p.gamma} · ${p.label}`);
    return b;
  });
  const presetGroup = el('div', { class: 'cum-presets', role: 'group', 'aria-labelledby': `${NAME}-presets-label` },
    el('span', { class: 'cum-presets__label', id: `${NAME}-presets-label` }, 'Presets (the paper’s runs)'),
    el('div', { class: 'cum-presets__row' }, presetBtns),
    el('p', { class: 'cum-presets__hint' }, 'Each also selects that run in the measured figure below.'));
  const svgB = sv('svg', { class: 'cum-svg cum-svg--b', 'aria-hidden': 'true', focusable: 'false' });
  const chartB = el('div', { class: 'cum-chart cum-chart--b', 'data-illus': '' }, svgB);
  const trayKey = el('span', { class: 'tray__key' });
  const trayVal = el('span', { class: 'tray__val' });
  const chipKey = el('span', { class: 'measured__key' });   // the measured run the slider sits on, if any
  const chipVal = el('span', { class: 'measured__val' });
  const chipSrc = el('span', { class: 'measured__src' });
  const chip = el('div', { class: 'measured cum-chip' }, chipKey, chipVal, chipSrc);
  const figB = el('figure', { class: 'fig fig--illus cum-fig cum-b', 'aria-labelledby': `${NAME}-b-title` },
    el('div', { class: 'fig__head' },
      el('h3', { class: 'fig__title', id: `${NAME}-b-title` }, 'A training score with history'),
      el('span', { class: 'badge badge--illus' }, 'Illustration · Eq. (2) · toy margins')),
    el('div', { class: 'controls cum-controls cum-b__controls' },
      el('div', { class: 'range-field' },
        el('div', { class: 'range-field__top' },
          el('label', { for: `${NAME}-gamma` }, 'History weight γ'), out),
        range,
        el('p', { class: 'range-field__hint', id: `${NAME}-gamma-hint` }, 'Weight on the earlier blocks’ margins; enters training only.')),
      presetGroup),
    el('ul', { class: 'legend cum-legend' },
      el('li', {}, swatch('cum-own'), el('span', {}, 'own margin ', el('i', {}, 'm'), el('sup', {}, '(', el('i', {}, 'd'), ')'))),
      el('li', {}, swatch('cum-hist'), el('span', { class: 'cum-legend__hist' }, 'inherited: γ × the earlier blocks’ margins'))),
    el('p', { class: 'cum-axis-title' }, 'Toy training margin ', el('i', {}, 'M'), el('sup', {}, '(', el('i', {}, 'd'), ')'),
      ` of the “${T.true_label}” image, per block`),
    chartB,
    el('p', { class: 'cum-pred' }, el('b', {}, 'Prediction (no γ):'), ` ${BEST} ✓, unchanged at every γ.`),
    el('div', { class: 'tray cum-tray' },
      el('div', { 'data-illus': '' }, trayKey, trayVal),
      chip),
    el('figcaption', { class: 'fig__caption' },
      el('i', { class: 'fig__lead' }, 'Illustration:'), ' the slider evaluates Eq. (2) on toy margins; the bars are not measurements. γ enters training only: at test time every variant predicts from the plain, unweighted sum of block goodness.',
      el('span', { class: 'src', 'data-illus': '' },
        `Toy margins: “${T.true_label}” minus “${T.margins_vs}” in the toy network, ${T.margins.map(toy).join(', ')} at blocks 0–3`)));

  // ================================================================== frame (C)
  const radiosC = D.rules.map((r) => {
    const id = `${NAME}-rule-${r.key}`;
    return [el('input', { type: 'radio', id, name: `${NAME}-rule`, value: r.key, checked: r.key === state.rule }),
      el('label', { for: id }, glyph(SITE, r.arm), ruleName(r.key))];
  });
  const svgC = sv('svg', { class: 'cum-svg cum-svg--c', 'aria-hidden': 'true', focusable: 'false' });
  const tip = el('div', { class: 'tip cum-tip', id: `${NAME}-tip`, role: 'tooltip', hidden: true },
    el('span', { class: 'tip__v' }), el('span', { class: 'tip__l' }), el('span', { class: 'tip__l' }),
    el('span', { class: 'tip__l cum-tip__src' }));
  const chartC = el('div', {
    class: 'cum-chart cum-chart--c', tabindex: '0', role: 'group', 'aria-roledescription': 'chart',
    'aria-describedby': `${NAME}-c-help`,
  }, svgC, tip);
  const statVal = el('span', { class: 'measured__val' });
  const statSrc = el('span', { class: 'measured__src' });
  const capC = el('p', { class: 'cum-c__cap', id: `${NAME}-c-cap` });
  // block 3 in all four runs: own share as a 100% bar (own = slate, the rest = inherited history)
  const miniRows = D.rules.map((r) => {
    const own = el('span', { class: 'cum-mini__own' });
    own.style.setProperty('--own-share', `${D.facts[r.share[3]].v}%`);
    return el('li', { class: 'cum-mini__row', 'data-rule': r.key },
      el('span', { class: 'cum-mini__name' }, glyph(SITE, r.arm), ruleName(r.key)),
      el('span', { class: 'cum-mini__bar', 'aria-hidden': 'true' }, own),
      el('span', { class: 'cum-mini__val' }, F(r.share[3])));
  });
  const mini = el('div', { class: 'cum-mini' },
    el('p', { class: 'cum-mini__key', id: `${NAME}-mini-key` }, 'Block 3’s own share in each run'),
    el('ul', { class: 'cum-mini__list', 'aria-labelledby': `${NAME}-mini-key` }, miniRows));
  const figC = el('figure', { class: 'fig cum-fig cum-c', 'aria-labelledby': `${NAME}-c-title` },
    el('div', { class: 'fig__head' },
      el('h3', { class: 'fig__title', id: `${NAME}-c-title` }, 'Free-riding, pictured'),
      el('span', { class: 'badge badge--measured' }, 'Measured')),
    el('div', { class: 'controls cum-controls' },
      el('fieldset', { class: 'ctl' },
        el('legend', { class: 'ctl__label' }, 'History rule'),
        el('div', { class: 'seg' }, radiosC))),
    el('div', { class: 'cum-c__body' },
      el('div', { class: 'cum-c__plot' },
        el('ul', { class: 'legend cum-legend' },
          el('li', {}, swatch('cum-own'), 'own goodness'),
          el('li', {}, swatch('cum-hist'), el('span', { class: 'cum-legend__hist' }, 'inherited history')),
          el('li', { class: 'cum-legend__note' }, '% = own share')),
        el('p', { class: 'cum-axis-title' }, 'Positive goodness of the true label: the training score'),
        chartC),
      el('div', { class: 'cum-c__read' },
        el('div', { class: 'measured cum-stat' },
          el('span', { class: 'measured__key' }, 'Measured · block 3’s own share'), statVal, statSrc),
        capC,
        mini,
        el('p', { class: 'cum-c__hint', id: `${NAME}-c-hint` },
          el('span', { class: 'only-hover' }, 'Hover a bar, or focus the chart and use the arrow keys, to read each block’s own goodness (Tab. 16). Esc closes the reading.'),
          el('span', { class: 'only-touch' }, 'Tap a bar to read each block’s own goodness (Tab. 16).')),
        el('p', { class: 'vh', id: `${NAME}-c-help` },
          'Use the arrow keys to read each block’s own goodness (Tab. 16); Home and End jump to block 0 and block 3; Esc closes the reading.'))),
    el('figcaption', { class: 'fig__caption' },
      'Seed 42 only. The γ = 0.7 run also carries the depth-scaled auxiliary and λ', el('sub', {}, 'depth'),
      ' = 0.2, which the γ = 0 and γ = 1 runs lack (Fig. 1 caption), so it differs from both in more than γ. ',
      `Healthier blocks here do not mean higher accuracy: Stage 1 ${F('c.acc_pair_s1#85.86%')} (hardness-gated) vs. ${F('c.acc_pair_s1#86.18%')} (constant γ = 0.7).`,
      el('span', { class: 'src' }, 'CIFAR-10 · 4 blocks, width 128 · seed 42 · end of Stage 1 · App. Fig. 6, Tab. 16')));

  live.append(el('div', { class: 'cum' }, figA, figB, figC, liveRegion));
  if (eq) figB.insertBefore(eq, figB.querySelector('.cum-b__controls'));
  root.dataset.state = 'live';

  // phones (< 640 px): the presets leave the slider's panel for one of their own under the readouts, so that the
  // slider sits right above the chart it drives; wider screens keep them beside the slider (focus is kept)
  const bControls = figB.querySelector('.cum-b__controls');
  const presetsBox = el('div', { class: 'controls cum-controls cum-b__presets' });
  const phone = window.matchMedia('(max-width: 639px)');
  const placePresets = guard(() => {
    const a = document.activeElement;
    const had = a && presetGroup.contains(a);
    if (phone.matches) {
      if (presetGroup.parentNode !== presetsBox) {
        presetsBox.append(presetGroup);
        figB.insertBefore(presetsBox, figB.querySelector(':scope > .fig__caption'));
      }
    } else if (presetGroup.parentNode !== bControls) {
      bControls.append(presetGroup);
      presetsBox.remove();
    }
    if (had) a.focus({ preventScroll: true });
  });
  placePresets();
  phone.addEventListener('change', placePresets);

  // ================================================================== drawing
  function drawStrips(bp, ff) {
    // backpropagation: four blocks, one loss at the end, one long backward arrow
    const bx = (i) => 4 + i * 34;
    for (const i of BLOCKS) {
      sv('rect', { x: bx(i), y: 10, width: 20, height: 20, rx: 4, class: 'cum-box' }, bp);
      if (i < 3) { sv('path', { d: `M${bx(i) + 22},20H${bx(i + 1) - 4}`, class: 'cum-fwd' }, bp); arrowHead(bp, bx(i + 1) - 2, 20, 'right', 'cum-fwd-head', 3); }
    }
    sv('path', { d: `M${bx(3) + 22},20H${bx(3) + 34}`, class: 'cum-fwd' }, bp);
    arrowHead(bp, bx(3) + 36, 20, 'right', 'cum-fwd-head', 3);
    sv('rect', { x: 142, y: 9, width: 40, height: 22, rx: 11, class: 'cum-pill' }, bp);
    tx(bp, 162, 24.5, 'loss', { class: 'cum-t cum-t--pill', 'text-anchor': 'middle' });
    sv('path', { d: 'M162,33C162,62 14,62 14,37', class: 'cum-back' }, bp);
    arrowHead(bp, 14, 33, 'up', 'cum-back-head', 4.2);
    // Forward-Forward: four blocks, a loss on each, short arrows, no gradient between blocks
    const fx = (i) => 12 + i * 58;
    for (const i of BLOCKS) {
      sv('rect', { x: fx(i), y: 2, width: 40, height: 20, rx: 10, class: 'cum-pill' }, ff);
      tx(ff, fx(i) + 20, 16.5, 'loss', { class: 'cum-t cum-t--pill', 'text-anchor': 'middle' });
      sv('path', { d: `M${fx(i) + 20},23V${38}`, class: 'cum-loss' }, ff);
      arrowHead(ff, fx(i) + 20, 42, 'down', 'cum-loss-head', 3.6);
      sv('rect', { x: fx(i) + 10, y: 43, width: 20, height: 20, rx: 4, class: 'cum-box' }, ff);
      if (i < 3) { sv('path', { d: `M${fx(i) + 32},53H${fx(i + 1) + 6}`, class: 'cum-fwd' }, ff); arrowHead(ff, fx(i + 1) + 8, 53, 'right', 'cum-fwd-head', 3); }
    }
  }

  // Frame (A) sits beside the taller frame (B) from 1000 px: its drawing takes the spare height (taller block
  // bars and prediction panel, up to a cap; any rest is split above and below), so the frame has no empty band.
  // The SVG is absolutely positioned inside .cum-net, which reserves the natural height (--cum-net-h).
  function drawA(W) {
    const H0 = paintA(W, 0);
    netA.style.setProperty('--cum-net-h', `${Math.ceil(H0)}px`);
    const avail = Math.floor(netA.clientHeight);
    const extra = Math.min(Math.max(0, avail - Math.ceil(H0)), W >= 520 ? 132 : 96);
    const H = extra > 0 ? paintA(W, extra) : H0;
    netA.style.setProperty('--cum-net-top', `${Math.max(0, Math.floor((avail - H) / 2))}px`);
  }
  function paintA(W, extra) {
    clear(svgA);
    const sel = state.label;
    const gsel = T.goodness[sel];
    const wide = W >= 520;
    const P = wide ? Math.round(clamp(W * 0.35, 198, 244)) : W;
    const N = wide ? W - P - 22 : W - 18;
    const img = wide ? 42 : 34;
    const gapI = wide ? 16 : 12;
    const gapB = wide ? 12 : 8;
    const bw = (N - img - gapI - 3 * gapB) / 4;
    const two = bw < 58;
    const bh = wide ? 40 : two ? 42 : 34;
    const bx = (d) => img + gapI + d * (bw + gapB);
    const cx = (d) => bx(d) + bw / 2;
    const railY = 10.5;
    const HB = (wide ? 132 : 46) + extra;
    const amax = GEO.a_scale_max;
    const base = railY + 30 + HB;
    const blkY = base + 8;
    const laneY = blkY + bh + 26;
    const netH = laneY + 26;

    // rail: the block goodness is summed and goes to the prediction
    const panelX = wide ? N + 22 : 0;
    const panelY = wide ? 0 : netH + 18;
    const railEnd = wide ? panelX - 1 : W - 7;
    sv('path', { d: `M${r1(cx(0))},${railY}H${r1(railEnd - (wide ? 6 : 0))}` + (wide ? '' : `V${r1(panelY - 6)}`), class: 'cum-rail' }, svgA);
    if (wide) arrowHead(svgA, railEnd, railY, 'right', 'cum-rail-head', 4);
    else arrowHead(svgA, railEnd, panelY - 1, 'down', 'cum-rail-head', 4);
    for (const d of BLOCKS) sv('path', { d: `M${r1(cx(d))},${railY}v7`, class: 'cum-rail' }, svgA);

    // block bars: the selected label's toy goodness per block (block 0 lightest)
    const barW = Math.min(24, bw * 0.42);
    for (const d of BLOCKS) {
      const h = (HB * gsel[d]) / amax;
      sv('path', { d: barPath(cx(d) - barW / 2, base - h, barW, base, 3), class: `cum-seg cum-seg--${d}` }, svgA);
      tx(svgA, cx(d), base - h - 6, toy(gsel[d]), { class: 'cum-t cum-t--toy', 'text-anchor': 'middle' });
    }
    sv('path', { d: `M${r1(bx(0) - 2)},${r1(base + 0.5)}H${r1(bx(3) + bw + 2)}`, class: 'cum-base' }, svgA);

    // image pictogram and the blocks
    const imgY = blkY + (bh - img) / 2;
    const pic = sv('g', { class: 'cum-pict' }, svgA);
    sv('rect', { x: 0.75, y: r1(imgY), width: img - 1.5, height: img, rx: 6, class: 'cum-pict__frame' }, pic);
    sv('path', { d: `M1.5,${r1(imgY + img * 0.82)}C${r1(img * 0.3)},${r1(imgY + img * 0.45)} ${r1(img * 0.62)},${r1(imgY + img * 0.5)} ${r1(img - 1.5)},${r1(imgY + img * 0.7)}V${r1(imgY + img - 1)}H1.5Z`, class: 'cum-pict__hill' }, pic);
    sv('circle', { cx: r1(img * 0.72), cy: r1(imgY + img * 0.3), r: r1(img * 0.12), class: 'cum-pict__sun' }, pic);
    tx(svgA, img / 2, imgY - 7, 'image', { class: 'cum-t cum-t--muted', 'text-anchor': 'middle' });
    const midY = blkY + bh / 2;
    sv('path', { d: `M${img + 2},${r1(midY)}H${r1(bx(0) - 5)}`, class: 'cum-fwd' }, svgA);
    arrowHead(svgA, bx(0) - 2, midY, 'right', 'cum-fwd-head', 3.4);
    for (const d of BLOCKS) {
      sv('rect', { x: r1(bx(d)), y: r1(blkY), width: r1(bw), height: bh, rx: 7, class: 'cum-box cum-box--block' }, svgA);
      if (two) {
        tx(svgA, cx(d), blkY + 17, 'block', { class: 'cum-t cum-t--small', 'text-anchor': 'middle' });
        tx(svgA, cx(d), blkY + 34, String(d), { class: 'cum-t cum-t--strong', 'text-anchor': 'middle' });
      } else {
        tx(svgA, cx(d), midY + 5, `block ${d}`, { class: 'cum-t cum-t--block', 'text-anchor': 'middle' });
      }
      if (d < 3) {
        sv('path', { d: `M${r1(bx(d) + bw + 2)},${r1(midY)}H${r1(bx(d + 1) - 5)}`, class: 'cum-fwd' }, svgA);
        arrowHead(svgA, bx(d + 1) - 2, midY, 'right', 'cum-fwd-head', 3.4);
      }
    }

    // the candidate label enters every block (dashed)
    const tagH = 24;
    const tagT = tx(svgA, 0, laneY + 4.5, `label: ${sel}?`, { class: 'cum-t cum-t--tag' });
    const tw = Math.ceil(textWidth(tagT, 7)) + 20;
    tagT.setAttribute('x', '10');
    sv('path', { d: `M${tw},${r1(laneY)}H${r1(cx(3))}`, class: 'cum-dash' }, svgA);
    for (const d of BLOCKS) {
      const y0 = cx(d) < tw + 4 ? laneY - tagH / 2 : laneY;
      sv('path', { d: `M${r1(cx(d))},${r1(y0)}V${r1(blkY + bh + 7)}`, class: 'cum-dash' }, svgA);
      arrowHead(svgA, cx(d), blkY + bh + 2, 'up', 'cum-dash-head', 3.6);
    }
    const tag = sv('rect', { x: 0.75, y: r1(laneY - tagH / 2), width: tw, height: tagH, rx: tagH / 2, class: 'cum-tag' }, svgA);
    svgA.insertBefore(tag, tagT);
    tx(svgA, tw + 10, laneY + 19, 'enters every block', { class: 'cum-t cum-t--small' });

    // prediction panel: three stacked bars (segments = blocks 0-3), the toy sum on top, ✓ under the largest
    const pw = wide ? P : W;
    const ph = wide ? Math.max(netH, 272) : 206;
    sv('rect', { x: r1(panelX + 0.75), y: r1(panelY + 0.75), width: r1(pw - 1.5), height: r1(ph - 1.5), rx: 12, class: 'cum-panel' }, svgA);
    tx(svgA, panelX + 14, panelY + 23, 'prediction:', { class: 'cum-t cum-t--strong' });
    tx(svgA, panelX + 14, panelY + 41, 'largest summed goodness', { class: 'cum-t' });
    const smax = Math.max(...T.labels.map((l) => T.sums[l]));
    const labY = panelY + ph - 36;
    const pBase = labY - 20;
    const HP = pBase - (panelY + 74);
    const colW = (pw - 28) / 3;
    const pbw = Math.min(24, colW * 0.42);
    T.labels.forEach((lab, i) => {
      const pcx = panelX + 14 + colW * (i + 0.5);
      const tot = (HP * T.sums[lab]) / smax;
      if (lab === sel) {
        sv('rect', { x: r1(pcx - colW / 2 + 3), y: r1(pBase - tot - 28), width: r1(colW - 6), height: r1(tot + 28 + 26), rx: 8, class: 'cum-sel' }, svgA);
      }
      let y = pBase;
      T.goodness[lab].forEach((g, d) => {
        const hseg = (HP * g) / smax;
        const top = y - hseg;
        const last = d === 3;
        sv('path', { d: barPath(pcx - pbw / 2, top + (last ? 0 : 1), pbw, y - (d === 0 ? 0 : 1), last ? 3 : 0), class: `cum-seg cum-seg--${d}` }, svgA);
        y = top;
      });
      tx(svgA, pcx, pBase - tot - 7, toy(T.sums[lab]), { class: 'cum-t cum-t--toy', 'text-anchor': 'middle' });
      tx(svgA, pcx, labY, lab === BEST ? `${lab} ✓` : lab, { class: `cum-t ${lab === sel ? 'cum-t--strong' : ''}`, 'text-anchor': 'middle' });
    });
    sv('path', { d: `M${r1(panelX + 12)},${r1(pBase + 0.5)}H${r1(panelX + pw - 12)}`, class: 'cum-base' }, svgA);
    // which segment is which block: block 0 lightest (bottom) to block 3 darkest (top)
    const legY = panelY + ph - 13;
    const legT = tx(svgA, panelX + 14, legY, 'segments: block 0', { class: 'cum-t cum-t--small' });
    let lx = panelX + 14 + Math.ceil(textWidth(legT, 6.4)) + 6;
    for (const d of BLOCKS) { sv('rect', { x: r1(lx), y: r1(legY - 10), width: 11, height: 11, rx: 2, class: `cum-seg cum-seg--${d}` }, svgA); lx += 13; }
    tx(svgA, lx + 3, legY, '3', { class: 'cum-t cum-t--small' });

    const H = wide ? Math.max(netH, ph) : panelY + ph;
    svgA.setAttribute('class', `cum-svg cum-svg--a${wide ? ' cum-svg--wide' : ''}`);
    svgA.setAttribute('viewBox', `0 0 ${W} ${r1(H)}`);
    svgA.setAttribute('width', W);
    svgA.setAttribute('height', r1(H));
    return H;
  }

  function drawB(W) {
    clear(svgB);
    const b = toyB(state.g100);
    const [d0, d1] = GEO.b_domain;
    const L = 34, R = 6, TOP = 44, BOT = 30;
    const plotH = Math.round(clamp(W * 0.3, 146, 176));
    const H = TOP + plotH + BOT;
    const y = (v) => TOP + plotH * (1 - (v - d0) / (d1 - d0));
    for (const t of GEO.b_ticks) {
      sv('path', { d: `M${L},${r1(y(t)) + 0.5}H${W - R}`, class: t === d0 ? 'cum-axis' : 'cum-grid' }, svgB);
      tx(svgB, L - 8, y(t) + 4.5, toy(t), { class: 'cum-t tick', 'text-anchor': 'end' });
    }
    const band = (W - L - R) / 4;
    const bw = Math.min(24, band * 0.46);
    for (const d of BLOCKS) {
      const cx = L + band * (d + 0.5);
      const x = cx - bw / 2;
      const yBase = y(0);
      const yOwn = y(b.own[d]);
      const yTot = y(b.own[d] + b.inh[d]);
      const hasHist = b.inh[d] > 0 && yOwn - 2 - yTot >= 1;
      sv('path', { d: barPath(x, yOwn, bw, yBase, hasHist ? 0 : 4), class: 'cum-own' }, svgB);
      if (hasHist) sv('path', { d: barPath(x + 0.75, yTot + 0.75, bw - 1.5, yOwn - 2, 4), class: 'cum-hist' }, svgB);
      if (b.inh[d] > 0) {
        tx(svgB, cx, yTot - 24, `+${toy(b.inh[d])}`, { class: 'cum-t cum-t--hist', 'text-anchor': 'middle' });
        tx(svgB, cx, yTot - 8, toy(b.own[d]), { class: 'cum-t cum-t--toy', 'text-anchor': 'middle' });
      } else {
        tx(svgB, cx, yOwn - 8, toy(b.own[d]), { class: 'cum-t cum-t--toy', 'text-anchor': 'middle' });
      }
      tx(svgB, cx, H - 9, `block ${d}`, { class: 'cum-t cum-t--muted', 'text-anchor': 'middle' });
    }
    svgB.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svgB.setAttribute('width', W);
    svgB.setAttribute('height', H);
  }

  // frame (C) chart: the hovered block wins over the keyboard/tap-selected one; both show the same reading
  let geoC = null;          // last layout of frame (C): scales, the highlight band and the bar tops
  let hover = -1;
  const shown = () => (hover >= 0 ? hover : state.active);
  function drawC(W) {
    clear(svgC);
    const r = RULES[state.rule];
    const own = GEO.own[r.key];
    const tot = GEO.total[r.key];
    const [d0, d1] = GEO.c_domain;
    const L = 28, R = 6, TOP = 30, BOT = 30;
    const plotH = Math.round(clamp(W * 0.42, 196, 252));
    const H = TOP + plotH + BOT;
    const y = (v) => TOP + plotH * (1 - (v - d0) / (d1 - d0));
    const band = (W - L - R) / 4;
    const bw = Math.min(24, band * 0.46);
    const hl = sv('rect', { x: L, y: 2, width: r1(band), height: H - 4, rx: 10, class: 'cum-active', visibility: 'hidden' }, svgC);
    for (const t of GEO.c_ticks) {
      sv('path', { d: `M${L},${r1(y(t)) + 0.5}H${W - R}`, class: t === d0 ? 'cum-axis' : 'cum-grid' }, svgC);
      tx(svgC, L - 8, y(t) + 4.5, String(t), { class: 'cum-t tick', 'text-anchor': 'end' });
    }
    for (const d of BLOCKS) {
      const cx = L + band * (d + 0.5);
      const x = cx - bw / 2;
      const yOwn = y(own[d]);
      const yTot = y(tot[d]);
      const hasHist = yOwn - 2 - yTot >= 1;
      sv('path', { d: barPath(x, yOwn, bw, y(0), hasHist ? 0 : 4), class: 'cum-own' }, svgC);
      if (hasHist) sv('path', { d: barPath(x + 0.75, yTot + 0.75, bw - 1.5, yOwn - 2, 4), class: 'cum-hist' }, svgC);
      tx(svgC, cx, yTot - 8, F(r.share[d]), { class: `cum-t ${d === 3 ? 'cum-t--key' : 'cum-t--share'}`, 'text-anchor': 'middle' });
      tx(svgC, cx, H - 9, `block ${d}`, { class: 'cum-t cum-t--muted', 'text-anchor': 'middle' });
    }
    for (const d of BLOCKS) {              // hit targets: the whole band of each block (>= 24 px)
      const hit = sv('rect', { x: r1(L + band * d), y: 0, width: r1(band), height: H, class: 'cum-hit' }, svgC);
      hit.addEventListener('pointerenter', guard(() => { hover = d; paintC(); }));
      hit.addEventListener('click', guard(() => { state.active = d; paintC(); }));
    }
    svgC.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svgC.setAttribute('width', W);
    svgC.setAttribute('height', H);
    geoC = { L, band, bw, y, tot, H, hl };
    paintC();
  }

  // ------------------------------------------------------------------ reading of one block (frame C)
  const [tipV, tipL1, tipL2, tipSrc] = Array.from(tip.children);
  function tipText(d) {            // value first, then label (own g+ from Tab. 16, own share from Fig. 6)
    const r = RULES[state.rule];
    const b3 = d === 3 && r.b3_own;
    return [`own ${F(r.own[d])}`, `block ${d} · own share ${F(r.share[d])}`,
      b3 ? `own ${F(r.b3_own)} of ${F(r.b3_total)} in total (App. C)` : '',
      `Tab. 16 · App. Fig. 6 · ${ruleName(r.key)}`];
  }
  function paintC() {                     // highlight band + tooltip, without redrawing the chart
    if (!geoC) return;
    const d = shown();
    if (d < 0) { geoC.hl.setAttribute('visibility', 'hidden'); tip.hidden = true; return; }
    geoC.hl.setAttribute('x', r1(geoC.L + geoC.band * d + 2));
    geoC.hl.setAttribute('width', r1(geoC.band - 4));
    geoC.hl.removeAttribute('visibility');
    const [v, l1, l2, src] = tipText(d);
    tipV.textContent = nb(v);
    tipL1.textContent = nb(l1);
    tipL2.textContent = nb(l2);
    tipL2.hidden = !l2;
    tipSrc.textContent = nb(src);
    tip.hidden = false;
    const cw = chartC.clientWidth;
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    const cx = geoC.L + geoC.band * (d + 0.5);
    const yTop = geoC.y(geoC.tot[d]) - 30;          // above the share label (may rise over the legend)
    let left, top;
    if (yTop - th >= -96) {
      left = clamp(cx - tw / 2, 0, Math.max(0, cw - tw));
      top = yTop - th;
    } else {                                       // no room above: beside the bar, toward the middle
      left = d <= 1 ? cx + geoC.bw / 2 + 10 : cx - geoC.bw / 2 - 10 - tw;
      left = clamp(left, 0, Math.max(0, cw - tw));
      top = clamp(geoC.y(geoC.tot[d]) - th / 2, 0, Math.max(0, geoC.H - th));
    }
    tip.style.setProperty('left', `${Math.round(left)}px`);
    tip.style.setProperty('top', `${Math.round(top)}px`);
  }
  function readBlock(d) {
    const r = RULES[state.rule];
    const b3 = d === 3 && r.b3_own ? `; ${F(r.b3_own)} of ${F(r.b3_total)} in total` : '';
    announce(`Block ${d}: own goodness ${F(r.own[d])}, own share ${F(r.share[d])}${b3}.`, false);
  }

  // ------------------------------------------------------------------ state updates
  function updateA(speak) {
    readA.textContent = nb(sentenceA(state.label));
    readA2.textContent = nb(marginsA(state.label));
    wA.redraw();
    if (speak) announce(`${readA.textContent} ${readA2.textContent}`, true);
  }
  function updateB(speak) {
    const g100 = state.g100;
    const p = presetFor(g100);
    out.textContent = gam(g100);
    range.setAttribute('aria-valuetext', `history weight γ = ${gam(g100)}${p ? `, ${p.say}` : ''}`);
    range.style.setProperty('--pct', `${((g100 / 100 - G.min) / (G.max - G.min)) * 100}%`);
    for (const btn of presetBtns) {
      if (Math.round(Number(btn.getAttribute('data-gamma')) * 100) === g100) btn.setAttribute('aria-current', 'true');
      else btn.removeAttribute('aria-current');
    }
    const run = p ? RULES[p.rule] : null;
    chip.classList.toggle('is-none', !run);
    if (run) {
      chipKey.textContent = nb(run.b3_own ? 'Measured · App. C, Fig. 6' : 'Measured · App. Fig. 6');
      chipVal.removeAttribute('data-illus');
      chipVal.textContent = F(run.share[3]);
      chipSrc.textContent = nb(`block 3’s own share in the trained ${ruleName(run.key)} run`);
    } else {                       // a γ the paper never trained: no measured outcome (the slider's value is toy)
      chipKey.textContent = nb('Measured · App. Fig. 6');
      chipVal.setAttribute('data-illus', '');
      chipVal.textContent = nb(`No trained run at γ = ${gam(g100)}`);
      chipSrc.textContent = nb('The trained runs here use γ = 0, 0.7 or 1 (the presets).');
    }
    const b = toyB(g100);
    trayKey.textContent = nb(`At γ = ${gam(g100)}, block 3’s toy training margin is`);
    trayVal.textContent = nb(`${toy(b.own[3])} own + ${toy(b.inh[3])} inherited`);
    wB.redraw();
    if (speak) announce(sentenceB(g100), true);
  }
  function updateC(speak) {
    const r = RULES[state.rule];
    statVal.textContent = F(r.share[3]);
    statSrc.textContent = nb(r.b3_own ? `${ruleName(r.key)} · App. C, Fig. 6` : `${ruleName(r.key)} · App. Fig. 6`);
    capC.textContent = nb(captionC(r.key));
    for (const row of miniRows) {
      if (row.getAttribute('data-rule') === r.key) row.setAttribute('aria-current', 'true'); else row.removeAttribute('aria-current');
    }
    chartC.setAttribute('aria-label', `Measured own and inherited goodness per block, ${ruleName(r.key)}. ` +
      `Own share of each block’s training score, blocks 0 to 3: ${r.share.map(F).join(', ')}.`);
    wC.redraw();
    if (speak) announce(capC.textContent, false);
  }

  // ------------------------------------------------------------------ events
  for (const [inp] of radiosA) {
    inp.addEventListener('change', guard(() => { if (inp.checked) { state.label = inp.value; updateA(true); } }));
  }
  range.addEventListener('input', guard(() => { state.g100 = Math.round(Number(range.value) * 100); updateB(false); }));
  range.addEventListener('change', guard(() => { state.g100 = Math.round(Number(range.value) * 100); updateB(true); }));
  for (const btn of presetBtns) {
    btn.addEventListener('click', guard(() => {
      const g = Number(btn.getAttribute('data-gamma'));
      range.value = String(g);
      state.g100 = Math.round(g * 100);
      updateB(false);
      const rk = btn.getAttribute('data-rule');
      const radio = radiosC.find(([inp]) => inp.value === rk)[0];
      radio.checked = true;
      state.rule = rk;
      updateC(false);
      announce(`${sentenceB(state.g100)} Measured figure: ${ruleName(rk)}. ${captionC(rk)}`, true);
    }));
  }
  for (const [inp] of radiosC) {
    inp.addEventListener('change', guard(() => { if (inp.checked) { state.rule = inp.value; updateC(true); } }));
  }
  chartC.addEventListener('keydown', guard((e) => {
    const k = e.key;
    let d = state.active;
    if (k === 'ArrowRight' || k === 'ArrowDown') d = d < 0 ? 0 : Math.min(3, d + 1);
    else if (k === 'ArrowLeft' || k === 'ArrowUp') d = d < 0 ? 3 : Math.max(0, d - 1);
    else if (k === 'Home') d = 0;
    else if (k === 'End') d = 3;
    else if (k === 'Escape') {
      if (state.active >= 0 || hover >= 0 || !tip.hidden) { e.preventDefault(); state.active = -1; hover = -1; paintC(); }
      return;
    } else return;
    e.preventDefault();
    hover = -1;
    state.active = d;
    paintC();
    readBlock(d);
  }));
  chartC.addEventListener('blur', guard(() => { state.active = -1; paintC(); }));
  chartC.addEventListener('pointerleave', guard(() => { hover = -1; paintC(); }));
  document.addEventListener('keydown', guard((e) => {  // Esc closes a hover reading wherever focus is (WCAG 1.4.13)
    if (e.key === 'Escape' && !tip.hidden && !chartC.contains(e.target)) { state.active = -1; hover = -1; paintC(); }
  }));

  const wA = onBox(netA, drawA, guard);
  const wB = onWidth(chartB, drawB, guard);
  const wC = onWidth(chartC, drawC, guard);
  updateA(false);
  updateB(false);
  updateC(false);
}

boot();
