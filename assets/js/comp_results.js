// Component "results" (SPEC_FINAL §5.3), ES module, no imports. Renders into #viz-results [data-live]:
//  (A) block-3 separation bars linked to paired accuracy changes (95% CIs, four readouts) in three settings;
//  (B) what moves accuracy: training rule vs. architecture vs. augmentation on one 0–100% axis.
// On phones (< 640 px) the segment controls show short labels, and the arm chips, the seed switch and the reading
// move under the charts, so that a tapped setting changes the chart right below it.
// (C) "Calibration, not a leaderboard" is the shell's static #calibration figure: never moved or hidden here.
// Numbers print only from SITE_DATA.results facts (verbatim; "±" spacing aside) and literals; text via
// textContent; colors via CSS classes bound to the theme tokens (no theme listener needed).
const NAME = 'results';
const NS = 'http://www.w3.org/2000/svg';
const root = document.getElementById(`viz-${NAME}`);
if (root) {
  try { mount(root); } catch (err) { console.error(`[${NAME}]`, err); root.dataset.state = 'broken'; }
}

function mount(root) {
  const D = (window.SITE_DATA || {})[NAME];
  if (!D || !D.facts || !D.settings) throw new Error(`window.SITE_DATA.${NAME} is missing (assets/data/${NAME}.js)`);
  const live = root.querySelector('[data-live]');
  if (!live) throw new Error('[data-live] is missing');
  const F = (k) => { const e = D.facts[k]; if (!e) throw new Error(`fact ${k} is missing`); return e; };
  const disp = (s) => s.replace(/(\d%?)\s*±\s*(\d)/g, '$1 ± $2');   // whitespace only
  const S = (k) => disp(F(k).s);
  const L = (t) => { if (!(D.literals || {})[t]) throw new Error(`literal ${t} is missing`); return t; };
  const num = (t) => parseFloat(String(t).replace(/−/g, '-').replace(/[+%\s]/g, ''));
  const head = (s) => s.split(/[±[]/)[0].trim();                                // "4.78±0.02" -> "4.78"
  const G = D.geometry;
  const RO = ['s1', 's1_tta', 's2', 's2_tta'];
  const stg = (r) => r.slice(0, 2);
  const roName = (r) => `${stg(r) === 's1' ? 'Stage 1, FF’s own sum' : 'Stage 2, probe (not FF)'}, ${r.endsWith('_tta') ? 'with flip' : 'single crop'}`;
  const SETS = Object.fromEntries(D.settings.map((s) => [s.key, s]));
  const UI = {
    c100: ['CIFAR-100 · 3 seeds', 'cumulative', 'CIFAR-100'],
    c10: ['CIFAR-10 · 32 experts', 'history-free', 'CIFAR-10'],
    ctrl: ['CIFAR-10 control · width 128', '(a) cumulative + auxiliary', 'CIFAR-10 control'],
  };

  /* ---------------------------------------------------------------- DOM helpers */
  function el(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === 'class') e.className = v; else if (k === 'text') e.textContent = v; else e.setAttribute(k, v === true ? '' : v);
    }
    e.append(...kids.flat().filter((k) => k !== null && k !== undefined && k !== false));
    return e;
  }
  const r1 = (x) => Math.round(x * 10) / 10;
  function sv(tag, attrs, ...kids) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === 'text') e.textContent = v; else e.setAttribute(k, typeof v === 'number' ? r1(v) : v);
    }
    e.append(...kids.filter(Boolean));
    return e;
  }
  const text = (cls, x, y, t, anchor) => sv('text', { class: cls, x, y, 'text-anchor': anchor, text: t });
  function shapeD(shape, cx, cy, r) {             // mark and glyph outlines centered at (cx, cy)
    const s = shape.replace('-hollow', ''), p = (...v) => v.map(r1).join(' ');
    if (s === 'square' || s === 'hollow') { const h = r * 0.86; return `M${p(cx - h, cy - h)}h${r1(2 * h)}v${r1(2 * h)}h${r1(-2 * h)}z`; }
    if (s === 'triangle') { const h = r * 1.12; return `M${p(cx, cy - h)}L${p(cx + h, cy + h * 0.8)}H${r1(cx - h)}z`; }
    if (s === 'diamond') { const h = r * 1.18; return `M${p(cx, cy - h)}L${p(cx + h, cy)}L${p(cx, cy + h)}L${p(cx - h, cy)}z`; }
    return `M${p(cx - r, cy)}a${p(r, r)} 0 1 0 ${r1(2 * r)} 0a${p(r, r)} 0 1 0 ${r1(-2 * r)} 0z`;
  }
  // keep a focused mark fully in view: Firefox does not scroll a focused SVG <g> into view while its <svg> is
  // partly visible (g.scrollIntoView() does nothing there either), so the window scrolls by the mark's overflow,
  // between the page's scroll padding at the top (site.css: under the sticky top bar) and 12 px above the bottom edge
  function reveal(node) {
    const r = node.getBoundingClientRect();
    if (!r.width && !r.height) return;
    const top = parseFloat(getComputedStyle(document.documentElement).scrollPaddingTop) || 72, bottom = innerHeight - 12;
    const dy = r.bottom > bottom && r.height <= bottom - top ? r.bottom - bottom : r.top < top ? r.top - top : 0;
    if (!dy) return;
    try { window.scrollBy({ top: dy, left: 0, behavior: 'instant' }); } catch (_) { window.scrollBy(0, dy); }
  }
  const glyph = (shape, style) => sv('svg', { class: `rs-glyph rs-s--${style}`, viewBox: '0 0 12 12', 'aria-hidden': 'true', focusable: 'false' },
    sv('path', { d: shapeD(shape, 6, 6, 4.4) }));
  const li = (g, t, attrs) => el('li', attrs, g, t);
  function measure(svg, t, cls) {
    const n = text(cls, 0, -100, t);
    svg.append(n);
    const w = n.getComputedTextLength();
    n.remove();
    return w;
  }
  function wrap(s, max) {
    const lines = [''];
    for (const w of s.split(' ')) {
      const cur = lines[lines.length - 1];
      if (cur && `${cur} ${w}`.length > max) lines.push(w); else lines[lines.length - 1] = cur ? `${cur} ${w}` : w;
    }
    return lines;
  }
  function radios(name, label, options, value) {         // options: [value, label, short label on phones?]
    const seg = el('div', { class: 'seg' });
    const inputs = options.map(([v, t, short]) => {
      const input = el('input', { type: 'radio', name, id: `${name}-${v}`, value: v });
      input.checked = v === value;
      seg.append(input, short
        ? el('label', { for: `${name}-${v}` }, el('span', { class: 'seg__long', text: t }), el('span', { class: 'seg__short', 'aria-hidden': 'true', text: short }))
        : el('label', { for: `${name}-${v}`, text: t }));
      return input;
    });
    return { box: el('fieldset', { class: 'ctl' }, el('legend', { class: 'ctl__label', text: label }), seg), inputs };
  }

  /* ---------------------------------------------------------------- state and scaffold */
  const state = { setting: 'c100', readout: 'all', hidden: new Set(), seeds: false, bp: false };
  const st = () => SETS[state.setting];
  const armOf = (s, k) => s.arms.find((a) => a.key === k);
  const shown = (s, k) => k === s.ref || !state.hidden.has(k);

  const liveRegion = el('p', { class: 'vh', id: 'results-live', 'aria-live': 'polite' });
  const tip = el('div', { class: 'tip', id: 'results-tip', role: 'tooltip', hidden: true });
  const setCtl = radios('results-setting', 'Setting', Object.entries(UI).map(([k, u]) => [k, u[0], u[2]]), state.setting);
  const roCtl = radios('results-readout', 'Readout', [['all', 'All'], ['s1', 'Stage 1: FF’s own sum', 'Stage 1'], ['s2', 'Stage 2: probe (not FF)', 'Stage 2']], state.readout);
  const chips = el('div', { class: 'rs-chips' });
  const seedsBox = el('input', { type: 'checkbox', id: 'results-seeds', 'aria-describedby': 'results-seeds-hint' });
  const seedsCtl = el('div', { class: 'ctl' }, el('span', { class: 'ctl__label', 'aria-hidden': 'true', text: 'Seeds' }),
    el('label', { class: 'chip rs-check', for: 'results-seeds' }, seedsBox, 'Show per-seed differences'),
    el('span', { class: 'rs-hint', id: 'results-seeds-hint', text: 'CIFAR-100 only' }));
  const take = el('p', { class: 'callout rs-take', id: 'results-take' });
  const sepSub = el('p', { class: 'rs-sub', id: 'results-sep-sub' });
  const sepChart = el('div', { class: 'rs-chart rs-chart--sep' });
  const sepBelow = el('div', { class: 'rs-below' });
  const accTitle = el('h4', { id: 'results-acc-title' });
  const accSub = el('p', { class: 'rs-sub', id: 'results-acc-sub' });
  const legend = el('ul', { class: 'legend rs-legend', 'aria-label': 'Legend' });
  const accChart = el('div', { class: 'rs-chart rs-chart--acc' });
  const notes = el('div');
  const srcA = el('p', { class: 'src' });
  const armsCtl = el('div', { class: 'ctl', role: 'group', 'aria-labelledby': 'results-arms-label' },
    el('span', { class: 'ctl__label', id: 'results-arms-label', text: 'Arms shown' }), chips);
  const ctlBox = el('div', { class: 'controls' }, setCtl.box, roCtl.box, armsCtl, seedsCtl);
  const panels = el('div', { class: 'rs-panels' },
      el('div', { class: 'rs-panel rs-panel--sep' }, el('h4', { id: 'results-sep-title', text: 'Block health: deepest-block separation' }), sepSub, sepChart, sepBelow),
      el('div', { class: 'rs-panel' }, accTitle, accSub, legend, accChart));
  const figA = el('figure', { class: 'fig rs-fig', id: 'results-health', 'aria-labelledby': 'results-a-title' },
    el('div', { class: 'fig__head' }, el('h3', { class: 'fig__title', id: 'results-a-title', text: 'Block health vs. accuracy' }),
      el('span', { class: 'badge badge--measured', text: 'Measured' })),
    el('p', { class: 'rs-lede' }, 'Pick a setting and a readout. ',
      el('span', { class: 'only-hover', text: 'Hover over or focus a bar or a point: the same arm lights up in both charts. Arrow keys move between marks; Esc hides the tooltip.' }),
      el('span', { class: 'only-touch', text: 'Tap a bar or a point to read its values: the same arm lights up in both charts.' })),
    ctlBox,
    take,
    panels,
    el('figcaption', { class: 'fig__caption rs-caption' }, notes,
      el('p', { class: 'rs-note', text: '95% CIs from resampling test images, conditional on the trained models. Repairs are not exact redistributions: their stable accuracy is an empirical finding, not a consequence of Prop. 3.2.' }),
      srcA));

  const bpChip = el('button', { type: 'button', class: 'chip rs-bp', id: 'results-bp', 'aria-pressed': 'false' },
    glyph('square', 'bp'), 'Show backprop on the same plain CNN');
  const mv = el('div', { class: 'rs-mv', role: 'group', 'aria-labelledby': 'results-b-title', 'aria-describedby': 'results-mv-help' });
  const figB = el('figure', { class: 'fig rs-fig', id: 'results-suspects', 'aria-labelledby': 'results-b-title' },
    el('div', { class: 'fig__head' }, el('h3', { class: 'fig__title', id: 'results-b-title', text: 'What moves accuracy?' }),
      el('span', { class: 'badge badge--measured', text: 'Measured' })),
    el('p', { class: 'rs-lede', id: 'results-mv-help' }, 'Three suspects on one axis: how far each moves CIFAR-10 test accuracy (single crop). ',
      el('span', { class: 'only-hover', text: 'Hover over or focus a dot for its value and source.' }),
      el('span', { class: 'only-touch', text: 'Tap a dot for its value and source.' })),
    el('div', { class: 'controls rs-controls--b' },
      el('div', { class: 'ctl' }, el('span', { class: 'ctl__label', 'aria-hidden': 'true', text: 'Reference' }), bpChip),
      el('ul', { class: 'legend', 'aria-label': 'Legend' }, li(glyph('circle', 'ff'), 'Forward-Forward (FF)'),
        li(glyph('square', 'bp'), 'backprop (BP)'), li(glyph('circle', 'start'), 'hollow: starting point'),
        el('li', {}, glyph('square', 'cum'), glyph('circle', 'free'), ' FF history rules: cumulative, history-free'))),
    mv,
    el('figcaption', { class: 'fig__caption rs-caption' },
      el('p', { class: 'rs-note', text: `How much each suspect moves accuracy: removing or gating the history ${L('under 1 pp')}; FF from a plain CNN to our stripped backbone ${S('s1.ff_plain_cnn')} → ${S('s1.ff_stripped')}; backprop with weak vs. strong augmentation ${S('s1.aug_6_4')}.` }),
      el('p', { class: 'rs-note', text: 'FF and BP recipes differ in crop range and operator rates: the FF-vs-BP rows compare augmentation pipelines, not one shared augmentation.' }),
      el('p', { class: 'src', text: `CIFAR-10 test · single crop · Tab. 1 (cumulative 4 seeds, history-free 3 seeds) and Tab. 2 (3 seeds; ${S('s1.ff_stripped')}: 1 seed)` })));
  live.replaceChildren(figA, figB, liveRegion, tip);

  /* ---------------------------------------------------------------- tooltip (value first; follows a focused mark) */
  let tipOwner = null, tipRaf = 0;
  function showTip(m) {
    if (!m || !m._tip) return;
    tip.replaceChildren(el('span', { class: 'tip__v', text: m._tip.v }), ...m._tip.l.map((t) => el('span', { class: 'tip__l', text: t })));
    tip.hidden = false;
    tipOwner = m;
    const r = (m._anchor || m).getBoundingClientRect(), tw = tip.offsetWidth, th = tip.offsetHeight;
    const x = Math.max(8, Math.min(document.documentElement.clientWidth - tw - 8, r.left + r.width / 2 - tw / 2));
    let y = r.top - th - 10;
    if (y < 64) y = r.bottom + 10;
    if (y + th > innerHeight - 8) y = Math.max(8, innerHeight - th - 8);
    tip.style.setProperty('left', `${Math.round(x)}px`);
    tip.style.setProperty('top', `${Math.round(y)}px`);
  }
  function hideTip() { tip.hidden = true; tipOwner = null; }
  window.addEventListener('scroll', () => {
    if (tip.hidden || !tipOwner) return;
    if (document.activeElement !== tipOwner) { hideTip(); return; }
    if (!tipRaf) tipRaf = requestAnimationFrame(() => { tipRaf = 0; if (tipOwner && !tip.hidden) showTip(tipOwner); });
  }, { passive: true });
  root.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !tip.hidden) { hideTip(); e.stopPropagation(); } });

  /* ---------------------------------------------------------------- linking: one arm lights up in both charts */
  function setHL(arm) {
    const ref = !!arm && arm === st().ref;
    for (const m of figA.querySelectorAll('.rs-mark[data-arm], .rs-legend [data-arm]')) {
      m.classList.toggle('is-dim', !!arm && m.dataset.arm !== arm && !(ref && m.closest('.rs-chart--acc')));
      m.classList.toggle('is-hl', !!arm && m.dataset.arm === arm);
    }
    figA.classList.toggle('is-ref-hl', ref);
  }

  /* ---------------------------------------------------------------- one Tab stop per chart, arrows between marks */
  function rove(box, onMark) {
    let active = null;
    const marks = () => Array.from(box.querySelectorAll('.rs-mark'));
    const stop = (m) => { for (const x of marks()) x.setAttribute('tabindex', x === m ? '0' : '-1'); };
    const pick = (m) => { stop(m); active = m.dataset.key; };          // remember only marks the reader chose
    box.addEventListener('keydown', (e) => {
      const ms = marks(), i = ms.indexOf(document.activeElement);
      if (i < 0) return;
      const j = { ArrowRight: i + 1, ArrowDown: i + 1, ArrowLeft: i - 1, ArrowUp: i - 1, Home: 0, End: ms.length - 1 }[e.key];
      if (j === undefined) return;
      e.preventDefault();
      const m = ms[Math.max(0, Math.min(ms.length - 1, j))];
      pick(m);
      m.focus();
    });
    box.addEventListener('focusin', (e) => {
      const m = e.target.closest && e.target.closest('.rs-mark');
      if (!m) return;
      pick(m);
      onMark(m);
      reveal(m);
      showTip(m);
      requestAnimationFrame(() => { if (tipOwner === m) showTip(m); });   // after any scroll into view
    });
    box.addEventListener('focusout', (e) => {
      if (e.relatedTarget && box.contains(e.relatedTarget)) return;
      onMark(null);
      if (tipOwner && box.contains(tipOwner)) hideTip();
    });
    box.addEventListener('pointerover', (e) => {
      const m = e.target.closest && e.target.closest('.rs-mark');
      if (m) { onMark(m); showTip(m); }
    });
    box.addEventListener('pointerout', (e) => {
      const m = e.target.closest && e.target.closest('.rs-mark');
      if (!m || (e.relatedTarget && e.relatedTarget.closest && e.relatedTarget.closest('.rs-mark') === m)) return;
      const f = document.activeElement;
      if (f && f.classList && f.classList.contains('rs-mark') && box.contains(f)) { onMark(f); showTip(f); } else { onMark(null); hideTip(); }
    });
    return (hadFocus) => {                        // after a redraw: keep one Tab stop (and the focus)
      const ms = marks();
      if (!ms.length) return;
      const m = ms.find((x) => x.dataset.key === active) || ms[0];
      stop(m);
      if (hadFocus) { m.focus({ preventScroll: true }); showTip(m); }
    };
  }
  const link = (m) => setHL(m ? m.dataset.arm : null);
  const restoreSep = rove(sepChart, link), restoreAcc = rove(accChart, link), restoreMv = rove(mv, () => {});
  function mark(cls, key, arm, label, tipv, tipl) {
    const g = sv('g', { class: `rs-mark ${cls}`, 'data-key': key, 'data-arm': arm, tabindex: '-1', role: 'img', 'aria-label': label });
    g._tip = { v: tipv, l: tipl };
    return g;
  }

  /* ---------------------------------------------------------------- (A) separation bars */
  function renderSep() {
    const s = st(), W = Math.max(260, Math.floor(sepChart.clientWidth)), narrow = W < 470;
    sepChart.parentElement.classList.toggle('is-narrow', narrow);
    const svg = sv('svg', { class: 'rs-svg', width: W, role: 'group', 'aria-labelledby': 'results-sep-title results-sep-sub' });
    sepChart.replaceChildren(svg);
    const bars = s.bars.filter((b) => shown(s, b.arm));
    const pitch = narrow ? 50 : 52, th = narrow ? 18 : 20, top = 22, H = top + 4 * pitch + 30;
    const names = (a) => (narrow ? [a.short] : wrap(a.name, 24));
    const labelW = Math.max(...s.bars.flatMap((b) => names(armOf(s, b.arm)).map((t) => measure(svg, t, 'rs-lbl'))));
    const x0 = Math.ceil(labelW) + 30, x1 = W - (narrow ? 40 : 46), [d0, d1] = G.sep_axis;
    const sx = (v) => x0 + ((x1 - x0) * (v - d0)) / (d1 - d0);
    svg.setAttribute('height', H);
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const yAx = top + 4 * pitch, off = ((4 - bars.length) * pitch) / 2;    // four rows reserved in every setting
    const grid = sv('g', { 'aria-hidden': 'true' });
    for (const t of G.sep_ticks) {
      grid.append(sv('line', { class: t ? 'rs-grid' : 'rs-axis', x1: sx(t), x2: sx(t), y1: top - 8, y2: yAx }), text('rs-tick tick', sx(t), yAx + 18, String(t), 'middle'));
    }
    const marks = sv('g'), ann = sv('g', { 'aria-hidden': 'true' }), geo = {};
    svg.append(grid, marks, ann);
    bars.forEach((b, i) => {
      const a = armOf(s, b.arm), f = F(b.sep), yc = top + off + i * pitch + pitch / 2, xe = sx(f.v), sd = f.sd !== undefined;
      geo[b.arm] = { yc, xe };
      const isRef = a.key === s.ref;
      const g = mark(`rs-s--${a.style}`, `sep-${a.key}`, a.key,
        `${a.name}: block-3 separation ${disp(f.s)} (${sd ? 'mean ± SD over 3 seeds' : 'seed 42'})${isRef ? '; the reference arm' : ''}`,
        disp(f.s), [`${a.name} · block-3 separation`, `${sd ? 'mean ± SD over 3 seeds' : 'seed 42'}${isRef ? ' · reference arm' : ''}`]);
      const bar = sv('path', { class: 'rs-bar', d: barD(x0, xe, yc, th) });
      g.append(sv('rect', { class: 'rs-hit', x: 0, y: yc - pitch / 2, width: W, height: pitch }),
        sv('rect', { class: 'rs-focus', x: 2, y: yc - pitch / 2 + 3, width: W - 4, height: pitch - 6, rx: 6 }),
        sv('path', { class: 'rs-shape', d: shapeD(a.shape, 9, yc, 5) }), bar);
      const ls = names(a);
      ls.forEach((t, k) => g.append(text('rs-lbl', 22, yc + 4.5 + (k - (ls.length - 1) / 2) * 16, t)));
      if (sd) {                                     // SD whisker at the tip (the SDs are 0.01–0.05)
        const lo = sx(f.v - f.sd), hi = sx(f.v + f.sd);
        g.append(sv('path', { class: 'rs-sd', d: `M${r1(lo)} ${r1(yc)}H${r1(hi)}M${r1(lo)} ${r1(yc - 4)}v8M${r1(hi)} ${r1(yc - 4)}v8` }));
      }
      g.append(text('rs-val', sx(f.v + (f.sd || 0)) + 7, yc + 4.5, b.label ? S(b.label) : head(f.s)));
      g._anchor = bar;
      marks.append(g);
    });
    if (s.key === 'c100' && geo.free) ann.append(text('rs-an', geo.free.xe, geo.free.yc - th / 2 - 6, S('abs.sep_up_to_5x'), 'end'));
    if (s.key === 'ctrl') decomp(svg, ann, s.decomp, geo, sx, th, narrow);
  }
  function barD(xa, xb, yc, th, r = 4) {           // square at the baseline, 4 px rounded data end
    const y0 = yc - th / 2, y1 = yc + th / 2;
    r = Math.max(0, Math.min(r, (xb - xa) / 2, th / 2));
    return `M${r1(xa)} ${r1(y0)}H${r1(xb - r)}A${r} ${r} 0 0 1 ${r1(xb)} ${r1(y0 + r)}V${r1(y1 - r)}A${r} ${r} 0 0 1 ${r1(xb - r)} ${r1(y1)}H${r1(xa)}Z`;
  }
  // control: brackets for where MGC's gain over (a) comes from, and the failed registered prediction for (b)
  function decomp(svg, ann, dc, geo, sx, th, narrow) {
    if (geo.b && geo.a) {
      const xp = sx(F(dc.pred).v), yb = geo.b.yc - th / 2;
      const p = S(dc.pred), t = [`registered prediction for (b): below ${p}, failed`, `prediction: below ${p}, failed`, `below ${p}: failed`]
        .find((x, i, all) => i === all.length - 1 || measure(svg, x, 'rs-an') <= xp - 9);
      ann.append(sv('path', { class: 'rs-pred', d: `M${r1(xp)} ${r1(yb - 17)}V${r1(yb - 5)}` }),
        sv('path', { class: 'rs-tipmark', d: `M${r1(xp - 4)} ${r1(yb - 7)}h8l-4 6z` }), text('rs-an', xp - 7, yb - 7, t, 'end'));
    }
    const bracket = (xa, xb, y, long, short, mid) => {
      ann.append(sv('path', { class: 'rs-brk', d: `M${r1(xa)} ${r1(y - 4)}V${r1(y)}H${r1(xb)}V${r1(y - 4)}` }));
      const t = measure(svg, long, 'rs-an') > (mid ? xb - xa : xa - 6) ? short : long;
      ann.append(mid ? text('rs-an', (xa + xb) / 2, y + 14, t, 'middle') : text('rs-an', xa - 5, y + 9, t, 'end'));
    };
    const part = (k, sh) => `${S(dc[k])} (${S(dc[sh])})`;
    if (geo.b && geo.a) bracket(sx(F(dc.from).v), sx(F(dc.mid).v), geo.b.yc + th / 2 + 6, `${part('aux', 'aux_share')} from dropping the auxiliary`, `${part('aux', 'aux_share')} auxiliary dropped`, true);
    if (geo.mgc && geo.b) bracket(sx(F(dc.mid).v), sx(F(dc.to).v), geo.mgc.yc + th / 2 + 6, `${part('comp', 'comp_share')} from the compensation`, `${part('comp', 'comp_share')} compensation`, false);
  }

  /* ---------------------------------------------------------------- (A) paired accuracy changes */
  function renderAcc() {
    const s = st(), W = Math.max(260, Math.floor(accChart.clientWidth)), narrow = W < 470;
    const left = narrow ? 34 : 40, right = W - 4, top = 10, plotH = narrow ? 230 : 236, bot = top + plotH;
    const [y0, y1] = G.delta_axis, [b0, b1] = G.band;
    const sy = (v) => top + (plotH * (y1 - v)) / (y1 - y0);
    const gap = narrow ? 12 : 22, colW = (right - left - gap) / 4;
    const cx = (i) => left + colW * (i + 0.5) + (i >= 2 ? gap : 0);
    const H = bot + (narrow ? 70 : 58);
    const svg = sv('svg', { class: 'rs-svg', width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'group', 'aria-labelledby': 'results-acc-title results-acc-sub' });
    accChart.replaceChildren(svg);
    const furn = sv('g', { 'aria-hidden': 'true' }, sv('rect', { class: 'rs-band', x: left, y: sy(b1), width: right - left, height: sy(b0) - sy(b1) }));
    for (const t of G.delta_ticks) {
      if (t) furn.append(sv('line', { class: t === b0 || t === b1 ? 'rs-edge' : 'rs-grid', x1: left, x2: right, y1: sy(t), y2: sy(t) }));
      furn.append(text('rs-tick tick', left - 7, sy(t) + 4.5, t ? `${t > 0 ? '+' : '−'}${Math.abs(t)}` : '0', 'end'));
    }
    const xs = left + colW * 2 + gap / 2;
    furn.append(sv('line', { class: 'rs-zero', x1: left, x2: right, y1: sy(0), y2: sy(0) }),
      sv('line', { class: 'rs-grid', x1: xs, x2: xs, y1: top - 4, y2: bot + 6 }), text('rs-an rs-onband', left + 6, sy(b1) + 15, `${S('set.pm1pp')} band`));
    svg.append(furn);
    const n = s.cmp.length, slot = narrow ? Math.min(24, colW / n) : Math.min(34, colW / (n + 0.9));   // hit strips tile the column on phones
    const seeds = state.seeds && s.key === 'c100';
    const items = [];
    RO.forEach((r, i) => s.cmp.forEach((c, k) => { if (shown(s, c.arm)) items.push({ r, c, a: armOf(s, c.arm), f: F(c.d[r]), x: cx(i) + (k - (n - 1) / 2) * slot }); }));
    const cols = Object.fromEntries(RO.map((r) => [r, sv('g', { class: `rs-col rs-col--${stg(r)}`, 'data-readout': r })]));
    const refName = UI[s.key][1];
    const refBar = s.bars.find((b) => b.arm === s.ref);
    for (const it of items) {
      const { r, c, a, f, x } = it;
      const leaves = f.lo < b0 || f.hi > b1;
      const leave = f.lo < b0 ? `the CI reaches below −1 pp${stg(r) === 's1' ? ': a Stage-1 cost' : ''}` : 'the CI leaves the ±1 pp band';
      const seedLine = seeds && c.seeds ? `per seed ${L('42/123/456')}: ${F(c.seeds[r]).s}` : null;
      const bar = s.bars.find((b) => b.arm === a.key);
      const name = `${a.name} − ${refName}`;
      const g = mark(`rs-s--${a.style}`, `ci-${a.key}-${r}`, a.key,
        `${name}, ${roName(r)}: ${disp(f.s)} pp${leaves ? `; ${leave}` : ', inside ±1 pp'}${seedLine ? `; ${seedLine}` : ''}`,
        `${disp(f.s)} pp`, [`${name} · ${roName(r)}`, s.n.startsWith('set.') ? `${S(s.n)} paired test predictions` : `paired, ${L(s.n)}`,
          ...(bar && bar.acc && refBar.acc ? [`accuracy (%): ${S(bar.acc[r])} vs. ${S(refBar.acc[r])} (mean ± SD)`] : []),
          ...(leaves ? [leave] : []), ...(seedLine ? [seedLine] : []), ...(s.key === 'c10' ? ['CP-FAIR − γ=0 in Tab. 3 (the paper’s sign)'] : [])]);
      g.append(sv('rect', { class: 'rs-hit', x: x - slot / 2, y: top, width: slot, height: plotH }),
        sv('rect', { class: 'rs-focus', x: x - Math.min(10, slot / 2 - 1), y: sy(f.hi) - 10, width: Math.min(20, slot - 2), height: sy(f.lo) - sy(f.hi) + 20, rx: 6 }),
        sv('path', { class: 'rs-wh', d: `M${r1(x)} ${r1(sy(f.hi))}V${r1(sy(f.lo))}M${r1(x - 4)} ${r1(sy(f.hi))}h8M${r1(x - 4)} ${r1(sy(f.lo))}h8` }));
      if (seedLine) {
        F(c.seeds[r]).s.split('/').forEach((p, j) => g.append(sv('path', { class: 'rs-seed', d: shapeD(a.shape, x + (narrow ? 6 : 8) + j * (narrow ? 3 : 4), sy(num(p)), narrow ? 2.6 : 3) })));
      }
      const m = sv('path', { class: 'rs-shape rs-ring', d: shapeD(a.shape, x, sy(f.v), narrow ? 4.6 : 5.2) });
      g.append(m);
      g._anchor = m;
      if (f.v < b0 || f.v > b1) outLabel(svg, g, it, items, sy, left, right, seedLine ? (narrow ? 18 : 26) : 12);
      cols[r].append(g);
    }
    svg.append(...RO.map((r) => cols[r]));
    if (!items.length) svg.append(text('rs-an rs-onband', (left + right) / 2, sy(0) - 12, 'Every comparison arm is hidden: turn one on above.', 'middle'));
    const labs = sv('g', { 'aria-hidden': 'true' });
    RO.forEach((r, i) => labs.append(text(`rs-xl rs-col--${stg(r)}`, cx(i), bot + 20, r.endsWith('_tta') ? '+ flip' : 'single crop', 'middle')));
    for (const [gx, k, l1, l2] of [[(cx(0) + cx(1)) / 2, 's1', 'Stage 1:', 'FF’s own sum'], [(cx(2) + cx(3)) / 2, 's2', 'Stage 2:', 'probe (not FF)']]) {
      if (narrow) labs.append(text(`rs-grp rs-col--${k}`, gx, bot + 42, l1, 'middle'), text(`rs-grp rs-col--${k}`, gx, bot + 59, l2, 'middle'));
      else labs.append(text(`rs-grp rs-col--${k}`, gx, bot + 44, `${l1} ${l2}`, 'middle'));
    }
    svg.append(labs);
  }
  // direct label of a point outside the band: value + "Stage-1 cost", on the side with room (fewer lines if short)
  function outLabel(svg, g, it, items, sy, left, right, dxR) {
    const { f, x, r } = it, v = head(f.s), y = sy(f.v);
    const opts = stg(r) === 's1' && f.v < G.band[0] ? [[v, 'Stage-1 cost'], [v, 'Stage-1', 'cost'], [v]] : [[v]];
    const place = (ls) => {
      const w = Math.max(...ls.map((t) => measure(svg, t, 'rs-an'))), yT = y - 8, yB = y + 8 + (ls.length - 1) * 15;
      const block = (o) => o !== it && sy(o.f.lo) + 6 > yT && sy(o.f.hi) - 6 < yB;     // whisker crosses the label
      if (x + dxR + w <= Math.min(right, ...items.filter((o) => o.x > x && block(o)).map((o) => o.x - 9))) return 'start';
      if (x - 12 - w >= Math.max(left + 2, ...items.filter((o) => o.x < x && block(o)).map((o) => o.x + 9))) return 'end';
      return null;
    };
    let lines = [v], side = 'start';
    for (const ls of opts) { const p = place(ls); if (p) { lines = ls; side = p; break; } }
    lines.forEach((t, k) => g.append(text(k ? 'rs-an' : 'rs-an rs-strong', side === 'start' ? x + dxR : x - 12, y + 4.5 + k * 15, t, side)));
  }

  /* ---------------------------------------------------------------- (A) text that follows the setting */
  function renderText() {
    const s = st(), ref = UI[s.key][1];
    sepSub.textContent = `block 3 · higher = healthier · ${s.key === 'c10' ? 'seed 42' : 'mean ± SD'}`;
    accTitle.textContent = `Accuracy change vs. ${ref} (pp)`;
    accSub.textContent = s.key === 'c10' ? 'paired 95% CIs · the paper’s sign: cumulative − history-free' : 'paired 95% CIs · below 0: lower accuracy than the reference';
    legend.replaceChildren(...s.cmp.map((c) => {
      const a = armOf(s, c.arm);
      return li(glyph(a.shape, a.style), s.key === 'ctrl' ? `${a.short} − (a)` : `${a.short} − ${ref}`, { 'data-arm': a.key, class: shown(s, a.key) ? null : 'is-off' });
    }), ...(state.seeds && s.key === 'c100' ? [li(glyph('circle', 'seedkey'), 'per seed (hollow)')] : []));
    chips.replaceChildren(...s.arms.filter((a) => a.key !== s.ref).map((a) => {
      const b = el('button', { type: 'button', class: 'chip rs-chip', id: `results-chip-${a.key}`, 'data-arm': a.key, 'aria-pressed': String(!state.hidden.has(a.key)) },
        glyph(a.shape, a.style), a.short);
      b.addEventListener('click', () => toggleArm(a.key));
      return b;
    }));
    seedsBox.disabled = s.key !== 'c100';
    seedsBox.checked = state.seeds;
    seedsCtl.classList.toggle('is-disabled', seedsBox.disabled);
    const reading = {
      c100: `separation goes from ${S('sep.c100.cum')} (cumulative) to ${S('sep.c100.gated')} (hardness-gated), ${S('sep.c100.g0')} (history-free) and ${S('sep.c100.mgc')} (MGC), yet every history-free and hardness-gated CI stays inside ${S('set.pm1pp')}; MGC’s two Stage-1 CIs reach below −1 pp, a Stage-1 cost the probe recovers.`,
      c10: `at seed 42, separation is ${S('s45.c10_sep_g0')} history-free vs. ${S('s45.c10_sep_cpfair')} cumulative, yet every paired CI stays inside ${S('set.pm1pp')}.`,
      ctrl: `across the four arms separation spans ${S('s45.ctrl_range')}, yet every paired 95% CI lies inside ${S('set.pm1pp')}.`,
    }[s.key];
    take.replaceChildren(el('b', { text: 'Reading: ' }), reading);
    sepBelow.replaceChildren(...below(s));
    notes.replaceChildren(...settingNotes(s));
    srcA.textContent = {
      c100: `CIFAR-100 · 4 blocks, width 256 · 3 seeds · separation: mean ± SD · accuracy: paired, 3 seeds × 10,000 test images, 95% CIs conditional on the trained models · Table 3${state.seeds ? ' · per seed: Tab. 12, App. B.1' : ''}`,
      c10: `CIFAR-10 · 4 blocks, width 256, 32 experts · separation: seed 42 · accuracy: paired, ${L('3 seeds × 10k test images')}, 95% CIs · Sec. 4.5, Table 3`,
      ctrl: `CIFAR-10 · 4 blocks, width 128 · 3 seeds per arm · SAM on · separation: mean ± SD · accuracy: paired, ${S('set.n30000')} test predictions, 95% CIs · Tab. 6, App. B.1.1`,
    }[s.key];
  }
  function below(s) {
    if (s.key === 'c100') {
      const go = el('button', { type: 'button', class: 'rs-link', id: 'results-goto-ctrl', text: 'Open the CIFAR-10 control' });
      go.addEventListener('click', () => { setSetting('ctrl'); setCtl.inputs[2].focus(); });
      return [el('div', { class: 'callout rs-caveat' }, el('p', {}, el('span', { class: 'rs-tag' }, glyph('diamond', 'mgc'), 'MGC caveat'),
        `MGC replaces the depth-scaled auxiliary that the history rules keep; in a CIFAR-10 control, ${S('s45.ctrl_92')} of MGC’s separation gain came from dropping it, not the compensation; its CIFAR-100 gain is not decomposed.`), go)];
    }
    if (s.key === 'c10') {
      return [el('p', { class: 'rs-small' }, 'One seed (42), so no SD whiskers. Both arms have 4 blocks of width 256 with 32 experts. ',
        'The history-free seed-42 run used λ', el('sub', { text: 'curr' }), ' = λ', el('sub', { text: 'depth' }),
        ` = 0 (no depth-scaled auxiliary and no depth-order term), while the cumulative arm keeps the auxiliary; ${L('seeds 123/456 used 0.25/0.2')} (Sec. 4.1, App. Tab. 22).`)];
    }
    return [el('ul', { class: 'rs-keys' }, ...s.arms.map((a) => li(glyph(a.shape, a.style), a.name)))];
  }
  function settingNotes(s) {
    if (s.key === 'c100') return [];
    if (s.key === 'c10') return [el('p', { class: 'rs-note', text: `Separation is from seed 42 (Sec. 4.5); the accuracy change pools ${L('3 seeds × 10k test images')} (Table 3). Both include the history-free seed-42 run, trained without the depth-scaled auxiliary and the depth-order term (Sec. 4.1, App. Tab. 22).` })];
    const dc = s.decomp;
    return [el('p', { class: 'rs-note', text: `Against a cumulative arm trained with SAM like MGC, removing the depth-scaled auxiliary alone lifts block-3 separation from ${S(dc.from)} to ${S(dc.mid)}. Of MGC’s gain of ${S(dc.gain)}, ${S(dc.aux)} (${S(dc.aux_share)}) comes from that removal and ${S(dc.comp)} (${S(dc.comp_share)}) from the compensation. Our prediction, registered before the runs, that this arm would stay below ${S(dc.pred)} failed. Every paired 95% CI lies inside ${S('set.pm1pp')}.` }),
      el('p', { class: 'rs-note', text: 'The runs mix GPU types and software stacks, so (b) minus (a) is confounded with them; the separation effects are large against this variation, the accuracy differences are not.' })];
  }
  const liveSentence = (s) => ({
    c100: `CIFAR-100: block-3 separation ${S('sep.c100.cum')} to ${S('sep.c100.mgc')}; history-free and hardness-gated CIs inside ${S('set.pm1pp')}; MGC’s two Stage-1 CIs reach below −1 pp.`,
    c10: `CIFAR-10, 32 experts: separation ${S('s45.c10_sep_g0')} (history-free) vs. ${S('s45.c10_sep_cpfair')} (cumulative), seed 42; every paired CI inside ${S('set.pm1pp')}.`,
    ctrl: `CIFAR-10 control: separation ${S('s45.ctrl_range')} across the four arms; ${S('b11.aux_413#4.13')} of MGC’s ${S('b11.gain_450#4.50')} gain comes from dropping the auxiliary; every paired 95% CI inside ${S('set.pm1pp')}.`,
  })[s.key];

  /* ---------------------------------------------------------------- (B) what moves accuracy */
  const ROWS = {
    rule: ['Training rule', 'history removed vs. cumulative (FF)', () => L('under 1 pp')],
    arch: ['Architecture', 'FF on a plain CNN → FF on our stripped backbone', () => `${S('s1.ff_plain_cnn')} → ${S('s1.ff_stripped')}`],
    aug: ['Augmentation', 'backprop, weak → strong', () => S('s1.aug_6_4')],
  };
  const WHO = {                                     // tooltip label and source per dot
    't1.cpfair.c2_test_no_tta': ['FF, cumulative (γ = 0.7), 32 experts', '4 seeds · Tab. 1'],
    't1.gamma0.c2_test_no_tta': ['FF, history-free (γ = 0), 32 experts', '3 seeds · Tab. 1', 'seed 42 trained without the depth-scaled auxiliary and the depth-order term (Sec. 4.1, App. Tab. 22)'],
    't2.cnn_ff.c1_test_no_tta': ['FF on a plain CNN', '3 seeds · Tab. 2'],
    't2.stripped_ff.c1_test_no_tta': ['FF on our stripped backbone', '1 seed · Tab. 2'],
    't2.cnn_bp.c1_test_no_tta': ['Backprop on the same plain CNN', '3 seeds · Tab. 2'],
    't2.stripped_bp_weak.c1_test_no_tta': ['Backprop on the stripped backbone, weak augmentation', '3 seeds · Tab. 2'],
    't2.stripped_bp_strong.c1_test_no_tta': ['Backprop on the stripped backbone, strong augmentation', '3 seeds · Tab. 2'],
  };
  const mvRows = D.suspects.map((row) => {
    const [name, desc, eff] = ROWS[row.key], track = el('div', { class: 'rs-mv__track' });
    const note = row.extra ? el('span', { class: 'rs-mv__note', text: `FF vs. BP on the same plain CNN: ${S(row.extra.note)}` }) : null;
    mv.append(el('div', { class: 'rs-mv__row' }, el('div', { class: 'rs-mv__name' }, el('b', { text: name }), el('span', { text: desc })), track,
      el('div', { class: 'rs-mv__effect' }, el('span', { class: 'rs-mv__big', text: eff() }), note)));
    return { row, track, note };
  });
  const mvAxis = el('div', { class: 'rs-mv__track' });
  mv.append(el('div', { class: 'rs-mv__row rs-mv__row--axis', 'aria-hidden': 'true' }, el('div', { class: 'rs-mv__name' }), mvAxis, el('div', { class: 'rs-mv__effect' })));

  function renderMoves() {
    const W = Math.max(220, Math.floor(mvRows[0].track.clientWidth)), narrow = W < 420, [a0, a1] = G.acc_axis;
    const sx = (v) => 8 + ((W - 16) * (v - a0)) / (a1 - a0), H = 66, yT = 36;
    for (const { row, track, note } of mvRows) {
      const svg = sv('svg', { class: 'rs-svg', width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'group', 'aria-label': `${ROWS[row.key][0]}: ${ROWS[row.key][1]}` });
      track.replaceChildren(svg);
      const grid = sv('g', { 'aria-hidden': 'true' });
      for (const t of G.acc_ticks) grid.append(sv('line', { class: t ? 'rs-grid' : 'rs-axis', x1: sx(t), x2: sx(t), y1: 4, y2: H - 2 }));
      svg.append(grid);
      const marks = sv('g');
      const dot = (end, start, x, y, lab) => {      // one dumbbell end: mark, label, tooltip
        const f = F(end.fact), [who, src, note] = WHO[end.fact];
        const style = end.arm === 'cum' || end.arm === 'free' ? end.arm : `${end.arm}${start ? '-start' : ''}`;
        const g = mark(`rs-dot rs-s--${style}`, `mv-${end.fact}`, null, `${who}: ${disp(f.s)} (CIFAR-10 test accuracy in percent, single crop; ${src})${note ? `; ${note}` : ''}`,
          disp(f.s), [who, `CIFAR-10 test accuracy (%), single crop · ${src}`, ...(note ? [note] : [])]);
        const m = sv('path', { class: 'rs-shape rs-ring', d: shapeD(end.arm === 'cum' || end.arm === 'bp' ? 'square' : 'circle', x, y, 5.6) });
        g.append(sv('rect', { class: 'rs-hit', x: x - 12, y: y - 12, width: 24, height: 24 }), sv('rect', { class: 'rs-focus', x: x - 10, y: y - 10, width: 20, height: 20, rx: 5 }), m,
          text('rs-val', lab[1], lab[2], lab[0], lab[3]));
        g._anchor = m;
        marks.append(g);
      };
      const fa = F(row.from.fact), fb = F(row.to.fact), xa = sx(fa.v), xb = sx(fb.v), sA = S(row.from.fact), sB = S(row.to.fact);
      if (row.key !== 'rule' && xb - xa > 30) {
        svg.append(sv('path', { class: 'rs-gap', d: `M${r1(xa + 7)} ${yT}H${r1(xb - 9)}M${r1(xb - 13)} ${yT - 4}l5 4-5 4`, 'aria-hidden': 'true' }));
      }
      if (row.key === 'rule') {                     // the two ends coincide at this scale: offset vertically
        dot(row.from, true, xa, yT - 8, [narrow ? `cum. ${sA}` : `cumulative ${sA}`, xa - 12, yT - 3.5, 'end']);
        dot(row.to, false, xb, yT + 8, [narrow ? `hist.-free ${sB}` : `history-free ${sB}`, xa - 12, yT + 12.5, 'end']);
      } else if (row.key === 'arch') {
        dot(row.from, true, xa, yT, [sA, xa, yT - 13, 'middle']);
        dot(row.to, false, xb, yT, [`${sB} (1 seed)`, Math.min(W - 2, xb + 30), yT - 13, 'end']);
        if (state.bp) {
          const xe = sx(F(row.extra.fact).v), sE = S(row.extra.fact);
          dot(row.extra, false, xe, yT + 17, [narrow ? `BP ${sE}` : `BP, same plain CNN ${sE}`, xe - 12, yT + 21.5, 'end']);
        }
      } else {
        dot(row.from, true, xa, yT, [`weak ${sA}`, xa - 12, yT + 4.5, 'end']);
        dot(row.to, false, xb, yT, [`strong ${sB}`, Math.min(W - 2, xb + 40), yT - 13, 'end']);
      }
      svg.append(marks);
      if (note) note.classList.toggle('is-hidden', !state.bp);
    }
    const ax = sv('svg', { class: 'rs-svg', width: W, height: 26, viewBox: `0 0 ${W} 26`, 'aria-hidden': 'true' }, sv('line', { class: 'rs-axis', x1: sx(a0), x2: sx(a1), y1: 1, y2: 1 }));
    for (const t of G.acc_ticks) ax.append(sv('line', { class: 'rs-axis', x1: sx(t), x2: sx(t), y1: 1, y2: 5 }), text('rs-tick tick', sx(t), 20, String(t), t === a0 ? 'start' : t === a1 ? 'end' : 'middle'));
    mvAxis.replaceChildren(ax, el('p', { class: 'rs-axislabel', text: 'CIFAR-10 test accuracy, single crop (%)' }));
  }

  /* ---------------------------------------------------------------- render, events */
  function renderA(keep) {
    const a = document.activeElement, had = keep && figA.contains(a) && a.classList.contains('rs-mark'), inSep = had && sepChart.contains(a);
    figA.dataset.readout = state.readout;
    figA.dataset.setting = state.setting;
    renderSep();
    renderAcc();
    restoreSep(inSep);
    restoreAcc(had && !inSep);
    if (!had) setHL(null);
  }
  function renderB(keep) {
    const had = keep && mv.contains(document.activeElement);
    renderMoves();
    restoreMv(had);
  }
  const say = (t) => { liveRegion.textContent = t; };
  function setSetting(k) {
    if (!SETS[k] || k === state.setting) return;
    state.setting = k;
    for (const x of setCtl.inputs) x.checked = x.value === k;
    hideTip();
    renderText();
    renderA(false);
    say(liveSentence(st()));
  }
  function toggleArm(k) {
    if (!state.hidden.delete(k)) state.hidden.add(k);
    hideTip();
    renderText();
    renderA(false);
    const b = document.getElementById(`results-chip-${k}`);
    if (b) b.focus();
    say(`${armOf(st(), k).name} ${state.hidden.has(k) ? 'hidden in' : 'shown in'} both charts.`);
  }
  for (const x of setCtl.inputs) x.addEventListener('change', () => { if (x.checked) setSetting(x.value); });
  for (const x of roCtl.inputs) {
    x.addEventListener('change', () => {
      if (!x.checked) return;
      state.readout = figA.dataset.readout = x.value;
      say({ all: 'Readout: all four columns.', s1: 'Readout: Stage 1, FF’s own sum; the Stage 2 columns are dimmed.',
        s2: 'Readout: Stage 2, probe (not FF); the Stage 1 columns are dimmed.' }[x.value]);
    });
  }
  seedsBox.addEventListener('change', () => {
    state.seeds = seedsBox.checked;
    renderText();
    renderA(false);
    say(state.seeds ? `Per-seed differences shown as hollow markers (seeds ${L('42/123/456')}).` : 'Per-seed differences hidden.');
  });
  bpChip.addEventListener('click', () => {
    state.bp = !state.bp;
    bpChip.setAttribute('aria-pressed', String(state.bp));
    renderB(false);
    const x = D.suspects.find((r) => r.extra).extra;
    say(state.bp ? `Backprop on the same plain CNN shown: ${S(x.fact)}; FF vs. BP on the same plain CNN: ${S(x.note)}.` : 'Backprop on the same plain CNN hidden.');
  });
  for (const box of [legend, chips]) {              // a legend item or chip previews its arm in both charts
    box.addEventListener('pointerover', (e) => { const t = e.target.closest('[data-arm]'); if (t) setHL(t.dataset.arm); });
    box.addEventListener('pointerout', () => setHL(null));
  }
  // redraw at the container width (re-laid out, not scaled), throttled with requestAnimationFrame
  const width = () => [sepChart.clientWidth, accChart.clientWidth, mvRows[0].track.clientWidth].map(Math.floor);
  let raf = 0, last = [];
  const ro = new ResizeObserver(() => {
    if (raf) return;
    raf = requestAnimationFrame(() => {
      raf = 0;
      const w = width();
      if (w[0] !== last[0] || w[1] !== last[1]) { hideTip(); renderA(true); }
      if (w[2] !== last[2]) renderB(true);
      last = w;
    });
  });
  // phones (< 640 px): Setting and Readout stay above the charts; the arm chips, the seed switch and the reading go
  // under them (a tapped setting changes the chart right below it). Focus is kept when nodes move.
  const moreBox = el('div', { class: 'controls rs-controls--more' });
  const phone = window.matchMedia('(max-width: 639px)');
  function placeA() {
    const a = document.activeElement, had = a && (armsCtl.contains(a) || seedsCtl.contains(a));
    if (phone.matches) {
      if (armsCtl.parentNode !== moreBox) {
        moreBox.append(armsCtl, seedsCtl);
        figA.insertBefore(moreBox, panels.nextSibling);
        figA.insertBefore(take, moreBox.nextSibling);
      }
    } else if (armsCtl.parentNode !== ctlBox) {
      ctlBox.append(armsCtl, seedsCtl);
      moreBox.remove();
      figA.insertBefore(take, panels);
    }
    if (had) a.focus({ preventScroll: true });
  }
  placeA();
  phone.addEventListener('change', () => { try { placeA(); } catch (err) { console.error(`[${NAME}]`, err); } });
  renderText();
  root.dataset.state = 'live';                       // show [data-live] first so the charts can measure their width
  renderA(false);
  renderB(false);
  last = width();
  for (const x of [sepChart, accChart, mvRows[0].track]) ro.observe(x);
}
