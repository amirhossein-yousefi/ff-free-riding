// Component "redistribution" (SPEC_FINAL.md §5.4): Proposition 3.2 in your hands. ES module, no imports.
//
// (A) Re-split the sum (toy numbers): the shared toy network (window.SITE_DATA.site.toy) as three stacked columns
//     (cat, dog, ship; blocks 0-3, block 3 outlined) over a blocks x labels grid of the toy values, beside the
//     per-block margins cat - dog and their sum. Mode "Re-split (keep every sum)" moves goodness for "dog" from
//     block 0 to block 3: every sum, and so the prediction, stays; block 3's margin crosses zero. Mode "Change
//     one sum" adds goodness for "dog" at block 3 only: above 1.7 dog's sum passes cat's and the prediction changes.
// (B) Retrained repairs are not re-splits (measured): the printed re-split endpoints (14% / 98% of the sum,
//     0 / 10,000 predictions change) beside the churn of retrained variants (13-26%; ~13% between history rules,
//     a 10 x 10 waffle; MGC vs. cumulative 22.7-26.2%), and a seed selector for "which one is right?" counts.
// On phones (< 640 px) the toy readouts (predictions changed, the sums, block 3 alone) move right under the slider.
//
// Contract (SPEC_FINAL §4.4): renders into [data-live] of #viz-redistribution with textContent only, then the
// caller marks the root live; on any exception: console.error("[redistribution]", err) and data-state="broken",
// so the static fallback stays. Never touches anything outside the root (it reads the shell's shape sprite by
// reference). Printed numbers come only from SITE_DATA.redistribution.facts[*].s (or verbatim substrings);
// toy values are computed here and render only inside [data-illus]. Colors are CSS classes bound to tokens,
// so theme changes need no script. Ids start with "redistribution-". One polite live region.
const NAME = 'redistribution';
const SVGNS = 'http://www.w3.org/2000/svg';
const MINUS = '−';
const NBSP = ' ';

/* ------------------------------------------------------------------ small DOM helpers (textContent only) */
function el(tag, attrs, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') n.className = v;
    else n.setAttribute(k, v === true ? '' : String(v));
  }
  for (const c of kids) if (c !== null && c !== undefined && c !== false) n.append(c);
  return n;
}
function sv(tag, attrs, parent) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v !== null && v !== undefined) n.setAttribute(k, String(v));
  if (parent) parent.appendChild(n);
  return n;
}
function stext(parent, x, y, str, cls, anchor) {
  const t = sv('text', { x: r1(x), y: r1(y), class: cls, 'text-anchor': anchor || 'start' }, parent);
  t.textContent = str;
  return t;
}
const r1 = (v) => Math.round(v * 10) / 10;
function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
// drawing width: whole CSS pixels, rounded down, so the SVG is never scaled below 1:1 (text stays >= 12 px)
const boxWidth = (box) => Math.floor(box.getBoundingClientRect().width);
// bar with a rounded data end (radius r) on the side `end` ('top', 'right' or 'left') and a square base
function barPath(x, y, w, h, r, end) {
  if (w <= 0 || h <= 0) return '';
  r = Math.max(0, Math.min(r, w / 2, h / 2, end === 'top' ? h : w));
  const X = r1(x), Y = r1(y), W = r1(w), H = r1(h);
  if (end === 'top') return `M${X},${r1(y + h)}V${r1(y + r)}Q${X},${Y} ${r1(x + r)},${Y}H${r1(x + w - r)}Q${r1(x + w)},${Y} ${r1(x + w)},${r1(y + r)}V${r1(y + h)}Z`;
  if (end === 'right') return `M${X},${Y}H${r1(x + w - r)}Q${r1(x + w)},${Y} ${r1(x + w)},${r1(y + r)}V${r1(y + h - r)}Q${r1(x + w)},${r1(y + h)} ${r1(x + w - r)},${r1(y + h)}H${X}Z`;
  if (end === 'left') return `M${r1(x + w)},${Y}H${r1(x + r)}Q${X},${Y} ${X},${r1(y + r)}V${r1(y + h - r)}Q${X},${r1(y + h)} ${r1(x + r)},${r1(y + h)}H${r1(x + w)}Z`;
  return `M${X},${Y}h${W}v${H}h${-W}z`;
}
// rough width of a 13 px label (for collision checks only)
const textW = (s, px = 13) => s.length * px * 0.56;

/* ------------------------------------------------------------------ the component */
function mount(root) {
  const all = window.SITE_DATA || {};
  const D = all[NAME];
  const SITE = all.site;
  if (!D || !D.facts || !D.seeds || !D.geometry || !D.toy) throw new Error(`window.SITE_DATA.${NAME} is missing or incomplete (assets/data/${NAME}.js)`);
  if (!SITE || !SITE.toy || !SITE.arms || !SITE.shapes) throw new Error('window.SITE_DATA.site is missing (assets/data/site_data.js)');
  const host = root.querySelector('[data-live]');
  if (!host) throw new Error('the mount has no [data-live] element');

  // printed numbers: only facts[key].s, never computed
  const fact = (key) => {
    const e = D.facts[key];
    if (!e || typeof e.s !== 'string') throw new Error(`fact ${key} is missing from ${NAME}.js`);
    return e.s;
  };
  const splitKey = (key) => { const at = key.indexOf('#'); return at < 0 ? [key, null] : [key.slice(0, at), key.slice(at + 1)]; };
  function setFact(span, key) {
    const [id, sub] = splitKey(key);
    span.setAttribute('data-fact', id);
    if (sub === null) span.removeAttribute('data-sub'); else span.setAttribute('data-sub', sub);
    span.textContent = fact(key);
    return span;
  }
  const factSpan = (key, cls) => setFact(el('span', { class: cls || 'nw' }), key);
  // a printed string with each seed's part wrapped (the selected seed is highlighted); text stays verbatim
  function seedSplit(fullKey, partKeys) {
    const full = fact(fullKey);
    const outer = el('span', { 'data-fact': fullKey });
    let pos = 0;
    partKeys.forEach((pk, i) => {
      const part = fact(pk);
      const at = full.indexOf(part, pos);
      if (at < 0) throw new Error(`${pk} does not occur in ${fullKey} after position ${pos}`);
      if (at > pos) outer.append(full.slice(pos, at));
      outer.append(el('span', { class: 'rd-seedval', 'data-seed': D.seeds.ids[i] }, part));
      pos = at + part.length;
    });
    if (pos < full.length) outer.append(full.slice(pos));
    if (outer.textContent !== full) throw new Error(`${fullKey} would not render verbatim`);
    return outer;
  }
  const arm = (key) => {
    const a = SITE.arms[key];
    if (!a) throw new Error(`unknown arm ${key}`);
    return a;
  };
  function glyph(key) {
    const a = arm(key);
    const s = sv('svg', { class: `glyph ${a.glyph}`, viewBox: '0 0 10 10', 'aria-hidden': 'true', focusable: 'false' });
    sv('use', { href: `#${SITE.shapes[a.shape]}` }, s);
    return s;
  }
  const armChip = (key, text) => el('span', { class: 'arm' }, glyph(key), text || arm(key).short);

  /* ---------------- toy model, in integer hundredths (exact; no float drift) */
  const TOY = SITE.toy;
  const T = D.toy;
  const LABELS = TOY.labels;
  const TRUE = TOY.true_label;
  const RIVAL = T.label;
  const hund = (x) => Math.round(Number(x) * 100);
  const BASE = Object.fromEntries(LABELS.map((l) => [l, TOY.goodness[l].map(hund)]));
  const NB = BASE[TRUE].length;
  const B3 = NB - 1;
  if (!BASE[RIVAL] || T.to !== B3) throw new Error('toy settings do not fit the shared toy network');
  const total = (arr) => arr.reduce((s, v) => s + v, 0);
  const argmax = (sum) => LABELS.reduce((best, l) => (sum[l] > sum[best] ? l : best), LABELS[0]);
  const PRED0 = argmax(Object.fromEntries(LABELS.map((l) => [l, total(BASE[l])])));

  // toy values: at most two decimals, trailing zeros dropped, U+2212 minus (SPEC_FINAL 5.0)
  const fmt = (h) => {
    let s = (Math.round(h) / 100).toFixed(2).replace(/0+$/, '').replace(/\.$/, '');
    if (s === '-0') s = '0';
    return s.replace('-', MINUS);
  };
  const sfmt = (h) => (h > 0 ? '+' : '') + fmt(h);

  const st = { mode: 'resplit', q: hund(T.resplit.value), a: hund(T.add.value), seed: D.seeds.ids[0] };
  function model() {
    const g = Object.fromEntries(LABELS.map((l) => [l, BASE[l].slice()]));
    let moved = 0;
    let added = 0;
    if (st.mode === 'resplit') { moved = st.q; g[RIVAL][T.from] -= moved; g[RIVAL][T.to] += moved; } else { added = st.a; g[RIVAL][T.to] += added; }
    const sum = Object.fromEntries(LABELS.map((l) => [l, total(g[l])]));
    const pred = argmax(sum);
    return {
      g, sum, pred, moved, added,
      margin: g[TRUE].map((v, d) => v - g[RIVAL][d]),
      msum: sum[TRUE] - sum[RIVAL],
      changed: pred === PRED0 ? 0 : 1,
    };
  }

  /* ================================================================ (A) Re-split the sum (toy) */
  const A = el('figure', { class: 'fig fig--toy rd-a', 'aria-labelledby': `${NAME}-a-title` });
  A.append(el('div', { class: 'fig__head' },
    el('h3', { class: 'fig__title', id: `${NAME}-a-title` }, 'Re-split the sum'),
    el('span', { class: 'badge badge--toy' }, `Toy numbers · Prop.${NBSP}3.2`)));

  // controls: mode (radios) and one range per mode
  const controls = el('div', { class: 'controls rd-controls', 'data-illus': '' });
  const modeSeg = el('div', { class: 'seg' });
  const modes = [['resplit', 'Re-split (keep every sum)', 'Re-split'], ['sum', 'Change one sum', null]];
  for (const [val, text, short] of modes) {
    const id = `${NAME}-mode-${val}`;
    modeSeg.append(el('input', { type: 'radio', name: `${NAME}-mode`, id, value: val, checked: val === st.mode }),
      short ? el('label', { for: id }, el('span', { class: 'seg__long' }, text), el('span', { class: 'seg__short', 'aria-hidden': 'true' }, short))
        : el('label', { for: id }, text));
  }
  controls.append(el('fieldset', { class: 'ctl rd-mode' }, el('legend', { class: 'ctl__label' }, 'Mode'), modeSeg));

  function rangeField(key, cfg, labelText, hintText, crossText) {
    const id = `${NAME}-${key}`;
    const out = el('output', { class: 'rd-out', id: `${id}-out`, for: id });
    const input = el('input', {
      type: 'range', class: 'range', id, min: cfg.min, max: cfg.max, step: cfg.step, value: cfg.value,
      'aria-describedby': `${id}-out ${id}-hint`,
    });
    const span = cfg.max - cfg.min;
    const ticks = el('div', { class: 'rd-ticks', 'aria-hidden': 'true' });
    const tick = (v, text, cls) => {
      const t = el('span', { class: `rd-tick ${cls || ''}` }, text);
      t.style.setProperty('--x', String((v - cfg.min) / span));
      ticks.append(t);
    };
    tick(cfg.min, fmt(hund(cfg.min)), 'rd-tick--start');
    tick(cfg.cross, `${fmt(hund(cfg.cross))}: ${crossText}`, 'rd-tick--mark');
    tick(cfg.max, fmt(hund(cfg.max)), 'rd-tick--end');
    const field = el('div', { class: 'range-field rd-range' },
      el('div', { class: 'range-field__top' }, el('label', { for: id }, labelText), out),
      input, ticks,
      el('p', { class: 'range-field__hint', id: `${id}-hint` }, hintText));
    return { field, input, out, cfg };
  }
  const RQ = rangeField('q', T.resplit, `Goodness for “${RIVAL}” moved from block ${T.from} to block ${T.to}`,
    `Toy values. ${cap(RIVAL)}’s block ${T.from} gives what its block ${T.to} gains, so every label keeps its sum.`,
    `block ${B3} flips`);
  const RA = rangeField('add', T.add, `Goodness added for “${RIVAL}” at block ${T.to} only`,
    `Toy values. Only ${RIVAL}’s sum grows; ${TRUE}’s stays put.`, 'sums cross');
  controls.append(RQ.field, RA.field);
  A.append(controls);

  // two panels: stacked columns over the goodness grid | margins per block and summed
  const panels = el('div', { class: 'rd-a__grid', 'data-illus': '' });
  const badge = el('p', { class: 'rd-pred' });
  const colsBox = el('div', { class: 'rd-chart rd-chart--cols' });
  const colsSvg = sv('svg', { class: 'rd-svg', role: 'img', 'aria-labelledby': `${NAME}-cols-desc`, focusable: 'false' });
  colsBox.append(colsSvg);
  const colsDesc = el('p', { class: 'vh', id: `${NAME}-cols-desc` });
  panels.append(el('div', { class: 'rd-panel' },
    el('div', { class: 'rd-panel__head' }, el('h4', { class: 'rd-panel__title' }, 'Goodness per label and block (toy)'), badge),
    colsBox, colsDesc,
    el('p', { class: 'rd-note' }, `Each column stacks one label’s goodness, block 0 at the bottom; its height is the sum FF compares.`)));
  const marBox = el('div', { class: 'rd-chart rd-chart--mar' });
  const marSvg = sv('svg', { class: 'rd-svg', role: 'img', 'aria-labelledby': `${NAME}-mar-desc`, focusable: 'false' });
  marBox.append(marSvg);
  const marDesc = el('p', { class: 'vh', id: `${NAME}-mar-desc` });
  const marNote = el('p', { class: 'rd-note' });
  panels.append(el('div', { class: 'rd-panel' },
    el('div', { class: 'rd-panel__head' }, el('h4', { class: 'rd-panel__title' }, `Margin, ${TRUE} ${MINUS} ${RIVAL} (toy)`)),
    marBox, marDesc, marNote));
  A.append(panels);

  // readouts (toy): the counter, the sums, block 3 alone
  const tray = el('div', { class: 'tray rd-tray' });
  const trayItem = (key) => {
    const v = el('span', { class: 'tray__val' });
    tray.append(el('div', { 'data-illus': '' }, el('span', { class: 'tray__key' }, key), v));
    return v;
  };
  const tChanged = trayItem('Predictions changed (toy)');
  const tSums = trayItem('Every label’s sum (toy)');
  const tB3 = trayItem(`Block ${B3} alone (toy)`);
  A.append(tray);
  // phones: the readouts follow the slider (they answer it), then the two charts
  const phone = window.matchMedia('(max-width: 639px)');
  function placeTray() {
    const want = phone.matches ? panels : null;               // before the panels on phones, after them otherwise
    if (want ? tray.nextSibling !== panels : tray.previousSibling !== panels) A.insertBefore(tray, want ? panels : panels.nextSibling);
  }
  placeTray();
  phone.addEventListener('change', () => { try { placeTray(); } catch (err) { console.error(`[${NAME}]`, err); } });

  A.append(el('figcaption', { class: 'fig__caption' },
    el('i', { class: 'fig__lead' }, 'Toy numbers'),
    ` to illustrate Proposition${NBSP}3.2: moving goodness between blocks while every label keeps its sum changes no prediction, yet it moves the per-block margins. Change a sum, and the prediction can change.`,
    el('span', { class: 'rd-quote' }, '“Per-block margin diagnostics are therefore not identifiable from summed-goodness predictions alone.”',
      el('span', { class: 'rd-quote__src' }, ` Proposition${NBSP}3.2`))));

  /* ================================================================ (B) Retrained repairs are not re-splits (measured) */
  const B = el('figure', { class: 'fig rd-b', 'aria-labelledby': `${NAME}-b-title` });
  B.append(el('div', { class: 'fig__head' },
    el('h3', { class: 'fig__title', id: `${NAME}-b-title` }, 'Retrained repairs are not re-splits'),
    el('span', { class: 'badge badge--measured' }, 'Measured')));

  // left: same network, goodness re-split (the two printed endpoints only)
  const strip = el('div', { class: 'rd-strip' });
  D.geometry.strip.forEach((s, i) => {
    const bar = el('div', { class: 'rd-strip__bar', 'aria-hidden': 'true' },
      el('span', { class: 'rd-strip__rest' }, i === 0 ? `blocks 0–${B3 - 1}` : ''),
      el('span', { class: 'rd-strip__b3' }, i === 0 ? '' : `block ${B3}`));
    bar.style.setProperty('--b3', String(s.block3 / 100));
    strip.append(el('div', { class: 'rd-strip__row' }, bar,
      el('p', { class: 'rd-strip__lbl' }, `block ${B3}: `, factSpan(s.key, 'nw rd-strong'), ' of sum')));
    if (i === 0) strip.append(el('p', { class: 'rd-strip__step' }, el('span', { 'aria-hidden': 'true' }, '↓ '), 're-split'));
  });
  const left = el('div', { class: 'rd-b__col' },
    el('h4', { class: 'rd-b__title' }, 'Same network, goodness re-split'),
    el('p', { class: 'rd-b__sub' }, 'goodness moved between blocks'),
    strip,
    el('p', { class: 'rd-b__keep' }, 'every label keeps its sum'),
    el('p', { class: 'rd-big' }, factSpan('s3.flips#0'), el('span', { class: 'rd-big__sep' }, ' / '), factSpan('s3.flips#10,000')),
    el('p', { class: 'rd-big__cap' }, 'predictions change'),
    el('p', { class: 'src' }, 'CIFAR-100 · cumulative model · seed 42'));

  // right: retrained variants (waffle for ∼13% between history rules; MGC vs. cumulative)
  const churn = fact('s3.churn_13');
  const churnDigits = (churn.match(/\d+/) || [''])[0];
  const W = D.geometry.waffle;
  const waffle = el('div', {
    class: 'rd-waffle', role: 'img', tabindex: '0',
    'aria-label': `about ${churnDigits} in 100 test predictions change`, 'aria-describedby': `${NAME}-trio-cap`,
  });
  for (let i = 0; i < W.cells; i++) waffle.append(el('span', { class: i < W.marked ? 'rd-cell rd-cell--on' : 'rd-cell' }));
  const right = el('div', { class: 'rd-b__col' },
    el('h4', { class: 'rd-b__title' }, 'Retrained with a repair'),
    el('p', { class: 'rd-b__sub' }, 'Retrained variants disagree on ', factSpan('abs.churn_13_26'), ' of test predictions.'),
    el('div', { class: 'rd-trio' }, waffle,
      el('div', { class: 'rd-trio__text' },
        el('p', { class: 'rd-big' }, factSpan('s3.churn_13')),
        el('p', { class: 'rd-big__cap', id: `${NAME}-trio-cap` }, 'of test predictions change between any two history rules (probe + flip)'),
        el('p', { class: 'rd-b__small' }, '(', factSpan('b7.trio_s2tta'), '; ', factSpan('b7.trio_s1'), ' for FF’s own sum)'))),
    el('div', { class: 'rd-mgc' },
      el('p', { class: 'rd-mgc__head' }, armChip('mgc'), ' vs. ', armChip('cum'), ': ', factSpan('s3.churn_mgc', 'nw rd-strong')),
      el('p', { class: 'rd-b__small' }, 'FF’s own sum ', seedSplit('b7.mgc_s1_seeds', D.seeds.mgc_s1),
        '; probe + flip ', seedSplit('b7.mgc_s2tta_seeds', D.seeds.mgc_s2tta), ` (seeds ${D.seeds.ids.join('/')})`)));
  B.append(el('div', { class: 'rd-b__row' }, left, right));

  // row 2: when two variants disagree, which one is right? (per seed)
  const seedSeg = el('div', { class: 'seg' });
  for (const sd of D.seeds.ids) {
    const id = `${NAME}-seed-${sd}`;
    seedSeg.append(el('input', { type: 'radio', name: `${NAME}-seed`, id, value: sd, checked: sd === st.seed }), el('label', { for: id }, sd));
  }
  const vals = { g0: [el('span', {}), el('span', {})], mgc: [el('span', {}), el('span', {})] };
  const tile = (title, sub, rows, foot) => el('div', { class: 'rd-tile' },
    el('p', { class: 'rd-tile__title' }, ...title),
    el('p', { class: 'rd-tile__sub' }, sub),
    el('dl', { class: 'rd-tile__rows' }, ...rows.map(([armKey, label, span]) => el('div', { class: 'rd-tile__row' },
      el('dt', {}, armChip(armKey, label)), el('dd', {}, span)))),
    el('p', { class: 'rd-tile__foot' }, foot));
  const row2 = el('div', { class: 'rd-who' },
    el('div', { class: 'rd-who__head' },
      el('h4', { class: 'rd-b__title', id: `${NAME}-who-title` }, 'When two variants disagree, which one is right?'),
      el('fieldset', { class: 'ctl rd-seed' }, el('legend', { class: 'ctl__label' }, 'Seed'), seedSeg)),
    el('div', { class: 'rd-tiles' },
      tile([armChip('free'), ' vs. ', armChip('cum')], 'probe + flip', [
        ['free', 'history-free only correct', vals.g0[0]], ['cum', 'cumulative only correct', vals.g0[1]]],
      'nearly balanced in correctness'),
      tile([armChip('mgc'), ' vs. ', armChip('cum')], 'FF’s own sum (Stage 1)', [
        ['cum', 'cumulative only correct', vals.mgc[0]], ['mgc', 'MGC only correct', vals.mgc[1]]],
      `cumulative-only correct outnumber MGC-only correct in every seed: the Stage-1${NBSP}cost`)),
    el('p', { class: 'rd-b__small rd-who__all' }, `All three seeds (${D.seeds.ids.join('/')}): `,
      seedSplit('b7.g0_counts', D.seeds.g0_pair), ' history-free/cumulative (probe + flip); ',
      seedSplit('b7.mgc_counts', D.seeds.mgc_pair), ' cumulative/MGC (FF’s own sum).'));
  B.append(row2);
  B.append(el('figcaption', { class: 'fig__caption' },
    `Re-splitting the sum changes no prediction (Prop.${NBSP}3.2). Repairs are not exact re-splits: they change what the network learns, and that accuracy stays put on these readouts (MGC’s Stage-1 cost aside) is empirical.`,
    el('span', { class: 'src' }, `CIFAR-100 · re-split: cumulative model, seed 42 · retrained: 3 seeds · Sec.${NBSP}3.2, App.${NBSP}B.7`)));

  // tooltip (value first, then label) for the waffle; filled only while shown
  const tip = el('div', { class: 'tip rd-tip', id: `${NAME}-tip`, role: 'tooltip', hidden: true });
  const live = el('p', { class: 'vh', 'aria-live': 'polite', id: `${NAME}-live` });
  host.append(A, B, tip, live);

  /* ================================================================ drawing */
  let last = null;
  const widths = new Map();

  function drawCols(m) {
    const Wd = Math.max(240, boxWidth(colsBox) || 520);
    widths.set(colsBox, Wd);
    clear(colsSvg);
    const narrow = Wd < 460;
    const gut = narrow ? 70 : 84;                 // grid row labels
    const padR = 6;
    const top = 30;                               // headroom for the sum labels
    const plotH = narrow ? 176 : 206;
    const yb = top + plotH;                       // baseline
    const yMax = hund(T.y_max);
    const y = (v) => yb - (v / yMax) * plotH;
    const band = (Wd - gut - padR) / LABELS.length;
    const cw = Math.round(Math.min(24, band * 0.34));          // chart grammar: bars at most 24 px thick
    const cx = (i) => gut + band * (i + 0.5);
    const labY = yb + 21;
    const row0 = labY + 27;
    const rowH = 21;
    const Ht = row0 + (NB - 1) * rowH + 10;
    colsSvg.setAttribute('viewBox', `0 0 ${Wd} ${Ht}`);
    colsSvg.setAttribute('width', Wd);
    colsSvg.setAttribute('height', Ht);

    const pi = LABELS.indexOf(m.pred);
    sv('rect', { class: 'rd-hl', x: r1(cx(pi) - cw / 2 - 12), y: 4, width: cw + 24, height: r1(labY + 8 - 4), rx: 8 }, colsSvg);
    sv('line', { class: 'rd-axis', x1: gut - 6, x2: Wd - padR, y1: yb + 0.5, y2: yb + 0.5 }, colsSvg);
    sv('line', { class: 'rd-gridrule', x1: 0, x2: Wd - padR, y1: labY + 9.5, y2: labY + 9.5 }, colsSvg);

    // the reference line: the true label's sum, which another label must pass to change the prediction
    // (its label sits in the left gutter, which is free above the grid)
    const yRef = r1(y(m.sum[TRUE])) + 0.5;
    sv('line', { class: 'rd-ref', x1: gut - 6, x2: Wd - padR, y1: yRef, y2: yRef }, colsSvg);
    stext(colsSvg, gut - 10, yRef + 4, `${TRUE}’s sum`, 'rd-t rd-t--note', 'end');

    LABELS.forEach((lab, i) => {
      const vals = m.g[lab];
      let topIdx = -1;
      vals.forEach((v, d) => { if (v > 0) topIdx = d; });
      let acc = 0;
      const x = cx(i) - cw / 2;
      vals.forEach((v, d) => {
        if (v <= 0) return;
        const yTop = y(acc + v) + (d < topIdx ? 1 : 0);
        const yBot = y(acc) - (acc > 0 ? 1 : 0);
        const h = yBot - yTop;
        if (h > 0.2) {
          sv('path', { class: `rd-seg rd-seg--b${d}`, d: barPath(x, yTop, cw, h, d === topIdx ? 4 : 0, 'top') }, colsSvg);
          if (d === B3 && h > 2.5) sv('path', { class: 'rd-ring', d: barPath(x + 1, yTop + 1, cw - 2, h - 1, d === topIdx ? 3 : 0, 'top') }, colsSvg);
        }
        acc += v;
      });
      const sumT = stext(colsSvg, cx(i), y(acc) - 9, fmt(acc), `rd-t rd-t--sum${lab === m.pred ? ' rd-t--pred' : ''}`, 'middle');
      if (lab === RIVAL && m.moved > 0) {           // the re-split keeps this sum: say so where the eye is
        const k = sv('tspan', { class: 'rd-t--kept', dx: 4 }, sumT);
        k.textContent = 'kept';
      }
      stext(colsSvg, cx(i), labY, lab === TRUE ? `${lab} (true)` : lab, `rd-t rd-t--lab${lab === m.pred ? ' rd-t--pred' : ''}`, 'middle');
    });

    // where the moved or added goodness sits (the rival's column)
    const ri = LABELS.indexOf(RIVAL);
    const xr = cx(ri) + cw / 2;
    const sR = m.sum[RIVAL];
    const nextLeft = ri + 1 < LABELS.length ? cx(ri + 1) - cw / 2 - 6 : Wd;   // the next column's left edge
    const nextTop = ri + 1 < LABELS.length ? y(m.sum[LABELS[ri + 1]]) - 24 : Wd;  // above its sum label
    if (m.moved > 0) {
      const g0 = m.g[RIVAL][T.from];
      const yA = g0 > 0 ? y(g0 / 2) : y(0) - 2;
      const yB = y(sR - BASE[RIVAL][T.to] - m.moved / 2);
      const bulge = narrow ? 15 : 22;
      sv('path', { class: 'rd-arc', d: `M${r1(xr + 3)},${r1(yA)}C${r1(xr + bulge)},${r1(yA)} ${r1(xr + bulge)},${r1(yB)} ${r1(xr + 8)},${r1(yB)}` }, colsSvg);
      sv('path', { class: 'rd-arrow', d: `M${r1(xr + 2)},${r1(yB)}l7,-4.2v8.4z` }, colsSvg);
      const lbl = `${fmt(m.moved)} moved`;
      const lx = xr + bulge + 4;
      const ly = (yA + yB) / 2 + 4;
      const lw = textW(lbl, 12);
      if (lx + lw < Wd - padR && (lx + lw < nextLeft || ly < nextTop)) stext(colsSvg, lx, ly, lbl, 'rd-t rd-t--note rd-t--move', 'start');
    } else if (m.added > 0) {
      const yT = y(sR) + 1;
      const yBt = y(sR - m.added) - 1;
      const bx = xr + 5;
      sv('path', { class: 'rd-brace', d: `M${r1(bx)},${r1(yT)}h5V${r1(yBt)}h-5` }, colsSvg);
      const lbl = `+${fmt(m.added)} added`;
      const ly = Math.min((yT + yBt) / 2 + 4, yBt - 2);
      stext(colsSvg, bx + 9, Math.max(ly, yT + 10), lbl, 'rd-t rd-t--note rd-t--move', 'start');
    }

    // the goodness grid: rows = blocks (block 3 on top, as in the stack), columns = labels
    for (let r = 0; r < NB; r++) {
      const d = B3 - r;
      const yy = row0 + r * rowH;
      sv('rect', { class: `rd-sw rd-seg--b${d}${d === B3 ? ' rd-sw--b3' : ''}`, x: 1, y: yy - 10, width: 11, height: 11, rx: 2 }, colsSvg);
      stext(colsSvg, 18, yy, `block ${d}`, `rd-t${d === B3 ? ' rd-t--b3' : ''}`, 'start');
      LABELS.forEach((lab, i) => {
        const v = m.g[lab][d];
        const v0 = BASE[lab][d];
        const hot = v !== v0;
        const t = stext(colsSvg, cx(i), yy, fmt(v), `rd-t rd-t--cell${hot ? ' rd-t--hot' : ''}`, 'middle');
        if (hot) {
          const a = sv('tspan', { class: 'rd-t--dir', dx: 3 }, t);
          a.textContent = v > v0 ? '↑' : '↓';
        }
      });
    }

    colsDesc.textContent = `Toy goodness per block for each label, block 0 to ${B3}: `
      + LABELS.map((l) => `${l} ${m.g[l].map(fmt).join(', ')}, sum ${fmt(m.sum[l])}`).join('; ')
      + `. FF predicts ${m.pred}, the label with the largest sum.`;
  }

  function drawMargins(m) {
    const Wd = Math.max(240, boxWidth(marBox) || 480);
    widths.set(marBox, Wd);
    clear(marSvg);
    const narrow = Wd < 460;
    const gut = narrow ? 74 : 92;
    const roomL = narrow ? 44 : 52;
    const roomR = narrow ? 44 : 52;
    const [lo, hi] = T.margin_domain.map(hund);
    const x0 = gut + roomL;
    const x1 = Wd - roomR;
    const x = (v) => x0 + ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * (x1 - x0);
    const top = 40;
    const pitch = narrow ? 31 : 36;
    const bh = narrow ? 16 : 18;
    const sumY = top + NB * pitch + 16;
    const Ht = sumY + bh + 40;
    marSvg.setAttribute('viewBox', `0 0 ${Wd} ${Ht}`);
    marSvg.setAttribute('width', Wd);
    marSvg.setAttribute('height', Ht);
    const xz = r1(x(0)) + 0.5;
    stext(marSvg, xz - 8, 16, `← ${RIVAL} ahead`, 'rd-t rd-t--sm', 'end');
    stext(marSvg, xz + 8, 16, `${TRUE} ahead →`, 'rd-t rd-t--sm', 'start');
    sv('line', { class: 'rd-gridrule', x1: 0, x2: Wd, y1: sumY - 9.5, y2: sumY - 9.5 }, marSvg);
    sv('line', { class: 'rd-axis', x1: xz, x2: xz, y1: top - 12, y2: sumY + bh + 8 }, marSvg);
    stext(marSvg, xz, sumY + bh + 24, '0', 'rd-t rd-t--sm', 'middle');

    const row = (yy, v, label, cls, isB3, sub) => {
      stext(marSvg, 0, yy + bh / 2 + 4.5, label, `rd-t${isB3 || sub ? ' rd-t--b3' : ''}`, 'start');
      if (sub) stext(marSvg, 0, yy + bh / 2 + 21, sub, 'rd-t rd-t--sm', 'start');
      const clipped = v < lo || v > hi;            // off the axis: the bar ends in a break mark, the label keeps the value
      if (v !== 0) {
        const xa = v > 0 ? x(0) + 1 : x(v);
        const w = Math.abs(x(v) - x(0)) - 1;
        if (w > 0.2) {
          sv('path', { class: `rd-mbar ${cls}`, d: barPath(xa, yy, w, bh, clipped ? 0 : 4, v > 0 ? 'right' : 'left') }, marSvg);
          if (isB3 && w > 2.5) sv('path', { class: 'rd-ring', d: barPath(xa + (v > 0 ? 0 : 1), yy + 1, w - 1, bh - 2, clipped ? 0 : 3, v > 0 ? 'right' : 'left') }, marSvg);
          if (clipped) {
            const bx = v < 0 ? xa + 7 : xa + w - 11;
            sv('path', { class: 'rd-break', d: `M${r1(bx)},${r1(yy + bh + 3)}l4,${-(bh + 6)}M${r1(bx + 5)},${r1(yy + bh + 3)}l4,${-(bh + 6)}` }, marSvg);
          }
        }
      }
      const lx = v > 0 ? x(v) + 6 : v < 0 ? x(v) - 6 : x(0) + 6;
      stext(marSvg, lx, yy + bh / 2 + 4.5, sfmt(v), `rd-t rd-t--val${sub ? ' rd-t--pred' : ''}`, v < 0 ? 'end' : 'start');
    };
    for (let r = 0; r < NB; r++) {
      const d = B3 - r;
      row(top + r * pitch, m.margin[d], `block ${d}`, `rd-seg--b${d}`, d === B3, null);
    }
    row(sumY, m.msum, 'sum', 'rd-mbar--sum', false, 'FF reads this');

    marDesc.textContent = `Toy margins, ${TRUE} minus ${RIVAL}, per block from block ${B3} down to block 0: `
      + Array.from({ length: NB }, (_, r) => sfmt(m.margin[B3 - r])).join(', ')
      + `; summed over blocks ${sfmt(m.msum)}, which decides the prediction (${m.pred}).`;
  }

  function render() {
    const m = model();
    last = m;
    // outputs and the WebKit progress fill
    for (const R of [RQ, RA]) {
      const v = hund(R.input.value);
      R.out.textContent = `${fmt(v)} (toy)`;
      R.input.style.setProperty('--pct', `${(100 * (Number(R.input.value) - R.cfg.min)) / (R.cfg.max - R.cfg.min)}%`);
    }
    RQ.field.hidden = st.mode !== 'resplit';
    RA.field.hidden = st.mode !== 'sum';
    // prediction badge
    const changed = m.changed > 0;
    clear(badge);
    badge.append('prediction: ', el('b', {}, `${m.pred} ${m.pred === TRUE ? '✓' : '✗'}`), changed ? ' (changed)' : '');
    badge.classList.toggle('is-changed', changed);
    drawCols(m);
    drawMargins(m);
    marNote.textContent = `FF reads only the sum row: it is ${m.msum > 0 ? 'above' : 'below'} 0, so FF predicts ${m.pred}.`;
    // readouts
    tChanged.textContent = String(m.changed);
    if (st.mode === 'resplit' || m.added === 0) tSums.textContent = 'kept';
    else tSums.textContent = `${RIVAL}’s ${fmt(m.sum[RIVAL] - m.added)} → ${fmt(m.sum[RIVAL])}`;
    const m3 = m.margin[B3];
    tB3.textContent = m3 > 0 ? `scores ${TRUE} above ${RIVAL}` : m3 === 0 ? `scores ${TRUE} and ${RIVAL} alike` : `now scores ${RIVAL} above ${TRUE}`;
    tB3.parentElement.classList.toggle('is-flipped', m3 <= 0);
    // spoken value of the active slider (toy values; the controls sit inside [data-illus])
    if (st.mode === 'resplit') {
      RQ.input.setAttribute('aria-valuetext', `${fmt(m.moved)} moved (toy); every sum kept; prediction ${m.pred}; block ${B3} ${tB3.textContent}`);
    } else {
      RA.input.setAttribute('aria-valuetext', `${fmt(m.added)} added (toy); ${RIVAL}’s sum ${fmt(m.sum[RIVAL])}; prediction ${m.pred}${m.changed ? ', changed' : ''}`);
    }
  }

  function reading() {
    const m = last || model();
    const span = el('span', { 'data-illus': '' });
    if (st.mode === 'resplit') {
      span.textContent = m.moved > 0
        ? `Toy: ${RIVAL}’s block-${T.to} goodness up by ${fmt(m.moved)} and its block-${T.from} goodness down by ${fmt(m.moved)}; every sum unchanged; prediction ${m.pred}, unchanged; block-${B3} margin now ${sfmt(m.margin[B3])}.`
        : `Toy: nothing moved; every sum unchanged; prediction ${m.pred}; block-${B3} margin ${sfmt(m.margin[B3])}.`;
    } else {
      const rel = m.sum[RIVAL] > m.sum[TRUE] ? 'above' : 'below';
      span.textContent = m.added > 0
        ? `Toy: ${RIVAL}’s block-${T.to} goodness up by ${fmt(m.added)}; ${RIVAL}’s sum now ${fmt(m.sum[RIVAL])}, ${rel} ${TRUE}’s ${fmt(m.sum[TRUE])}; prediction ${m.pred}, ${m.changed ? 'changed' : 'unchanged'}.`
        : `Toy: nothing added; every sum unchanged; prediction ${m.pred}.`;
    }
    return span;
  }
  function announce(node) { clear(live); live.append(node); }

  function renderSeed() {
    const i = D.seeds.ids.indexOf(st.seed);
    setFact(vals.g0[0], D.seeds.g0[i][0]);
    setFact(vals.g0[1], D.seeds.g0[i][1]);
    setFact(vals.mgc[0], D.seeds.mgc[i][0]);
    setFact(vals.mgc[1], D.seeds.mgc[i][1]);
    for (const n of root.querySelectorAll('.rd-seedval')) n.classList.toggle('is-on', n.getAttribute('data-seed') === st.seed);
  }
  const seedReading = () => `Seed ${st.seed}: history-free only correct ${vals.g0[0].textContent}, cumulative only correct ${vals.g0[1].textContent} (probe + flip); cumulative only correct ${vals.mgc[0].textContent}, MGC only correct ${vals.mgc[1].textContent} (FF’s own sum).`;

  /* ================================================================ events */
  modeSeg.addEventListener('change', (e) => {
    if (!e.target.matches('input[type=radio]')) return;
    st.mode = e.target.value;
    render();
    announce(reading());
  });
  for (const [R, k] of [[RQ, 'q'], [RA, 'a']]) {
    R.input.addEventListener('input', () => { st[k] = hund(R.input.value); render(); });
    R.input.addEventListener('change', () => { st[k] = hund(R.input.value); render(); announce(reading()); });
  }
  seedSeg.addEventListener('change', (e) => {
    if (!e.target.matches('input[type=radio]')) return;
    st.seed = e.target.value;
    renderSeed();
    announce(document.createTextNode(seedReading()));
  });

  // tooltip on the waffle: hover and focus; Esc hides; it never gates (its quote is also in the tiles)
  let tipOn = false;
  // above the waffle, so it never covers the value it annotates; below when the sticky top bar is in the way
  function placeTip() {
    const r = waffle.getBoundingClientRect();
    const t = tip.getBoundingClientRect();
    const vw = document.documentElement.clientWidth;
    const x = Math.max(8, Math.min(r.left, vw - t.width - 8));
    let yy = r.top - t.height - 10;
    if (yy < 68) yy = r.bottom + 10;
    tip.style.setProperty('--tip-x', `${Math.round(x)}px`);
    tip.style.setProperty('--tip-y', `${Math.round(yy)}px`);
  }
  function showTip() {
    if (!tipOn) {
      clear(tip);
      tip.append(el('span', { class: 'tip__v' }, churn),
        el('span', { class: 'tip__l' }, '“For the trio pairs under S2 TTA, the disagreements are nearly balanced in correctness.” App. B.7; S2 TTA: probe + flip.'));
      tip.hidden = false;
      tipOn = true;
    }
    placeTip();
  }
  function hideTip() {
    if (!tipOn) return;
    tip.hidden = true;
    clear(tip);
    tipOn = false;
  }
  waffle.addEventListener('pointerenter', showTip);
  waffle.addEventListener('pointerleave', () => { if (document.activeElement !== waffle) hideTip(); });
  waffle.addEventListener('focus', showTip);
  waffle.addEventListener('blur', hideTip);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && tipOn) hideTip(); });
  for (const ev of ['scroll', 'resize']) window.addEventListener(ev, () => { if (tipOn) placeTip(); }, { passive: true });

  render();
  renderSeed();

  // after the root is live: draw at the real widths, then redraw whenever a chart box changes width
  return function afterLive() {
    if (last) { drawCols(last); drawMargins(last); }
    if ('ResizeObserver' in window) {
      let raf = 0;
      const ro = new ResizeObserver(() => {
        if (raf) return;
        raf = requestAnimationFrame(() => {
          raf = 0;
          if (!last) return;
          if (boxWidth(colsBox) !== widths.get(colsBox)) drawCols(last);
          if (boxWidth(marBox) !== widths.get(marBox)) drawMargins(last);
        });
      });
      ro.observe(colsBox);
      ro.observe(marBox);
    }
  };
}

function cap(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

/* ------------------------------------------------------------------ boot (after every helper is initialized) */
const root = document.getElementById(`viz-${NAME}`);
if (root) {
  try {
    const afterLive = mount(root);
    root.dataset.state = 'live';
    afterLive();
  } catch (err) {
    console.error(`[${NAME}]`, err);
    root.dataset.state = 'broken';
  }
}
