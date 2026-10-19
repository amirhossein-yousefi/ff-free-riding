// Shell script (ES module, no imports): global chrome only. Components live in comp_<name>.js and never
// depend on this file (they may listen to the document event "ff:theme"). Nothing here touches storage,
// cookies or the network; the theme choice lives in memory and resets on reload (back to the system setting).
const root = document.documentElement;
root.classList.remove('no-js');
root.classList.add('js');

const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const scrollBehavior = () => (reduceMotion.matches ? 'auto' : 'smooth');

/* ---- theme: a two-state switch (light <-> dark), seeded from the theme in effect, so every click changes what
   the reader sees. Until the first click the page follows the system setting (mode "auto"); the choice lives in
   memory only. The button names the theme a click gives; dispatches "ff:theme" {mode, dark} ---- */
const darkQuery = window.matchMedia('(prefers-color-scheme: dark)');
const themeBtn = document.getElementById('theme-toggle');
let themeMode = 'auto';
function isDark() { return themeMode === 'dark' || (themeMode === 'auto' && darkQuery.matches); }
function announceTheme() {
  document.dispatchEvent(new CustomEvent('ff:theme', { detail: { mode: themeMode, dark: isDark() } }));
}
function labelTheme() {
  if (!themeBtn) return;
  const next = isDark() ? 'light' : 'dark';
  const label = `Switch to the ${next} theme`;
  themeBtn.setAttribute('aria-label', label);
  themeBtn.title = label;
  const use = themeBtn.querySelector('use');
  if (use) use.setAttribute('href', `#i-theme-${next}`);
}
if (themeBtn) {
  themeBtn.addEventListener('click', () => {
    themeMode = isDark() ? 'light' : 'dark';
    root.setAttribute('data-theme', themeMode);
    labelTheme();
    announceTheme();
  });
  labelTheme();
}
darkQuery.addEventListener('change', () => { if (themeMode === 'auto') { labelTheme(); announceTheme(); } });

/* ---- contents menu: close on link, Esc, outside click ---- */
const menu = document.querySelector('.menu');
if (menu) {
  menu.addEventListener('click', (e) => { if (e.target.closest('a')) menu.open = false; });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && menu.open) { menu.open = false; menu.querySelector('summary').focus(); }
  });
  document.addEventListener('click', (e) => { if (menu.open && !menu.contains(e.target)) menu.open = false; });
}

/* ---- scroll-spy (aria-current on the top-bar links) and the reading-progress line ---- */
const topbar = document.getElementById('topbar');
const navLinks = Array.from(document.querySelectorAll('.toc a[href^="#"]'));
/* every section counts: in a section without its own top-bar link (Read this first, Scope, Resources ...) no link is
   marked current, instead of the previous section's link staying highlighted */
const targets = Array.from(document.querySelectorAll('section[id]'));
let ticking = false;
function onScroll() {
  ticking = false;
  const max = Math.max(1, root.scrollHeight - window.innerHeight);
  if (topbar) topbar.style.setProperty('--progress', String(Math.min(1, Math.max(0, window.scrollY / max))));
  const line = window.innerHeight * 0.4;
  let current = null;
  for (const t of targets) { if (t.getBoundingClientRect().top <= line) current = t; }
  for (const a of navLinks) {
    const on = current && a.getAttribute('href') === '#' + current.id;
    if (on) a.setAttribute('aria-current', 'true'); else a.removeAttribute('aria-current');
  }
}
window.addEventListener('scroll', () => { if (!ticking) { ticking = true; requestAnimationFrame(onScroll); } }, { passive: true });
window.addEventListener('resize', () => { if (!ticking) { ticking = true; requestAnimationFrame(onScroll); } });
onScroll();

/* ---- phone folds: <details class="fold"> read as plain blocks on wide screens (open; the summary is hidden by CSS)
   and start collapsed on phones (< 640 px), where they shorten the page; crossing the breakpoint resets them.
   Without JavaScript they stay open. Runs before the deep-link opener, which reopens a fold that holds the target. ---- */
const phoneQuery = window.matchMedia('(max-width: 639px)');
const folds = Array.from(document.querySelectorAll('details.fold'));
function setFolds() { for (const f of folds) f.open = !phoneQuery.matches; }
setFolds();
phoneQuery.addEventListener('change', setFolds);

/* ---- deep links into <details> (FAQ items, "Theorem 3.1 as stated", glossary): open on load and on hashchange ---- */
function openHashTarget() {
  let id = '';
  try { id = decodeURIComponent(location.hash.slice(1)); } catch (_) { return; }
  if (!id) return;
  const el = document.getElementById(id);
  if (!el) return;
  let opened = false;
  for (let d = el.tagName === 'DETAILS' ? el : el.closest('details'); d; d = d.parentElement && d.parentElement.closest('details')) {
    if (!d.open) { d.open = true; opened = true; }
  }
  if (opened) requestAnimationFrame(() => el.scrollIntoView({ block: 'start', behavior: scrollBehavior() }));
}
window.addEventListener('hashchange', openHashTarget);
openHashTarget();

/* ---- keyboard focus lands fully in view: browsers scroll a focused element only while it is entirely off screen,
   so a tall card, chart or thumbnail can stay half below the bottom edge (or half under the sticky top bar). On
   keyboard focus (:focus-visible) an element that fits between the page's scroll padding at the top (site.css: the
   top bar plus 16 px) and 12 px above the bottom edge is scrolled fully into view, at once. Pointer focus (a click or
   a tap) never scrolls. SVG chart marks are revealed by their components with the same limits. ---- */
document.addEventListener('focusin', (e) => {
  const el = e.target;
  if (!(el instanceof HTMLElement) || el === document.body || el.closest('#topbar')) return;
  let keyboard = true;
  try { keyboard = el.matches(':focus-visible'); } catch (_) { keyboard = true; }
  if (!keyboard) return;
  const r = el.getBoundingClientRect();
  if (!r.width && !r.height) return;
  const top = parseFloat(getComputedStyle(root).scrollPaddingTop) || 72;
  const bottom = window.innerHeight - 12;
  if ((r.top >= top && r.bottom <= bottom) || r.height > bottom - top) return;   // in view, or taller than the view
  const dy = r.bottom > bottom ? r.bottom - bottom : r.top - top;
  try { window.scrollBy({ top: dy, left: 0, behavior: 'instant' }); } catch (_) { window.scrollBy(0, dy); }
});

/* ---- "Expand all" for the FAQ (progressive enhancement; the button is hidden without JS) ---- */
for (const btn of document.querySelectorAll('[data-expand-all]')) {
  const scope = document.getElementById(btn.getAttribute('data-expand-all'));
  if (!scope) continue;
  const items = Array.from(scope.querySelectorAll('details'));
  const sync = () => {
    const all = items.every((d) => d.open);
    btn.textContent = all ? 'Collapse all' : 'Expand all';
    btn.setAttribute('aria-expanded', String(all));
  };
  btn.hidden = false;
  btn.addEventListener('click', () => {
    const open = !items.every((d) => d.open);
    items.forEach((d) => { d.open = open; });
    sync();
  });
  items.forEach((d) => d.addEventListener('toggle', sync));
  sync();
}

/* ---- copy buttons: [data-copy="<id of the <pre>>"], announced in the polite live region (aria-describedby) ---- */
for (const btn of document.querySelectorAll('[data-copy]')) {
  const src = document.getElementById(btn.getAttribute('data-copy'));
  const live = document.getElementById(btn.getAttribute('aria-describedby') || '') || null;
  if (!src) continue;
  btn.hidden = false;
  let timer = 0;
  btn.addEventListener('click', async () => {
    const text = src.textContent.replace(/\s+$/, '') + '\n';
    let ok = false;
    try { await navigator.clipboard.writeText(text); ok = true; } catch (_) { ok = false; }
    if (!ok) {
      const range = document.createRange(); range.selectNodeContents(src);
      const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(range);
    }
    if (live) {
      live.textContent = ok ? 'Copied.' : 'Selected: press Ctrl+C or Cmd+C to copy.';
      clearTimeout(timer);
      timer = setTimeout(() => { live.textContent = ''; }, 2500);
    }
    btn.dataset.copied = ok ? 'true' : 'false';
  });
}

/* ---- wide tables and the BibTeX: focusable scroll regions only while they overflow (keyboard users can scroll them) ---- */
const wraps = Array.from(document.querySelectorAll('.tbl-wrap, .code--bib pre'));
// the caption's own words only: its source line (.src) and badges are separate elements whose text would
// otherwise run into it ("… on CIFAR-100" + "4 blocks …" read as "CIFAR-1004 blocks")
function captionName(cap) {
  if (!cap) return 'Table';
  const parts = [];
  for (const n of cap.childNodes) {
    if (n.nodeType === 1 && n.matches('.src, .badge')) continue;
    parts.push(n.textContent);
  }
  return parts.join('').replace(/\s+/g, ' ').trim() || 'Table';
}
function markOverflow(w) {
  if (w.scrollWidth > w.clientWidth + 1) {
    w.setAttribute('tabindex', '0');
    w.setAttribute('role', 'region');
    if (!w.hasAttribute('aria-label')) {
      const name = w.matches('pre') ? 'BibTeX' : captionName(w.querySelector('caption'));
      w.setAttribute('aria-label', `${name} (scrolls sideways)`);
    }
  } else if (w.getAttribute('tabindex') === '0') {
    w.removeAttribute('tabindex'); w.removeAttribute('role'); w.removeAttribute('aria-label');
  }
}
if ('ResizeObserver' in window) {
  const ro = new ResizeObserver((entries) => entries.forEach((e) => markOverflow(e.target)));
  wraps.forEach((w) => ro.observe(w));
} else {
  wraps.forEach(markOverflow);
}
for (const d of document.querySelectorAll('details')) d.addEventListener('toggle', () => wraps.forEach(markOverflow));

/* ---- the poster session in the reader's own time zone (progressive enhancement): the static text gives the Paris time
   and the NeurIPS schedule's listing; this fills the hidden "In your time zone" row of #visit from the UTC instants in
   its data-session-start / data-session-end attributes, in the page's language and the browser's time zone. A reader
   whose clock reads like Paris during the session (Paris, Amsterdam, Berlin ...) sees "Same as Paris time"; a session
   that crosses midnight where the reader is shows two dated times and the zone once. Nothing is stored or sent; on any
   failure the row stays hidden. ---- */
for (const box of document.querySelectorAll('[data-session-start][data-session-end]')) {
  const row = box.querySelector('[data-local-row]');
  if (!row) continue;
  try {
    const start = new Date(box.getAttribute('data-session-start'));
    const end = new Date(box.getAttribute('data-session-end'));
    if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime()) || end <= start) continue;
    const lang = root.lang || 'en-US';
    const zone = (new Intl.DateTimeFormat().resolvedOptions().timeZone || '').replace(/_/g, ' ');
    let sameAsParis = false;
    try {   // the same wall-clock date and time as Paris at both instants (a browser without time-zone data: false)
      const wall = (timeZone) => new Intl.DateTimeFormat('en-US', { timeZone, year: 'numeric', month: 'numeric', day: 'numeric', hour: 'numeric', minute: 'numeric' });
      const here = wall(undefined);
      const paris = wall('Europe/Paris');
      sameAsParis = [start, end].every((t) => here.format(t) === paris.format(t));
    } catch (_) { sameAsParis = false; }
    const hm = { hour: 'numeric', minute: '2-digit' };
    const day = new Intl.DateTimeFormat(lang, { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' });
    let lines;
    if (sameAsParis) {
      lines = ['Same as Paris time', '', zone];
    } else if (day.format(start) === day.format(end)) {
      const hours = new Intl.DateTimeFormat(lang, { ...hm, timeZoneName: 'short' });
      lines = [day.format(start),
        typeof hours.formatRange === 'function' ? hours.formatRange(start, end) : `${hours.format(start)} – ${hours.format(end)}`, zone];
    } else {   // the two hours cross midnight where the reader is: two dated times (no year), then the zone once
      const at = new Intl.DateTimeFormat(lang, { weekday: 'short', month: 'short', day: 'numeric', ...hm });
      const named = new Intl.DateTimeFormat(lang, { ...hm, timeZoneName: 'short' });
      const abbr = (t) => (named.formatToParts(t).find((p) => p.type === 'timeZoneName') || {}).value || '';
      const names = [...new Set([abbr(start), abbr(end)])].filter(Boolean).join(' / ');
      lines = [`${at.format(start)} –`, at.format(end), [names, zone].filter(Boolean).join(' · ')];
    }
    const put = (sel, text) => { const el = row.querySelector(sel); if (el) el.textContent = text; };
    put('[data-local-date]', lines[0]);
    put('[data-local-hours]', lines[1]);
    put('[data-local-zone]', lines[2]);
    row.hidden = false;
  } catch (_) { /* the static Paris time and the NeurIPS schedule's listing stay */ }
}

window.FF_SHELL_READY = true;
