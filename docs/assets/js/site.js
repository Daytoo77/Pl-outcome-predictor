/* site.js: theme toggle, mobile nav, entrance animation, number formatting.
   No dependencies. Loaded with `defer`; every page is usable without it. */
(function () {
  'use strict';
  var root = document.documentElement;
  var KEY = 'pl-theme';
  var ORDER = ['system', 'light', 'dark'];

  function stored() { try { return localStorage.getItem(KEY); } catch (e) { return null; } }
  function store(v) { try { if (v) localStorage.setItem(KEY, v); else localStorage.removeItem(KEY); } catch (e) { /* storage unavailable */ } }
  function apply(v) {
    if (v === 'light' || v === 'dark') root.setAttribute('data-theme', v);
    else root.removeAttribute('data-theme');
  }

  function initTheme() {
    var btn = document.querySelector('[data-theme-toggle]');
    if (!btn) return;
    var state = stored();
    if (ORDER.indexOf(state) === -1) state = 'system';
    function label() {
      var next = ORDER[(ORDER.indexOf(state) + 1) % ORDER.length];
      return 'Theme: ' + state + '. Switch to ' + next + '.';
    }
    function render() {
      btn.setAttribute('data-state', state);
      btn.setAttribute('aria-label', label());
      btn.title = label();
    }
    btn.addEventListener('click', function () {
      state = ORDER[(ORDER.indexOf(state) + 1) % ORDER.length];
      store(state === 'system' ? null : state);
      apply(state);
      render();
    });
    render();
  }

  function initNav() {
    var nav = document.querySelector('.site-nav');
    var btn = nav && nav.querySelector('[data-nav-toggle]');
    if (!btn) return;
    function setOpen(open) {
      if (open) nav.setAttribute('data-open', ''); else nav.removeAttribute('data-open');
      btn.setAttribute('aria-expanded', open ? 'true' : 'false');
    }
    btn.addEventListener('click', function () { setOpen(!nav.hasAttribute('data-open')); });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && nav.hasAttribute('data-open')) { setOpen(false); btn.focus(); }
    });
    document.addEventListener('click', function (e) {
      if (nav.hasAttribute('data-open') && !nav.contains(e.target)) setOpen(false);
    });
  }

  function initReveal() {
    var els = document.querySelectorAll('.reveal');
    if (!els.length) return;
    var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce || !('IntersectionObserver' in window)) {
      Array.prototype.forEach.call(els, function (el) { el.classList.add('in'); });
      return;
    }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { en.target.classList.add('in'); io.unobserve(en.target); }
      });
    }, { threshold: 0.08, rootMargin: '0px 0px -4% 0px' });
    Array.prototype.forEach.call(els, function (el) { io.observe(el); });
  }

  function initTocSpy() {
    var links = document.querySelectorAll('.toc-list a[href^="#"]');
    if (!links.length || !('IntersectionObserver' in window)) return;
    var byId = {};
    Array.prototype.forEach.call(links, function (a) { byId[a.getAttribute('href').slice(1)] = a; });
    var targets = Object.keys(byId).map(function (id) { return document.getElementById(id); }).filter(Boolean);
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (!en.isIntersecting) return;
        Array.prototype.forEach.call(links, function (a) { a.removeAttribute('aria-current'); });
        byId[en.target.id].setAttribute('aria-current', 'true');
      });
    }, { rootMargin: '-20% 0px -70% 0px', threshold: 0 });
    targets.forEach(function (t) { io.observe(t); });
  }

  /* shared formatting: probabilities and percentages always to one decimal place */
  window.PL = window.PL || {};
  window.PL.pct = function (x) { return (Number(x) * 100).toFixed(1) + '%'; };
  window.PL.fix = function (x, d) { return Number(x).toFixed(d == null ? 3 : d); };
  window.PL.esc = function (s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  };
  /* UK kickoff time zone: British Summer Time runs from the last Sunday of March
     to the last Sunday of October. */
  window.PL.ukZone = function (isoDate) {
    var d = new Date(isoDate + 'T12:00:00Z');
    var y = d.getUTCFullYear();
    function lastSunday(month) { var x = new Date(Date.UTC(y, month + 1, 0)); x.setUTCDate(x.getUTCDate() - x.getUTCDay()); return x; }
    var start = lastSunday(2), end = lastSunday(9);
    return (d >= start && d < end) ? 'BST' : 'GMT';
  };

  function init() { initTheme(); initNav(); initReveal(); initTocSpy(); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
