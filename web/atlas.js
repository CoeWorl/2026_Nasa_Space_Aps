/* GROUNDTRUTH atlas — shared data layer and drawing devices.

   One classic script, no build step, no dependencies. Every page does:

       ATLAS.boot("evidence", function (ctx) { ... });

   boot() loads the exported products, draws the masthead and the chapter
   nav, restores the shared selection from the URL, and then hands control
   to the page.

   SHARED SELECTION. Selected acquisition, track, coverage-screen state,
   tide-match tolerance and selected marsh polygon live in the query string,
   so a selection survives a page change, a reload, and a pasted link. Chapter
   links carry the current state with them.

   THE SCIENCE STAYS IN PYTHON. This file computes nothing inferential. Every
   slope, interval, p-value and error metric is read from models.json or
   robustness.json. The only arithmetic here is differences, medians and
   scales for drawing.
*/
"use strict";

var ATLAS = (function () {

  var DATA = new URL(new URLSearchParams(location.search).get("data") || "./data-app/", location.href);

  var ORBITS = {
    descending_069: { label: "Descending 069", short: "descending", color: "#367DA4", shape: "circle" },
    ascending_133:  { label: "Ascending 133",  short: "ascending",  color: "#C7774C", shape: "square" }
  };
  var INK = "#243846", MUTED = "#5f6e77", FAINT = "#93a0a6", RULE = "#d6d0c1",
      MARSH = "#788A70", PAPER = "#F3F0E8", SEL = "#111c24", WARN = "#9c5a2a";
  var THRESHOLDS = [0.05, 0.10, 0.20];

  var PAGES = [
    { id: "finding",    file: "finding.html",    no: "01", title: "The finding",       blurb: "What the data support, with every slope on one scale." },
    { id: "observe",    file: "observe.html",    no: "02", title: "One observation",   blurb: "A radar image, two gauges, and the instant the satellite sampled." },
    { id: "evidence",   file: "evidence.html",   no: "03", title: "Both directions",   blurb: "The two direction-specific regressions, on a scale you choose." },
    { id: "robustness", file: "robustness.html", no: "04", title: "Robustness",        blurb: "What happens to the slope when the assumptions are pushed." },
    { id: "holdout",    file: "holdout.html",    no: "05", title: "Held out",          blurb: "Does the correction work on dates the model never saw?" },
    { id: "marshes",    file: "marshes.html",    no: "06", title: "Marsh by marsh",    blurb: "Forty-five polygons, shrunken slopes, and a dependence-aware null." },
    { id: "matched",    file: "matched.html",    no: "07", title: "Tide-matched",      blurb: "Comparing passes that saw the same water level." },
    { id: "context",    file: "context.html",    no: "08", title: "Restoration",       blurb: "Context only: where a tide-aware comparison could later be tested." }
  ];

  // --- URL-backed state ------------------------------------------------------
  var state = {
    selected: null, orbit: "descending_069", includePartial: false,
    matchThresh: 0.10, region: null
  };
  var MODELS, SERIES, HYDRO, ENV, ACQ, BASE, PROJECTS, ROBUST, PTS = {}, BYID = {};
  var PAGE = null, RENDER = null;

  function readURL() {
    var q = new URLSearchParams(location.search);
    if (q.get("sel")) state.selected = q.get("sel");
    if (ORBITS[q.get("orbit")]) state.orbit = q.get("orbit");
    if (q.get("partial")) state.includePartial = q.get("partial") === "1";
    if (q.get("thr")) state.matchThresh = parseFloat(q.get("thr"));
    if (q.get("region")) state.region = q.get("region");
  }
  function sharedParams() {
    var q = new URLSearchParams();
    var raw = new URLSearchParams(location.search).get("data");
    if (raw) q.set("data", raw);
    if (state.selected) q.set("sel", state.selected);
    q.set("orbit", state.orbit);
    if (state.includePartial) q.set("partial", "1");
    if (state.matchThresh !== 0.10) q.set("thr", String(state.matchThresh));
    if (state.region) q.set("region", state.region);
    return q;
  }
  function pushURL() {
    history.replaceState(null, "", location.pathname + "?" + sharedParams().toString());
    var q = sharedParams().toString();
    document.querySelectorAll("nav.chapters a[data-file], .onward a[data-file]").forEach(function (a) {
      a.href = a.dataset.file + (a.dataset.file === "explore.html" ? "" : "?" + q);
    });
  }
  function hrefFor(file) { return file + "?" + sharedParams().toString(); }

  // --- small helpers ---------------------------------------------------------
  var NS = "http://www.w3.org/2000/svg";
  function el(tag, attrs, kids) {
    var n = document.createElementNS(NS, tag);
    attrs = attrs || {};
    for (var k in attrs) if (attrs[k] != null) n.setAttribute(k, attrs[k]);
    [].concat(kids || []).forEach(function (c) {
      n.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
    return n;
  }
  function h(tag, cls, html) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html != null) n.innerHTML = html;
    return n;
  }
  function scale(d0, d1, r0, r1) { var m = (r1 - r0) / ((d1 - d0) || 1); return function (v) { return r0 + (v - d0) * m; }; }
  function ticks(lo, hi, n) {
    n = n || 5;
    var s0 = (hi - lo || 1) / n, mag = Math.pow(10, Math.floor(Math.log10(s0))), nr = s0 / mag,
        st = (nr < 1.5 ? 1 : nr < 3 ? 2 : nr < 7 ? 5 : 10) * mag, t = [];
    for (var v = Math.ceil(lo / st) * st; v <= hi + 1e-9; v += st) t.push(+v.toFixed(10));
    return t;
  }
  function ext(a) { return [Math.min.apply(null, a), Math.max.apply(null, a)]; }
  function median(a) {
    if (!a.length) return NaN;
    var s = a.slice().sort(function (x, y) { return x - y; }), m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }
  function sgn(v, d) { d = d == null ? 2 : d; return v == null || !isFinite(v) ? "—" : (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d); }
  function fx(v, d) { d = d == null ? 2 : d; return v == null || !isFinite(v) ? "—" : (v < 0 ? "−" : "") + Math.abs(v).toFixed(d); }
  function pct(v, d) { return v == null || !isFinite(v) ? "—" : (v * 100).toFixed(d == null ? 0 : d) + "%"; }
  function pv(p) { return p == null ? "—" : p < 0.001 ? "<0.001" : p.toFixed(3); }
  function ci(a, d) { return "[" + fx(a[0], d) + ", " + fx(a[1], d) + "]"; }
  function days(a, b) { return Math.round(Math.abs(a - b) / 864e5); }
  function fmtDate(t) { return new Date(t).toLocaleDateString("en-US", { timeZone: "UTC", year: "numeric", month: "long", day: "numeric" }); }
  function monthTicks(t0, t1) {
    var out = [], d0 = new Date(t0);
    for (var d = new Date(Date.UTC(d0.getUTCFullYear(), d0.getUTCMonth() + 1, 1)); d.getTime() <= t1; d.setUTCMonth(d.getUTCMonth() + 2)) out.push(d.getTime());
    return out;
  }
  function monthFmt(t) {
    var d = new Date(t);
    return d.toLocaleString("en", { month: "short", timeZone: "UTC" }) + (d.getUTCMonth() <= 1 ? " ’" + String(d.getUTCFullYear()).slice(2) : "");
  }
  function axes(svg, x, y, P, W, H, xt, yt, xf, yf) {
    var ax = el("g", { class: "axis" });
    xt.forEach(function (t) {
      ax.appendChild(el("line", { x1: x(t), x2: x(t), y1: P.t, y2: H - P.b }));
      ax.appendChild(el("text", { x: x(t), y: H - P.b + 15, "text-anchor": "middle" }, xf(t)));
    });
    yt.forEach(function (t) {
      ax.appendChild(el("line", { x1: P.l, x2: W - P.r, y1: y(t), y2: y(t) }));
      ax.appendChild(el("text", { x: P.l - 6, y: y(t) + 3.5, "text-anchor": "end" }, yf(t)));
    });
    svg.appendChild(ax);
    return ax;
  }

  // --- the observation mark --------------------------------------------------
  function mark(p, x, y, opt) {
    opt = opt || {};
    var o = ORBITS[p.orbit], r = opt.r == null ? 4.2 : opt.r, rr = opt.sel ? r + 2.2 : r,
        fade = opt.fade ? 0.25 : 1, click = opt.click !== false;
    var a = {
      class: "m", fill: p.ok ? o.color : PAPER, "fill-opacity": fade,
      stroke: opt.sel ? SEL : o.color, "stroke-opacity": fade,
      "stroke-width": opt.sel ? 2.2 : (p.ok ? 0 : 1.6)
    };
    if (opt.fill) { a.fill = opt.fill; a.stroke = opt.stroke || o.color; }
    var shape = o.shape === "circle"
      ? el("circle", Object.assign({ cx: x, cy: y, r: rr }, a))
      : el("rect", Object.assign({ x: x - rr * 0.9, y: y - rr * 0.9, width: rr * 1.8, height: rr * 1.8 }, a));
    var g = el("g", {
      class: click ? "pt" : "", tabindex: click ? 0 : null, role: click ? "button" : null,
      "aria-label": o.short + " " + p.date + ", water level " + p.h.toFixed(2) + " metres" + (p.ok ? "" : ", failed the coverage screen")
    }, [shape]);
    g.appendChild(el("title", {}, opt.title ||
      (p.date + " · " + o.label + "\nh " + fx(p.h) + " m · marsh−upland " + fx(p.y) + " dB" + (p.ok ? "" : "\nfailed the coverage screen"))));
    if (click) {
      g.onclick = function () { select(p.id); };
      g.onkeydown = function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(p.id); } };
    }
    return g;
  }
  function im(orbit, hollow) {
    var o = ORBITS[orbit], f = hollow ? PAPER : o.color;
    var s = o.shape === "circle"
      ? '<circle cx="6" cy="6" r="4.6" fill="' + f + '" stroke="' + o.color + '" stroke-width="1.6"/>'
      : '<rect x="1.6" y="1.6" width="8.8" height="8.8" fill="' + f + '" stroke="' + o.color + '" stroke-width="1.6"/>';
    return '<svg class="im" viewBox="0 0 12 12" aria-hidden="true">' + s + "</svg>";
  }
  function def(term, explanation) {
    return '<span class="defn" title="' + explanation.replace(/"/g, "&quot;") + '">' + term + "</span>";
  }

  // --- model accessors -------------------------------------------------------
  function variantOf(orbit) {
    return (orbit === "ascending_133" && !state.includePartial && MODELS.tracks[orbit].series.coverage_screened)
      ? "coverage_screened" : "all_dates";
  }
  function model(orbit) { return MODELS.tracks[orbit].series[variantOf(orbit)]; }
  function robust(orbit, variant) { return ROBUST.tracks[orbit].variants[variant || variantOf(orbit)]; }
  function inFit(p) { return variantOf(p.orbit) === "all_dates" || p.ok; }
  function fitPoints(orbit) { return PTS[orbit].filter(inFit); }

  function select(id) {
    if (!BYID[id]) return;
    state.selected = id;
    state.orbit = BYID[id].orbit;
    redraw();
  }
  function set(key, value) { state[key] = value; redraw(); }
  function redraw() {
    var keep = window.scrollY;
    var host = document.getElementById("app");
    host.innerHTML = "";
    RENDER({ host: host });
    pushURL();
    window.scrollTo(0, keep);
  }

  // --- chrome ----------------------------------------------------------------
  function drawLocator(host) {
    if (!BASE || !BASE.locator) { host.remove(); document.querySelector(".mast").style.gridTemplateColumns = "1fr"; return; }
    var L = BASE.locator,
        all = L.boroughs.reduce(function (acc, b) { return acc.concat(b.rings.reduce(function (c, r) { return c.concat(r); }, [])); }, []).concat(L.window);
    var xs = ext(all.map(function (c) { return c[0]; })), ys = ext(all.map(function (c) { return c[1]; })),
        k = Math.cos(40.65 * Math.PI / 180), W = 132, s = (W - 4) / ((xs[1] - xs[0]) * k), H = (ys[1] - ys[0]) * s + 4;
    function X(c) { return 2 + (c[0] - xs[0]) * k * s; }
    function Y(c) { return 2 + (ys[1] - c[1]) * s; }
    function path(ring) { return ring.map(function (c, i) { return (i ? "L" : "M") + X(c).toFixed(1) + " " + Y(c).toFixed(1); }).join("") + "Z"; }
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H.toFixed(1) });
    L.boroughs.forEach(function (b) { b.rings.forEach(function (r) { svg.appendChild(el("path", { d: path(r), fill: INK })); }); });
    svg.appendChild(el("path", { d: path(L.window), fill: ORBITS.descending_069.color, "fill-opacity": 0.22, stroke: ORBITS.descending_069.color, "stroke-width": 1.4 }));
    svg.appendChild(el("title", {}, "Brooklyn and Queens, with the Jamaica Bay study window"));
    host.appendChild(svg);
  }

  function drawChrome(page) {
    var mast = document.querySelector(".mast");
    drawLocator(document.getElementById("locator"));
    var nav = h("nav", "chapters");
    PAGES.forEach(function (p) {
      var a = h("a", null, '<span class="n">' + p.no + "</span><span class=\"t\">" + p.title + "</span>");
      a.dataset.file = p.file;
      a.href = hrefFor(p.file);
      if (p.id === page) a.setAttribute("aria-current", "page");
      nav.appendChild(a);
    });
    var legacy = h("a", "legacy", '<span class="n">—</span><span class="t">Legacy one-pager</span>');
    legacy.dataset.file = "explore.html";
    legacy.href = "explore.html";
    legacy.title = "The original single-page explorer, kept unchanged";
    nav.appendChild(legacy);
    mast.parentNode.insertBefore(nav, mast.nextSibling);
  }

  function onward(host, currentId) {
    var i = PAGES.findIndex(function (p) { return p.id === currentId; });
    var box = h("div", "onward");
    [PAGES[i - 1], PAGES[i + 1]].forEach(function (p, k) {
      if (!p) return;
      var a = h("a", null, '<span class="n">' + (k ? "NEXT · " : "BACK · ") + p.no + "</span>" +
        '<span class="t">' + p.title + '</span><span class="d">' + p.blurb + "</span>");
      a.dataset.file = p.file;
      a.href = hrefFor(p.file);
      box.appendChild(a);
    });
    host.appendChild(box);
    host.appendChild(h("div", "foot",
      "GROUNDTRUTH · NASA Space Apps · NISAR L2 GCOV, HH and HV · USGS 01311850 and 01311875 · " +
      "marsh polygons from OpenStreetMap · every figure is drawn from the exported products, which Python computed · " +
      '<a href="explore.html">legacy one-pager</a>'));
  }

  // --- reusable figures ------------------------------------------------------

  // Radar map with marsh polygons, gauges, restoration locators and the
  // missing-footprint hatch for the selected pass.
  function drawMap(host, opt) {
    opt = opt || {};
    var p = BYID[state.selected], wrap = h("div", "mapwrap");
    host.appendChild(wrap);
    if (!BASE) { wrap.appendChild(h("p", "cap", "basemap.json not found; run make_basemap.py to draw the map.")); return; }
    var W = BASE.width, H = BASE.height;
    var svg = el("svg", { class: "map", viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": "Radar map of Jamaica Bay with marsh polygons" });
    svg.appendChild(el("image", { href: BASE.image, x: 0, y: 0, width: W, height: H, preserveAspectRatio: "none" }));
    BASE.regions.forEach(function (r) {
      var poly = el("polygon", {
        class: "region" + (r.region_id === state.region ? " on" : ""),
        points: r.ring.map(function (c) { return c.join(","); }).join(" ")
      });
      if (opt.regionFill) {
        var f = opt.regionFill(r.region_id);
        if (f) { poly.setAttribute("fill", f.fill); poly.setAttribute("fill-opacity", f.opacity == null ? 0.78 : f.opacity); poly.setAttribute("stroke", f.stroke || "#55664f"); }
      }
      poly.appendChild(el("title", {}, opt.regionTitle ? opt.regionTitle(r.region_id) : "Marsh polygon " + r.region_id + " (OpenStreetMap). Click for its series."));
      poly.onclick = function () { state.region = state.region === r.region_id ? null : r.region_id; redraw(); };
      svg.appendChild(poly);
    });
    var cov = BASE.coverage[state.selected];
    if (cov) svg.appendChild(el("image", { href: cov.image, x: 0, y: 0, width: W, height: H, preserveAspectRatio: "none", "pointer-events": "none" }));
    (BASE.projects || []).forEach(function (q) {
      var x = q.px[0], y = q.px[1];
      var g = el("g", {}, [el("path", { d: "M" + x + " " + (y - 5) + "L" + (x + 5) + " " + y + "L" + x + " " + (y + 5) + "L" + (x - 5) + " " + y + "Z", fill: PAPER, stroke: INK, "stroke-width": 1.2 })]);
      var f = PROJECTS && PROJECTS.features.filter(function (z) { return z.properties.project_id === q.project_id; })[0];
      g.appendChild(el("title", {}, (f ? f.properties.project_name : q.project_id) + "\n" + q.geometry_kind + " — not a project footprint"));
      svg.appendChild(g);
    });
    Object.keys(BASE.gauges || {}).forEach(function (k) {
      var g = BASE.gauges[k], x = g.px[0], y = g.px[1];
      svg.appendChild(el("path", { d: "M" + (x - 4.5) + " " + (y + 4) + "L" + (x + 4.5) + " " + (y + 4) + "L" + x + " " + (y - 5) + "Z", fill: INK, stroke: PAPER, "stroke-width": 1 }));
      svg.appendChild(el("text", { x: x + 7, y: y + 4, fill: PAPER, stroke: INK, "stroke-width": 2.6, "paint-order": "stroke", style: "font:600 9.5px var(--sans)" }, k[0].toUpperCase() + k.slice(1) + " gauge"));
    });
    var km = 5000 / BASE.pixel_m, bx = W - km - 14, by = H - 14;
    svg.appendChild(el("rect", { x: bx - 6, y: by - 15, width: km + 12, height: 22, fill: PAPER, "fill-opacity": 0.85 }));
    svg.appendChild(el("path", { d: "M" + bx + " " + (by - 4) + "V" + by + "H" + (bx + km) + "V" + (by - 4), fill: "none", stroke: INK, "stroke-width": 1.2 }));
    svg.appendChild(el("text", { x: bx + km / 2, y: by - 5, "text-anchor": "middle", fill: INK, style: "font:9px var(--mono)" }, "5 km"));
    svg.appendChild(el("text", { x: 14, y: 24, fill: INK, style: "font:600 10px var(--sans)", "text-anchor": "middle" }, "N"));
    svg.appendChild(el("path", { d: "M14 28V44M10.5 33L14 28L17.5 33", fill: "none", stroke: INK, "stroke-width": 1.2 }));
    wrap.appendChild(svg);
    if (opt.legend !== false) host.appendChild(greyLegend(cov, p));
    return wrap;
  }

  // The grayscale key the map needs: which end of the ramp is a strong return.
  function greyLegend(cov, p) {
    var box = h("div");
    var d = BASE.display || { db_min: -24, db_max: 2, gamma: 0.8, rgb_at_db_min: [226, 232, 232], rgb_at_db_max: [74, 92, 102] };
    function rampColor(t) {
      var g = Math.pow(t, d.gamma || 1);
      return "rgb(" + [0, 1, 2].map(function (i) {
        return Math.round(d.rgb_at_db_min[i] + (d.rgb_at_db_max[i] - d.rgb_at_db_min[i]) * g);
      }).join(",") + ")";
    }
    var W = 330, H = 40;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, style: "max-width:330px;margin-top:10px", role: "img", "aria-label": "Key for the radar base image, in decibels" });
    // The ramp is drawn from the same stretch the PNG was written with, so the
    // key cannot drift away from the picture.
    for (var i = 0; i < 60; i++) {
      svg.appendChild(el("rect", { x: i * (W / 60), y: 2, width: W / 60 + 0.6, height: 13, fill: rampColor(i / 59) }));
    }
    svg.appendChild(el("rect", { x: 0, y: 2, width: W, height: 13, fill: "none", stroke: INK, "stroke-width": 0.8 }));
    [0, 0.5, 1].forEach(function (f) {
      var v = d.db_min + (d.db_max - d.db_min) * f;
      svg.appendChild(el("text", { x: f * W, y: 27, "text-anchor": f === 0 ? "start" : f === 1 ? "end" : "middle", class: "labm", style: "font-size:10px" }, fx(v, 0) + " dB"));
    });
    svg.appendChild(el("text", { x: 0, y: 38, class: "axl", style: "font-size:10px" }, "weak return · open water"));
    svg.appendChild(el("text", { x: W, y: 38, "text-anchor": "end", class: "axl", style: "font-size:10px" }, "strong return · city, vegetation"));
    box.appendChild(svg);
    box.appendChild(h("p", "cap",
      (cov ? '<span class="hatch"></span><b>Hatched: no valid radar on ' + p.date + ".</b> This pass covered " +
        pct(cov.marsh_coverage_fraction) + " of the marsh pixels and " + pct(cov.upland_coverage_fraction) +
        " of the upland reference, so its bay-wide median describes a different area from the other dates. " : "") +
      "The base is the temporal median of " + MODELS.tracks.descending_069.series.all_dates.n +
      " descending HH images at " + BASE.pixel_m.toFixed(0) + " m, on the fixed " + fx(d.db_min, 0) + " to " + fx(d.db_max, 0) +
      " dB stretch above. The stretch never changes with the selection. <b>Ink is a strong radar return</b> " +
      "(city, vegetation); pale is a weak return (open water) — the ramp runs the opposite way from the usual " +
      "grayscale convention, so the built city reads as ink on paper. " +
      "It is not a shoreline or a flood map and it does not move with the tide. " +
      "Green outlines are the sampled marsh polygons; ◆ are approximate restoration locators, not footprints; ▲ are the two gauges."));
    return box;
  }

  function drawStaff(host, p) {
    var W = 84, H = 422, top = 26, bot = H - 34, lo = -1.0, hi = 1.25,
        y = scale(lo, hi, bot, top), x0 = 30, w = 16, o = ORBITS[p.orbit];
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": "Tide staff: " + p.h.toFixed(2) + " metres NAVD88 at the selected overpass" });
    svg.appendChild(el("text", { x: x0 + w / 2, y: 12, "text-anchor": "middle", class: "axl" }, "m NAVD88"));
    svg.appendChild(el("rect", { x: x0, y: y(p.h), width: w, height: bot - y(p.h), fill: o.color, "fill-opacity": 0.22 }));
    svg.appendChild(el("rect", { x: x0, y: top, width: w, height: bot - top, fill: "none", stroke: INK, "stroke-width": 1.2 }));
    for (var v = lo; v <= hi + 1e-9; v += 0.25) {
      var major = Math.abs(v * 2 - Math.round(v * 2)) < 1e-9;
      svg.appendChild(el("line", { x1: x0, x2: x0 + (major ? w : w * 0.45), y1: y(v), y2: y(v), stroke: INK, "stroke-width": Math.abs(v) < 1e-9 ? 2 : 0.8 }));
      if (major) svg.appendChild(el("text", { x: x0 - 5, y: y(v) + 3.5, class: "labm", style: "font-size:10px" }, sgn(v, 1)));
    }
    PTS[p.orbit].forEach(function (q) {
      svg.appendChild(el("line", { x1: x0 + w + 3, x2: x0 + w + 9, y1: y(q.h), y2: y(q.h), stroke: o.color, "stroke-width": 1, "stroke-opacity": 0.55 }));
    });
    var e = p.env;
    [["I", e.usgs_inwood_water_level_m_navd88], ["R", e.usgs_rockaway_water_level_m_navd88]].forEach(function (pair) {
      if (pair[1] == null) return;
      svg.appendChild(el("line", { x1: x0 + w + 12, x2: x0 + w + 19, y1: y(pair[1]), y2: y(pair[1]), stroke: INK, "stroke-width": 1.2 }));
      svg.appendChild(el("text", { x: x0 + w + 21, y: y(pair[1]) + 3.5, class: "labm", style: "font-size:9px" }, pair[0]));
    });
    svg.appendChild(el("path", { d: "M" + (x0 - 3) + " " + y(p.h) + "H" + (x0 + w + 3), stroke: SEL, "stroke-width": 2.4 }));
    svg.appendChild(el("path", { d: "M" + (x0 - 3) + " " + y(p.h) + "l-6 -4v8z", fill: SEL }));
    svg.appendChild(el("text", { x: x0 + w / 2, y: H - 14, "text-anchor": "middle", class: "labm", style: "font-size:13px;font-weight:600" }, sgn(p.h) + " m"));
    svg.appendChild(el("text", { x: x0 + w / 2, y: H - 2, "text-anchor": "middle", class: "axl", style: "font-size:9px" }, "at overpass"));
    svg.appendChild(el("title", {}, "Tide staff. Bar: mean of the two gauges at the selected overpass. I, R: each gauge. Side ticks: every pass on this track."));
    host.appendChild(svg);
  }

  // A forest plot: estimates on one scale, with a zero reference.
  function forest(host, rows, opt) {
    opt = opt || {};
    var W = opt.width || 760, rowH = opt.rowH || 34, labelW = opt.labelW || 236,
        P = { l: labelW, r: opt.right || 168, t: 26, b: 34 }, H = P.t + rows.length * rowH + P.b;
    var lo = Math.min.apply(null, rows.map(function (r) { return r.ci ? r.ci[0] : r.value; }).concat([0]));
    var hi = Math.max.apply(null, rows.map(function (r) { return r.ci ? r.ci[1] : r.value; }).concat([0]));
    var pad = (hi - lo) * 0.08 || 0.1;
    var x = scale(lo - pad, hi + pad, P.l, W - P.r);
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": opt.label || "Coefficient plot" });
    ticks(lo - pad, hi + pad, 5).forEach(function (t) {
      svg.appendChild(el("line", { x1: x(t), x2: x(t), y1: P.t - 6, y2: H - P.b, stroke: RULE }));
      svg.appendChild(el("text", { x: x(t), y: H - P.b + 15, "text-anchor": "middle", class: "axis" , style:"fill:" + MUTED + ";font:10.5px var(--mono)"}, sgn(t, 1)));
    });
    svg.appendChild(el("line", { x1: x(0), x2: x(0), y1: P.t - 6, y2: H - P.b, stroke: INK, "stroke-width": 1.3 }));
    svg.appendChild(el("text", { x: x(0), y: P.t - 12, "text-anchor": "middle", class: "axl" }, "no relationship"));
    svg.appendChild(el("text", { x: (P.l + W - P.r) / 2, y: H - 6, "text-anchor": "middle", class: "axl" }, opt.xlabel || "slope, dB per metre of water level"));
    rows.forEach(function (r, i) {
      var cy = P.t + i * rowH + rowH / 2, color = r.color || INK;
      if (r.rule) svg.appendChild(el("line", { x1: 0, x2: W, y1: P.t + i * rowH - 1, y2: P.t + i * rowH - 1, stroke: RULE }));
      svg.appendChild(el("text", { x: labelW - 12, y: cy + 4, "text-anchor": "end", class: "lab", style: "font-size:12px" + (r.bold ? ";font-weight:600" : "") }, r.label));
      if (r.sub) svg.appendChild(el("text", { x: labelW - 12, y: cy + 15, "text-anchor": "end", class: "axl", style: "font-size:10px" }, r.sub));
      if (r.ci) {
        svg.appendChild(el("line", { x1: x(r.ci[0]), x2: x(r.ci[1]), y1: cy, y2: cy, stroke: color, "stroke-width": r.bold ? 2.2 : 1.6, "stroke-opacity": r.faint ? 0.5 : 1 }));
        [r.ci[0], r.ci[1]].forEach(function (v) {
          svg.appendChild(el("line", { x1: x(v), x2: x(v), y1: cy - 4.5, y2: cy + 4.5, stroke: color, "stroke-width": r.bold ? 2.2 : 1.6, "stroke-opacity": r.faint ? 0.5 : 1 }));
        });
      }
      var shape = r.shape === "square"
        ? el("rect", { x: x(r.value) - 4.4, y: cy - 4.4, width: 8.8, height: 8.8, fill: r.hollow ? PAPER : color, stroke: color, "stroke-width": 1.8 })
        : el("circle", { cx: x(r.value), cy: cy, r: r.bold ? 5.4 : 4.4, fill: r.hollow ? PAPER : color, stroke: color, "stroke-width": 1.8 });
      svg.appendChild(shape);
      svg.appendChild(el("text", { x: W - P.r + 10, y: cy + 4, class: "labm", style: "font-size:11.5px" },
        fx(r.value) + (r.ci ? "  " + ci(r.ci) : "")));
      svg.appendChild(el("title", {}, r.label + (r.sub ? " — " + r.sub : "") + "\n" + fx(r.value) + (r.ci ? " dB/m, 95% CI " + ci(r.ci) : "")));
    });
    host.appendChild(svg);
    return svg;
  }

  // --- boot ------------------------------------------------------------------
  function getJSON(name, optional) {
    return fetch(new URL(name, DATA)).then(function (r) {
      if (!r.ok) throw new Error(name + ": HTTP " + r.status);
      return r.json();
    }).catch(function (e) { if (optional) return null; throw e; });
  }

  function boot(page, render) {
    PAGE = page; RENDER = render;
    readURL();
    Promise.all([
      getJSON("models.json"), getJSON("timeseries.json"), getJSON("hydrographs.json"),
      getJSON("environment.json"), getJSON("acquisitions.json"),
      getJSON("basemap.json", true), getJSON("restoration_projects.geojson", true),
      getJSON("robustness.json", true)
    ]).then(function (all) {
      MODELS = all[0]; SERIES = all[1]; HYDRO = all[2]; ENV = all[3]; ACQ = all[4];
      BASE = all[5]; PROJECTS = all[6]; ROBUST = all[7];
      var env = {}, acq = {};
      ENV.observations.forEach(function (o) { env[o.acquisition_id] = o; });
      ACQ.acquisitions.forEach(function (a) { acq[a.acquisition_id] = a; });
      Object.keys(ORBITS).forEach(function (orbit) {
        PTS[orbit] = SERIES.pooled[orbit].map(function (p) {
          return {
            id: p.acquisition_id, orbit: orbit, date: p.date, t: Date.parse(p.date + "T00:00:00Z"),
            h: p.mean_water_level_m_navd88, y: p.marsh_relative_db, hh: p.marsh_hh_db,
            upland: p.upland_hh_db, hv: p.marsh_hv_db, yhv: p.marsh_relative_hv_db,
            ok: p.passes_coverage_screen, env: env[p.acquisition_id] || { quality_flags: [] },
            acq: acq[p.acquisition_id] || {}
          };
        });
        PTS[orbit].sort(function (a, b) { return a.t - b.t; });
        PTS[orbit].forEach(function (p) { BYID[p.id] = p; });
      });
      if (!state.selected || !BYID[state.selected]) state.selected = PTS.descending_069[0].id;
      state.orbit = BYID[state.selected].orbit;
      drawChrome(page);
      redraw();
      var host = document.getElementById("app");
      onward(host.parentNode, page);
    }).catch(function (e) {
      document.getElementById("app").innerHTML =
        '<div class="err"><b>Could not load the data products.</b><br>' + e.message + "<br><br>" +
        "Expected JSON under <code>" + DATA.href + "</code>. Generate with <code>tide_adjust.py</code>, " +
        "<code>make_basemap.py</code> and <code>make_robustness.py</code>, or pass <code>?data=&lt;url&gt;</code>.</div>";
    });
  }

  return {
    boot: boot, select: select, set: set, redraw: redraw, state: state,
    ORBITS: ORBITS, THRESHOLDS: THRESHOLDS, PAGES: PAGES,
    INK: INK, MUTED: MUTED, FAINT: FAINT, RULE: RULE, MARSH: MARSH, PAPER: PAPER, SEL: SEL, WARN: WARN,
    el: el, h: h, scale: scale, ticks: ticks, ext: ext, median: median, axes: axes,
    sgn: sgn, fx: fx, pct: pct, pv: pv, ci: ci, days: days, fmtDate: fmtDate,
    monthTicks: monthTicks, monthFmt: monthFmt, mark: mark, im: im, def: def,
    forest: forest, drawMap: drawMap, drawStaff: drawStaff, hrefFor: hrefFor,
    variantOf: variantOf, model: model, robust: robust, inFit: inFit, fitPoints: fitPoints,
    get MODELS() { return MODELS; }, get SERIES() { return SERIES; }, get HYDRO() { return HYDRO; },
    get ENV() { return ENV; }, get ACQ() { return ACQ; }, get BASE() { return BASE; },
    get PROJECTS() { return PROJECTS; }, get ROBUST() { return ROBUST; },
    get PTS() { return PTS; }, get BYID() { return BYID; }
  };
})();
