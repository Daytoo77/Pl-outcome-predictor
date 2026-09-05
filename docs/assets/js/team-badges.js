/* team-badges.js: self-hosted club badges, no external crest assets (no logo-rights issue,
   no external CDN). Each badge is a coloured disc with a 1-3 letter monogram in the club's
   real primary shirt colour, generated as inline SVG. Deterministic per name; unknown teams
   fall back to a neutral grey disc with their first two letters.

   window.PL.badge(name, size)          -> an <svg> element
   window.PL.badgeHTML(name, size)      -> the same, as an HTML string (for innerHTML use)
*/
(function () {
  'use strict';

  // Real (approximate) primary shirt/badge colours, ink chosen for contrast on that fill.
  var TEAMS = {
    "Arsenal":         { m: "AFC", fill: "#EF0107", ink: "#ffffff" },
    "Aston Villa":     { m: "AVL", fill: "#95BFE5", ink: "#1a1a2e" },
    "Bournemouth":     { m: "BOU", fill: "#DA291C", ink: "#ffffff" },
    "Brentford":       { m: "BRE", fill: "#E30613", ink: "#ffffff" },
    "Brighton":        { m: "BHA", fill: "#0057B8", ink: "#ffffff" },
    "Burnley":         { m: "BUR", fill: "#6C1D45", ink: "#ffffff" },
    "Chelsea":         { m: "CHE", fill: "#034694", ink: "#ffffff" },
    "Coventry":        { m: "COV", fill: "#78D0F3", ink: "#1a1a2e" },
    "Crystal Palace":  { m: "CRY", fill: "#1B458F", ink: "#ffffff" },
    "Everton":         { m: "EVE", fill: "#003399", ink: "#ffffff" },
    "Fulham":          { m: "FUL", fill: "#000000", ink: "#ffffff" },
    "Hull":            { m: "HUL", fill: "#F18A00", ink: "#1a1a2e" },
    "Ipswich":         { m: "ITF", fill: "#3A64A3", ink: "#ffffff" },
    "Leeds":           { m: "LUFC", fill: "#FFCD00", ink: "#1a1a2e" },
    "Leicester":       { m: "LCFC", fill: "#003090", ink: "#ffffff" },
    "Liverpool":       { m: "LIV", fill: "#C8102E", ink: "#ffffff" },
    "Luton":           { m: "LTFC", fill: "#F78F1E", ink: "#1a1a2e" },
    "Man City":        { m: "MCI", fill: "#6CABDD", ink: "#1a1a2e" },
    "Man United":      { m: "MUFC", fill: "#DA291C", ink: "#ffffff" },
    "Newcastle":       { m: "NUFC", fill: "#241F20", ink: "#ffffff" },
    "Norwich":         { m: "NCFC", fill: "#FFF200", ink: "#1a1a2e" },
    "Nott'm Forest":   { m: "NFFC", fill: "#DD0000", ink: "#ffffff" },
    "Sheffield United":{ m: "SUFC", fill: "#EE2737", ink: "#ffffff" },
    "Southampton":     { m: "SFC", fill: "#D71920", ink: "#ffffff" },
    "Sunderland":      { m: "SAFC", fill: "#EB172B", ink: "#ffffff" },
    "Tottenham":       { m: "THFC", fill: "#132257", ink: "#ffffff" },
    "Watford":         { m: "WFC", fill: "#FBEE23", ink: "#1a1a2e" },
    "West Brom":       { m: "WBA", fill: "#122F67", ink: "#ffffff" },
    "West Ham":        { m: "WHU", fill: "#7A263A", ink: "#ffffff" },
    "Wolves":          { m: "WOL", fill: "#FDB913", ink: "#1a1a2e" },
    "Burton":          { m: "BAFC", fill: "#FDE401", ink: "#1a1a2e" },
    "Portsmouth":      { m: "POR", fill: "#001489", ink: "#ffffff" },
    "Derby":           { m: "DCFC", fill: "#000000", ink: "#ffffff" },
    "Birmingham":      { m: "BCFC", fill: "#0000FF", ink: "#ffffff" },
    "Blackburn":       { m: "BRFC", fill: "#009EE0", ink: "#ffffff" },
    "Bolton":          { m: "BWFC", fill: "#8B1D41", ink: "#ffffff" },
    "Charlton":        { m: "CAFC", fill: "#D2122E", ink: "#ffffff" },
    "Middlesbrough":   { m: "MFC", fill: "#DA1F27", ink: "#ffffff" },
    "Reading":         { m: "RFC", fill: "#004494", ink: "#ffffff" },
    "Stoke":           { m: "SCFC", fill: "#E03A3E", ink: "#ffffff" },
    "Swansea":         { m: "SWA", fill: "#121212", ink: "#ffffff" },
    "Wigan":           { m: "WAFC", fill: "#1B5AA5", ink: "#ffffff" },
    "Cardiff":         { m: "CCFC", fill: "#0070B5", ink: "#ffffff" },
    "Huddersfield":    { m: "HTAFC", fill: "#0E63AD", ink: "#ffffff" },
    "Bradford":        { m: "BCAFC", fill: "#8A1D3E", ink: "#ffffff" },
    "Blackpool":       { m: "BFC", fill: "#F68712", ink: "#1a1a2e" }
  };
  var FALLBACK_FILL = "#6e6e73", FALLBACK_INK = "#ffffff";

  function entry(name) {
    var t = TEAMS[name];
    if (t) return t;
    var letters = String(name || "?").replace(/[^A-Za-z]/g, "").slice(0, 3).toUpperCase() || "?";
    return { m: letters, fill: FALLBACK_FILL, ink: FALLBACK_INK };
  }

  function svgString(name, size) {
    size = size || 28;
    var t = entry(name);
    var fs = t.m.length >= 4 ? size * 0.28 : size * 0.36;
    var label = (name || "Unknown club") + " badge";
    return '<svg viewBox="0 0 40 40" width="' + size + '" height="' + size +
      '" role="img" aria-label="' + label.replace(/"/g, '&quot;') + '" focusable="false">' +
      '<circle cx="20" cy="20" r="19" fill="' + t.fill + '" stroke="rgba(0,0,0,0.12)" stroke-width="1"/>' +
      '<text x="20" y="20" text-anchor="middle" dominant-baseline="central" ' +
      'font-family="-apple-system,BlinkMacSystemFont,\'Segoe UI\',Inter,system-ui,sans-serif" ' +
      'font-weight="700" font-size="' + (fs * 40 / size) + '" fill="' + t.ink + '">' + t.m + '</text>' +
      '</svg>';
  }

  window.PL = window.PL || {};
  window.PL.badgeHTML = svgString;
  window.PL.badge = function (name, size) {
    var wrap = document.createElement('span');
    wrap.innerHTML = svgString(name, size);
    return wrap.firstElementChild;
  };
})();
