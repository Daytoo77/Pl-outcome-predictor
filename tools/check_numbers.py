"""Every number the site displays must trace to a committed artefact. This checks that it does.

    python tools/check_numbers.py

Elements in docs/**/*.html carry data-src="<artefact>#<path>" and, for numbers, data-fmt. The
element's text is compared with the value at that path:

    data-fmt="3"      float shown to 3 decimals (tolerance half a unit in the last place)
    data-fmt="pct1"   fraction shown as a percentage to 1 decimal
    data-fmt="int"    integer; "int," allows thousands separators
    (no data-fmt)     exact string match

Artefact names resolve to data/, models/ or docs/data/. Paths use dots for keys, [i] for list
indices and [key=value] for list lookups, e.g. ui_meta.json#rolling_origin[3].edge or
ui_predictions.json#[id=9531].market.H. Stdlib only.
"""
from __future__ import annotations

import json
import os
import re
import sys
from html.parser import HTMLParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
ARTEFACT_DIRS = [os.path.join(ROOT, "data"), os.path.join(ROOT, "models"), os.path.join(DOCS, "data")]
_cache = {}


def artefact(name):
    if name not in _cache:
        for d in ARTEFACT_DIRS:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                _cache[name] = json.load(open(p, encoding="utf-8"))
                break
        else:
            raise KeyError(f"artefact not found: {name}")
    return _cache[name]


TOKEN = re.compile(r"\[([^\]]*)\]|([^.\[\]]+)")


def lookup(obj, path):
    for m in TOKEN.finditer(path):
        bracket, key = m.group(1), m.group(2)
        if key is not None:
            obj = obj[key] if isinstance(obj, dict) else obj[int(key)]
        elif "=" in bracket:
            k, v = bracket.split("=", 1)
            matches = [x for x in obj if str(x.get(k)) == v]
            if len(matches) != 1:
                raise KeyError(f"[{bracket}] matched {len(matches)} items")
            obj = matches[0]
        else:
            obj = obj[int(bracket)]
    return obj


class Collector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []          # open elements carrying data-src: [src, fmt, line, text parts]
        self.depths = []         # nesting depth at which each tracked element opened
        self.depth = 0
        self.found = []
        self.void = {"br", "img", "meta", "link", "input", "hr", "source", "wbr"}

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in self.void:
            return
        self.depth += 1
        if a.get("data-src"):
            self.stack.append([a["data-src"], a.get("data-fmt"), self.getpos()[0], []])
            self.depths.append(self.depth)

    def handle_endtag(self, tag):
        if tag in self.void:
            return
        if self.depths and self.depths[-1] == self.depth:
            src, fmt, line, parts = self.stack.pop()
            self.depths.pop()
            self.found.append((src, fmt, line, "".join(parts).strip()))
        self.depth -= 1

    def handle_data(self, data):
        for item in self.stack:
            item[3].append(data)


def normalise(text):
    return text.replace("−", "-").replace(",", "").replace("+", "").replace("%", "").strip()


def check(text, fmt, value):
    """Return None if the displayed text matches the artefact value under fmt, else a message."""
    if fmt is None:
        return None if text == str(value) else f"text {text!r} != {value!r}"
    try:
        shown = float(normalise(text))
    except ValueError:
        return f"text {text!r} is not a number"
    if fmt.startswith("int"):
        return None if abs(shown - float(value)) < 1e-9 else f"shown {text} != {value}"
    if fmt == "pct1":
        target, tol = float(value) * 100, 0.05 + 1e-9
    else:
        dp = int(fmt)
        target, tol = float(value), 0.5 * 10 ** (-dp) + 1e-9
    return None if abs(shown - target) <= tol else f"shown {text} vs artefact {value} (fmt {fmt})"


def main():
    problems, n = [], 0
    for dirpath, _, files in os.walk(DOCS):
        for f in sorted(files):
            if not f.endswith(".html"):
                continue
            path = os.path.join(dirpath, f)
            rel = os.path.relpath(path, ROOT)
            c = Collector()
            c.feed(open(path, encoding="utf-8").read())
            for src, fmt, line, text in c.found:
                n += 1
                try:
                    name, jpath = src.split("#", 1)
                    value = lookup(artefact(name), jpath)
                except (KeyError, IndexError, ValueError, TypeError) as e:
                    problems.append(f"{rel}:{line}: {src}: cannot resolve ({e})")
                    continue
                msg = check(text, fmt, value)
                if msg:
                    problems.append(f"{rel}:{line}: {src}: {msg}")
    print(f"{n} displayed figures checked against their artefacts")
    for p in problems:
        print("MISMATCH:", p)
    if problems:
        print(f"{len(problems)} mismatch(es)")
        return 1
    print("every figure matches its artefact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
