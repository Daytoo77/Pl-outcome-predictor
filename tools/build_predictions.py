"""Assemble docs/predictions.html as one self-contained file.

    python tools/build_predictions.py

Inlines the shared CSS (docs/assets/css/*.css), the site script (docs/assets/js/site.js) and the
committed predictions data (data/ui_predictions.json, data/ui_meta.json) into
tools/predictions.template.html. The result works offline as a single file, from a local clone,
and after every model export the page is rebuilt with one command. Stdlib only, deterministic.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
TEMPLATE = os.path.join(ROOT, "tools", "predictions.template.html")
OUT = os.path.join(DOCS, "predictions.html")


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def json_for_script(path):
    """Compact JSON, safe inside a <script> element (no `</` sequences, no U+2028/2029)."""
    obj = json.load(open(path, encoding="utf-8"))
    text = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    return text.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def main():
    html = read(TEMPLATE)
    subs = {
        "/*__BASE_CSS__*/": read(os.path.join(DOCS, "assets", "css", "base.css")),
        "/*__COMPONENTS_CSS__*/": read(os.path.join(DOCS, "assets", "css", "components.css")),
        "/*__SITE_JS__*/": read(os.path.join(DOCS, "assets", "js", "site.js")),
        "/*__BADGES_JS__*/": read(os.path.join(DOCS, "assets", "js", "team-badges.js")),
        "__PREDICTIONS_JSON__": json_for_script(os.path.join(ROOT, "data", "ui_predictions.json")),
        "__META_JSON__": json_for_script(os.path.join(ROOT, "data", "ui_meta.json")),
    }
    for key, value in subs.items():
        if key not in html:
            print(f"template is missing placeholder {key}", file=sys.stderr)
            return 1
        html = html.replace(key, value, 1)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(html)
    print(f"wrote {os.path.relpath(OUT, ROOT)}  ({os.path.getsize(OUT):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
