"""Fail on dead internal links in docs/.

    python tools/check_links.py

Checks every href / src in docs/**/*.html that points inside the site (relative paths, or paths
under the Pages base), including #fragments, plus the URLs in sitemap.xml. External http(s) links
are listed but not fetched, so the check is deterministic and offline. Stdlib only.
"""
from __future__ import annotations

import os
import re
import sys
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
BASE_PATH = "/pl-outcome-predictor/"
SITE = "https://daytoo77.github.io" + BASE_PATH
SKIP_SCHEMES = ("http:", "https:", "mailto:", "data:", "javascript:", "tel:")


class Collector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []      # (attr, value, line)
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if a.get("id"):
            self.ids.add(a["id"])
        if tag == "a" and a.get("name"):
            self.ids.add(a["name"])
        for attr in ("href", "src"):
            v = a.get(attr)
            if v:
                self.links.append((attr, v, self.getpos()[0]))


def parse(path):
    c = Collector()
    with open(path, encoding="utf-8") as fh:
        c.feed(fh.read())
    return c


def html_files():
    for dirpath, _, files in os.walk(DOCS):
        for f in files:
            if f.endswith(".html"):
                yield os.path.join(dirpath, f)


def resolve(from_file, target):
    """Map a link target to (filesystem path, fragment) inside docs/, or None if external."""
    if target.startswith(SKIP_SCHEMES) or target.startswith("//"):
        return None
    parts = urlsplit(target)
    path, frag = unquote(parts.path), parts.fragment
    if not path:
        return from_file, frag
    if path.startswith(BASE_PATH):
        fs = os.path.join(DOCS, path[len(BASE_PATH):].replace("/", os.sep))
    elif path.startswith("/"):
        return "ABSOLUTE:" + path, frag
    else:
        fs = os.path.normpath(os.path.join(os.path.dirname(from_file), path.replace("/", os.sep)))
    if os.path.isdir(fs):
        fs = os.path.join(fs, "index.html")
    return fs, frag


def main():
    pages = {p: parse(p) for p in html_files()}
    problems, external, checked = [], set(), 0
    for page, c in pages.items():
        rel = os.path.relpath(page, ROOT)
        for attr, value, line in c.links:
            r = resolve(page, value)
            if r is None:
                external.add(value)
                continue
            fs, frag = r
            checked += 1
            if fs.startswith("ABSOLUTE:"):
                problems.append(f"{rel}:{line}: {attr}=\"{value}\" is site-root absolute; use a relative path or the Pages base")
                continue
            if not os.path.isfile(fs):
                problems.append(f"{rel}:{line}: {attr}=\"{value}\" -> missing file {os.path.relpath(fs, ROOT)}")
                continue
            if frag and fs.endswith(".html"):
                ids = pages[fs].ids if fs in pages else parse(fs).ids
                if frag not in ids:
                    problems.append(f"{rel}:{line}: {attr}=\"{value}\" -> no element with id \"{frag}\" in {os.path.relpath(fs, ROOT)}")
    # sitemap entries must map to real files
    sm = os.path.join(DOCS, "sitemap.xml")
    if os.path.isfile(sm):
        for loc in re.findall(r"<loc>(.*?)</loc>", open(sm, encoding="utf-8").read()):
            checked += 1
            if not loc.startswith(SITE):
                problems.append(f"docs/sitemap.xml: {loc} is not under {SITE}")
                continue
            tail = loc[len(SITE):] or "index.html"
            if not os.path.isfile(os.path.join(DOCS, tail.replace("/", os.sep))):
                problems.append(f"docs/sitemap.xml: {loc} -> missing docs/{tail}")
    print(f"{len(pages)} pages, {checked} internal targets checked, {len(external)} distinct external links not fetched")
    for p in problems:
        print("DEAD:", p)
    if problems:
        print(f"{len(problems)} dead internal link(s)")
        return 1
    print("no dead internal links")
    return 0


if __name__ == "__main__":
    sys.exit(main())
