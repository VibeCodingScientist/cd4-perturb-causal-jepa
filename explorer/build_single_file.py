#!/usr/bin/env python3
"""
build_single_file.py — bundle the CD4+ Perturbation Explorer into ONE
self-contained HTML file that opens via file:// with NO network access.

Produces: explorer_bundle.html (next to this script).

Strategy: use explorer.html as the template and, in place, replace the
external <link>/<script> references with inline <style>/<script> blocks,
preserving the exact ordering the browser would have loaded them in. All
fonts are embedded as base64 data: URIs and all data/*.json files are
injected as window.__APP_DATA__ before app.js runs, so nothing hits the
network at runtime.

Python 3, standard library only.
"""

import base64
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def read_text(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def read_bytes(path):
    with open(path, "rb") as fh:
        return fh.read()


def die(msg):
    sys.stderr.write("build_single_file.py: ERROR: %s\n" % msg)
    sys.exit(1)


def inline_fonts(fonts_css, fonts_dir):
    """Replace url("./NAME.woff2") in fonts.css with base64 data: URIs."""
    def repl(match):
        rel = match.group(1)
        name = os.path.basename(rel)
        woff2_path = os.path.join(fonts_dir, name)
        if not os.path.isfile(woff2_path):
            # Leave untouched rather than break; font simply falls back.
            sys.stderr.write("  warn: font not found, left as-is: %s\n" % name)
            return match.group(0)
        b64 = base64.b64encode(read_bytes(woff2_path)).decode("ascii")
        return 'url("data:font/woff2;base64,%s")' % b64

    # Match url("./Something.woff2") with optional whitespace/quote styles.
    pattern = re.compile(r'url\(\s*["\']?\.?/?([^"\')]+\.woff2)["\']?\s*\)')
    return pattern.sub(repl, fonts_css)


def build_css():
    """Read style.css, resolve its @import of fonts.css (with inlined fonts)."""
    style_path = os.path.join(HERE, "style.css")
    if not os.path.isfile(style_path):
        die("style.css not found at %s" % style_path)
    css = read_text(style_path)

    fonts_dir = os.path.join(HERE, "assets", "fonts")
    fonts_css_path = os.path.join(fonts_dir, "fonts.css")

    if os.path.isfile(fonts_css_path):
        fonts_css = read_text(fonts_css_path)
        fonts_css = inline_fonts(fonts_css, fonts_dir)
    else:
        sys.stderr.write("  warn: fonts.css not found; dropping @import\n")
        fonts_css = ""

    # Replace the @import line for fonts.css with the resolved font-face rules.
    import_re = re.compile(
        r'@import\s+url\(\s*["\']?[^"\')]*fonts\.css["\']?\s*\)\s*;'
    )
    if import_re.search(css):
        css = import_re.sub(lambda m: fonts_css, css, count=1)
    else:
        # No @import found — prepend fonts so faces are still available.
        sys.stderr.write("  warn: fonts.css @import not found in style.css\n")
        css = fonts_css + "\n" + css

    return css


def build_data_script():
    """Inline every data/*.json as window.__APP_DATA__ keyed by basename."""
    data_dir = os.path.join(HERE, "data")
    payload = {}
    if not os.path.isdir(data_dir):
        die("data/ directory not found at %s" % data_dir)

    names = sorted(f for f in os.listdir(data_dir) if f.endswith(".json"))
    if not names:
        die("no JSON files found in %s" % data_dir)

    for fname in names:
        key = fname[:-len(".json")]
        raw = read_text(os.path.join(data_dir, fname))
        try:
            payload[key] = json.loads(raw)
        except ValueError as exc:
            die("invalid JSON in %s: %s" % (fname, exc))

    # Serialize compactly; escape </ so a nested "</script>" can't close the tag.
    js = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    js = js.replace("</", "<\\/")
    return "window.__APP_DATA__ = %s;" % js, list(payload.keys())


def script_block(path, label):
    if not os.path.isfile(path):
        die("expected script not found: %s (%s)" % (path, label))
    body = read_text(path)
    # Guard against a stray </script> in vendored code closing our tag early.
    body = body.replace("</script>", "<\\/script>")
    return "<script>\n%s\n</script>" % body


def main():
    template_path = os.path.join(HERE, "explorer.html")
    if not os.path.isfile(template_path):
        die("explorer.html not found at %s" % template_path)
    html = read_text(template_path)

    # --- 1. Inline CSS: replace the <link rel="stylesheet"> line. ---
    css = build_css()
    style_tag = "<style>\n%s\n</style>" % css
    link_re = re.compile(
        r'<link\b[^>]*rel=["\']stylesheet["\'][^>]*>', re.IGNORECASE
    )
    if not link_re.search(html):
        die('could not find <link rel="stylesheet"> in explorer.html')
    html = link_re.sub(lambda m: style_tag, html, count=1)

    # --- 2. Build inline data script (goes before app.js). ---
    data_script, data_keys = build_data_script()
    data_tag = "<script>\n%s\n</script>" % data_script

    # --- 3. Inline every <script src="..."> in html order. ---
    # We resolve each src to a local file and swap in an inline block. The
    # data script is prepended to the app.js block so it lands right before it.
    src_re = re.compile(r'<script\b[^>]*\bsrc=["\']([^"\']+)["\'][^>]*>\s*</script>',
                        re.IGNORECASE)

    def replace_src(match):
        src = match.group(1)
        rel = src.lstrip("./")
        abs_path = os.path.join(HERE, rel)
        block = script_block(abs_path, src)
        # Insert offline data immediately before app.js so loadOne() finds it,
        # and after d3 (app.js is loaded after d3 in the template).
        if os.path.basename(rel) == "app.js":
            return data_tag + "\n  " + block
        return block

    html, n = src_re.subn(replace_src, html)
    if n == 0:
        die("no <script src> tags found to inline in explorer.html")

    # --- 4. Sanity checks: data before app.js, d3 before app.js. ---
    idx_data = html.find("window.__APP_DATA__")
    idx_app = html.find("App.registerPanel")  # unique-ish token from app.js
    if idx_app < 0:
        idx_app = html.find("loadOne")  # fallback token from app.js
    idx_d3 = html.find("d3 = ")
    if idx_d3 < 0:
        idx_d3 = html.find("//# sourceMappingURL")  # d3 minified tail marker
    if idx_data < 0:
        die("window.__APP_DATA__ missing from output")
    if idx_app < 0:
        die("app.js body missing from output")
    if idx_data > idx_app:
        die("window.__APP_DATA__ not placed before app.js body")
    # d3 must appear before app.js body.
    idx_d3_first = html.find("d3=function")
    if idx_d3_first < 0:
        idx_d3_first = html.lower().find("d3")
    if idx_d3_first < 0 or idx_d3_first > idx_app:
        sys.stderr.write("  warn: could not positively confirm d3 before app.js\n")

    # --- 5. Write output. ---
    out_path = os.path.join(HERE, "explorer_bundle.html")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    size_kb = os.path.getsize(out_path) / 1024.0
    print("Wrote: %s" % out_path)
    print("Size:  %.1f KB" % size_kb)
    print("Data:  window.__APP_DATA__ keys = %s" % ", ".join(data_keys))
    print("Runs fully offline via file:// - no network required.")


if __name__ == "__main__":
    main()
