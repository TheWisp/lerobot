# /// script
# dependencies = ["markdown-it-py>=3,<5"]
# ///
"""Serve repository Markdown as formatted pages without maintaining HTML copies.

Run: uv run scripts/preview_markdown.py --port 18935
"""

import argparse
import html
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from markdown_it import MarkdownIt

STYLE = """
:root { color-scheme: light dark; font: 17px/1.65 system-ui, sans-serif; }
body { max-width: 900px; margin: 40px auto; padding: 0 24px 64px; }
h1, h2, h3 { line-height: 1.25; scroll-margin-top: 24px; }
h2 { margin-top: 2.4em; padding-top: .8em; border-top: 1px solid #8885; }
a { color: light-dark(#1658a3, #8bc5ff); }
code { background: #8882; border-radius: 4px; padding: .1em .3em; }
pre { overflow: auto; padding: 16px; background: #8882; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; border-bottom: 1px solid #8885; padding: 10px; }
img { max-width: 100%; height: auto; }
header { font-size: .85em; opacity: .8; margin-bottom: 28px; }
"""


def render(source):
    parser = MarkdownIt("commonmark").enable("table")
    tokens = parser.parse(source)
    seen = {}
    for index, token in enumerate(tokens):
        if token.type == "heading_open":
            title = tokens[index + 1].content
            slug = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
            count = seen.get(slug, 0)
            seen[slug] = count + 1
            token.attrSet("id", slug + (f"-{count}" if count else ""))
    return parser.renderer.render(tokens, parser.options, {})


class Preview(SimpleHTTPRequestHandler):
    def do_GET(self):
        url = urlsplit(self.path)
        path = Path(self.translate_path(url.path)).resolve()
        if not path.is_relative_to(Path(self.directory).resolve()):
            self.send_error(403)
            return
        if path.suffix != ".md" or not path.is_file() or url.query == "raw":
            super().do_GET()
            return
        content = (
            '<!doctype html><html><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{html.escape(path.name)}</title><style>{STYLE}</style></head><body>"
            '<header>Rendered from repository Markdown · <a href="?raw">View source</a></header>'
            f"<main>{render(path.read_text())}</main></body></html>"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18935)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    def handler(*arguments, **kwargs):
        return Preview(*arguments, directory=str(root), **kwargs)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"Markdown preview: http://127.0.0.1:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
