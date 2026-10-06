"""Static dashboard process.

Serves ``frontend/`` on ``MM_FRONTEND_PORT`` (default 3000) and points the
browser at the FastAPI backend on ``MM_PORT`` (default 8088) via
``window.MM_API_ORIGIN``. Also serves ``lib/`` at ``/lib`` if present.

Usage (repo root, venv active):

    python -m frontend.dev_server
"""

from __future__ import annotations

import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from backend import settings

FRONTEND_DIR = settings.BASE_DIR / "frontend"
LIB_DIR = settings.BASE_DIR / "lib"


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def log_message(self, format, *args):
        print(f"frontend {self.address_string()} {format % args}")

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_GET(self):
        if self.path.split("?", 1)[0] == "/config.js":
            self._send_config()
            return
        lib_path = self.path.split("?", 1)[0]
        if lib_path.startswith("/lib/") and LIB_DIR.exists():
            self._send_lib(lib_path[len("/lib/") :])
            return
        super().do_GET()

    def _send_config(self):
        body = (
            "window.MM_API_ORIGIN = "
            + json.dumps(settings.API_ORIGIN)
            + ";\n"
            + "window.MM_FRONTEND_ORIGIN = "
            + json.dumps(settings.FRONTEND_ORIGIN)
            + ";\n"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/javascript; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_lib(self, relative):
        target = (LIB_DIR / relative).resolve()
        if not str(target).startswith(str(LIB_DIR.resolve())) or not target.is_file():
            self.send_error(404, "lib file not found")
            return
        data = target.read_bytes()
        ctype = self.guess_type(str(target))
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    host = settings.FRONTEND_HOST
    port = settings.FRONTEND_PORT
    httpd = ThreadingHTTPServer((host, port), partial(DashboardHandler))
    print(f"frontend http://{host}:{port}  ->  backend {settings.API_ORIGIN}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("frontend stopped")


if __name__ == "__main__":
    main()
