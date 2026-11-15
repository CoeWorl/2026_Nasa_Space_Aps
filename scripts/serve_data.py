"""Serve the viewer and the analysis rasters over HTTP, for local development.

Two explicit roots, because serving the repository root would expose .env
and secrets/netrc to anything that can reach the port:

    /            -> web/            the standalone viewer
    /data/...    -> data/interim/   rasters and provenance
    /data/index.json                synthesised listing of what is there

Supports HTTP Range, which GeoTIFF readers use to fetch only the tiles
they need, and sends CORS headers so a separate Vite dev server on
another port can read from here.

This is a development tool. It is not hardened, and `make serve` binds it
to 127.0.0.1 so it is not reachable from the network.
"""

from __future__ import annotations

import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB_ROOT = Path(os.environ.get("WEB_ROOT", "/web")).resolve()
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/data")).resolve() / "interim"
# The small JSON/GeoJSON data products for the app pages, served read-only at
# /data-app/. A third explicit root, for the same reason as the other two.
APP_ROOT = (DATA_ROOT.parent / "processed" / "jamaica_bay" / "app").resolve()
# SERVE_PORT is the repo's own name for it; PORT is what most launchers set.
PORT = int(os.environ.get("SERVE_PORT") or os.environ.get("PORT") or "8080")

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".geojson": "application/geo+json; charset=utf-8",
    ".tif": "image/tiff",
    ".png": "image/png",
    ".svg": "image/svg+xml",
}
# Only these are servable from the data root.
DATA_SUFFIXES = {".tif", ".json", ".png"}
# Only these are servable from the app-products root.
APP_SUFFIXES = {".json", ".geojson"}

RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


class Handler(BaseHTTPRequestHandler):
    server_version = "groundtruth-dev"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(f"  {self.address_string()} {fmt % args}\n")

    # -- routing ------------------------------------------------------

    def resolve(self, path: str) -> tuple[Path | None, bytes | None]:
        """Map a URL path to a file, or to synthesised bytes."""
        path = path.split("?", 1)[0].split("#", 1)[0]

        if path in ("/", "/index.html"):
            return WEB_ROOT / "index.html", None

        if path.startswith("/data-app/"):
            name = path[len("/data-app/") :]
            candidate = (APP_ROOT / name).resolve()
            if APP_ROOT not in candidate.parents:
                return None, None
            if candidate.suffix.lower() not in APP_SUFFIXES:
                return None, None
            return candidate, None

        if path == "/data/index.json":
            return None, self.build_index()

        if path.startswith("/data/"):
            name = path[len("/data/") :]
            candidate = (DATA_ROOT / name).resolve()
            # Containment check defeats ../ traversal.
            if DATA_ROOT not in candidate.parents:
                return None, None
            if candidate.suffix.lower() not in DATA_SUFFIXES:
                return None, None
            return candidate, None

        candidate = (WEB_ROOT / path.lstrip("/")).resolve()
        if WEB_ROOT not in candidate.parents and candidate != WEB_ROOT:
            return None, None
        return candidate, None

    def build_index(self) -> bytes:
        """Describe the published rasters.

        Delegates to build_index.py, which is also what writes the file
        uploaded to R2. One implementation means the local contract and
        the Cloudflare contract cannot drift apart.
        """
        from build_index import build_index

        return json.dumps(build_index(DATA_ROOT), indent=2).encode()

    # -- responses ----------------------------------------------------

    def send_common(self, content_type: str, length: int) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Range")
        self.send_header(
            "Access-Control-Expose-Headers",
            "Content-Range, Content-Length, Accept-Ranges",
        )
        self.send_header("Cache-Control", "no-store")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Range")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_HEAD(self) -> None:
        self.respond(body=False)

    def do_GET(self) -> None:
        self.respond(body=True)

    def respond(self, *, body: bool) -> None:
        target, synthetic = self.resolve(self.path)

        if synthetic is not None:
            self.send_response(200)
            self.send_common("application/json; charset=utf-8", len(synthetic))
            self.end_headers()
            if body:
                self.wfile.write(synthetic)
            return

        if target is None or not target.is_file():
            message = b"not found\n"
            self.send_response(404)
            self.send_common("text/plain; charset=utf-8", len(message))
            self.end_headers()
            if body:
                self.wfile.write(message)
            return

        size = target.stat().st_size
        content_type = CONTENT_TYPES.get(
            target.suffix.lower(), "application/octet-stream"
        )

        # Range: what lets a GeoTIFF reader fetch single tiles instead of
        # the whole raster.
        header = self.headers.get("Range")
        match = RANGE_RE.match(header.strip()) if header else None
        if match:
            raw_start, raw_end = match.groups()
            if raw_start:
                start = int(raw_start)
                end = int(raw_end) if raw_end else size - 1
            else:
                # Suffix form: bytes=-N means the last N bytes.
                length = int(raw_end or 0)
                start = max(0, size - length)
                end = size - 1
            end = min(end, size - 1)

            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                return

            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_common(content_type, length)
            self.end_headers()
            if body:
                with target.open("rb") as handle:
                    handle.seek(start)
                    self.wfile.write(handle.read(length))
            return

        self.send_response(200)
        self.send_common(content_type, size)
        self.end_headers()
        if body:
            self.wfile.write(target.read_bytes())


def main() -> int:
    if not WEB_ROOT.is_dir():
        print(f"viewer directory {WEB_ROOT} is missing", file=sys.stderr)
        return 1
    if not DATA_ROOT.is_dir():
        print(f"data directory {DATA_ROOT} is missing", file=sys.stderr)
        return 1

    rasters = sorted(DATA_ROOT.glob("*.tif"))
    print(f"serving viewer from {WEB_ROOT}")
    if APP_ROOT.is_dir():
        products = sorted(p.name for p in APP_ROOT.glob("*.json")) + sorted(
            p.name for p in APP_ROOT.glob("*.geojson")
        )
        print(f"serving data products from {APP_ROOT} ({len(products)} file(s)) at /data-app/")
        print(f"  bay.html -> http://127.0.0.1:{PORT}/bay.html")
    else:
        print(f"no data products at {APP_ROOT} yet (run tide_adjust.py) - the app pages will show an error")
    print(f"serving rasters from {DATA_ROOT} ({len(rasters)} GeoTIFF(s))")
    for raster in rasters:
        print(f"  /data/{raster.name}")
    if not rasters:
        print("  (none yet - run 'make fetch DATE=...' first)")
    print()
    print(f"open http://localhost:{PORT}/")
    print("press ctrl-c to stop")

    # 0.0.0.0 inside the container (the Makefile publishes it on 127.0.0.1 only);
    # set SERVE_BIND=127.0.0.1 when running on the host directly.
    server = ThreadingHTTPServer((os.environ.get("SERVE_BIND", "0.0.0.0"), PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
