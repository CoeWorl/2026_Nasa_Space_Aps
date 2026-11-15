/**
 * GROUNDTRUTH artifact API.
 *
 * Serves analysis artifacts out of R2 with correct HTTP Range semantics,
 * which is the whole point: a Cloud Optimized GeoTIFF is only "cloud
 * optimized" if the client can fetch individual tiles. geotiff.js in the
 * browser issues range requests; if this Worker answered 200 with the
 * whole object every time, a click-to-inspect on one pixel would pull
 * eight megabytes.
 *
 * The bytes are streamed through untouched. This Worker never decodes,
 * re-encodes or reprojects a raster - that work belongs to the Python
 * processor, which owns and tests the science.
 */

interface Env {
  DATA: R2Bucket;
}

interface ByteRange {
  start: number;
  end: number;
}

const CORS: Record<string, string> = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
  "Access-Control-Allow-Headers": "Range, Content-Type",
  "Access-Control-Expose-Headers":
    "Content-Range, Content-Length, Accept-Ranges, ETag",
};

/** Only these keys are reachable. Never expose arbitrary R2 keys. */
const KEY_PATTERN = /^(data|results)\/[A-Za-z0-9._/-]+$/;
const ALLOWED_SUFFIX = /\.(tif|tiff|json|geojson|png|csv)$/;

/**
 * The atlas pages and their two shared assets, served from R2 so one origin
 * hosts page + data. An allowlist, not a prefix match: nothing under app/ is
 * reachable unless it is named here, and a page added to web/ has to be added
 * here too. "bay" is the baseline explorer and is also what "/" serves.
 */
const PAGES = [
  "bay",
  "finding",
  "observe",
  "evidence",
  "robustness",
  "holdout",
  "marshes",
  "matched",
  "context",
];
const ASSETS = ["atlas.css", "atlas.js"];
const ASSET_TYPES: Record<string, string> = {
  css: "text/css; charset=utf-8",
  js: "text/javascript; charset=utf-8",
};

const CONTENT_TYPES: Record<string, string> = {
  tif: "image/tiff",
  tiff: "image/tiff",
  json: "application/json; charset=utf-8",
  geojson: "application/geo+json",
  png: "image/png",
  csv: "text/csv; charset=utf-8",
  html: "text/html; charset=utf-8",
};

/**
 * Parse a single byte range, including the suffix form `bytes=-N`.
 *
 * Returns null for anything malformed or unsatisfiable, which the caller
 * turns into a 416. Multipart ranges are deliberately unsupported: no
 * GeoTIFF reader needs them, and a partial implementation would be worse
 * than an honest rejection.
 */
export function parseRange(value: string, size: number): ByteRange | null {
  const match = /^bytes=(\d*)-(\d*)$/.exec(value.trim());
  if (!match) return null;

  const [, rawStart, rawEnd] = match;
  if (rawStart === "" && rawEnd === "") return null;
  if (size === 0) return null;

  let start: number;
  let end: number;

  if (rawStart === "") {
    // Suffix range: the last N bytes. GeoTIFF readers use this to find
    // the trailing directory of a BigTIFF.
    const suffix = Number(rawEnd);
    if (!Number.isSafeInteger(suffix) || suffix <= 0) return null;
    start = Math.max(0, size - suffix);
    end = size - 1;
  } else {
    start = Number(rawStart);
    const requestedEnd = rawEnd === "" ? size - 1 : Number(rawEnd);
    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(requestedEnd)) {
      return null;
    }
    if (start >= size || requestedEnd < start) return null;
    end = Math.min(requestedEnd, size - 1);
  }

  return { start, end };
}

function contentTypeFor(key: string): string {
  const suffix = key.split(".").pop()?.toLowerCase() ?? "";
  return CONTENT_TYPES[suffix] ?? "application/octet-stream";
}

function plain(status: number, message: string): Response {
  return new Response(message, {
    status,
    headers: { ...CORS, "Content-Type": "text/plain; charset=utf-8" },
  });
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS });
    }

    if (url.pathname === "/health") {
      return Response.json(
        { status: "ok", service: "groundtruth-api" },
        { headers: CORS },
      );
    }

    if (url.pathname === "/api") {
      return Response.json(
        {
          service: "groundtruth-api",
          endpoints: {
            health: "/health",
            app: "/",
            index: "/data/index.json",
            artifact: "/data/<name>.tif",
            chapters: "/finding.html",
            products: "/data-app/<name>.json",
          },
          notes: "Artifacts support HTTP Range; see Accept-Ranges.",
        },
        { headers: CORS },
      );
    }

    if (request.method !== "GET" && request.method !== "HEAD") {
      return new Response("Method not allowed", {
        status: 405,
        headers: { ...CORS, Allow: "GET, HEAD, OPTIONS" },
      });
    }

    // The atlas pages and their shared stylesheet and script, served from R2.
    // Their own fetches for /data-app/<file> are handled by the alias below,
    // so page and data come from one origin with no CORS and no ?data= needed.
    // "/" is the baseline explorer. Everything else resolves by name.
    const name = url.pathname === "/" ? "bay.html" : url.pathname.slice(1);
    const bare = name.endsWith(".html") ? name.slice(0, -5) : name;
    const isPage = PAGES.includes(bare);
    const isAsset = ASSETS.includes(name);
    if (isPage || isAsset) {
      const file = isPage ? `${bare}.html` : name;
      const key = `app/${file}`;
      const head = await env.DATA.head(key);
      if (!head) return plain(404, `${file} not seeded; run seed.sh`);
      const headers = new Headers(CORS);
      head.writeHttpMetadata(headers);
      const ext = file.slice(file.lastIndexOf(".") + 1);
      headers.set("Content-Type", ASSET_TYPES[ext] ?? CONTENT_TYPES.html);
      headers.set("ETag", head.httpEtag);
      headers.set("Cache-Control", "public, max-age=60");
      headers.set("Content-Length", String(head.size));
      if (request.method === "HEAD") {
        return new Response(null, { status: 200, headers });
      }
      const body = await env.DATA.get(key);
      if (!body) return plain(404, "Not found");
      return new Response(body.body, { status: 200, headers });
    }

    let key: string;
    try {
      key = decodeURIComponent(url.pathname.slice(1));
    } catch {
      return plain(400, "Invalid path");
    }

    // Alias: /data-app/<file> -> R2 key data/app/<file>. Keeps one data base
    // (./data-app/) working whether the page is served by `make serve` or here.
    if (key.startsWith("data-app/")) {
      key = "data/app/" + key.slice("data-app/".length);
    }

    if (
      !KEY_PATTERN.test(key) ||
      key.split("/").includes("..") ||
      !ALLOWED_SUFFIX.test(key)
    ) {
      return plain(404, "Not found");
    }

    // head() before a ranged get() costs one extra metadata operation but
    // gives the true object size, which Content-Range must report even
    // when only a slice is returned.
    const metadata = await env.DATA.head(key);
    if (!metadata) return plain(404, "Not found");

    const headers = new Headers(CORS);
    metadata.writeHttpMetadata(headers);
    headers.set("Content-Type", contentTypeFor(key));
    headers.set("Accept-Ranges", "bytes");
    headers.set("ETag", metadata.httpEtag);
    headers.set(
      "Cache-Control",
      key.endsWith("index.json")
        ? "public, max-age=30"
        : "public, max-age=300",
    );

    if (request.method === "HEAD") {
      headers.set("Content-Length", String(metadata.size));
      return new Response(null, { status: 200, headers });
    }

    const rangeHeader = request.headers.get("Range");
    const range = rangeHeader ? parseRange(rangeHeader, metadata.size) : null;

    if (rangeHeader && !range) {
      headers.set("Content-Range", `bytes */${metadata.size}`);
      return new Response(null, { status: 416, headers });
    }

    const object = await env.DATA.get(
      key,
      range
        ? { range: { offset: range.start, length: range.end - range.start + 1 } }
        : undefined,
    );
    if (!object) return plain(404, "Not found");

    if (range) {
      headers.set(
        "Content-Range",
        `bytes ${range.start}-${range.end}/${metadata.size}`,
      );
      headers.set("Content-Length", String(range.end - range.start + 1));
    } else {
      headers.set("Content-Length", String(metadata.size));
    }

    return new Response(object.body, {
      status: range ? 206 : 200,
      headers,
    });
  },
} satisfies ExportedHandler<Env>;
