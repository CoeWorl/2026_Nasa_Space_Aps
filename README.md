# GROUNDTRUTH

NISAR surface-change explorer — NASA Space Apps Challenge 2026.

GROUNDTRUTH turns pairs of NISAR radar observations into georeferenced
change rasters, quality masks, and machine-readable statistics, and serves them
to an interactive map.

**Status:** real NISAR data flowing. The container streams an area of
interest straight out of a multi-gigabyte granule over HTTP range reads and
writes a georeferenced GeoTIFF plus provenance. The change algorithm, the
Cloudflare API, and the frontend come next.

---

## Quick start

The only prerequisite is **Docker** (Docker Desktop on macOS or Windows,
Docker Engine on Linux). No local Python, GDAL, HDF5, or NASA credentials are
needed for any command on this page.

```bash
cd 2026_Nasa_Space_Aps
make verify
```

`make verify` is the whole check: it creates `.env` and the `data/` tree, builds
the image, and proves the container can read and write the mounted data
directory in both directions. It ends with:

```text
================================================================
VERIFY PASSED
  The processor image builds, sees ./data at /data, and can read
  and write it. Put NISAR granules in ./data/raw/nisar/ and they
  are available immediately, with no rebuild.
================================================================
```

Anything else is a failure with the reason printed above it.

The first build downloads the base image and the pinned Python packages, so
expect a few minutes. Later builds are cached and take seconds.

---

## Everyday commands

| Command | What it does |
|---|---|
| `make setup` | Create `.env`, the `data/` tree, and the credential placeholder |
| `make build` | Build the processor image |
| `make verify` | Full acceptance check (build + mount + selftest) |
| `make info` | Resolved settings and mount status |
| `make data` | List what the container sees under `/data` |
| `make selftest` | Write and read back an HDF5 file and a GeoTIFF |
| `make test` | Run the unit tests inside the container |
| `make lint` | Run ruff inside the container |
| `make shell` | Interactive shell in the container |
| `make survey` | List real NISAR granules over the study area (no login) |
| `make inspect` | Open a granule by range reads, print its real HDF5 paths |
| `make fetch DATE=2026-10-04` | Stream that date's AOI window to `data/interim` |
| `make quicklook` | Render `data/interim` GeoTIFFs to PNG and print dB stats |
| `make serve` | Serve the rasters and reference viewer on localhost:8080 |
| `make lock` | Regenerate `backend/uv.lock` after a dependency change |

Without `make`, every target is a plain Compose command:

```bash
docker compose --profile tools build processor
docker compose --profile tools run --rm processor groundtruth info
```

The `tools` profile exists because the processor is a command-line program, not
a server: plain `docker compose up` would start a container that exits
immediately.

---

## Getting NISAR data

Granules are 5.7 GB each and we do not download them. `earthaccess.open()`
hands h5py a file object that fetches byte ranges on demand, and GCOV is
chunked 512x512, so reading a city-sized window costs a few hundred megabytes
of transfer and leaves a few megabytes on disk.

```bash
make survey                    # what exists over the AOI (no login needed)
make inspect                   # the granule's real HDF5 paths (needs login)
make fetch DATE=2026-10-04     # stream that date's window to data/interim
```

What the study area currently offers, from `make survey`:

- **GCOV**, track 069 descending, frame 068, dual-pol HH/HV
- 29 acquisitions on a 12-day repeat, Nov 2025 to Oct 2026
- `grids/frequencyA` is 10 m, `frequencyB` is 80 m, both EPSG:32618

A pair is only comparable when track, direction, frame, polarization **and
processing maturity** all match. Maturity varies *within* a single collection
(the first two dates on this track are provisional, the rest final), so it is
read per granule rather than inferred from the collection name.

Each fetch writes two files into `data/interim/`: the GeoTIFF, and a
`.provenance.json` recording the granule id, acquisition time, track, frame,
maturity, the exact variable read, the grid, and the validity rule applied.

---

## Looking at the data

Three ways, in increasing order of effort.

**PNG quicklook** needs nothing but Docker:

```bash
make quicklook
```

It writes a PNG beside each GeoTIFF and prints the decibel distribution.
Over a city, water should sit near -24 dB and built-up areas above 0 dB; if
that structure is missing, something upstream is wrong. Use it constantly -
it catches flipped axes and broken masks that statistics hide.

**QGIS or ArcGIS Pro**: drag `data/interim/*.tif` straight in. Both read COG
with no plugin. Worth doing once deliberately, to confirm the raster lands on
the real coastline rather than 5 m off it.

**The browser**:

```bash
make serve     # then open http://localhost:8080/
```

`web/index.html` is a **reference viewer, not the application**. It exists to
prove the contract the React app will be built against:

- the float32 COG loads in a browser straight from HTTP
- the dB stretch and colour ramp are right
- clicking a pixel returns a real measurement, not a colour

A browser cannot render float32 with NaN nodata natively. `geotiff.js` reads
the actual values and the page builds the image itself, so measurements
survive into the UI. Clicking issues an HTTP range request for that single
pixel at full resolution, which is why the readout is the measurement rather
than the colour of a downsampled texture. The two functions worth copying are
`readRaster()` and `inspectAt()`.

### Contract for the frontend

| Property | Value |
|---|---|
| Format | Cloud Optimized GeoTIFF, tiled 512x512, DEFLATE |
| CRS | EPSG:32618 (WGS 84 / UTM 18N) |
| Data type | float32, `NaN` = nodata |
| Quantity | linear intensity; display as `10 * log10(value)` |
| Typical display range | -13 to +12 dB (2nd to 98th percentile) |
| Served from | `/data/<name>.tif`, with `/data/index.json` listing what exists |

`make serve` supports HTTP Range and sends CORS headers, so a Vite dev server
on another port can read from it. It binds to 127.0.0.1 only, and serves just
`web/` and `data/interim/` - never the repository root, which holds `.env` and
`secrets/`.

---

## Cloudflare

`cloudflare/` is a Worker that serves analysis artifacts from R2 with correct
HTTP Range semantics. Range is the point: a Cloud Optimized GeoTIFF is only
cloud optimized if the client can fetch single tiles. Without it, inspecting
one pixel would pull eight megabytes.

The Worker never decodes or reprojects a raster. It streams bytes. The science
stays in the Python processor, which owns and tests it.

### Local, with no Cloudflare account

Wrangler simulates R2, so the whole path is testable before anyone logs in:

```bash
make index                     # write data/interim/index.json
cd cloudflare && npm install
./seed.sh                      # load artifacts into LOCAL simulated R2
npx wrangler dev               # Worker on http://localhost:8787
```

Point the reference viewer at it:

```
http://localhost:8080/?api=http://localhost:8787
```

The viewer's reading code is identical either way. That is what makes the
artifact contract, not the storage backend, the thing the frontend depends on.

### Deployed

Live at **https://groundtruth-api.groundtruth-api.workers.dev**

```bash
cd cloudflare
npx wrangler login                          # once
npx wrangler r2 bucket create groundtruth-dev
./seed.sh --remote                          # upload artifacts
npx wrangler deploy
```

R2 must be enabled on the account in the dashboard first, separately from the
API token's r2 scope, or bucket creation fails with code 10042. After a first
deploy the workers.dev certificate takes a couple of minutes; until it is
issued, TLS fails while DNS already resolves.

The raster viewer runs against it with:

```
http://localhost:8080/?api=https://groundtruth-api.groundtruth-api.workers.dev
```

The endpoint is public and unauthenticated, which suits NASA open data.

### The explore app (temporal views)

`seed.sh` also uploads the data products and the `explore.html` page, so the
Worker serves the whole temporal app from one origin:

| URL | Serves |
|---|---|
| `/explore.html` | the page, from R2 key `app/explore.html` |
| `/data-app/<name>.json` | a data product, aliased to R2 key `data/app/<name>` |

The products come from `tide_adjust.py`; regenerate and reseed with:

```bash
uv run --project backend python -u backend/scripts/tide_adjust.py --out-dir data/processed/jamaica_bay/app
cd cloudflare && ./seed.sh --remote        # or ./seed.sh for local simulated R2
npx wrangler deploy                        # or npx wrangler dev
```

Then open `https://groundtruth-api.groundtruth-api.workers.dev/explore.html`.
The page defaults to a same-origin `./data-app/` base, so no query string is
needed; pass `?data=<url>` only to point it at a different host. Locally,
`make serve` serves the same page at `http://localhost:8080/explore.html`.

### Verified behaviour

| Check | Result |
|---|---|
| `HEAD` | 200, `Content-Length`, `Accept-Ranges: bytes`, `ETag` |
| `Range: bytes=0-1023` | 206, `Content-Range: bytes 0-1023/8312627` |
| `Range: bytes=-512` | 206, correct suffix window |
| Unsatisfiable range | 416 with `Content-Range: bytes */<size>` |
| `data/../secrets/netrc` | 404 |
| Cross-origin from the viewer | CORS headers present, 206 responses |
| Same pixel: local file, local R2, deployed R2 | `-5.92 dB`, linear `0.2559852600097656`, pixel 597,1552 |
| `/explore.html` | 200 `text/html` from R2 |
| `/data-app/models.json` | 200 `application/json`, descending slope `-0.914 dB/m` |
| `/data-app/regions.geojson` | 200 `application/geo+json` |
| `Range: bytes=0-99` on a product | 206 |
| `/data-app/../../secrets/netrc`, missing key | 404 |

### Not built yet

Queues, a Durable Object coordinator, and `POST /api/analyses` come after the
change kernel exists. A Queue consumer is still a Worker with Worker CPU
limits; it does not become a Python/GDAL environment by being asynchronous.

---

## Where data goes

```text
data/
├── raw/
│   ├── nisar/      # NISAR granules (HDF5) — put downloads here
│   └── nyc/        # ancillary GIS layers for the study area
├── interim/        # scratch; safe to delete at any time
└── processed/      # analysis artifacts: change.tif, statistics.json, ...
```

This directory is **mounted** into the container at `/data`, never copied into
the image. A file dropped into `data/raw/nisar/` on your machine is visible to
the container immediately — no rebuild. The directory layout is tracked in git
via `.gitkeep` files; the data files themselves are ignored.

Inside the container the same tree is at `/data`, and `configs/` is mounted
read-only at `/configs`.

---

## Configuration

Two kinds of configuration, kept apart on purpose:

- **`.env`** — environment and deployment values: paths, ports, log level,
  credentials. Created from `.env.example` by `make setup`. Never committed.
  It holds no R2 keys: artifacts are published by POSTing them to the
  Cloudflare Worker, which writes to R2 through its native binding, so the
  Python container never handles bucket credentials.
- **`configs/*.yaml`** — scientific parameters: study area, polarization,
  thresholds, output format. Committed, because an analysis must be
  reproducible from its config file and input manifest rather than from
  someone's shell.

`.env.example` documents every variable. Edit `.env` freely; it is yours.

### NASA Earthdata credentials

Only needed to **download** granules. Everything above works without them.

`make setup` creates `secrets/netrc` — a placeholder, unless you already have an
Earthdata entry in `~/.netrc`, in which case it is copied. Replace the
placeholder values:

```text
machine urs.earthdata.nasa.gov
  login YOUR_EARTHDATA_USERNAME
  password YOUR_EARTHDATA_PASSWORD
```

The file is mounted read-only into the container as a Compose secret at
`/run/secrets/earthdata_netrc`. It is gitignored and never enters the image.
Register for an account at <https://urs.earthdata.nasa.gov/users/new>.

---

## Reproducibility

- **Pinned dependencies.** `backend/uv.lock` pins the exact resolved version
  of every package, for every platform, and the image installs with
  `uv sync --locked` — which *fails the build* if the lock and
  `pyproject.toml` have drifted apart. Unlike a `pip freeze` dump, this lock
  is cross-platform, so a teammate on arm64 and a judge on amd64 get the same
  versions.
- **Pinned tooling.** uv itself is copied into the image at a fixed version
  from its official image, so there is no `curl | sh` install step that could
  change between builds.
- **Pinned base image.** `python:3.12-slim-bookworm`, with the Debian release
  in the tag so the system package set does not drift.
- **No host state.** The build does not read anything outside the repository.
- **Both architectures.** Builds on arm64 (Apple Silicon) and amd64 from
  manylinux wheels.
- **Verifiable.** `make verify` checks the claims on this page rather than
  asserting them.

To regenerate the lock after changing a dependency in
`backend/pyproject.toml`:

```bash
make lock
```

That runs uv in a container, so it works without a local uv install. Review
`git diff backend/uv.lock`, then commit it together with the change that
caused it.

### Optional: working without Docker

If you would rather run the code directly, the project is a standard uv
project:

```bash
cd backend
uv sync --all-extras       # creates .venv from uv.lock
uv run pytest
uv run ruff check .
uv run groundtruth info    # expects DATA_ROOT to point at ../data
```

Docker remains the reference environment: it is what `make verify` checks and
what the cloud worker will run.

---

## Repository layout

```text
.
├── GROUNDTRUTH.md      # full technical specification
├── DOCKER_PLAN.md      # plan for the current infrastructure phase
├── compose.yaml        # local orchestration and volume mounts
├── Makefile            # command aliases
├── .env.example        # documented environment variables
├── backend/
│   ├── Dockerfile
│   ├── pyproject.toml      # dependencies, unpinned by design
│   ├── uv.lock             # exact resolved versions, committed
│   ├── src/groundtruth/
│   │   ├── cli.py      # info / check-data / selftest
│   │   └── config.py   # pydantic-settings
│   └── tests/unit/
├── configs/nyc.yaml    # scientific parameters
├── data/               # mounted at /data; contents gitignored
├── scripts/
│   ├── setup.sh        # idempotent first-time setup
│   └── verify.sh       # acceptance check
└── secrets/            # local credential files; gitignored
```

---

## Troubleshooting

**`Cannot connect to the Docker daemon`** — Docker Desktop is not running.

**`permission denied` writing to `data/`** (Linux) — `make setup` writes your
uid and gid into `.env` as `DOCKER_UID` / `DOCKER_GID` so the container writes
as you. Run `make setup` again if you edited `.env` by hand.

**`groundtruth info` reports `MISSING`** — you ran the container with bare
`docker run` instead of through Compose, so the volumes were never mounted.

**The build is slow the first time** — it downloads the base image and the
scientific stack. It is cached afterwards; `docker system prune` undoes that.

**The build fails with `the lockfile is not up-to-date`** — someone changed
`pyproject.toml` without regenerating the lock. Run `make lock` and commit the
result. This failure is deliberate: a build must never quietly resolve
different versions than the lock records.

## License

MIT — see [LICENSE](LICENSE).
