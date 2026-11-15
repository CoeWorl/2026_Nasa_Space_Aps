# GROUNDTRUTH — short aliases for the Docker workflow.
# Every target is safe to run repeatedly.
#
# Docker is the only prerequisite. uv runs inside the containers, so
# nothing here needs a local Python or a local uv install.

COMPOSE  := docker compose --profile tools
RUN      := $(COMPOSE) run --rm processor
UV_IMAGE := ghcr.io/astral-sh/uv:0.12.1-python3.12-bookworm-slim

.PHONY: help setup build info data selftest verify shell test lint lock survey inspect fetch quicklook index serve clean-interim

help:
	@echo "GROUNDTRUTH"
	@echo ""
	@echo "  make setup     create .env, data/ tree and local secrets placeholder"
	@echo "  make build     build the processor image"
	@echo "  make verify    build, run the selftest, check the files on the host"
	@echo ""
	@echo "  make info      show resolved settings and mount status"
	@echo "  make data      list what the container sees under /data"
	@echo "  make selftest  write and read back an HDF5 file and a GeoTIFF"
	@echo "  make test      run the unit tests inside the container"
	@echo "  make lint      run ruff inside the container"
	@echo "  make shell     interactive shell in the processor container"
	@echo ""
	@echo "  make survey    list real NISAR granules over the study area (no login)"
	@echo "  make inspect   open one granule by range reads, print its real HDF5 paths"
	@echo "  make fetch DATE=2026-10-04   stream that date's AOI window to data/interim"
	@echo "  make quicklook render data/interim GeoTIFFs to PNG and print dB stats"
	@echo "  make index     write data/interim/index.json for upload to R2"
	@echo "  make serve     serve the rasters and reference viewer on localhost:8080"
	@echo "  make lock      regenerate backend/uv.lock after a dependency change"
	@echo "  make clean-interim  delete scratch output in data/interim"

setup:
	@./scripts/setup.sh

build: setup
	$(COMPOSE) build processor

info: build
	$(RUN) groundtruth info

data: build
	$(RUN) groundtruth check-data

selftest: build
	$(RUN) groundtruth selftest

verify:
	@./scripts/verify.sh

# Phase 0 feasibility: what NISAR data actually exists over the AOI.
# Uses the public CMR search API, so it needs no Earthdata account.
survey: build
	$(COMPOSE) run --rm -v "$(CURDIR)/scripts:/scripts:ro" processor \
		python /scripts/survey_nisar.py

# Phase 0 exit criterion: the real variable paths, read over the network
# without downloading the granule. Needs Earthdata credentials.
inspect: build
	$(COMPOSE) run --rm -v "$(CURDIR)/scripts:/scripts:ro" processor \
		python /scripts/inspect_granule.py

# Stream one acquisition's area of interest. Nothing is downloaded whole.
#   make fetch DATE=2026-10-04 [POL=HH] [BBOX=w,s,e,n]
DATE ?=
POL  ?= HH
BBOX ?= -74.03,40.70,-73.90,40.88
# Write the artifact index as a file, for uploading to R2 alongside
# the rasters. The local server synthesises the same JSON per request.
index: build
	$(COMPOSE) run --rm -v "$(CURDIR)/scripts:/scripts:ro" processor \
		python /scripts/build_index.py

# Serve the rasters and the reference viewer for local development.
# Bound to 127.0.0.1 so it is not reachable from the network.
SERVE_PORT ?= 8080
serve: build
	@echo "viewer:  http://localhost:$(SERVE_PORT)/"
	@echo "atlas:   http://localhost:$(SERVE_PORT)/finding.html"
	@echo "legacy:  http://localhost:$(SERVE_PORT)/explore.html"
	@echo "rasters: http://localhost:$(SERVE_PORT)/data/index.json"
	$(COMPOSE) run --rm \
		-p "127.0.0.1:$(SERVE_PORT):$(SERVE_PORT)" \
		-e SERVE_PORT="$(SERVE_PORT)" \
		-v "$(CURDIR)/scripts:/scripts:ro" \
		-v "$(CURDIR)/web:/web:ro" \
		processor python /scripts/serve_data.py

# Eyeball the rasters without QGIS or a browser.
quicklook: build
	$(COMPOSE) run --rm -v "$(CURDIR)/scripts:/scripts:ro" processor \
		python /scripts/quicklook.py

fetch: build
	@test -n "$(DATE)" || { echo "usage: make fetch DATE=YYYY-MM-DD"; exit 2; }
	$(RUN) groundtruth fetch-window \
		--date "$(DATE)" --polarization "$(POL)" --bbox "$(BBOX)"

test: build
	$(RUN) pytest -q -p no:cacheprovider /app/tests

lint: build
	$(RUN) ruff check /app/src /app/tests

shell: build
	$(COMPOSE) run --rm --entrypoint bash processor

# Regenerate the lockfile in a container, as your own user so the file
# does not come back owned by root. Review the diff before committing.
lock:
	docker run --rm \
		--user "$$(id -u):$$(id -g)" \
		-e UV_CACHE_DIR=/tmp/uv-cache \
		-e HOME=/tmp \
		-v "$(CURDIR)/backend:/app" \
		-w /app \
		$(UV_IMAGE) lock
	@echo "backend/uv.lock regenerated — review 'git diff backend/uv.lock'"

clean-interim:
	find data/interim -type f ! -name .gitkeep -delete
	@echo "data/interim cleared"
