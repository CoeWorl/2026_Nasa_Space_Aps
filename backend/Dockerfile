# GROUNDTRUTH scientific processor.
#
# Debian slim, not Alpine: the geospatial wheels (rasterio, pyproj,
# shapely, h5py) are manylinux builds and musl makes them a problem.
#
# Dependencies are installed by uv from the committed uv.lock, so this
# image contains the same package versions on every machine.
#
# The image contains code and dependencies only. NISAR data is mounted
# at /data by compose.yaml and never enters the build context.

FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.title="groundtruth-processor" \
      org.opencontainers.image.description="NISAR surface-change analysis" \
      org.opencontainers.image.source="https://github.com/JZT3/2026_Nasa_Space_Aps" \
      org.opencontainers.image.licenses="MIT"

# Pinned uv, copied from its official image: no curl | sh, no network
# install step that could drift between builds.
COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /uvx /usr/local/bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Keep the venv out of /app so a bind mount over the source can
    # never shadow it.
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    # Use the interpreter already in this image; do not fetch another.
    UV_PYTHON_DOWNLOADS=never \
    PATH="/opt/venv/bin:$PATH"

# Command-line inspection tools for Phase 0 feasibility work:
#   h5ls / h5dump  - discover NISAR HDF5 variable paths
#   gdalinfo       - check CRS, geotransform and subdatasets
#
# Deliberately NOT installing libgdal-dev / libproj-dev / libgeos-dev:
# nothing here compiles against them (the wheels bundle their own
# libraries), they add about a gigabyte, and mixing system GDAL with
# wheel GDAL is a known source of version mismatches.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gdal-bin \
        hdf5-tools \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create the unprivileged user and its directories BEFORE installing
# anything into them. A `chown -R` after the install would rewrite every
# file in the venv into a new layer and add roughly half a gigabyte to
# the image for no benefit.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app /opt/venv /data /configs \
    && chown appuser:appuser /app /opt/venv /data /configs

USER appuser
WORKDIR /app

# Leave empty for the development image, which includes pytest and ruff
# so anyone can run the tests without a local Python.
# A lean production image: --build-arg UV_SYNC_FLAGS=--no-dev
ARG UV_SYNC_FLAGS=""

# Dependencies first, without the project itself, so editing source code
# does not invalidate this layer. --locked fails the build if uv.lock and
# pyproject.toml have drifted apart.
COPY --chown=appuser:appuser pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project ${UV_SYNC_FLAGS}

# Then the project. Tests ship in the image so they can be run without
# a local Python environment.
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser tests/ ./tests/
RUN uv sync --locked ${UV_SYNC_FLAGS}

# compose.yaml runs the container as the host uid, which does not own
# /app, so tool caches must go somewhere writable.
ENV RUFF_CACHE_DIR=/tmp/ruff-cache \
    PYTHONPYCACHEPREFIX=/tmp/pycache

# The console script from /opt/venv/bin, already on PATH. Not `uv run`:
# that would try to re-sync the environment at container start.
CMD ["groundtruth", "--help"]
