#!/usr/bin/env bash
# GROUNDTRUTH first-time setup. Idempotent: safe to run again.
# Creates .env, the data directory tree, and a local secrets placeholder.
# Never overwrites an existing .env.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

echo "GROUNDTRUTH setup in $repo_root"

# --- .env ----------------------------------------------------------
if [ -f .env ]; then
  echo "  [keep]   .env already exists"
else
  cp .env.example .env
  echo "  [create] .env from .env.example"
fi

# --- host uid/gid so container-written files stay yours on Linux ----
uid="$(id -u)"
gid="$(id -g)"
for pair in "DOCKER_UID=$uid" "DOCKER_GID=$gid"; do
  key="${pair%%=*}"
  if grep -q "^${key}=" .env; then
    # BSD and GNU sed disagree on -i; rewrite via a temp file instead.
    tmp="$(mktemp)"
    sed "s|^${key}=.*|${pair}|" .env > "$tmp" && mv "$tmp" .env
  else
    printf '%s\n' "$pair" >> .env
  fi
done
echo "  [set]    DOCKER_UID=$uid DOCKER_GID=$gid in .env"

# --- data tree ------------------------------------------------------
for dir in data/raw/nisar data/raw/nyc data/interim data/processed; do
  mkdir -p "$dir"
  touch "$dir/.gitkeep"
done
echo "  [ok]     data/ tree ready (raw/nisar, raw/nyc, interim, processed)"

# --- earthdata credential placeholder --------------------------------
mkdir -p secrets
touch secrets/.gitkeep
if [ -f secrets/netrc ]; then
  echo "  [keep]   secrets/netrc already exists"
elif [ -f "$HOME/.netrc" ] && grep -q "urs.earthdata.nasa.gov" "$HOME/.netrc" 2>/dev/null; then
  cp "$HOME/.netrc" secrets/netrc
  chmod 600 secrets/netrc
  echo "  [copy]   secrets/netrc from ~/.netrc (Earthdata entry found)"
else
  cat > secrets/netrc <<'NETRC'
machine urs.earthdata.nasa.gov
  login REPLACE_WITH_YOUR_EARTHDATA_USERNAME
  password REPLACE_WITH_YOUR_EARTHDATA_PASSWORD
NETRC
  chmod 600 secrets/netrc
  echo "  [create] secrets/netrc placeholder (edit before downloading data)"
fi

echo ""
echo "Setup complete. Next:"
echo "    make verify     # build the image and prove the data mount works"
