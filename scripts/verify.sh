#!/usr/bin/env bash
# GROUNDTRUTH acceptance check for the Docker + data setup.
#
# Builds the image, runs the selftest inside the container, and then
# verifies from the HOST that the files the container wrote really
# appeared. Prints PASS or FAIL and exits accordingly.
#
# Docker is the only prerequisite. No local Python, uv, or NASA
# credentials are needed.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

compose="docker compose --profile tools"

fail() { echo ""; echo "VERIFY FAILED: $*" >&2; exit 1; }

echo "=== 0. setup ==================================================="
./scripts/setup.sh

echo ""
echo "=== 1. docker available ========================================"
command -v docker >/dev/null 2>&1 || fail "docker is not installed"
docker version --format '  client {{.Client.Version}}, server {{.Server.Version}}' \
  || fail "docker daemon is not running"

echo ""
echo "=== 2. build the processor image ==============================="
echo "  (uv sync --locked fails the build if uv.lock and pyproject.toml disagree,"
echo "   so a successful build is also a lockfile integrity check)"
$compose build processor || fail "image build failed"

echo ""
echo "=== 3. settings and mounts ====================================="
$compose run --rm processor groundtruth info || fail "groundtruth info failed"

echo ""
echo "=== 4. host file visible inside the container =================="
marker="data/raw/nisar/_verify_marker.txt"
echo "written on the host at $(date -u +%FT%TZ)" > "$marker"
# Capture the output rather than piping into `grep -q`: grep exits at the
# first match, the writer gets SIGPIPE, and `pipefail` would report a
# mount failure that never happened.
listing="$($compose run --rm processor groundtruth check-data 2>&1)" || {
  find "$marker" -delete 2>/dev/null || true
  fail "groundtruth check-data failed"
}
find "$marker" -delete 2>/dev/null || true
case "$listing" in
  *_verify_marker.txt*) echo "  [OK] host -> container" ;;
  *) fail "a file created on the host was not visible in the container" ;;
esac

echo ""
echo "=== 5. selftest (container writes to the mount) ================"
find data/interim -maxdepth 1 -type f -name 'selftest.*' -delete 2>/dev/null || true
$compose run --rm processor groundtruth selftest || fail "selftest failed"

echo ""
echo "=== 6. container output visible on the host ===================="
for f in data/interim/selftest.h5 data/interim/selftest.tif; do
  [ -s "$f" ] || fail "$f was not written to the host (bind mount is not working)"
  echo "  [OK] $f ($(wc -c < "$f" | tr -d ' ') bytes)"
done
echo "  [OK] container -> host"

echo ""
echo "=== 7. unit tests inside the container ========================="
$compose run --rm processor pytest -q -p no:cacheprovider /app/tests \
  || fail "unit tests failed inside the container"

echo ""
echo "=== 8. data is not baked into the image ========================"
size_bytes="$(docker image inspect groundtruth-processor:dev --format '{{.Size}}')"
echo "  image size: $(( size_bytes / 1024 / 1024 )) MiB"
contents="$(docker run --rm --entrypoint sh groundtruth-processor:dev \
  -c 'ls -A /data 2>/dev/null | wc -l')"
[ "$(echo "$contents" | tr -d '[:space:]')" = "0" ] \
  || fail "the image contains data at /data; it must only be mounted"
echo "  [OK] /data is empty in the image; data comes from the mount"

echo ""
echo "================================================================"
echo "VERIFY PASSED"
echo "  The processor image builds, sees ./data at /data, and can read"
echo "  and write it. Put NISAR granules in ./data/raw/nisar/ and they"
echo "  are available immediately, with no rebuild."
echo "================================================================"
