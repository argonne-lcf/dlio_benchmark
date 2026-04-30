#!/usr/bin/env bash
# run_fast_ci.sh — Run the Fast CI test suite locally.
#
# Mirrors .github/workflows/fast-ci.yml as closely as possible.
# Completes in ~10 minutes.  Uses the existing .venv by default.
#
# Usage:
#   ./tests/run_fast_ci.sh              # use existing .venv (fastest)
#   ./tests/run_fast_ci.sh --reinstall  # recreate .venv from scratch first
#   ./tests/run_fast_ci.sh -k test_gen  # pass extra args to pytest

set -uo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PATH="${REPO_DIR}/.venv"
REINSTALL=0
EXTRA_ARGS=()

for arg in "$@"; do
    if [[ "$arg" == "--reinstall" ]]; then
        REINSTALL=1
    else
        EXTRA_ARGS+=("$arg")
    fi
done

# ── Mirror CI environment variables exactly ───────────────────────────────────
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export DLIO_OBJECT_STORAGE_TESTS=0
export DLIO_MAX_AUTO_THREADS=2
export DFTRACER_ENABLE=1
export RDMAV_FORK_SAFE=1

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

# ── Install / reinstall if requested ─────────────────────────────────────────
if [[ $REINSTALL -eq 1 ]]; then
    echo -e "${YELLOW}[CI]${NC} Recreating venv at ${VENV_PATH} ..."
    rm -rf "${VENV_PATH}"
    python3.12 -m venv "${VENV_PATH}"
    source "${VENV_PATH}/bin/activate"
    pip install --quiet --upgrade pip
    pip install --quiet ".[test]"
elif [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
    echo -e "${YELLOW}[CI]${NC} No .venv found — creating one ..."
    python3.12 -m venv "${VENV_PATH}"
    source "${VENV_PATH}/bin/activate"
    pip install --quiet --upgrade pip
    pip install --quiet ".[test]"
else
    source "${VENV_PATH}/bin/activate"
fi

# ── Run fast CI tests ─────────────────────────────────────────────────────────
echo -e "${YELLOW}[CI]${NC} Running fast CI tests (tests/test_fast_ci.py) ..."
cd "${REPO_DIR}"

RESULTS_XML="tests/fast-ci-results-local.xml"

python -m pytest tests/test_fast_ci.py \
    --tb=short -v \
    --junitxml="${RESULTS_XML}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
    ; code=$?

# Exit 134 = SIGABRT from TF+OpenMPI process teardown after tests already
# passed — same tolerance as fast-ci.yml.
if [[ "${code}" -eq 134 ]]; then
    echo -e "${YELLOW}[CI]${NC} Exit 134 (SIGABRT teardown) — checking results XML ..."
    python3 - <<PY
import xml.etree.ElementTree as ET, sys
tree = ET.parse("${RESULTS_XML}")
suite = tree.getroot().find('testsuite') or tree.getroot()
failures = int(suite.get('failures', 0)) + int(suite.get('errors', 0))
tests    = int(suite.get('tests', 0))
print(f"Tests: {tests}  Failures/Errors: {failures}")
sys.exit(1 if failures > 0 else 0)
PY
elif [[ "${code}" -ne 0 ]]; then
    echo -e "${RED}[FAIL]${NC} pytest exited with code ${code}"
    exit "${code}"
fi

echo -e "${GREEN}[PASS]${NC} Fast CI tests complete."
