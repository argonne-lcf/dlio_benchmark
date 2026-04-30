#!/usr/bin/env bash
# run_integration_tests.sh — Run the full DLIO integration test suite locally.
#
# Clones the current working tree (including uncommitted changes) into a
# fresh temp directory, creates a venv, installs DLIO, then runs every
# non-S3 integration test step in sequence.
#
# Usage:
#   ./run_integration_tests.sh              # run all steps, venv = via-setup
#   ./run_integration_tests.sh via-reqs     # use requirements-test.txt install
#   ./run_integration_tests.sh -k test_gen  # pass extra args to every pytest call
#   SKIP_INSTALL=1 ./run_integration_tests.sh   # reuse existing workdir venv

set -uo pipefail

# ── configuration ─────────────────────────────────────────────────────────────
VENV_MODE="${1:-via-setup}"   # via-setup | via-reqs
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORK_DIR="/tmp/dlio-integration-tests"
VENV_PATH="${WORK_DIR}/.venv"
PYTHON="python3.12"

# Mirror CI environment variables exactly
export CC=gcc-10
export CXX=g++-10
export DFTRACER_BUILD_TYPE=Debug
export DFTRACER_ENABLE=1
export DFTRACER_LOG_LEVEL=INFO
export DLIO_MAX_AUTO_THREADS=2
export DLIO_OBJECT_STORAGE_TESTS=0
export GOTCHA_DEBUG=1
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export RDMAV_FORK_SAFE=1
export PYTHONPATH="${WORK_DIR}:${PYTHONPATH:-}"

if [[ "${VENV_MODE}" == "via-setup" ]]; then
    export DLIO_EXEC="dlio_benchmark"
else
    export DLIO_EXEC="${VENV_PATH}/bin/python3 -m dlio_benchmark.main"
fi

# ── helpers ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
PASS_COUNT=0; FAIL_COUNT=0; SKIP_COUNT=0
declare -a FAILURES=()

log()  { echo -e "${YELLOW}[CI]${NC} $*"; }
ok()   { echo -e "${GREEN}[PASS]${NC} $*"; ((PASS_COUNT++)) || true; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; ((FAIL_COUNT++)) || true; FAILURES+=("$*"); }

run_step() {
    local name="$1"; shift
    log "── $name ──"
    if bash -c "$*"; then
        ok "$name"
    else
        fail "$name"
        # Continue — don't abort the whole run on one failure
    fi
}

# ── Step 0: sync working tree to WORK_DIR ────────────────────────────────────
log "Syncing working tree → ${WORK_DIR}"
mkdir -p "${WORK_DIR}"

# rsync the repo (respects .gitignore via --exclude-from, but simpler to
# exclude the big stuff explicitly).  We always sync so uncommitted edits
# are included.
rsync -a --delete \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.venv/' \
    --exclude='target/' \
    --exclude='outputs/' \
    --exclude='data/' \
    --exclude='checkpoints/' \
    --exclude='output/' \
    --exclude='hydra_log/' \
    --exclude='*.egg-info/' \
    --exclude='dlio_test_output/' \
    "${REPO_DIR}/" "${WORK_DIR}/"

log "Sync complete.  WORK_DIR=${WORK_DIR}"

cd "${WORK_DIR}"

# ── Step 1: venv + install ────────────────────────────────────────────────────
if [[ "${SKIP_INSTALL:-0}" == "1" && -d "${VENV_PATH}" ]]; then
    log "Reusing existing venv at ${VENV_PATH} (SKIP_INSTALL=1)"
else
    log "Creating venv: ${VENV_PATH} (mode=${VENV_MODE})"
    rm -rf "${VENV_PATH}"
    ${PYTHON} -m venv "${VENV_PATH}"
    source "${VENV_PATH}/bin/activate"
    pip install --upgrade pip -q

    if [[ "${VENV_MODE}" == "via-setup" ]]; then
        log "Installing via pip install .[test]"
        pip install ".[test]"
    else
        log "Installing via requirements-test.txt"
        pip install -r requirements-test.txt
        # via-reqs: package not installed, so PYTHONPATH must be set (done above)
    fi
fi

source "${VENV_PATH}/bin/activate"

MPI="mpirun -np 2"

# ── Step 2: Preflight ─────────────────────────────────────────────────────────
run_step "Preflight runtime imports" python3 - <<'PY'
import importlib, sys
required = ["dftracer.python", "dftracer.dftracer"]
optional = ["dgen_py"]
failures = []
for mod in required:
    try: importlib.import_module(mod)
    except Exception as exc: failures.append(f"{mod}: {exc}")
optional_failures = []
for mod in optional:
    try: importlib.import_module(mod)
    except Exception as exc: optional_failures.append(f"{mod}: {exc}")
if failures:
    print("Preflight FAILED:")
    for f in failures: print(f"  {f}")
    sys.exit(1)
if optional_failures:
    for f in optional_failures: print(f"WARNING: {f}")
print("Preflight import check passed")
PY

# ── Step 3: test_dataset_dimension_gen_data ───────────────────────────────────
run_step "test_dataset_dimension_gen_data" \
    "pytest tests/dlio_dataset_dimension_test.py -n 4 -v && rm -rf outputs"

# ── Step 4: test_checkpoint_epoch ────────────────────────────────────────────
run_step "test_checkpoint_epoch" bash <<'STEP'
set -e
for k in \
    "test_checkpoint_epoch[tensorflow-1024-optimizers0-2-layer_params0-0-True]" \
    "test_checkpoint_epoch[pytorch-1024-optimizers1-2-layer_params1-0-True]" \
    "test_checkpoint_epoch[tensorflow-1024-optimizers2-2-layer_params2-3-True]" \
    "test_checkpoint_epoch[pytorch-1024-optimizers3-2-layer_params3-3-True]" \
    "test_checkpoint_epoch[tensorflow-1024-optimizers4-1-layer_params4-0-True]" \
    "test_checkpoint_epoch[pytorch-1024-optimizers5-1-layer_params5-0-True]" \
    "test_checkpoint_epoch[tensorflow-1024-optimizers6-2-layer_params6-0-False]" \
    "test_checkpoint_epoch[pytorch-1024-optimizers7-2-layer_params7-0-False]" \
    "test_checkpoint_epoch[tensorflow-1024-optimizers8-2-layer_params8-3-False]" \
    "test_checkpoint_epoch[pytorch-1024-optimizers9-2-layer_params9-3-False]" \
    "test_checkpoint_epoch[tensorflow-1024-optimizers10-1-layer_params10-0-False]" \
    "test_checkpoint_epoch[pytorch-1024-optimizers11-1-layer_params11-0-False]"; do
    mpirun -np 2 pytest -k "${k}" -v
done
rm -rf data
STEP

# ── Step 5: test_checkpoint_ksm_config ───────────────────────────────────────
run_step "test_checkpoint_ksm_config" \
    "mpirun -np 2 pytest -k test_checkpoint_ksm_config -v && rm -rf data"

# ── Step 6: test_checkpoint_step ─────────────────────────────────────────────
run_step "test_checkpoint_step" \
    "mpirun -np 2 pytest -k test_checkpoint_step -v"

# ── Step 7: test_gen_data ─────────────────────────────────────────────────────
run_step "test_gen_data" bash <<'STEP'
set -e
for k in png-tensorflow npz-tensorflow jpeg-tensorflow tfrecord-tensorflow hdf5-tensorflow indexed_binary-tensorflow mmap_indexed_binary-tensorflow; do
    mpirun -np 2 pytest -k "test_gen_data[${k}]" -v
done
rm -rf data
STEP

# ── Step 8: test_custom_storage_root_gen_data ─────────────────────────────────
run_step "test_custom_storage_root_gen_data" bash <<'STEP'
set -e
for k in png-tensorflow npz-tensorflow jpeg-tensorflow tfrecord-tensorflow hdf5-tensorflow indexed_binary-tensorflow mmap_indexed_binary-tensorflow; do
    mpirun -np 2 pytest -k "test_storage_root_gen_data[${k}]" -v
done
rm -rf data
STEP

# ── Step 9: test_train (True variants) ───────────────────────────────────────
run_step "test_train (True variants)" bash <<'STEP'
set -e
for k in \
    png-tensorflow-tensorflow-True npz-tensorflow-tensorflow-True jpeg-tensorflow-tensorflow-True \
    tfrecord-tensorflow-tensorflow-True hdf5-tensorflow-tensorflow-True csv-tensorflow-tensorflow-True \
    png-pytorch-pytorch-True npz-pytorch-pytorch-True jpeg-pytorch-pytorch-True \
    hdf5-pytorch-pytorch-True csv-pytorch-pytorch-True \
    png-tensorflow-dali-True npz-tensorflow-dali-True jpeg-tensorflow-dali-True \
    hdf5-tensorflow-dali-True csv-tensorflow-dali-True \
    png-pytorch-dali-True npz-pytorch-dali-True jpeg-pytorch-dali-True \
    hdf5-pytorch-dali-True csv-pytorch-dali-True \
    indexed_binary-tensorflow-tensorflow-True indexed_binary-pytorch-pytorch-True \
    indexed_binary-tensorflow-dali-True indexed_binary-pytorch-dali-True \
    mmap_indexed_binary-tensorflow-tensorflow-True mmap_indexed_binary-pytorch-pytorch-True \
    mmap_indexed_binary-tensorflow-dali-True mmap_indexed_binary-pytorch-dali-True; do
    mpirun -np 2 pytest -k "test_train[${k}]" -v
done
STEP

# ── Step 10: test_train (False variants) ─────────────────────────────────────
run_step "test_train (False variants)" bash <<'STEP'
set -e
for k in \
    png-tensorflow-tensorflow-False npz-tensorflow-tensorflow-False jpeg-tensorflow-tensorflow-False \
    tfrecord-tensorflow-tensorflow-False hdf5-tensorflow-tensorflow-False csv-tensorflow-tensorflow-False \
    png-pytorch-pytorch-False npz-pytorch-pytorch-False jpeg-pytorch-pytorch-False \
    hdf5-pytorch-pytorch-False csv-pytorch-pytorch-False \
    png-tensorflow-dali-False npz-tensorflow-dali-False jpeg-tensorflow-dali-False \
    hdf5-tensorflow-dali-False csv-tensorflow-dali-False \
    png-pytorch-dali-False npz-pytorch-dali-False jpeg-pytorch-dali-False \
    hdf5-pytorch-dali-False csv-pytorch-dali-False \
    indexed_binary-tensorflow-tensorflow-False indexed_binary-pytorch-pytorch-False \
    indexed_binary-tensorflow-dali-False indexed_binary-pytorch-dali-False \
    mmap_indexed_binary-tensorflow-tensorflow-False mmap_indexed_binary-pytorch-pytorch-False \
    mmap_indexed_binary-tensorflow-dali-False mmap_indexed_binary-pytorch-dali-False; do
    mpirun -np 2 pytest -k "test_train[${k}]" -v
done
rm -rf data
STEP

# ── Step 11: test_custom_storage_root_train ───────────────────────────────────
run_step "test_custom_storage_root_train" bash <<'STEP'
set -e
for k in \
    png-tensorflow npz-tensorflow jpeg-tensorflow tfrecord-tensorflow hdf5-tensorflow csv-tensorflow \
    png-pytorch npz-pytorch jpeg-pytorch hdf5-pytorch csv-pytorch \
    indexed_binary-tensorflow indexed_binary-pytorch \
    mmap_indexed_binary-tensorflow mmap_indexed_binary-pytorch; do
    mpirun -np 2 pytest -k "test_custom_storage_root_train[${k}]" -v
done
rm -rf data
STEP

# ── Step 12: test_eval ────────────────────────────────────────────────────────
run_step "test_eval" "mpirun -np 2 pytest -k test_eval -v"

# ── Step 13: test_multi_threads ───────────────────────────────────────────────
run_step "test_multi_threads" bash <<'STEP'
set -e
for k in tensorflow-0 tensorflow-1 tensorflow-2 pytorch-0 pytorch-1 pytorch-2; do
    mpirun -np 2 pytest -k "test_multi_threads[${k}]" -v
done
rm -rf data
STEP

# ── Step 14: test-pytorch-multiprocessing-context ────────────────────────────
run_step "test-pytorch-multiprocessing-context" bash <<'STEP'
set -e
for k in "0-None" "1-fork" "2-forkserver" "2-spawn"; do
    mpirun -np 2 pytest -k "test_pytorch_multiprocessing_context[${k}]" -v
done
rm -rf data
STEP

# ── Step 15: test_subset ──────────────────────────────────────────────────────
run_step "test_subset" bash <<'STEP'
rm -rf output data checkpoints
mpirun -np 2 pytest -k test_subset -v
rm -rf data
STEP

# ── Step 16: test-tf-loader-tfrecord ─────────────────────────────────────────
run_step "test-tf-loader-tfrecord" bash <<'STEP'
set -e
rm -rf output data checkpoints
mpirun -np 2 ${DLIO_EXEC} workload=resnet50_tf \
    ++workload.dataset.num_files_train=4 ++workload.dataset.num_samples_per_file=16 \
    ++workload.workflow.train=False ++workload.workflow.generate_data=True
mpirun -np 2 ${DLIO_EXEC} workload=resnet50_tf \
    ++workload.dataset.num_files_train=4 ++workload.dataset.num_samples_per_file=16 \
    ++workload.workflow.train=True ++workload.workflow.generate_data=False \
    ++workload.train.computation_time=0.01 ++workload.train.epochs=1
rm -rf data
STEP

# ── Step 17: test-torch-loader-npz ───────────────────────────────────────────
run_step "test-torch-loader-npz" bash <<'STEP'
set -e
rm -rf output data checkpoints
mpirun -np 2 ${DLIO_EXEC} workload=unet3d_a100 \
    ++workload.train.computation_time=0.05 ++workload.evaluation.eval_time=0.01 \
    ++workload.workflow.train=False ++workload.workflow.generate_data=True \
    ++workload.dataset.num_files_train=8 ++workload.dataset.num_files_eval=8 \
    ++workload.reader.read_threads=2 \
    ++workload.dataset.record_length=4096 ++workload.dataset.record_length_stdev=0
mpirun -np 2 ${DLIO_EXEC} workload=unet3d_a100 \
    ++workload.train.computation_time=0.05 ++workload.evaluation.eval_time=0.01 \
    ++workload.train.epochs=1 ++workload.workflow.train=True ++workload.workflow.generate_data=False \
    ++workload.dataset.num_files_train=8 ++workload.dataset.num_files_eval=8 \
    ++workload.reader.read_threads=0 \
    ++workload.dataset.record_length=4096 ++workload.dataset.record_length_stdev=0
mpirun -np 2 ${DLIO_EXEC} workload=unet3d_a100 \
    ++workload.train.computation_time=0.05 ++workload.evaluation.eval_time=0.01 \
    ++workload.train.epochs=1 ++workload.workflow.train=True ++workload.workflow.generate_data=False \
    ++workload.dataset.num_files_train=8 ++workload.dataset.num_files_eval=8 \
    ++workload.reader.read_threads=0 \
    ++workload.dataset.record_length=4096 ++workload.dataset.record_length_stdev=0 \
    ++workload.reader.odirect=True
rm -rf data
STEP

# ── Step 18: test-tf-loader-npz ──────────────────────────────────────────────
run_step "test-tf-loader-npz" bash <<'STEP'
set -e
rm -rf output data checkpoints
mpirun -np 2 ${DLIO_EXEC} workload=unet3d_a100 \
    ++workload.framework=tensorflow ++workload.data_reader.data_loader=tensorflow \
    ++workload.train.computation_time=0.05 ++workload.evaluation.eval_time=0.01 \
    ++workload.train.epochs=2 ++workload.workflow.train=False ++workload.workflow.generate_data=True \
    ++workload.dataset.num_files_train=16 ++workload.dataset.num_files_eval=16 \
    ++workload.reader.read_threads=2 \
    ++workload.dataset.record_length=4096 ++workload.dataset.record_length_stdev=0
mpirun -np 2 ${DLIO_EXEC} workload=unet3d_a100 \
    ++workload.framework=tensorflow ++workload.data_reader.data_loader=tensorflow \
    ++workload.train.computation_time=0.05 ++workload.evaluation.eval_time=0.01 \
    ++workload.train.epochs=2 ++workload.workflow.train=True ++workload.workflow.generate_data=False \
    ++workload.dataset.num_files_train=16 ++workload.dataset.num_files_eval=16 \
    ++workload.reader.read_threads=2 \
    ++workload.dataset.record_length=4096 ++workload.dataset.record_length_stdev=0
rm -rf data
STEP

# ── Step 19: test_unet3d ──────────────────────────────────────────────────────
run_step "test_unet3d" bash <<'STEP'
set -e
rm -rf output data checkpoints
mpirun -np 2 ${DLIO_EXEC} workload=unet3d_a100 ++workload.workflow.generate_data=True ++workload.dataset.num_files_train=42
mpirun -np 2 ${DLIO_EXEC} workload=unet3d_h100 ++workload.workflow.generate_data=True ++workload.dataset.num_files_train=42
mpirun -np 2 ${DLIO_EXEC} workload=unet3d_h100 ++workload.workflow.generate_data=True ++workload.dataset.num_files_train=42 ++workload.dataset.format=synthetic
rm -rf data
STEP

# ── Step 20: test_resnet50 ────────────────────────────────────────────────────
run_step "test_resnet50" bash <<'STEP'
set -e
rm -rf output data checkpoints
mpirun -np 2 ${DLIO_EXEC} workload=resnet50_a100 ++workload.workflow.generate_data=True ++workload.dataset.num_files_train=8 ++workload.reader.read_threads=1
mpirun -np 2 ${DLIO_EXEC} workload=resnet50_h100 ++workload.workflow.generate_data=True ++workload.dataset.num_files_train=8 ++workload.reader.read_threads=1
mpirun -np 2 ${DLIO_EXEC} workload=resnet50_h100 ++workload.workflow.generate_data=True ++workload.dataset.num_files_train=8 ++workload.reader.read_threads=1 ++workload.dataset.format=synthetic
rm -rf data
STEP

# ── Step 21: test_cosmoflow ───────────────────────────────────────────────────
run_step "test_cosmoflow" bash <<'STEP'
set -e
rm -rf output data checkpoints
mpirun -np 2 ${DLIO_EXEC} workload=cosmoflow_a100 ++workload.workflow.generate_data=True ++workload.dataset.num_files_train=16
mpirun -np 2 ${DLIO_EXEC} workload=cosmoflow_h100 ++workload.workflow.generate_data=True ++workload.dataset.num_files_train=16
mpirun -np 2 ${DLIO_EXEC} workload=cosmoflow_h100 ++workload.workflow.generate_data=True ++workload.dataset.num_files_train=16 ++workload.dataset.format=synthetic
rm -rf data
STEP

# ── Step 22: test_computation_time_distribution ──────────────────────────────
run_step "test_computation_time_distribution" bash <<'STEP'
rm -rf output data checkpoints
mpirun -np 2 pytest -k test_computation_time_distribution -v
rm -rf data
STEP

# ── Step 23: test_llama_8b ────────────────────────────────────────────────────
run_step "test_llama_8b" bash <<'STEP'
rm -rf output data checkpoints
mpirun -np 2 ${DLIO_EXEC} workload=llama_8b_zero3 ++workload.model.parallelism.data=1024 ++workload.checkpoint.mode=subset
STEP

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo -e "  ${GREEN}PASSED${NC}: ${PASS_COUNT}   ${RED}FAILED${NC}: ${FAIL_COUNT}"
echo "════════════════════════════════════════════════════════"
if [[ ${#FAILURES[@]} -gt 0 ]]; then
    echo -e "${RED}Failed steps:${NC}"
    for f in "${FAILURES[@]}"; do echo "  - $f"; done
    exit 1
fi
echo -e "${GREEN}All steps passed.${NC}"
