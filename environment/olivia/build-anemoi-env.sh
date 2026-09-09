#!/usr/bin/env bash
#
# Build (or rebuild) the `nese-anemoi` GPU environment on Olivia and install a
# Lmod modulefile for it. Repeatable: safe to re-run.
#
#   # from an aarch64 `accel` node (no GPU needed for the build):
#   salloc --nodes=1 --cpus-per-task=64 --mem=200G --gpus-per-node=0 \
#          --qos=devel --time=01:00:00 --partition=accel --account=nn2993k
#   bash environment/olivia/build-anemoi-env.sh
#
# Unlike the CPU `nese` env (Tykky / conda-containerize), this is a custom
# Apptainer image: conda-forge has no linux-aarch64 CUDA PyTorch, and NRIS says to
# use the NGC PyTorch container. We extend that container (arm64) with the anemoi
# suite + editable anemoi-core / nese-inference / nese-training. See
# environment/olivia/README.md.
#
set -euo pipefail

# --- config (override via environment) --------------------------------------
PREFIX="${NESE_ANEMOI_ENV_PREFIX:-/cluster/projects/nn2993k/timith/sw/nese-anemoi}"
SIF="${NESE_ANEMOI_SIF:-$PREFIX/nese-anemoi.sif}"
BASE_SIF="${NESE_ANEMOI_BASE_SIF:-/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif}"
NGC_URL="${NESE_ANEMOI_NGC_URL:-docker://nvcr.io/nvidia/pytorch:25.06-py3}"
MODULES_ROOT="${NESE_ANEMOI_MODULES_ROOT:-/cluster/projects/nn2993k/timith/sw/modules}"
ANEMOI_CORE_DIR="${NESE_ANEMOI_CORE_DIR:-/cluster/projects/nn2993k/timith/anemoi-core}"
NESE_INFERENCE_DIR="${NESE_INFERENCE_DIR:-/cluster/projects/nn2993k/timith/nese-inference}"
NESE_TRAINING_DIR="${NESE_TRAINING_DIR:-/cluster/projects/nn2993k/timith/nese/training}"
HTTP_PROXY_URL="${NESE_ANEMOI_HTTP_PROXY:-http://10.63.2.48:3128/}"
APPTAINER_CACHEDIR="${NESE_ANEMOI_APPTAINER_CACHEDIR:-/cluster/projects/nn2993k/timith/sw/.apptainer-cache}"
export APPTAINER_CACHEDIR

ANEMOI_CORE_BRANCH="feature/ordering-kwarg-schema"
ANEMOI_CORE_COMMIT="af652540f"   # nest_ordering HEALPix schema kwarg; used by training.yaml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEF_IN="$SCRIPT_DIR/nese-anemoi.def.in"
REQ="$SCRIPT_DIR/requirements-anemoi.txt"
MODFILE_IN="$SCRIPT_DIR/modulefile-anemoi.lua.in"

echo ">> prefix         : $PREFIX"
echo ">> image          : $SIF"
echo ">> base image     : $BASE_SIF"
echo ">> anemoi-core    : $ANEMOI_CORE_DIR"
echo ">> nese-inference : $NESE_INFERENCE_DIR"
echo ">> nese-training  : $NESE_TRAINING_DIR"
echo ">> modules root   : $MODULES_ROOT"

# --- 1. guards -------------------------------------------------------------
if [ "$(uname -m)" != "aarch64" ]; then
    cat >&2 <<EOF
!! $(uname -m) host -- the image is aarch64 and its %post runs arm64 binaries.
!! Run this from an Olivia 'accel' node, e.g.:

   salloc --nodes=1 --cpus-per-task=64 --mem=200G --gpus-per-node=0 \\
          --qos=devel --time=01:00:00 --partition=accel --account=nn2993k
EOF
    exit 1
fi
[ -n "${SLURM_JOB_ID:-}" ] || echo "!! warning: no SLURM allocation -- the build is heavy, use salloc on 'accel'" >&2

for d in "$ANEMOI_CORE_DIR/graphs" "$ANEMOI_CORE_DIR/models" "$ANEMOI_CORE_DIR/training" \
         "$NESE_INFERENCE_DIR" "$NESE_TRAINING_DIR"; do
    [ -d "$d" ] || { echo "!! missing source dir: $d" >&2; exit 1; }
done
[ -f "$DEF_IN" ] || { echo "!! missing $DEF_IN" >&2; exit 1; }
[ -f "$REQ" ]    || { echo "!! missing $REQ" >&2; exit 1; }
[ -f "$MODFILE_IN" ] || { echo "!! missing $MODFILE_IN" >&2; exit 1; }

if git -C "$ANEMOI_CORE_DIR" rev-parse --git-dir >/dev/null 2>&1; then
    _branch="$(git -C "$ANEMOI_CORE_DIR" rev-parse --abbrev-ref HEAD)"
    if [ "$_branch" != "$ANEMOI_CORE_BRANCH" ] || \
       ! git -C "$ANEMOI_CORE_DIR" merge-base --is-ancestor "$ANEMOI_CORE_COMMIT" HEAD 2>/dev/null; then
        echo "!! warning: $ANEMOI_CORE_DIR is on '$_branch' and/or missing $ANEMOI_CORE_COMMIT" >&2
        echo "!!          (training.yaml needs the nest_ordering schema kwarg from '$ANEMOI_CORE_BRANCH')" >&2
    fi
fi

if [ ! -f "$BASE_SIF" ]; then
    cat >&2 <<EOF
!! base image not found: $BASE_SIF
!! NRIS stages it; if it is gone, pull our own copy (needs the proxy + a large
!! APPTAINER_CACHEDIR; NGC anon pulls may need APPTAINER_DOCKER_USERNAME=\$oauthtoken
!! + an NGC API key):

   mkdir -p "$(dirname "$BASE_SIF")"
   apptainer pull --arch arm64 "$BASE_SIF" "$NGC_URL"

!! then re-run, or set NESE_ANEMOI_BASE_SIF to an existing arm64 PyTorch .sif.
EOF
    exit 1
fi

# --- 2. proxy (build %post does NOT inherit the caller's env) ---------------
export http_proxy="$HTTP_PROXY_URL" https_proxy="$HTTP_PROXY_URL"
export APPTAINERENV_http_proxy="$HTTP_PROXY_URL" APPTAINERENV_https_proxy="$HTTP_PROXY_URL"
mkdir -p "$APPTAINER_CACHEDIR" "$PREFIX" "$(dirname "$SIF")"

# --- 3. render the def ----------------------------------------------------
VERSION="$(date +%Y%m%d)"
# Keep the build dir off /cluster so `apptainer build` does not implicitly bind a
# /cluster path that then collides with our explicit --bind /cluster.
GEN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/nese-anemoi-build.XXXXXX")"
case "$GEN_DIR" in /cluster/*) GEN_DIR="$(mktemp -d /tmp/nese-anemoi-build.XXXXXX)" ;; esac
trap 'rm -rf "$GEN_DIR"' EXIT
# %files source paths resolve against the build CWD, not the def location, so pass
# the requirements file by absolute path.
sed -e "s#@BASE_SIF@#${BASE_SIF}#g" \
    -e "s#@REQUIREMENTS_TXT@#${REQ}#g" \
    -e "s#@HTTP_PROXY@#${HTTP_PROXY_URL}#g" \
    -e "s#@ANEMOI_CORE_DIR@#${ANEMOI_CORE_DIR}#g" \
    -e "s#@NESE_INFERENCE_DIR@#${NESE_INFERENCE_DIR}#g" \
    -e "s#@NESE_TRAINING_DIR@#${NESE_TRAINING_DIR}#g" \
    -e "s#@VERSION@#${VERSION}#g" \
    "$DEF_IN" > "$GEN_DIR/nese-anemoi.def"

# --- 4. build ------------------------------------------------------------
# .tmp + mv so an interrupted build never leaves a half-written .sif in place.
# --bind /cluster: the %post editable installs read source trees from /cluster;
# the def's %setup creates the mount point (the NGC base has no /cluster and a
# def-file build cannot auto-create bind destinations).
TMP_SIF="$SIF.tmp.$$"
echo ">> building $SIF (this is slow -- pip resolve + arm64 wheels)"
( cd "$GEN_DIR" && apptainer build --fakeroot --bind /cluster "$TMP_SIF" "$GEN_DIR/nese-anemoi.def" )
mv -f "$TMP_SIF" "$SIF"
echo ">> wrote $SIF"

# --- 5. modulefile -----------------------------------------------------
mkdir -p "$MODULES_ROOT/nese-anemoi"
MODFILE="$MODULES_ROOT/nese-anemoi/olivia.lua"
sed -e "s#@PREFIX@#${PREFIX}#g" \
    -e "s#@SIF@#${SIF}#g" \
    -e "s#@BASE_SIF@#${BASE_SIF}#g" \
    -e "s#@VERSION@#${VERSION}#g" \
    "$MODFILE_IN" > "$MODFILE"
echo ">> wrote modulefile $MODFILE"

cat <<EOF

Done. To use the environment:

    module use $MODULES_ROOT      # already in the user's bashrc
    module load nese-anemoi

Verify (imports + CUDA visibility; no training run):

    apptainer exec --bind /cluster "\$NESE_ANEMOI_SIF" python -c \\
      "import sys, anemoi.models, anemoi.graphs, anemoi.training, nese_training, nese.inference; print(sys.version)"
    apptainer exec --bind /cluster "\$NESE_ANEMOI_SIF" python -c \\
      "from importlib.metadata import entry_points; print('nese-zarr' in {e.name for e in entry_points(group='anemoi.inference.outputs')})"
    apptainer exec --bind /cluster "\$NESE_ANEMOI_SIF" anemoi-training --help
    apptainer exec --bind /cluster "\$NESE_ANEMOI_SIF" pip check   # known anemoi-inference pin warnings only

    # on a GPU node:
    apptainer exec --nv --bind /cluster "\$NESE_ANEMOI_SIF" python -c \\
      "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

See environment/olivia/README.md for training job scripts.
EOF
