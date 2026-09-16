#!/usr/bin/env bash
#
# Build (or rebuild) the containerised `nese` environment on Olivia and install a
# Lmod modulefile for it. Repeatable: safe to re-run.
#
#   bash environment/olivia/build-env.sh
#
# Native conda/pip installs are forbidden on Olivia, so the environment is wrapped
# with hpc-container-wrapper (Tykky). After this runs:
#
#   module use "$MODULES_ROOT"      # once; add to ~/.bashrc to make it permanent
#   module load nese
#
set -euo pipefail

# --- config (override via environment) ---------------------------------------
PREFIX="${NESE_ENV_PREFIX:-/cluster/projects/nn2993k/timith/sw/nese}"
DATAPREP_DIR="${NESE_DATAPREP_DIR:-/cluster/projects/nn2993k/timith/nese-dataprep}"
VERIFY_DIR="${NESE_VERIFY_DIR:-/cluster/projects/nn2993k/timith/nese-verify}"
MODULES_ROOT="${NESE_MODULES_ROOT:-/cluster/projects/nn2993k/timith/sw/modules}"
# GCC-14.3.0 toolchain, not the newer GCC-15.2.0 one: the latter's zlib is built for
# Zen5 and SIGILLs git/mksquashfs on the Zen4 login nodes. Same MPI ABI (libmpi.so.40).
OPENMPI_MOD="${NESE_OPENMPI_MOD:-OpenMPI/5.0.9-GCC-14.3.0}"
HCW_MOD="${NESE_HCW_MOD:-hpc-container-wrapper/0.5.1-CPU}"
HTTP_PROXY_URL="${NESE_HTTP_PROXY:-http://10.63.2.48:3128/}"
# Dedicated apptainer cache: isolated from ~/.apptainer so a bad layer from an
# interrupted pull can't poison later builds (that failure shows up as
# "mksquashfs ... illegal instruction"). Wipe it and retry if a pull dies.
APPTAINER_CACHEDIR="${NESE_APPTAINER_CACHEDIR:-/cluster/projects/nn2993k/timith/sw/.apptainer-cache}"
export APPTAINER_CACHEDIR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_YAML="$SCRIPT_DIR/environment.yaml"

echo ">> prefix        : $PREFIX"
echo ">> nese-dataprep : $DATAPREP_DIR"
echo ">> nese-verify   : $VERIFY_DIR"
echo ">> modules root  : $MODULES_ROOT"
echo ">> openmpi        : $OPENMPI_MOD"

[ -f "$ENV_YAML" ] || { echo "!! missing $ENV_YAML" >&2; exit 1; }
[ -d "$DATAPREP_DIR" ] || { echo "!! missing $DATAPREP_DIR" >&2; exit 1; }
[ -d "$VERIFY_DIR" ] || { echo "!! missing $VERIFY_DIR" >&2; exit 1; }

# --- 1. internet access from the login node ---------------------------------
export http_proxy="$HTTP_PROXY_URL"
export https_proxy="$HTTP_PROXY_URL"
mkdir -p "$APPTAINER_CACHEDIR"

# --- 2. clean stale Tykky build temp dirs ----------------------------------
# Tykky builds under /tmp/$USER/cw-* and its cleanup sometimes fails (apptainer
# leaves write-protected dirs behind), which can wedge later runs.
_tmpbase="/tmp/${USER}"
if compgen -G "$_tmpbase/cw-*" > /dev/null 2>&1; then
    echo ">> clearing stale $_tmpbase/cw-* build dirs"
    chmod -R u+w "$_tmpbase"/cw-* 2>/dev/null || true
    rm -rf "$_tmpbase"/cw-* || true
fi

# --- 3. modules -----------------------------------------------------------
# Just the container wrapper. The build compiles nothing against MPI (conda-forge's
# mpi4py links the external OpenMPI shim), so OpenMPI is a runtime-only dep.
module load NRIS/CPU
module load "$HCW_MOD"

# --- 4. base environment (conda + pip section, no MPI compiled here) --------
# A valid Tykky install has share/conf.yaml; a bare dir is a failed/partial build.
if [ -f "$PREFIX/share/conf.yaml" ]; then
    echo ">> $PREFIX is a valid install -- skipping 'conda-containerize new', updating in place"
else
    if [ -e "$PREFIX" ]; then
        echo ">> $PREFIX exists but is not a valid Tykky install -- removing and rebuilding"
        rm -rf "$PREFIX"
    fi
    mkdir -p "$(dirname "$PREFIX")"
    conda-containerize new --mamba --prefix "$PREFIX" "$ENV_YAML"
fi

# --- 5. post-install: editable nese-dataprep + nese-verify -----------------
# Bake the source dirs into a generated copy of the post-install script so it does
# not depend on Tykky forwarding environment into the build container.
GEN_DIR="$(mktemp -d)"
trap 'rm -rf "$GEN_DIR"' EXIT
sed -e "s#@DATAPREP_DIR@#${DATAPREP_DIR}#g" \
    -e "s#@VERIFY_DIR@#${VERIFY_DIR}#g" \
    "$SCRIPT_DIR/post-install.sh.in" > "$GEN_DIR/post-install.sh"
chmod +x "$GEN_DIR/post-install.sh"
conda-containerize update "$PREFIX" --post-install "$GEN_DIR/post-install.sh"

# --- 6. modulefile --------------------------------------------------------
VERSION="$(date +%Y%m%d)"
mkdir -p "$MODULES_ROOT/nese"
MODFILE="$MODULES_ROOT/nese/olivia.lua"
sed -e "s#@PREFIX@#${PREFIX}#g" \
    -e "s#@OPENMPI_MOD@#${OPENMPI_MOD}#g" \
    -e "s#@VERSION@#${VERSION}#g" \
    -e "s#@ENV_YAML@#${ENV_YAML}#g" \
    "$SCRIPT_DIR/modulefile.lua.in" > "$MODFILE"
echo ">> wrote modulefile $MODFILE"

cat <<EOF

Done. To use the environment:

    module use $MODULES_ROOT      # add to ~/.bashrc to make permanent
    module load nese

Verify:

    which python
    python -c "from mpi4py import MPI; print(MPI.Get_library_version())"
    python -c "import nese.dataprep, inspect; print(inspect.getfile(nese.dataprep))"
    python -c "import nese.verify, inspect; print(inspect.getfile(nese.verify))"

See environment/olivia/README.md for running MPI jobs.
EOF
