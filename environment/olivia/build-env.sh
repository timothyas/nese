#!/usr/bin/env bash
#
# Build (or rebuild) the containerised `nese` environment on Olivia and install a
# Lmod modulefile for it. Repeatable: safe to re-run.
#
#   bash environment/olivia/build-env.sh
#   bash environment/olivia/build-env.sh --modulefile-only   # just re-render the modulefile
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
LOGISTICS_DIR="${NESE_LOGISTICS_DIR:-/cluster/projects/nn2993k/timith/nese-logistics}"
DATAPREP_DIR="${NESE_DATAPREP_DIR:-/cluster/projects/nn2993k/timith/nese-dataprep}"
VERIFY_DIR="${NESE_VERIFY_DIR:-/cluster/projects/nn2993k/timith/nese-verify}"
MODULES_ROOT="${NESE_MODULES_ROOT:-/cluster/projects/nn2993k/timith/sw/modules}"
# A version present in both the Zen4 (NRIS/Login) and Zen5 (NRIS/CPU) trees, so the one
# modulefile works on login and compute nodes. Not GCC-15.2.0: its zlib SIGILLs
# git/mksquashfs on the Zen4 login nodes. Same MPI ABI (libmpi.so.40) across 5.0.x.
OPENMPI_MOD="${NESE_OPENMPI_MOD:-OpenMPI/5.0.8-GCC-14.3.0}"
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
echo ">> nese-logistics: $LOGISTICS_DIR"
echo ">> nese-dataprep : $DATAPREP_DIR"
echo ">> nese-verify   : $VERIFY_DIR"
echo ">> modules root  : $MODULES_ROOT"
echo ">> openmpi        : $OPENMPI_MOD"

write_modulefile() {
    local modfile="$MODULES_ROOT/nese/olivia.lua"
    mkdir -p "$MODULES_ROOT/nese"
    sed -e "s#@PREFIX@#${PREFIX}#g" \
        -e "s#@OPENMPI_MOD@#${OPENMPI_MOD}#g" \
        -e "s#@VERSION@#$(date +%Y%m%d)#g" \
        -e "s#@ENV_YAML@#${ENV_YAML}#g" \
        "$SCRIPT_DIR/modulefile.lua.in" > "$modfile"
    echo ">> wrote modulefile $modfile"
}

if [ "${1:-}" = "--modulefile-only" ]; then
    write_modulefile
    exit 0
fi

[ -f "$ENV_YAML" ] || { echo "!! missing $ENV_YAML" >&2; exit 1; }
[ -d "$LOGISTICS_DIR" ] || { echo "!! missing $LOGISTICS_DIR" >&2; exit 1; }
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
# Just the container wrapper. hpc-container-wrapper is only in the Zen5 tree, so this
# loads NRIS/CPU even on a login node; NRIS's "load NRIS/Login instead" warning is
# expected here. The build compiles nothing against MPI (conda-forge's
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

# --- 5. post-install: editable nese-logistics, nese-dataprep, nese-verify  -
# Bake the source dirs into a generated copy of the post-install script so it does
# not depend on Tykky forwarding environment into the build container.
GEN_DIR="$(mktemp -d)"
trap 'rm -rf "$GEN_DIR"' EXIT
sed -e "s#@LOGISTICS_DIR@#${LOGISTICS_DIR}#g" \
    -e "s#@DATAPREP_DIR@#${DATAPREP_DIR}#g" \
    -e "s#@VERIFY_DIR@#${VERIFY_DIR}#g" \
    "$SCRIPT_DIR/post-install.sh.in" > "$GEN_DIR/post-install.sh"
chmod +x "$GEN_DIR/post-install.sh"
conda-containerize update "$PREFIX" --post-install "$GEN_DIR/post-install.sh"

write_modulefile

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
