# Olivia environment setup

Olivia forbids native `conda`/`pip` installs.
This README documents how to make CPU and GPU environments that are accessed by
loading a modulefile, i.e., by running `module load <environment>`.

## CPU Environment: `nese`

```bash
module use /cluster/projects/nn2993k/timith/sw/modules   # add to ~/.bashrc to persist
module load nese

python ...
nese-dataprep inspect
nese-verify ...
```

Built as a container with
[hpc-container-wrapper](https://documentation.sigma2.no/hpc_machines/olivia/software_stack.html#python-r-and-ana-conda)
(Tykky).
There is no `conda activate` / `conda run -n nese` here.
Instead, we load a
modulefile that puts the container's `bin/` on `PATH`.

### Running MPI jobs

The `nese` module loads `OpenMPI/5.0.9-GCC-14.3.0` (so `mpi4py` finds the system
`libmpi`) and sets the transport vars for single-node jobs:

```bash
OMPI_MCA_pml=ob1
OMPI_MCA_btl=sm,self
```

So we don't need to set these in SLURM scripts.
Launch with `srun`, since single-node `cxi` can break under `mpirun`.

```bash
srun -n "$SLURM_NTASKS" nese-dataprep create recipe.yaml
# GPU node only: export OMPI_MCA_btl=sm,smcuda,self
```

For multi-node jobs: **override** the module's single-node vars in the job script.

```bash
export OMPI_MCA_pml=cm
export OMPI_MCA_mtl=ofi
export OMPI_MCA_mtl_ofi_av=table
export PRTE_MCA_ras_base_launch_orted_on_hn=1
export FI_PROVIDER=cxi
export FI_CXI_RX_MATCH_MODE=hybrid
srun -n "$SLURM_NTASKS" nese-dataprep create recipe.yaml
```

Source: <https://documentation.sigma2.no/hpc_machines/olivia/openmpi.html>

### Build / Rebuild Process

```bash
bash environment/olivia/build-env.sh
```

This:
1. sets the login-node proxy and loads `NRIS/CPU` + `hpc-container-wrapper` (OpenMPI
   is a runtime-only dep, pulled in by the modulefile, not needed for the build);
2. `conda-containerize new` from `environment/olivia/environment.yaml` (skipped if the
   prefix already exists);
3. `conda-containerize update` to install `nese-dataprep` and `nese-verify` editable
   (`pip install -e`); both provide subpackages of the shared `nese` namespace;
4. writes the modulefile to `$NESE_MODULES_ROOT/nese/olivia.lua`.

Defaults (override by exporting before running):

| var | default |
|---|---|
| `NESE_ENV_PREFIX` | `/cluster/projects/nn2993k/timith/sw/nese` |
| `NESE_DATAPREP_DIR` | `/cluster/projects/nn2993k/timith/nese-dataprep` |
| `NESE_VERIFY_DIR` | `/cluster/projects/nn2993k/timith/nese-verify` |
| `NESE_MODULES_ROOT` | `/cluster/projects/nn2993k/timith/sw/modules` |
| `NESE_OPENMPI_MOD` | `OpenMPI/5.0.9-GCC-14.3.0` |
| `NESE_HTTP_PROXY` | `http://10.63.2.48:3128` |

The build needs internet (proxy) and can be slow, so run it in `tmux`/`screen` or a
short interactive job.
The HTTP proxy is exported into the module as `HTTP_PROXY_URL` and
`HTTPS_PROXY_URL`, so that all applications, like `nese-dataprep` are ready for
internet access.

`nese-dataprep` and `nese-verify` source code can be edited, since they are
editable installs.
It is only necessary to re-run `build-env.sh` if there are changes to:
* their package dependencies,
* entry points (e.g., new CLI commands),
* versions, or
* `environment/olivia/environment.yaml`

### First Usage: Login to Copernicus Marine

Before creating data with `nese-dataprep`, login to copernicus marine with

```bash
module load nese                    # sets https_proxy
copernicusmarine login
nese-dataprep inspect cmems_l3_sla  # sanity-check, accesses CMEMS
```

### Build Troubleshooting

- **Used GCC-14.3.0 toolchain** rather than the newer GCC-15.2.0 one because the
  latter's zlib SIGILLs `git` on the Zen4 login nodes.
- **`mksquashfs ... signal: illegal instruction (core dumped)`** during the base
  image fetch: something on `LD_LIBRARY_PATH` gave `mksquashfs` a library built for
  Zen5 while the login node is Zen4 -- usually a `GCC-15.2.0`-toolchain module (e.g.
  `OpenMPI/5.0.10-GCC-15.2.0`) loaded in the shell. Run the build from a login shell
  with only `NRIS/CPU` + `hpc-container-wrapper` loaded.
  A corrupt apptainer cache layer can also cause it; the script uses a dedicated
  `APPTAINER_CACHEDIR` (`$NESE_APPTAINER_CACHEDIR`, default `.../sw/.apptainer-cache`);
  wipe it and re-run: `rm -rf /cluster/projects/nn2993k/timith/sw/.apptainer-cache`.
- **`errSymlink is not user-visible`** at the end of a pull is a harmless apptainer
  cleanup bug; the image is fine. The script also clears stale `/tmp/$USER/cw-*`
  build dirs on each run.
- A failed build can leave an empty/partial prefix dir; the script detects that
  (no `share/conf.yaml`) and rebuilds from scratch.

### Note on cross-machine compatibility

`environment/olivia/environment.yaml` is a hand-maintained copy of
`environment/environment.yaml` that drops `mpich` / `mpi=*=mpich`, replaces them
with `openmpi=5.0.*=external_*` (a contentless build that defers to the system
OpenMPI), and adds the `conda-forge/label/mpi-external` channel those builds live
on. Mirror any other change to `environment/environment.yaml` here.

## GPU Environment: `nese-anemoi`

The module is thin: it sets `NESE_ANEMOI_SIF` and the proxy and puts **nothing** on
`PATH`.
Run tools explicitly with `apptainer`:

```bash
module load nese-anemoi
apptainer exec --nv --bind /cluster "$NESE_ANEMOI_SIF" python ...
apptainer exec --nv --bind /cluster "$NESE_ANEMOI_SIF" anemoi-training train --config-name=training ...
```

The `nese-anemoi` environment is for training, inference, and other
workflows involving anemoi packages.
Due to some build issues (e.g., cartopy has no aarch64 wheel and forces
`numpy>=2`) the specialized plotting, notebook, and documentation packages are
not included.
matplotlib is included, however, for basic diagnostic plots during training.

- `anemoi-train-multinode.slurm` -- multiple nodes. NCCL over Slingshot needs the
  host libfabric / NCCL / aws-ofi-nccl / OpenMPI bound into the container; the
  script carries the NRIS bind list and module set (confirm versions with
  `module spider`).

### Build / rebuild

`nese-anemoi` is **not** a Tykky container: on
Olivia every GPU node is a GH200 (aarch64 Grace CPU + Hopper GPU), conda-forge has
no linux-aarch64 CUDA build of PyTorch, and NRIS's guidance is to use the NVIDIA
NGC PyTorch container. So `nese-anemoi` is a custom Apptainer image: NRIS's staged
NGC arm64 PyTorch container (`pytorch_nvidia_25.06_arm64.sif` -- torch 2.8 / CUDA
12.9 / Python 3.12) extended with the anemoi PyPI packages and editable
`anemoi-core/{graphs,models,training}`, `nese-inference`, `nese/training`. torch /
torchvision / CUDA / NCCL / **flash-attn** / Transformer Engine are the container's
and are never touched by pip (the NGC image already ships `flash_attn`, so there is
no source build). The `%post` install runs **without `--upgrade`**, so every
package NGC already provides (numpy 1.26, scipy, numcodecs, ...) is kept and only
the missing anemoi bits are added, resolved against NGC's numpy.

The image is aarch64 and its `%post` runs arm64 binaries, so the build must run on
an `accel` node (login nodes are `x86_64`).
However, we can do a proper build by requesting 0 GPUs:

```bash
salloc --nodes=1 --cpus-per-task=64 --mem=200G --gpus-per-node=0 \
       --qos=devel --time=01:00:00 --partition=accel --account=nn2993k
bash environment/olivia/build-anemoi-env.sh
```

This:
1. checks the host is aarch64, the source trees exist, and `anemoi-core` is on
   `feature/ordering-kwarg-schema` (training configs need its `nest_ordering`
   schema kwarg);
2. exports the proxy (as `http_proxy` *and* `APPTAINERENV_http_proxy` -- `%post`
   does not inherit the caller's env);
3. renders `nese-anemoi.def.in` and runs
   `apptainer build --fakeroot --bind /cluster` (`.tmp` + `mv`, so an interrupted
   build leaves no half-written `.sif`);
4. writes the modulefile to `$NESE_ANEMOI_MODULES_ROOT/nese-anemoi/olivia.lua`.

Defaults (override by exporting before running):

| var | default |
|---|---|
| `NESE_ANEMOI_ENV_PREFIX` | `/cluster/projects/nn2993k/timith/sw/nese-anemoi` |
| `NESE_ANEMOI_SIF` | `$NESE_ANEMOI_ENV_PREFIX/nese-anemoi.sif` |
| `NESE_ANEMOI_BASE_SIF` | `/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif` |
| `NESE_ANEMOI_NGC_URL` | `docker://nvcr.io/nvidia/pytorch:25.06-py3` |
| `NESE_ANEMOI_MODULES_ROOT` | `/cluster/projects/nn2993k/timith/sw/modules` |
| `NESE_ANEMOI_CORE_DIR` | `/cluster/projects/nn2993k/timith/anemoi-core` |
| `NESE_INFERENCE_DIR` | `/cluster/projects/nn2993k/timith/nese-inference` |
| `NESE_TRAINING_DIR` | `/cluster/projects/nn2993k/timith/nese/training` |
| `NESE_ANEMOI_HTTP_PROXY` | `http://10.63.2.48:3128` |

The base image is referenced straight from NRIS's staged path, it is not copied.
If NRIS removes or bumps it, we will need to rebuild.

`.py` edits in `anemoi-core`, `nese-inference`, `nese/training` are live (editable
installs, source bind-mounted from `/cluster`). Rebuild only after changing a
**dependency**, **entry point**, or **version**, or after editing
`requirements-anemoi.txt` / `nese-anemoi.def.in`.

### How to add packages

- **Permanent:** add it to `requirements-anemoi.txt` (PyPI), or to the `%post`
  `apt-get` line (system lib) / the editable loop (another local repo) in
  `nese-anemoi.def.in`, then rebuild.
- **Throwaway (no rebuild):** an Apptainer overlay --
  `apptainer overlay create --sparse --size 5120 ov.img`, then
  `apptainer exec --overlay ov.img "$NESE_ANEMOI_SIF" pip install <pkg>` in a
  `--gpus-per-node=0` job, and run with `--overlay ov.img:ro`.

### Note on cross-machine compatibility

`environment/olivia/requirements-anemoi.txt` is a hand-maintained pip translation
of `environment/environment-anemoi.yaml` (the Mac/conda reference) for the NGC
container. Mirror changes there, applying:

- both files are training+inference only -- the plotting / notebook / docs stack
  is in neither (see the yaml header);
- drop everything NGC already ships (`pytorch`, `torchvision`, `numpy`, `scipy`,
  `numba`, `pandas`, `matplotlib`, `zarr`, `numcodecs`, `pydantic`, `networkx`,
  ...) -- the install runs without `--upgrade` so NGC's versions win anyway; keep
  only a short "guardrail" section pinning ranges that matter (`numpy<2`,
  `zarr<3`, ...);
- Use `python=3.12` instead of `python=3.13`. Note anemoi states 3.12 anyway;
- conda -> pip renames: `pytorch_geometric` -> `torch-geometric`;
- `anemoi-core` packages are editable (`%post`), not listed.

### Build Troubleshooting

- **`exec format error`** running the image: you are on a login node (x86_64).
  Build and run on `accel`.
- **`%post` network failures** during build: the proxy did not reach the build
  sandbox -- check `APPTAINERENV_http_proxy` is exported (the build script does
  this).
- **`mount /cluster: destination /cluster doesn't exist in container`** during
  `%post`: the def's `%setup` (`mkdir ${APPTAINER_ROOTFS}/cluster`) creates the
  bind target -- a def-file build cannot auto-create it. If you also see
  `destination is already in the mount point list`, run the build from a dir that
  is not under `/cluster` (the build script uses a `/tmp` build dir for this).
- **`uv: not found`** / **`externally-managed-environment`** in `%post`: the raw
  NGC image has no `uv` and its pip is PEP-668 marked. The def uses
  `python -m pip install --break-system-packages` for this; if a future base image
  behaves differently, adjust the `PIP=` line in `nese-anemoi.def.in`.
- **`ResolutionImpossible` / `conflicting dependencies`** in `%post`: NGC's own
  `/etc/pip/constraint.txt` pins the whole NVIDIA ecosystem, including loose deps
  the anemoi suite needs newer. The def ignores that file and builds a minimal
  constraint pinning only the compiled stack (`torch`/`torchvision`/`torchaudio`/
  `triton`/`numpy`/`flash-attn`/`transformer-engine`) to the image's versions, and
  installs without `--upgrade`. If a package still hard-requires `numpy>=2` (NGC
  ships 1.26), pin *that package* to its last numpy-1.26-compatible release in
  `requirements-anemoi.txt` rather than letting numpy move.
- **`cartopy` / plotting**: intentionally not in the image (no aarch64 wheel,
  forces `numpy>=2`). If you need map projections with coastlines for eval on a
  GPU node, add cartopy via an overlay: `apt`-provide `libgeos-dev libproj-dev`
  outside the container is not possible, so build the overlay from a def that
  `%post`s `env -u PIP_CONSTRAINT pip install cartopy`. Simpler: plot in the CPU
  `nese` env.
- **torch got clobbered**:
  `apptainer exec --nv "$NESE_ANEMOI_SIF" python -c "import torch; print(torch.__version__, torch.version.cuda)"`
  must print `2.8...` / `12.9`. A `+cpu` build or a plain-PyPI version means a
  requirement dodged `PIP_CONSTRAINT` -- pin it in `requirements-anemoi.txt`.
- **`pip check`** warnings about `anemoi-inference==0.11.2` vs the editable
  `anemoi-models` / `-graphs` / `-training` versions are expected and harmless.
- **`Boolean1DMask` CPU/GPU device mismatch** at scaler construction on the first
  GPU run -- see `experimental/local-sla-16yr/scaling_up.md`.
- **Benign build warnings** (image is fine):
  - `pynvml 12.0.0 requires nvidia-ml-py<13` -- NGC's `pynvml` is a deprecated
    re-export stub; anemoi needs `nvidia-ml-py>=13`. Nothing on the train/infer
    path imports `pynvml`.
  - `nvidia-dali-cuda120 requires packaging<=24.2` -- a dep pulled `packaging` 26;
    DALI is unused.
  - `%test`: `mkdir /root/.config/... Read-only file system` from matplotlib /
    earthkit-data -- `$HOME` is read-only *during the build test only*. At runtime
    `apptainer exec` bind-mounts your real `$HOME`. The example job scripts also
    set a per-job `MPLCONFIGDIR`.
  - `bind mounting /cluster ... already in the mount point list` -- harmless dedup
    warning; the bind works.

## File System Notes

I'm new to Olivia, so these notes detail the setup in terms of file systems used
and environment specs+how-to's

Filesystems
* [Project Area](https://documentation.sigma2.no/files_storage/clusters.html#project-area)
  for repos and persistent data
    * `/cluster/projects/<project>/<username>`
    * 1 TiB per project (currently we have 15 TiB, using ~11)
    * shared amongst users
* [Project Work Area](https://documentation.sigma2.no/files_storage/clusters.html#project-work-area)
  for semi-persistent files...
    * `/cluster/work/projects/<project>/<username>`
    * Once file system is 70% full, files older than 21 days are deleted
    * After 90%, files older than 17 days are deleted
* [Job Scratch Area](https://documentation.sigma2.no/files_storage/clusters.html#job-scratch-area)
  for temporary scratch during jobs
    * `LOCALSCRATCH=/localscratch/$SLURM_JOB_ID` for CPUs
    * `/cluster/gpuscratch/jobs/$SLURM_JOB_ID` for GPUs
    * files are removed automatically after running
    * this is local to disk, so only useful for caching etc with multi-node jobs
