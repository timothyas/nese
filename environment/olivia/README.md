# Olivia environment setup

Olivia forbids native `conda`/`pip` installs, so the `nese` environment is built as
a container with [hpc-container-wrapper](https://documentation.sigma2.no/hpc_machines/olivia/software_stack.html#python-r-and-ana-conda)
(Tykky). There is no `conda activate` / `conda run -n nese` here.
Instead, load a
modulefile that puts the container's `bin/` on `PATH`.

## Build / rebuild

```bash
bash environment/olivia/build-env.sh
```

This:
1. sets the login-node proxy and loads `NRIS/CPU` + `hpc-container-wrapper` (OpenMPI
   is a runtime-only dep, pulled in by the modulefile, not needed for the build);
2. `conda-containerize new` from `environment/olivia/environment.yaml` (skipped if the
   prefix already exists);
3. `conda-containerize update` to install `nese-dataprep` editable (`pip install -e`);
4. writes the modulefile to `$NESE_MODULES_ROOT/nese/olivia.lua`.

Defaults (override by exporting before running):

| var | default |
|---|---|
| `NESE_ENV_PREFIX` | `/cluster/projects/nn2993k/timith/sw/nese` |
| `NESE_DATAPREP_DIR` | `/cluster/projects/nn2993k/timith/nese-dataprep` |
| `NESE_MODULES_ROOT` | `/cluster/projects/nn2993k/timith/sw/modules` |
| `NESE_OPENMPI_MOD` | `OpenMPI/5.0.9-GCC-14.3.0` |

The build needs internet (proxy) and can be slow, so run it in `tmux`/`screen` or a
short interactive job.

### Troubleshooting

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

## Use

```bash
module use /cluster/projects/nn2993k/timith/sw/modules   # add to ~/.bashrc to persist
module load nese

python ...
nese-dataprep inspect
```

`module load nese` also exports the Olivia HTTP proxy
(`http_proxy`/`https_proxy` = `10.63.2.48:3128`), which login and compute nodes
both need for internet access; `module unload nese` reverts it.

Editing `nese-dataprep` source is live (editable install). Re-run `build-env.sh`
only after changing its **dependencies**, **entry points**, or **version**, or after
editing `environment/olivia/environment.yaml`.

### Copernicus Marine

Note that before creating data with `nese-dataprep` you'll want to login to
copernicus marine with

```bash
module load nese          # sets https_proxy
copernicusmarine login
```

And before building a big dataset we can sanity check that it worked with

```bash
nese-dataprep inspect cmems_l3_sla
```

## Running MPI jobs

The `nese` module loads `OpenMPI/5.0.9-GCC-14.3.0` (so `mpi4py` finds the system
`libmpi`) and sets the **single-node** transport vars (`OMPI_MCA_pml=ob1`,
`OMPI_MCA_btl=sm,self`) -- enough for serial work and single-node jobs. The
GCC-14.3.0 toolchain is used rather than the newer GCC-15.2.0 one because the
latter's zlib SIGILLs `git` on the Zen4 login nodes. Launch with `srun`
(single-node `cxi` can break under `mpirun`).

Single-node (CPU): nothing extra needed.

```bash
srun -n "$SLURM_NTASKS" nese-dataprep create recipe.yaml
# GPU node only: export OMPI_MCA_btl=sm,smcuda,self
```

Multi-node: **override** the module's single-node vars in the job script.

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

## Keeping the two env files in sync

`environment/olivia/environment.yaml` is a hand-maintained copy of
`environment/environment.yaml` that drops `mpich` / `mpi=*=mpich`, replaces them
with `openmpi=5.0.*=external_*` (a contentless build that defers to the system
OpenMPI), and adds the `conda-forge/label/mpi-external` channel those builds live
on. Mirror any other change to `environment/environment.yaml` here.

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
