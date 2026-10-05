# Handoff: verify `nese-submit` on Olivia

You are taking over the last verification step of a change made on the Mac. Read
`$WORK/nese/CLAUDE.md` first. On Olivia you **must** follow Sigma2's AI policy
(https://www.sigma2.no/policy-agentic-ai-tools-hpc-systems). Keep login-node work light: the
commands below only submit jobs, read logs and list files. Training, inference and
verification all run as SLURM jobs.

## What changed and why

Inference and verify need the training `run_id`, because checkpoints live in
`runs/v3/checkpoint/<run_id>/`. MLflow used to pick that id only once training started, so the
downstream jobs couldn't be queued together with training. Now:

- **nese-logistics**: a new `nese-submit` CLI (`nese.logistics.launch`). It writes a new MLflow
  run, tagged `dry_run=True`, straight into the offline file store (no mlflow needed), then
  `sbatch`es each script with `--export=ALL,NESE_RUN_ID=<id>`, chained by `afterok` with
  `--kill-on-invalid-dep=yes`.
- **nese-verify**: `plot-loss` plots the run named by `training.run_id` and writes to
  `runs/v3/<run_id>/`. Without one, it plots the latest run as before.
- **This experiment**: `training.yaml` sets `training.run_id: ${oc.env:NESE_RUN_ID,null}` and
  `on_resume_create_child: false`. The inference and verify yamls and `plot_*.py` read and write
  `runs/v3/$NESE_RUN_ID/...`.

anemoi trains such a run from scratch: `_check_dry_run` in
`anemoi-core/training/src/anemoi/training/train/train.py`. Once a checkpoint exists, the same
id resumes from it.

## Already verified on the Mac

- Unit tests: nese-logistics (60 passed, `launch.py` 100% covered, including `nese-submit`
  run against a fake `sbatch`) and nese-verify (162 passed).
- mlflow 3.15.1 reads the written run (`get_run`, `search_runs`, artifacts), including inside an
  experiment that mlflow itself created.
- anemoi's `AnemoiMLflowLogger(offline=True, run_id=<prepared>, on_resume_create_child=False)`
  sees `_parent_dry_run=True`, logs into that run, and finalizes it `FINISHED`. No second run is
  created.
- `prepare` on the real `training.yaml` resolves with no `SLURM_*` variables set.
  anemoi-inference's `RunConfiguration.load` and nese-verify's config loader resolve the new
  paths, and an unset `NESE_RUN_ID` fails with `Environment variable 'NESE_RUN_ID' not found`.

## Not verified yet (your job)

1. A real 4-rank training job started from a prepared run. The code says rank 0 sees the dry
   run and ranks 1–3 find no checkpoint and pass `ckpt_path=None`, but that has never run.
2. `NESE_RUN_ID` reaching the Python processes through `sbatch` → `srun` → `apptainer exec`.
3. `--kill-on-invalid-dep` cancelling the downstream jobs.
4. Resubmitting the same id resumes into the same run.
5. `nese-submit` on the x86 login node after the env rebuild.

## Steps

All from `$WORK/nese/experimental/diagnose-uvgxt` unless noted.

0. **Setup.** Pull the three repos the user committed to (nese-logistics, nese, nese-verify;
   ask which branch). The `nese` env installs nese-logistics and nese-verify editable from
   `/cluster/projects/nn2993k/timith/`, so check those checkouts are the ones updated.
   `nese-submit` is a new entry point, so the `nese` env needs rebuilding. That is the user's
   call: ask them to run it, or for the go-ahead to run it yourself, following
   `$WORK/nese/environment/olivia/README.md` ("Build / Rebuild Process"). Then check:
   ```bash
   module purge; module use /cluster/projects/nn2993k/timith/sw/modules; module load nese
   which nese-submit && nese-submit --help
   ```
   The user already confirmed that `sbatch` works from the env's python.
1. **A short training script.** Copy `submit_training.sh` to `submit_training.smoke.sh`
   (don't commit it) and append to the `anemoi-training train` line:
   ```
   training.max_steps=${SMOKE_STEPS:-40} dataloader.limit_batches.training=20 dataloader.limit_batches.validation=4
   ```
   That is 2 epochs, so 2 checkpoints.
2. **Count the existing runs**, so a stray second run shows up later:
   `ls runs/v3/logs/mlflow/*/ | wc -l`.
3. **Submit the chain:**
   ```bash
   nese-submit -n launch-smoke training.yaml submit_training.smoke.sh submit_inference.sh submit_verify.sh
   ```
   Record the run id and the three job ids.
4. **Check training** (`slurm/training-v3-<job>.out`):
   - `Run id: <id>`, `Parent run is a Dry Run: True`, `Dry run: True`, `Starting from checkpoint: False`
   - no traceback from **any** rank, and the job ends COMPLETED (`sacct -j <job>`)
   - `runs/v3/checkpoint/<id>/inference-last.ckpt` exists
   - `runs/v3/logs/mlflow/<exp id>/<id>/meta.yaml` has `status: 3` (FINISHED) and `metrics/` is
     not empty
   - the run count from step 2 went up by exactly one
5. **Check inference and verify:** `runs/v3/<id>/inference-validation` and
   `runs/v3/<id>/inference-2015` exist. The verify job wrote `runs/v3/<id>/loss_curves*.pdf`,
   metrics under the inference dirs, and the movie. Its log shows plot-loss picked `<id>`.
6. **Resume:** `NESE_RUN_ID=<id> SMOKE_STEPS=60 sbatch submit_training.smoke.sh`. Expect
   `Resuming training from last checkpoint`, metric files in the **same** run dir gaining lines
   past step 40, and no new run dir.
7. **Cancel:**
   `nese-submit -n launch-cancel training.yaml submit_training.smoke.sh submit_inference.sh submit_verify.sh`,
   then `scancel <training job>` while it is pending. The two dependent jobs should end
   CANCELLED (`sacct -j <a>,<b>`), not sit pending with `DependencyNeverSatisfied`.
8. **Clean up, with the user's OK:** delete `submit_training.smoke.sh`, and for both smoke run
   ids: `runs/v3/logs/mlflow/<exp id>/<id>`, `runs/v3/checkpoint/<id>`, `runs/v3/plots/<id>`
   and `runs/v3/<id>`.

## If something fails

Report it with the log excerpt. Don't patch anemoi-core to get past it. Likely places:
- A rank other than 0 crashes or hangs in step 4 (finding 1 above). See `last_checkpoint` and
  `_check_dry_run` in `train.py`.
- `KeyError`/`Environment variable 'NESE_RUN_ID' not found` inside a job: the variable didn't
  get through `srun`/`apptainer`. Check for `--cleanenv` or an `--export` override.
- `nese-submit: command not found`: the `nese` module isn't loaded, or the env wasn't rebuilt
  after the pull. `python -m nese.logistics.launch ...` takes the same arguments and works
  without the rebuild.
- mlflow can't read the run on Olivia (`get_run` raises): compare
  `runs/v3/logs/mlflow/<exp>/<id>/meta.yaml` with an older run's meta written by anemoi.

## Report back

The run id, job ids and states, a pass/fail for each of steps 4–7 with the log lines that
show it, and anything that differed from this document.
