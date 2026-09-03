# Scripts

Every number and table in `../main.tex` is produced here. All run in the `nese-anemoi`
environment and read the mesh from `../../anemoi-hpx6-24h/dataset.zarr` (override with
`--dataset`).

```
conda run -n nese-anemoi python table_phi.py
```

| script | produces |
|---|---|
| `table_phi.py` | Table `tab:phi` — neighbour spacings and stencil anisotropy (§3.2) |
| `table_pixel_shape.py` | Table `tab:sides` — true pixel side lengths vs `sqrt(4 pi / n_pix)` (§3.1) |
| `check_weights.py` | Table `tab:weights` — two-point flux vs the implemented stencil (§3.1) |
| `table_sigma.py` | Table `tab:sigma` — `sigma/sigma_interior` vs correlation range |
| `table_coast.py` | Table `tab:coast`, the cell counts and isolated-cell locations of §3.3 |
| `table_validation.py` | Table `tab:validation` — discrete vs analytical Matérn |
| `check_normalisation.py` | §2 verification: unit marginal variance, `E[x'C^-1x] = N`, and that `sigma` acts on `x` *before* `A` |
| `check_alternatives.py` | §3.3 — the boundary-stencil diagnosis and the two rejected fixes |

`matern.py` holds the shared `Mesh` class and operator; `common.py` holds argument
parsing. The table scripts print LaTeX rows, so a changed number can be pasted straight
back into `main.tex`.

Randomised quantities (anything involving `sigma`) move in the third significant figure
between seeds; `--n-samples` and `--seed` control this. `run_all.sh` runs everything.
