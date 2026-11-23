# Update Forcings

This picks up from `../all-altimeter/v2p2` as the current baseline to improve
upon.
There are several aspects about anemoi forcing variables that are suboptimal,
and I would like to deal with them in order to understand how important they
are.

First, there is no distinction between static forcings and time-evolving
forcings, so static forcing variables (e.g., elevation) are repeated for every
input timestep.
The repeated information is wasteful and, more importantly, could influence
training evolution and bring more importance to the static fields.

Secondly, the definition of forcing variables is something that we know at all
times during the prediction. So for a forecasting task, we know forcing
variables at t > t0 in addition to (t0-Ndt, ... t0-dt, t0).
But, anemoi only provides forcings at the input times (t0-Ndt, ... t0-dt, t0).

This experiment directory tests the impact of changing both of these factors to
see how important they are.

## Limitations

The extension supports a single forecast step only, so it does not work for forecasting
applications that roll out over many timesteps.
- `KnownForcingForecaster` refuses a rollout or validation rollout above 1. Advancing the
  input window would roll the forcing-only t0+Δt slot into the past as if it had been
  observed.
- The `known_forcing` runner in nese-inference only allows `lead_time` equal to the
  checkpoint's timestep. Δt comes from the checkpoint's own task `timestep`, so it is 24h
  here, while a 6h model would allow exactly one 6h step.
- The runner builds the input window by multiplying the checkpoint's input and output date
  indices by that timestep. This assumes the task's offsets are evenly spaced by `timestep`,
  which holds for `KnownForcingForecaster`.

Supporting rollout would mean writing the forcings for each new target time into the future
slot at every step, in both the training task's `advance_input` and the inference runner.

## Results

### v0

This design is cleaner and makes me feel good, but it doesn't meaningfully
change anything in terms of performance.
RMSE is less than 1% different, and the fields look qualitatively the same.

Conclusion: this may matter more for a model setup with more important forcings,
but for now it's no big deal.
