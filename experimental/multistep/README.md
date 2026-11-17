# Multistep

This picks up from `../update-forcings-aa/runs/v0`, which is really a small update
to `../all-altimeter/runs/v2p2`.
The idea here is to try a fundamentally different learning task.
Previously, we wanted to forecast from `{t-2dt, t-dt, t}` -> `t+dt`.
Now, lets try to predict full state estimates for multiple timestamps.
Specifically, from sparse observations at `{t-dt, t, t+dt}`, predict full or
dense states at those same time stamps.

Now, we will need to mask data in order to avoid just memorizing data where we
have it.
To make this comparable to the previous forecasting task (input 3 states,
predict 1), lets mask 75% of the input states randomly at training time.
With this masking, we will in all likelihood eventually let the model see all
parts of all data samples.
But, the training and validation split will still be the same, so we can monitor
overfitting that way.

A note about giving the model apparently future times, that is why do we give it
`t+dt`?
We could also view this as: given sparse observations at `{t-2dt, t-dt, t}`
predict dense estimates for those timestamps.
I used the original notation about because, from a product generation
perspective, I would suspect the middle estimate to be the best as it is
"constrained" by past and future state estimates.
Additionally, it would be natural to generate a reanalysis by only using the
middle time step and slide the 3 step window across the observation record.

## Implementation Notes

* Everywhere assumes we always use `[-𝚫t, 0, +𝚫t] -> [-𝚫t, 0, +𝚫t]`, keeping the
  central point as the true target.
  This could be generalized in several ways:
    * We could use more inputs than outputs, e.g.
      `[-2𝚫t, -𝚫t, 0, +𝚫t, +2𝚫t] -> [-𝚫t, 0, +𝚫t]`
    * It could be uncentered, e.g.
      `[-2𝚫t, -𝚫t, 0, +𝚫t] -> [-2𝚫t, -𝚫t, 0, +𝚫t]`
