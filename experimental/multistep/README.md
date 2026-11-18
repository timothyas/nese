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

## Results

### Version 0

This hides a fraction of `(cell, time)`, so it hides any cell regardless of
whether or not it was observed.

Here I used `mask_fraction = 0.25`, different from what I originally proposed
above, although what I originally wrote is a little confusing.
Here, this meant that this fraction of cells was removed by setting
`obs_count_{x} = 0`, effectively a mask.

It is hard to directly compare to the forecasting setup.
On the one hand, the forecast setup receives 3 timesteps on input, and scores
them on one.
From that perspective, it's like the model sees 75% of the observations with 25%
masked.
On the other hand, if we think of observed vs unobserved cells, in
`../unseen-ft-all-altimeter/README.md` it was shown that the typical loss is
made up of ~75% grid cells that are unseen at input time.

In short, this motivates a sensitivity study once the hidden fraction definition
is settled.

#### RMSE:
* The RMSE scores are overall excellent, significantly beating persistence in
  all seen/unseen combinations.
* The `fhr` doesn't matter, each has the same accuracy relatively speaking.
* For each `fhr` selected, the lowest error is achieved for observations in that
  time slot (~0.9 cm).
* Error is a little higher at grid cells that are observed at input time, but at
  different forecast hours (~1.3 cm).
* Error at unseen grid cells, determined by the `input_hidden` mask, is highest
  of all again, but signficantly lower than before (2.3 cm vs 5.5 cm with
  forecasting)
* Those numbers are for SLA, but improvements are across the board for all
  variables

#### Extrema:
* Extreme values for all variables match the observed values surprisingly well

#### Qualitative plots:
* This is where we see some issues. Plots are visibly noisy
* "unobserved patches" exist as in the forecasting task, except here things are
  much noisier

#### Interpretation / conclusion:
The model is not regularized enough, and spitting out predictions that fit the
data too well.
Possible paths forward, after fixing the masking:
* Test different mask fractions, e.g. 0.25 and 0.75 for now. Does a higher mask
  fraction return something closer to the forecasting setup?
  Is the better fit + noisy predictions just coming from the fact that I copied
  this setup over (same `weight_decay`), but the misfit / weight decay ratio is
  very different here?
* Could a variable mask fraction schedule be appropriate? Give the model tons of
  observations at first, and then continually stress its ability to fill in the
  gaps as training proceeds?
* If not, I need to think about why this is so different from the forecasting
  setup.
* Or, is it really time to add a Matern-type regularization here?
