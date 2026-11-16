# No Latent Skip

This picks up from ../update-forcings-aa/runs/v0, which is really a small update
to ../all-altimeter/runs/v2p2.
The sensitivities that propagate through the model look overly pointed, and my
concern is that the latent skip is part of the reason for the imprint of
observation locations.
Here, lets try removing that.
It's a simple boolean in the yaml.
