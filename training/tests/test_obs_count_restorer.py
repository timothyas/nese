"""Gap restoration and truth composition on the inverse pass.

Both only ever reach the diagnostic plots, which is exactly what makes them worth
testing: nothing downstream will complain if they are wrong, the figures will just
quietly show the wrong truth. The composition in particular has to stay out of the
forward pass -- merging the missions into one field is the model's job, and a merge that
leaked into training would be handing it the answer.
"""

import numpy as np
import pytest
import torch
from nese_training.processors import ObsCountNaNRestorer

GRID = 8
GROUPS = ["cryosat_2", "jason_2", "topex_poseidon"]
SPEC = {"sla": {"observed_as": "sla_filtered", "groups": "auto", "count": "obs_count_{group}"}}


class FakeIndices:
    """``data.output`` as the restorer reads it: a name map plus the column order.

    ``elevation`` is forcing-only, so it sits in the name map but not in ``output.full``
    -- the same offset between global index and tensor column that the real config has.
    """

    def __init__(self):
        names = ["elevation", "sla"]
        for g in GROUPS:
            names += [f"sla_filtered_{g}", f"obs_count_{g}"]
        name_to_index = {n: i for i, n in enumerate(names)}
        full = np.array([i for n, i in name_to_index.items() if n != "elevation"])
        output = type("_O", (), {"name_to_index": name_to_index, "full": full})
        self.data = type("_D", (), {"output": output})
        self.names = names
        self.columns = {n: k for k, n in enumerate(n for n in names if n != "elevation")}


def make_tensor(values: dict, counts: dict) -> tuple[torch.Tensor, FakeIndices]:
    """A ``data.output.full``-shaped tensor, with gaps already filled as the imputer
    leaves them (a real number everywhere, no NaN)."""
    idx = FakeIndices()
    x = torch.zeros(1, 1, 1, GRID, len(idx.columns))
    x[..., idx.columns["sla"]] = 99.0  # the placeholder column, to be overwritten
    for g in GROUPS:
        x[..., idx.columns[f"sla_filtered_{g}"]] = torch.tensor(values[g], dtype=torch.float32)
        x[..., idx.columns[f"obs_count_{g}"]] = torch.tensor(counts[g], dtype=torch.float32)
    return x, idx


def make_restorer(idx: FakeIndices, composed: dict | None = None) -> ObsCountNaNRestorer:
    return ObsCountNaNRestorer(
        masked_by={f"sla_filtered_{g}": f"obs_count_{g}" for g in GROUPS},
        data_indices=idx,
        composed=composed,
    )


ONE_EACH = {
    "values": {"cryosat_2": [1.0] * 8, "jason_2": [3.0] * 8, "topex_poseidon": [5.0] * 8},
    # cells 0-1 seen by cryosat only, 2-3 by jason only, 4-5 by both, 6-7 by nobody
    "counts": {
        "cryosat_2": [1, 1, 0, 0, 1, 1, 0, 0],
        "jason_2": [0, 0, 1, 1, 3, 3, 0, 0],
        "topex_poseidon": [0] * 8,
    },
}


def test_forward_pass_is_a_no_op():
    """Training must not see the merge -- or the gaps."""
    x, idx = make_tensor(ONE_EACH["values"], ONE_EACH["counts"])
    before = x.clone()
    out = make_restorer(idx, SPEC)(x, inverse=False)
    assert torch.equal(out, before)


def test_model_output_tensors_pass_through_untouched():
    """Predictions stay dense; only the truth regains gaps."""
    x, idx = make_tensor(ONE_EACH["values"], ONE_EACH["counts"])
    pred = torch.full((1, 1, 1, GRID, 1), 0.5)
    out = make_restorer(idx, SPEC)(pred.clone(), inverse=True)
    assert torch.equal(out, pred)


def test_composed_truth_is_the_count_weighted_mean_of_live_missions():
    x, idx = make_tensor(ONE_EACH["values"], ONE_EACH["counts"])
    out = make_restorer(idx, SPEC)(x, inverse=True)
    sla = out[..., idx.columns["sla"]].reshape(-1)

    assert sla[0] == pytest.approx(1.0)  # cryosat only
    assert sla[2] == pytest.approx(3.0)  # jason only
    # both: (1*1 + 3*3) / (1 + 3)
    assert sla[4] == pytest.approx(2.5)
    assert torch.isnan(sla[6])  # nobody looked


def test_composed_coverage_is_the_union_not_one_mission():
    """The defect this replaced: a single-mission truth column is blank wherever that
    mission was not flying, however many others were."""
    counts = dict(ONE_EACH["counts"], jason_2=[0] * 8)  # the copied mission is absent
    x, idx = make_tensor(ONE_EACH["values"], counts)
    out = make_restorer(idx, SPEC)(x, inverse=True)
    seen = ~torch.isnan(out[..., idx.columns["sla"]].reshape(-1))

    assert seen.tolist() == [True, True, False, False, True, True, False, False]


def test_a_mission_that_never_flew_does_not_drag_the_mean():
    """All-zero counts must contribute nothing, not a zero-valued observation."""
    x, idx = make_tensor(ONE_EACH["values"], ONE_EACH["counts"])
    out = make_restorer(idx, SPEC)(x, inverse=True)
    # topex_poseidon has value 5.0 everywhere but zero counts throughout
    assert out[..., idx.columns["sla"]].reshape(-1)[0] == pytest.approx(1.0)


def test_gaps_in_a_mission_column_do_not_leak_into_the_composition():
    """The mask loop runs first and writes NaN into unobserved cells, and NaN*0 is NaN,
    so the composition has to clear them rather than rely on the weights."""
    x, idx = make_tensor(ONE_EACH["values"], ONE_EACH["counts"])
    out = make_restorer(idx, SPEC)(x, inverse=True)
    sla = out[..., idx.columns["sla"]].reshape(-1)
    assert not torch.isnan(sla[:6]).any(), sla


def test_mission_columns_still_get_their_gaps_back():
    x, idx = make_tensor(ONE_EACH["values"], ONE_EACH["counts"])
    out = make_restorer(idx, SPEC)(x, inverse=True)
    cryo = out[..., idx.columns["sla_filtered_cryosat_2"]].reshape(-1)
    assert torch.isnan(cryo).tolist() == [False, False, True, True, False, False, True, True]


def test_without_a_composed_spec_the_field_column_is_left_alone():
    """Backwards compatible: no `composed`, no merge."""
    x, idx = make_tensor(ONE_EACH["values"], ONE_EACH["counts"])
    out = make_restorer(idx, composed=None)(x, inverse=True)
    assert torch.equal(out[..., idx.columns["sla"]], torch.full((1, 1, 1, GRID), 99.0))
