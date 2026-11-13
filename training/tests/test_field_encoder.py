"""The field encoder must not care which, or how many, groups observe a field."""

import pytest
import torch
from anemoi.graphs.edges import KNNEdges
from anemoi.graphs.nodes import HEALPixNodes
from anemoi.models.data_indices.collection import IndexCollection
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from nese_training.field_encoder import FieldEncoder
from nese_training.field_encoder import FieldEncoderEncProcDec
from nese_training.field_encoder import resolve_field_columns

CHANNELS = 8
GROUPS = ["jason_2", "topex_poseidon", "topex_poseidon_interleaved"]
FORCINGS = ["elevation", "cos_latitude"]
SPEC = {"sla": {"observed_as": "sla_filtered", "groups": "auto", "count": "obs_count_{group}"}}


@pytest.fixture
def encoder() -> FieldEncoder:
    torch.manual_seed(0)
    return FieldEncoder({"sla": 1, "uvgxt": 2}, CHANNELS)


def observations(
    n_groups: int, observed: torch.Tensor | None = None, components: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.randn(2, 5, n_groups, components)
    counts = observed if observed is not None else torch.randint(0, 3, (2, 5, n_groups)).float()
    return values, counts


FIELDS = pytest.mark.parametrize(("field", "components"), [("sla", 1), ("uvgxt", 2)])


@FIELDS
def test_permuting_groups_leaves_latent_unchanged(encoder, field, components):
    values, counts = observations(4, torch.ones(2, 5, 4), components)
    order = torch.tensor([2, 0, 3, 1])
    torch.testing.assert_close(
        encoder.encode(field, values, counts), encoder.encode(field, values[..., order, :], counts[..., order])
    )


@FIELDS
def test_unobserved_group_has_no_effect(encoder, field, components):
    values, counts = observations(3, torch.tensor([1.0, 2.0, 0.0]).expand(2, 5, 3), components)
    changed = values.clone()
    changed[..., 2, :] = 100.0
    torch.testing.assert_close(encoder.encode(field, values, counts), encoder.encode(field, changed, counts))


@FIELDS
def test_adding_an_unobserved_group_leaves_latent_unchanged(encoder, field, components):
    values, counts = observations(2, torch.ones(2, 5, 2), components)
    extra_values = torch.cat((values, torch.randn(2, 5, 1, components)), dim=-2)
    extra_counts = torch.cat((counts, torch.zeros(2, 5, 1)), dim=-1)
    torch.testing.assert_close(encoder.encode(field, values, counts), encoder.encode(field, extra_values, extra_counts))


def test_latent_is_zero_where_nothing_observed(encoder):
    values, _ = observations(3)
    assert torch.all(encoder.encode("sla", values, torch.zeros(values.shape[:-1])) == 0)


def test_vector_latent_depends_on_both_components(encoder):
    values, counts = observations(2, torch.ones(2, 5, 2), components=2)
    assert not torch.allclose(encoder.encode("uvgxt", values, counts), encoder.encode("uvgxt", values.flip(-1), counts))


def test_latent_is_observed_mean_not_count_weighted(encoder):
    values, _ = observations(2)
    single = encoder.encode("sla", values, torch.tensor([1.0, 0.0]).expand(2, 5, 2))
    many = encoder.encode("sla", values, torch.tensor([7.0, 0.0]).expand(2, 5, 2))
    torch.testing.assert_close(single, many)


def names(groups: list[str] = GROUPS) -> list[str]:
    return (
        [f"sla_filtered_{group}" for group in groups]
        + [f"obs_count_{group}" for group in groups]
        + FORCINGS
        + ["sla"]
    )


def test_resolve_field_columns_separates_counts_from_passthrough():
    name_to_index = {name: i for i, name in enumerate(names()[:-1])}
    columns = resolve_field_columns(SPEC, name_to_index)
    values, counts = columns.fields["sla"]
    assert values == [[0], [1], [2]]
    assert counts == [3, 4, 5]
    assert columns.passthrough == [6, 7]


def test_resolve_field_columns_groups_vector_components():
    variables = [f"{component}_{group}" for component in ("ugxt", "vgxt") for group in GROUPS] + names()[:-1]
    name_to_index = {name: i for i, name in enumerate(variables)}
    columns = resolve_field_columns(SPEC | {"uvgxt": {"components": ["ugxt", "vgxt"]}}, name_to_index)
    values, counts = columns.fields["uvgxt"]
    assert values == [[0, 3], [1, 4], [2, 5]]
    assert counts == columns.fields["sla"][1]
    assert columns.passthrough == [12, 13]


def test_resolve_field_columns_requires_counts():
    name_to_index = {f"sla_filtered_{group}": i for i, group in enumerate(GROUPS)}
    with pytest.raises(ValueError, match="obs_count_jason_2"):
        resolve_field_columns(SPEC, name_to_index)


def toy_graph() -> HeteroData:
    graph = HEALPixNodes(resolution=2, name="data").update_graph(HeteroData())
    graph = HEALPixNodes(resolution=1, name="hidden").update_graph(graph)
    attributes = {"edge_length": {"_target_": "anemoi.graphs.edges.attributes.EdgeLength", "norm": "unit-max"}}
    for source, target in [("data", "hidden"), ("hidden", "hidden"), ("hidden", "data")]:
        graph = KNNEdges(source, target, num_nearest_neighbours=3).update_graph(graph, attributes)
    return graph.to("cpu")


def toy_model(
    n_step_input: int = 2, groups: list[str] = GROUPS, **model_options
) -> tuple[FieldEncoderEncProcDec, IndexCollection]:
    variables = names(groups)
    data_config = OmegaConf.create(
        {
            "forcing": variables[:-1],
            "diagnostic": ["sla"],
        }
    )
    indices = IndexCollection(data_config, {name: i for i, name in enumerate(variables)})
    mapper = {
        "num_chunks": 1,
        "mlp_hidden_ratio": 2,
        "num_heads": 2,
        "sub_graph_edge_attributes": ["edge_length"],
        "trainable_size": 0,
        "graph_attention_backend": "pyg",
        "layer_kernels": {},
    }
    config = OmegaConf.create(
        {
            "training": {"training_loss": {"datasets": {"data": {"observations": SPEC}}}},
            "model": {
                "num_channels": 16,
                "model": {
                    "_target_": "nese_training.field_encoder.FieldEncoderEncProcDec",
                    "hidden_nodes_name": "hidden",
                    "latent_skip": True,
                    "fields": "${training.training_loss.datasets.data.observations}",
                    "field_channels": CHANNELS,
                    **model_options,
                },
                "trainable_parameters": {"data": 0, "hidden": 0, "data2hidden": 0, "hidden2data": 0, "hidden2hidden": 0},
                "layer_kernels": {},
                "encoder": {"_target_": "anemoi.models.layers.mapper.GraphTransformerForwardMapper", **mapper},
                "decoder": {"_target_": "anemoi.models.layers.mapper.GraphTransformerBackwardMapper", **mapper},
                "processor": {
                    "_target_": "anemoi.models.layers.processor.GraphTransformerProcessor",
                    "num_layers": 1,
                    **mapper,
                },
                "residual": {"_target_": "anemoi.models.layers.residual.SkipConnection", "step": -1},
                "bounding": [],
            },
        }
    )
    model = FieldEncoderEncProcDec(
        model_config=config,
        data_indices={"data": indices},
        statistics={"data": {}},
        n_step_input=n_step_input,
        n_step_output=1,
        graph_data=toy_graph(),
    )
    return model, indices


def test_input_dim_counts_field_latents_not_groups():
    model, _ = toy_model(n_step_input=2)
    node_attributes = model.node_attributes.attr_ndims["data"]
    assert model.input_dim["data"] == 2 * (CHANNELS + len(FORCINGS)) + node_attributes


def test_forward_emits_the_diagnostic_on_every_node():
    model, indices = toy_model(n_step_input=2)
    grid = model._graph_data["data"].num_nodes
    x = torch.randn(1, 2, 1, grid, len(indices.model.input))

    y = model({"data": x})["data"]

    assert y.shape == (1, 1, 1, grid, 1)
    assert torch.isfinite(y).all()


def stack_inputs(indices: IndexCollection, columns: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([columns[name] for name in indices.model.input.name_to_index], dim=-1)


@pytest.mark.parametrize(
    "groups",
    [GROUPS + ["sentinel_3a"], GROUPS[:1], ["cryosat_2", "haiyang_2b"]],
    ids=["more", "fewer", "renamed"],
)
def test_weights_transfer_to_a_different_set_of_groups(groups):
    trained, _ = toy_model()
    other, indices = toy_model(groups=groups)
    other.load_state_dict(trained.state_dict())

    grid = other._graph_data["data"].num_nodes
    y = other({"data": torch.randn(1, 2, 1, grid, len(indices.model.input))})["data"]
    assert torch.isfinite(y).all()


def test_unobserved_extra_group_leaves_prediction_unchanged():
    trained, trained_indices = toy_model()
    extended, extended_indices = toy_model(groups=GROUPS + ["sentinel_3a"])
    extended.load_state_dict(trained.state_dict())

    grid = trained._graph_data["data"].num_nodes
    shape = (1, 2, 1, grid)
    columns = {name: torch.randn(shape) for name in names()}
    columns |= {f"obs_count_{group}": torch.randint(0, 3, shape).float() for group in GROUPS}
    extra = {"sla_filtered_sentinel_3a": torch.randn(shape), "obs_count_sentinel_3a": torch.zeros(shape)}

    trained.eval(), extended.eval()
    with torch.no_grad():
        expected = trained({"data": stack_inputs(trained_indices, columns)})["data"]
        actual = extended({"data": stack_inputs(extended_indices, columns | extra)})["data"]
    torch.testing.assert_close(actual, expected)


def test_resolve_field_columns_moves_static_out_of_passthrough():
    name_to_index = {name: i for i, name in enumerate(names()[:-1])}
    columns = resolve_field_columns(SPEC, name_to_index, static=["elevation"])
    assert columns.static == [6]
    assert columns.passthrough == [7]


@pytest.mark.parametrize("static", ["obs_count_jason_2", "not_an_input"])
def test_resolve_field_columns_rejects_misplaced_static(static):
    name_to_index = {name: i for i, name in enumerate(names()[:-1])}
    with pytest.raises(ValueError, match=static):
        resolve_field_columns(SPEC, name_to_index, static=[static])


def test_future_forcing_steps_must_leave_an_observed_step():
    with pytest.raises(ValueError, match="future_forcing_steps"):
        toy_model(n_step_input=2, future_forcing_steps=2)


KNOWN_FORCING = {"static_forcing": ["elevation"], "future_forcing_steps": 1}


def test_input_dim_counts_static_once_and_future_dynamic_forcings():
    model, _ = toy_model(n_step_input=3, **KNOWN_FORCING)
    node_attributes = model.node_attributes.attr_ndims["data"]
    assert model.input_dim["data"] == 2 * (CHANNELS + 1) + 1 + 1 + node_attributes


def test_default_options_reproduce_the_original_layout():
    torch.manual_seed(0)
    default, indices = toy_model()
    torch.manual_seed(0)
    explicit, _ = toy_model(static_forcing=[], future_forcing_steps=0)
    grid = default._graph_data["data"].num_nodes
    x = torch.randn(1, 2, 1, grid, len(indices.model.input))
    default.eval(), explicit.eval()
    with torch.no_grad():
        torch.testing.assert_close(explicit({"data": x})["data"], default({"data": x})["data"], rtol=0, atol=0)


def predict_perturbed(step: int, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Predictions from a 3-step window (t0-dt, t0, t0+dt) before and after perturbing one input."""
    torch.manual_seed(0)
    model, indices = toy_model(n_step_input=3, **KNOWN_FORCING)
    grid = model._graph_data["data"].num_nodes
    shape = (1, 3, 1, grid)
    columns = {name: torch.randn(shape) for name in names()}
    columns |= {f"obs_count_{group}": torch.ones(shape) for group in GROUPS}
    x = stack_inputs(indices, columns)
    perturbed = x.clone()
    perturbed[:, step, ..., indices.model.input.name_to_index[name]] += 10.0
    model.eval()
    with torch.no_grad():
        return model({"data": x})["data"], model({"data": perturbed})["data"]


@pytest.mark.parametrize("name", ["sla_filtered_jason_2", "obs_count_jason_2", "elevation"])
def test_future_slot_reads_no_observation_or_static(name):
    expected, actual = predict_perturbed(step=-1, name=name)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_static_forcing_is_read_from_t0_only():
    expected, actual = predict_perturbed(step=0, name="elevation")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(("step", "name"), [(-1, "cos_latitude"), (1, "elevation"), (1, "sla_filtered_jason_2")])
def test_known_inputs_reach_the_prediction(step, name):
    expected, actual = predict_perturbed(step=step, name=name)
    assert not torch.allclose(actual, expected)
