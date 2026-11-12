"""Write the network dimensions of an inference checkpoint to dims.tex and symbols_table.tex.

Adapted from ``../../local-sla-16yr/figures/network-diagram/extract_dims.py`` for a model with
the field encoder, static and future forcings, radius attention and physics diagnostics.
"""

import argparse
import datetime
import textwrap
from pathlib import Path

import torch


def gc_km(a, b):
    """Great-circle distance in km between two (lat, lon) tensors, degrees."""
    p1, p2 = torch.deg2rad(a), torch.deg2rad(b)
    dlat, dlon = p2[:, 0] - p1[:, 0], p2[:, 1] - p1[:, 1]
    h = (dlat / 2).sin() ** 2 + p1[:, 0].cos() * p2[:, 0].cos() * (dlon / 2).sin() ** 2
    return 2 * 6371.0 * torch.asin(h.clamp(max=1.0).sqrt())


def degrees(edge_index, num_dst, prefix):
    """Incoming-neighbourhood size per destination node, summarised."""
    degree = torch.bincount(edge_index[1], minlength=num_dst)
    return {
        f"{prefix}degmin": degree.min().item(),
        f"{prefix}degmax": degree.max().item(),
        f"{prefix}degmean": f"{degree.float().mean().item():.1f}",
        f"{prefix}degzero": int((degree == 0).sum().item()),
    }


def load_dims(ckpt_path, dataset=None):
    """Pull every dimension off the live modules in the checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ckpt.model

    names = list(model.encoder.keys())
    if dataset is None:
        if len(names) != 1:
            raise SystemExit(f"checkpoint has several datasets {names}; pass --dataset")
        dataset = names[0]

    enc = model.encoder[dataset]
    dec = model.decoder[dataset]
    proc = model.processor
    proc_attn = proc.proc[0].attention
    block = enc.proc
    node_attrs = model.node_attributes
    hidden = model._graph_name_hidden
    indices = model.data_indices[dataset]
    columns = model.field_columns[dataset]

    d = {
        "dataset": dataset,
        "Nx": node_attrs.num_nodes[dataset],
        "Nz": node_attrs.num_nodes[hidden],
        "Nax": node_attrs.attr_ndims[dataset],
        "Naz": node_attrs.attr_ndims[hidden],
        "Nu": model.num_input_channels[dataset],
        "Npsi": model.input_dim[dataset],
        "Nphi": block.lin_query.out_features,
        "NH": block.num_heads,
        "Nc": block.out_channels_conv,
        "Nae": block.lin_edge.in_features,
        "Nf": block.node_dst_mlp.mlp[0].out_features,
        "NEE": model.encoder_graph_provider[dataset].edge_index_base.shape[1],
        "NDE": model.decoder_graph_provider[dataset].edge_index_base.shape[1],
        "NPE": model.processor_graph_provider.edge_index_base.shape[1],
        "Nout": model.num_output_channels[dataset],
        "Nnet": model.output_dim[dataset],
        "NL": proc.num_layers,
        "Nchunk": proc.num_chunks,
    }

    d["NT"] = model.n_observed_steps
    d["NTf"] = model.future_forcing_steps
    d["Nfield"] = len(columns.fields)
    d["Nfc"] = model.field_channels
    d["Npass"] = len(columns.passthrough)
    d["Nstatic"] = len(columns.static)
    d["Nobs"] = sum(len(sum(values, [])) + len(counts) for values, counts in columns.fields.values())
    d["fieldnamesplain"] = ", ".join(columns.fields)
    d["fieldgroups"] = ", ".join(f"{field} {len(values)}" for field, (values, _) in columns.fields.items())

    name_of = {index: name for name, index in indices.model.input.name_to_index.items()}

    def tt_names(index_list):
        return ", ".join(r"{\tt " + name_of[i].replace("_", r"\_") + "}" for i in index_list)

    d["passnames"] = tt_names(columns.passthrough)
    d["staticnames"] = tt_names(columns.static)

    d["Nprog"] = len(model._internal_output_idx[dataset])
    d["Ndiag"] = len(indices.model.output.diagnostic)
    d["Nphys"] = d["Nout"] - d["Nnet"]
    output_name = {index: name for name, index in indices.model.output.name_to_index.items()}
    network = model._network_output_idx[dataset]
    d["netnamesplain"] = ", ".join(output_name[i] for i in network)
    d["physnamesplain"] = ", ".join(name for i, name in sorted(output_name.items()) if i not in network)

    def trainable_width(obj):
        return 0 if obj.trainable is None else obj.trainable.shape[1]

    d["Naxtrain"] = trainable_width(node_attrs.trainable_tensors[dataset])
    d["Naztrain"] = trainable_width(node_attrs.trainable_tensors[hidden])
    d["Naetrain"] = trainable_width(model.encoder_graph_provider[dataset].trainable)

    d["fieldparams"] = sum(p.numel() for p in model.field_encoder.parameters())
    d["encparams"] = sum(p.numel() for p in enc.parameters())
    d["decparams"] = sum(p.numel() for p in dec.parameters())
    d["procparams"] = sum(p.numel() for p in proc.parameters())
    d["proclayerparams"] = sum(p.numel() for p in proc.proc[0].parameters())
    d["nodeparams"] = sum(
        t.trainable.numel() for t in node_attrs.trainable_tensors.values() if t.trainable is not None
    )
    d["edgeparams"] = sum(
        gp.trainable.trainable.numel()
        for provider in (model.encoder_graph_provider, model.decoder_graph_provider)
        for gp in provider.values()
        if gp.trainable.trainable is not None
    )
    d["graphparams"] = d["nodeparams"] + d["edgeparams"]
    d["totalparams"] = sum(p.numel() for p in model.parameters())
    d["totalparamsM"] = f"{d['totalparams'] / 1e6:.2f}"
    d["graphpct"] = round(100 * d["graphparams"] / d["totalparams"])

    d |= degrees(model.encoder_graph_provider[dataset].edge_index_base, d["Nz"], "")
    d |= degrees(model.decoder_graph_provider[dataset].edge_index_base, d["Nx"], "dec")
    d |= degrees(model.processor_graph_provider.edge_index_base, d["Nz"], "proc")

    def latlon_deg(name):
        ll = dict(node_attrs.named_buffers())[f"latlons_{name}"]
        return torch.rad2deg(
            torch.stack([torch.atan2(ll[:, 0], ll[:, 2]), torch.atan2(ll[:, 1], ll[:, 3])], dim=1)
        )

    ll_hidden, ll_data = latlon_deg(hidden), latlon_deg(dataset)

    src, dst = model.encoder_graph_provider[dataset].edge_index_base
    d["enccutoffkm"] = f"{gc_km(ll_data[src], ll_hidden[dst]).max().item():.0f}"
    src, dst = model.processor_graph_provider.edge_index_base
    proc_km = gc_km(ll_hidden[src], ll_hidden[dst]).max().item()
    d["proccutoffkm"] = f"{proc_km:.0f}"
    d["procreachkm"] = f"{proc.num_layers * proc_km:.0f}"  # one stencil radius per layer

    layout = d["NT"] * (d["Nfield"] * d["Nfc"] + d["Npass"]) + d["NTf"] * d["Npass"] + d["Nstatic"] + d["Nax"]
    assert d["Npsi"] == layout, f"input_dim={d['Npsi']} is not the assumed layout ({layout})"
    assert d["Nu"] == d["Nobs"] + d["Npass"] + d["Nstatic"], "inputs are not fields + passthrough + static"
    assert d["Nphi"] == d["NH"] * d["Nc"], "num_heads * head_dim != num_channels"
    assert d["Nout"] == d["Nprog"] + d["Ndiag"], "output channels are not prognostic + diagnostic"
    assert dec.node_data_extractor[-1].out_features == d["Nnet"], (
        "the decoder's extractor no longer emits the network output width"
    )
    assert proc.num_channels == d["Nphi"], "the processor no longer runs at the latent width"
    assert proc_attn.num_heads * proc_attn.head_dim == d["Nphi"], (
        "the processor's attention is narrower than the latent width"
    )
    assert type(proc).__name__ == "RadiusTransformerProcessor", "the processor no longer attends by radius"
    assert not hasattr(dec, "emb_nodes_src"), "the backward mapper has grown a source embedding"
    return {k: (int(v) if isinstance(v, (int, torch.Tensor)) else v) for k, v in d.items()}


def write_dims(d, path, header):
    lines = [header, ""]
    for key, value in d.items():
        if key == "dataset":
            continue
        lines.append(rf"\newcommand{{\val{key}}}{{{value}}}")
    path.write_text("\n".join(lines) + "\n")


# Column break between the notation table's two side-by-side tabulars.
SPLIT = object()

ROWS = [
    # (symbol macro, description, value expression)
    (r"\coordx", "data-space node coordinate", None),
    (r"\Nx", "number of data-space nodes", r"\valNx"),
    (r"\Nu", "number of model input variables", r"\valNu"),
    (r"\Nfield", r"number of encoded fields (\valfieldnamesplain)", r"\valNfield"),
    (r"\Nfc", r"latent channels per field ({\tt field\_channels})", r"\valNfc"),
    (r"\Npass", r"dynamic forcings (\valpassnames)", r"\valNpass"),
    (r"\Nstatic", r"static forcings, once (\valstaticnames)", r"\valNstatic"),
    (r"\NT", r"observed input timesteps ({\tt multistep\_input})", r"\valNT"),
    (r"\NTf", r"forcing-only target timesteps ({\tt future\_forcing\_steps})", r"\valNTf"),
    (r"\attrx", r"data-node attributes (sin/cos lat-lon\valtrainabletrail)", None),
    (r"\Nax", "width of $\\attrx$", r"\valNaxsplit"),
    (r"\Npsi", r"input width, $\NT(\Nfield\Nfc + \Npass) + \NTf\Npass + \Nstatic + \Nax$", r"\valNpsi"),
    (r"\coordz", "latent-space node coordinate", None),
    (r"\Nz", "number of latent-space nodes", r"\valNz"),
    (r"\attrz", r"latent-node attributes (sin/cos lat-lon\valtrainabletrail)", None),
    (r"\Naz", r"width of $\attrz$", r"\valNazsplit"),
    (r"\Nphi", "number of latent channels ({\\tt num\\_channels})", r"\valNphi"),
    SPLIT,
    (r"\Nout", r"output channels, $\Nout = \Nprog + \Ndiag$", r"\valNout"),
    (r"\Nnet", r"of which emitted by the decoder (\valnetnamesplain)", r"\valNnet"),
    (r"\Nphys", r"of which diagnosed by physics (\valphysnamesplain)", r"\valNphys"),
    (r"\attre", r"edge attributes (length, direction\valtrainabletrail)", None),
    (r"\Nae", r"width of $\attre$", r"\valNaesplit"),
    (r"\NEE", "number of encoder edges (data $\\to$ latent)", r"\valNEE"),
    (r"\NPE", "number of processor edges (latent $\\to$ latent)", r"\valNPE"),
    (r"\NDE", "number of decoder edges (latent $\\to$ data)", r"\valNDE"),
    (r"\NH", "number of attention heads", r"\valNH"),
    (r"\Nc", r"channels per head, $\Nc = \Nphi/\NH$", r"\valNcsplit"),
    (r"\Nf", r"MLP hidden width ({\tt mlp\_hidden\_ratio})", r"\valNfsplit"),
    (r"\NL", r"processor layers ({\tt num\_layers})", r"\valNL"),
]


def write_table(d, path, header):
    def split(total, static, trainable):
        return f"{total}" if trainable == 0 else f"{total} = {static} + {trainable}"

    derived = [
        rf"\newcommand{{\valNaxsplit}}{{{split(d['Nax'], d['Nax'] - d['Naxtrain'], d['Naxtrain'])}}}",
        rf"\newcommand{{\valNazsplit}}{{{split(d['Naz'], d['Naz'] - d['Naztrain'], d['Naztrain'])}}}",
        rf"\newcommand{{\valNaesplit}}{{{split(d['Nae'], d['Nae'] - d['Naetrain'], d['Naetrain'])}}}",
        rf"\newcommand{{\valNcsplit}}{{${d['Nphi']}/{d['NH']} = {d['Nc']}$}}",
        rf"\newcommand{{\valNfsplit}}{{${d['Nf'] // d['Nphi']}\Nphi = {d['Nf']}$}}",
    ]
    trail = r" $\|$ trainable" if d["graphparams"] else ""
    derived.append(rf"\newcommand{{\valtrainabletrail}}{{{trail}}}")

    columns = [[]]
    for row in ROWS:
        if row is SPLIT:
            columns.append([])
            continue
        symbol, desc, value = row
        cell = "--" if value is None else value
        columns[-1].append(rf"${symbol}$ & {desc} & {cell} \\")

    macros = []
    for name, body in zip(("symbolrowsone", "symbolrowstwo"), columns):
        macros += ["", "\\newcommand{\\" + name + "}{%", *body, "}"]

    path.write_text("\n".join([header, "", *derived, *macros, ""]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--dataset", default=None, help="encoder key, if the model has several")
    ap.add_argument("-o", "--outdir", type=Path, default=Path(__file__).parent)
    args = ap.parse_args()

    d = load_dims(args.ckpt, args.dataset)
    header = (
        f"% GENERATED by extract_dims.py -- do not edit.\n"
        f"% source: {args.ckpt}\n"
        f"% dataset: {d['dataset']}\n"
        f"% date: {datetime.date.today().isoformat()}"
    )
    write_dims(d, args.outdir / "dims.tex", header)
    write_table(d, args.outdir / "symbols_table.tex", header)

    print(textwrap.dedent(f"""\
        dataset       {d['dataset']}
        N_x / N_z     {d['Nx']} data / {d['Nz']} latent nodes
        edges         {d['NEE']} enc / {d['NPE']} proc / {d['NDE']} dec
        N_u           {d['Nu']} = {d['Nobs']} field + {d['Npass']} dynamic + {d['Nstatic']} static
        fields        {d['fieldgroups']} groups, {d['Nfc']} channels each
        N_psi         {d['Npsi']} = {d['NT']}*({d['Nfield']}*{d['Nfc']} + {d['Npass']}) + {d['NTf']}*{d['Npass']} + {d['Nstatic']} + {d['Nax']}
        N_phi / N_H   {d['Nphi']} / {d['NH']} heads of {d['Nc']}
        processor     {d['NL']} layers, stencil {d['proccutoffkm']} km, {d['procdegmean']} neighbours
        out channels  {d['Nout']} = {d['Nnet']} decoder ({d['netnamesplain']}) + {d['Nphys']} physics ({d['physnamesplain']})
        params        {d['totalparamsM']} M (field encoder {d['fieldparams']})"""))
    print(f"\nwrote {args.outdir/'dims.tex'} and {args.outdir/'symbols_table.tex'}")


if __name__ == "__main__":
    main()
