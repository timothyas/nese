"""Read the encoder dimensions out of an anemoi inference checkpoint.

Writes two LaTeX fragments used by network_diagram.tex:

    dims.tex           \newcommand{\valNx}{49152} ...   (numbers only)
    symbols_table.tex  the tabular body for the notation table

The figure itself never sees a number -- it is drawn entirely with the symbol
macros in notation.tex.  Only this table is populated from the checkpoint.
"""

import argparse
import datetime
import textwrap
from pathlib import Path

import torch


def load_encoder_dims(ckpt_path, dataset=None):
    """Pull every dimension off the live modules in the checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ckpt.model

    names = list(model.encoder.keys())
    if dataset is None:
        if len(names) != 1:
            raise SystemExit(f"checkpoint has several datasets {names}; pass --dataset")
        dataset = names[0]

    enc = model.encoder[dataset]
    block = enc.proc
    node_attrs = model.node_attributes
    hidden = model._graph_name_hidden

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
        "Nout": model.num_output_channels[dataset],
    }

    # Static vs trainable split of the attribute widths, for the table.
    d["Naxstatic"] = node_attrs.trainable_tensors[dataset].trainable.shape[1]
    d["Nazstatic"] = node_attrs.trainable_tensors[hidden].trainable.shape[1]
    d["Naestatic"] = model.encoder_graph_provider[dataset].edge_attr.shape[1]

    # Parameter budget, for the note about how much capacity is pinned to the
    # graph rather than shared across it.
    d["encparams"] = sum(p.numel() for p in enc.parameters())
    d["nodeparams"] = sum(
        t.trainable.numel() for t in node_attrs.trainable_tensors.values() if t.trainable is not None
    )
    d["edgeparams"] = model.encoder_graph_provider[dataset].trainable.trainable.numel()
    d["graphparams"] = d["nodeparams"] + d["edgeparams"]
    d["totalparams"] = sum(p.numel() for p in model.parameters())
    d["graphpct"] = round(100 * d["graphparams"] / d["totalparams"])

    # The incoming neighbourhood N(z_j) is a property of the graph, not a stored
    # dimension, so summarise it here rather than typing numbers into the caption.
    dst = model.encoder_graph_provider[dataset].edge_index_base[1]
    degree = torch.bincount(dst, minlength=d["Nz"])
    d["degmin"] = degree.min().item()
    d["degmax"] = degree.max().item()
    d["degmean"] = f"{degree.float().mean().item():.1f}"
    d["degzero"] = int((degree == 0).sum().item())

    # multistep is not stored directly; it falls out of the input width.
    num, rem = divmod(d["Npsi"] - d["Nax"], d["Nu"])
    if rem or num < 1:
        raise SystemExit(
            f"cannot decompose input_dim={d['Npsi']} as N_T*{d['Nu']} + {d['Nax']}; "
            "the assumed input layout no longer holds"
        )
    d["NT"] = num

    assert d["Npsi"] == d["NT"] * d["Nu"] + d["Nax"]
    assert d["Nphi"] == d["NH"] * d["Nc"], "num_heads * head_dim != num_channels"
    return {k: (int(v) if isinstance(v, (int, torch.Tensor)) else v) for k, v in d.items()}


def write_dims(d, path, header):
    lines = [header, ""]
    for key, value in d.items():
        if key == "dataset":
            continue
        lines.append(rf"\newcommand{{\val{key}}}{{{value}}}")
    path.write_text("\n".join(lines) + "\n")


ROWS = [
    # (symbol macro, description, value expression)
    (r"\coordx", "data-space node coordinate", None),
    (r"\Nx", "number of data-space nodes", r"\valNx"),
    (r"\datavec", "data feature vector at one timestep", None),
    (r"\Nu", "number of data features per timestep", r"\valNu"),
    (r"\NT", "number of input timesteps ({\\tt multistep\\_input})", r"\valNT"),
    (r"\attrx", "data-node attributes (sin/cos lat-lon $\\|$ trainable)", None),
    (r"\Nax", "width of $\\attrx$", r"\valNaxsplit"),
    (r"\inputvec", r"assembled data-node input, $\inputvec = [\datavec_{t-\NT+1},\,\dots,\,\datavec_{t},\,\attrx]$", None),
    (r"\Npsi", r"width of $\inputvec$", r"\valNpsisplit"),
    (r"\coordz", "latent-space node coordinate", None),
    (r"\Nz", "number of latent-space nodes", r"\valNz"),
    (r"\attrz", "latent-node attributes (sin/cos lat-lon $\\|$ trainable)", None),
    (r"\Naz", r"width of $\attrz$", r"\valNazsplit"),
    (r"\latentvec", "latent state vector", None),
    (r"\Nphi", "number of latent channels ({\\tt num\\_channels})", r"\valNphi"),
    (r"\attre", "encoder edge attributes (length, direction $\\|$ trainable)", None),
    (r"\Nae", r"width of $\attre$", r"\valNaesplit"),
    (r"\NEE", "number of encoder edges (data $\\to$ latent)", r"\valNEE"),
    (r"\nbhd(\coordz_j)", r"data-space nodes with an edge into latent node $\coordz_j$", None),
    (r"\NH", "number of attention heads", r"\valNH"),
    (r"\Nc", r"channels per head, $\Nc = \Nphi/\NH$", r"\valNcsplit"),
    (r"\ff", r"MLP, $\ff:\mathbb{R}^{\Nphi}\to\mathbb{R}^{\Nphi}$, GELU", None),
    (r"\Nf", r"MLP hidden width ({\tt mlp\_hidden\_ratio})", r"\valNfsplit"),
]


def write_table(d, path, header):
    derived = [
        rf"\newcommand{{\valNaxsplit}}{{{d['Nax']} = {d['Nax'] - d['Naxstatic']} + {d['Naxstatic']}}}",
        rf"\newcommand{{\valNazsplit}}{{{d['Naz']} = {d['Naz'] - d['Nazstatic']} + {d['Nazstatic']}}}",
        rf"\newcommand{{\valNaesplit}}{{{d['Nae']} = {d['Naestatic']} + {d['Nae'] - d['Naestatic']}}}",
        rf"\newcommand{{\valNpsisplit}}{{$\NT\Nu + \Nax = "
        rf"{d['NT']}\cdot{d['Nu']} + {d['Nax']} = {d['Npsi']}$}}",
        rf"\newcommand{{\valNcsplit}}{{${d['Nphi']}/{d['NH']} = {d['Nc']}$}}",
        rf"\newcommand{{\valNfsplit}}{{$4\Nphi = {d['Nf']}$}}",
    ]

    body = []
    for symbol, desc, value in ROWS:
        cell = "--" if value is None else value
        body.append(rf"${symbol}$ & {desc} & {cell} \\")

    path.write_text(
        "\n".join([header, "", *derived, "", r"\newcommand{\symbolrows}{%", *body, "}", ""])
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--dataset", default=None, help="encoder key, if the model has several")
    ap.add_argument("-o", "--outdir", type=Path, default=Path(__file__).parent)
    args = ap.parse_args()

    d = load_encoder_dims(args.ckpt, args.dataset)
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
        N_x           {d['Nx']}
        N_z           {d['Nz']}
        N_EE          {d['NEE']}
        N_u           {d['Nu']}
        N_T           {d['NT']}
        N_ax          {d['Nax']}
        N_psi         {d['Npsi']} = {d['NT']}*{d['Nu']} + {d['Nax']}
        N_az          {d['Naz']}
        N_ae          {d['Nae']}
        N_phi         {d['Nphi']}
        N_H           {d['NH']}
        N_c           {d['Nc']}
        N_f           {d['Nf']}
        out channels  {d['Nout']}"""))
    print(f"\nwrote {args.outdir/'dims.tex'} and {args.outdir/'symbols_table.tex'}")


if __name__ == "__main__":
    main()
