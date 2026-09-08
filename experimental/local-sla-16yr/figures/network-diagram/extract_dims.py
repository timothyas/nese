"""Read the encoder and decoder dimensions out of an anemoi inference checkpoint.

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
        "Nout": model.num_output_channels[dataset],
        "NL": proc.num_layers,
        "Nw": proc_attn.window_size,
        "Nchunk": proc.num_chunks,
    }

    # Output channels split into prognostic (which take the residual add) and
    # diagnostic (which do not).  With no prognostic channel the residual add in
    # _assemble_output touches an empty index set and the whole output is emitted
    # by the decoder's extractor alone -- the point the decoder figure has to make,
    # so it is read off the checkpoint rather than asserted in a caption.
    d["Nprog"] = len(model._internal_output_idx[dataset])
    d["Ndiag"] = len(indices.model.output.diagnostic)
    diagnostic = set(indices.model.output.diagnostic.tolist())
    diagnames = [name for name, i in indices.model.output.name_to_index.items() if i in diagnostic]
    d["diagnamesplain"] = ", ".join(diagnames)
    d["diagnames"] = ", ".join(
        r"{\tt " + name.replace("_", r"\_") + "}" for name in diagnames
    )

    # Trainable share of the attribute widths, for the table.  `trainable` is None,
    # not an empty tensor, when trainable_size is 0.
    def trainable_width(obj):
        return 0 if obj.trainable is None else obj.trainable.shape[1]

    d["Naxtrain"] = trainable_width(node_attrs.trainable_tensors[dataset])
    d["Naztrain"] = trainable_width(node_attrs.trainable_tensors[hidden])
    d["Naetrain"] = trainable_width(model.encoder_graph_provider[dataset].trainable)

    # Parameter budget, for the note about how much capacity is pinned to the
    # graph rather than shared across it.
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
    d["graphpct"] = round(100 * d["graphparams"] / d["totalparams"])

    # The incoming neighbourhood N(z_j) is a property of the graph, not a stored
    # dimension, so summarise it here rather than typing numbers into the caption.
    for prefix, provider, num_dst in (
        ("", model.encoder_graph_provider, d["Nz"]),
        ("dec", model.decoder_graph_provider, d["Nx"]),
    ):
        dst = provider[dataset].edge_index_base[1]
        degree = torch.bincount(dst, minlength=num_dst)
        d[f"{prefix}degmin"] = degree.min().item()
        d[f"{prefix}degmax"] = degree.max().item()
        d[f"{prefix}degmean"] = f"{degree.float().mean().item():.1f}"
        d[f"{prefix}degzero"] = int((degree == 0).sum().item())

    # The processor's window is a band on the node *index*, and the hidden nodes are
    # HEALPix in ring order, so the window is a zonal strip -- all longitudes, a few
    # degrees of latitude.  How many degrees is a property of the graph, so measure it
    # rather than asserting it in a caption.
    latlons = dict(node_attrs.named_buffers())[f"latlons_{hidden}"]
    lat = torch.rad2deg(torch.atan2(latlons[:, 0], latlons[:, 2]))
    spans = [
        (lambda w: (w.max() - w.min()).item())(lat[max(0, j - d["Nw"]) : j + d["Nw"] + 1])
        for j in range(0, d["Nz"], 97)
    ]
    span = sorted(spans)[len(spans) // 2]
    d["windowlat"] = f"{span:.0f}"
    # One layer reaches half a window in each direction, so the whole stack reaches
    # N_L times that -- the meridional limit on how far information can travel.
    d["windowreach"] = f"{proc.num_layers * span / 2:.0f}"
    d["ringordered"] = bool(torch.all(lat[1:] - lat[:-1] <= 1e-4))

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
    assert d["Nout"] == d["Nprog"] + d["Ndiag"], "output channels are not prognostic + diagnostic"
    assert dec.node_data_extractor[-1].out_features == model.output_dim[dataset], (
        "the decoder's extractor no longer emits the model output width"
    )
    assert proc.num_channels == d["Nphi"], "the processor no longer runs at the latent width"
    assert proc_attn.num_heads * proc_attn.head_dim == d["Nphi"], (
        "the processor's attention is narrower than the latent width; the figure draws them equal"
    )
    assert d["Nw"] is not None, "the processor figure draws a sliding window; this one attends globally"
    assert d["ringordered"], (
        "hidden nodes are no longer in ring order, so the index window is not a zonal band"
    )
    assert not hasattr(dec, "emb_nodes_src"), (
        "the backward mapper has grown a source embedding; the decoder figure draws it without one"
    )
    return {k: (int(v) if isinstance(v, (int, torch.Tensor)) else v) for k, v in d.items()}


def write_dims(d, path, header):
    lines = [header, ""]
    for key, value in d.items():
        if key == "dataset":
            continue
        lines.append(rf"\newcommand{{\val{key}}}{{{value}}}")
    path.write_text("\n".join(lines) + "\n")


# The notation table is set as two tabulars side by side -- one table tall enough
# to overflow the page is silently shipped as an overfull box on a page of its
# own, so the split is structural, not cosmetic.  SPLIT marks the column break.
SPLIT = object()

ROWS = [
    # (symbol macro, description, value expression)
    (r"\coordx", "data-space node coordinate", None),
    (r"\Nx", "number of data-space nodes", r"\valNx"),
    (r"\datavec", "data feature vector at one timestep", None),
    (r"\Nu", "number of data features per timestep", r"\valNu"),
    (r"\NT", "number of input timesteps ({\\tt multistep\\_input})", r"\valNT"),
    (r"\attrx", r"data-node attributes (sin/cos lat-lon\valtrainabletrail)", None),
    (r"\Nax", "width of $\\attrx$", r"\valNaxsplit"),
    (r"\inputvec", r"assembled data-node input, $\inputvec = [\datavec_{t-\NT+1},\,\dots,\,\datavec_{t},\,\attrx]$", None),
    (r"\Npsi", r"width of $\inputvec$, $\NT\Nu + \Nax$", r"\valNpsi"),
    (r"\coordz", "latent-space node coordinate", None),
    (r"\Nz", "number of latent-space nodes", r"\valNz"),
    (r"\attrz", r"latent-node attributes (sin/cos lat-lon\valtrainabletrail)", None),
    (r"\Naz", r"width of $\attrz$", r"\valNazsplit"),
    (r"\latentvec", "latent state vector", None),
    (r"\Nphi", "number of latent channels ({\\tt num\\_channels})", r"\valNphi"),
    SPLIT,
    (r"\outvec", "model output in data space", None),
    (r"\Nout", r"number of output channels, $\Nout = \Nprog + \Ndiag$", r"\valNout"),
    (r"\Nprog", r"number of prognostic output channels ($\idxprog$)", r"\valNprog"),
    (r"\Ndiag", r"number of diagnostic output channels ($\idxdiag$)", r"\valNdiag"),
    (r"\attre", r"edge attributes (length, direction\valtrainabletrail)", None),
    (r"\Nae", r"width of $\attre$", r"\valNaesplit"),
    (r"\NEE", "number of encoder edges (data $\\to$ latent)", r"\valNEE"),
    (r"\NDE", "number of decoder edges (latent $\\to$ data)", r"\valNDE"),
    (r"\nbhd(\coordz_j)", r"data-space nodes with an edge into latent node $\coordz_j$", None),
    (r"\nbhd(\coordx_j)", r"latent-space nodes with an edge into data node $\coordx_j$", None),
    (r"\NH", "number of attention heads", r"\valNH"),
    (r"\Nc", r"channels per head, $\Nc = \Nphi/\NH$", r"\valNcsplit"),
    (r"\ff", r"MLP, $\ff:\mathbb{R}^{\Nphi}\to\mathbb{R}^{\Nphi}$, GELU", None),
    (r"\Nf", r"MLP hidden width ({\tt mlp\_hidden\_ratio})", r"\valNfsplit"),
    (r"\NL", r"processor layers ({\tt num\_layers})", r"\valNL"),
    (r"\Nw", r"attention window half-width, in latent nodes ({\tt window\_size})", r"\valNw"),
]



def write_table(d, path, header):
    # An attribute width with no trainable part is written as a bare number rather
    # than as "n = n + 0".
    def split(total, static, trainable):
        return f"{total}" if trainable == 0 else f"{total} = {static} + {trainable}"

    derived = [
        rf"\newcommand{{\valNaxsplit}}{{{split(d['Nax'], d['Nax'] - d['Naxtrain'], d['Naxtrain'])}}}",
        rf"\newcommand{{\valNazsplit}}{{{split(d['Naz'], d['Naz'] - d['Naztrain'], d['Naztrain'])}}}",
        rf"\newcommand{{\valNaesplit}}{{{split(d['Nae'], d['Nae'] - d['Naetrain'], d['Naetrain'])}}}",
        rf"\newcommand{{\valNcsplit}}{{${d['Nphi']}/{d['NH']} = {d['Nc']}$}}",
        rf"\newcommand{{\valNfsplit}}{{$4\Nphi = {d['Nf']}$}}",
    ]

    # "between 3 and 3" reads badly, and a k-nearest-neighbour graph gives exactly
    # that, so the decoder's degree sentence is phrased from the numbers.
    if d["decdegmin"] == d["decdegmax"]:
        phrase = rf"holds exactly {d['decdegmax']} latent nodes"
    else:
        phrase = (
            rf"holds between {d['decdegmin']} and {d['decdegmax']} latent nodes, "
            rf"{d['decdegmean']} on average"
        )
    derived.append(rf"\newcommand{{\valdecdegphrase}}{{{phrase}}}")

    # Whether the attributes carry a trainable tail is a config choice
    # (`trainable_parameters` / `trainable_size`), so the wording follows the
    # checkpoint rather than being fixed in the caption.
    if d["graphparams"]:
        derived += [
            r"\newcommand{\valtrainabletrail}{ $\|$ trainable}",
            r"\newcommand{\valtrainablenote}{The attribute widths split as"
            r" (sin/cos lat-lon) $\|$ (trainable), and the edge attribute width as"
            r" (length, direction) $\|$ (trainable); the trainable parts are"
            r" zero-initialised and are pinned to specific nodes and edges.}",
        ]
    else:
        derived += [
            r"\newcommand{\valtrainabletrail}{}",
            r"\newcommand{\valtrainablenote}{Every {\tt trainable\_size} is currently zero,"
            r" so the node and edge attributes are purely geometric and no"
            r" parameter is pinned to an individual node or edge.}",
        ]

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
        N_x           {d['Nx']}
        N_z           {d['Nz']}
        N_EE          {d['NEE']}
        N_DE          {d['NDE']}
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
        N_L / N_w     {d['NL']} layers / window {d['Nw']} nodes = {d['windowlat']} deg latitude
        out channels  {d['Nout']} = {d['Nprog']} prognostic + {d['Ndiag']} diagnostic ({d['diagnamesplain']})"""))
    print(f"\nwrote {args.outdir/'dims.tex'} and {args.outdir/'symbols_table.tex'}")


if __name__ == "__main__":
    main()
