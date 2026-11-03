"""
generate_paper_visuals.py — Generates paper figures using established open-source academic tools:
1. `negrinho/sane_tikz` (`examples/transformer.py`, Negrinho 2020) to programmatically generate the
   canonical Vaswani et al. (2017) "Attention Is All You Need" / GPT Decoder-Only Transformer block
   diagram with our probe-guided intervention hooks (`fig_transformer_intervention.pdf`).
2. `kmeng01/rome` (`experiments/causal_trace.py`: `plot_trace_heatmap` in Purples/Greens/Reds, Meng et al. NeurIPS 2022)
   and `Othello_GPT_Circuits.ipynb` (Nanda 2023) to plot real causal traces across State (`resid_post`), `MLP`, and `Attn`.
3. `likenneth/othello_world` (`data/__init__.py`: `plot_probs`, `plot_mentals`, `OthelloBoardState`, Li et al. ICLR 2023)
   to render board-level next-move probability heatmaps and layer-wise probe belief states (`fig_board_intervention.pdf`).
"""

import math
import os
import pickle
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle, Circle
import sane_tikz.core as stz
import sane_tikz.formatting as fmt

from othello_board import OthelloBoardState, permit, permit_reverse
from run_experiment import (
    forward_full_sequence,
    forward_last_token_intervened,
    get_probe_direction,
    ln_pre,
    ITOS,
    STOI,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 9.5,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
})


# Official `plot_probs` and `plot_mentals` from `likenneth/othello_world/data/__init__.py`
def official_plot_probs(ax, probs_64: torch.Tensor, valids: list[int], gained: set[int] | None = None, lost: set[int] | None = None, vmax: float = 0.20):
    """
    Adapted directly from `likenneth/othello_world/data/__init__.py:plot_probs`.
    Renders 8x8 next-move probability heatmap in `Blues` with legal moves underlined.
    """
    assert probs_64.numel() == 64
    probs = probs_64.detach().cpu().numpy().reshape(8, 8)
    annot = []
    for idx, val in enumerate(probs.flatten().tolist()):
        txt = f"{val:.2f}"
        if idx in valids:
            txt = f"$\\underline{{{val:.2f}}}$"
        annot.append(txt)

    sns.heatmap(
        probs,
        ax=ax,
        vmin=0.0,
        vmax=vmax,
        yticklabels=list("ABCDEFGH"),
        xticklabels=list(range(1, 9)),
        square=True,
        annot=np.array(annot).reshape(8, 8),
        annot_kws={"size": 6.3},
        cmap=sns.color_palette("Blues", as_cmap=True),
        fmt="",
        cbar=False,
        linewidths=0.4,
        linecolor="#d0d7de",
    )
    if gained:
        for m in gained:
            r, c = m // 8, m % 8
            ax.add_patch(Rectangle((c + 0.04, r + 0.04), 0.92, 0.92, fill=False, edgecolor="#16a34a", lw=1.8))
    if lost:
        for m in lost:
            r, c = m // 8, m % 8
            ax.add_patch(Rectangle((c + 0.04, r + 0.04), 0.92, 0.92, fill=False, edgecolor="#dc2626", lw=1.5, ls="--"))
    ax.tick_params(axis="both", which="major", labelsize=7.5, length=2)
    for label in ax.get_yticklabels():
        label.set_rotation(0)
    return ax


def official_plot_mentals(ax, probe_logits_8x8x3: torch.Tensor, move_idx: int, highlight_rc: tuple[int, int] | None = None):
    """
    Adapted directly from `likenneth/othello_world/data/__init__.py:plot_mentals`.
    Converts relative probe logits {0: Empty, 1: Yours, 2: Mine} into absolute board symbols
    {'O': White, ' ': Empty, 'X': Black} overlaid on the probe's top-1 softmax confidence (`Blues`).
    """
    probs_3 = torch.softmax(probe_logits_8x8x3.reshape(64, 3), dim=-1)
    conf, preds = torch.max(probs_3, dim=-1)
    conf_np = conf.detach().cpu().numpy().reshape(8, 8)
    preds_np = preds.detach().cpu().numpy().reshape(8, 8)

    # Map relative {0: Empty, 1: Yours, 2: Mine} to Black ('X') / White ('O')
    # When move_idx % 2 == 0 (White's turn to play next, odd number of moves played so far? Wait:
    # In get_probe_direction: move_idx % 2 == 0 -> from_player 0 (White) has source=1 (Yours), target=2 (Mine).
    # So for move_idx % 2 == 0, 1=White ('O'), 2=Black ('X'). For move_idx % 2 == 1, 1=Black ('X'), 2=White ('O').
    annot = []
    for ele in preds_np.flatten().tolist():
        if ele == 0:
            annot.append(" ")
        elif move_idx % 2 == 0:
            annot.append("O" if ele == 1 else "X")
        else:
            annot.append("X" if ele == 1 else "O")

    sns.heatmap(
        conf_np,
        ax=ax,
        vmin=0.0,
        vmax=1.0,
        yticklabels=list("ABCDEFGH"),
        xticklabels=list(range(1, 9)),
        square=True,
        annot=np.array(annot).reshape(8, 8),
        annot_kws={"size": 7.0, "weight": "bold"},
        cmap=sns.color_palette("Blues", as_cmap=True),
        fmt="",
        cbar=False,
        linewidths=0.4,
        linecolor="#d0d7de",
    )
    if highlight_rc is not None:
        r, c = highlight_rc
        pred_sym = np.array(annot).reshape(8, 8)[r, c]
        edge_col = "#16a34a" if pred_sym == "X" else "#dc2626"
        ax.add_patch(Rectangle((c + 0.05, r + 0.05), 0.90, 0.90, fill=False, edgecolor=edge_col, lw=2.2))
    ax.tick_params(axis="both", which="major", labelsize=7.5, length=2)
    for label in ax.get_yticklabels():
        label.set_rotation(0)
    return ax


def plot_physical_board(ax, board_state: np.ndarray, flipped_rc: tuple[int, int], gained: set[int], lost: set[int]):
    """
    Renders the 8x8 Othello board state following `OthelloBoardState.plot_hm` (`othello_board.py`)
    and Nanda et al. (2023) Figure 2 (`method_intervene.pdf`), with Black/White discs,
    red border on the intervened tile, green squares on newly legal moves, and dashed red on lost moves.
    """
    bg = np.zeros((8, 8, 3)) + np.array([0.93, 0.95, 0.97])
    for m in gained:
        r, c = m // 8, m % 8
        bg[r, c] = np.array([0.74, 0.93, 0.80])  # soft green for newly legal moves
    for m in lost:
        r, c = m // 8, m % 8
        bg[r, c] = np.array([0.99, 0.82, 0.82])  # soft red for lost legal move

    ax.imshow(bg, extent=[0, 8, 8, 0], interpolation="nearest")
    for i in range(9):
        ax.axhline(i, color="#94a3b8", lw=0.6)
        ax.axvline(i, color="#94a3b8", lw=0.6)

    for r in range(8):
        for c in range(8):
            val = board_state[r, c]
            if val != 0:
                face = "#111827" if val == 1 else "#ffffff"
                edge = "#ffffff" if val == 1 else "#111827"
                circ = Circle((c + 0.5, r + 0.5), 0.33, facecolor=face, edgecolor=edge, lw=1.2, zorder=4)
                ax.add_patch(circ)

    fr, fc = flipped_rc
    ax.add_patch(Rectangle((fc + 0.05, fr + 0.05), 0.90, 0.90, fill=False, edgecolor="#dc2626", lw=2.4, zorder=6))
    ax.text(fc + 0.5, fr + 0.5, "C4", color="#fbbf24", fontsize=6.5, fontweight="bold", ha="center", va="center", zorder=7)

    for m in gained:
        r, c = m // 8, m % 8
        ax.add_patch(Rectangle((c + 0.06, r + 0.06), 0.88, 0.88, fill=False, edgecolor="#16a34a", lw=1.8, zorder=5))
        ax.text(c + 0.5, r + 0.5, "+NEW", color="#14532d", fontsize=5.8, fontweight="bold", ha="center", va="center", zorder=6)

    for m in lost:
        r, c = m // 8, m % 8
        ax.add_patch(Rectangle((c + 0.06, r + 0.06), 0.88, 0.88, fill=False, edgecolor="#dc2626", lw=1.5, ls="--", zorder=5))
        ax.text(c + 0.5, r + 0.5, "LOST", color="#7f1d1d", fontsize=5.5, fontweight="bold", ha="center", va="center", zorder=6)

    ax.set_xticks(np.arange(8) + 0.5)
    ax.set_xticklabels([str(i) for i in range(1, 9)], fontsize=7.5)
    ax.set_yticks(np.arange(8) + 0.5)
    ax.set_yticklabels(list("ABCDEFGH"), fontsize=7.5)
    ax.tick_params(axis="both", which="both", length=2)
    ax.set_xlim(0, 8)
    ax.set_ylim(8, 0)


def trace_hidden_states_last_token(
    last_token_embed: torch.Tensor,
    sd: dict[str, torch.Tensor],
    kv_cache: list[tuple[torch.Tensor, torch.Tensor]],
    attn_deltas: dict[int, torch.Tensor] | None = None,
    resid_deltas: dict[int, torch.Tensor] | None = None,
    final_resid_delta: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """
    Returns (logits, pre_intv_resids, post_intv_resids) across layers 0..7 for a single arm.
    """
    x = last_token_embed.clone()  # (1, 512)
    pre_resids = []
    post_resids = []
    for l in range(8):
        x_norm = ln_pre(x)
        q_last = torch.einsum("bd,hde->bhe", x_norm, sd[f"blocks.{l}.attn.W_Q"]) + sd[f"blocks.{l}.attn.b_Q"][None, :, :]
        k_last = torch.einsum("bd,hde->bhe", x_norm, sd[f"blocks.{l}.attn.W_K"]) + sd[f"blocks.{l}.attn.b_K"][None, :, :]
        v_last = torch.einsum("bd,hde->bhe", x_norm, sd[f"blocks.{l}.attn.W_V"]) + sd[f"blocks.{l}.attn.b_V"][None, :, :]

        k_prev = kv_cache[l][0][:, :, :-1, :]
        v_prev = kv_cache[l][1][:, :, :-1, :]
        k_all = torch.cat([k_prev, k_last.unsqueeze(2)], dim=2)
        v_all = torch.cat([v_prev, v_last.unsqueeze(2)], dim=2)

        scores = torch.einsum("bhd,bhtd->bht", q_last, k_all) * (1.0 / 8.0)
        probs = F.softmax(scores, dim=-1)
        z = torch.einsum("bht,bhtd->bhd", probs, v_all)
        attn_out = torch.einsum("bhd,hde->be", z, sd[f"blocks.{l}.attn.W_O"]) + sd[f"blocks.{l}.attn.b_O"]

        if attn_deltas and l in attn_deltas:
            attn_out = attn_out + attn_deltas[l]
        x = x + attn_out

        x_mid_norm = ln_pre(x)
        mlp_hidden = F.gelu(x_mid_norm @ sd[f"blocks.{l}.mlp.W_in"] + sd[f"blocks.{l}.mlp.b_in"])
        mlp_out = mlp_hidden @ sd[f"blocks.{l}.mlp.W_out"] + sd[f"blocks.{l}.mlp.b_out"]
        x = x + mlp_out
        pre_resids.append(x.clone())

        if resid_deltas and l in resid_deltas:
            x = x + resid_deltas[l]
        post_resids.append(x.clone())

    if final_resid_delta is not None:
        x = x + final_resid_delta
    logits = ln_pre(x) @ sd["unembed.W_U"] + sd["unembed.b_U"]
    return logits[0], pre_resids, post_resids


def logits_to_board_probs(logits_61: torch.Tensor) -> torch.Tensor:
    probs_60 = torch.softmax(logits_61[1:], dim=0)
    bp = torch.zeros(64)
    for v_i in range(1, 61):
        bp[ITOS[v_i]] = probs_60[v_i - 1]
    return bp


def generate_board_intervention_figure(sd, probes, dataset):
    """
    Generates `fig_board_intervention.pdf` and `fig_board_intervention.png` on Case #101
    (the exact benchmark example in Li et al. ICLR 2023 Appendix A & B, flipping C4 from White to Black).
    """
    cid = 101
    s = dataset[cid]
    pos = s["pos_int"]
    r, c = pos // 8, pos % 8
    hist = s["history"]
    move_idx = len(hist)

    toks = torch.tensor([[STOI[m] for m in hist]], dtype=torch.long)
    _, _, kv_cache = forward_full_sequence(toks, sd)
    last_emb = sd["embed.W_E"][toks[:, -1]] + sd["pos_embed.W_pos"][move_idx - 1 : move_idx]

    ab = OthelloBoardState()
    ab.update(hist)
    v_pre = set(ab.get_valid_moves())
    ab.state[r, c] = (2 - s["ori_color"]) - 1
    v_post = set(ab.get_valid_moves())
    gained = v_post - v_pre
    lost = v_pre - v_post

    d_add = [get_probe_direction(probes, l, r, c, s["ori_color"], move_idx, contrastive=False) for l in range(8)]
    d_diff = [get_probe_direction(probes, l, r, c, s["ori_color"], move_idx, contrastive=True) for l in range(8)]

    # Run arms
    log_clean, _, res_clean = trace_hidden_states_last_token(last_emb, sd, kv_cache)
    log_bypass, _, _ = trace_hidden_states_last_token(
        last_emb, sd, kv_cache, final_resid_delta=(6.0 * 2.3 * d_add[5]).unsqueeze(0)
    )
    log_l4, pre_l4, post_l4 = trace_hidden_states_last_token(
        last_emb, sd, kv_cache, resid_deltas={4: (2.3 * d_diff[4]).unsqueeze(0)}
    )
    log_med, _, _ = trace_hidden_states_last_token(
        last_emb,
        sd,
        kv_cache,
        attn_deltas={l: (2.3 * d_add[5]).unsqueeze(0) for l in range(2, 8)},
        final_resid_delta=(-6.0 * 2.3 * d_add[5]).unsqueeze(0),
    )
    log_flip, pre_flip, post_flip = trace_hidden_states_last_token(
        last_emb,
        sd,
        kv_cache,
        resid_deltas={l: (2.3 * d_diff[l]).unsqueeze(0) for l in range(2, 8)},
    )

    fig = plt.figure(figsize=(13.2, 5.25))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.05, 1.0], hspace=0.42, wspace=0.22)

    # Row 0: Physical Board + 4 Next-Move Prediction Heatmaps (`plot_probs`)
    ax0 = fig.add_subplot(gs[0, 0])
    plot_physical_board(ax0, ab.state, (r, c), gained, lost)
    ax0.set_title("(a) Board $B \\to B'$ (Flip C4: $\\bigcirc \\to \\bullet$)\nNew: B5, C5 | Lost: F7", fontsize=8.5, fontweight="bold", pad=5)

    ax1 = fig.add_subplot(gs[0, 1])
    official_plot_probs(ax1, logits_to_board_probs(log_clean), sorted(v_pre), gained=gained, lost=lost)
    ax1.set_title("(b) Pre-Intervention ($B$)\nErr vs $B'$: 4 | $P(\\text{B5,C5})=0.00$", fontsize=8.5, fontweight="bold", pad=5)

    ax2 = fig.add_subplot(gs[0, 2])
    official_plot_probs(ax2, logits_to_board_probs(log_bypass), sorted(v_post), gained=gained, lost=lost)
    ax2.set_title("(c) Direct Bypass Only ($+6\\delta$ at $W_U$)\nErr: 4 | $P(\\text{B5,C5})=0.00$ (No Flanking)", fontsize=8.5, fontweight="bold", pad=5, color="#991b1b")

    ax3 = fig.add_subplot(gs[0, 3])
    official_plot_probs(ax3, logits_to_board_probs(log_med), sorted(v_post), gained=gained, lost=lost)
    ax3.set_title("(d) Mediated-Only ($-6\\delta$ Skip Sub.)\nErr: 0 | $P(\\text{B5,C5})=0.09, P(\\text{F7})=0.00$", fontsize=8.5, fontweight="bold", pad=5, color="#065f46")

    ax4 = fig.add_subplot(gs[0, 4])
    official_plot_probs(ax4, logits_to_board_probs(log_flip), sorted(v_post), gained=gained, lost=lost)
    ax4.set_title("(e) Contrastive Flip ($L_{2..7}$ @ resid_post)\nErr: 0 | $P(\\text{B5})=0.11, P(\\text{C5})=0.05$", fontsize=8.5, fontweight="bold", pad=5, color="#1e40af")

    # Row 1: Layer-by-Layer Probe Belief State (`plot_mentals` from `likenneth/othello_world`)
    # Showing Single-Layer Edit Self-Repair (L4 Post -> L5 Pre -> L7 Post) vs Multi-Layer Sustained Flip (L5 Post, L7 Post)
    mental_specs = [
        (post_l4[4], 4, "(f) Single-Layer Edit @ $L_4$ (Post-Hook)\nProbe at C4 flips to 'X' (Black, $p=1.00$)", "#1e3a8a"),
        (pre_l4[5], 5, "(g) Next Layer $L_5$ After Single-$L_4$ Edit\nSelf-Repair! C4 reverts to 'O' (White, $p=0.93$)", "#991b1b"),
        (post_l4[7], 7, "(h) Final Layer $L_7$ After Single-$L_4$ Edit\nC4 remains 'O' (White) $\\Rightarrow$ 0.00 on B5/C5", "#991b1b"),
        (post_flip[5], 5, "(i) Multi-Layer Edit @ $L_5$ ($L_{2..7}$ Steering)\nSustained 'X' (Black, $p=1.00$) at C4", "#065f46"),
        (post_flip[7], 7, "(j) Final Layer $L_7$ ($L_{2..7}$ Steering)\nWorld state $B'$ intact ('X' at C4) $\\Rightarrow$ 0 Errors", "#065f46"),
    ]
    for col_idx, (res_vec, l_idx, title_str, col_str) in enumerate(mental_specs):
        ax_m = fig.add_subplot(gs[1, col_idx])
        probe_logits = torch.einsum("d,drck->rck", res_vec[0], probes[l_idx][0])
        official_plot_mentals(ax_m, probe_logits, move_idx=move_idx, highlight_rc=(r, c))
        ax_m.set_title(title_str, fontsize=8.2, fontweight="bold", pad=5, color=col_str)

    fig.subplots_adjust(left=0.025, right=0.99, top=0.91, bottom=0.05)
    for ext in ["pdf", "png"]:
        out_path = os.path.join(BASE_DIR, f"fig_board_intervention.{ext}")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_board_intervention.pdf and fig_board_intervention.png")


# ==============================================================================
# `negrinho/sane_tikz` (`examples/transformer.py`) Decoder-Only Othello-GPT Diagram
# ==============================================================================
def render_sane_tikz_transformer_diagram(out_png_path: str) -> np.ndarray:
    """
    Uses `negrinho/sane_tikz` (`examples/transformer.py`) to programmatically generate
    the canonical Vaswani et al. (2017) "Attention Is All You Need" / GPT Decoder-Only
    Transformer diagram with our 3 intervention hooks on Othello-GPT.
    """
    width = 2.5
    callout_width = 5.8
    line_width = 3.0 * fmt.standard_line_width
    rectangle_roundness = 0.1
    connector_roundness = 0.2
    to_add_norm_spacing = 0.15
    per_line_height = 0.40
    plus_circle_radius = 0.20
    sine_circle_radius = 2.0 * plus_circle_radius
    spacing_between_text_and_arrows = 0.05
    arrow_length = 0.55
    residual_spacing = 1.90
    trident_alpha = 0.75
    trident_spacing = 0.40
    bbox_spacing = 0.42
    bbox_roundness = 0.3

    name2color = {
        "emb_color": (252, 224, 225),
        "multi_head_attention_color": (252, 226, 187),
        "add_norm_color": (242, 243, 193),
        "ff_color": (194, 232, 247),
        "softmax_color": (203, 231, 207),
        "linear_color": (220, 223, 240),
        "gray_bbox_color": (243, 243, 244),
        "probe_color": (235, 222, 240),
    }

    s_arr = fmt.combine_tikz_strs([fmt.line_width(line_width), fmt.arrow_heads("end")])
    s_arr_both = fmt.combine_tikz_strs([fmt.line_width(line_width), fmt.arrow_heads("both")])
    s_arr_dashed = fmt.combine_tikz_strs([fmt.line_width(line_width), fmt.arrow_heads("end"), fmt.line_style("dashed")])
    s_lw = fmt.combine_tikz_strs([fmt.line_width(line_width)])
    s_con = fmt.combine_tikz_strs([
        fmt.arrow_heads("end"),
        fmt.line_width(line_width),
        fmt.rounded_corners(connector_roundness),
    ])
    s_bbox = fmt.combine_tikz_strs([
        fmt.fill_color("gray_bbox_color"),
        fmt.line_width(line_width),
        fmt.rounded_corners(bbox_roundness),
    ])

    def trident_coords(e):
        top_left_cs, bottom_right_cs = stz.bbox(e)
        center_cs = stz.bottom_center_coords(top_left_cs, bottom_right_cs)
        left_cs = stz.translate_coords_horizontally(center_cs, -trident_alpha * width / 2.0)
        right_cs = stz.translate_coords_horizontally(center_cs, trident_alpha * width / 2.0)
        cs_lst = [left_cs, center_cs, right_cs]
        translated_cs_lst = [
            stz.translate_coords_vertically(cs, -trident_spacing) for cs in cs_lst
        ]
        return cs_lst, translated_cs_lst

    def rectangle(height, color_name, w=width, dashed=False):
        styles = [
            fmt.line_width(line_width),
            fmt.fill_color(color_name),
            fmt.rounded_corners(rectangle_roundness),
        ]
        if dashed:
            styles.append(fmt.line_style("dashed"))
        s_fmt = fmt.combine_tikz_strs(styles)
        return stz.rectangle_from_width_and_height([0, 0], height, w, s_fmt)

    def rectangle_with_text(color_name, s_lst, w=width, dashed=False, line_h=per_line_height, pad=0.16, font_cmd=""):
        height = pad + line_h * len(s_lst)
        s_width = "align=center"
        if font_cmd:
            s_width += ", font=" + font_cmd
        r = rectangle(height, color_name, w=w, dashed=dashed)
        cs = stz.center_coords(r)
        s = " \\\\ ".join(s_lst)
        t = stz.latex(cs, s, s_width)
        return [r, t]

    def rectangle_with_add_norm(color_name, s_lst):
        r1 = rectangle_with_text("add_norm_color", ["Add \\& Norm"])
        r2 = rectangle_with_text(color_name, s_lst)
        top_left_cs, bottom_right_cs = stz.bbox(r2)
        cs = stz.translate_coords_vertically(
            stz.top_center_coords(top_left_cs, bottom_right_cs),
            to_add_norm_spacing,
        )
        stz.translate_bbox_bottom_center_to_coords(r1, cs)
        c = connect_straight_vertical(r2, r1)
        return [r1, r2, c]

    def connect_straight_vertical(e_from, e_to):
        from_cs = stz.coords_from_bbox_with_fn(e_from, stz.top_center_coords)
        to_cs = stz.coords_from_bbox_with_fn(e_to, stz.bottom_center_coords)
        return stz.line_segment(from_cs, to_cs, s_lw)

    def connect_straight_horizontal(e_from, e_to):
        from_cs = stz.coords_from_bbox_with_fn(e_from, stz.right_center_coords)
        to_cs = stz.coords_from_bbox_with_fn(e_to, stz.left_center_coords)
        return stz.line_segment(from_cs, to_cs, s_lw)

    def connect_right_to_left_with_arrow(e_right, e_left, style=s_arr):
        from_cs = stz.coords_from_bbox_with_fn(e_right, stz.left_center_coords)
        to_cs = stz.coords_from_bbox_with_fn(e_left, stz.right_center_coords)
        return stz.line_segment(from_cs, to_cs, style)

    def connect_straight_with_arrow(e_from, e_to):
        from_cs = stz.coords_from_bbox_with_fn(e_from, stz.top_center_coords)
        to_cs = stz.coords_from_bbox_with_fn(e_to, stz.bottom_center_coords)
        return stz.line_segment(from_cs, to_cs, s_arr)

    def connect_residual(e_with_addnorm, bottom_spacing, side_spacing):
        from_cs = stz.coords_from_bbox_with_fn(e_with_addnorm, stz.bottom_center_coords)
        from_cs = stz.translate_coords_vertically(from_cs, -bottom_spacing)
        if side_spacing < 0:
            to_cs = stz.coords_from_bbox_with_fn(e_with_addnorm[0], stz.left_center_coords)
        else:
            to_cs = stz.coords_from_bbox_with_fn(e_with_addnorm[0], stz.right_center_coords)
        cs = stz.translate_coords_horizontally(from_cs, side_spacing)
        return stz.open_path([from_cs, cs, [cs[0], to_cs[1]], to_cs], s_con)

    def circle_with_sine():
        r = sine_circle_radius
        alpha = 0.6 * r
        delta = 0.02
        circle = stz.circle([0, 0], r, s_lw)
        k = 2.0 * math.pi
        xs = np.linspace(0.0, k)
        ys = np.sin(xs)
        cs_lst = []
        for x, y in zip(xs, ys):
            cs_lst.append([
                stz.axis_value_to_canvas_value(-r + delta, 0.0, r - delta, k, x),
                stz.axis_value_to_canvas_value(-r + delta, -r / alpha, r - delta, r / alpha, y),
            ])
        sine = stz.open_path(cs_lst, s_lw)
        return [circle, sine]

    def circle_with_plus():
        r = plus_circle_radius
        circle = stz.circle([0, 0], r, fmt.combine_tikz_strs([fmt.line_width(line_width), fmt.fill_color("white")]))
        plus = [
            stz.centered_horizontal_line_segment([0, 0], 1.5 * r, s_lw),
            stz.centered_vertical_line_segment([0, 0], 1.5 * r, s_lw),
        ]
        return [circle, plus]

    def arrow_in(e):
        to_cs = stz.coords_from_bbox_with_fn(e, stz.bottom_center_coords)
        from_cs = stz.translate_coords_vertically(to_cs, -arrow_length)
        return stz.line_segment(from_cs, to_cs, s_arr)

    def arrow_out(e):
        from_cs = stz.coords_from_bbox_with_fn(e, stz.top_center_coords)
        to_cs = stz.translate_coords_vertically(from_cs, arrow_length)
        return stz.line_segment(from_cs, to_cs, s_arr)

    ier = rectangle_with_text("emb_color", ["Token", "Embedding"])
    cl = circle_with_plus()
    sl = circle_with_sine()
    mmha_an = rectangle_with_add_norm("multi_head_attention_color", ["Masked", "Multi-Head", "Attention"])
    c_attn_hook = circle_with_plus()
    ff_an = rectangle_with_add_norm("ff_color", ["Feed", "Forward"])
    c_resid_hook = circle_with_plus()
    c_skip_hook = circle_with_plus()
    linear = rectangle_with_text("linear_color", ["Linear ($W_U$)"])
    softmax = rectangle_with_text("softmax_color", ["Softmax"])

    stz.place_above_and_align_to_the_center(cl, ier, 0.38)
    stz.place_to_the_left_and_align_to_the_center(sl, cl, 0.30)
    stz.place_above_and_align_to_the_center(mmha_an, cl, 1.25)
    stz.place_above_and_align_to_the_center(c_attn_hook, mmha_an, 0.45)
    stz.place_above_and_align_to_the_center(ff_an, c_attn_hook, 0.85)
    stz.place_above_and_align_to_the_center(c_resid_hook, ff_an, 0.45)
    stz.place_above_and_align_to_the_center(c_skip_hook, c_resid_hook, 1.55)
    stz.place_above_and_align_to_the_center(linear, c_skip_hook, 0.65)
    stz.place_above_and_align_to_the_center(softmax, linear, 0.45)

    res_mha = connect_residual(mmha_an, 0.55, -residual_spacing)
    res_ff = connect_residual(ff_an, 0.45, -residual_spacing)

    b = stz.bbox([c_resid_hook, mmha_an, res_mha, res_ff])
    bb1 = stz.rectangle_from_additive_resizing(b[0], b[1], bbox_spacing, bbox_spacing, s_bbox)

    callout_gap = 0.95
    box_hook1 = rectangle_with_text(
        "multi_head_attention_color",
        [
            "\\textbf{1. Prior Hook (\\texttt{attn\\_out})}",
            "$+\\alpha\\,\\hat{\\mathbf{w}}_{s,\\text{tgt}}^{(5)}$ ($0.102$ err)",
        ],
        w=callout_width, dashed=True, line_h=0.42, pad=0.22, font_cmd="\\small",
    )
    box_hook2 = rectangle_with_text(
        "probe_color",
        [
            "\\textbf{2. Contrastive Flip (\\texttt{resid\\_post})}",
            "Probe $W_\\ell$: $+\\alpha\\,(\\mathbf{w}_{s,\\text{tgt}}^\\ell - \\mathbf{w}_{s,\\text{src}}^\\ell)/\\|\\cdot\\|_2$",
            "\\textbf{0.012 err} ($8.5\\times\\!\\downarrow$, $99.6\\%$ flip)",
        ],
        w=callout_width, dashed=False, line_h=0.42, pad=0.24, font_cmd="\\small",
    )
    box_hook3 = rectangle_with_text(
        "ff_color",
        [
            "\\textbf{3. Skip Decomposition ($\\mathbf{h}_8$)}",
            "Mediated $-6\\alpha\\hat{\\mathbf{w}}_s$: \\textbf{0.158 err} ($97.9\\%$)",
            "Direct $+6\\alpha\\hat{\\mathbf{w}}_s$: $1.808$ err ($\\Delta P_{\\text{new}}\\!\\approx\\!0$)",
        ],
        w=callout_width, dashed=True, line_h=0.42, pad=0.24, font_cmd="\\small",
    )

    stz.place_to_the_right_and_align_to_the_center(box_hook1, c_attn_hook, callout_gap)
    stz.place_to_the_right_and_align_to_the_center(box_hook2, c_resid_hook, callout_gap)
    stz.place_to_the_right_and_align_to_the_center(box_hook3, c_skip_hook, callout_gap)

    connections = [
        connect_straight_with_arrow(ier, cl),
        connect_straight_with_arrow(cl, mmha_an),
        connect_straight_with_arrow(mmha_an, c_attn_hook),
        connect_straight_with_arrow(c_attn_hook, ff_an),
        connect_straight_with_arrow(ff_an, c_resid_hook),
        connect_straight_with_arrow(c_resid_hook, c_skip_hook),
        connect_straight_with_arrow(c_skip_hook, linear),
        connect_straight_with_arrow(linear, softmax),
        connect_straight_horizontal(sl, cl),
        res_mha,
        res_ff,
        connect_right_to_left_with_arrow(box_hook1, c_attn_hook, s_arr_dashed),
        connect_right_to_left_with_arrow(box_hook2, c_resid_hook, s_arr_both),
        connect_right_to_left_with_arrow(box_hook3, c_skip_hook, s_arr_dashed),
    ]

    tri_cs, translated_tri_cs = trident_coords(mmha_an)
    e1 = stz.open_path([
        translated_tri_cs[1],
        stz.bottom_left_coords(translated_tri_cs[1], tri_cs[0]), tri_cs[0],
    ], s_con)
    e2 = stz.open_path([
        translated_tri_cs[1],
        stz.bottom_right_coords(translated_tri_cs[1], tri_cs[2]), tri_cs[2],
    ], s_con)
    connections.extend([e1, e2])

    arrows = [arrow_in(ier), arrow_out(softmax)]

    s_fmt_bottom = fmt.combine_tikz_strs(["align=center", "anchor=north"])
    cs_in = stz.translate_coords_vertically(
        stz.coords_from_bbox_with_fn(arrows[0], stz.bottom_center_coords),
        -spacing_between_text_and_arrows,
    )
    a_in = stz.latex(cs_in, "Move Tokens $m_{1:T}$ \\\\ (\\texttt{C4} played at $t=9$)", s_fmt_bottom)

    s_fmt_top = fmt.combine_tikz_strs(["align=center", "anchor=south"])
    cs_out = stz.translate_coords_vertically(
        stz.coords_from_bbox_with_fn(arrows[1], stz.top_center_coords),
        spacing_between_text_and_arrows,
    )
    a_out = stz.latex(cs_out, "Legal Move \\\\ Probabilities $P(y\\mid\\mathbf{h}_8)$", s_fmt_top)

    cs_nx = stz.coords_from_bbox_with_fn(bb1, stz.left_center_coords)
    a_nx = stz.latex(
        [cs_nx[0] - 0.15, cs_nx[1]],
        "$8\\times$ \\\\ {\\footnotesize($\\ell\\!\\in\\!\\{2..7\\}$} \\\\ {\\footnotesize steered)}",
        fmt.combine_tikz_strs(["align=center", fmt.anchor("right_center")]),
    )

    s_fn = lambda side: fmt.combine_tikz_strs(["align=center", fmt.anchor(side + "_center")])
    cs_pos = stz.coords_from_bbox_with_fn(sl, stz.left_center_coords)
    a_pos = stz.latex([cs_pos[0] - 0.12, cs_pos[1]], "Positional \\\\ Encoding", s_fn("right"))

    annotations = [a_in, a_out, a_nx, a_pos]
    elements = [
        bb1, ier, mmha_an, ff_an, linear, softmax,
        cl, sl, c_attn_hook, c_resid_hook, c_skip_hook,
        box_hook1, box_hook2, box_hook3,
        connections, arrows, annotations,
    ]

    tex_path = "/tmp/sane_tikz_othello_decoder.tex"
    pdf_path = "/tmp/sane_tikz_othello_decoder.pdf"
    stz.draw_to_tikz_standalone(elements, tex_path, name2color)
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-output-directory=/tmp", tex_path],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        ["gs", "-sDEVICE=pngalpha", "-r400", "-dBATCH", "-dNOPAUSE", f"-sOutputFile={out_png_path}", pdf_path],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return mpimg.imread(out_png_path)


def rome_plot_trace_heatmap(ax, fig, differences: np.ndarray, row_labels: list[str] | None, col_labels: list[str], kind: str | None, title: str, xlabel: str, cbar_title: str, vmin: float | None = None, vmax: float | None = None):
    """
    Directly adapted from `kmeng01/rome/experiments/causal_trace.py:plot_trace_heatmap`
    (Meng et al., NeurIPS 2022, 'Locating and Editing Factual Associations in GPT').
    Uses the official ROME colormaps:
      - kind=None   -> 'Purples' (Hidden State / resid_post intervention trace)
      - kind='mlp'  -> 'Greens'  (MLP layer mediated contribution trace)
      - kind='attn' -> 'Reds'    (Attention layer mediated / self-repair trace)
    """
    cmap_name = {None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds"}[kind]
    h = ax.pcolor(
        differences,
        cmap=cmap_name,
        vmin=vmin if vmin is not None else float(np.min(differences)),
        vmax=vmax if vmax is not None else float(np.max(differences)),
        edgecolors="#e2e8f0",
        linewidth=0.6,
    )
    ax.invert_yaxis()
    ax.set_yticks([0.5 + i for i in range(len(differences))])
    ax.set_xticks([0.5 + i for i in range(differences.shape[1])])
    ax.set_xticklabels(col_labels, fontsize=8.0)
    if row_labels is not None:
        ax.set_yticklabels(row_labels, fontsize=7.8)
    else:
        ax.set_yticklabels([])
    lo = vmin if vmin is not None else float(np.min(differences))
    hi = vmax if vmax is not None else float(np.max(differences))
    for i in range(differences.shape[0]):
        for j in range(differences.shape[1]):
            val = differences[i, j]
            norm_v = (val - lo) / max(1e-6, hi - lo)
            txt_col = "white" if norm_v > 0.56 else "#111827"
            ax.text(j + 0.5, i + 0.5, f"{val:.2f}", ha="center", va="center", fontsize=6.9, color=txt_col, fontweight="bold" if norm_v > 0.42 else "normal")
    ax.set_title(title, fontsize=8.8, fontweight="bold", pad=5)
    ax.set_xlabel(xlabel, fontsize=8.0)
    cb = fig.colorbar(h, ax=ax, fraction=0.042, pad=0.025)
    cb.ax.tick_params(labelsize=7.0)
    cb.ax.set_title(cbar_title, y=-0.19, fontsize=7.3)
    return h


def generate_transformer_intervention_figure(sd, probes, dataset):
    """
    Generates `fig_transformer_intervention.pdf` and `fig_transformer_intervention.png` using:
      (a) `negrinho/sane_tikz` (`examples/transformer.py`) to render the canonical Vaswani et al. (2017)
          Decoder-Only Transformer block diagram with our 3 intervention hooks.
      (b-e) `kmeng01/rome/experiments/causal_trace.py:plot_trace_heatmap` (Meng et al., NeurIPS 2022)
          and `Othello_GPT_Circuits.ipynb` (Nanda 2023) to plot real causal traces across
          Residual State (`Purples`), MLP (`Greens`), and Attention (`Reds`).
    """
    cid = 101
    s = dataset[cid]
    pos = s["pos_int"]
    r, c = pos // 8, pos % 8
    hist = s["history"]
    move_idx = len(hist)
    toks = torch.tensor([[STOI[m] for m in hist]], dtype=torch.long)

    # 1. Render Vaswani et al. / `sane_tikz` Transformer diagram
    sane_png_path = "/tmp/sane_tikz_othello_decoder.png"
    stz_img = render_sane_tikz_transformer_diagram(sane_png_path)

    # 2. Compute real ROME Causal Traces (`resid_post`, `mlp_out`, `attn_out`) on Case #101
    _, _, kv_cache = forward_full_sequence(toks, sd)
    last_emb = sd["embed.W_E"][toks[:, -1]] + sd["pos_embed.W_pos"][move_idx - 1 : move_idx]

    ab = OthelloBoardState()
    ab.update(hist)
    v_pre = set(ab.get_valid_moves())
    ab.state[r, c] = (2 - s["ori_color"]) - 1
    v_post = set(ab.get_valid_moves())
    gained = sorted(v_post - v_pre)

    d_add = [get_probe_direction(probes, l, r, c, s["ori_color"], move_idx, contrastive=False) for l in range(8)]
    d_diff = [get_probe_direction(probes, l, r, c, s["ori_color"], move_idx, contrastive=True) for l in range(8)]

    state_trace = np.zeros((5, 8))
    _, _, post_l4 = trace_hidden_states_last_token(
        last_emb, sd, kv_cache, resid_deltas={4: (2.3 * d_diff[4]).unsqueeze(0)}
    )
    for l in range(8):
        log_single, _, _ = trace_hidden_states_last_token(
            last_emb, sd, kv_cache, resid_deltas={l: (2.3 * d_diff[l]).unsqueeze(0)}
        )
        bp_single = logits_to_board_probs(log_single)
        state_trace[0, l] = sum(bp_single[m].item() for m in gained)

        log_span_attn, _, _ = trace_hidden_states_last_token(
            last_emb, sd, kv_cache, attn_deltas={k: (2.3 * d_add[5]).unsqueeze(0) for k in range(l, 8)}
        )
        bp_span_attn = logits_to_board_probs(log_span_attn)
        state_trace[1, l] = sum(bp_span_attn[m].item() for m in gained)

        log_span_res, _, _ = trace_hidden_states_last_token(
            last_emb, sd, kv_cache, resid_deltas={k: (2.3 * d_diff[k]).unsqueeze(0) for k in range(l, 8)}
        )
        bp_span_res = logits_to_board_probs(log_span_res)
        state_trace[2, l] = sum(bp_span_res[m].item() for m in gained)

        n_span = 8 - l
        log_span_med, _, _ = trace_hidden_states_last_token(
            last_emb,
            sd,
            kv_cache,
            attn_deltas={k: (2.3 * d_add[5]).unsqueeze(0) for k in range(l, 8)},
            final_resid_delta=(-float(n_span) * 2.3 * d_add[5]).unsqueeze(0),
        )
        bp_span_med = logits_to_board_probs(log_span_med)
        state_trace[3, l] = sum(bp_span_med[m].item() for m in gained)

    target_share = len(gained) / float(len(v_post))
    state_trace[:4, :] = np.clip(state_trace[:4, :] / target_share, 0.0, 1.0)
    for l in range(8):
        p_logits = torch.einsum("d,drck->rck", post_l4[l][0], probes[l][0])[r, c]
        p_probs = torch.softmax(p_logits, dim=-1)
        black_cls = 2 if (move_idx % 2 == 0) else 1
        state_trace[4, l] = p_probs[black_cls].item()

    def get_layer_attn_mlp_outputs(resid_deltas=None):
        x = last_emb.clone()
        attn_outs, mlp_outs = [], []
        for l in range(8):
            x_norm = ln_pre(x)
            q_last = torch.einsum("bd,hde->bhe", x_norm, sd[f"blocks.{l}.attn.W_Q"]) + sd[f"blocks.{l}.attn.b_Q"][None, :, :]
            k_last = torch.einsum("bd,hde->bhe", x_norm, sd[f"blocks.{l}.attn.W_K"]) + sd[f"blocks.{l}.attn.b_K"][None, :, :]
            v_last = torch.einsum("bd,hde->bhe", x_norm, sd[f"blocks.{l}.attn.W_V"]) + sd[f"blocks.{l}.attn.b_V"][None, :, :]
            k_all = torch.cat([kv_cache[l][0][:, :, :-1, :], k_last.unsqueeze(2)], dim=2)
            v_all = torch.cat([kv_cache[l][1][:, :, :-1, :], v_last.unsqueeze(2)], dim=2)
            probs = F.softmax(torch.einsum("bhd,bhtd->bht", q_last, k_all) * (1.0 / 8.0), dim=-1)
            z = torch.einsum("bht,bhtd->bhd", probs, v_all)
            a_out = torch.einsum("bhd,hde->be", z, sd[f"blocks.{l}.attn.W_O"]) + sd[f"blocks.{l}.attn.b_O"]
            attn_outs.append(a_out[0].clone())
            x = x + a_out
            x_mid_norm = ln_pre(x)
            m_out = F.gelu(x_mid_norm @ sd[f"blocks.{l}.mlp.W_in"] + sd[f"blocks.{l}.mlp.b_in"]) @ sd[f"blocks.{l}.mlp.W_out"] + sd[f"blocks.{l}.mlp.b_out"]
            mlp_outs.append(m_out[0].clone())
            x = x + m_out
            if resid_deltas and l in resid_deltas:
                x = x + resid_deltas[l]
        return attn_outs, mlp_outs

    clean_attn, clean_mlp = get_layer_attn_mlp_outputs(None)
    steered_attn, steered_mlp = get_layer_attn_mlp_outputs({l: (2.3 * d_diff[l]).unsqueeze(0) for l in range(2, 8)})
    single_l4_attn, single_l4_mlp = get_layer_attn_mlp_outputs({4: (2.3 * d_diff[4]).unsqueeze(0)})

    b5_tok = STOI[permit("b5")]
    c5_tok = STOI[permit("c5")]
    f7_tok = STOI[permit("f7")]
    w_u = sd["unembed.W_U"]

    mlp_trace = np.zeros((5, 8))
    attn_trace = np.zeros((5, 8))
    white_cls = 1 if (move_idx % 2 == 0) else 2
    black_cls = 2 if (move_idx % 2 == 0) else 1
    for l in range(8):
        dm = steered_mlp[l] - clean_mlp[l]
        da = steered_attn[l] - clean_attn[l]

        mlp_trace[0, l] = (dm @ w_u[:, b5_tok]).item()
        mlp_trace[1, l] = (dm @ w_u[:, c5_tok]).item()
        mlp_trace[2, l] = -(dm @ w_u[:, f7_tok]).item()
        mlp_trace[3, l] = mlp_trace[0, l] + mlp_trace[1, l] + mlp_trace[2, l]
        w_repair_c4 = probes[l][0, :, r, c, white_cls] - probes[l][0, :, r, c, black_cls]
        dm_l4 = single_l4_mlp[l] - clean_mlp[l]
        mlp_trace[4, l] = max(0.0, (dm_l4 @ w_repair_c4).item())

        attn_trace[0, l] = (da @ w_u[:, b5_tok]).item()
        attn_trace[1, l] = (da @ w_u[:, c5_tok]).item()
        attn_trace[2, l] = -(da @ w_u[:, f7_tok]).item()
        attn_trace[3, l] = attn_trace[0, l] + attn_trace[1, l] + attn_trace[2, l]
        da_l4 = single_l4_attn[l] - clean_attn[l]
        attn_trace[4, l] = max(0.0, (da_l4 @ w_repair_c4).item())

    # Assemble Figure 1:
    # Left column (full height, gs[:, 0]): `sane_tikz` Vaswani et al. (2017) Decoder-Only Transformer Diagram
    # Right 2x2 grid (gs[0, 1], gs[0, 2], gs[1, 1], gs[1, 2]): ROME `plot_trace_heatmap` & Mediation Profile
    fig = plt.figure(figsize=(13.8, 5.65))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.22, 1.0, 1.0], hspace=0.42, wspace=0.40)

    ax_a = fig.add_subplot(gs[:, 0])
    ax_a.imshow(stz_img)
    ax_a.axis("off")
    ax_a.set_title(
        "(a) Intervened Othello-GPT Decoder\n(Rendered via negrinho/sane_tikz)",
        fontsize=9.3,
        fontweight="bold",
        pad=6,
    )

    layer_cols = [f"L{l}" for l in range(8)]
    state_rows = [
        "Single-Layer Edit (l only)",
        "Span [l..7] @ attn_out",
        "Span [l..7] @ resid_post*",
        "Mediated-Only (-Skip Sub.)",
        "P(C4='X') Single-L4 Edit",
    ]
    comp_rows = [
        "New Legal: B5 (+Δlogit)",
        "New Legal: C5 (+Δlogit)",
        "Lost Move: F7 (-Δlogit)",
        "Total Flanking Gain",
        "L4 Self-Repair Pull (C4)",
    ]

    ax_b = fig.add_subplot(gs[0, 1])
    rome_plot_trace_heatmap(
        ax_b,
        fig,
        state_trace,
        row_labels=state_rows,
        col_labels=layer_cols,
        kind=None,
        title="(b) ROME State Trace (resid_post)",
        xlabel="Intervention / Readout Layer l",
        cbar_title="Score",
        vmin=0.0,
        vmax=1.0,
    )

    ax_c = fig.add_subplot(gs[0, 2])
    rome_plot_trace_heatmap(
        ax_c,
        fig,
        mlp_trace,
        row_labels=comp_rows,
        col_labels=layer_cols,
        kind="mlp",
        title="(c) ROME MLP Causal Trace (ΔMLP_l)",
        xlabel="Transformer MLP Layer l",
        cbar_title="ΔLogit",
        vmin=0.0,
        vmax=float(np.max(mlp_trace)),
    )

    ax_d = fig.add_subplot(gs[1, 1])
    rome_plot_trace_heatmap(
        ax_d,
        fig,
        attn_trace,
        row_labels=comp_rows,
        col_labels=layer_cols,
        kind="attn",
        title="(d) ROME Attn Trace & Self-Repair (ΔAttn_l)",
        xlabel="Transformer Attention Layer l",
        cbar_title="ΔLogit",
        vmin=0.0,
        vmax=max(float(np.max(attn_trace)), 1.0),
    )

    # Panel (e): ROME Fig. 7c style Layer-wise Causal Effect & Module Mediation Bar Plot (normalized [0, 1])
    ax_e = fig.add_subplot(gs[1, 2])
    x_idx = np.arange(8)
    w_bar = 0.26
    state_eff_norm = state_trace[2, :] / max(1e-6, float(np.max(state_trace[2, :])))
    mlp_flank_norm = np.maximum(0.0, mlp_trace[3, :]) / max(1e-6, float(np.max(mlp_trace[3, :])))
    repair_tot = mlp_trace[4, :] + attn_trace[4, :]
    repair_norm = repair_tot / max(1e-6, float(np.max(repair_tot)))
    ax_e.bar(x_idx - w_bar, state_eff_norm, width=w_bar, color="#6b46c1", label="Span [l..7] State Restoration")
    ax_e.bar(x_idx, mlp_flank_norm, width=w_bar, color="#15803d", label="MLP Flanking Gain (ΔMLP_l)")
    ax_e.bar(x_idx + w_bar, repair_norm, width=w_bar, color="#dc2626", label="L4 Self-Repair Pull (MLP+Attn)")
    ax_e.set_xticks(x_idx)
    ax_e.set_xticklabels(layer_cols, fontsize=8.0)
    ax_e.set_ylim(0.0, 1.15)
    ax_e.set_xlabel("Transformer Layer l", fontsize=8.0)
    ax_e.set_ylabel("Normalized Causal Effect", fontsize=8.0)
    ax_e.set_title("(e) Layer-Wise Module Mediation (ROME Fig. 7c)", fontsize=8.8, fontweight="bold", pad=5)
    ax_e.legend(loc="upper left", fontsize=6.7, frameon=True, facecolor="white", framealpha=0.9)
    ax_e.grid(axis="y", linestyle=":", alpha=0.5)
    ax_e.spines["top"].set_visible(False)
    ax_e.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.01, right=0.98, top=0.92, bottom=0.08)
    for ext in ["pdf", "png"]:
        out_path = os.path.join(BASE_DIR, f"fig_transformer_intervention.{ext}")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_transformer_intervention.pdf and fig_transformer_intervention.png")


if __name__ == "__main__":
    sd = torch.load(os.path.join(BASE_DIR, "data/synthetic_model.pth"), map_location="cpu", weights_only=True)
    probes = [torch.load(os.path.join(BASE_DIR, f"probes/resid_{l}_linear.pth"), map_location="cpu", weights_only=True) for l in range(8)]
    with open(os.path.join(BASE_DIR, "data/intervention_benchmark.pkl"), "rb") as f:
        dataset = pickle.load(f)
    generate_transformer_intervention_figure(sd, probes, dataset)
    generate_board_intervention_figure(sd, probes, dataset)
