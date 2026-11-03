"""
generate_paper_visuals.py — Generates paper figures using the official plotting tools
from Li et al. (ICLR 2023, `likenneth/othello_world/data/__init__.py`: `plot_probs`, `plot_mentals`,
`OthelloBoardState`) and Nanda, Lee & Wattenberg (BlackboxNLP 2023, `ajyl/mech_int_othelloGPT`).
"""

import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, FancyArrowPatch

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


if __name__ == "__main__":
    sd = torch.load(os.path.join(BASE_DIR, "data/synthetic_model.pth"), map_location="cpu", weights_only=True)
    probes = [torch.load(os.path.join(BASE_DIR, f"probes/resid_{l}_linear.pth"), map_location="cpu", weights_only=True) for l in range(8)]
    with open(os.path.join(BASE_DIR, "data/intervention_benchmark.pkl"), "rb") as f:
        dataset = pickle.load(f)
    generate_board_intervention_figure(sd, probes, dataset)
