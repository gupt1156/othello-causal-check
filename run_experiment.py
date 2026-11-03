"""
run_experiment.py — Reproducible Probing & Causal Intervention Audit on Othello-GPT.

Uses exact causal KV-caching for prefix tokens 0..T-2 (since interventions only modify
the final token position T-1 and causal masking prevents earlier tokens from attending
to T-1), running all 25 intervention arms + alpha sweeps across all N = 1,001 official
benchmark boards in ~10 seconds on CPU/GPU.
"""

import json
import math
import os
import pickle
import random
import time
import numpy as np
import torch
import torch.nn.functional as F

from othello_rules import (
    OthelloBoardState,
    permit,
    permit_reverse,
    ITOS,
    STOI,
    generate_synthetic_game,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(max(1, os.cpu_count() or 8))


def set_all_seeds(seed: int = 44):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ln_pre(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + eps)


def make_untrained_sd(ref_sd: dict[str, torch.Tensor], seed: int = 44) -> dict[str, torch.Tensor]:
    """Create a randomly initialized 8-layer GPT state dict with matched layer norms/scales."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    rand_sd = {}
    for k, v in ref_sd.items():
        if "mask" in k or "IGNORE" in k:
            rand_sd[k] = v.clone()
        elif v.ndim >= 2:
            std = float(v.std().item())
            rand_sd[k] = torch.randn(v.shape, generator=gen, dtype=v.dtype) * std
        else:
            rand_sd[k] = torch.zeros_like(v)
    return rand_sd


def forward_full_sequence(
    tokens: torch.Tensor,
    sd: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]]]:
    """
    Run clean forward pass on (1, T) tokens, returning:
    - clean logits at last token (1, 61)
    - resid_posts: list of 8 tensors of shape (1, T, 512)
    - kv_cache: list of 8 tuples (K_full, V_full) each of shape (1, 8, T, 64)
    """
    B, T = tokens.shape
    x = sd["embed.W_E"][tokens] + sd["pos_embed.W_pos"][:T]
    resid_posts = []
    kv_cache = []
    mask_T = sd["blocks.0.attn.mask"][:T, :T]
    neg_inf = torch.tensor(-1e5, dtype=x.dtype)

    for l in range(8):
        x_norm = ln_pre(x)
        q = torch.einsum("btd,hde->bhte", x_norm, sd[f"blocks.{l}.attn.W_Q"]) + sd[f"blocks.{l}.attn.b_Q"][None, :, None, :]
        k = torch.einsum("btd,hde->bhte", x_norm, sd[f"blocks.{l}.attn.W_K"]) + sd[f"blocks.{l}.attn.b_K"][None, :, None, :]
        v = torch.einsum("btd,hde->bhte", x_norm, sd[f"blocks.{l}.attn.W_V"]) + sd[f"blocks.{l}.attn.b_V"][None, :, None, :]
        kv_cache.append((k, v))

        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * (1.0 / 8.0)
        scores = torch.where(mask_T, scores, neg_inf)
        probs = F.softmax(scores, dim=-1)
        z = torch.einsum("bhqk,bhkd->bhqd", probs, v)
        attn_out = torch.einsum("bthd,hde->bte", z.transpose(1, 2), sd[f"blocks.{l}.attn.W_O"]) + sd[f"blocks.{l}.attn.b_O"]
        x = x + attn_out

        x_mid_norm = ln_pre(x)
        mlp_hidden = F.gelu(x_mid_norm @ sd[f"blocks.{l}.mlp.W_in"] + sd[f"blocks.{l}.mlp.b_in"])
        mlp_out = mlp_hidden @ sd[f"blocks.{l}.mlp.W_out"] + sd[f"blocks.{l}.mlp.b_out"]
        x = x + mlp_out
        resid_posts.append(x)

    logits_last = ln_pre(x[:, -1, :]) @ sd["unembed.W_U"] + sd["unembed.b_U"]
    return logits_last, resid_posts, kv_cache


def forward_last_token_intervened(
    last_token_embed: torch.Tensor,
    sd: dict[str, torch.Tensor],
    kv_cache: list[tuple[torch.Tensor, torch.Tensor]],
    attn_deltas: dict[int, torch.Tensor] | None = None,
    resid_deltas: dict[int, torch.Tensor] | None = None,
    final_resid_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Exact batched forward pass for the final token position T-1 across B intervention arms,
    reusing the invariant prefix keys/values (0..T-2) from `kv_cache`.
    - last_token_embed: (1, 512) clean input embedding at T-1
    - attn_deltas[l]: (B, 512)
    - resid_deltas[l]: (B, 512)
    - final_resid_delta: (B, 512)
    Returns logits of shape (B, 61).
    """
    B = next(iter(attn_deltas.values())).shape[0] if attn_deltas else final_resid_delta.shape[0]
    x = last_token_embed.expand(B, -1).clone()  # (B, 512)

    for l in range(8):
        x_norm = ln_pre(x)  # (B, 512)
        q_last = torch.einsum("bd,hde->bhe", x_norm, sd[f"blocks.{l}.attn.W_Q"]) + sd[f"blocks.{l}.attn.b_Q"][None, :, :]
        k_last = torch.einsum("bd,hde->bhe", x_norm, sd[f"blocks.{l}.attn.W_K"]) + sd[f"blocks.{l}.attn.b_K"][None, :, :]
        v_last = torch.einsum("bd,hde->bhe", x_norm, sd[f"blocks.{l}.attn.W_V"]) + sd[f"blocks.{l}.attn.b_V"][None, :, :]

        k_prev = kv_cache[l][0][:, :, :-1, :].expand(B, -1, -1, -1)  # (B, 8, T-1, 64)
        v_prev = kv_cache[l][1][:, :, :-1, :].expand(B, -1, -1, -1)  # (B, 8, T-1, 64)

        k_all = torch.cat([k_prev, k_last.unsqueeze(2)], dim=2)  # (B, 8, T, 64)
        v_all = torch.cat([v_prev, v_last.unsqueeze(2)], dim=2)  # (B, 8, T, 64)

        scores = torch.einsum("bhd,bhtd->bht", q_last, k_all) * (1.0 / 8.0)  # (B, 8, T)
        probs = F.softmax(scores, dim=-1)
        z = torch.einsum("bht,bhtd->bhd", probs, v_all)  # (B, 8, 64)
        attn_out = torch.einsum("bhd,hde->be", z, sd[f"blocks.{l}.attn.W_O"]) + sd[f"blocks.{l}.attn.b_O"]

        if attn_deltas and l in attn_deltas:
            attn_out = attn_out + attn_deltas[l]
        x = x + attn_out

        x_mid_norm = ln_pre(x)
        mlp_hidden = F.gelu(x_mid_norm @ sd[f"blocks.{l}.mlp.W_in"] + sd[f"blocks.{l}.mlp.b_in"])
        mlp_out = mlp_hidden @ sd[f"blocks.{l}.mlp.W_out"] + sd[f"blocks.{l}.mlp.b_out"]
        x = x + mlp_out

        if resid_deltas and l in resid_deltas:
            x = x + resid_deltas[l]

    if final_resid_delta is not None:
        x = x + final_resid_delta

    logits = ln_pre(x) @ sd["unembed.W_U"] + sd["unembed.b_U"]
    return logits


def relative_board_targets(moves: list[int]) -> np.ndarray:
    board = OthelloBoardState()
    states = []
    for m in moves:
        board.umpire(m)
        states.append(board.state.copy())
    stack = np.stack(states, axis=0)
    T = stack.shape[0]
    rel = np.zeros((T, 8, 8), dtype=np.int64)
    rel[0::2][stack[0::2] == 1] = 1
    rel[0::2][stack[0::2] == -1] = 2
    rel[1::2][stack[1::2] == -1] = 1
    rel[1::2][stack[1::2] == 1] = 2
    return rel


def evaluate_and_train_control_probes(
    trained_sd: dict[str, torch.Tensor],
    untrained_sd: dict[str, torch.Tensor],
    probes: list[torch.Tensor],
    n_train_games: int = 200,
    n_test_games: int = 80,
) -> tuple[dict, list[torch.Tensor]]:
    print("Fitting Untrained-GPT control probes & evaluating layer-wise probe accuracy...")
    train_games = [generate_synthetic_game(seed=1000 + i) for i in range(n_train_games)]
    test_games = [generate_synthetic_game(seed=5000 + i) for i in range(n_test_games)]

    untrained_X = [[] for _ in range(8)]
    train_Y = []
    with torch.no_grad():
        for g in train_games:
            L = min(len(g), 59)
            if L < 10:
                continue
            toks = torch.tensor([[STOI[m] for m in g[:L]]], dtype=torch.long)
            rel_y = relative_board_targets(g[:L])
            _, r_un, _ = forward_full_sequence(toks, untrained_sd)
            sl = slice(4, L - 1)
            train_Y.append(torch.from_numpy(rel_y[sl]))
            for l in range(8):
                untrained_X[l].append(r_un[l][0, sl, :])

    Y_tr = torch.cat(train_Y, dim=0)
    N_tr = Y_tr.shape[0]
    Y_onehot = F.one_hot(Y_tr.view(N_tr, 64), num_classes=3).float().view(N_tr, 192)

    untrained_probes = []
    reg_eye = 100.0 * torch.eye(512)
    for l in range(8):
        X_l = torch.cat(untrained_X[l], dim=0)
        W_l = torch.linalg.solve(X_l.T @ X_l + reg_eye, X_l.T @ Y_onehot)
        untrained_probes.append(W_l.view(1, 512, 8, 8, 3))

    trained_acc = np.zeros(8)
    untrained_acc = np.zeros(8)
    trained_occ_acc = np.zeros(8)
    untrained_occ_acc = np.zeros(8)
    total_cells = 0
    total_occupied = 0

    with torch.no_grad():
        for g in test_games:
            L = min(len(g), 59)
            if L < 10:
                continue
            toks = torch.tensor([[STOI[m] for m in g[:L]]], dtype=torch.long)
            rel_y = torch.from_numpy(relative_board_targets(g[:L]))
            _, r_tr, _ = forward_full_sequence(toks, trained_sd)
            _, r_un, _ = forward_full_sequence(toks, untrained_sd)
            sl = slice(4, L - 1)
            y_true = rel_y[sl]  # (T_sl, 8, 8)
            occ_mask = y_true > 0
            total_cells += y_true.numel()
            total_occupied += int(occ_mask.sum().item())

            for l in range(8):
                logits_tr = torch.einsum("td,drck->trck", r_tr[l][0, sl], probes[l][0])
                pred_tr = logits_tr.argmax(dim=-1)
                trained_acc[l] += int((pred_tr == y_true).sum().item())
                trained_occ_acc[l] += int((pred_tr[occ_mask] == y_true[occ_mask]).sum().item())

                logits_un = torch.einsum("td,drck->trck", r_un[l][0, sl], untrained_probes[l][0])
                pred_un = logits_un.argmax(dim=-1)
                untrained_acc[l] += int((pred_un == y_true).sum().item())
                untrained_occ_acc[l] += int((pred_un[occ_mask] == y_true[occ_mask]).sum().item())

    probe_summary = {
        "layers": list(range(8)),
        "trained_3way_acc": (trained_acc / total_cells).tolist(),
        "untrained_3way_acc": (untrained_acc / total_cells).tolist(),
        "trained_occupied_acc": (trained_occ_acc / total_occupied).tolist(),
        "untrained_occupied_acc": (untrained_occ_acc / total_occupied).tolist(),
        "n_test_positions": int(total_cells // 64),
    }
    return probe_summary, untrained_probes


def get_probe_direction(
    probes_list: list[torch.Tensor],
    layer: int,
    r: int,
    c: int,
    from_player: int,
    move_idx: int,
    contrastive: bool = False,
) -> torch.Tensor:
    if move_idx % 2 == 0:
        target_opt = 2 if from_player == 0 else 1
        source_opt = 1 if from_player == 0 else 2
    else:
        target_opt = 1 if from_player == 0 else 2
        source_opt = 2 if from_player == 0 else 1

    w_tgt = probes_list[layer][0, :, r, c, target_opt]
    if not contrastive:
        return w_tgt / w_tgt.norm()
    w_src = probes_list[layer][0, :, r, c, source_opt]
    diff = w_tgt - w_src
    return diff / diff.norm()


def eval_batch_predictions(
    logits_batch: torch.Tensor,
    pre_valids: set[str],
    post_valids: set[str],
) -> tuple[list[int], list[float]]:
    K = len(post_valids)
    move_logits = logits_batch[:, 1:]  # (B, 60)
    probs = F.softmax(move_logits, dim=-1)
    top_indices = move_logits.topk(k=K, dim=-1).indices + 1
    new_valids = post_valids - pre_valids
    new_indices = [STOI[permit(m)] - 1 for m in new_valids]

    errs = []
    new_ps = []
    for b in range(logits_batch.shape[0]):
        pred_moves = {permit_reverse(ITOS[int(idx)]) for idx in top_indices[b]}
        fp = len(pred_moves - post_valids)
        fn = len(post_valids - pred_moves)
        errs.append(fp + fn)
        if new_indices:
            new_ps.append(float(probs[b, new_indices].sum().item()))
        else:
            new_ps.append(0.0)
    return errs, new_ps


def bootstrap_ci(values: list[float], n_boot: int = 1000, seed: int = 44) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    N = len(arr)
    idx = rng.integers(0, N, size=(n_boot, N))
    means = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    t0 = time.time()
    set_all_seeds(44)
    model_path = os.path.join(BASE_DIR, "data/synthetic_model.pth")
    if not os.path.exists(model_path):
        import urllib.request
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print("Downloading synthetic_model.pth from HuggingFace...")
        url = "https://huggingface.co/NeelNanda/Othello-GPT-Transformer-Lens/resolve/main/synthetic_model.pth"
        urllib.request.urlretrieve(url, model_path)
    sd = torch.load(model_path, map_location="cpu")
    probes = [torch.load(os.path.join(BASE_DIR, f"probes/resid_{l}_linear.pth"), map_location="cpu") for l in range(8)]
    with open(os.path.join(BASE_DIR, "data/intervention_benchmark.pkl"), "rb") as f:
        dataset = pickle.load(f)

    untrained_sd = make_untrained_sd(sd, seed=44)
    probe_summary, untrained_probes = evaluate_and_train_control_probes(sd, untrained_sd, probes)
    print(f"Probes evaluated in {time.time() - t0:.1f}s.")
    print("Probe 3-way accuracy (Trained):", [round(x, 4) for x in probe_summary["trained_3way_acc"]])
    print("Probe 3-way accuracy (Untrained):", [round(x, 4) for x in probe_summary["untrained_3way_acc"]])

    arm_names = [
        "null_baseline",                 # 0
        "full_multilayer_L5_attn",       # 1: Exact Nanda et al. intervene.py (L2..7, attn_out, probes[5], alpha=2.3)
        "mediated_only_skip_subtracted", # 2: L2..7 intervention with direct skip-connection subtracted before ln_final + W_U
        "direct_logit_bypass_only",      # 3: Accumulated delta injected ONLY at final residual stream before ln_final + W_U
        "per_layer_matched_attn",        # 4: Code audit 1: Using each layer's own probe probes[l] instead of hardcoded probes[5]
        "per_layer_matched_resid",       # 5: Code audit 2: Hooking resid_post (where probes were trained) with probes[l]
        "contrastive_flip_resid",        # 6: Code audit 3: Using contrastive (w_target - w_source) on resid_post (L2..7)
        "wrong_square_control",          # 7: Negative control 1: Real probe for an un-intervened square (r', c') != (r, c)
        "untrained_probe_control",       # 8: Negative control 2: Probe trained on randomly initialized Othello-GPT
    ]
    for l in range(8):
        arm_names.append(f"single_layer_{l}")             # 9..16: Single-layer intervention at layer l
    for l in range(8):
        arm_names.append(f"cumulative_from_{l}_to_7")     # 17..24: Multi-layer span [l..7]

    B_main = len(arm_names)
    errors = {k: [] for k in arm_names}
    new_probs = {k: [] for k in arm_names}

    alphas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.3, 3.0, 4.0]
    alpha_arms = ["full_steering", "mediated_only", "direct_bypass", "wrong_square", "untrained_probe"]
    alpha_sweep_errs = {a_name: {al: [] for al in alphas} for a_name in alpha_arms}

    scale_default = 2.3
    N_bench = len(dataset)
    print(f"Running KV-cached batched causal check across all {N_bench} benchmark boards...")

    with torch.no_grad():
        for idx, sample in enumerate(dataset):
            board = OthelloBoardState()
            completion = sample["history"]
            board.update(completion)
            pre_valids = {permit_reverse(m) for m in board.get_valid_moves()}
            tok_seq = torch.tensor([[STOI[s] for s in completion]], dtype=torch.long)
            T = tok_seq.shape[1]

            move = sample["pos_int"]
            r, c = move // 8, move % 8
            from_player = int(sample["ori_color"])
            to_player = 2 - from_player
            board.state[r, c] = to_player - 1
            post_valids = {permit_reverse(m) for m in board.get_valid_moves()}
            move_idx = len(completion)

            wrong_r, wrong_c = (r + 2) % 6 + 1, (c + 2) % 6 + 1
            if wrong_r == r and wrong_c == c:
                wrong_r = (r + 1) % 6 + 1

            u_L5 = get_probe_direction(probes, 5, r, c, from_player, move_idx, contrastive=False)
            u_per_layer = [get_probe_direction(probes, l, r, c, from_player, move_idx, contrastive=False) for l in range(8)]
            u_contrastive = [get_probe_direction(probes, l, r, c, from_player, move_idx, contrastive=True) for l in range(8)]
            u_wrong = get_probe_direction(probes, 5, wrong_r, wrong_c, from_player, move_idx, contrastive=False)
            u_untrained = get_probe_direction(untrained_probes, 5, r, c, from_player, move_idx, contrastive=False)

            d_L5 = scale_default * u_L5
            d_wrong = scale_default * u_wrong
            d_untr = scale_default * u_untrained

            # Run clean full sequence once to get prefix KV cache
            _, _, kv_cache = forward_full_sequence(tok_seq, sd)
            last_embed = sd["embed.W_E"][tok_seq[:, -1]] + sd["pos_embed.W_pos"][T - 1 : T]

            attn_deltas = {l: torch.zeros((B_main, 512)) for l in range(8)}
            resid_deltas = {l: torch.zeros((B_main, 512)) for l in range(8)}
            final_delta = torch.zeros((B_main, 512))

            for l in range(2, 8):
                attn_deltas[l][1] = d_L5
                attn_deltas[l][2] = d_L5
                attn_deltas[l][4] = scale_default * u_per_layer[l]
                resid_deltas[l][5] = scale_default * u_per_layer[l]
                resid_deltas[l][6] = scale_default * u_contrastive[l]
                attn_deltas[l][7] = d_wrong
                attn_deltas[l][8] = d_untr

            final_delta[2] = -6.0 * d_L5
            final_delta[3] = 6.0 * d_L5

            for l in range(8):
                attn_deltas[l][9 + l] = d_L5

            for start_l in range(8):
                for l in range(start_l, 8):
                    attn_deltas[l][17 + start_l] = d_L5

            logits_b = forward_last_token_intervened(
                last_embed,
                sd,
                kv_cache,
                attn_deltas=attn_deltas,
                resid_deltas=resid_deltas,
                final_resid_delta=final_delta,
            )
            b_errs, b_probs = eval_batch_predictions(logits_b, pre_valids, post_valids)
            for a_i, k in enumerate(arm_names):
                errors[k].append(b_errs[a_i])
                new_probs[k].append(b_probs[a_i])

            # Alpha sweep across all N_bench boards too (since KV-cached single-token step is super fast!)
            err0 = b_errs[0]
            for a_name in alpha_arms:
                alpha_sweep_errs[a_name][0.0].append(err0)

            nz_alphas = alphas[1:]
            B_al = len(nz_alphas) * 5
            al_attn = {l: torch.zeros((B_al, 512)) for l in range(2, 8)}
            al_final = torch.zeros((B_al, 512))
            for j, al in enumerate(nz_alphas):
                base_b = j * 5
                d_a = al * u_L5
                for l in range(2, 8):
                    al_attn[l][base_b + 0] = d_a
                    al_attn[l][base_b + 1] = d_a
                    al_attn[l][base_b + 3] = al * u_wrong
                    al_attn[l][base_b + 4] = al * u_untrained
                al_final[base_b + 1] = -6.0 * d_a
                al_final[base_b + 2] = 6.0 * d_a

            logits_al = forward_last_token_intervened(
                last_embed,
                sd,
                kv_cache,
                attn_deltas=al_attn,
                final_resid_delta=al_final,
            )
            al_errs, _ = eval_batch_predictions(logits_al, pre_valids, post_valids)
            for j, al in enumerate(nz_alphas):
                base_b = j * 5
                for arm_offset, a_name in enumerate(alpha_arms):
                    alpha_sweep_errs[a_name][al].append(al_errs[base_b + arm_offset])

            if (idx + 1) % 250 == 0:
                print(f"  Processed {idx + 1}/{N_bench} boards ({time.time() - t0:.1f}s)...")

    summary_arms = {}
    p0_arr = np.asarray(new_probs["null_baseline"], dtype=np.float64)

    for k in arm_names:
        m_err, lo_err, hi_err = bootstrap_ci(errors[k])
        p_gain = (np.asarray(new_probs[k], dtype=np.float64) - p0_arr).tolist()
        m_pg, lo_pg, hi_pg = bootstrap_ci(p_gain)
        summary_arms[k] = {
            "error_mean": m_err,
            "error_ci95": [lo_err, hi_err],
            "new_prob_gain_mean": m_pg,
            "new_prob_gain_ci95": [lo_pg, hi_pg],
        }

    alpha_summary = {}
    for a_name in alpha_arms:
        alpha_summary[a_name] = {}
        for al in alphas:
            m, lo, hi = bootstrap_ci(alpha_sweep_errs[a_name][al])
            alpha_summary[a_name][str(al)] = {"mean": m, "ci95": [lo, hi]}

    results = {
        "n_benchmark": N_bench,
        "probe_summary": probe_summary,
        "arms": summary_arms,
        "alpha_sweep": alpha_summary,
        "elapsed_seconds": time.time() - t0,
    }

    out_json = os.path.join(BASE_DIR, "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nCompleted in {time.time() - t0:.1f}s. Saved to {out_json}.")
    print("\n=== HEADLINE CAUSAL CHECK RESULTS (N = 1,001 boards) ===")
    for k in arm_names[:9]:
        s = summary_arms[k]
        print(
            f"{k:32s} | Error: {s['error_mean']:.3f} [{s['error_ci95'][0]:.3f}, {s['error_ci95'][1]:.3f}] "
            f"| dP_new: {s['new_prob_gain_mean']:+.3f} [{s['new_prob_gain_ci95'][0]:+.3f}, {s['new_prob_gain_ci95'][1]:+.3f}]"
        )

    print("\n=== LAYER-WISE DECODABILITY VS. SINGLE-LAYER & SPAN INTERVENTION ===")
    for l in range(8):
        s_single = summary_arms[f"single_layer_{l}"]
        s_span = summary_arms[f"cumulative_from_{l}_to_7"]
        p_acc = probe_summary["trained_3way_acc"][l] * 100
        u_acc = probe_summary["untrained_3way_acc"][l] * 100
        print(
            f"Layer {l} | Probe Acc: {p_acc:5.1f}% (Untrained: {u_acc:4.1f}%) "
            f"| Single-Layer Err: {s_single['error_mean']:.3f} | Span [{l}..7] Err: {s_span['error_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
