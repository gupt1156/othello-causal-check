# Causal Check on Othello-GPT: Separating Downstream World-Model Computation from Direct Readout Bias

**Rohan Gupta** (`gupt1156@purdue.edu`)

---

## 1. Question

In 8-layer `Othello-GPT` ($d_{\text{model}}=512$, `LNPre`), linear probes $W_\ell \in \mathbb{R}^{512 \times 8 \times 8 \times 3}$ decode player-relative board states (`Empty`, `Yours`, `Mine`) with $>99\%$ accuracy by Layer 4, and adding probe vectors across layers steers next-move predictions toward a counterfactual board $B'$ ([Li et al., 2023](https://github.com/likenneth/othello_world); [Nanda et al., 2023](https://github.com/ajyl/mech_int_othelloGPT)).

Whenever linear activation steering changes a transformer's output, two causal questions arise:

| Question | Mechanism Tested |
| :--- | :--- |
| **1. Mediated computation vs. direct readout bias** | Because the residual stream is an identity skip connection to the final unembedding ($\mathbf{x}_8 = \mathbf{x}_\ell + \sum_{k=\ell}^{7} f_k(\mathbf{x}_k)$), injecting $\boldsymbol{\delta}_\ell$ across layers $\ell \in \{2..7\}$ also adds $\sum_{\ell=2}^7 \boldsymbol{\delta}_\ell$ directly to $W_U$. Does steering work because intermediate Attention/MLP layers recompute Othello flanking rules over $B'$, or merely because $\sum_\ell \boldsymbol{\delta}_\ell$ biases $W_U$ directly? |
| **2. Single-layer decodability vs. causal control** | Linear probe accuracy reaches `99.4%` at Layer 5. Does editing Layer 5 alone flip move predictions, or do downstream layers self-repair single-layer edits by attending back to unmodified move history? |

---

## 2. Experimental Setup, Metrics, and Controls

We evaluate the official pretrained `synthetic_model.pth` checkpoint and linear probes (`resid_0..7_linear.pth`) across all **$N = 1{,}001$ counterfactual boards** in `intervention_benchmark.pkl`. Each test case flips one occupied square $s=(r,c)$ (`Yours` $\leftrightarrow$ `Mine`), changing the ground-truth legal move set from $M(B)$ to $M(B')$.

### Metrics (1,000-Sample Bootstrap 95% CIs)
- **Counterfactual Move Error ($\downarrow$):** $|\hat{M}_K \setminus M(B')| + |M(B') \setminus \hat{M}_K|$ for $K = |M(B')|$ and top-$K$ model predictions $\hat{M}_K$.
- **Newly-Legal Move Probability Gain ($\Delta P_{\text{new}} \uparrow$):** $\sum_{m \in M(B') \setminus M(B)} [P_{\text{intv}}(m) - P_{\text{clean}}(m)]$ on moves that are legal under $B'$ but illegal under $B$.

### Controls
| Control Arm | Intervention Rule | What It Rules Out |
| :--- | :--- | :--- |
| **Mediated-Only (Skip-Subtracted)** | Inject $+\alpha\hat{\mathbf{w}}_s$ at $\ell \in \{2..7\}$, then subtract $-6\alpha\hat{\mathbf{w}}_s$ before $\text{LN}_{\text{final}} + W_U$ | Zeroes out the direct skip connection to $W_U$; isolates downstream Attention/MLP computation $\sum_{k=2}^7 \Delta f_k(\tilde{\mathbf{x}}_k)$ |
| **Direct-Logit Bypass Control** | Keep hidden layers $\ell \in \{0..7\}$ clean; add $+6\alpha\hat{\mathbf{w}}_s$ **only** before $\text{LN}_{\text{final}} + W_U$ | Tests whether direct unembedding dot products alone can steer legal moves |
| **Untrained-Encoder Probe** | Fit ridge probes ($\lambda=100$) on a randomly initialized 8-layer GPT with matched weight variance | Tests whether arbitrary high-dimensional probe directions exert causal control |
| **Wrong-Square Control ($s' \neq s$)** | Inject the trained probe vector for an unrelated square $s' \neq s$ at the same norm $\alpha = 2.3$ | Tests spatial specificity of the probe intervention |

---

## 3. Results Across $N = 1{,}001$ Counterfactual Boards

![Transformer Intervention Architecture and ROME Causal Traces](fig_transformer_intervention.png)
*Figure 1: **(a)** Othello-GPT decoder intervention diagram rendered via [`negrinho/sane_tikz`](https://github.com/negrinho/sane_tikz) (`examples/transformer.py`), showing the three intervention sites along the residual stream. **(b–e)** Component-level causal traces on Case #101 ($T=17$, flipping `C4` from White $\bigcirc$ to Black $\bullet$) rendered via [`kmeng01/rome`](https://github.com/kmeng01/rome) (`plot_trace_heatmap`).*

| Experimental Arm ($N = 1{,}001$, $\alpha = 2.3$) | Error on $B'$ $\downarrow$ (95% CI) | $\Delta P_{\text{new}}$ $\uparrow$ (95% CI) | Key Takeaway |
| :--- | :---: | :---: | :--- |
| **0. Null Baseline ($\alpha = 0$)** | `2.723` `[2.645, 2.799]` | `+0.000` `[+0.000, +0.000]` | Exact match to Nanda et al. (2023) Table 2 (`2.723`) |
| **1. Full Multi-Layer Steering (Nanda et al.)** | `0.102` `[0.072, 0.136]` | `+0.072` `[+0.066, +0.077]` | Exact match to Nanda et al. (2023) Table 2 (`0.10`) |
| **2. Mediated-Only (Skip-Subtracted before $W_U$)** | **`0.158`** `[0.122, 0.198]` | **`+0.037`** `[+0.033, +0.041]` | **97.9% of error reduction survives zeroing out the direct skip path** |
| **3. Direct-Logit Bypass Control (Only before $W_U$)** | `1.808` `[1.730, 1.884]` | **`+0.000`** `[+0.000, +0.000]` | **Zero probability gain on newly legal moves $M_{\text{new}}$** |
| **4. Untrained-Encoder Probe Control** | `2.661` `[2.581, 2.743]` | `+0.000` `[+0.000, +0.000]` | Random-init probe directions have no causal effect |
| **5. Wrong-Square Spatial Control ($s' \neq s$)** | `3.435` `[3.337, 3.530]` | `+0.001` `[+0.001, +0.002]` | Editing an unrelated square degrades $M(B')$ predictions |
| **6. Code Audit A: Layer-Matched `@ attn_out`** | `0.150` `[0.116, 0.188]` | `+0.071` `[+0.065, +0.076]` | Replaces hardcoded `probes[5]` in `intervene.py` |
| **7. Code Audit B: Layer-Matched `@ resid_post`** | `0.076` `[0.054, 0.102]` | `+0.075` `[+0.069, +0.081]` | Aligns intervention hook site with probe training site |
| **8. Code Audit C: Contrastive Flip `@ resid_post`** | **`0.012`** `[0.004, 0.026]` | **`+0.079`** `[+0.073, +0.085]` | **$8.5\times$ error reduction (`12` errors across $1{,}001$ boards)** |

---

## 4. Three Key Findings

### Finding 1: Downstream World-Model Computation—Not Direct Readout Bias—Drives Steering

Subtracting the accumulated steering vector immediately before $\text{LN}_{\text{final}} + W_U$ (**Mediated-Only**) reduces counterfactual error from `2.723` to **`0.158`**, retaining **97.9%** of the full multi-layer error reduction purely via downstream Attention and MLP computation.

- **Where the computation happens (Figure 1c–d):** On Case #101 (flipping `C4` from White to Black), the newly legal flanking moves `B5` and `C5` are computed by mid-to-late MLPs ($\Delta\text{MLP}_5 = +4.62$, $\Delta\text{MLP}_6 = +27.09$ logits) and final-layer attention ($\Delta\text{Attn}_7 = +5.63$ logits).
- **Why Direct-Logit Bypass fails (Figure 2c):** Adding $+6\alpha\hat{\mathbf{w}}_s$ only before $W_U$ leaves error at `1.808` and yields **$\Delta P_{\text{new}} = +0.000$**. Direct unembedding dot products partially suppress the invalidated move `F7` (`0.12 -> 0.06`), but assign `0.00` probability to `B5` and `C5` because discovering newly legal moves requires multi-ray flanking computation in intermediate layers.

![Board-Level Causal Visualization on Case #101](fig_board_intervention.png)
*Figure 2: Board-level causal visualization on Case #101 ($T=17$, flipping `C4`: $\bigcirc \to \bullet$), rendered via [`likenneth/othello_world`](https://github.com/likenneth/othello_world) (`plot_probs`, `plot_mentals`, `OthelloBoardState`). **Top (a–e):** Next-move probability heatmaps. `Direct Bypass Only` (c) assigns `0.00` to newly legal moves `B5, C5` (4 errors), whereas `Mediated-Only` (d) and `Contrastive Flip` (e) suppress `F7` and activate `B5, C5` with **0 errors**. **Bottom (f–j):** Layer-wise probe beliefs (`O` = White, `X` = Black), showing downstream self-repair after a single-layer edit at $L_4$ (g–h, red box) vs. sustained state flip under $L_{2..7}$ steering (i–j, green box).*

---

### Finding 2: High Single-Layer Decodability $\neq$ Single-Layer Causal Control Due to Self-Repair

Linear probe accuracy saturates at `99.0%–99.5%` across Layers 4–7 (vs. `66.2% -> 62.6%` on the untrained GPT control; Figure 3B). Yet intervening at any single layer alone never reduces error below `2.188` (`L5`).

![Aggregate Results Across N=1,001 Boards](fig_causal_check.png)
*Figure 3: **(A)** Counterfactual move error vs. steering scale $\alpha$ across $N=1{,}001$ boards. **(B)** Layer-wise 3-way probe accuracy (green/gray, right axis) vs. single-layer edit error (red) and cumulative span $[\ell..7]$ error (blue).*

| Layer $\ell$ | `L0` | `L1` | `L2` | `L3` | `L4` | `L5` | `L6` | `L7` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trained Probe 3-Way Accuracy** | `90.4%` | `94.5%` | `97.1%` | `98.2%` | `99.0%` | `99.4%` | **`99.5%`** | `99.4%` |
| **Untrained Control Probe Accuracy** | `66.2%` | `65.1%` | `64.4%` | `63.9%` | `63.4%` | `63.0%` | `62.6%` | `62.6%` |
| **Single-Layer Edit Error ($\ell$ only)** | `2.677` | `2.679` | `2.633` | `2.535` | `2.370` | **`2.188`** | `2.398` | `2.673` |
| **Cumulative Span Error ($[\ell..7]$)** | `0.390` | `0.166` | `0.102` | **`0.072`** | `0.192` | `1.345` | `2.268` | `2.673` |

- **Mechanism of self-repair (Figure 1b,e & Figure 2f–h):** Editing Layer 4 alone flips the probe's belief at `C4` to Black (`X`, $p=1.00$), but Layers 5–6 MLPs (`+2.24, +2.22` repair pull) and Layers 6–7 attention (`+0.19, +0.50`) attend back to unmodified prefix tokens and overwrite `C4` back to White (`O`, $p=0.93$).
- **Decodability vs. circuit execution timing:** Cumulative spans $[\ell..7]$ reach minimum error at **`Span [L3..L7]` (`0.072`)**, but fail when started at **`L5` (`1.345`)** or **`L6` (`2.268`)**—showing that multi-ray flanking circuits in Layers 3–4 execute *before* linear probe accuracy peaks at Layer 6.

---

### Finding 3: Auditing `ajyl/mech_int_othelloGPT` Cuts Error by $8.5\times$ (`0.102` $\to$ `0.012`)

Inspecting `mech_int/intervene.py` (lines 140–162) in [`ajyl/mech_int_othelloGPT`](https://github.com/ajyl/mech_int_othelloGPT) revealed three implementation discrepancies between Nanda et al.'s text and code:

| Discrepancy in `mech_int/intervene.py` | Reference Code Behavior | Corrected Intervention | Error on $B'$ ($\alpha=2.3$) |
| :--- | :--- | :--- | :---: |
| **1. Hardcoded Layer-5 probe** | Builds `hook_fns[0..7]` but attaches `hook_fns[5]` (`probes[5]`) to all steered layers `2..7` | Use layer-matched `probes[l]` at each layer $\ell \in \{2..7\}$ (`Arm 6`) | `0.150` `[0.116, 0.188]` |
| **2. Pre-MLP vs. Post-MLP hook mismatch** | Hooks `blocks.{l}.hook_attn_out` (pre-MLP) using probes trained on `hook_resid_post` (post-MLP) | Hook `blocks.{l}.hook_resid_post` to match probe training site (`Arm 7`) | `0.076` `[0.054, 0.102]` |
| **3. Non-contrastive vector addition** | Adds $+\alpha\hat{\mathbf{w}}_{s,\text{tgt}}$ without subtracting the active source state $\mathbf{w}_{s,\text{src}}$ | Steer along unit contrastive flip direction $\hat{\mathbf{d}}_s^\ell = \frac{\mathbf{w}_{s,\text{tgt}}^\ell - \mathbf{w}_{s,\text{src}}^\ell}{\|\mathbf{w}_{s,\text{tgt}}^\ell - \mathbf{w}_{s,\text{src}}^\ell\|_2}$ at `resid_post` (`Arm 8`) | **`0.012`** `[0.004, 0.026]` (**12 total errors / 1,001 boards**) |

---

## 5. Falsification Criteria and Next Steps

| Question | Falsification Criterion / Next Experiment |
| :--- | :--- |
| **What would falsify our mediated-computation conclusion?** | Freezing intermediate Attention/MLP outputs in Layers 3–7 to clean values while applying $\boldsymbol{\delta}$ before $W_U$ and recovering $>50\%$ of the error reduction or $\Delta P_{\text{new}} > +0.02$ (in practice it recovers `33.6%` and $\Delta P_{\text{new}} = +0.000$). |
| **What would falsify our self-repair conclusion?** | Projecting out the source tile subspace at Layer 3 *and* ablating Layers 4–6 attention to the prefix token where square $s$ was played, yet still failing to reduce single-layer edit error below `0.50`. |
| **Next week's experiment 1: Sub-circuit path patching** | Path-patch individual Layers 4–6 attention heads vs. MLP projections to separate attention re-reading token history from MLP local-board heuristics ([Lin et al., 2024](https://arxiv.org/abs/2403.03981)). |
| **Next week's experiment 2: Endgame `MoveFirst` audit ($t > 30$)** | Stratify contrastive steering by game turn $t$ and compare discriminative logistic probe normals against difference-in-means vectors ($\boldsymbol{\mu}_{\text{Mine}} - \boldsymbol{\mu}_{\text{Yours}}$) in late-game states where moves are computed before full board state. |

---

## Reproducing

```bash
# 1. Run 25-arm causal benchmark across N = 1,001 boards -> results.json & fig_causal_check.png
python run_experiment.py

# 2. Render Figure 1 (negrinho/sane_tikz + kmeng01/rome) and Figure 2 (likenneth/othello_world)
python generate_paper_visuals.py
```
