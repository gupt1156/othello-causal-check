# Causal Check on Othello-GPT: Separating Downstream World-Model Computation from Direct Readout Bias

**Rohan Gupta** · Purdue University · `gupt1156@purdue.edu`

---

## 1. Setup and Causal Decomposition

In 8-layer `Othello-GPT` ($d_{\text{model}}=512$, `LNPre`), linear probes $W_\ell \in \mathbb{R}^{512 \times 8 \times 8 \times 3}$ trained on residual stream activations decode player-relative board states (`Empty`, `Yours`, `Mine`) with over $99\%$ accuracy by Layer 4, and adding probe directions across layers steers next-move predictions toward a counterfactual board $B'$ (Li et al., 2023; Nanda et al., 2023). Because the residual stream forms an identity skip connection to the final unembedding ($\mathbf{x}_8 = \mathbf{x}_\ell + \sum_{k=\ell}^{7} f_k(\mathbf{x}_k)$), injecting a steering vector $\boldsymbol{\delta}_\ell$ across intermediate layers $\ell \in \{2..7\}$ also adds $\sum_{\ell=2}^7 \boldsymbol{\delta}_\ell$ directly into the readout matrix $W_U$ (Figure 1a). This raises the question of whether steering works because intermediate attention and MLP layers recompute Othello flanking rules over $B'$, or simply because $\sum_\ell \boldsymbol{\delta}_\ell$ acts as a direct logit bias at $W_U$. A related question is why single-layer interventions fail even at layers where linear probe accuracy exceeds $99.4\%$.

We test both questions using the pretrained `synthetic_model.pth` checkpoint and linear probes (`resid_0..7_linear.pth`) across all $N = 1{,}001$ counterfactual boards in `intervention_benchmark.pkl`. Each benchmark board flips a single occupied square $s=(r,c)$ between `Yours` and `Mine`, changing the ground-truth legal move set from $M(B)$ to $M(B')$. Across all boards (with 1,000-sample bootstrap $95\%$ CIs), we measure counterfactual move error $|\hat{M}_K \setminus M(B')| + |M(B') \setminus \hat{M}_K|$ for $K=|M(B')|$ and top-$K$ predicted moves $\hat{M}_K$, alongside the probability gain $\Delta P_{\text{new}} = \sum_{m \in M(B') \setminus M(B)} [P_{\text{intv}}(m) - P_{\text{clean}}(m)]$ on moves that become newly legal under $B'$.

To separate downstream circuit computation from direct unembedding bias, our **Mediated-Only** intervention injects $+\alpha\hat{\mathbf{w}}_s$ at layers $\ell \in \{2..7\}$ and subtracts $-6\alpha\hat{\mathbf{w}}_s$ immediately before $\text{LN}_{\text{final}} + W_U$ so the direct skip path sums to zero. Conversely, the **Direct-Logit Bypass** control leaves all hidden layers clean and adds $+6\alpha\hat{\mathbf{w}}_s$ only before $\text{LN}_{\text{final}} + W_U$. We also evaluate an **Untrained-Encoder** control (ridge probes $\lambda=100$ fit on a randomly initialized 8-layer GPT with matched weight variance over 200 games) and a **Wrong-Square** spatial control ($s' \neq s$ at $\alpha=2.3$).

![Transformer Intervention Architecture](fig_transformer_intervention.png)

| Experimental Arm ($N = 1{,}001$ counterfactual boards, $\alpha = 2.3$) | Error on $B'$ $\downarrow$ (95% CI) | $\Delta P_{\text{new}}$ $\uparrow$ (95% CI) | Mechanistic Role / Key Finding |
| :--- | :---: | :---: | :--- |
| 0. Null Baseline ($\alpha = 0$, un-intervened model) | `2.723` `[2.645, 2.799]` | `+0.000` `[+0.000, +0.000]` | Exact match to Nanda et al. (2023) Table 2 (`2.723`) |
| 1. Full Multi-Layer Steering (Nanda et al. `intervene.py`) | `0.102` `[0.072, 0.136]` | `+0.072` `[+0.066, +0.077]` | Exact match to Nanda et al. (2023) Table 2 (`0.10`) |
| **2. Mediated-Only (Skip-Subtracted before $\text{LN}_{\text{final}} + W_U$)** | **`0.158`** `[0.122, 0.198]` | **`+0.037`** `[+0.033, +0.041]` | **97.9% of error reduction survives zeroing skip path** |
| **3. Direct-Logit Bypass Control (Only before $\text{LN}_{\text{final}} + W_U$)** | `1.808` `[1.730, 1.884]` | **`+0.000`** `[+0.000, +0.000]` | **Zero probability gain on newly legal moves $M_{\text{new}}$** |
| 4. Untrained-Encoder Probe Control (Random-init GPT probe) | `2.661` `[2.581, 2.743]` | `+0.000` `[+0.000, +0.000]` | Random-projection directions exert no causal control |
| 5. Wrong-Square Spatial Control (Steer $s' \neq s$ at $\alpha = 2.3$) | `3.435` `[3.337, 3.530]` | `+0.001` `[+0.001, +0.002]` | Editing an unrelated square degrades legal move set $M(B')$ |
| 6. Code Audit A: Layer-Matched `probes[l]` @ `hook_attn_out` | `0.150` `[0.116, 0.188]` | `+0.071` `[+0.065, +0.076]` | Replaces hardcoded `probes[5]` in `intervene.py` |
| 7. Code Audit B: Layer-Matched `probes[l]` @ `hook_resid_post` | `0.076` `[0.054, 0.102]` | `+0.075` `[+0.069, +0.081]` | Aligns intervention hook site with probe training site |
| **8. Code Audit C: Contrastive Flip $\hat{\mathbf{d}}_s^\ell$ @ `hook_resid_post`** | **`0.012`** `[0.004, 0.026]` | **`+0.079`** `[+0.073, +0.085]` | **$8.5\times$ error reduction (`12` total errors / $1{,}001$ boards)** |

---

## 2. Results

![Board-Level Intervention Visualization](fig_board_intervention.png)

![Causal Check Diagnostic Plot](fig_causal_check.png)

**Mediated computation vs. direct readout bias.** Subtracting the accumulated steering vector before $\text{LN}_{\text{final}} + W_U$ (`Mediated-Only`, Table 1 and Figure 3A) reduces counterfactual move error from `2.723` to **`0.158`**, retaining **97.9%** of the full multi-layer error reduction purely through downstream attention and MLP updates $\sum_{k=2}^7 \Delta f_k(\tilde{\mathbf{x}}_k)$. Component-level causal traces on Case #101 (flipping `C4` from White to Black; Figure 1c–d) show that this recovery is driven by mid-to-late MLPs ($\Delta\text{MLP}_5 = +4.62$, $\Delta\text{MLP}_6 = +27.09$ logits) and final-layer attention ($\Delta\text{Attn}_7 = +5.63$ logits). In contrast, applying $+6\alpha\hat{\mathbf{w}}_s$ only before $W_U$ (`Direct-Logit Bypass`) leaves move error at `1.808` and yields **$\Delta P_{\text{new}} = +0.000$**: on Case #101 (Figure 2c), direct unembedding dot products partially suppress the invalidated move `F7` (`0.12 -> 0.06`) but assign `0.00` probability to the newly legal flanking moves `B5` and `C5`. Both the untrained-encoder probe (`2.661` error) and wrong-square steering (`3.435` error) fail to steer predictions toward $M(B')$.

**Single-layer decodability vs. downstream self-repair.** Although 3-way linear probe accuracy reaches `99.0%` to `99.5%` in Layers 4–7 (compared to `66.2% -> 62.6%` on the untrained GPT; Figure 3B), editing any single layer alone never reduces move error below `2.188` (Layer 5). Layer-wise probe beliefs on Case #101 (Figure 1b,e and Figure 2f–h) show why: editing Layer 4 alone flips the probe's belief at `C4` to Black (`X`, $p=1.00$), but Layers 5–6 MLPs (`+2.24, +2.22` logit pull toward White) and Layers 6–7 attention (`+0.19, +0.50`) attend back to the unmodified move prefix and overwrite `C4` back to White (`O`, $p=0.93$). Cumulative layer spans $[\ell..7]$ reach minimum error at `L3..7` (`0.072`) and `L2..7` (`0.102`), but degrade sharply when started at `L5` (`1.345`) or `L6` (`2.268`), indicating that multi-ray flanking circuits in Layers 3–4 execute before linear probe accuracy peaks at Layer 6.

**Codebase audit and contrastive steering.** Inspecting `mech_int/intervene.py` (lines 140–162) in `ajyl/mech_int_othelloGPT` uncovered three implementation bugs in the reference benchmark: the loop hardcodes `hook_fns[5]` (`probes[5]`) across all steered layers $\ell \in \{2..7\}$, attaches hooks to `hook_attn_out` (pre-MLP) using probes trained on `hook_resid_post` (post-MLP), and adds $+\alpha\hat{\mathbf{w}}_{s,\text{tgt}}$ without subtracting the active source class vector $\mathbf{w}_{s,\text{src}}$. Moving the hook to `hook_resid_post` with layer-matched probes lowers error from `0.102` to `0.076` (Arm 7), and steering along the unit contrastive flip direction $\hat{\mathbf{d}}_s^\ell = (\mathbf{w}_{s,\text{tgt}}^\ell - \mathbf{w}_{s,\text{src}}^\ell)/\|\cdot\|_2$ at `hook_resid_post` reduces counterfactual error by $8.5\times$ to **`0.012 [0.004, 0.026]`** (`12` errors across $1{,}001$ boards; Arm 8).

---

## 3. Falsification and Next Steps

Our conclusion that steering operates through downstream circuit computation rather than direct readout bias would be falsified if freezing intermediate attention and MLP outputs to their clean values while applying $\boldsymbol{\delta}$ before $W_U$ recovered more than $50\%$ of the error reduction or achieved $\Delta P_{\text{new}} > +0.02$ (in practice it recovers $33.6\%$ and $\Delta P_{\text{new}} = +0.000$). Similarly, our self-repair explanation for single-layer failure would be falsified if projecting out the source tile direction at Layer 3 while ablating Layers 4–6 attention to the prefix token where square $s$ was played still failed to push single-layer edit error below $0.50$. With another week, we would path-patch individual Layers 4–6 attention heads against MLP projections to separate prefix re-reading from local board heuristics (Lin et al., 2024), and stratify contrastive steering by game turn $t > 30$ to compare logistic probe normals against difference-in-means vectors ($\boldsymbol{\mu}_{\text{Mine}} - \boldsymbol{\mu}_{\text{Yours}}$) in late-game states.
