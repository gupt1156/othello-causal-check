# Causal Check on Othello-GPT: Separating Downstream World-Model Computation from Direct Readout Bias

**Author:** Rohan Gupta (`gupt1156@purdue.edu`)

---

## 1. Question, Experimental Setup, and Controls

In 8-layer `Othello-GPT` ($d_{\text{model}} = 512$, `LNPre`), linear probes $W_\ell \in \mathbb{R}^{512 \times 8 \times 8 \times 3}$ decode player-relative board states (`{Empty, Yours, Mine}`) with $>99\%$ accuracy by Layer 4, and adding probe directions across layers steers predictions toward a counterfactual board $B'$ (Nanda et al., 2023). We test two causal questions that arise whenever activation steering alters a model's output:
1. **Mediated world-model computation vs. direct skip-connection bias:** Because the residual stream forms an identity skip connection to the final unembedding ($\mathbf{x}_8 = \mathbf{x}_\ell + \sum_{k=\ell}^{7} f_k(\mathbf{x}_k)$), injecting $\boldsymbol{\delta}_\ell$ across layers $\ell \in \{2..7\}$ also adds $\sum_{\ell=2}^7 \boldsymbol{\delta}_\ell$ directly to $W_U$ (Figure 1a). Does steering succeed because intermediate Attention/MLP layers recompute Othello flanking rules over $B'$, or merely because $\sum_\ell \boldsymbol{\delta}_\ell$ biases $W_U$ directly?
2. **Decodability vs. single-layer causal control:** Does $99.4\%$ linear probe accuracy at Layer 5 imply that editing Layer 5 alone flips model predictions, or do downstream layers self-repair single-layer edits by attending back to unmodified prefix tokens?

**Setup, Metrics, and Controls:**
- **Model & Benchmark:** Pretrained 8-layer `synthetic_model.pth` and linear probes (`resid_0..7_linear.pth`) evaluated across all $N = 1{,}001$ counterfactual boards in `intervention_benchmark.pkl` (Li et al., 2023; Nanda et al., 2023), where flipping one occupied tile $s=(r,c)$ (`Yours` $\leftrightarrow$ `Mine`) changes ground-truth legal moves from $M(B)$ to $M(B')$.
- **Metrics (1,000-sample bootstrap 95% CIs):**
  1. **Counterfactual Move Error ($\downarrow$):** $|\hat{M}_K \setminus M(B')| + |M(B') \setminus \hat{M}_K|$ for $K = |M(B')|$ and top-$K$ predictions $\hat{M}_K$.
  2. **Newly-Legal Move Probability Gain ($\Delta P_{\text{new}} \uparrow$):** $\sum_{m \in M(B') \setminus M(B)} [P_{\text{intv}}(m) - P_{\text{clean}}(m)]$.
- **Controls:**
  1. **Mediated-Only vs. Direct-Logit Bypass:** `Mediated-Only` injects $+\alpha\hat{\mathbf{w}}_s$ at $\ell \in \{2..7\}$ and subtracts $-6\alpha\hat{\mathbf{w}}_s$ before `ln_final` + $W_U$ so the direct skip path sums to zero; `Direct-Logit Bypass` leaves hidden layers clean and adds $+6\alpha\hat{\mathbf{w}}_s$ only before `ln_final` + $W_U$.
  2. **Untrained-Encoder Probe Control:** Ridge probes ($\lambda = 100$) fit on a randomly initialized 8-layer GPT with matched weight variance over 200 synthetic games.
  3. **Wrong-Square Spatial Control ($s' \neq s$):** Inject the trained probe vector for an unrelated square $s' \neq s$ at the same norm $\alpha = 2.3$.

---

## 2. Figures and Results

### Figure 1: Transformer Intervention Architecture & ROME Causal Traces (`negrinho/sane_tikz` + `kmeng01/rome:plot_trace_heatmap`)
![Transformer Intervention Architecture](fig_transformer_intervention.png)

### Figure 2: Board-Level Causal Visualization on Case #101 (`likenneth/othello_world`: `plot_probs`, `plot_mentals`)
![Board-Level Intervention Visualization](fig_board_intervention.png)

### Figure 3: Aggregate Results Across $N = 1{,}001$ Benchmark Boards
![Causal Check Diagnostic Plot](fig_causal_check.png)

### Table 1: Causal Path Decomposition & Codebase Audit ($N = 1{,}001$, $\alpha = 2.3$)

| Experimental Arm | Error on $B'$ $\downarrow$ (95% CI) | $\Delta P_{\text{new}}$ $\uparrow$ (95% CI) | Mechanistic Role / Key Finding |
| :--- | :---: | :---: | :--- |
| **0. Null Baseline ($\alpha = 0$)** | `2.723` `[2.645, 2.799]` | `+0.000` `[+0.000, +0.000]` | Exact match to Nanda et al. (2023) Table 2 (`2.723`) |
| **1. Full Multi-Layer Steering (Nanda et al.)** | `0.102` `[0.072, 0.136]` | `+0.072` `[+0.066, +0.077]` | Exact match to Nanda et al. (2023) Table 2 (`0.10`) |
| **2. Mediated-Only (Skip-Subtracted before $W_U$)** | **`0.158`** `[0.122, 0.198]` | **`+0.037`** `[+0.033, +0.041]` | **97.9% of error reduction survives zeroing skip path** |
| **3. Direct-Logit Bypass Control (Only before $W_U$)** | `1.808` `[1.730, 1.884]` | **`+0.000`** `[+0.000, +0.000]` | **Zero probability gain on newly legal moves $M_{\text{new}}$** |
| **4. Untrained-Encoder Probe Control** | `2.661` `[2.581, 2.743]` | `+0.000` `[+0.000, +0.000]` | Random-projection directions exert no causal control |
| **5. Wrong-Square Spatial Control ($s' \neq s$)** | `3.435` `[3.337, 3.530]` | `+0.001` `[+0.001, +0.002]` | Editing an unrelated square degrades legal move set $M(B')$ |
| **6. Code Audit A: Layer-Matched `@ attn_out`** | `0.150` `[0.116, 0.188]` | `+0.071` `[+0.065, +0.076]` | Replaces hardcoded `probes[5]` in `intervene.py` |
| **7. Code Audit B: Layer-Matched `@ resid_post`** | `0.076` `[0.054, 0.102]` | `+0.075` `[+0.069, +0.081]` | Aligns intervention hook site with probe training site |
| **8. Code Audit C: Contrastive Flip `@ resid_post`** | **`0.012`** `[0.004, 0.026]` | **`+0.079`** `[+0.073, +0.085]` | **$8.5\times$ error reduction (`12` errors / $1{,}001$ boards)** |

### Experimental Findings

1. **Downstream world-model computation—not direct readout bias—drives counterfactual steering:** Subtracting the accumulated steering vector before $\text{LN}_{\text{final}} + W_U$ (`Mediated-Only`, Table 1 & Figure 3A) reduces counterfactual move error from `2.723` to **`0.158`**, retaining **97.9%** of the full error reduction purely via downstream Attention/MLP computation $\sum_{k=2}^7 \Delta f_k(\tilde{\mathbf{x}}_k)$ (specifically $\Delta\text{MLP}_5 = +4.62$, $\Delta\text{MLP}_6 = +27.09$, and $\Delta\text{Attn}_7 = +5.63$ logit gain on Case #101; Figure 1c–d). Conversely, applying $+6\alpha\hat{\mathbf{w}}_s$ only before $W_U$ (`Direct-Logit Bypass`) leaves error at `1.808` with **$\Delta P_{\text{new}} = +0.000$**: on Case #101 (Figure 2c), direct unembedding dot products partially suppress invalidated move `F7` (`0.12 -> 0.06`) but assign `0.00` probability to newly legal flanking moves `B5` and `C5`. `Untrained-Encoder` (`2.661` err) and `Wrong-Square` (`3.435` err) controls confirm spatial and representational specificity.
2. **High single-layer decodability $\neq$ single-layer causal control due to downstream self-repair:** Trained probe accuracy saturates at `99.0%–99.5%` in `L4–L7` (vs. `66.2% -> 62.6%` on the untrained GPT as random weights scramble token history; Figure 3B), yet single-layer edits never reduce error below `2.188` (`L5`). Figure 1b,e and Figure 2f–h show why: editing `L4` alone flips the probe's belief at `C4` to Black (`X`, $p=0.91\text{--}1.00$), but downstream `L5–L6` MLPs (`+2.24, +2.22` repair pull) and `L6–L7` attention (`+0.19, +0.50`) attend back to unmodified history tokens and overwrite `C4` back to White (`O`, $p=0.93$). Cumulative spans $[\ell..7]$ achieve minimum error at **`Span [L3..L7]` (`0.072`)** and **`Span [L2..L7]` (`0.102`)**, but fail when started at **`L5` (`1.345`)** or **`L6` (`2.268`)**—showing that multi-ray flanking circuits in Layers 3–4 execute before linear decodability peaks at Layer 6.
3. **Auditing `ajyl/mech_int_othelloGPT` cuts counterfactual error by $8.5\times$ (`0.102 -> 0.012`):** Inspecting `mech_int/intervene.py` (lines 140–162) revealed three discrepancies between Nanda et al.'s text and code: (i) hardcoding `hook_fns[5]` (`probes[5]`) across all steered layers $\ell \in \{2..7\}$; (ii) hooking `hook_attn_out` (pre-MLP) using probes trained on `hook_resid_post` (post-MLP); and (iii) adding only $+\alpha\hat{\mathbf{w}}_{s,\text{tgt}}$ without subtracting the active source state $\mathbf{w}_{s,\text{src}}$. Aligning the hook to `resid_post` with layer-matched probes drops error to `0.076` (Arm 7), and steering along the unit **contrastive flip vector** $\hat{\mathbf{d}}_s^\ell = (\mathbf{w}_{s,\text{tgt}}^\ell - \mathbf{w}_{s,\text{src}}^\ell)/\|\cdot\|_2$ at `resid_post` slashes error to **`0.012 [0.004, 0.026]`** (Arm 8; 12 errors across 1,001 boards, 99.6% probe flip).

---

## 3. Falsification Criteria and Next Week's Experiments

- **What would change our conclusion:** (1) Our mediated-computation conclusion would be falsified if freezing intermediate `Attn/MLP` outputs to clean values while applying $\boldsymbol{\delta}$ before $W_U$ recovered $>50\%$ of the error reduction or $\Delta P_{\text{new}} > +0.02$ (it recovers `33.6%` and $\Delta P_{\text{new}} = +0.000$). (2) Our self-repair conclusion would be falsified if projecting out the source tile subspace at `L3` *and* ablating `L4–L6` attention to the move token where $s$ was played still failed to reduce single-layer edit error below `0.50`.
- **Next week's plan:** (1) *Sub-circuit path patching of self-repair:* Path-patch individual `L4–L6` attention heads vs. MLP projections to separate attention re-reading token history from MLP local-board heuristics (Lin et al., 2024). (2) *Endgame `MoveFirst` audit ($t > 30$):* Stratify contrastive steering by game turn $t$ and compare discriminative probe normal vectors against difference-in-means vectors ($\boldsymbol{\mu}_{\text{Mine}} - \boldsymbol{\mu}_{\text{Yours}}$) in late-game states where moves are computed before full board state.

---

## Reproducing

```bash
python run_experiment.py
python generate_paper_visuals.py
pdflatex report.tex && pdflatex report.tex
```
