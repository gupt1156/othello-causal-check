# Causal Check on Othello-GPT: Separating World-Model Computation from Direct Readout Bias

**Rohan Gupta** (`gupt1156@purdue.edu`)

---

Othello-GPT is an 8-layer transformer trained only on sequences of legal Othello moves, with no built-in knowledge of the $8 \times 8$ board or game rules. Linear probes trained on its internal activations can decode which piece is on each square (`Empty`, `Yours`, or `Mine`) with over 99% accuracy by Layer 4, and pushing activations along a probe direction after flipping a piece steers the model into predicting legal moves for the edited board ([Li et al., 2023](https://github.com/likenneth/othello_world); [Nanda et al., 2023](https://github.com/ajyl/mech_int_othelloGPT)).

We ran a causal audit across all 1,001 counterfactual boards in the Othello-GPT intervention benchmark to test whether steering actually uses the model's internal rule computation or merely biases the final output layer via skip connections, why single-layer edits fail despite 99.4% probe accuracy, and whether the reference benchmark code matches the paper.

## Summary of Results

| Question | Answer |
| :--- | :--- |
| **Is steering just biasing the final output layer via the skip connection?** | **No.** Subtracting the entire steering vector right before the final output layer still preserves **97.9%** of the error reduction (`2.723 -> 0.158` errors). |
| **Can the direct skip connection to the output layer find newly legal moves on its own?** | **No.** Adding the steering vector only at the final readout layer gives **+0.000** probability gain on newly legal moves (`1.808` errors). |
| **Does 99.4% probe accuracy at Layer 5 mean editing Layer 5 alone flips the model's move?** | **No.** Editing Layer 5 alone leaves error at `2.188`. Layers 5–7 attend back to the unedited move history and overwrite ("self-repair") single-layer edits. |
| **Does the layer with the highest probe accuracy give the best causal control?** | **No.** Probe accuracy peaks at Layer 6 (`99.5%`), but steering from Layer 5 or 6 onward fails (`1.345` and `2.268` errors) because flanking rules are already computed in Layers 3–4. |
| **Do probes trained on an untrained transformer steer moves?** | **No.** An untrained GPT with matched weight variance gets `2.661` errors and `+0.000` probability gain on newly legal moves. |
| **Does the official `mech_int/intervene.py` code match the paper?** | **No.** It hardcodes Layer 5's probe into all layers, hooks before the MLP instead of after, and adds the target state without subtracting the current state. Fixing all three cuts error by **8.5x** (`0.102 -> 0.012`, or 12 errors across 1,001 boards). |

The headline numbers across all 1,001 counterfactual boards (with 1,000-sample bootstrap 95% CIs):

```
                                              move errors on edited board    prob gain on newly legal moves
                                                  (lower is better)                (higher is better)
un-intervened model (clean board)                 2.723 [2.645, 2.799]              +0.000 [+0.000, +0.000]
full multi-layer steering (Nanda et al.)          0.102 [0.072, 0.136]              +0.072 [+0.066, +0.077]
  minus the direct skip path (mediated-only)      0.158 [0.122, 0.198]              +0.037 [+0.033, +0.041]  <- 97.9% of reduction survives
  direct skip path only (no hidden layers)        1.808 [1.730, 1.884]              +0.000 [+0.000, +0.000]  <- zero new legal moves found
untrained encoder probe control                   2.661 [2.581, 2.743]              +0.000 [+0.000, +0.000]  <- random encoder has no effect
wrong-square control (steer unrelated square)     3.435 [3.337, 3.530]              +0.001 [+0.001, +0.002]  <- degrades legal move predictions
contrastive flip at resid_post (after bug fixes)  0.012 [0.004, 0.026]              +0.079 [+0.073, +0.085]  <- 8.5x fewer errors (12 / 1,001 boards)
```

---

## 1. The Question and Why It Matters for World Models Now

When a probe decodes a world-state variable from a model's activations and steering along that probe changes the model's output, two things can go wrong:

1. **Shortcut bias vs. real world-model computation:** In a transformer, every layer adds its output to a shared residual stream that connects straight to the final output layer. When you add a steering vector across middle layers (Layers 2 to 7), that vector also travels untouched down the skip connection into the final readout weights. Does the intervention work because intermediate Attention and MLP layers actually recompute which moves are legal on the edited board, or just because the steering vector acts as a direct bias on the final output logits?
2. **Reading state vs. controlling predictions:** If a linear probe reads out the board state with 99.4% accuracy at Layer 5, does editing Layer 5 alone change the model's move predictions, or do later layers ignore single-layer edits?

**Why this matters now:** Linear probes and activation steering are standard tools for claiming that video, robotics, and market models have learned internal world models. If a representation is 99% decodable at a given layer yet causally ignored unless edited across a specific circuit span—or if steering works mainly by biasing the final readout layer—standard probing evals will overstate whether a model can actually be used for counterfactual planning.

---

## 2. Method: Model, Data, Metrics, and Controls

- **Model:** The pretrained 8-layer, 8-head Othello-GPT checkpoint (`synthetic_model.pth`, width 512) and its per-layer linear probes (`resid_0..7_linear.pth`) from Nanda et al. (2023).
- **Data:** All **1,001 counterfactual test boards** from `intervention_benchmark.pkl`. Each test case takes a game history producing board $B$, flips a single occupied square between `Yours` and `Mine` to create a counterfactual board $B'$, and tests whether the model's predicted legal moves match the new board $B'$.
- **Metrics (reported with 1,000-sample bootstrap 95% CIs):**
  - **Move errors on the edited board (lower is better):** If the edited board $B'$ has $K$ legal moves, we take the model's top $K$ predicted moves and count how many are wrong (illegal moves predicted plus legal moves missed). This is the exact error metric in Nanda et al. (2023) Table 2.
  - **Probability gain on newly legal moves (higher is better):** The total increase in probability assigned to moves that were illegal on the original board $B$ and only become legal after flipping the tile.
- **Controls to rule out alternative explanations:**
  - **Mediated-only (subtracting the skip connection):** We add the steering vector at Layers 2–7 so middle layers process the edited board, then **subtract the accumulated steering vector right before the final LayerNorm and output layer**. The direct skip path to the output layer sums to zero; any remaining effect must come from intermediate Attention and MLP layers.
  - **Direct-readout bypass (skip connection only):** We leave all hidden layers untouched and add the total steering vector **only** right before the final output layer. This tests how much of the effect comes from directly biasing the final readout weights.
  - **Untrained-encoder probe control:** We initialize an identical 8-layer GPT with random weights (matched variance), fit ridge probes on 200 games of its activations, and steer the trained model along those directions.
  - **Wrong-square control:** We steer using the trained probe direction for a different square than the one that was flipped, at the same intervention strength ($\alpha = 2.3$).

---

## 3. Experiments and Findings

### Finding 1: Intermediate Layers Recompute the Rules; Direct Readout Bias Finds Zero New Legal Moves

![Transformer Intervention Architecture and ROME Causal Traces](fig_transformer_intervention.png)
*Figure 1: **(a)** Othello-GPT decoder diagram rendered via [`negrinho/sane_tikz`](https://github.com/negrinho/sane_tikz) (`examples/transformer.py`), showing the three intervention points along the residual stream: the original pre-MLP hook (`attn_out`), the post-MLP probe site (`resid_post`), and the skip-connection subtraction before the final linear readout (`h_8`). **(b–e)** Layer-by-layer causal traces on Case #101 (flipping square `C4` from White to Black at move 17) rendered via [`kmeng01/rome`](https://github.com/kmeng01/rome) (`plot_trace_heatmap`).*

When we subtract out the direct skip connection before the final output layer (**Mediated-only**), move errors still drop from `2.723` to **`0.158`**, keeping **97.9%** of the full multi-layer intervention's error reduction (`0.102`).

Tracing individual layers on Case #101 (flipping `C4` from White to Black, which makes `B5` and `C5` newly legal and invalidates `F7`) shows where this happens:
- **MLPs in Layers 5 and 6 compute the new flanking moves (Figure 1c–d):** Layer 5 and Layer 6 MLPs boost the logits of the newly legal moves `B5` and `C5` and suppress the invalidated move `F7` by `+4.62` and `+27.09` logits, followed by `+5.63` logits from Layer 7 attention.
- **Direct readout bias cannot discover new legal moves (Figure 2c):** Adding the steering vector only at the final readout layer (`Direct Bypass Only`) reduces error modestly from `2.723` to `1.808` because dot products with the output weights slightly lower the probability of the old move `F7` (`0.12 -> 0.06`), but it assigns **`0.00` probability** to the newly legal moves `B5` and `C5` (`+0.000` across all 1,001 boards). Discovering that flipping `C4` opens up `B5` and `C5` requires multi-step flanking calculation inside the middle layers.

![Board-Level Causal Visualization on Case #101](fig_board_intervention.png)
*Figure 2: Board-level visualization on Case #101 (flipping `C4` from White to Black), rendered via [`likenneth/othello_world`](https://github.com/likenneth/othello_world) (`plot_probs`, `plot_mentals`, `OthelloBoardState`). **Top (a–e):** Move probability heatmaps. `Direct Bypass Only` (c) assigns `0.00` to the newly legal moves `B5` and `C5` (4 errors), whereas `Mediated-Only` (d) and `Contrastive Flip` (e) suppress `F7` (`0.12 -> 0.00`) and activate `B5` and `C5` with **0 errors**. **Bottom (f–j):** Probe beliefs across layers (`O` = White, `X` = Black), showing how a single-layer edit at Layer 4 is overwritten by Layer 5 (g–h, red box), whereas steering Layers 2–7 sustains the flipped piece through Layer 7 (i–j, green box).*

---

### Finding 2: 99.4% Probe Accuracy at Layer 5 Gives Almost No Single-Layer Control Due to Self-Repair

![Aggregate Results Across N=1,001 Boards](fig_causal_check.png)
*Figure 3: **(A)** Move error on the edited board across intervention scales $\alpha$. **(B)** Layer-by-layer 3-way probe accuracy (green/gray, right axis) compared against single-layer edit error (red) and cumulative multi-layer span `[l..7]` error (blue).*

| Layer | `L0` | `L1` | `L2` | `L3` | `L4` | `L5` | `L6` | `L7` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trained Probe Accuracy (3-Way)** | `90.4%` | `94.5%` | `97.1%` | `98.2%` | `99.0%` | `99.4%` | **`99.5%`** | `99.4%` |
| **Untrained GPT Probe Accuracy** | `66.2%` | `65.1%` | `64.4%` | `63.9%` | `63.4%` | `63.0%` | `62.6%` | `62.6%` |
| **Single-Layer Edit Error (edit layer `l` only)** | `2.677` | `2.679` | `2.633` | `2.535` | `2.370` | **`2.188`** | `2.398` | `2.673` |
| **Multi-Layer Span Error (edit layers `l..7`)** | `0.390` | `0.166` | `0.102` | **`0.072`** | `0.192` | `1.345` | `2.268` | `2.673` |

Even though linear probes decode the board with `99.0%–99.5%` accuracy in Layers 4–7 (while the untrained GPT control decays from `66.2%` to `62.6%` as random layers scramble the input tokens), editing any single layer alone never gets move error below `2.188` (Layer 5). Two mechanisms explain this gap:

- **Downstream self-repair (Figure 1b,e & Figure 2f–h):** When we edit Layer 4 alone on Case #101, Layer 4's probe belief at `C4` cleanly flips to Black (`X`, probability `1.00`, Figure 2f). Immediately in Layer 5, however, the MLP (`+2.24` repair pull) and later attention heads re-read the unedited move history and flip `C4` right back to White (`O`, probability `0.93`, Figure 2g–h).
- **Probe accuracy peaks *after* the causal circuit has already run:** Intervening across a span of layers `l..7` works best when started at **Layer 3 (`0.072` errors)**, but fails if started at **Layer 5 (`1.345` errors)** or **Layer 6 (`2.268` errors)**—even though probe accuracy is highest at Layers 5 and 6. By Layer 5, the flanking-detection circuits in Layers 3 and 4 have already run on the unedited board state.

---

### Finding 3: Fixing Three Bugs in `mech_int/intervene.py` Cuts Move Error by 8.5x (`0.102 -> 0.012`)

While reproducing Nanda et al. (2023), we inspected `mech_int/intervene.py` (lines 140–162) in [`ajyl/mech_int_othelloGPT`](https://github.com/ajyl/mech_int_othelloGPT) and found three implementation bugs where the code diverges from the paper's description:

| Bug in `mech_int/intervene.py` | What the Reference Code Does | What We Changed | Move Error on $B'$ |
| :--- | :--- | :--- | :---: |
| **Baseline (`intervene.py` as written)** | — | — | `0.102` `[0.072, 0.136]` |
| **1. Hardcoded Layer-5 probe** | Builds per-layer hooks `hook_fns[0..7]` but passes `hook_fns[5]` (`probes[5]`) into all steered layers `2..7` | Use each layer's own probe `probes[l]` at layer `l` | `0.150` `[0.116, 0.188]` |
| **2. Hook site mismatch** | Injects vectors at `hook_attn_out` (before the MLP), even though the probes were trained on `hook_resid_post` (after the MLP) | Move the intervention hook to `hook_resid_post` where the probes were trained | `0.076` `[0.054, 0.102]` |
| **3. One-sided vector addition** | Adds the new piece's probe vector (`+w_target`) without subtracting the old piece's vector (`-w_source`), leaving the old state active | Steer along the unit **contrastive flip vector** `(w_target - w_source) / \|\|w_target - w_source\|\|` at `hook_resid_post` | **`0.012`** `[0.004, 0.026]` (**12 errors across 1,001 boards**) |

Fixing the hook site and steering along the contrastive flip direction (`w_target - w_source`) flips the probe's belief on 99.6% of boards and reduces total benchmark errors from **102 errors to 12 errors** across the 1,001 boards.

---

## 4. What Would Change Our Conclusion, and What We Would Run Next Week

- **What result would change our conclusion:**
  1. *On mediated computation vs. direct readout bias:* Our conclusion would be refuted if freezing intermediate Attention and MLP outputs in Layers 3–7 to their clean (un-intervened) values while applying the steering vector at the final output layer recovered more than 50% of the error reduction or achieved `> +0.02` probability gain on newly legal moves (in practice it recovers 33.6% of total error via old-move suppression and `+0.000` on newly legal moves).
  2. *On downstream self-repair:* Our explanation that single-layer edits fail because later layers re-read earlier activations would be refuted if projecting out the old piece's direction at Layer 3 *and* blocking Layers 4–6 attention heads from attending to the move token where square $s$ was originally played still failed to push single-layer edit error below `0.50`.
- **What we would run with another week:**
  1. **Decomposing Attention vs. MLP self-repair:** Path-patch individual attention heads vs. MLP neurons in Layers 4–6 to measure how much of the self-repair comes from attention heads re-reading the move prefix versus MLP neurons enforcing local $3 \times 3$ board heuristics ([Lin et al., 2024](https://arxiv.org/abs/2403.03981)).
  2. **Late-game `MoveFirst` audit (turns `t > 30`):** Split the 1,001 benchmark boards by game turn $t$ and compare logistic-probe normal vectors against simple difference-in-means vectors (`mean(Mine) - mean(Yours)`) in endgame states, where models often compute legal moves before representing the full board.

---

## 5. What I Want to Work On With Physical Reasoning

I want to work on **Evaluating World Models Under Intervention**—specifically building [**`physicalreasoning/forward` Issue #5 (*"Design an intervention test for world models"*)**](https://github.com/physicalreasoning/forward/issues/5), using the same untrained-encoder and shortcut-control discipline as [`causal-jepa`](https://github.com/physicalreasoning/causal-jepa).

In my research at the **CausalML Lab** (modeling multi-agent failures via soft interventions) and on **hierarchical traffic scene synthesis**, models with nearly identical observational rollout loss routinely diverge when tested under counterfactual interventions. Concretely, I want to build a lightweight, deterministic 2D physics and traffic intervention benchmark for `forward` where:
1. Two world models (e.g., a pixel-reconstruction dynamics model vs. a latent-state / JEPA or object-factored model) are trained to matched observational validation loss on the same trajectories;
2. The simulator is reset to held-out states and re-run under unshown impulse or mass changes; and
3. Models are evaluated against a pre-registered gate on their **counterfactual consistency gap** alongside untrained-encoder and direct-readout controls.

---

## Reproduce

```bash
# 1. Run all 25 causal intervention & control arms across N = 1,001 boards (~2 min on GPU)
#    Writes full metrics + bootstrap CIs to results.json and plots fig_causal_check.png
python run_experiment.py

# 2. Generate Figure 1 (via negrinho/sane_tikz + kmeng01/rome) and Figure 2 (via likenneth/othello_world)
python generate_paper_visuals.py
```

*AI tools note:* AI coding assistants (Antigravity / Gemini) were used to scaffold the PyTorch forward-pass hooks, KV-cached batching, and plotting scripts (`sane_tikz`, `rome`, `othello_world`); the experimental design, skip-connection decomposition controls, `mech_int/intervene.py` codebase audit, and analysis were verified directly against executed benchmark runs.
