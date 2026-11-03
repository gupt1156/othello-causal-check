# Causal Check on Othello-GPT: Does Steering Use the World Model or Just Bias the Output Layer?

**Rohan Gupta** (`gupt1156@purdue.edu`)

---

## 1. Background and the Question We Are Testing

### How do we know if a model actually understands the world it predicts?

A central problem in evaluating world models is that **a model can make accurate predictions without understanding the underlying state of the world**. Even if you train a diagnostic ("probe") that reads out the state of the world from a model's hidden layers, that still does not prove the model *uses* that representation to make decisions. The representation might just be a passive byproduct of the input history.

To test this cleanly, [Li et al. (2023)](https://github.com/likenneth/othello_world) and [Nanda et al. (2023)](https://github.com/ajyl/mech_int_othelloGPT) introduced **Othello-GPT**:
- **The game:** In Othello, two players take turns placing Black and White discs on an $8 \times 8$ board. A move is only legal if it flanks (sandwiches) a line of the opponent's pieces between the new disc and an existing disc of your own color.
- **The model (`Othello-GPT`):** An 8-layer GPT transformer trained *only* on sequences of legal moves (e.g., `D3, C5, F6, ...`). It is never shown the $8 \times 8$ board grid and is never given the rules of Othello; it only predicts which square token can legally be played next.
- **Reading the internal board (Linear Probes):** Despite only seeing move sequences, if you train a simple linear classifier (a **linear probe**) on the model's hidden activations after Layer 4, you can read out the exact state of all 64 squares on the board (`Empty`, `Yours`, or `Mine`) with **over 99% accuracy**.
- **The causal check (Activation Steering):** To test whether Othello-GPT actually *uses* this internal board to pick its next move, you perform a **causal intervention** (also called **activation steering**). Mid-game, you reach into the model's hidden activations and push one square's representation from `White` to `Black`, effectively flipping a piece on the model's internal board (for example, flipping square `C4` from White to Black in Figure 2a below). Flipping that single piece changes which moves are legal on the board. If the model genuinely uses its internal board state, its next-move predictions should switch to the moves that are legal on the *new*, edited board.

### The two questions prior work left open

Nanda et al. (2023) showed that if you inject the probe's "flip piece" vector across Layers 2 through 7, the model's predictions switch to the legal moves of the edited board. However, two fundamental questions remain:

1. **Does the intervention work because middle layers recompute the rules of Othello, or just because the added vector takes a shortcut straight to the final output layer?**
   In a Transformer, every layer reads from and writes to a shared pass-through channel called the **residual stream** (the vertical arrow on the left of Figure 1a), which connects straight to the final output layer. When you add a vector at Layers 2 through 7, that vector does not just go into the middle layers; it also travels untouched down the pass-through connection right into the final output weights. Did the model's predictions change because the middle layers actually recomputed which Othello moves are legal on the edited board, or simply because the injected vector directly nudged the final output scores?

2. **If the board is already 99.4% readable at Layer 5, why doesn't editing Layer 5 alone work?**
   By Layer 5, the linear probe reads the board with 99.4% accuracy. Yet if you flip a piece only at Layer 5 (instead of across Layers 2 to 7 together), the model's move predictions barely change at all. Why does a representation that is 99.4% accurate at a single layer fail to control the model when you edit that layer?

**Why this matters for world models now:** Linear probes and activation steering are widely used to claim that video, robotics, and market models have learned actionable world representations. If a representation is 99% readable at a layer yet ignored by the rest of the network when edited, or if steering works merely by biasing the final output layer via skip connections, standard probing benchmarks will overstate whether a learned world model can support counterfactual planning.

---

## 2. Method: Model, Data, Metrics, and Controls

We test the official pretrained 8-layer `Othello-GPT` checkpoint (`synthetic_model.pth`) and its per-layer linear probes (`resid_0..7_linear.pth`) across all **1,001 counterfactual test boards** in `intervention_benchmark.pkl` from Li et al. (2023) and Nanda et al. (2023).

On each of the 1,001 boards, one occupied square $s$ is flipped between `Yours` and `Mine`, changing the set of ground-truth legal moves from the original board $B$ to a new counterfactual board $B'$. We report 1,000-sample bootstrap 95% confidence intervals on two plain-English metrics:

- **Move errors on the edited board (lower is better):** If the edited board $B'$ has $K$ legal moves, we take the model's top $K$ predicted moves and count how many are wrong (illegal moves predicted plus legal moves missed). An un-intervened model makes **2.723 errors** per board on average.
- **Probability gained on newly legal moves (higher is better):** Flipping a piece makes some previously illegal moves newly legal on $B'$. We measure how much extra probability the model assigns to these newly legal moves after the intervention.

### Controls to separate real computation from shortcuts

| Control Arm | What We Do | What It Tests |
| :--- | :--- | :--- |
| **1. Full Multi-Layer Steering (Nanda et al. baseline)** | Add the probe vector for the flipped square across Layers 2 to 7 and let it pass all the way to the output layer. | Reproduces Nanda et al. (2023) Table 2 (`0.102` move errors). |
| **2. Mediated-Only (Remove the output shortcut)** | Add the probe vector across Layers 2 to 7 so the middle layers process the edited board, then **subtract the accumulated vector right before the final output layer**. | Cancels out the direct pass-through shortcut to the output layer. Any improvement that survives *must* come from middle Attention and MLP layers recomputing the board. |
| **3. Direct-Readout Bypass (Shortcut only, no middle layers)** | Leave all hidden layers (Layers 0 to 7) completely untouched, and add the total steering vector **only** right before the final output layer. | Tests how much of the effect comes purely from directly biasing the final output layer without any middle-layer computation. |
| **4. Untrained-Transformer Control** | Take a randomly initialized 8-layer GPT with the same architecture and weight scale, fit linear probes on 200 games of its activations, and steer the trained model with those vectors. | Tests whether arbitrary high-dimensional probe directions can steer the model. |
| **5. Wrong-Square Control** | Steer the model using the trained probe vector for a *different* square than the one that was actually flipped. | Tests whether the intervention is spatially specific to the flipped square. |

---

## 3. Summary of Results

| Question | Finding |
| :--- | :--- |
| **Is steering just biasing the final output layer via the pass-through connection?** | **No.** Subtracting the entire injected vector right before the final output layer (**Mediated-Only**) still reduces move errors from `2.723` to **`0.158`**, preserving **97.9%** of the full intervention effect. |
| **Can the direct shortcut to the output layer discover newly legal moves on its own?** | **No.** Adding the vector only at the final output layer (**Direct-Readout Bypass**) gives **`+0.000`** probability gain on newly legal moves (`1.808` errors). |
| **Does 99.4% probe accuracy at Layer 5 mean editing Layer 5 alone changes the model's move?** | **No.** Editing Layer 5 alone leaves error at `2.188`. Later layers (Layers 5 to 7) look back at the unedited move history and overwrite ("self-repair") single-layer edits. |
| **Does the layer with the highest probe accuracy give the best causal control?** | **No.** Probe accuracy peaks at Layer 6 (`99.5%`), but steering from Layer 5 or 6 onward fails (`1.345` and `2.268` errors) because Othello flanking rules are already computed earlier, in Layers 3 and 4. |
| **Do probes from an untrained transformer steer moves?** | **No.** Steering with probes from an untrained GPT gets `2.661` errors and `+0.000` probability gain on newly legal moves. |
| **Does the official `mech_int/intervene.py` benchmark code match the paper?** | **No.** We found three bugs in the reference repo: it hardcodes Layer 5's probe into all layers, hooks before the MLP instead of after, and adds the new piece vector without subtracting the old piece vector. Fixing all three cuts move error by **8.5x** (`0.102 -> 0.012`, or just **12 total errors across 1,001 boards**). |

Here are the headline numbers across all 1,001 counterfactual boards (with 95% bootstrap confidence intervals):

```
                                              move errors on edited board    prob gain on newly legal moves
                                                  (lower is better)                (higher is better)
0. Un-intervened model (clean board)              2.723 [2.645, 2.799]              +0.000 [+0.000, +0.000]
1. Full multi-layer steering (Nanda et al.)       0.102 [0.072, 0.136]              +0.072 [+0.066, +0.077]
2. Mediated-only (shortcut subtracted out)        0.158 [0.122, 0.198]              +0.037 [+0.033, +0.041]  <- 97.9% of reduction survives
3. Direct-readout bypass (shortcut only)          1.808 [1.730, 1.884]              +0.000 [+0.000, +0.000]  <- zero new legal moves found
4. Untrained transformer probe control            2.661 [2.581, 2.743]              +0.000 [+0.000, +0.000]  <- random encoder has no effect
5. Wrong-square control (steer unrelated square)  3.435 [3.337, 3.530]              +0.001 [+0.001, +0.002]  <- degrades legal move predictions
6. Fix Bug 1: Layer-matched probes @ attn_out     0.150 [0.116, 0.188]              +0.071 [+0.065, +0.076]  <- replaces hardcoded Layer-5 probe
7. Fix Bug 2: Layer-matched probes @ resid_post   0.076 [0.054, 0.102]              +0.075 [+0.069, +0.081]  <- matches hook to probe training site
8. Fix Bug 3: Contrastive flip @ resid_post       0.012 [0.004, 0.026]              +0.079 [+0.073, +0.085]  <- 8.5x fewer errors (12 / 1,001 boards)
```

---

## 4. Detailed Visuals and Walkthrough of Findings

### Finding 1: Middle Layers Recompute the Rules; the Output Shortcut Finds Zero New Legal Moves

![Transformer Intervention Architecture and ROME Causal Traces](fig_transformer_intervention.png)
*Figure 1: **(a)** Othello-GPT architecture diagram generated via [`negrinho/sane_tikz`](https://github.com/negrinho/sane_tikz) (`examples/transformer.py`), showing the three places we intervene along the residual stream: (1) the original pre-MLP hook (`attn_out`), (2) the post-MLP probe site (`resid_post`), and (3) the skip-connection subtraction right before the final output layer (`h_8`). **(b-e)** Layer-by-layer causal traces on Case #101 (flipping square `C4` from White to Black at move 17) generated via [`kmeng01/rome`](https://github.com/kmeng01/rome) (`plot_trace_heatmap`).*

When we cancel out the direct pass-through connection right before the final output layer (**Mediated-Only**), move errors still drop from `2.723` to **`0.158`**, retaining **97.9%** of the full multi-layer intervention's error reduction (`0.102`).

We can see exactly how the network computes the new moves by looking at a concrete board (**Case #101** in Figures 1 and 2, where flipping square `C4` from White to Black makes `B5` and `C5` newly legal and makes `F7` illegal):
- **MLPs in Layers 5 and 6 compute the new flanking moves (Figure 1c-d):** Once the board state at `C4` is flipped in earlier layers, the Feed-Forward (MLP) blocks in Layer 5 and Layer 6 boost the output scores of the newly legal moves `B5` and `C5` and suppress the invalidated move `F7` by `+4.62` and `+27.09` logits, followed by `+5.63` logits from Layer 7 attention.
- **Why the direct shortcut fails (Figure 2c):** When we skip the middle layers and add the steering vector only at the final output layer (`Direct Bypass Only`), the vector slightly lowers the score of the old move `F7` (`0.12 -> 0.06`), which reduces average error from `2.723` to `1.808`. However, it assigns **`0.00` probability** to the newly legal moves `B5` and `C5` (`+0.000` across all 1,001 boards). Without the middle MLP and Attention layers running their flanking checks, the final output layer has no way to know that flipping `C4` makes `B5` and `C5` legal.

![Board-Level Causal Visualization on Case #101](fig_board_intervention.png)
*Figure 2: Board-level visualization on Case #101 (flipping `C4` from White to Black), generated via [`likenneth/othello_world`](https://github.com/likenneth/othello_world) (`plot_probs`, `plot_mentals`, `OthelloBoardState`). **Top row (a-e):** Move probability heatmaps on the $8 \times 8$ board. `Direct Bypass Only` (c) assigns `0.00` to the newly legal moves `B5` and `C5` (4 errors), whereas `Mediated-Only` (d) and `Contrastive Flip` (e) suppress `F7` (`0.12 -> 0.00`) and activate `B5` and `C5` with **0 errors**. **Bottom row (f-j):** What the linear probe believes is on each square (`O` = White, `X` = Black). Editing Layer 4 alone flips `C4` to Black `X` at Layer 4 (f), but Layer 5 immediately overwrites `C4` back to White `O` (g-h, red box). Steering across Layers 2 to 7 keeps `C4` flipped to `X` all the way to Layer 7 (i-j, green box).*

---

### Finding 2: 99.4% Probe Accuracy at Layer 5 Gives Almost No Single-Layer Control Because Later Layers Overwrite It

![Aggregate Results Across N=1,001 Boards](fig_causal_check.png)
*Figure 3: **(A)** Move error on the edited board across intervention strengths $\alpha$. **(B)** Layer-by-layer 3-way probe accuracy (green/gray, right axis) compared against single-layer edit error (red) and multi-layer span `[l..7]` error (blue).*

| Layer | `L0` | `L1` | `L2` | `L3` | `L4` | `L5` | `L6` | `L7` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trained Probe Accuracy (3-Way)** | `90.4%` | `94.5%` | `97.1%` | `98.2%` | `99.0%` | `99.4%` | **`99.5%`** | `99.4%` |
| **Untrained GPT Probe Accuracy** | `66.2%` | `65.1%` | `64.4%` | `63.9%` | `63.4%` | `63.0%` | `62.6%` | `62.6%` |
| **Single-Layer Edit Error (edit layer `l` only)** | `2.677` | `2.679` | `2.633` | `2.535` | `2.370` | **`2.188`** | `2.398` | `2.673` |
| **Multi-Layer Span Error (edit layers `l..7`)** | `0.390` | `0.166` | `0.102` | **`0.072`** | `0.192` | `1.345` | `2.268` | `2.673` |

Even though linear probes read the board with `99.0% to 99.5%` accuracy in Layers 4 to 7 (while the untrained GPT control drops from `66.2%` to `62.6%` as random layers scramble the move history), editing any single layer on its own never gets move error below `2.188` (Layer 5). Two things explain why:

1. **Later layers "self-repair" single-layer edits (Figure 1b,e & Figure 2f-h):** When we edit Layer 4 alone on Case #101, Layer 4's probe belief at `C4` cleanly flips to Black (`X`, probability `1.00`, Figure 2f). Immediately in Layer 5, however, the Layer 5 MLP (`+2.24` pull toward White) and later attention heads re-read the unedited move history from the rest of the board and overwrite `C4` right back to White (`O`, probability `0.93`, Figure 2g-h).
2. **Probe accuracy peaks *after* the move-calculation circuit has already started:** Intervening across a continuous span of layers `l..7` works best when started at **Layer 3 (`0.072` errors)**, and fails when started at **Layer 5 (`1.345` errors)** or **Layer 6 (`2.268` errors)**, even though probe accuracy is highest at Layers 5 and 6. By Layer 5, the flanking-detection circuits in Layers 3 and 4 have already run on the old, unedited board state.

---

### Finding 3: Fixing Three Bugs in `mech_int/intervene.py` Cuts Move Error by 8.5x (`0.102 -> 0.012`)

While reproducing Nanda et al. (2023), we audited `mech_int/intervene.py` (lines 140 to 162) in [`ajyl/mech_int_othelloGPT`](https://github.com/ajyl/mech_int_othelloGPT) and found three bugs where the released code does not match the paper's description:

| Bug in `mech_int/intervene.py` | What the Released Code Does | How We Fixed It | Move Error on $B'$ |
| :--- | :--- | :--- | :---: |
| **Reference baseline (`intervene.py` as released)** | Unmodified reference script | Unmodified | `0.102` `[0.072, 0.136]` |
| **Bug 1: Hardcoded Layer-5 probe** | Builds per-layer hooks `hook_fns[0..7]` but passes `hook_fns[5]` (`probes[5]`) into every steered layer `2..7` | Use each layer's own probe `probes[l]` at layer `l` (`Arm 6`) | `0.150` `[0.116, 0.188]` |
| **Bug 2: Hook site mismatch** | Injects vectors at `hook_attn_out` (before the MLP), even though the probes were trained on `hook_resid_post` (after the MLP) | Move the intervention hook to `hook_resid_post` where the probes were actually trained (`Arm 7`) | `0.076` `[0.054, 0.102]` |
| **Bug 3: One-sided vector addition** | Adds the new piece's probe vector (`+w_target`) without subtracting the old piece's vector (`-w_source`), leaving the old piece active in the residual stream | Steer along the unit **contrastive flip vector** `(w_target - w_source) / \|\|w_target - w_source\|\|` at `hook_resid_post` (`Arm 8`) | **`0.012`** `[0.004, 0.026]` (**12 errors across 1,001 boards**) |

Aligning the hook site with where the probes were trained and steering along the contrastive flip direction (`w_target - w_source`) flips the probe's belief on 99.6% of boards and cuts total benchmark errors from **102 errors down to 12 errors** across the 1,001 boards.

---

## 5. What Would Change Our Conclusion, and What We Would Run Next Week

- **What result would change our conclusion:**
  1. *On middle-layer computation vs. output shortcut:* Our conclusion would be refuted if freezing intermediate Attention and MLP outputs in Layers 3 to 7 to their clean (un-intervened) values while applying the steering vector at the final output layer recovered more than 50% of the error reduction or achieved `> +0.02` probability gain on newly legal moves (in practice it recovers 33.6% via old-move suppression and `+0.000` on newly legal moves).
  2. *On self-repair overwriting single-layer edits:* Our explanation would be refuted if erasing the old piece's direction at Layer 3 *and* blocking Layers 4 to 6 attention heads from looking back at the move token where square $s$ was originally played still failed to bring single-layer edit error below `0.50`.
- **What we would run with another week:**
  1. **Separating Attention vs. MLP self-repair:** Patch individual attention heads vs. MLP neurons in Layers 4 to 6 to measure how much of the overwrite comes from attention heads re-reading the move history versus MLP neurons enforcing local $3 \times 3$ board patterns ([Lin et al., 2024](https://arxiv.org/abs/2403.03981)).
  2. **Late-game `MoveFirst` audit (turns `t > 30`):** Split the 1,001 benchmark boards by game turn $t$ and compare logistic-probe normal vectors against simple difference-in-means vectors (`mean(Mine) - mean(Yours)`) in late-game states, where models often compute legal moves before representing the full board.

---

## 6. What I Want to Work On With Physical Reasoning

I want to work on **Evaluating World Models Under Intervention**, specifically building [**`physicalreasoning/forward` Issue #5 (*"Design an intervention test for world models"*)**](https://github.com/physicalreasoning/forward/issues/5), using the same untrained-encoder and shortcut-control discipline as [`causal-jepa`](https://github.com/physicalreasoning/causal-jepa).

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
