# Causal Check on Othello-GPT: Does Steering Use the World Model or Just Bias the Output Layer?

**Rohan Gupta** (`gupt1156@purdue.edu`)

---

## 1. Background and the Question Tested

### Testing whether a model understands the world it predicts

A central problem in evaluating world models is that **a model can make accurate predictions without understanding the underlying state of the world**. Even when a diagnostic ("probe") reads out the state of the world from a model's hidden layers, that still does not prove the model *uses* that representation to make decisions. The representation might just be a passive byproduct of the input history.

To test this cleanly, [Li et al. (2023)](https://github.com/likenneth/othello_world) and [Nanda et al. (2023)](https://github.com/ajyl/mech_int_othelloGPT) introduced **Othello-GPT**:
- **The game:** In Othello, two players take turns placing Black and White discs on an `8x8` board. A move is only legal if it flanks (sandwiches) a line of the opponent's pieces between the new disc and an existing disc of the active player's color.
- **The model (`Othello-GPT`):** An 8-layer GPT transformer trained *only* on sequences of legal moves (e.g., `D3, C5, F6, ...`). It is never shown the `8x8` board grid and is never given the rules of Othello; it only predicts which square token can legally be played next.
- **Reading the internal board (Linear Probes):** Despite only seeing move sequences, a simple linear classifier (a **linear probe**) trained on the model's hidden activations after Layer 4 decodes the exact state of all 64 squares on the board (`Empty`, `Yours`, or `Mine`) with **over 99% accuracy**.
- **The causal check (Activation Steering):** Testing whether Othello-GPT actually *uses* this internal board to pick its next move requires a **causal intervention** (also called **activation steering**). Mid-game, the intervention reaches into the model's hidden activations and pushes one square's representation from `White` to `Black`, effectively flipping a piece on the model's internal board (for example, flipping square `C4` from White to Black in Figure 2a below). Flipping that single piece changes which moves are legal on the board. If the model genuinely uses its internal board state, its next-move predictions should switch to the moves that are legal on the *new*, edited board.

### The two questions prior work left open

Nanda et al. (2023) showed that injecting the probe's "flip piece" vector across Layers 2 through 7 switches the model's predictions to the legal moves of the edited board. However, two fundamental questions remain:

1. **Does the intervention work because middle layers recompute the rules of Othello, or just because the added vector takes a shortcut straight to the final output layer?**
   In a Transformer, every layer reads from and writes to a shared pass-through channel called the **residual stream** (the vertical arrow on the left of Figure 1a), which connects straight to the final output layer. Adding a vector at Layers 2 through 7 does not just feed into the middle layers; that vector also travels untouched down the pass-through connection right into the final output weights. Did the model's predictions change because the middle layers actually recomputed which Othello moves are legal on the edited board, or simply because the injected vector directly nudged the final output scores?

2. **If the board is already 99.4% readable at Layer 5, why doesn't editing Layer 5 alone work?**
   By Layer 5, the linear probe reads the board with 99.4% accuracy. Yet flipping a piece only at Layer 5 (instead of across Layers 2 to 7 together) barely changes the model's move predictions at all. Why does a representation that is 99.4% accurate at a single layer fail to control the model when edited at that layer?

**Why this matters for world models now:** Linear probes and activation steering are widely used to claim that video, robotics, and market models have learned actionable world representations. If a representation is 99% readable at a layer yet ignored by the rest of the network when edited, or if steering works merely by biasing the final output layer via skip connections, standard probing benchmarks will overstate whether a learned world model can support counterfactual planning.

---

## 2. Method: Model, Data, Metrics, and Controls

The evaluation uses the official pretrained 8-layer `Othello-GPT` checkpoint (`synthetic_model.pth`) and its per-layer linear probes (`resid_0..7_linear.pth`) across all **1,001 counterfactual test boards** in `intervention_benchmark.pkl` from Li et al. (2023) and Nanda et al. (2023).

On each of the 1,001 boards, one occupied square `s` is flipped between `Yours` and `Mine`, changing the set of ground-truth legal moves from the original board `B` to a new counterfactual board `B'`. All results report 1,000-sample bootstrap 95% confidence intervals on two plain-English metrics:

- **Move errors on the edited board (lower is better):** If the edited board `B'` has `K` legal moves, this metric takes the model's top `K` predicted moves and counts how many are wrong (illegal moves predicted plus legal moves missed). An un-intervened model makes **2.723 errors** per board on average.
- **Probability gained on newly legal moves (higher is better):** Flipping a piece makes some previously illegal moves newly legal on `B'`. This metric measures how much extra probability the model assigns to these newly legal moves after the intervention.

### Controls to separate real computation from shortcuts

| Control Arm | Intervention Procedure | What It Tests |
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

Headline numbers across all 1,001 counterfactual boards (with 95% bootstrap confidence intervals):

```
                                              move errors on edited board    prob gain on newly legal moves
                                                  (lower is better)                (higher is better)
0. Un-intervened model (clean board)              2.723 [2.645, 2.799]              +0.000 [+0.000, +0.000]
1. Full multi-layer steering (Nanda et al.)       0.102 [0.072, 0.136]              +0.072 [+0.066, +0.077]
2. Mediated-only (shortcut subtracted out)        0.158 [0.122, 0.198]              +0.037 [+0.033, +0.041]  <- 97.9% of reduction survives
3. Direct-readout bypass (shortcut only)          1.808 [1.730, 1.884]              +0.000 [+0.000, +0.000]  <- zero new legal moves found
4. Untrained transformer probe control            2.661 [2.581, 2.743]              +0.000 [+0.000, +0.000]  <- random encoder has no effect
5. Wrong-square control (steer unrelated square)  3.435 [3.337, 3.530]              +0.001 [+0.001, +0.002]  <- degrades legal move predictions
```

---

## 4. Visual Walkthrough of Findings

### Finding 1: Middle Layers Recompute the Rules; the Output Shortcut Finds Zero New Legal Moves

> **Core result:** Removing the direct pass-through shortcut to the output layer (**Mediated-Only**) still reduces move error from `2.723` to **`0.158`**, preserving **97.9%** of the full steering improvement (`0.102`). Applying the vector only at the output layer (**Direct Bypass Only**) yields **`+0.000`** probability gain on newly legal moves.

![Transformer Intervention Architecture and ROME Causal Traces](fig_transformer_intervention.png)
*Figure 1: Architecture diagram (a, via [`negrinho/sane_tikz`](https://github.com/negrinho/sane_tikz)) and layer-by-layer causal traces on Case #101 (b-e, via [`kmeng01/rome`](https://github.com/kmeng01/rome)).*

**How to read Figure 1 (Case #101: flipping square `C4` from White to Black makes `B5`, `C5` newly legal and `F7` illegal):**
- **Panel `(a)` (Intervention sites):** Steering vectors are injected into the middle layers (`Layers 2 to 7`) and then **subtracted right before the final output layer (`h_8`)** so no added vector can directly bias the output weights.
- **Panels `(c, d)` (Where new legal moves are computed):** Once `C4` is flipped in earlier layers, the **Layer 5 and Layer 6 MLP blocks** recompute Othello's flanking rules, boosting the newly legal moves (`B5`, `C5`) and suppressing the newly illegal move (`F7`) by `+4.62` and `+27.09` logits.

![Board-Level Causal Visualization on Case #101](fig_board_intervention.png)
*Figure 2: Board-level move predictions (a-e) and internal probe states (f-j) on Case #101, generated via [`likenneth/othello_world`](https://github.com/likenneth/othello_world).*

**How to read Figure 2 (`8x8` board walkthrough on Case #101):**

| Figure 2 Panels | Condition | What Happens on the Board |
| :--- | :--- | :--- |
| **Top `(a, b)`** | **Clean vs. Edited Board** | Flipping `C4` from White to Black makes `F7` illegal and makes `B5` and `C5` newly legal. |
| **Top `(c)`** | **Direct Shortcut Only** | Assigns **`0.00` probability** to newly legal moves `B5` and `C5` (`4 errors`). Without middle layers running flanking checks, the output layer has no way to know that flipping `C4` makes `B5` and `C5` legal. |
| **Top `(d, e)`** | **Mediated-Only / Full Steering** | Middle layers suppress old move `F7` (`0.12 -> 0.00`) and activate newly legal moves `B5` and `C5` (**`0 errors`**). |
| **Bottom `(f-h)`** | **Single-Layer Edit (`Layer 4` only)** | Editing Layer 4 flips the probe readout at `C4` to Black (`X`) at Layer 4 `(f)`, but Layer 5 immediately overwrites `C4` back to White (`O`) `(g, h, red box)`. |
| **Bottom `(i, j)`** | **Multi-Layer Edit (`Layers 2 to 7`)** | Sustaining the edit across Layers 2 to 7 keeps `C4` flipped to Black (`X`) all the way through Layer 7 `(i, j, green box)`. |

---

### Finding 2: 99.4% Probe Accuracy at Layer 5 Gives Almost No Single-Layer Control Because Later Layers Overwrite It

> **Core result:** Even though linear probes read the board with **99.4% accuracy** at Layer 5, editing Layer 5 alone leaves move error at **`2.188`** (barely better than the `2.723` un-intervened baseline).

![Aggregate Results Across N=1,001 Boards](fig_causal_check.png)
*Figure 3: **(A)** Move error across intervention strengths `α`. **(B)** Layer-by-layer probe accuracy (green/gray) vs. single-layer edit error (red) and multi-layer span `[l..7]` error (blue).*

| Layer | `L0` | `L1` | `L2` | `L3` | `L4` | `L5` | `L6` | `L7` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trained Probe Accuracy (3-Way)** | `90.4%` | `94.5%` | `97.1%` | `98.2%` | `99.0%` | `99.4%` | **`99.5%`** | `99.4%` |
| **Untrained GPT Probe Accuracy** | `66.2%` | `65.1%` | `64.4%` | `63.9%` | `63.4%` | `63.0%` | `62.6%` | `62.6%` |
| **Single-Layer Edit Error (edit layer `l` only)** | `2.677` | `2.679` | `2.633` | `2.535` | `2.370` | **`2.188`** | `2.398` | `2.673` |
| **Multi-Layer Span Error (edit layers `l..7`)** | `0.390` | `0.166` | `0.102` | **`0.072`** | `0.192` | `1.345` | `2.268` | `2.673` |

**Why high single-layer probe accuracy fails to give single-layer control:**

1. **Later layers "self-repair" single-layer edits (Figure 1b,e and Figure 2f-h):**
   When a single layer (such as Layer 4) is edited on its own, its internal board representation at `C4` cleanly flips to Black (`X`, probability `1.00` in Figure 2f). In the very next layer, however, the Layer 5 MLP (`+2.24` pull toward White) and later attention heads re-read the unedited move history from earlier tokens and overwrite `C4` right back to White (`O`, probability `0.93` in Figure 2g-h).
2. **Probe accuracy peaks *after* the move-calculation circuit has already started (Figure 3B):**
   Probe accuracy is highest at Layers 5 and 6 (`99.4%` and `99.5%`), yet starting a multi-layer intervention at Layer 5 (`1.345` errors) or Layer 6 (`2.268` errors) largely fails. Causal control is strongest when starting at **Layer 3 (`0.072` errors)** because Othello's flanking rules are computed in Layers 3 and 4; by Layer 5, the move-selection circuit has already run on the old, unedited board state.

---

## 5. Falsification Criteria and Next Steps

- **What result would change these conclusions:**
  1. *On middle-layer computation vs. output shortcut:* The mediated-computation conclusion would be refuted if freezing intermediate Attention and MLP outputs in Layers 3 to 7 to their clean (un-intervened) values while applying the steering vector at the final output layer recovered more than 50% of the error reduction or achieved `> +0.02` probability gain on newly legal moves (in practice it recovers 33.6% via old-move suppression and `+0.000` on newly legal moves).
  2. *On self-repair overwriting single-layer edits:* The self-repair explanation would be refuted if erasing the old piece's direction at Layer 3 *and* blocking Layers 4 to 6 attention heads from looking back at the move token where square `s` was originally played still failed to bring single-layer edit error below `0.50`.
- **Next experiments with another week:**
  1. **Separating Attention vs. MLP self-repair:** Patch individual attention heads vs. MLP neurons in Layers 4 to 6 to measure how much of the overwrite comes from attention heads re-reading the move history versus MLP neurons enforcing local `3x3` board patterns ([Lin et al., 2024](https://arxiv.org/abs/2403.03981)).
  2. **Late-game `MoveFirst` audit (turns `t > 30`):** Split the 1,001 benchmark boards by game turn `t` and compare logistic-probe normal vectors against simple difference-in-means vectors (`mean(Mine) - mean(Yours)`) in late-game states, where models often compute legal moves before representing the full board.

---

## 6. What I Want to Work On With Physical Reasoning

I want to work on [**`physicalreasoning/forward` Issue #5 (*"Design an intervention test for world models"*)**](https://github.com/physicalreasoning/forward/issues/5), applying the same untrained-encoder and shortcut-control discipline used in [`causal-jepa`](https://github.com/physicalreasoning/causal-jepa).

In my research on multi-agent failure modeling (using soft interventions on Structural Causal Models at the **CausalML Lab**) and **hierarchical traffic scene synthesis**, I have seen that models with nearly identical observational rollout loss routinely diverge when tested under counterfactual interventions. With Physical Reasoning, I would love to build a lightweight 2D physics and traffic intervention benchmark in `forward` that:
1. Trains two world models (for example, a pixel-reconstruction dynamics model vs. a latent-state / JEPA or object-factored model) to matched observational validation loss on identical trajectories;
2. Resets the simulator to held-out states and re-runs under unshown impulse or mass changes; and
3. Scores models against a pre-registered gate on their **counterfactual consistency gap** alongside untrained-encoder and direct-readout controls.

---

## Reproduce

```bash
# 1. Run all causal intervention & control arms across N = 1,001 boards (~2 min on GPU)
#    Writes full metrics + bootstrap CIs to results.json and plots fig_causal_check.png
python run_experiment.py

# 2. Generate Figure 1 (via negrinho/sane_tikz + kmeng01/rome) and Figure 2 (via likenneth/othello_world)
python generate_paper_visuals.py
```

*AI tools note:* AI coding assistants (Antigravity / Gemini) were used to scaffold the PyTorch forward-pass hooks, KV-cached batching, and plotting scripts (`sane_tikz`, `rome`, `othello_world`); the experimental design, skip-connection decomposition controls, and analysis were verified directly against executed benchmark runs.
