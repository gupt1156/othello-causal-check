# Causal Check on Othello-GPT: Does Steering Use the World Model or Just Bias the Output Layer?

**Rohan Gupta** (`gupt1156@purdue.edu`)

---

## 1. The Question and Why It Matters Now

A central problem in evaluating world models is that **a model can make accurate predictions without understanding the underlying state of the world**. Even when a diagnostic ("probe") reads out the state of the world from a model's hidden layers, that still does not prove the model *uses* that representation to make decisions.

To test this cleanly, [Li et al. (2023)](https://github.com/likenneth/othello_world) and [Nanda et al. (2023)](https://github.com/ajyl/mech_int_othelloGPT) introduced **Othello-GPT**:
- **The game:** In Othello, two players take turns placing Black and White discs on an `8x8` board. A move is only legal if it flanks (sandwiches) a line of the opponent's pieces between the new disc and an existing disc of the active player's color.
- **The model (`Othello-GPT`):** An 8-layer GPT transformer trained *only* on sequences of legal moves (e.g., `E6, F4, D3, D6, E3, C4, ...`). It is never shown the `8x8` board grid and is never given the rules of Othello; it only predicts which square token can legally be played next.
- **Reading and editing the internal board:** Despite only seeing move sequences, a simple linear classifier (a **linear probe**) trained on the model's hidden activations decodes the exact state of all 64 squares (`Empty`, `Yours`, or `Mine`) with **over 99% accuracy**. Furthermore, reaching into hidden Layers 2 through 7 mid-game and pushing one square's representation from `White` to `Black` (**activation steering**) causes the model's next-move predictions to switch to the legal moves of the *edited* board.

### The causal question left open

In a Transformer, every layer reads from and writes to a shared pass-through channel called the **residual stream**, which connects straight to the final output layer. Adding a "flip piece" vector across Layers 2 through 7 does not just feed into the middle layers; that vector also travels untouched down the pass-through connection right into the final output weights.

> **Question:** When steering flips a piece on Othello-GPT's internal board, do the model's move predictions change because the middle layers actually recompute the rules of Othello over the edited board, or simply because the injected vector travels down the pass-through connection and directly biases the final output layer?

**Why this matters for world models now:** Linear probes and activation steering are widely used to claim that video, robotics, and market models have learned actionable world representations. If steering works merely by biasing the final output layer via skip connections, standard intervention benchmarks will overstate whether a learned world model can support counterfactual planning.

---

## 2. Method: Model, Data, Metrics, and Controls

<p align="center">
  <img src="fig_official_intervention.png" width="540" alt="Pre-Intervention vs Post-Intervention Board and Predictions (Li et al., 2023)">
</p>

*Figure from [Li et al. (2023)](https://arxiv.org/abs/2210.13382): **Bottom row (`World State`):** Mid-game, the intervention flips a single piece on the model's internal board (here, square `E6` in the blue box is flipped from Black to White). **Top row (`Prediction`):** Because `E6` is now White, moves `E7` and `F6` are no longer legal for White (`16% -> 0%`), while square `B3` becomes newly legal (`0% -> 17%`).*

The evaluation uses the official pretrained 8-layer `Othello-GPT` checkpoint (`synthetic_model.pth`) and its per-layer linear probes (`resid_0..7_linear.pth`) across all **1,001 counterfactual test boards** in `intervention_benchmark.pkl` from Li et al. (2023) and Nanda et al. (2023).

On each of the 1,001 boards, one occupied square is flipped between `Yours` and `Mine` (like square `E6` above), changing the ground-truth legal moves from the original board `B` to a new edited board `B'`. All results report 1,000-sample bootstrap 95% confidence intervals on two plain-English metrics:

- **Move errors on the edited board (lower is better):** If the edited board `B'` has `K` legal moves (such as the `K = 5` red squares in the top-right `Post-Intervention` board above: `B3, B4, C6, D3, F4`), this metric takes the model's top `K` predicted moves and counts how many do not match `B'` (illegal moves predicted plus legal moves missed). Without intervention, the model still predicts moves for the old board and makes **2.723 errors** per board on average.
- **Probability gained on newly legal moves (higher is better):** Flipping a piece makes some previously illegal moves newly legal on `B'` (such as square `B3` above, which jumps from `0%` to `17%`). This metric measures how much extra probability the model assigns to these newly legal moves after the intervention.

### Controls to separate middle-layer rule computation from the output shortcut

| Control Arm | Intervention Procedure | What It Tests |
| :--- | :--- | :--- |
| **1. Full Multi-Layer Steering (Nanda et al. baseline)** | Add the probe vector for the flipped square across Layers 2 to 7 and let it pass all the way to the output layer. | Reproduces Nanda et al. (2023) Table 2 (`0.102` move errors). |
| **2. Mediated-Only (Remove the output shortcut)** | Add the probe vector across Layers 2 to 7 so the middle layers process the edited board, then **subtract the accumulated vector right before the final output layer**. | Cancels out the direct pass-through shortcut to the output layer. Any improvement that survives *must* come from the middle layers recomputing legal moves on the edited board. |
| **3. Direct-Readout Bypass (Shortcut only, skip middle layers)** | Leave all hidden layers (Layers 0 to 7) completely untouched, and add the total steering vector **only** right before the final output layer. | Tests how much of the effect comes purely from directly biasing the final output layer without any middle-layer computation. |
| **4. Untrained-Transformer Control** | Take a randomly initialized 8-layer GPT with the same architecture and weight scale, fit linear probes on 200 games of its activations, and steer the trained model with those vectors. | Tests whether random probe directions from an untrained network can steer the model. |

---

## 3. Results and Conclusion

| Intervention Condition | Move Errors on Edited Board *(lower is better)* | 95% Confidence Interval | Prob. Gain on Newly Legal Moves *(higher is better)* | 95% Confidence Interval | Plain-English Takeaway |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **0. Un-intervened model** *(clean board)* | `2.723` | `[2.645, 2.799]` | `+0.000` | `[+0.000, +0.000]` | Baseline when predicting moves for the old board |
| **1. Full multi-layer steering** *(Nanda et al.)* | **`0.102`** | `[0.072, 0.136]` | **`+0.072`** | `[+0.066, +0.077]` | Steering Layers 2 to 7 switches moves to the edited board |
| **2. Mediated-only** *(shortcut subtracted out)* | **`0.158`** | `[0.122, 0.198]` | **`+0.037`** | `[+0.033, +0.041]` | **97.9% of error reduction survives** without the shortcut |
| **3. Direct-readout bypass** *(shortcut only)* | `1.808` | `[1.730, 1.884]` | **`+0.000`** | `[+0.000, +0.000]` | **Zero new legal moves found** when middle layers are skipped |
| **4. Untrained transformer control** | `2.661` | `[2.581, 2.743]` | `+0.000` | `[+0.000, +0.000]` | Random untrained encoder vectors have no effect |

![Board-Level Causal Visualization on Case #101](fig_board_intervention.png)
*Figure: Board-level move predictions on benchmark Case #101 (flipping square `C4` from White to Black), generated via [`likenneth/othello_world`](https://github.com/likenneth/othello_world).*

### Plain-English Explanation of the Results

1. **Middle layers actively recompute the rules of Othello (`Mediated-Only`, Panel `d`):**
   On Case #101 above (Panel `a`), flipping `C4` from White to Black makes `F7` illegal (`LOST`, red box) and makes `B5` and `C5` newly legal (`+NEW`, green boxes). When the steering vector is added in the middle layers (Layers 2 to 7) and completely subtracted right before the final output layer (**Mediated-Only**, Panel `d`), the model turns off the invalidated move `F7` (`0.12 -> 0.00`), activates the newly legal moves `B5` and `C5` with **0 errors**, and preserves **97.9%** of the full steering error reduction across all 1,001 boards (`2.723 -> 0.158`).
2. **The direct shortcut to the output layer cannot find new legal moves (`Direct Bypass Only`, Panel `c`):**
   Skipping the middle layers and adding the steering vector *only* at the final output layer slightly lowers the score of old move `F7` (`0.12 -> 0.06`), which reduces average error from `2.723` to `1.808`, but it assigns **`0.00` probability** to the newly legal moves `B5` and `C5` (**`+0.000`** across all 1,001 boards). Without the middle layers checking which pieces are flanked on the board, the final output layer has no way to know that flipping `C4` makes `B5` and `C5` legal.
3. **Conclusion:**
   Othello-GPT genuinely uses its internal board state in the middle layers to compute legal moves rather than relying on a pass-through shortcut to the output layer. More broadly, any causal intervention test for a world model should include a **Mediated-Only** control (blocking direct skip connections to the readout layer) and an **Untrained-Model** control so shortcut bias is never mistaken for world-model reasoning.

---

## 4. What Would Change This Conclusion and Next Steps

- **What result would change this conclusion:**
  The conclusion that middle layers recompute the rules would be disproved if locking the middle layers (Layers 2 to 7) to their original unedited outputs while applying the steering vector only at the final output layer still eliminated more than half of the move errors or boosted newly legal moves. In practice, the output shortcut alone finds **zero** newly legal moves (`+0.000`).
- **What to run next with another week:**
  Even though the linear probe reads the board with **99.4% accuracy** by Layer 5, editing Layer 5 *alone* (instead of across Layers 2 to 7 together) leaves move error high at `2.188`. With another week, the next experiment is to test whether later layers undo single-layer edits by looking back at the unedited move tokens in the input history, and whether blocking attention to the flipped square's original move token makes a single-layer edit work.

---

## 5. What I Want to Work On With Physical Reasoning

I want to work on [**`physicalreasoning/forward` Issue #5 (*"Design an intervention test for world models"*)**](https://github.com/physicalreasoning/forward/issues/5), applying the same untrained-encoder and shortcut-control discipline used in [`causal-jepa`](https://github.com/physicalreasoning/causal-jepa).

In my research on multi-agent failure modeling (using soft interventions on Structural Causal Models at the **CausalML Lab**) and **hierarchical traffic scene synthesis**, I have seen that models with nearly identical observational rollout loss routinely diverge when tested under counterfactual interventions. With Physical Reasoning, I would love to build a lightweight 2D physics and traffic intervention benchmark in `forward` that:
1. Trains two world models (for example, a pixel-reconstruction dynamics model vs. a latent-state / JEPA or object-factored model) to matched observational validation loss on identical trajectories;
2. Resets the simulator to held-out states and re-runs under unshown impulse or mass changes; and
3. Scores models against a pre-registered gate on their **counterfactual consistency gap** alongside untrained-encoder and direct-readout controls.

---

## Reproduce

```bash
# 1. Run all causal intervention & control arms across N = 1,001 boards (~2 min on GPU)
python run_experiment.py

# 2. Generate the board-level intervention visualization (via likenneth/othello_world)
python generate_paper_visuals.py
```

*AI tools note:* AI coding assistants (Antigravity / Gemini) were used to scaffold the PyTorch forward-pass hooks, KV-cached batching, and plotting scripts (`othello_world`); the experimental design, skip-connection decomposition controls, and analysis were verified directly against executed benchmark runs.
