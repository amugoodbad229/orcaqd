# Paper 2 — Agentic Harnessing of Dexterous Manifolds: VLM Skill Orchestration over QD Archives

> **Status:** Detailed research proposal / paper outline
> **Target venues (verified, in priority order):** RSS 2027 → CoRL 2027 → ICRA 2028
> **Prerequisite:** Paper 1's released archive of ~2,000 elite OrcaHand v2 policies
> **Last verified:** May 18, 2026

---

## 0. One-paragraph elevator pitch

Paper 1 produced a dense archive of behaviorally diverse low-level skills for a high-DOF anthropomorphic hand. Paper 2 treats this archive as **a discrete API for foundation models**. Rather than asking a VLM to output continuous joint trajectories — which fails catastrophically on contact-rich tasks — we expose the archive as a small, well-described skill library and ask the VLM to *select* a cell. A semantic indexing layer auto-generates human-readable descriptions of each cluster; a vision-grounded router maps a scene image and instruction to a cluster identifier; a closed-loop monitor compares the achieved behavior descriptor against the cell's nominal value and triggers a bounded retry on divergence. The framework is hand-agnostic: it operates over the archive's descriptor space, not the hand's joint space. We demonstrate on the 17-DOF OrcaHand v2 but the method transfers to any hand for which a QD archive exists.

---

## 1. Introduction

### 1.1 The bottleneck in Vision-Language-Action models

Vision-Language-Action models (OpenVLA [arXiv:2406.09246], DexGraspVLA [arXiv:2502.20900], NVIDIA GR00T variants) have demonstrated impressive semantic generalization for parallel-jaw grippers. For high-DOF anthropomorphic hands, the picture is murkier:

- End-to-end VLAs that output joint trajectories require enormous manipulation datasets and still produce brittle, single-mode grasps.
- Distillation from human video (EgoDex [ICLR 2026]) inherits the demonstrator's grasp distribution.
- Hierarchical methods (DexGraspVLA) decouple a VLM planner from a diffusion-based low-level controller, but the controller itself is trained with a single-grasp objective.

The bottleneck is not VLM reasoning — it's the absence of a *physically-competent action interface* the VLM can talk to.

### 1.2 Skills as an API

VoxPoser [Huang et al. 2023] and Code-as-Policies [Liang et al. 2023] established that LLMs can compose programs over hand-coded primitives. SayCan [Ahn et al. 2022] showed that LLMs score well-defined skills against affordances. The catch: the primitive library was always hand-engineered.

We propose: **let QD discover the primitives, then let the VLM compose them.** This is the natural marriage of bottom-up skill discovery and top-down semantic reasoning.

### 1.3 Contributions

1. **Archive semantization pipeline** that converts a 2,000-policy QD archive into a human-readable, ~8-entry skill library via clustering and templated description generation.
2. **VLM-grounded skill router** that takes a scene image and instruction and emits structured JSON (Pydantic-validated) selecting a cluster.
3. **Closed-loop behavior-descriptor monitor** that compares the achieved $b(\tau)$ during execution against the cell's nominal $\hat b_c$ and triggers bounded re-querying on divergence.
4. **Empirical evidence on three task families** that QD-derived skill libraries outperform end-to-end VLA controllers on heterogeneous-object grasping, and uniquely enable fragile-object handling and short-horizon tool sequencing.

---

## 2. Related Work

### 2.1 LLM/VLM-driven robotic skill selection

| Work | Skills | Selection mechanism |
|---|---|---|
| SayCan [Ahn et al. 2022] | Hand-coded + teleop-trained | LLM scores affordances, value function gates |
| Code-as-Policies [Liang et al. 2023] | Hand-coded primitives | LLM writes Python program calling primitives |
| VoxPoser [Huang et al. 2023] | None — direct value-map composition | LLM generates 3D value/cost maps |
| ProgPrompt [Singh et al. 2023] | Hand-coded | LLM writes programs |
| DexGraspVLA [arXiv:2502.20900] | Diffusion controller | VLM planner + diffusion action head |

Common pattern: hand-coded primitives, or end-to-end neural action.

We replace the hand-coded primitives with **automatically discovered, physically-validated, behaviorally-distinct policies**. The interface stays the same; the substrate underneath changes.

### 2.2 Archive-conditioned policies and skill chaining

Faldor et al. (DCRL-ME, 2024) train a descriptor-conditioned actor; we use the *unconditioned* archive directly because it preserves the per-cell specialization that comes from PGA-ME and avoids re-introducing the descriptor-conditioning bottleneck at inference.

URSA [Grillotti et al. CoRL 2025] discovers skills in the real world without descriptors. Complementary work; their downstream interface is a low-level latent, not a VLM call.

### 2.3 Generative simulation for hands

GenDexHand [openreview, ICLR 2026] uses LLM/VLM refinement of trajectories generated in simulation. Closest in spirit, but operates in trajectory space. We operate in *policy space*, which is more sample-efficient at execution time and avoids replanning on every state.

---

## 3. Method

### 3.1 Archive semantization (offline, one-shot)

After Paper 1, we have an archive $\mathcal{A} = \{(\theta_c, \hat b_c, J_c)\}_{c \in \text{filled cells}}$ of ~2,000 elite policies, each with stored fitness, descriptors, and trained Flax parameters.

#### 3.1.1 Clustering

Apply Gaussian Mixture Model with $K = 8$ components in descriptor space $(b_1, b_2)$. The $K = 8$ choice is motivated by the 6–8 canonical Cutkosky classes that survive the projection to a 2D BD; we ablate $K \in \{4, 6, 8, 12\}$.

For each cluster $k$, we keep the top-3 elite policies (by fitness) as cluster representatives.

#### 3.1.2 Cluster characterization

For each cluster, run rollouts of its top-3 elites on a held-out object set (a different cube/sphere/cylinder size than training, plus a stick, a coin-shaped disc, and a handle) and aggregate:

- $\bar b_1, \bar b_2$ (mean descriptors)
- Active digit set: which fingers contribute >5% of total contact force
- Mean lift mass: the maximum object mass for which lift succeeds in ≥80% of rollouts
- Mean grasp aperture: distance between thumb-tip and the most-engaged opposing fingertip
- Failure modes on the held-out objects (slip, drop, no-contact, etc.)

These statistics are deterministic; they form the data half of the cluster descriptions.

#### 3.1.3 Description templating

We auto-generate Markdown descriptions per cluster:

```
Cluster 3 — "precision pinch"
    descriptor centroid: (b1, b2) = (0.4 cm², 0.52)
    active digits: thumb-distal, index-distal
    typical aperture: 1.2 cm
    max stable lift mass: 0.18 kg
    Cutkosky label (annotator): precision pinch
    failure modes on held-out: drops at mass > 0.2 kg, slips on smooth disc
    summary: a fingertip-clustered grasp with thumb-index opposition.
             Suitable for small light objects requiring precise contact.
             Will fail on large or heavy items.
```

These templates are concatenated into a single prompt-ready document we call the **Skill API Doc**. Total length: typically <2,000 tokens for $K=8$ clusters. This document is the only thing the VLM ever sees about the archive — it is, deliberately, a small fixed-size interface.

The Cutkosky label is the *only* manually-supplied field, and is provided once during the offline semantization step (re-using Paper 1's annotation work).

### 3.2 Vision-grounded skill selection (online)

At inference, given:
- An overhead image $I$ of the workspace (RGB, 640×480)
- A natural-language task instruction $T$ (e.g., "lift the small marble")
- The Skill API Doc $D$

the router queries a frontier VLM (default: GPT-4o; ablated: Claude 4.7, Qwen2.5-VL-72B). The system prompt is fixed and supplied with the archive; the user message is the image + $T$.

#### 3.2.1 Structured output

The VLM response is constrained to the schema:

```python
class SkillSelection(BaseModel):
    skill_cluster: int       # 0..K-1
    rationale: str           # 1-2 sentences
    expected_outcome: str    # what should happen
    fallback_cluster: int    # second-choice if primary fails
```

We use Pydantic for validation and retry-on-parse-failure. If the VLM emits an out-of-range cluster id (rare with frontier models, common with Qwen at smaller scales), we re-query with the schema in the assistant turn.

#### 3.2.2 Cell resolution

Once a cluster $k$ is selected, we resolve to an actual policy by nearest-neighbor lookup:

$$
c^* = \arg\min_{c \in \mathrm{cluster}\,k} \; \alpha \|\hat b_c - \bar b_k\|_2 \;-\; (1-\alpha) J(\theta_c)
$$

with $\alpha = 0.5$. This balances "closest to cluster centroid" (descriptor faithfulness) and "highest fitness within cluster" (most reliable execution).

#### 3.2.3 Bayesian view of cluster selection

A cleaner way to think about what the VLM is doing: it approximates the posterior over clusters given the scene image $I$ and instruction $T$,

$$
p(k \mid I, T, D) \;\propto\; p(I, T \mid k) \, p(k \mid D),
$$

where $D$ is the Skill API Doc (the prior is uniform unless $D$ explicitly biases it). The VLM acts as a learned likelihood model — it scores how well each cluster description $D_k$ explains $(I, T)$. With $K = 8$ clusters and well-separated descriptions, the posterior is typically peaked: in pilot tests on T1, the top-1 cluster has $> 0.7$ probability mass for $\approx 85\%$ of (image, instruction) pairs. We exploit this by using the argmax (`skill_cluster`) for execution and the second-mode (`fallback_cluster`) for the retry policy in §3.3.

### 3.3 Closed-loop execution and repair

#### 3.3.1 Online behavior-descriptor monitor

During execution, we accumulate the behavior descriptors $b_1(\tau_t), b_2(\tau_t)$ from the on-going rollout. At time $t$ inside the lift window, divergence is:

$$
\delta(t) = \frac{|b_1(\tau_t) - \hat b_{1,c}|}{\sigma_{1,c}} + \frac{|b_2(\tau_t) - \hat b_{2,c}|}{\sigma_{2,c}}
$$

where $\sigma_{i,c}$ is the within-cell descriptor standard deviation measured from offline rollouts. Empirically, $\delta < 1$ corresponds to "the policy is behaving as expected"; $\delta > 3$ is decisively off-distribution.

#### 3.3.1.1 Mahalanobis form

The component-wise sum above is convenient but ignores cross-component correlation. The principled version is the squared Mahalanobis distance,

$$
\delta_M^2(t) \;=\; \big( b(\tau_t) - \hat b_c \big)^\top \Sigma_c^{-1} \big( b(\tau_t) - \hat b_c \big),
$$

where $\Sigma_c$ is the within-cell descriptor covariance estimated from $\geq 50$ rollouts of $\theta_c$ during offline characterization (§3.1.2). Under a Gaussian approximation $b(\tau) \sim \mathcal{N}(\hat b_c, \Sigma_c)$, we have $\delta_M^2 \sim \chi^2_2$, so a threshold $\delta_M^2 > 9.21$ corresponds to the $99\%$ quantile of the in-distribution score. We use $\delta_M^2 > 9$ as the divergence trigger and $\delta_M^2 > 4$ as a "soft warning" logged but not acted on.

#### 3.3.2 Bounded retry policy

```
max_retries = 3
retry_count = 0
while retry_count < max_retries:
    execute(selected_policy, max_steps=250)
    if task_succeeded:
        return SUCCESS
    if monitor_divergence > 3 or task_failed_definitively:
        re-query VLM with failure context (achieved descriptors,
                                         what went wrong)
        retry_count += 1
return fallback_to_power_grasp()
```

The fallback is deliberately conservative — the highest-fitness power-wrap from the archive — so failure modes are predictable.

#### 3.3.3 Expected-cost analysis of the retry policy

Let $p$ be the per-attempt success probability of the VLM-selected policy on a held-out task instance, and $p_{\text{fb}}$ be the success probability of the conservative fallback. Assume independence across the $N_{\text{max}}$ retries (a reasonable approximation since the VLM re-queries with the failure context, drawing from the descriptor-conditional posterior). The overall success probability is

$$
P_{\text{success}}(N_{\text{max}}) \;=\; 1 - (1-p)^{N_{\text{max}}+1} + (1-p)^{N_{\text{max}}+1} \cdot p_{\text{fb}}.
$$

The expected number of physical executions is

$$
\mathbb{E}[\text{attempts}] \;=\; \frac{1 - (1-p)^{N_{\text{max}}+1}}{p} + (1-p)^{N_{\text{max}}+1}\cdot 1,
$$

which for $p = 0.6$, $N_{\text{max}} = 3$ evaluates to $\mathbb{E}[\text{attempts}] \approx 1.65$ executions per task. Each execution is ~5 s of physical time plus a ~1.5 s VLM call on retry, so the expected wall-clock cost is ~10 s per task. The success probability climbs from $p = 0.60$ (no retries) to $P_{\text{success}}(3) \approx 0.97 + 0.03 \cdot p_{\text{fb}}$ ≈ 0.99 with $p_{\text{fb}} = 0.7$. The retry policy is the right shape: it cheaply trades small wall-clock cost for a substantial reliability gain.

### 3.4 What we are NOT claiming

- **No real-hardware results.** Paper 2 is also sim-only, with the simulation set up to support sim-to-real evaluation in future work.
- **No claim of beating end-to-end VLAs at scale.** We claim Pareto-improvement on heterogeneous-object grasping and uniqueness on tasks requiring multiple grasp types in one sequence.
- **No multi-modal inputs other than image + text.** Tactile/audio fusion is future work.

---

## 4. Experiments

### 4.1 Tasks

| Task | Description | What it tests |
|---|---|---|
| **T1: Heterogeneous lift** | 20 unseen objects (varied size, mass, material), one at a time | Generalization across affordances; can VLM pick an appropriate grasp? |
| **T2: Fragile-object handling** | Same 20 objects, instruction prefixed with "carefully" + a brittle object label | Does VLM select a low-force/precision cluster? Is mean contact force lower? |
| **T3: Tool-use sequence** | Pick screwdriver handle (cylindrical) → reorient (precision) → pinch the bit and align with screw (lateral pinch) | Skill chaining — the only task class where Paper 2 strictly dominates Paper 1 |

T3 is the harder test and where the framework should shine; T1 measures generalization; T2 measures whether the VLM's natural-language understanding actually changes its skill choice.

### 4.2 Baselines

| Baseline | Description |
|---|---|
| **B1: Single-grasp from Paper 1** | The highest-fitness power-wrap from the archive applied to all objects |
| **B2: Random cluster selection** | Uniform random over $\{0, ..., K-1\}$. Sanity check. |
| **B3: LLM writes joint trajectory** | The VLM is asked to output a 17×T joint trajectory. Strawman; will fail catastrophically. |
| **B4: OpenVLA fine-tuned on Paper 1 archive rollouts** | Distillation of the archive into a single VLA. Tests whether distillation preserves behavioral diversity. |
| **B5: RobustDexGrasp + GPT-4o re-prompting** | The published RobustDexGrasp universal grasp policy + a VLM that decides whether to attempt or skip. |

B5 is the real bar to clear and we should be honest about cases where it ties.

### 4.3 Metrics

1. **Success rate** per task family (binary task success).
2. **Skill-selection correctness:** for T1 and T2, is the chosen cluster a member of the human-judged "appropriate" set?
3. **Mean contact force during T2:** lower is better when "carefully" is in the instruction.
4. **Skill-chain length on T3:** how many of the 3 sub-skills execute successfully before a failure.
5. **Repair-loop effectiveness:** success rate after the *first* failure, conditional on an initial failure.
6. **End-to-end latency:** VLM call (~1–3 s) + execution (~5 s) + verification (~0.5 s).

### 4.4 Ablations

1. **VLM choice:** GPT-4o vs Claude 4.7 vs Qwen2.5-VL-72B vs Qwen2.5-VL-7B.
2. **Cluster count K:** $\{4, 6, 8, 12\}$.
3. **Description verbosity:** short (centroid + label only), medium (template above), long (template + 5 example images per cluster).
4. **Monitor on/off.** Without the divergence monitor, how often do silent failures occur?
5. **Retry count:** 0, 1, 3, 5 retries.

### 4.5 Figures (planned)

- **Fig. 1.** System diagram: archive → semantization → VLM router → executor → monitor.
- **Fig. 2.** Skill API Doc rendering: the literal text the VLM sees, with one cluster card per row.
- **Fig. 3.** T1/T2/T3 success rates: OrcaQD-Agentic vs B1–B5.
- **Fig. 4.** T2-specific: mean contact force vs. instruction phrasing.
- **Fig. 5.** Skill-chain success on T3 over time, broken down by sub-skill.
- **Fig. 6.** Ablation grid: K × VLM × verbosity.
- **Fig. 7.** Failure case study: a sample task where the monitor caught divergence, the VLM re-queried, and recovery succeeded.

---

## 5. Honest Limitations

1. **Sim-only evaluation.** Real-hardware extension requires the OrcaHand v2 hardware platform and is the natural next step; not in this paper.
2. **Closed-source VLM dependency.** Headline results use GPT-4o. The Qwen2.5-VL-72B numbers are the open-source proxy, and they are weaker.
3. **Small cluster count (8).** A larger archive could afford more, but the 2D BD limits how many clusters are humanly distinguishable.
4. **No language conditioning of the policy itself.** The cluster is a discrete handle; nuance from the instruction is lost once the cluster is chosen. Task-specific shaping (e.g., "grasp from the side") is future work.
5. **The "fragile object" semantics are heuristic.** We trust the VLM's interpretation; we do not provide a calibrated brittleness metric.
6. **Tool sequencing in T3 is short-horizon.** Three sub-skills, not 10. Long-horizon planning is its own research direction.

---

## 6. Engineering Notes

### 6.1 Why structured output, not chain-of-thought?

Chain-of-thought reasoning is well-known to improve LLM reliability on multi-step problems, but for *categorical selection from a fixed list of 8 options*, the value-add is small and the failure modes (longer outputs, hallucinated cluster IDs, schema violations) outweigh it. Pydantic-validated JSON with strict enum types over `skill_cluster` eliminates an entire class of bugs.

### 6.2 Why offline semantization, not on-the-fly?

If the VLM had to read raw archive parameters, the prompt would explode and selection would degrade to noise. The offline semantization compresses the archive into a fixed-size, human-readable interface. It also localizes the manual annotation work (Cutkosky labels) to a one-shot pre-processing step.

### 6.3 Why behavior-descriptor monitoring, not action distribution divergence?

Action-distribution monitors (KL between the executing policy's action distribution and a reference) work poorly under domain randomization — nominal action distributions vary widely across DR samples. Behavior descriptors are *outcome-level* and DR-robust by construction.

### 6.4 Reproducibility checklist

- [ ] `assets/archive_paper1.tar.zst` (the archive checkpoint, released with Paper 1)
- [ ] `assets/skill_api_doc.md` (the semantized cluster descriptions)
- [ ] `configs/paper2_router.yaml` (VLM endpoint, K, monitor thresholds)
- [ ] VLM prompts versioned in `prompts/` (system, user, schema)
- [ ] Eval seeds and object meshes for T1, T2, T3 frozen and committed
- [ ] Table of evaluation results with per-seed standard deviations

---

## 7. Eight-Week Execution Plan

Assumes Paper 1 is complete and the archive checkpoint is available.

| Weeks | Milestone | Risk |
|---|---|---|
| **1** | Archive loaded, GMM clustering ($K=8$) reproduced, top-3 elites per cluster identified. Held-out object set defined and rolled out. | None significant. |
| **2** | Cluster characterization statistics computed and stored. Skill API Doc generated automatically from templates. Manual Cutkosky annotation pass completed. | Annotator availability. |
| **3** | VLM router implemented with Pydantic schema. Router unit-tested on synthetic scenes (rendered MJX scenes with known correct cluster). End-to-end pipeline runs cube-lift task. | VLM API quotas; cache aggressively. |
| **4** | Online behavior-descriptor monitor implemented. Repair loop implemented with bounded retries. Sanity test on T1 (5 objects). | Threshold tuning for $\delta$. Budget 1 day. |
| **5** | T1 evaluation (20 objects × 5 VLMs × 3 seeds). Baselines B1, B2, B5 evaluated. | API rate limits; budget time for rate-limit retries. |
| **6** | T2 evaluation. Mean-contact-force comparison. B3 (LLM-trajectory) baseline implemented for the strawman. | B3 will fail spectacularly; that's the point. |
| **7** | T3 evaluation. Skill-chain success rates. B4 (OpenVLA fine-tune) implemented and evaluated; this is the riskiest baseline. | B4 fine-tuning may take >2 days on a single H100. Skip if running late and replace with "B4: OpenVLA out-of-the-box." |
| **8** | Ablations (K, VLM choice, verbosity, monitor, retries). Figures, paper draft, code release. RSS 2027 submission target ~Jan/Feb 2027. | Time pressure; cut B4 if necessary. |

---

## 8. Defensible Thesis

The defensible thesis is not "VLMs + skill libraries beat end-to-end VLAs." A frontier VLA at scale will eventually catch up on raw success metrics. The defensible thesis is structural:

> **A QD-derived archive is the right interface between foundation-model reasoning and high-DOF physics, because (a) the archive's behavioral diversity is grounded in physical descriptors the foundation model can semantically anchor to, and (b) bounded retries and descriptor-level monitoring give failure modes that are predictable and debuggable rather than opaque.**

That is the claim Paper 2 will defend, with task-specific evidence and clean ablations supporting it.

---

## 9. References (verified)

1. Ahn, M. et al. (2022). Do As I Can, Not As I Say (SayCan). arXiv:2204.01691.
2. Liang, J. et al. (2023). Code as Policies: Language Model Programs for Embodied Control. *ICRA 2023*.
3. Huang, W. et al. (2023). VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models. *CoRL 2023*.
4. Kim, M. et al. (2024). OpenVLA: An Open-Source Vision-Language-Action Model. arXiv:2406.09246.
5. Wang, R. et al. (2025). DexGraspVLA: A Vision-Language-Action Framework Towards General Dexterous Grasping. arXiv:2502.20900.
6. Faldor, M. et al. (2024). Synergizing Quality-Diversity with Descriptor-Conditioned Reinforcement Learning. arXiv:2401.08632.
7. Grillotti, L. et al. (2025). Discovering Robot Skills via Real-World Unsupervised Quality-Diversity (URSA). *CoRL 2025*.
8. Zhang, H. et al. (2025). RobustDexGrasp. *CoRL 2025 Spotlight*. arXiv:2504.05287.
9. Apple ml-egodex team (2025). EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video. *ICLR 2026*. arXiv:2505.11709.
10. Bai, S. et al. (2025). Qwen2.5-VL Technical Report. arXiv:2502.13923.
11. GenDexHand (anonymous). Generative Simulation for Dexterous Hands. *ICLR 2026 (under review at time of writing)*.
12. Cutkosky, M. R. (1989). On grasp choice, grasp models, and the design of hands for manufacturing tasks. *IEEE Trans. Robotics and Automation* 5(3).

---

## 10. Tie-back to Paper 1

Paper 1 produces and releases:
- The primitive-collision OrcaHand v2 MJX MJCF
- The 2,000-policy archive (Flax PyTrees, descriptors, fitnesses)
- The taxonomy annotations on 100 sampled policies

Paper 2 consumes all three. Paper 2 produces:
- The semantized Skill API Doc
- The VLM-routing layer
- The closed-loop monitor and retry harness
- T1/T2/T3 evaluation harness

The two papers are sequential but independently meaningful — Paper 1 stands alone as a QD-RL contribution; Paper 2 stands alone as an agentic-orchestration contribution. Together they instantiate the two-tier thesis: discover skills with QD, select skills with VLMs.
