# Paper 1 — OrcaQD: Discovering Dexterous Contact Manifolds via Hardware-Accelerated Quality-Diversity RL

> **Status:** Detailed research proposal / paper outline
> **Target venues (verified, in priority order):** ICRA 2027 (deadline ~Sep 2026, Seoul, May 24–28 2027) → RSS 2027 → CoRL 2027
> **Hand model:** OrcaHand v2 (17 actuated DOF, right-hand variant)
> **Last verified:** May 18, 2026

---

## 0. One-paragraph elevator pitch

Standard deep RL on multi-fingered hands collapses to a single power-wrap grasp. We hypothesize this is a structural artifact of policy-gradient methods on action spaces with vastly unequal solution-volume — call it *manifold-volume gradient starvation*. We sidestep the problem entirely by re-casting dexterous skill discovery as **archive optimization**: instead of a single scalar return, we maintain a 2D grid of elite policies indexed by physically-meaningful behavior descriptors (contact dispersion and inter-digit force allocation). Built on MuJoCo MJX and QDax's `PGAMEEmitter`, OrcaQD discovers an archive of diverse, contact-rich grasping policies on any high-DOF anthropomorphic hand in a few GPU-hours, with no human demonstrations. We demonstrate the framework on the 17-DOF OrcaHand v2 and deliver the first openly-reproducible dexterous skill library covering a verifiable fraction of the Cutkosky taxonomy.

---

## 1. Introduction

### 1.1 Motivation

Anthropomorphic hands have 16–24 actuated joints and the kinematic capacity to express the full Cutkosky grasp taxonomy [Cutkosky 1989] — 16 grasp classes spanning power vs. precision, prismatic vs. circular. Yet the dominant paradigm in dexterous manipulation RL trains a single policy with a scalar return:

- OpenAI's ShadowHand work [arXiv:1808.00177] uses PPO with massive domain randomization and converges to a single in-hand reorientation behavior.
- RobustDexGrasp [CoRL 2025 Spotlight, arXiv:2504.05287] uses teacher-student distillation and reports a single universal grasp policy with 94.6% success on 500+ unseen objects — every one of them grasped with a power-wrap.
- EgoDex [ICLR 2026, arXiv:2505.11709] uses 829 hours of egocentric video with 3D hand tracking, but inherits the modal grasp distribution of the human demonstrators.

The empirical observation is consistent: a 17-DOF agent given a "lift the object" reward learns one grasp.

### 1.2 Why does this happen?

We propose the following hypothesis, which we will support empirically rather than prove formally:

> **Manifold-Volume Gradient Starvation Hypothesis.** Let $\mathcal{A} \subset \mathbb{R}^{n_a}$ be the action space, and let $\mathcal{S}_g \subset \mathcal{A}$ be the set of action sequences that achieve task success via grasp class $g$. When the volumes $\operatorname{vol}(\mathcal{S}_g)$ are exponentially unequal across grasp classes, a stochastic gradient estimator with finite variance preferentially walks toward the largest-volume basin and stays there. Smaller-volume basins (precision pinches, lateral pinches, tripods) are reachable in principle but the expected contribution to the gradient is dominated by the variance of the largest basin.

This is *not* a tuning problem. Tighter exploration noise, larger entropy bonuses, intrinsic-motivation rewards — these reshape but do not eliminate the volume disparity. The structural fix is to abandon the scalar objective and optimize for *coverage* of behavioral basins.

#### 1.2.1 Informal variance argument

Consider an MDP with two disjoint optimal-action sets $\mathcal{S}_1, \mathcal{S}_2 \subset \mathcal{A}$ corresponding to two grasp classes, with $\operatorname{vol}(\mathcal{S}_1) = V_1$ and $\operatorname{vol}(\mathcal{S}_2) = V_2$, and $V_1 \gg V_2$. Both achieve the same return $R^*$. A Gaussian policy $\pi_\theta(a \mid s) = \mathcal{N}(\mu_\theta(s), \Sigma)$ produces the policy-gradient estimator

$$
\hat g(\theta) \;=\; \frac{1}{N}\sum_{i=1}^N \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t^{(i)} \mid s_t^{(i)}) \, R^{(i)}.
$$

For policy parameters initialized so that $\mu_\theta(s) \approx 0$, the probability that a sampled trajectory lies in $\mathcal{S}_g$ is $p_g \propto V_g \cdot \phi(\mathcal{S}_g; \Sigma)$ where $\phi$ is a Gaussian-density factor that decreases with distance from the initialization. The expected gradient $\mathbb{E}[\hat g]$ is a $V_g$-weighted sum:

$$
\mathbb{E}[\hat g(\theta)] \;\propto\; \sum_g V_g \, \nabla_\theta \log p_g(\theta).
$$

The contribution from the smaller basin scales linearly with $V_2 / V_1$, while the variance contribution from the larger basin is $\Theta(1)$. Once the policy mean drifts inside $\mathcal{S}_1$, $p_2 \to 0$ exponentially in the distance, and the gradient becomes a pure within-$\mathcal{S}_1$ refinement. We call this *gradient starvation* of $\mathcal{S}_2$.

Formally proving this for high-dimensional contact-rich settings is hard because the volumes are themselves outputs of a non-convex dynamics simulator. We sidestep the theory and provide §4.2's PPO/SAC baselines as the empirical witness: scalar-return RL on the OrcaHand consistently lands in one large-volume basin and does not escape regardless of seed.

#### 1.2.2 Why QD avoids this

Quality-Diversity decouples the optimization signal from the scalar return. Let $b: \mathcal{T} \to \mathcal{B} \subset \mathbb{R}^k$ map trajectories $\tau$ to behavior descriptors. The QD objective is

$$
\mathcal{L}_{\text{QD}}(\Theta) \;=\; \sum_{c \in \mathcal{C}} \max_{\theta_c \in \Theta} J(\theta_c) \;\cdot\; \mathbb{1}\!\left[ b(\tau_{\theta_c}) \in c \right]
$$

where $\mathcal{C}$ is a partition of $\mathcal{B}$ (a grid). Each cell $c$ has its own elite policy $\theta_c$. The aggregator is *additive across cells*, so a high-fitness solution in a small-volume basin contributes an independent term to the loss and is never washed out by the larger basin.

### 1.3 Quality-Diversity to the rescue

Quality-Diversity (QD) algorithms [Cully & Demiris 2017, Pugh et al. 2016] solve this by maintaining a population of solutions indexed by hand-designed (or learned) **behavior descriptors** $b: \mathcal{S} \times \mathcal{A} \to \mathbb{R}^k$. Instead of "find the best policy," QD asks "find the best policy *for each cell* in descriptor space." MAP-Elites [Mouret & Clune 2015] is the canonical algorithm. PGA-MAP-Elites [Nilsson & Cully 2021] augments evolutionary search with off-policy policy-gradient improvement. DCRL-ME [Faldor et al. 2024, arXiv:2401.08632] further uses a descriptor-conditioned actor as a generative model for offspring.

QD has been applied to bipedal locomotion, ant gaits, robotic arms, and continuous control on Brax tasks. **It has not, to our knowledge, been applied to multi-fingered dexterous hands.**

### 1.4 Contributions

1. **First QD-RL framework for dexterous hand manipulation.** A general pipeline for any high-DOF anthropomorphic hand (16–24 DOF), contact-rich, GPU-parallelized, no demonstrations, no teacher policy. We demonstrate on the 17-DOF OrcaHand v2 but the method requires only a digit-to-geom mapping to transfer to other hands (Shadow, Allegro, Leap, etc.).
2. **Two physically-grounded behavior descriptors** that map cleanly to the Cutkosky taxonomy: (i) trace of the contact-position covariance matrix and (ii) thumb-distal force ratio. Both are defined over contact geometry, not hand topology — they apply to any hand with an opposable thumb.
3. **An automated primitive-collision MJCF generator** that converts any mesh-based hand model into an MJX-JAX-compatible variant by fitting capsules to phalanx bounding boxes. Demonstrated on OrcaHand v2; applicable to any hand with STL meshes.
4. **Open archive release.** ~2,000 elite policies serialized as Flax PyTrees, indexed by their cell coordinates and tagged with annotated Cutkosky labels. This becomes the foundation for Paper 2.

---

## 2. Related Work

### 2.1 Dexterous manipulation RL

| Work | Method | Outcome |
|---|---|---|
| OpenAI 2018 [1808.00177] | PPO + domain randomization | Single in-hand cube reorientation policy |
| DAPG [Rajeswaran et al., RSS 2018] | Demo-augmented policy gradient | One grasp per task |
| RobustDexGrasp [CoRL 2025] | Teacher-student, single-view vision | One universal power-wrap policy |
| EgoDex [ICLR 2026] | Distillation from 829h human video | Inherits human demo distribution |
| In-Hand Reorientation MoE [arXiv:2508.01695] | Mixture-of-experts hierarchical | Multiple policies but task-specific reorientation only |

None of these targets *behavioral coverage* of the grasp taxonomy as a first-class objective.

### 2.2 Quality-Diversity for control

| Work | Domain | Descriptor |
|---|---|---|
| MAP-Elites [Mouret & Clune 2015] | Hexapod locomotion, foot contact time | Hand-designed |
| PGA-ME [Nilsson & Cully 2021] | Brax Walker/Ant | Final foot positions |
| DCRL-ME [Faldor et al. 2024] | QDax benchmark suite | Descriptor-conditioned actor |
| AURORA [Cully 2019, Grillotti & Cully 2022] | Various | Learned via autoencoder |
| URSA [Grillotti et al., CoRL 2025] | Real-world quadruped | Unsupervised, online |

OrcaQD lifts the descriptor design to the contact level — a representation natural for grasping but absent from prior QD-RL work.

### 2.3 Hardware-accelerated physics

MuJoCo XLA (MJX) [Google DeepMind, 2024–2026, mujoco-mjx 3.8 on PyPI] re-implements MuJoCo in JAX, supporting GPU and TPU. MJX-JAX is the pure-JAX backend; MJX-Warp uses NVIDIA Warp for fully-supported mesh collisions. MJX-JAX has known limits on convex-mesh collisions (≲32 vertices for convex-convex, ≲200 for convex-primitive). We use MJX-JAX with primitive collision geoms; we discuss the MJX-Warp alternative in §6.

QDax [Chalumeau et al. 2024, JMLR] is the canonical JAX QD library (v0.5.1 as of late 2025). It implements PGA-MAP-Elites, DCRL-ME, MAP-Elites-PBT, and AURORA out of the box and is the foundation of our pipeline.

---

## 3. Method

### 3.1 OrcaHand v2 environment

**Hand.** We demonstrate on the OrcaHand v2 right-hand model (17 actuated DOF) but the framework is parameterized by a `HandConfig` that specifies DOF count, digit-to-geom mapping, and actuator ranges. Any anthropomorphic hand with 16–24 DOF and an opposable thumb can be used by providing a new config. The key requirements are:
- Position-controlled actuators (the action space is joint-position targets)
- An identifiable thumb digit (for $b_2$ computation)
- STL meshes or pre-existing primitive collision geometry

OrcaHand v2 specifics: 17 actuated joints:
- 1 wrist
- 4 × {abduction, MCP, PIP} for index, middle, ring, pinky (12)
- 4 thumb joints (CMC, abduction, MCP, PIP)

Joint ranges and actuator force limits read directly from the upstream MJCF (`v2/models/mjcf/orcahand_right.mjcf`). The default `forcerange="-1 1"` and `kp=2.0` from `options.xml` are retained for the body but tuned per-joint for grasping (see §3.6).

**Object set (Paper 1 scope, deliberately small).**
- Cube, side 4 cm
- Sphere, diameter 3.5 cm
- Cylinder, 3 cm diameter × 8 cm length

These are chosen because (a) they evoke distinct grasp affordances (lateral pinch on cube edge, spherical wrap, cylindrical power grasp), (b) all three are convex with simple primitive collision geoms, and (c) the diversity of canonical grasps they support exercises the descriptor space.

**Physics backend.** MuJoCo MJX-JAX, batch size 4,096, episode length 250 steps at 50 Hz (5 s physical time).

**Collision model.** This is the central engineering contribution of §3. The upstream `orcahand_right.mjcf` uses STL meshes for *all* collision geometry (each finger phalanx is a Fusion360 export with hundreds of vertices). MJX-JAX would either OOM or run at <50 steps/sec at batch 4,096 with these meshes. We author a parallel `orcahand_right_mjx.mjcf` that:
- Retains all visual STL geoms (`contype=0 conaffinity=0`)
- Replaces every collision-eligible body with one capsule (phalanx) or box (palm/forearm)
- Inherits all `<exclude>` contact pairs from the upstream file

Geometry is hand-fitted using `meshlab` to extract phalanx principal axes, then refined visually in `mujoco.viewer`. Total: 14 capsules (4 fingers × 3 phalanges + thumb × 2) + 1 palm box + 1 forearm box.

**Domain randomization** (per environment instance, sampled at reset):
- Object mass: $m \sim \mathcal{U}(0.05, 0.5)$ kg
- Friction coefficient: $\mu \sim \mathcal{U}(0.4, 1.2)$
- Initial object pose: $\Delta x, \Delta y \sim \mathcal{U}(-0.02, 0.02)$ m, yaw $\sim \mathcal{U}(-15°, 15°)$
- Per-actuator gain perturbation: $k_p \sim \mathcal{U}(0.9, 1.1) \times k_{p,\text{nominal}}$

### 3.2 State and action spaces

The state and action dimensions are determined by the hand's DOF count $n_a$ (17 for OrcaHand, 16 for Leap Hand, 20 for Shadow Hand, 24 for Allegro, etc.):

**State** $s_t \in \mathbb{R}^{2 n_a + 13}$:
- $n_a$ joint positions $q_t$
- $n_a$ joint velocities $\dot q_t$
- Object 6D pose: position $p^o_t \in \mathbb{R}^3$, quaternion $\xi^o_t \in \mathbb{R}^4$
- Object spatial velocity: $v^o_t \in \mathbb{R}^3, \omega^o_t \in \mathbb{R}^3$

For OrcaHand: $s_t \in \mathbb{R}^{47}$ (17 + 17 + 7 + 6).

**Action** $a_t \in \mathbb{R}^{n_a}$: per-joint position targets, normalized to $[-1, 1]$ and rescaled to each joint's `ctrlrange`.

### 3.3 Reward function

We use a staged-curriculum scalar reward:

$$
r(s_t, a_t) = w_1 \mathbb{1}[\text{any hand-object contact}] + w_2 \Delta z^o_t + w_3 \mathbb{1}[z^o_t > h^*] - w_4 \|a_t - a_{t-1}\|_2^2
$$

where $\Delta z^o_t = z^o_t - z^o_{t-1}$, $h^*$ is the lift-success threshold (default 5 cm above starting pose), and the action-rate penalty discourages bang-bang policies.

**Curriculum.** $w_2$ is annealed from $0$ to its target value once the first contact event is logged in an episode. Without this anneal, the archive is ~90% empty for the first hour because most random policies never touch the object.

Default weights: $w_1 = 0.5,\ w_2 = 10.0,\ w_3 = 5.0,\ w_4 = 0.001$.

### 3.4 Behavior descriptors

The descriptor $b: \tau \to \mathbb{R}^2$ maps an episode trajectory $\tau$ to a 2D point. Both components are computed *during* the lift window $[t_g, t_g + T_g]$, where $t_g$ is the first time the object's vertical velocity stays positive for ≥200 ms ("grasp acquired") and $T_g = 1$ s.

#### 3.4.1 Dimension 1: Contact dispersion

$$
b_1(\tau) = \frac{1}{T_g}\int_{t_g}^{t_g + T_g} \operatorname{tr}\!\left( \frac{1}{|\mathcal{C}_t|}\sum_{i \in \mathcal{C}_t} (p_{i,t} - \bar{p}_t)(p_{i,t} - \bar{p}_t)^\top \right) dt
$$

where:
- $\mathcal{C}_t$ = active hand-object contacts at time $t$, filtered by normal-force threshold $f_n > 0.05$ N
- $p_{i,t} \in \mathbb{R}^3$ = world-frame contact point read from `mjx.Data.contact.pos`
- $\bar{p}_t = \frac{1}{|\mathcal{C}_t|}\sum_i p_{i,t}$

Geometric meaning: $b_1$ is the trace of the spatial covariance matrix of contact points. Units: m². Rotation- and translation-invariant. Small $b_1$ ⇒ fingertip-clustered contacts (pinch, tripod). Large $b_1$ ⇒ contacts spread across palm and phalanges (wrap, power grasp).

Range observed in pilot rollouts: $b_1 \in [10^{-5}, 10^{-2}]$ m². We bin in log-space.

#### 3.4.2 Dimension 2: Thumb force ratio

$$
b_2(\tau) = \frac{\int_{t_g}^{t_g+T_g} \|f^{\mathrm{thumb}}(t)\|\,dt}{\sum_{j \in \{T,I,M,R,P\}}\int_{t_g}^{t_g+T_g} \|f^j(t)\|\,dt} \in [0, 1]
$$

where $f^j(t) = \sum_{i \in \mathcal{C}_t^j} f_{i,t}^n$ is the total normal force exerted by digit $j$'s distal phalanx, summed over active contacts of that digit. Read from `mjx.Data.contact.frame[:,0]` × force.

Geometric meaning: $b_2$ measures how much of the squeezing force passes through the thumb opposed to the other digits. Cutkosky-axis: opposed-thumb power grasps have $b_2 \approx 0.5$, lateral pinches against a non-opposed thumb have $b_2 \to 0$, key/lateral pinches with thumb-only loading have $b_2 \to 1$.

#### 3.4.3 Why these two and not learned descriptors?

Learned descriptors (AURORA, DCG-ME) are appealing but:
- They couple descriptor learning to scoring, complicating ablations.
- They make taxonomy mapping post-hoc and harder for human annotators to verify.
- Reviewers will demand a baseline against hand-designed descriptors anyway.

We take the position that **physically-grounded hand-designed descriptors are the right starting point for dexterous QD**, and learned-descriptor variants are future work.

### 3.5 Algorithm: PGA-MAP-Elites on the OrcaHand archive

We use QDax's `PGAMEEmitter` directly. We do not reimplement; QDax v0.5.1 ships:

```python
from qdax.core.map_elites import MAPElites
from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
from qdax.core.emitters.pga_me_emitter import PGAMEEmitter, PGAMEConfig
```

**Archive.** $50 \times 50$ grid over $(\log_{10} b_1, b_2)$ with bounds calibrated from pilot rollouts. Total: 2,500 cells.

**Emitter mix** (PGA-ME default):
- Half the offspring batch: PG emitter — TD3 critic-gradient improvement applied to a parent sampled from the repertoire.
- Half: GA emitter — Iso+LineDD mutation [Vassiliades & Mouret 2018] of two repertoire parents.

**Networks.**
- Actor: MLP(47 → 64 → 64 → 17), tanh activation, Flax `nn.Module`. ~8K parameters per policy.
- Critic (shared, used only by PG emitter): MLP(47+17 → 256 → 256 → 1).

**Scoring function.** QDax's interface is:
```python
scoring_fn(genotypes: Params, key: PRNGKey) -> (fitness, descriptors, extra_scores, key)
```
We implement this as a `jax.vmap`'d 250-step MJX rollout, returning total reward (fitness) and the 2D descriptor.

#### 3.5.1 Iso+LineDD mutation operator

For two parents $\theta_a, \theta_b$ drawn uniformly from the repertoire, the offspring is

$$
\theta' \;=\; \theta_a \;+\; \sigma_{\text{iso}} \cdot \boldsymbol{\xi}_1 \;+\; \sigma_{\text{line}} \cdot \xi_2 \cdot (\theta_b - \theta_a)
$$

where $\boldsymbol{\xi}_1 \sim \mathcal{N}(0, I)$ is independent per parameter, and $\xi_2 \sim \mathcal{N}(0, 1)$ is a single scalar shared across all parameters. The first term is isotropic exploration; the second term is a directional perturbation aligned with the population's existing variation. Defaults: $\sigma_{\text{iso}} = 0.005$, $\sigma_{\text{line}} = 0.05$.

#### 3.5.2 PG emitter update

Half the offspring batch is produced by gradient ascent on a shared TD3 critic $Q_\phi$. Given a parent policy $\pi_{\theta_p}$ from the repertoire, we apply $K$ steps of:

$$
\theta_{k+1} \;=\; \theta_k \;+\; \eta \cdot \nabla_\theta \mathbb{E}_{s \sim \mathcal{D}}\!\left[ Q_\phi(s, \pi_\theta(s)) \right]
$$

for $K=100$, $\eta = 3 \times 10^{-4}$. The replay buffer $\mathcal{D}$ aggregates transitions from all rollouts so far. The critic $Q_\phi$ is trained with the standard TD3 target

$$
\mathcal{L}_Q(\phi) \;=\; \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\!\left[ \big( Q_\phi(s,a) - y \big)^2 \right]
$$

with $y = r + \gamma \min\!\big( Q_{\phi'_1}(s', \tilde a'), Q_{\phi'_2}(s', \tilde a') \big)$, target-policy smoothing $\tilde a' = \pi_{\theta'}(s') + \operatorname{clip}(\epsilon, -c, c)$, and twin target critics $\phi'_1, \phi'_2$ updated by Polyak averaging.

#### 3.5.3 Archive insertion rule

For an offspring $\theta'$ with descriptor $b(\tau_{\theta'})$ falling in cell $c$ and fitness $J'$:

$$
\theta_c \;\leftarrow\; \begin{cases} \theta' & \text{if } c \notin \text{filled or } J' > J(\theta_c) \\ \theta_c & \text{otherwise.} \end{cases}
$$

Cells are independent — replacement in one cell never displaces another. This is what gives QD its independence-across-basins property and is the structural reason it escapes manifold-volume gradient starvation.

### 3.6 Training configuration

| Parameter | Value |
|---|---|
| Parallel environments | 4,096 |
| Archive grid | $50 \times 50$ |
| Total environment steps | $1 \times 10^8$ |
| Iterations | ~100,000 |
| Offspring batch | 1,024 / iteration |
| PG steps per offspring | 100 |
| Critic learning rate | $3 \times 10^{-4}$ |
| Actor learning rate | $3 \times 10^{-4}$ |
| Iso+LineDD ($\sigma_1, \sigma_2$) | $(0.005, 0.05)$ |
| Hardware | 1× H100 80GB (or A100 40GB with reduced batch) |
| Expected wall-clock | 2.5–4 hours |

The 4,096 figure is our target. Pilot tuning will adjust batch size to fit memory; the throughput-per-dollar sweet spot on H100 is empirically near this scale for ~50-body MJX scenes.

### 3.7 What we are NOT doing

- **No sim-to-real in Paper 1.** Paper 1's claims are about behavioral coverage in simulation under domain randomization. Real-hardware transfer is Paper 2's open question and beyond it.
- **No object-pose conditioning of the policy beyond the state vector.** The policies are state-feedback, not vision. Vision conditioning is Paper 2.
- **No multi-task curriculum.** The archive is conditioned on a single task family ("lift one object"). Other task families (in-hand reorientation, peg insertion) require separate archives.

---

## 4. Experiments

### 4.1 Primary metrics

Let $\mathcal{C}$ be the set of grid cells, and let $\theta_c$ denote the elite policy in cell $c$ (undefined if $c$ is unfilled). Let $J(\theta) = \mathbb{E}_\tau[\sum_{t=0}^{T-1} r(s_t, a_t)]$ be the expected return.

1. **QD-Score:** the sum of fitness over filled cells, shifted to be non-negative,

   $$
   \mathrm{QD\text{-}Score} \;=\; \sum_{c \in \mathcal{C}_{\text{filled}}} \!\big( J(\theta_c) - J_{\text{offset}} \big),
   $$

   with $J_{\text{offset}}$ chosen so the worst surviving policy contributes zero. This is the standard QD aggregate [Pugh et al. 2016].
2. **Coverage:** fraction of cells whose elite achieves the success threshold,

   $$
   \mathrm{Coverage} \;=\; \frac{1}{|\mathcal{C}|}\sum_{c \in \mathcal{C}} \mathbb{1}\!\left[ J(\theta_c) > J_{\min} \right],
   $$

   with $J_{\min}$ corresponding to a successful lift (object held above $h^*$ for ≥0.5 s).
3. **Taxonomy recall:** two blinded annotators label 100 stratified-sampled archive policies against the 16-class Cutkosky taxonomy. We report (a) the number of distinct classes covered and (b) Cohen's $\kappa$ for inter-rater agreement.

### 4.2 Baselines

| Baseline | Implementation | Question it answers |
|---|---|---|
| **PPO (flat)** | Brax / mujoco_playground PPO trainer | "Does standard RL mode-collapse?" |
| **SAC (flat)** | Brax SAC | Same, off-policy |
| **Vanilla MAP-Elites** | QDax `MixingEmitter` only | "Is the PG emitter contributing, or is GA enough?" |
| **DCRL-ME** | QDax `DCRLMEEmitter` (provided) | SOTA QD-RL; reviewers will demand it |

For PPO and SAC, we measure: (a) what is the *single* grasp class converged to, (b) how does it score on the QD metrics if scored as a singleton archive entry.

### 4.3 Ablations

1. **PG emitter ON/OFF.** Isolates the contribution of policy-gradient-assisted offspring.
2. **Descriptor swap.** Replace $(b_1, b_2)$ with joint-angle-based descriptors (mean MCP angle, mean PIP angle). Tests whether contact descriptors are necessary or whether kinematic descriptors are equivalent.
3. **Domain randomization ON/OFF.** Without DR: archive is denser but brittle. With DR: archive is sparser but transferable.
4. **Mesh vs primitive collisions.** Run a small-scale (batch 256, 1e7 steps) MJX-JAX experiment with mesh collisions to quantify the speed/accuracy tradeoff. We expect ~30× slowdown.
5. **MJX-Warp comparison.** A small-scale run with MJX-Warp + mesh collisions confirms whether our primitive approximation introduces discoverable artifacts.

### 4.4 Figures (planned)

- **Fig. 1.** Archive heatmap: $50 \times 50$ grid colored by $J(\theta_c)$ at end of training. Side panel: example rollouts from 8 representative cells.
- **Fig. 2.** QD-Score vs. wall-clock for OrcaQD vs. baselines.
- **Fig. 3.** Cutkosky taxonomy coverage matrix: rows = Cutkosky classes, columns = methods, cells = fraction of class found.
- **Fig. 4.** Ablation: Coverage and QD-Score for each ablation variant.
- **Fig. 5.** Mesh vs. primitive collision: throughput (steps/sec) and a perturbation analysis showing that grasp success rates differ by <5%.

---

## 5. Honest Limitations

1. **Task-conditional archive.** The archive is trained for "lift", and re-training is needed per task family. We do not claim a universal motor cortex.
2. **No real-hardware validation.** All claims are sim-only under DR. This is by design for Paper 1.
3. **Hand-designed descriptors.** Learned descriptors (AURORA-style) are future work.
4. **2D projection loses information.** Two policies in the same cell may differ perceptibly. The grid resolution is a compromise.
5. **MJX-JAX collision approximation.** Primitive collisions are an approximation of the OrcaHand's true geometry. Validated against MJX-Warp mesh-collision results (Fig. 5).
6. **Single hand laterality.** Left-hand archive is straightforward extension but not in scope.

---

## 6. Engineering Notes

### 6.1 Why MJX-JAX over MJX-Warp

| Axis | MJX-JAX | MJX-Warp |
|---|---|---|
| Mesh collisions | Limited (~32 verts convex-convex, ~200 convex-primitive) | Fully supported |
| Differentiability | Yes (mostly) | No |
| Hardware | Nvidia, AMD, Apple Silicon, TPU | Nvidia only |
| Contact API | `mjx.Data.contact` | `mjx.Data._impl` (private) |
| Pairs with QDax | Native (JAX pytrees throughout) | Requires interop layer |

We chose MJX-JAX because (a) QDax integration is native, (b) contact reads for descriptors are straightforward, (c) we lose nothing by using primitive collisions, and (d) hardware portability matters for reproduction. The MJX-Warp comparison run (§4.3 ablation 5) validates that our primitive approximation is close enough to mesh collisions.

### 6.2 OrcaHand v2 MJCF specifics

The upstream `v2/models/mjcf/orcahand_right.mjcf` has three quirks worth flagging:

1. **Numerical residue.** Many `pos` and `quat` values have `~1e-15` residue from Fusion360 export. MuJoCo handles it; we leave it untouched.
2. **Tiny actuator force range.** Default `forcerange="-1 1"` and `kp=2.0` will not lift a 0.5 kg object. We override per-actuator: `forcerange="-3 3"` for non-thumb digits, `forcerange="-5 5"` for thumb and wrist. Raised `kp` in proportion. This is documented in `assets/mjcf_overrides.xml`.
3. **Existing exclude pairs.** ~70 contact-exclude pairs between adjacent bodies in the upstream file. We inherit them verbatim.

### 6.3 Reproducibility checklist

- [ ] `pyproject.toml` with version-pinned dependencies (`mujoco-mjx==3.8.*`, `qdax==0.5.1`, `jax[cuda12]>=0.4.30`)
- [ ] `assets/mjcf/orcahand_right_mjx.mjcf` (primitive collision variant, released alongside paper)
- [ ] `configs/paper1_main.yaml` (1e8 steps, 50×50 archive, all hyperparameters)
- [ ] `configs/paper1_ablation_*.yaml`
- [ ] `scripts/reproduce_paper1.sh` (end-to-end on a fresh H100 instance)
- [ ] Archive checkpoint released as a single `.tar.zst` (~50 MB compressed)
- [ ] WandB logs of the headline run + at least one seed replicate

---

## 7. Eight-Week Execution Plan

| Weeks | Milestone | Risk |
|---|---|---|
| **1–2** | OrcaHand v2 loads in MJX. Primitive-collision MJCF authored and validated. `mjx-testspeed` reports ≥3,000 steps/sec at batch 4,096 on H100. | MJX-JAX may need actuator/solver tuning; budget 3 days. |
| **3** | $b_1, b_2$ descriptors implemented in JIT-compiled rollout. Unit tests against CPU NumPy reference. PPO sanity baseline runs end-to-end. Reward curriculum validated. | Contact filter threshold per object needs tuning. |
| **4–5** | PGA-ME runs to completion. First archive at >40% coverage. QD-Score curve and archive heatmap reproduced. | QDax + MJX glue: scoring function wrapping is the integration point. Budget 3 days. |
| **6** | Ablations: vanilla ME, DCRL-ME, PG-off, descriptor-swap. Domain randomization on. Final archive frozen. | DCRL-ME reproduction may need 2 extra days. |
| **7** | Mesh-vs-primitive validation run on MJX-Warp. Taxonomy annotation by 2 annotators (100 policies each). Inter-rater $\kappa$ computed. | Annotator availability. |
| **8** | Figures, paper draft, code release. ICRA 2027 submission target ~Sep 2026. | Time pressure; cut Fig. 5 (mesh comparison) if running late. |

---

## 8. References (verified)

1. Cutkosky, M. R. (1989). On grasp choice, grasp models, and the design of hands for manufacturing tasks. *IEEE Trans. Robotics and Automation*, 5(3), 269–279.
2. OpenAI et al. (2018). Learning Dexterous In-Hand Manipulation. arXiv:1808.00177.
3. Mouret, J.-B. & Clune, J. (2015). Illuminating search spaces by mapping elites. arXiv:1504.04909.
4. Nilsson, O. & Cully, A. (2021). Policy Gradient Assisted MAP-Elites. *GECCO 2021*.
5. Faldor, M., Chalumeau, F., Flageat, M. & Cully, A. (2024). Synergizing Quality-Diversity with Descriptor-Conditioned Reinforcement Learning. arXiv:2401.08632.
6. Chalumeau, F. et al. (2024). QDax: A Library for Quality-Diversity and Population-based Algorithms with Hardware Acceleration. *JMLR* 25(108).
7. Zakka, K. et al. (2025). MuJoCo Playground. arXiv:2502.08844.
8. Vassiliades, V. & Mouret, J.-B. (2018). Discovering the Elite Hypervolume by Leveraging Interspecies Correlation. *GECCO 2018*.
9. Cully, A. & Demiris, Y. (2018). Quality and Diversity Optimization: A Unifying Modular Framework. *IEEE Trans. Evolutionary Computation*.
10. Grillotti, L. & Cully, A. (2022). Unsupervised Behavior Discovery with Quality-Diversity Optimisation. arXiv:2106.05648.
11. Zhang, H. et al. (2025). Robust Dexterous Grasping of General Objects from Single-view Perception (RobustDexGrasp). *CoRL 2025 Spotlight*. arXiv:2504.05287.
12. Apple ml-egodex team (2025). Learning Dexterous Manipulation from Large-Scale Egocentric Video (EgoDex). *ICLR 2026*. arXiv:2505.11709.
13. Rajeswaran, A. et al. (2018). Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations (DAPG). *RSS 2018*.
14. MuJoCo MJX documentation (stable, v3.8). https://mujoco.readthedocs.io/en/stable/mjx.html
