# Paper Plan: Movement-Scoring Graph Policies for Heterogeneous Traffic Networks

## Purpose and scope

This paper presents a compact representation and control interface for applying one shared graph-neural-network traffic-signal policy to OSM-derived networks with different topology, movement sets, and phase counts. The work is positioned as a clear systems and representation contribution supported by preliminary multi-city evidence. It does not claim a new reinforcement-learning algorithm, universal baseline superiority, or conclusive cross-city generalization.

The first draft should be comprehensive. Later revisions may shorten, reorder, or move details to appendices. Every statement about the implementation or experiment must be traceable to maintained documentation, current code, the iteration-85 evidence bundle, or an original external source.

## Working title

**Movement-Scoring Graph Policies for Traffic Signal Control Across Heterogeneous City Networks**

Alternative titles:

- **From Movement Scores to Legal Phases: A Shared Graph Policy for Heterogeneous Traffic Networks**
- **Shared Movement-Scoring Policies for OSM-Derived Traffic Signal Networks**

## Central research question

Can a traffic-signal policy operate across road networks with different topology and phase structure by learning shared movement scores instead of network-specific phase actions?

## Central thesis

The controller separates traffic representation, learned prioritization, and legal action construction:

1. directed `LaneGroup` nodes represent corridor state;
2. `Movement` nodes represent legal signal-controlled turns from an input LaneGroup to an output LaneGroup;
3. a shared typed GNN produces one scalar priority per Movement;
4. a junction-specific binary incidence matrix sums Movement scores into phase logits;
5. a runtime mask removes temporarily unavailable phases;
6. conflict-derived phase sets ensure that the policy acts only over supported compatible signal states.

This factorization lets the learned parameter shapes remain independent of city graph size, the number of controlled movements, and the number of phases at each junction.

## Defensible contribution statement

The contribution is a transparent movement-level network encoding and structured control interface that combines a shared city-graph GNN with junction-local, conflict-derived phase spaces. The iteration-85 experiment provides preliminary evidence that the resulting scratch-trained policy can remain competitive across four rollout cities and transfer well to Freiburg, which generated no PPO rollouts.

The paper should describe iteration 85 as the completed endpoint of the reported run, not as a Freiburg-optimized checkpoint. Freiburg was periodically observed during development, so it is a held-out evaluation or validation city rather than an untouched final test set.

## Intended claim boundaries

The paper may claim:

- one parameter set operates on all five evaluated city graphs;
- the architecture supports variable graph, movement, and phase counts by construction;
- Freiburg generated no PPO rollouts;
- learned control exceeded both reported baselines on Heidelberg and Freiburg throughput at iteration 85;
- learned control was close to the strongest baseline in Karlsruhe and Stuttgart and weaker in Mannheim;
- the result is encouraging preliminary evidence for cross-network transfer.

The paper must not claim:

- state-of-the-art performance;
- superiority in every city;
- conclusive generalization;
- an untouched Freiburg test protocol;
- that architecture alone guarantees transfer;
- that conflict synthesis constitutes formal real-world signal-safety verification;
- that the additive movement-to-phase principle is independently novel.

## Mathematical core

Define a heterogeneous directed graph

\[
G=(V_L\cup V_M,E),
\]

where `LaneGroup` nodes \(V_L\) represent directed corridors and `Movement` nodes \(V_M\) represent controlled turns. Junctions own movements and phase sets but are not GNN nodes.

For each Movement \(m\), the shared policy produces

\[
s_m=f_\theta(G,X)_m.
\]

For junction \(j\), let \(M_j\) be its local movements, \(P_j\) its synthesized selectable phases, and

\[
A_j\in\{0,1\}^{|P_j|\times |M_j|}
\]

its phase-incidence matrix. Phase logits are

\[
\boldsymbol{\ell}_j=A_j\mathbf{s}_j,
\qquad
\ell_{j,p}=\sum_{m\in M_j}A_{j,p,m}s_m.
\]

The runtime legal mask \(q_{j,t}\in\{0,1\}^{|P_j|}\) yields

\[
\tilde\ell_{j,p}=
\begin{cases}
\ell_{j,p}, & q_{j,t,p}=1,\\
-\infty, & q_{j,t,p}=0,
\end{cases}
\]

and PPO uses the per-junction categorical policy

\[
\pi_\theta(a_{j,t}=p\mid G_t,X_t)
=\operatorname{softmax}(\tilde{\boldsymbol\ell}_j)_p.
\]

The sum is an intentional inductive bias: a phase serving several positively scored compatible movements accumulates their priorities. This is a design rationale, not an ablation-backed claim that summation is always preferable to normalized aggregation.

## Detailed section plan

### Abstract

The abstract should contain five elements:

1. Fixed global phase-action heads bind learned controllers to particular intersection schemas.
2. The proposed representation separates directed corridor state, controlled movements, and locally legal phases.
3. A shared GNN scores movements and binary incidence matrices sum scores into phase logits.
4. The controller was trained from scratch with PPO rollouts from Karlsruhe, Mannheim, Stuttgart, and Heidelberg and evaluated additionally on Freiburg.
5. At iteration 85 it was competitive on several rollout cities and exceeded max-pressure and queue baselines on Freiburg, while the single-run development protocol limits the generalization claim.

Avoid detailed hyperparameters and percentage-heavy result prose in the abstract.

### 1. Introduction

Motivate the structural problem rather than generic urban congestion:

- intersections expose different numbers and definitions of legal phases;
- a conventional fixed neural output associates weights with fixed action indices;
- padding or canonical mapping can hide structural differences or require a predefined intersection template;
- movements are common control primitives across heterogeneous junctions.

Introduce the hierarchy:

\[
\text{corridor state}\rightarrow\text{movement priority}\rightarrow\text{compatible phase score}.
\]

State contributions explicitly:

1. a city-level LaneGroup/Movement graph that distinguishes controlled and uncontrolled junction transitions;
2. a movement-scoring policy with non-learned local incidence aggregation and runtime legality masks;
3. an OSM-to-SUMO pipeline that derives selectable phases from controlled links and request conflicts;
4. a preliminary scratch-PPO study across five heterogeneous city scenarios.

End with the research question and calibrated result statement.

### 2. Related work

Organize by conceptual relationship.

#### Movement pressure and phase scoring

Explain that max-pressure assigns priorities to movements from upstream/downstream imbalance and scores a stage or phase from the served movements. Cite Varaiya and PressLight. This is the clearest lineage for additive movement-to-phase aggregation.

#### Movement- and phase-structured learned control

Discuss FRAP as a movement-feature, phase-demand, and phase-competition architecture with symmetry-oriented sharing. Discuss Advanced-XLight as evidence that traffic-state representation and movement pressure remain central to learned TSC performance.

#### Shared and transferable policies

Discuss AttendLight, MetaLight, GESA, and inductive graph RL. Distinguish rapid adaptation, canonical mapping, parameter sharing, and zero-shot evaluation.

#### Closest comparison: TransferLight

TransferLight uses a hierarchical local segment-to-movement-to-phase graph, learned attention, weight-tied intersection agents, domain-randomized training networks, and learned phase energies. This work instead constructs a city-level LaneGroup/Movement message graph, ends the learned actor at scalar movement scores, and applies a fixed local incidence sum over conflict-derived phases. TransferLight has a stronger zero-shot evaluation protocol; the present work emphasizes a simpler, explicit control interface and an OSM/SUMO construction workflow.

Do not imply an implementation comparison or reproduce reported performance across incompatible benchmarks.

### 3. Problem formulation

Define:

- a city network and its controllable junctions;
- local Movement sets and selectable phase sets;
- a target phase decision every ten simulated seconds;
- legal actions as the intersection of the synthesized phase set and runtime availability;
- the shared-parameter objective across several city environments.

Clarify terminology. In SUMO, a signal state string is a simultaneous state over controlled links. In this paper, a selectable phase is one synthesized green state corresponding to a maximal compatible set of atomic controlled-link groups. Yellow transitions are execution states, not policy actions.

### 4. Movement-level graph policy

#### 4.1 LaneGroups

Describe LaneGroups as directed corridors between controller-relevant endpoints. Opposite directions are different nodes. Corridor contraction may cross unambiguous unsignalized continuation but stops before ambiguous branches or reachability shortcuts.

Summarize dynamic and static feature categories. Keep the complete 29-feature schema in an appendix.

#### 4.2 Movements

Define a Movement as a legal controlled input-LaneGroup to output-LaneGroup turn associated with a traffic light. Explain why this node is the correct interface between upstream demand and downstream supply. Mention the four Movement features without overstating their sophistication.

#### 4.3 Typed message passing

List the four directed edge types:

- input LaneGroup to Movement;
- output LaneGroup to Movement;
- Movement to input LaneGroup;
- Movement to output LaneGroup.

Explain that message direction is not traffic direction. Output-to-Movement messages expose downstream storage and spillback context.

At non-controllable junctions, directed LaneGroup-to-LaneGroup connectors carry information with weight

\[
w=\exp(-t_{\mathrm{ff}}/30\ \mathrm{s}).
\]

Signalized junctions do not receive bypass connectors. The reference checkpoint uses one macro-hop.

#### 4.4 Movement scoring and local phase aggregation

Present the central equations. Explain that the same scorer operates regardless of the number of movements. Each junction supplies a ragged local incidence matrix, so phase-logit dimensions are determined by the current network rather than by learned output weights.

Include a small worked matrix example. The calculation should use internally consistent scores and phase sums and show one phase unavailable after aggregation.

#### 4.5 Runtime legality and transitions

Explain minimum green, continued phases, deterministic yellow transitions, and the runtime assertion that an action allowed by the PPO mask is accepted by the controller. Forced single-action decisions train the critic but do not contribute actor or entropy loss.

### 5. Conflict-derived phase synthesis

Explain that phases are not read directly from OSM signal tags.

Pipeline:

1. shape the OSM network;
2. use SUMO `netconvert` to derive lane connections, controlled-link indices, junction request indices, and foes information;
3. form atomic controlled-link groups that must activate together;
4. reject internally conflicting groups;
5. mark cross-group conflict using SUMO foes in either direction;
6. additionally conflict different incoming approaches that enter the same outgoing edge;
7. form the compatibility graph;
8. enumerate all maximal cliques with Bron-Kerbosch;
9. convert each maximal clique to a selectable green phase.

Emphasize maximal versus maximum. Retaining all maximal compatible sets preserves smaller protected phases that cannot be extended, rather than only retaining the largest movement set.

Explain the boundary of the safety claim: phases are compatible under the implemented SUMO-derived and added merge-conflict rules.

### 6. OSM-derived city workflow

Describe cached OSM sources, build recipes, replayable pruning decisions, generated SUMO networks, routes, additional files, graph inspection, and demand calibration. Explain that scenario construction includes manual shaping and synthetic demand; OSM geometry alone is not a calibrated traffic benchmark.

Keep operational command examples out of the main paper unless included in a reproducibility appendix.

### 7. PPO training

Give the conceptual training procedure:

- shared actor-critic initialized from scratch;
- persistent libsumo workers collect variable-size city trajectories;
- one categorical action is sampled independently for every controllable junction from its currently legal phase logits;
- generalized advantage estimation and clipped PPO update the shared parameters;
- critic values are pooled per traffic light from local movement embeddings;
- fixed-length nonterminal rollouts bootstrap from the next state.

Main-text setup facts:

- rollout cities: Karlsruhe, Mannheim, Stuttgart, Heidelberg;
- 32 persistent workers;
- 40 rollout jobs per update, ten per rollout city;
- 350 policy decisions per rollout;
- two update epochs;
- no value warm-up;
- hidden dimension 64;
- one macro-hop;
- 29 LaneGroup and four Movement features.

Move the complete reward coefficients and configuration table to an appendix, but mention that the optimized reward emphasizes local throughput with global, progress, gridlock, and speed-change terms.

### 8. Experimental protocol

State that iteration 85 is the completed endpoint of the reported run. Confirm this wording against the console log before finalization.

Evaluation:

- learned actions sampled at temperature 1.0;
- seeds 100 through 105;
- demand scale 1.0;
- 1,200 simulated seconds;
- 120 policy decision opportunities at ten-second intervals;
- deterministic max-pressure and longest-queue baselines over the same synthesized legal phase sets.

Freiburg generated no PPO rollouts but was periodically evaluated during development. It is therefore held out from rollout generation, not an untouched final test city.

Define metrics:

- throughput as completed vehicles per simulated hour;
- completion as completed divided by inserted vehicles;
- wait density according to the repository implementation;
- completed-trip averages as conditional on trip completion.

Explain why completion and wait density accompany completed-trip waiting time.

### 9. Results

#### 9.1 Main iteration-85 comparison

Use a grouped throughput plot with learned, max-pressure, and queue values per city. Add error bars from the seed-level export. Visually separate Freiburg from rollout cities.

Use a result table containing throughput, completion, and wait density. The exact mean values are sourced from the iteration-85 evaluation export and maintained result report.

Interpret city by city:

- Karlsruhe: learned close to max pressure and above queue;
- Mannheim: learned materially below both throughput baselines;
- Stuttgart: learned effectively tied with queue and close to max pressure;
- Heidelberg: learned above both throughput baselines;
- Freiburg: learned substantially above both baselines in throughput, with higher completion and lower wait density.

Avoid significance language. Six fixed seeds and a single training run do not support it.

#### 9.2 Learning trajectory

Use one combined plot if it remains legible. Color denotes city. Learned trajectories use solid lines; baseline references use lighter dashed or dotted lines. If this becomes too dense, retain a compact learned-only overview in the main text and move per-city comparisons to the appendix.

Explain that Freiburg improvement occurred despite zero rollout generation, while acknowledging that periodic evaluation made it visible during development.

#### 9.3 Multi-metric interpretation

Optionally include a small throughput-completion plane or move it to the appendix. Use it to show that high throughput should be interpreted jointly with how many inserted vehicles complete. Discuss Mannheim as a meaningful weakness rather than smoothing it into the aggregate story.

### 10. Discussion

Discuss what the experiment supports:

- the representation is operational across heterogeneous city graphs;
- one scratch-trained parameter set reaches competitive performance across several scenarios;
- Freiburg provides encouraging held-out-rollout evidence;
- the simple movement-score interface is sufficient to produce useful local phase choices.

Discuss plausible benefits without causal claims:

- shared movement semantics avoid fixed global phase identities;
- downstream LaneGroup context can expose supply and spillback;
- incidence aggregation keeps the learned head independent of local phase count;
- conflict-derived phases separate learning from signal compatibility.

Discuss the relation to TransferLight carefully: both use movements as an intermediate abstraction, but they differ in graph scope, learned phase representation, aggregation, training distribution, and empirical protocol.

### 11. Limitations

List concrete limitations:

- one independent training run;
- one completed endpoint;
- six fixed periodic evaluation seeds;
- Freiburg observed during development;
- learned evaluation includes categorical sampling variability;
- synthetic demand and manually shaped OSM scenarios;
- relatively low completion in some cities at the finite horizon;
- Mannheim underperformance;
- no learned generalist baseline such as TransferLight was reimplemented;
- no causal ablation of LaneGroup/Movement encoding, message passing, reward components, or phase aggregation.

### 12. Future work

Organize future work into four programs.

#### Stronger generalization protocol

- multiple independent training seeds;
- frozen fresh scenario seeds;
- confidence intervals and paired seed-level comparisons;
- another OSM city excluded from training, monitoring, and checkpoint decisions;
- broader topology and demand distributions;
- longer-horizon evaluation under different congestion regimes.

#### Representation and communication ablations

- zero-hop local movement scorer versus typed message passing;
- removal of downstream-supply messages;
- removal of unsignalized connector edges;
- one versus several macro-hops;
- alternative corridor contraction rules;
- sensitivity to detector extent and feature schema.

#### Reward and phase-interface studies

- individual reward-component removal and weight sensitivity;
- local versus global throughput terms;
- progress and gridlock shaping under oversaturation;
- incidence sum versus mean, max, learned weighted sum, or attention;
- explicit phase-size normalization or saturation-flow weighting;
- phase-synthesis alternatives, including non-maximal operational phases and transition-aware scoring.

The paper should explain the current rationale for summation: simultaneously serving more positively scored compatible movements can represent greater discharge opportunity. Future comparisons would test when that bias helps or hurts.

#### Operational realism

- calibrated origin-destination demand;
- pedestrians, public transport, and heterogeneous vehicle classes;
- realistic signal timing constraints and ring-barrier programs;
- robustness to sensing noise and missing detector information;
- simulation-to-field transfer and safety validation.

### 13. Conclusion

Restate the representation result rather than claiming benchmark dominance. The final sentence should emphasize that movements provide a shared learned vocabulary while synthesized phases remain local to each junction.

## Planned figures

### Figure 1: Whole-method overview

Use `docs/assets/movement-scoring-generalist-policy.svg` as the working figure. It should show two heterogeneous graph/action instances passing through one shared GNN and then through different local incidence matrices and masks.

The generated raster experiment is not selected for the paper because it is less precise and less visually coherent.

### Figure 2: LaneGroup/Movement graph

Use or adapt `docs/assets/movement-graph-3x3.png`. The caption must explain that junctions are spatial anchors rather than GNN nodes and that unsignalized connectors differ from signalized Movement paths.

### Figure 3: Worked phase-incidence calculation

Create a compact deterministic diagram or equation block with:

- four or six movement scores;
- two or three binary incidence rows;
- correct summed phase logits;
- one masked phase;
- one selected legal phase.

This may be embedded into Figure 1 if the overview remains readable.

### Figure 4: Phase synthesis

Create a deterministic three-stage diagram:

1. controlled links and conflicts;
2. atomic-group compatibility graph;
3. all maximal compatible phases.

Include a smaller maximal set to visually distinguish maximal from maximum.

### Figure 5: Throughput comparison

Grouped bars or points with seed dispersion. Separate Freiburg visually. Use consistent policy colors throughout the paper.

### Figure 6: Learning trajectories

Prefer one combined main-text panel. Move crowded per-city panels and PPO diagnostics to the appendix.

### Optional appendix figure: throughput-completion plane

Plot city-policy means and dispersion if readable. This is supporting interpretation, not a primary claim.

## Planned tables

### Table 1: Conceptual comparison with prior work

Columns:

- method;
- graph or traffic representation;
- agent scope;
- action construction;
- support for varying phase sets;
- parameter sharing;
- cross-network evaluation.

Rows: max pressure, PressLight, FRAP, GESA, TransferLight, this work. Verify every cell from the original paper.

### Table 2: City structural statistics

For each city report:

- rollout/evaluation role;
- total junction count;
- controllable signalized junction count;
- LaneGroups;
- Movements;
- unsignalized connectors;
- total typed message edges;
- total synthesized selectable phases;
- mean and range of phases per controlled junction;
- a network-size measure such as lane length.

Generate these values from current saved scenarios and code rather than estimating them.

### Table 3: Iteration-85 results

Report policy-by-city throughput, completion, and wait density. Include dispersion either in the table or Figure 5, but avoid excessive duplication.

### Appendix table: training and evaluation configuration

Include architecture, workers, rollout allocation, reward weights, demand ranges, PPO settings, evaluation seeds, horizon, interval, action sampling, and baselines.

## Core references to verify and cite

- Varaiya, *Max Pressure Control of a Network of Signalized Intersections*, Transportation Research Part C, 2013.
- Wei et al., *PressLight: Learning Max Pressure Control to Coordinate Traffic Signals in Arterial Network*, KDD 2019.
- Zheng et al., *Learning Phase Competition for Traffic Signal Control*, CIKM 2019.
- Wei et al., *CoLight: Learning Network-level Cooperation for Traffic Signal Control*, CIKM 2019.
- Chen et al., *Toward A Thousand Lights*, AAAI 2020.
- Zang et al., *MetaLight*, AAAI 2020.
- Oroojlooy et al., *AttendLight*, NeurIPS 2020.
- Zhang et al., *Expression Might Be Enough*, ICML 2022.
- Devailly et al., inductive/model-based graph RL for traffic-signal control.
- Jiang et al., *A General Scenario-Agnostic Reinforcement Learning for Traffic Signal Control*, IEEE T-ITS 2024.
- Schmidt et al., *TransferLight: Zero-Shot Traffic Signal Control on any Road-Network*.
- Schulman et al., *Proximal Policy Optimization Algorithms*.
- Bron and Kerbosch, maximal-clique enumeration.
- SUMO and official SUMO OSM/network-import documentation.

## Repository sources of truth

- `README.md`
- `docs/architecture.md`
- `docs/city_pipeline.md`
- `docs/training_and_evaluation.md`
- `docs/results/city_first_pass_throughput_scratch_32_worker.md`
- `configs/training/city_first_pass_throughput_scratch_32_worker.yaml`
- movement graph, model, phase-logit, runtime, synthesis, PPO, and evaluation modules under `src/movement/`
- iteration-85 evidence under `artifacts/ppo_runs/city_first_pass_throughput_progress_025_sample_eval_v3/`

Historical files under `docs/outdated/` may be used only to recover reasoning that can be independently verified against current code or maintained documentation.

## Drafting and verification workflow

1. Draft from this plan and maintained sources only.
2. Use citation placeholders only where the original paper still needs method-level verification.
3. Do not invent city structural statistics; mark their table as pending extraction.
4. Use exact iteration-85 result values from the evidence export.
5. Keep claims conditional and distinguish architectural support from empirical evidence.
6. After the first draft, extract city statistics and generate missing figures.
7. Resolve every citation placeholder from primary sources.
8. Audit each equation against code and each experimental number against the evidence bundle.
9. Revise for narrative structure and concision only after technical verification.
