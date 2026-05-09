# Beyond Final Answers: Trajectory-Level Hallucination Detection in Multi-Agent Industrial Workflows

**STATGR5293 — Generative AI Using LLMs**
Section 007 (Wednesday) — Columbia University, Spring 2026

**Instructor:** Parijat Dube
**Project PI:** Dhaval C. Patel (IBM Research)

---

## Abstract

Large Language Models (LLMs) are increasingly deployed as autonomous agents that reason, use tools, and act across multiple steps. Yet most hallucination benchmarks evaluate only the final output, missing failures that originate in intermediate Thought–Action–Observation steps. This project introduces **Trajel**, a framework and dataset for auditing trajectory-level hallucinations in multi-agent industrial workflows, built on top of IBM Research's AssetOpsBench benchmark. We define a five-type hallucination taxonomy — factual, referential, logical, procedural, and scope-based — grounded as structural predicates over the execution trace. We release a dataset of 225 expert-annotated agent trajectories labeled under this taxonomy by human reviewers from Columbia University and IBM Research, with a human-identified hallucination rate of 68.3%. We benchmark three supervised detection paradigms (BERT, NLI, Longformer) alongside an LLM-as-a-Judge baseline, finding that all supervised models remain below 0.70 AUC. A signal analysis of execution-quality flags reveals that the clarity-and-justification signal alone achieves AUC = 0.908, substantially outperforming all trained classifiers and suggesting lightweight runtime monitors as the most practical path to real-time hallucination detection.

---

## 1. Introduction

The transition from static LLMs to autonomous agentic systems represents a fundamental shift in how AI is deployed. In high-stakes domains such as industrial monitoring and infrastructure maintenance, agents are no longer text generators — they are decision-making entities that parse sensor signals, follow multi-step procedures, and coordinate across tool-augmented workflows. With this autonomy comes a more dangerous and complex failure mode: **trajectory-level hallucination**.

In an agentic setting, a hallucination is not simply a factual error in a final response. It is a structural deviation from evidence that propagates through a sequential, tool-mediated trajectory, often causing cascading failures downstream. A hallucination at step $t$ corrupts the state consumed by every subsequent step $t' > t$, meaning the final answer can appear correct while the reasoning that produced it was broken three steps earlier.

Despite this, the science of hallucination evaluation has remained largely tethered to static benchmarks. Existing approaches — SQuAD, TruthfulQA, HotpotQA — treat every task as an isolated input–output pair and never examine the trace. This makes it nearly impossible to diagnose whether a system failed because it misunderstood the environment or because it invented a state that did not exist.

This project addresses that gap. We make the following contributions:

1. **A five-type hallucination taxonomy** that formalizes hallucination as a structural predicate over the Thought–Action–Observation trace, distinguishing factual, referential, logical, procedural, and scope-based failures.
2. **The Trajel dataset**, comprising 225 expert-annotated trajectories from IBM Research's AssetOpsBench multi-agent benchmark, labeled through a two-phase protocol combining LLM-as-a-Judge automation with blind human review from two institutions.
3. **Three supervised detection paradigms** — subtask-level BERT classification, trajectory-level natural language inference (NLI), and long-context Longformer modeling — benchmarked against the LLM-as-a-Judge baseline.
4. **Empirical validation of execution-quality signals**, showing that runtime flags are strongly predictive of hallucination and that a single signal achieves AUC = 0.908, outperforming all trained classifiers.

---

## 2. Related Work

### 2.1 Agent Benchmarks and Multi-Step Evaluation

ReAct [Yao et al., 2023] interleaves reasoning and action to expose intermediate traces, while AgentBench [Liu et al., 2023] evaluates LLMs as agents across diverse tasks and shows sharp performance degradation on long-horizon problems. WebArena [Deng et al., 2024] and HotpotQA [Yang et al., 2018] provide human-annotated environments and multi-hop reasoning chains respectively, but large-scale human-labeled agent trajectory datasets remain scarce. Critically, none of these benchmarks formalize hallucination as a structural phenomenon within the trace.

### 2.2 Hallucination in Tool-Augmented Settings

ToolBH [Wu et al., 2024] and MIRAGE-Bench [Zhang et al., 2025] both demonstrate that hallucinations in agents predominantly arise during intermediate reasoning and tool use rather than in final outputs. However, neither formalizes hallucination types as structural predicates over the trace. TruthfulQA [Lin et al., 2022] and MIRAGE [Dong et al., 2025] show that scaling alone does not eliminate hallucinations under complex reasoning, reinforcing the need for benchmarks that vary model size, interaction complexity, and alignment strategy together.

### 2.3 Hallucination Taxonomies

Cognitive Mirage [Ye et al., 2023] categorizes factual, logical, and contextual errors. MIRAGE-Bench introduces an agentic taxonomy over instruction, history, and observation inconsistencies. Critically, none of these prior taxonomies separate **procedural violations** (broken workflow ordering) from **scope-based violations** (a correct claim made by the wrong agent) — a distinction that is essential in multi-agent industrial settings and central to our work.

### 2.4 Multi-Agent Failure Analysis

Cemri et al. [2025] introduce MAST, a 14-mode failure taxonomy validated on 1,600+ annotated traces across seven multi-agent frameworks. Their categories cover general MAS failures but do not isolate hallucination as a distinct phenomenon, nor do they formalize failure types as structural predicates over the execution trace. TRAJECT-Bench [He et al., 2026] evaluates tool-use correctness along agentic trajectories but does not address hallucination detection or multi-agent coordination.

### 2.5 Positioning Trajel

Table 1 situates Trajel against existing benchmarks across six axes. To our knowledge, Trajel is the first benchmark to combine industrial multi-agent trajectories with full trajectory-level evaluation, a structurally grounded five-type taxonomy, expert human annotations from two institutions, and LLM-as-a-Judge baselines.

**Table 1: Comparison of agentic evaluation benchmarks.**

| Benchmark | Industrial | Multi-Agent | Trajectory | Taxonomy | Human | LLM Judge |
|---|---|---|---|---|---|---|
| AgentBench | No | Yes | No | No | No | No |
| WebArena | No | Yes | No | No | Yes | No |
| MIRAGE-Bench | No | No | Partial | Yes | No | Yes |
| ToolBH | No | Yes | Partial | Yes | No | Yes |
| MAST | No | Yes | Full | Partial | Yes | Yes |
| TRAJECT-Bench | No | No | Full | No | No | No |
| AssetOpsBench | Yes | Yes | Partial | No | Yes | Yes |
| **Trajel (Ours)** | **Yes** | **Yes** | **Full** | **Yes** | **Yes** | **Yes** |

---

## 3. Problem Formulation

### 3.1 Trajectory Structure

We model an agentic workflow as a compound AI system $\Phi = (\mathcal{M}, \mathcal{C}, \mathcal{T}_{\text{tool}})$, where $\mathcal{M} = \langle M_1, \ldots, M_K \rangle$ is a set of LLM-driven agent modules, $\mathcal{C}$ is the orchestrator (Plan-and-Execute), and $\mathcal{T}_{\text{tool}}$ is the tool set. In AssetOpsBench, $K = 4$ with agents $\mathcal{A} = \{\text{IoT, FSMR, TSFM, WO}\}$ covering perception, state modeling, temporal forecasting, and execution.

Execution proceeds as a sequence of steps, each a Thought–Action–Observation triple produced by one agent:

> **Definition 1 (Trajectory).** A step is $s_t = (a_t, \tau_t, \alpha_t, \omega_t)$ where $a_t \in \mathcal{A}$, $\tau_t$ is the thought (reasoning), $\alpha_t$ is the action (tool invocation), and $\omega_t = T(\alpha_t)$ is the observation (tool return). A trajectory is the ordered trace $\mathcal{T} = (s_1, \ldots, s_N)$.

Let $E_t = \{\omega_1, \ldots, \omega_{t-1}\}$ denote the evidence set at step $t$, and $K$ the task specification. In AssetOpsBench, $\mathcal{T}$ is serialized as a JSON array — a causally ordered information stream in which every step has access to all prior evidence and may therefore reference, misreference, or fabricate upstream content.

### 3.2 Hallucination Taxonomy

> **Definition 2 (Hallucination).** Let $g_t \in \{\tau_t, \alpha_t\}$. A hallucination is the predicate $h(g_t \mid E_t, K, a_t) = 1$ iff $g_t \not\models E_t \lor g_t \not\models K \lor g_t \not\models \text{role}(a_t)$, where $\models$ denotes semantic entailment and $\text{role}(a_t)$ encodes the operational mandate of agent $a_t$.

We refine this into five categories $\mathcal{H} = \{h^F, h^R, h^L, h^P, h^S\}$:

- **Factual ($h^F$):** Asserts a claim contradicted by ground-truth data. Detectable from a single step in isolation.
- **Referential ($h^R$):** References an entity or prior result absent from the trace history. The model "remembers" something that never happened.
- **Logical ($h^L$):** Reasoning does not follow from its premises, even when those premises are correct. A broken inference chain.
- **Procedural ($h^P$):** Skips, reorders, or fabricates a required workflow step. Invisible without knowledge of the prescribed workflow.
- **Scope ($h^S$):** Agent acts outside its assigned mandate. Content may be correct but originates from the wrong agent. Unique to multi-agent settings.

The five types are mutually clarifying rather than disjoint: a single step may exhibit several simultaneously, motivating a multi-label formulation. They are also ordered by the context required for detection: $h^F$ requires only the step and ground truth; $h^R, h^L$ additionally require $E_t$; $h^P$ further requires $K$; and $h^S$ further requires $\text{role}(a_t)$. This ordering predicts the relative strengths of subtask-level versus trajectory-level detectors.

### 3.3 Research Questions

- **RQ1 (Prevalence):** What is the empirical distribution of hallucination types, and are certain types concentrated in specific agents or trace positions?
- **RQ2 (Localization):** Can we identify the originating step and distinguish hallucination from co-occurring execution errors?
- **RQ3 (Detection Modeling):** How do subtask-level classification, trajectory-level NLI, and long-context modeling compare?
- **RQ4 (Predictive Signals):** Which execution-quality signals most reliably predict hallucination early enough for real-time intervention?

---

## 4. The Trajel Dataset

### 4.1 Composition

Trajel comprises 225 annotated trajectories generated by the AssetOpsBench multi-agent framework. Each trajectory is a complete execution trace — a JSON-serialized sequence of Thought–Action–Observation steps interleaved across four domain agents (IoT, FSMR, TSFM, WO) — produced in response to one of 42 industrial operations questions spanning sensor retrieval, anomaly detection, failure-mode identification, and work-order generation.

Trajectories are generated by 6 distinct model configurations, yielding a model × question matrix that enables controlled comparison of hallucination behavior across architectures on identical tasks.

**Table 2: Trajel dataset overview.**

| Statistic | Value |
|---|---|
| Total annotated trajectories | 225 |
| Unique model configurations | 6 |
| Unique task questions | 42 |
| Annotation institutions | 2 (Columbia University, IBM Research) |
| Human-identified hallucination rate | 68.3% (153/224) |
| LLM-judge-identified hallucination rate | 79.1% (178/225) |
| Single-type hallucinations | 79 (51.3%) |
| Multi-type hallucinations | 75 (48.7%) |

### 4.2 Annotation Protocol

Each trajectory is evaluated independently by two parties: an LLM-as-a-Judge and a human reviewer. The human reviewer annotates **blind** — without access to the LLM judge's output — to prevent anchoring bias.

For each trajectory, annotators record four fields:
1. **Hallucination presence:** Binary label (hallucinated or correct).
2. **Hallucination type(s):** One or more categories from $\mathcal{H}$.
3. **Localization:** The specific step and component (Thought, Action, Observation, or Response) where the hallucination originates.
4. **Rationale:** Free-text explanation of the reviewer's judgment.

**Review principles.** Agent failure is not automatically classified as hallucination unless there is clear evidence of explicit information fabrication. Repetition alone does not indicate hallucination, and self-correcting behavior reflects adaptive reasoning. The primary objective is to determine both the specific hallucination category and the precise location at which it occurs.

Human reviewers are drawn from two institutions: Columbia University (148 reviews) and IBM Research (77 reviews). Agreement rates are comparable across institutions (Columbia: 64.9%, IBM: 59.7%), suggesting consistent annotation standards and cross-institutional reliability.

### 4.3 Annotation Quality

**Table 3: LLM-as-a-Judge vs. human annotation (n = 224).**

| | Human: No Hall. | Human: Hall. |
|---|---|---|
| **Judge: No Hall.** | 35 (TN) | 12 (FN) |
| **Judge: Hall.** | 36 (FP) | 141 (TP) |

The LLM judge achieves 92.2% recall against human labels but only 79.7% precision, with a Cohen's κ of 0.456 (moderate agreement). The judge missed 12 hallucinations that humans caught (false negatives) and over-flagged 36 correct trajectories (false positives). In 32 additional cases, both identified a hallucination but disagreed on type or location — showing that hallucination *detection* is easier than hallucination *classification*, and motivating the taxonomy-aware evaluation our benchmark provides.

**Table 4: Per-type Cohen's κ between LLM judge and human annotators.**

| Type | Both | Agent Only | Human Only | Neither | Cohen's κ |
|---|---|---|---|---|---|
| Scope ($h^S$) | 33 | 9 | 16 | 167 | 0.656 |
| Procedural ($h^P$) | 78 | 26 | 17 | 104 | 0.613 |
| Factual ($h^F$) | 50 | 24 | 15 | 136 | 0.595 |
| Logical ($h^L$) | 4 | 6 | 17 | 198 | 0.211 |
| Referential ($h^R$) | 3 | 7 | 14 | 201 | 0.176 |

The per-type κ pattern is consistent with the context-ordering hypothesis: types requiring only local evidence show moderate-to-substantial agreement (κ ≥ 0.595), while types requiring cross-step reasoning show only slight agreement (κ ≤ 0.211). This confirms that referential and logical hallucinations require genuine human expertise to annotate reliably.

---

## 5. Methodology

### 5.1 LLM-as-a-Judge

The LLM-as-a-Judge serves as the automated baseline and the mechanism by which Trajel labels were initially generated at scale. The judge receives four inputs: the task question, the characteristic expected answer, the full Thought–Action–Observation trace, and the agent's final response. It returns a structured JSON verdict specifying hallucination presence, type(s), location, and rationale.

The prompt encodes the five-type taxonomy with concrete examples of what constitutes and does not constitute a hallucination (e.g., selecting an available but suboptimal tool is *not* a hallucination; calling a tool that does not exist *is*). The judge outputs binary labels only and thus does not produce calibrated probability scores, preventing direct AUC comparison against supervised classifiers.

### 5.2 Detection Modeling (RQ3)

The taxonomy established that hallucination types differ in the scope of context required for detection. This ordering motivates three complementary modeling paradigms, each operating at a different contextual granularity.

**Paradigm 1: Subtask-level Classification (BERT).** A fine-tuned BERT-based classifier operates on individual steps $s_t$, receiving the concatenation of $\tau_t$, $\alpha_t$, and $\omega_t$ as input and predicting whether the step contains a hallucination. This paradigm captures local cues — lexical anomalies, contradiction between thought and observation, tool-call malformations — without awareness of the broader trajectory. By the taxonomy's context ordering, this paradigm should be most effective for factual hallucinations and least effective for procedural and scope-based types.

**Paradigm 2: Trajectory-level NLI.** A natural language inference formulation treats hallucination detection as an entailment problem. For each step $s_t$, the trajectory history $\{s_1, \ldots, s_{t-1}\}$ serves as the premise and the current step's thought and action serve as the hypothesis. The model predicts whether the hypothesis is entailed by, neutral with respect to, or contradicted by the premise. This paradigm targets trace-wide consistency: referential hallucinations (claims about nonexistent prior outputs) and logical hallucinations (conclusions that do not follow from stated premises) should surface as contradiction or neutral judgments respectively.

**Paradigm 3: Long-context Modeling (Longformer).** A Longformer-based classifier ingests the full serialized trajectory $\mathcal{T}$ as a single input sequence. The Longformer's sparse attention mechanism enables processing of traces that exceed standard transformer context windows. This paradigm is designed for global context modeling: detecting procedural hallucinations (which require comparing the executed sequence against the expected workflow) and scope-based hallucinations (which require tracking agent identity across the full trace).

These three paradigms are complementary rather than competing: subtask-level classification offers efficiency at the cost of limited context; trajectory-level NLI provides pairwise consistency checks; long-context modeling captures global structure at greater computational cost.

### 5.3 Execution-Quality Signal Analysis (RQ4)

Beyond supervised classification, we investigate which execution-quality signals available during or immediately after agent execution are most predictive of hallucination. We operationalize five signal families from the AssetOpsBench evaluation framework:

- **Task Completion (TC):** Was the assigned task completed?
- **Data Retrieval Accuracy (DRA):** Were relevant data sources successfully retrieved?
- **Result Verification (RV):** Was the output verified against source observations?
- **Agent Sequence Correctness (ASC):** Did the agent invoke tools in the prescribed order?
- **Clarity and Justification (CJ):** Did the agent provide clear, grounded reasoning?

Each signal is a binary flag. If sufficiently predictive, these signals could support lightweight real-time monitors — guardrails integrated into the agent loop that flag or halt execution when hallucination risk exceeds a threshold.

---

## 6. Experiments and Results

### 6.1 Experimental Setup

All analyses are conducted on the 224 trajectories with complete human annotations. We report precision, recall, and F1 per hallucination type for the LLM judge, ROC-AUC for supervised classifiers, and Pearson correlation and estimated AUC for execution-quality signals. ROC-AUC is emphasized as the primary metric for its robustness under class imbalance.

### 6.2 Type Distribution (RQ1)

**Table 5: Hallucination type distribution (human evaluation).**

| Type | Count | % | Key Implication |
|---|---|---|---|
| Procedural ($h^P$) | 95 | 38.5% | Invisible to output-only eval |
| Factual ($h^F$) | 65 | 26.3% | Most detectable |
| Scope ($h^S$) | 49 | 19.8% | Unique to multi-agent settings |
| Logical ($h^L$) | 21 | 8.5% | Hardest to annotate |
| Referential ($h^R$) | 17 | 6.9% | Requires full trace context |
| **Total occurrences** | **247** | — | Multi-label: 48.7% of traces |

Procedural hallucinations dominate at 38.5%, a finding of significant practical importance: procedural failures — skipping required diagnostic steps, fabricating workflow completion, acting on nonexistent tool outputs — are **invisible to evaluation methods that check only factual accuracy**. This validates the need for trajectory-level evaluation and confirms that purely factual benchmarks systematically undercount hallucination in agentic systems.

Notably, 48.7% of hallucinated trajectories exhibit multiple types simultaneously. The most common co-occurring pairs are procedural–factual (n = 27) and procedural–scope (n = 26), indicating that control-flow violations frequently co-occur with other failure modes. This validates the multi-label formulation in our taxonomy design.

### 6.3 Cross-Model Analysis (RQ1)

**Table 6: Per-model hallucination rates (human evaluation).**

| Model | N | Hall. Rate | $h^F$ | $h^R$ | $h^L$ | $h^P$ | $h^S$ |
|---|---|---|---|---|---|---|---|
| Model 6 | 42 | 52.4% | 7 | 0 | 2 | 17 | 2 |
| Model 19 | 42 | 64.3% | 9 | 6 | 2 | 16 | 18 |
| Model 7 | 41 | 68.3% | 13 | 2 | 6 | 17 | 6 |
| Model 12 | 42 | 71.4% | 13 | 1 | 4 | 20 | 6 |
| Model 16 | 15 | 80.0% | 5 | 1 | 1 | 7 | 4 |
| Model 17 | 42 | 81.0% | 18 | 7 | 6 | 18 | 13 |

Hallucination rates range from 52.4% (Model 6) to 81.0% (Model 17) on identical tasks. Beyond aggregate rates, type profiles differ qualitatively: Model 19 exhibits a disproportionate share of scope-based hallucinations (18 of 27 hallucinated trajectories), while Model 6 shows almost none. These patterns are invisible under a binary label, reinforcing the diagnostic value of the five-type taxonomy.

### 6.4 Localization Analysis (RQ2)

**Table 7: Hallucination localization within trajectories.**

| Step Component | Count | | Task Position | Count |
|---|---| |---|---|
| Action | 70 | | Task 1 | 36 |
| Response | 58 | | Task 2 | 44 |
| Thought | 40 | | Task 3 | **63 ← peak** |
| Observation | 11 | | Task 4 | 43 |
| | | | Task 5+ | 51 (declining) |

Two patterns emerge. First, hallucinations most frequently localize to **Actions** (70) and **Responses** (58) — the externally visible components — rather than Thoughts (40) or Observations (11). The primary failure mode is not internal reasoning per se, but the translation of reasoning into tool invocations and output claims. This has direct implications for guardrail design: monitors targeting action validity and response verification will cover more than 75% of hallucination sites.

Second, hallucinations peak at Task 3 (63 occurrences) and decline monotonically thereafter. This mid-trajectory concentration is consistent with the hypothesis that hallucination risk increases with accumulated context but decreases as the remaining scope narrows.

### 6.5 Per-Type Detection Quality (RQ1, RQ3)

**Table 8: LLM-as-a-Judge per-type detection performance.**

| Type | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Procedural ($h^P$) | 78 | 26 | 17 | 0.750 | 0.821 | 0.784 |
| Factual ($h^F$) | 50 | 24 | 15 | 0.676 | 0.769 | 0.719 |
| Scope ($h^S$) | 32 | 9 | 16 | 0.780 | 0.667 | 0.719 |
| Logical ($h^L$) | 4 | 6 | 17 | 0.400 | 0.190 | 0.258 |
| Referential ($h^R$) | 3 | 7 | 14 | 0.300 | 0.176 | 0.222 |

The LLM judge's performance varies dramatically across types. It achieves F1 scores of 0.784 and 0.719 on procedural and factual hallucinations respectively, but only 0.258 and 0.222 on logical and referential types. This disparity is consistent with the context-ordering hypothesis: procedural and factual hallucinations have more overt surface cues, while logical and referential hallucinations require deeper cross-step reasoning.

The judge's failure mode also differs by type: for procedural hallucinations, the primary error is over-flagging (26 FP vs. 17 FN), while for referential and logical types, the dominant error is under-detection (14 and 17 FN respectively). The LLM judge is a reasonable first-pass filter for procedural and factual types, but human review remains essential for referential and logical ones.

### 6.6 Supervised Detection Models (RQ3)

**Table 9: Supervised detection model comparison on Trajel.**

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Majority-class baseline | 0.683 | 1.000 | 0.812 | 0.500 |
| LLM-as-a-Judge (zero-shot) | 0.797 | 0.922 | 0.855 | — |
| BERT (subtask) | 0.548 | 0.639 | 0.590 | 0.613 |
| NLI (trajectory-level) | 0.500 | 0.643 | 0.563 | **0.689** |
| Longformer (long-context) | 0.471 | 0.615 | 0.533 | 0.599 |

All three supervised models improve over their respective pre-trained baselines after fine-tuning, confirming that the Trajel dataset contains learnable signal. The NLI model achieves the highest ROC-AUC (0.689), consistent with the prediction that trajectory-level context improves ranking quality for types requiring cross-step reasoning.

However, all fine-tuned models fall short of the zero-shot LLM judge on F1 (0.855), indicating that task-specific supervised training on 225 trajectories is insufficient to match the general reasoning capability of large prompted models. Performance below 0.70 AUC across all paradigms confirms that **trajectory-level hallucination detection remains an unsolved challenge**.

### 6.7 Execution-Quality Signal Analysis (RQ4)

**Table 10: Execution-quality signals as univariate predictors of hallucination (n = 213).**

| Signal | Hall. rate (present) | Hall. rate (absent) | Pearson r | Est. AUC |
|---|---|---|---|---|
| Task Completion (TC) | 2.3% | 90.0% | −0.786 | 0.853 |
| Data Retrieval Accuracy (DRA) | 43.0% | 89.6% | −0.502 | 0.771 |
| Result Verification (RV) | 6.4% | 91.0% | −0.784 | 0.863 |
| Agent Sequence Correct (ASC) | 43.7% | 86.6% | −0.453 | 0.738 |
| **Clarity & Justification (CJ)** | **9.1%** | **94.3%** | **−0.833** | **0.908** |

All five signals are strongly negatively correlated with hallucination. The clarity-and-justification signal is the strongest univariate predictor (r = −0.833, AUC = 0.908): trajectories lacking clear, grounded reasoning hallucinate at a rate of 94.3%, versus only 9.1% for trajectories with high-clarity reasoning.

The CJ signal alone achieves AUC = 0.908, **substantially outperforming the best supervised classifier** (NLI, AUC = 0.689). When both CJ and RV signals are absent simultaneously, the hallucination rate reaches 97.1%, suggesting an effective kill-switch condition for agent orchestrators. This finding shifts the practical recommendation from post-hoc classification toward **lightweight runtime monitors integrated into the agent loop**.

---

## 7. Discussion

### 7.1 Why Procedural Hallucinations Dominate

The dominance of procedural hallucinations (38.5%) is the most practically significant finding in this work. Procedural failures — an agent claiming task completion without running the required diagnostic, or invoking a downstream tool before its dependency is satisfied — are structurally invisible to any evaluation method that checks only the final answer. A factual benchmark could in principle catch all factual errors; a procedural error requires comparing the executed trace against the prescribed workflow. This is a fundamentally different evaluation regime, and our taxonomy is designed precisely to surface it.

### 7.2 The Multi-Label Problem

Nearly half of all hallucinated trajectories (48.7%) carry multiple types simultaneously. The most common co-occurring patterns — procedural–factual and procedural–scope — reflect a structural logic: once an agent violates the prescribed workflow, it is also likely to fabricate outputs (factual) or act beyond its mandate (scope). A single-label evaluation would mischaracterize nearly half of real failures, reducing a structured diagnostic to a binary indicator that loses exactly the information needed to improve the system.

### 7.3 Execution Signals as Practical Guardrails

The signal analysis result — that a single binary flag achieves AUC = 0.908 — has a strong practical implication. Training BERT, NLI, or Longformer classifiers requires labeled data, fine-tuning compute, and ongoing maintenance. Execution-quality signals are already produced by the AssetOpsBench evaluation framework at inference time, require no training, and integrate naturally into the agent orchestration loop. The clarity-and-justification signal is the strongest: if an agent cannot justify its reasoning clearly and ground it in prior observations, hallucination is nearly certain (94.3% rate). This suggests a practical guardrail design: monitor CJ continuously and halt or re-route execution when it drops below threshold.

### 7.4 Where Automated Judges Fail

The per-type κ analysis reveals a clear gap: the LLM judge is reliable for procedural, scope, and factual types (κ ≥ 0.595) but nearly fails on logical (κ = 0.211) and referential (κ = 0.176) types. Logical hallucinations require the judge to track whether a conclusion actually follows from prior premises — a form of multi-hop reasoning that even large models struggle with in the annotation setting. Referential hallucinations require verifying that every entity referenced in a thought actually appeared earlier in the trace, a retrieval-and-verification task that is sensitive to context window positioning and attention drift. These are the blind spots of current automated evaluation, and they are precisely where human review remains irreplaceable.

---

## 8. Limitations and Future Work

**Domain scope.** Trajel comprises 225 trajectories from a single industrial domain (data center asset operations). The type distribution and detection difficulty may differ in other domains such as healthcare, finance, or open-ended web tasks.

**Model coverage.** The six model configurations share a common orchestration framework (AssetOpsBench). Hallucination behavior under alternative orchestrators or agent architectures remains untested.

**Annotation agreement on subtle types.** Per-type κ drops to 0.176 for referential and 0.211 for logical hallucinations, indicating that these categories require richer annotation protocols. Future protocols could require annotators to explicitly trace each referential claim back to a specific prior step before labeling.

**Dataset scale.** All supervised detection models are benchmarked on 225 labeled trajectories — a small dataset for fine-tuning. The gap between supervised classifier AUC (0.689) and LLM-judge F1 (0.855) motivates scaling the dataset and exploring hybrid architectures that combine supervised grounding with LLM-derived features.

**Future directions** include scaling to new domains, richer annotation protocols for referential and logical types, hybrid classifier architectures, and extension to multi-model ensembles where inter-agent disagreement can be measured directly rather than proxied by execution flags.

---

## 9. Conclusion

We introduced **Trajel**, a trajectory-aware benchmark and dataset for hallucination detection in multi-agent industrial workflows. Our five-type taxonomy formalizes hallucination as a structural predicate over the Thought–Action–Observation execution trace, enabling diagnostic analysis that binary labels cannot provide. Empirical evaluation on 225 expert-annotated trajectories reveals three principal findings:

1. **Procedural hallucinations are the dominant failure mode** (38.5%), invisible to output-only evaluation, validating the need for trajectory-level benchmarks.
2. **Nearly half of hallucinated trajectories (48.7%) exhibit multiple types simultaneously**, confirming that single-label formulations mischaracterize a large share of real failures.
3. **Automated detectors with high binary accuracy still systematically misclassify the subtlest types**: per-type κ drops to 0.176 for referential and 0.211 for logical hallucinations.

Signal analysis reveals that the clarity-and-justification execution flag alone achieves AUC = 0.908, substantially outperforming all trained classifiers. This shifts the practical recommendation toward lightweight runtime monitors integrated into the agent orchestration loop — a path that is both more effective and more deployable than post-hoc classification.

We believe Trajel will serve as a foundation for future work on transparent, auditable, and safer multi-agent deployment in high-stakes industrial settings.

---

## References

[1] Patel, D., Lin, S., Rayfield, J., Zhou, N., Vaculin, R., Martinez, N., O'donncha, F., Kalagnanam, J. (2025). AssetOpsBench: Benchmarking AI Agents for Task Automation in Industrial Asset Operations and Maintenance. *arXiv:2506.03828*.

[2] Yao, S., et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *arXiv:2210.03629*.

[3] Liu, X., et al. (2023). AgentBench: Evaluating LLMs as Agents. *arXiv:2308.03688*.

[4] Lin, S., et al. (2022). TruthfulQA: Measuring How Models Mimic Human Falsehoods. *ACL 2022*.

[5] Wu, et al. (2024). ToolBH: A Multi-level Hallucination Diagnostic Benchmark for Tool-Augmented LLMs. *EMNLP 2024*.

[6] Zhang, W., et al. (2025). MIRAGE-Bench: LLM Agent is Hallucinating and Where to Find Them. *arXiv:2507.21017*.

[7] Ye, H., et al. (2023). Cognitive Mirage: A Review of Hallucinations in Large Language Models. *arXiv:2309.06794*.

[8] Dong, B., et al. (2025). MIRAGE: Assessing Hallucination in Multimodal Reasoning Chains of MLLMs. *arXiv:2505.24238*.

[9] Deng, X., et al. (2024). WebArena: A Realistic Web Environment for Building Autonomous Agents. *arXiv:2307.13854*.

[10] Yang, Z., et al. (2018). HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. *EMNLP 2018*.

[11] Cemri, M., et al. (2025). Why Do Multi-Agent LLM Systems Fail? *NeurIPS 2025, Datasets and Benchmarks Track*.

[12] He, P., et al. (2026). TRAJECT-Bench: A Trajectory-Aware Benchmark for Evaluating Agentic Tool Use. *ICLR 2026*.

[13] Darwish, A.M., et al. (2025). Mitigating LLM Hallucinations Using a Multi-Agent Framework. *Information, 16(7):517*.

[14] Agrawal, L.A., Tan, S., Soylu, D., et al. (2026). GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning. *ICLR 2026, Oral. arXiv:2507.19457*.
