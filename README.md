# Trajectory-Level Hallucination Detection in Multi-Agent Industrial Workflows

**STATGR5293 — Generative AI Using LLMs — Final Project**

---

## Overview

This project extends [AssetOpsBench](https://github.com/IBM/AssetOpsBench) (IBM Research), a benchmark for AI-driven failure diagnosis of industrial assets (chillers, pumps, compressors). Given a sensor reading or anomaly report, AssetOpsBench dispatches tasks across four FastMCP tool servers — IoT, FSMR, TSFM, and WO — using a Plan-Execute orchestrator to identify root causes, predict failure modes, and generate maintenance work orders automatically.

Each scenario produces a sequential **Thought → Action → Observation** trace across multiple agent calls. This project focuses on detecting when that trace goes wrong.

---

## The Problem

Standard hallucination benchmarks treat every task as an isolated input → output pair. In a multi-step agent loop, this misses the real failure modes:

- **Mid-trace failures**: a hallucinated action at step *t* corrupts everything downstream — the final answer can look correct while the reasoning was broken three steps earlier
- **Type blindness**: the most common failure mode (procedural hallucination) is completely invisible to factual checks; most hallucinated traces carry more than one failure type simultaneously
- **Detector gap**: existing automated judges catch obvious hallucinations but systematically miss the subtler types — high overall accuracy masks critical blind spots

---

## This Project: Trajel Dataset

We introduce **Trajel**, a framework and dataset for trajectory-level hallucination detection built on top of AssetOpsBench. The core contributions are:

### 1. Five-Type Hallucination Taxonomy

| Type | Definition |
|---|---|
| **Factual** | Asserts a claim contradicted by ground-truth data |
| **Referential** | References an entity or prior result absent from the trace |
| **Logical** | Reasoning doesn't follow from its premises |
| **Procedural** | Skips, reorders, or fabricates a required workflow step |
| **Scope** | Agent acts outside its assigned mandate |

### 2. Trajel Dataset (`data/trajel_dataset.csv`)

225 expert-annotated agent trajectories:
- 6 LLM model configurations
- 42 industrial AssetOps task questions
- Human annotations from Columbia University and IBM Research
- Each entry: hallucination presence, type(s), localization, free-text rationale, and LLM-judge vs. human agreement

Human-identified hallucination rate: **68.3%**. Nearly half of hallucinated trajectories (48.7%) carry multiple types simultaneously.

### 3. ML Detection Models (`hallucination_detection/classifiers.ipynb`)

Three complementary detection paradigms:

- **BERT (subtask-level)**: classifies individual Thought–Action–Observation steps
- **NLI (trajectory-level)**: treats trajectory history as premise and current step as hypothesis
- **Longformer (long-context)**: ingests the full serialized trajectory for global structural detection

### 4. LLM-as-a-Judge Prompt (`hallucination_detection/llm_judge_prompt.py`)

A structured evaluation prompt that classifies hallucination presence, type, location, and rationale for any agent trajectory. Used as the automated baseline in the Trajel evaluation pipeline.

### 5. Data Analysis (`hallucination_detection/analysis.ipynb`)

Exploratory analysis of the Trajel dataset: type distributions, per-model hallucination rates, localization patterns, and execution-quality signal correlations.

---

## Key Results

| Finding | Value |
|---|---|
| Human hallucination rate | 68.3% |
| Most common type | Procedural (38.5%) |
| Multi-type trajectories | 48.7% |
| LLM-judge binary recall | 92.2% |
| LLM-judge F1 on logical type | 0.258 |
| Best classifier AUC (NLI) | 0.689 |
| Best single signal AUC (Clarity & Justification) | **0.908** |

The clarity-and-justification execution signal alone outperforms every trained classifier — suggesting lightweight runtime monitors are the most practical path to real-time hallucination detection.

---

## Repository Structure

```
genai-using-llms-final-project/
├── data/
│   └── trajel_dataset.csv                  # Expert-annotated trajectory dataset
├── hallucination_detection/
│   ├── classifiers.ipynb                   # BERT / NLI / Longformer detection models
│   ├── analysis.ipynb                      # Dataset EDA and signal analysis
│   └── llm_judge_prompt.py                 # LLM-as-a-Judge evaluation prompt
├── assetopsbench/
│   └── src/                                # AssetOpsBench agent source (IBM Research)
│       ├── workflow/                        # Plan-Execute orchestrator
│       ├── llm/                             # LiteLLM backend
│       └── servers/                         # IoT, FMSR, TSFM, Utilities MCP servers
└── performance_optimization/
    ├── caching/                             # Placeholder resolver optimization
    ├── parallelization/                     # Async N×M FMSR parallelization
    └── quantization/
        ├── code/                            # Benchmarking scripts
        ├── charts/                          # Per-model precision vs. accuracy/latency charts
        └── analysis/                        # Per-model quantization analysis
```

---

## Built On

- [AssetOpsBench](https://github.com/IBM/AssetOpsBench) — IBM Research industrial multi-agent benchmark
- Patel et al. (2025). *AssetOpsBench: Benchmarking AI Agents for Task Automation in Industrial Asset Operations and Maintenance.* arXiv:2506.03828

---

## Course

**STATGR5293 — Section 007 (Wednesday)**
Generative AI Using LLMs — Columbia University, Spring 2026

**Instructor:** Parijat Dube
**Project PI:** Dhaval C. Patel (IBM Research)
