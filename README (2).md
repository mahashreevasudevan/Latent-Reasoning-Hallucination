# Right Answer, Wrong Reason: Latent Reasoning Hallucinations in LLMs

> Presented at the **BCS Lovelace Colloquium 2026**, University of Birmingham

We all know LLMs can give wrong answers. This project asks a harder question: what happens when a model gives the *right* answer for the *wrong* reason?

This research investigates **latent reasoning hallucinations** — cases where a model's final answer is correct but the chain-of-thought reasoning used to justify it is flawed, fabricated, or logically disconnected. These failures are invisible to standard accuracy benchmarks, which makes them particularly dangerous in high-stakes settings like medicine and law.

---

## Key Objectives

- Find out how often latent reasoning hallucinations occur across open-source models of varying capability
- Identify which question types and cognitive properties make them more likely
- Understand whether reasoning quality degrades predictably with model size
- Build a structural framework for predicting which questions are vulnerable
- Measure reasoning stability — whether correct reasoning paths hold across prompt variants

---

## Methodology

Six open-source models were tested across three datasets using standard chain-of-thought prompting. Every response was manually labelled by two independent raters using a three-level schema.

**Datasets**

| Dataset | Questions | Type | Difficulty |
|---|---|---|---|
| GSM8K | 30 | Arithmetic word problems | Easy / Medium / Hard |
| AQuA-RAT | 30 | Algebraic multiple-choice | Easy / Medium / Hard |
| Custom Cognitive-Trap | 150 | Misleading framing, causal traps, conditional dependencies | Hard |

The custom dataset was hand-built specifically to satisfy both conditions for latent hallucinations — questions with retrievable answers that still demand non-trivial reasoning to justify.

**Labelling Schema**

Each response was classified as one of three categories:

- **Robust** — correct answer, sound reasoning throughout
- **Shallow-but-correct** — correct answer but compressed or under-justified reasoning
- **Flawed** — incorrect or fabricated reasoning steps

A **latent hallucination** is any response labelled *Flawed* where the final answer is nonetheless correct.

Inter-rater reliability: Cohen's κ = 0.33–0.37, reflecting the genuine difficulty of distinguishing shallow from flawed reasoning. Labels are treated as an analytical lens, not a ground truth.

---

## Model Pipeline

**Models tested**

| Model | Size | Tier |
|---|---|---|
| Llama 3.1 8B | 8B | Weak |
| GPT-OSS 20B | 20B | Mid |
| Llama 4 Scout 17B | 17B | Mid |
| Qwen3 32B | 32B | Mid-strong |
| Llama 3.3 70B | 70B | Strong |
| GPT-OSS 120B | 120B | Very strong |

All models were accessed via the **Groq API** using a standard chain-of-thought prompt. Data collection scripts are in the `/code` directory. Results were saved to CSV and labelled manually before analysis.

**Pipeline flow**

```
Datasets → Models → CoT Prompt → 540 Responses → Manual Labelling → Analysis
```

The stability analysis added a further 18 repeated and rephrased questions to test whether reasoning paths are consistent across prompt variants.

---

## Challenges Addressed

**1. Accuracy is blind to reasoning quality**
Standard benchmarks like GSM8K and AQuA-RAT only check whether the final answer is correct. A model can score 100% while producing entirely post-hoc reasoning. This project addresses that gap directly.

**2. No framework for predicting vulnerability**
Prior work showed that chain-of-thought can be unfaithful. Nobody had mapped *which* question types trigger latent hallucinations or *why*. This project proposes the first predictive framework.

**3. Manual labelling at scale is hard**
Reasoning quality judgements are subjective. The moderate inter-rater agreement (κ = 0.33–0.37) reflects this honestly and is reported transparently rather than smoothed over.

**4. Reasoning instability is hard to measure**
Most evaluations run each question once. The stability analysis here systematically varies prompt phrasing to measure whether correct reasoning is consistent or coincidental.

---

## Results

**Latent hallucinations are rare and model-specific**

Across 540 total responses, 6 latent hallucinations were found. Every single one was in Llama 3.1 8B, on the custom cognitive-trap dataset. Zero latent hallucinations appeared in any other model or on either benchmark dataset.

**Question structure predicts failure type**

All 6 instances share the same profile: reasoning load 2-4 out of 5, retrievable answers, and a specific cognitive vulnerability. The model retrieves the correct answer from training and then constructs reasoning to justify it after the fact. Four vulnerability types were identified:

| Vulnerability Type | Description |
|---|---|
| Surface Mismatch | Question surface implies the wrong answer |
| Procedural Trap | A familiar method is applied unnecessarily |
| Causal Trap | The obvious causal direction must be rejected |
| Conditional Dependency | The answer requires resolving a self-referential condition |

**Reasoning is fundamentally unstable**

Across 18 repeated and rephrased questions, not one produced a consistent reasoning path, even when the final answer stayed the same. The reasoning is post-hoc, not principled.

**Reasoning quality follows capability**

AQuA-RAT results show a clear gradient: weak models produce flawed reasoning, mid-tier models produce a mix, and strong models produce shallow-but-correct responses. Qwen3 32B is an exception — its explicit think blocks expose failures that hidden chain-of-thought would mask.

**The two-condition framework**

Latent hallucinations need two conditions at once:

1. The correct answer is retrievable without genuine reasoning
2. Non-trivial reasoning is still required to justify the answer

GSM8K fails condition 1 — answers must be computed. AQuA-RAT fails condition 1 — algebraic errors produce wrong answers. Only the custom dataset satisfies both, which explains exactly why latent hallucinations appeared there and nowhere else.

---

## Impact

This work has three practical implications.

**For evaluation:** Accuracy alone can't tell you whether a model is reasoning faithfully. Evaluation frameworks need to assess reasoning quality independently of answer correctness. This is most urgent for small models deployed in resource-constrained environments, where latent hallucinations are most likely.

**For deployment:** The two-condition framework gives practitioners a principled way to identify which question types carry higher hallucination risk before deploying a model on them.

**For trust:** A model that says the right thing for the wrong reason teaches users the wrong lesson. In medicine, law, and scientific assistance, the reasoning is the decision. Getting it wrong — silently — is worse than getting it wrong visibly.

---

## Technology and Tools

**Languages and libraries**

- Python 3.12
- `pandas` — data handling and CSV processing
- `matplotlib`, `numpy` — chart and diagram generation
- `groq` — API access for all model inference
- `datasets` — loading GSM8K and AQuA-RAT from HuggingFace

**APIs and models**

- [Groq API](https://console.groq.com) — fast inference for all 6 models
- [Cerebras API](https://cloud.cerebras.ai) — used for Qwen3-235B collection script

**Datasets**

- [GSM8K](https://huggingface.co/datasets/openai/gsm8k) — grade school math word problems
- [AQuA-RAT](https://huggingface.co/datasets/deepmind/aqua_rat) — algebraic multiple-choice with rationales
- Custom cognitive-trap dataset — hand-built, included in this repository

**Poster and visualisation**

- PowerPoint — poster design (A1 portrait)
- `matplotlib` — all charts and diagrams
- Presented at BCS Lovelace Colloquium 2026

---

## Repository Structure

```
├── data/
│   ├── gsm8k_sample.csv
│   ├── aqua_sample.csv
│   ├── results_20260309_095314.csv
│   ├── results_hard_20260309_201712.csv
│   ├── results_benchmarks_20260311_105709.csv
│   ├── results_gptoss_20260310_211744.csv
│   ├── results_qwen3_20260310_153432.csv
│   ├── results_small_models_20260310_213412.csv
│   └── stability_20260310_102000.csv
├── labelled/
│   ├── results_benchmarks_labelled.csv
│   ├── results_hard_labelled.csv
│   ├── results_gptoss_labelled.csv
│   ├── results_qwen3_labelled.csv
│   └── results_small_models_labelled.csv
├── code/
│   ├── sample_datasets.py
│   ├── collect_benchmarks.py
│   ├── collect_gptoss.py
│   ├── collect_small_models.py
│   └── collect_qwen3_235b.py
├── visualisation/
│   ├── poster_charts_plain.py
│   ├── poster_charts_from_data.py
│   └── poster_diagrams.py
└── README.md
```

---

## References

1. Wei et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS 2022.*
2. Turpin et al. (2023). Language Models Don't Always Say What They Think. *NeurIPS 2023.*
3. Arcuschin et al. (2025). CoT Reasoning In The Wild Is Not Always Faithful. *ArXiv.*
4. Li et al. (2024). Deceptive Semantic Shortcuts on Reasoning Chains. *NAACL 2024.*
