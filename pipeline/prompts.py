import re

# Papers with confidence below this value are flagged with ⚑ in the terminal.
# This helps you spot papers that need a manual check before handing
# results to the professor. Does not affect the CSV — only the terminal output.
REVIEW_THRESHOLD = 75


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Fast screen: Efficiency vs Scaling vs Other
# Goal: route the paper to the right Stage 2 prompt.
# Expected output: 2 lines only.
# ══════════════════════════════════════════════════════════════════════════════
STAGE1_PROMPT = """You are screening an LLM research paper for a professor whose research question is:
  "Are recent papers making LLMs more EFFICIENT, or more CAPABLE through SCALE?"

Classify this paper into exactly one of:

A - LLM Efficiency  : Primary goal is making LLMs cheaper, faster, or smaller to run.
                      (quantization, pruning, LoRA/PEFT, distillation, efficient attention,
                       sparse models, memory reduction, faster inference)

B - LLM Scaling     : Primary goal is making LLMs more capable by scaling up.
                      (scaling laws, larger models, emergent abilities, compute-optimal
                       training, studying performance vs size/data/compute)

C - Other           : Primary contribution is clearly neither efficiency nor scaling.
                      (new architectures as standalone contributions, alignment/RLHF,
                       benchmarks, multimodal, domain applications)

IMPORTANT: If the paper improves efficiency IN ORDER TO scale larger → B (Scaling).
           If the paper reduces cost of running existing model sizes → A (Efficiency).

---
EXAMPLES
---
Title: QLoRA: Efficient Finetuning of Quantized LLMs
Stage1: A
Reasoning: Combines 4-bit quantization with LoRA to make fine-tuning 65B models feasible on one GPU — goal is cheaper fine-tuning of existing model sizes.

Title: Training Compute-Optimal Large Language Models
Stage1: B
Reasoning: Studies how to allocate compute between model size and data to maximize capability at scale — a scaling study about getting more capable models.

Title: MMLU: Massive Multitask Language Understanding
Stage1: C
Reasoning: Proposes a benchmark to evaluate LLM knowledge across 57 subjects — evaluation contribution, not efficiency or scaling.

Title: LLaMA: Open and Efficient Foundation Language Models
Stage1: B
Reasoning: Trains 7B-65B foundation models and shows smaller models trained longer match larger ones — primarily about compute-optimal scaling, not reducing inference cost.

Title: SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot
Stage1: A
Reasoning: One-shot pruning algorithm that compresses 175B models to 50% sparsity with no retraining — goal is cheaper inference of existing large models.

---
Reply in this EXACT format — no extra text:
Stage1: <A or B or C>
Reasoning: <one sentence — what the paper does and why it maps to A, B, or C>"""


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2A — Deep classification for Efficiency / Scaling papers
# Only runs when Stage 1 returned A or B.
# Focused entirely on the Efficiency vs Scaling distinction.
# ══════════════════════════════════════════════════════════════════════════════
STAGE2A_PROMPT = """You are helping a professor classify an LLM research paper.
The professor's research question: "Is this paper making LLMs more EFFICIENT or more CAPABLE through SCALE?"

Your job: confirm or correct the initial screen, then provide a confident classification.

---
TWO CATEGORIES
---
LLM Efficiency
  The paper's PRIMARY goal is reducing the cost of running LLMs — less memory, faster inference,
  fewer FLOPs — while preserving quality at existing model sizes.
  Techniques: quantization, pruning, distillation, LoRA/PEFT, sparse activation,
              efficient attention (FlashAttention), early exit, gradient checkpointing.

LLM Scaling
  The paper's PRIMARY goal is understanding or pushing LLM capability through scale —
  training larger models, studying how performance grows, finding compute-optimal strategies.
  Techniques: scaling laws, emergent ability studies, compute-optimal training,
              large model training (GPT-3, PaLM, Chinchilla), data scaling.

---
HOW TO DECIDE WHEN IT IS AMBIGUOUS
---
Ask: What problem is this paper written to solve?
  → "Existing models are too expensive to run"           → Efficiency
  → "We want more capable models"                        → Scaling
  → "We want to train larger models without more money"  → Scaling (goal is capability)
  → "We want the same model to run on cheaper hardware"  → Efficiency (goal is cost)

Specific hard cases:
  - Compute-optimal training (e.g. Chinchilla) → Scaling. The goal is better models, not cheaper inference.
  - LoRA, QLoRA, PEFT methods → Efficiency. The goal is cheaper fine-tuning of existing sizes.
  - Mixture of Experts — depends on framing:
      "Sparse activation reduces FLOPs per token" → Efficiency
      "MoE lets us train effectively larger models" → Scaling
  - Knowledge distillation → Efficiency. Transferring capability to smaller models = cheaper.
  - Data-efficient training → Scaling. Better use of data to reach better capability.

---
CONFIDENCE RUBRIC
---
90-100 : Clearly one category. No meaningful argument for the other exists.
75-89  : One category is clearly stronger, but a reasonable person could argue the other.
60-74  : Genuine ambiguity. Both have real merit. Pick the stronger one and explain why.
Below 60: Too uncertain — pick your best guess, lower confidence, flag for review.

---
FEW-SHOT EXAMPLES
---
Example 1 — Clear Efficiency
Title: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
Screening note: Initial screen said Efficiency.
Reasoning: FlashAttention rewrites the attention kernel to be IO-aware, reducing memory reads and speeding up computation without changing the mathematical result. The goal is to make existing Transformer models faster and cheaper — not to scale to larger models. Scaling is ruled out because the paper does not study how performance changes with model size.
Category: LLM Efficiency
Confidence: 97
Justification: FlashAttention's contribution is an IO-aware attention algorithm that reduces memory usage and wall-clock time for exact self-attention. The goal is making existing model sizes cheaper to train and run, not enabling larger models. It is not a scaling paper — no scaling laws, no emergent abilities, no study of capability vs compute. Directly relevant to the professor's efficiency question.

Example 2 — Clear Scaling
Title: Scaling Laws for Neural Language Models
Screening note: Initial screen said Scaling.
Reasoning: The entire paper studies power-law relationships between model size, data, compute, and loss. There is no efficiency technique proposed. The goal is understanding how to get more capable models through scale, not how to run existing models cheaper.
Category: LLM Scaling
Confidence: 98
Justification: This paper derives empirical scaling laws showing how language model loss improves predictably with model size, dataset size, and compute budget. It answers the question "how much capability do we get per FLOP at scale?" — a core scaling question. No efficiency technique is introduced. Directly central to the professor's scaling question.

Example 3 — Ambiguous, Efficiency wins
Title: LoRA: Low-Rank Adaptation of Large Language Models
Screening note: Initial screen said Efficiency.
Reasoning: LoRA reduces fine-tuning memory by training only low-rank matrices, making 175B fine-tuning accessible on commodity hardware. One could argue Training & Alignment since it is a fine-tuning method, but the primary motivation is cost reduction for existing model sizes, not improving model capability.
Category: LLM Efficiency
Confidence: 84
Justification: LoRA's core contribution is making fine-tuning of large models computationally accessible by reducing trainable parameters via low-rank decomposition. The paper's stated goal is cost reduction — not making models more capable. One could argue this is a training method, but the efficiency framing is primary and the professor's efficiency question is directly answered.

Example 4 — Ambiguous, Scaling wins
Title: LLaMA: Open and Efficient Foundation Language Models
Screening note: Initial screen said Scaling.
Reasoning: LLaMA's key finding is that smaller models trained significantly longer on more tokens match larger models — a compute-optimal scaling study. The efficiency framing ("open and efficient") refers to accessibility, not reducing inference cost. The core contribution is a training and scaling strategy.
Category: LLM Scaling
Confidence: 72
Justification: LLaMA's primary contribution is demonstrating compute-optimal training — that 7B-65B models trained on more tokens can match much larger models trained on fewer tokens. This is a scaling finding about how to allocate compute for maximum capability. The efficiency framing is secondary; the paper's core question is about training strategy for capable foundation models. Genuine ambiguity warrants lower confidence.

---
NOW CLASSIFY THE PAPER ABOVE
---
Work through this:
  1. What specific problem is this paper solving?
  2. Is the goal cheaper/smaller LLMs, or more capable/larger LLMs?
  3. What is the strongest argument for the other category, and why does it lose?

Reply in this EXACT format — no extra text, no markdown:
Reasoning: <2-3 sentences — what the paper does, which category wins and why, what argument for the other category you considered and rejected>
Category: <LLM Efficiency or LLM Scaling>
Confidence: <0-100>
Justification: <3 sentences — (1) what the paper contributes specifically, (2) why this category and not the other, (3) how it relates to the professor's research question>"""


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2B — Brief labelling for Other papers
# Only runs when Stage 1 returned C.
# These papers are outside the professor's primary interest.
# Goal: give a short label and one-line explanation — nothing more.
# ══════════════════════════════════════════════════════════════════════════════
STAGE2B_PROMPT = """You are helping a professor who is studying LLM efficiency and scaling research.
This paper has been screened and is NOT about efficiency or scaling.

Your only job: write a short label describing what the paper IS about, and one sentence explaining it.

The label should follow this format:
  Other — <brief topic>

Examples of good labels:
  Other — Model Architecture
  Other — Training & Alignment
  Other — Benchmark / Evaluation
  Other — Multimodal
  Other — Domain Application

---
EXAMPLES
---
Title: Training language models to follow instructions with human feedback
Reasoning: The paper proposes RLHF to align model outputs with human preferences — a training methodology, not efficiency or scaling.
Category: Other — Training & Alignment
Confidence: 96
Justification: This paper introduces RLHF as a training technique to align LLM outputs with human intent. It is outside the professor's efficiency/scaling scope.

Title: Attention Is All You Need
Reasoning: The paper proposes the Transformer — a new sequence model architecture built on self-attention. The architectural design is the contribution, not efficiency or scaling.
Category: Other — Model Architecture
Confidence: 98
Justification: This paper introduces the Transformer architecture, replacing recurrent models with pure self-attention. It is outside the professor's efficiency/scaling scope.

---
Reply in this EXACT format — no extra text, no markdown:
Reasoning: <1-2 sentences — what the paper does and why it is not efficiency or scaling>
Category: Other — <brief topic>
Confidence: <0-100>
Justification: <1-2 sentences — what the paper contributes and note it is outside the professor's scope>"""


# ── Parsers ───────────────────────────────────────────────────────────────────
def parse_stage1(raw: str) -> dict:
    """Extract the Stage 1 result (A, B, or C) from the LLM response.
    Defaults to C (Other) if parsing fails, so Stage 2 still runs safely.
    """
    result = {"primary": "C", "reasoning": ""}

    match = re.search(r'(?i)stage1\s*:\s*([ABC])', raw)
    if match:
        result["primary"] = match.group(1).upper()

    reason = re.search(r'(?i)reasoning\s*:\s*(.+)', raw)
    if reason:
        result["reasoning"] = reason.group(1).strip()

    return result


def parse_classification(raw: str) -> dict:
    """Extract Reasoning, Category, Confidence, and Justification from the LLM response.

    Uses regex instead of simple string splitting so it handles formatting
    variations the LLM might produce (bold markers, extra spaces, etc.).
    Both Stage 2A and 2B use the same output format, so one parser handles both.
    """
    result = {
        "reasoning"    : "",
        "category"     : "Unknown",
        "confidence"   : "0",
        "justification": "",
    }

    reason_match = re.search(
        r'(?i)\*{0,2}reasoning\*{0,2}\s*:\s*(.+?)(?=\n\s*\*{0,2}category\*{0,2}\s*:)',
        raw, re.DOTALL
    )
    if reason_match:
        result["reasoning"] = reason_match.group(1).strip()

    cat_match = re.search(r'(?i)\*{0,2}category\*{0,2}\s*:\s*(.+)', raw)
    if cat_match:
        result["category"] = cat_match.group(1).strip()

    conf_match = re.search(r'(?i)\*{0,2}confidence\*{0,2}\s*:\s*(\d+)', raw)
    if conf_match:
        result["confidence"] = conf_match.group(1).strip()
    else:
        # Fallback: handle formats like "85%" or "85 / 100"
        fallback = re.search(r'(?i)confidence.*?(\d{1,3})\s*(?:%|\/\s*100)?', raw)
        if fallback:
            result["confidence"] = fallback.group(1).strip()

    just_match = re.search(r'(?i)\*{0,2}justification\*{0,2}\s*:\s*(.+)', raw, re.DOTALL)
    if just_match:
        justification = just_match.group(1).strip()
        # Trim off anything that looks like a new field starting after the justification
        justification = re.split(
            r'\n\s*\*{0,2}(?:reasoning|category|confidence)\*{0,2}\s*:',
            justification, flags=re.IGNORECASE
        )[0]
        result["justification"] = justification.strip()

    return result
