# Research Paper Analysis Pipeline

Classifies LLM research papers from PDF — extracts key sections and uses a local LLM (Ollama) to classify each paper as **LLM Efficiency**, **LLM Scaling**, or **Other**.

---

## Project Structure

```
├── main.py                   # Entry point — run this to process all papers
├── requirements.txt          # Python dependencies
├── .gitignore                # Excludes generated files from git
├── pipeline/
│   ├── pdf_extractor.py      # Extracts title, abstract, introduction, conclusion from PDFs
│   ├── classifier.py         # Calls Ollama, runs two-stage LLM classification, saves results
│   └── prompts.py            # LLM prompts and response parsers
├── papers/
│   ├── pdf/                  # ← Place your input PDF files here
│   └── clean_text/           # Auto-generated: extracted text per paper (for inspection)
├── output/                   # Auto-generated: per-paper JSON with chunks + classification
└── results/
    ├── results.csv           # Final results — open this after each run
    └── results.json          # Detailed results with reasoning and confidence
```

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Configure Ollama**

Update these two lines in `pipeline/classifier.py` to point to your Ollama server:
```python
OLLAMA_URL   = "http://<your-host>:<port>/api/generate"
OLLAMA_MODEL = "gpt-oss:120b"
```

**3. Add PDFs**

Place your research paper PDFs inside `papers/pdf/`.

---

## Running

```bash
python main.py
```

Already classified papers are **skipped automatically** — only new or failed papers are processed on each run.

**What happens:**
1. Scans `papers/pdf/` for PDF files
2. Skips papers that are already successfully classified
3. Extracts Title, Abstract, Introduction, Conclusion from each PDF
4. Saves extracted text to `papers/clean_text/<paper_id>.txt`
5. Saves structured chunks to `output/<paper_id>.json`
6. Runs two-stage LLM classification on each paper
7. Writes final results to `results/results.csv` and `results/results.json`

---

## Output

### results.csv

| Column | Description |
|---|---|
| `paper_id` | ArXiv ID or filename — use this to look up the paper |
| `title` | Paper title extracted from the PDF |
| `classification` | Category assigned by the LLM |
| `accuracy` | LLM confidence score (0–100%) |
| `justification` | 2–3 sentence explanation of why the paper was classified this way |

### Classification

Papers are classified using a **two-stage** approach:

**Stage 1** — Quick screen:
- `A` = LLM Efficiency
- `B` = LLM Scaling
- `C` = Other

**Stage 2A** (for A/B papers) — Confirms Efficiency vs Scaling with deep reasoning.

**Stage 2B** (for C papers) — Assigns a short label: `Other — Model Architecture`, `Other — Training & Alignment`, etc.

| Category | What it means |
|---|---|
| `LLM Efficiency` | Makes LLMs cheaper, faster, or smaller (quantization, pruning, LoRA, distillation, efficient attention) |
| `LLM Scaling` | Makes LLMs more capable through scale (scaling laws, larger models, emergent abilities) |
| `Other — ...` | Neither efficiency nor scaling — label describes what it actually is |

Papers below **75% confidence** are flagged with `⚑ NEEDS REVIEW` in the terminal.

---

## Pipeline Modules

### `pipeline/pdf_extractor.py`
- Opens PDFs with **PyMuPDF**
- Extracts title using font-size detection (falls back to PDF metadata)
- Extracts Abstract, Introduction, Conclusion using regex with multiple pattern variants and position-based fallbacks
- Warns in the terminal if any section could not be found

### `pipeline/classifier.py`
- Calls Ollama with **temperature=0** (deterministic — same paper always gets the same result)
- Retries up to **3 times** on failure with a 5s delay
- Papers that fail all retries are marked `ERROR` and retried on the next run

### `pipeline/prompts.py`
- Defines the three LLM prompts: `STAGE1_PROMPT`, `STAGE2A_PROMPT`, `STAGE2B_PROMPT`
- Parses LLM responses using regex to handle formatting variations

---

## Requirements

- Python 3.10+
- A running [Ollama](https://ollama.com) instance accessible from your machine
- PDF files placed in `papers/pdf/`
