# Research Paper Analysis Pipeline

A pipeline that reads LLM research papers (PDF) and automatically classifies them using an AI model.

> **What does it do?**
> You give it a folder of research papers. It reads each paper, understands what it's about, and tells you whether the paper is focused on making AI models more **efficient** (cheaper/faster) or more **capable** through **scaling** (bigger/smarter).

---

## How It Works

```
PDF Paper
    ↓
1. pdf_extractor.py   → Reads the PDF and pulls out the key sections
                        (Title, Abstract, Introduction, Conclusion)
    ↓
2. classifier.py      → Sends those sections to an AI model (LLM)
                        and runs a two-step classification
    ↓
3. prompts.py         → Contains the instructions given to the AI
                        and reads back its answers
    ↓
Result → Saved to results/results.csv and results/results.json
```

---

## Project Structure

```
├── main.py                   # Run this to start the pipeline
├── requirements.txt          # Python packages needed
├── .gitignore                # Files excluded from git
├── pipeline/
│   ├── pdf_extractor.py      # Step 1 - reads and extracts text from PDFs
│   ├── classifier.py         # Step 2 - sends text to AI and classifies
│   └── prompts.py            # Step 3 - AI prompts and response readers
├── papers/
│   └── pdf/                  # ← Put your PDF files here
├── output/                   # Auto-generated: one JSON file per paper
└── results/
    ├── results.csv           # Final results table
    └── results.json          # Detailed results with reasoning
```

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Configure the AI server**

Open `pipeline/classifier.py` and update these two lines to point to your server:
```python
OLLAMA_URL   = "http://<your-host>:<port>/api/generate"
OLLAMA_MODEL = "gpt-oss:120b"
```

> If you don't know these values, ask your professor or supervisor for the server address.

**3. Add your PDF files**

Copy your research paper PDFs into the `papers/pdf/` folder.

---

## Running

```bash
python main.py
```

- Every paper is **re-classified on every run**
- Failed papers are marked as `ERROR` in the output

---

## Understanding the Results

Open `results/results.csv` after the run. You'll see:

| Column | What it means |
|---|---|
| `paper_id` | The paper's filename or ArXiv ID |
| `title` | Title of the paper |
| `classification` | What category the AI assigned |
| `accuracy` | How confident the AI is (0–100%) |
| `justification` | Why the AI chose that category |

### Categories

| Category | Meaning |
|---|---|
| `LLM Efficiency` | The paper is about making AI models cheaper, faster, or smaller to run |
| `LLM Scaling` | The paper is about making AI models more powerful by training bigger models |
| `Other — ...` | The paper is about something else (e.g. benchmarks, new architectures, alignment) |

> Papers with confidence **below 75%** are flagged with `⚑ NEEDS REVIEW` in the terminal — these should be checked manually.

---

## How Classification Works

The AI classifies each paper in **two steps**:

**Step 1 — Quick Screen**
The AI does a fast read and decides:
- Is this paper about **Efficiency**? (making models cheaper/faster)
- Is this paper about **Scaling**? (making models bigger/smarter)
- Or is it about something **else**?

**Step 2 — Deep Classification**
Based on Step 1, the AI does a deeper analysis:
- For Efficiency/Scaling papers → confirms or corrects the Step 1 result with full reasoning
- For Other papers → assigns a short descriptive label (e.g. `Other — Benchmark / Evaluation`)

This two-step approach improves accuracy by giving the AI a chance to reconsider its first answer.

---

## Troubleshooting

**The pipeline shows `ERROR` for a paper**
> The AI server was unreachable or timed out. The paper will be retried automatically on the next run. Check that your server URL is correct.

**A paper shows low confidence (below 75%)**
> The AI was unsure about the classification. Review it manually — the justification column will explain why.

**No PDFs found**
> Make sure your PDF files are inside the `papers/pdf/` folder.

---

## Requirements

- Python 3.10+
- Access to an Ollama server (ask your supervisor for the server address)
- PDF files placed in `papers/pdf/`
