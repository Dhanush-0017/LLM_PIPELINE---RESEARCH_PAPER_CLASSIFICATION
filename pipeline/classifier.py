import json
import requests
import os
import time

from pipeline.prompts import (
    STAGE1_PROMPT,
    STAGE2A_PROMPT,
    STAGE2B_PROMPT,
    parse_stage1,
    parse_classification,
)

# ── Configuration ─────────────────────────────────────────────────────────────
# Update OLLAMA_URL and OLLAMA_MODEL to match your server setup.
OLLAMA_URL   = "http://10.230.100.240:17020/api/generate"
OLLAMA_MODEL = "gpt-oss:120b"
OUTPUT_DIR   = "output"
MAX_RETRIES  = 3   # how many times to retry if the LLM call fails
RETRY_DELAY  = 5   # seconds to wait between retries

# Maps Stage 1 output (A/B/C) to a human-readable label for the screening hint
STAGE1_LABEL = {"A": "LLM Efficiency", "B": "LLM Scaling", "C": None}


# ── LLM Communication ─────────────────────────────────────────────────────────
def call_llm(prompt: str, timeout: int = 300) -> str:
    """Send a prompt to the Ollama LLM and return its response.

    Retries up to MAX_RETRIES times if the call fails (e.g. network timeout,
    server error). Temperature is set to 0 so the same paper always gets
    the same classification — important for reproducibility.

    Raises RuntimeError if all retries fail.
    """
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "temperature": 0},
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()["response"].strip()
        except (requests.exceptions.RequestException, KeyError, ValueError) as e:
            last_error = e
            if attempt < MAX_RETRIES:
                print(f"\n  [Retry {attempt}/{MAX_RETRIES}] LLM error: {e}. Retrying in {RETRY_DELAY}s...", end="", flush=True)
                time.sleep(RETRY_DELAY)
    raise RuntimeError(f"LLM failed after {MAX_RETRIES} attempts: {last_error}")


# ── Data Loading ──────────────────────────────────────────────────────────────
def load_chunks(paper_id: str) -> dict:
    """Load the JSON file created by chunker.py for a given paper."""
    file_path = os.path.join(OUTPUT_DIR, f"{paper_id}.json")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_already_processed(data: dict) -> bool:
    """Check if this paper already has a valid classification.
    Papers marked ERROR are NOT skipped — they get retried on the next run.
    """
    category = data.get("classification", {}).get("category", "").strip()
    return bool(category) and category != "ERROR"


# ── Prompt Building ───────────────────────────────────────────────────────────
def build_input(data: dict) -> str:
    """Assemble the paper's Title, Abstract, Introduction, and Conclusion
    into a single string to send to the LLM.
    Only includes sections that were successfully extracted.
    """
    sections = {c["section"]: c["text"] for c in data["chunks"]}
    parts    = [f"Title: {data['title']}"]
    for section in ("abstract", "introduction", "conclusion"):
        text = sections.get(section, "").strip()
        if text:
            parts.append(f"{section.capitalize()}:\n{text}")
    return "\n\n".join(parts)


def build_stage1_prompt(paper_input: str) -> str:
    """Combine the paper text with the Stage 1 screening prompt."""
    return f"{paper_input}\n\n{STAGE1_PROMPT}"


def build_stage2_prompt(paper_input: str, stage1: dict) -> str:
    """Choose the right Stage 2 prompt based on the Stage 1 result.

    - Stage 1 said A (Efficiency) or B (Scaling):
        Use STAGE2A — focused 2-category prompt with a screening hint
        so Stage 2 starts with context but can still disagree.

    - Stage 1 said C (Other):
        Use STAGE2B — secondary classification into Model Architecture,
        Training & Alignment, LLM Evaluation, etc.
    """
    primary = stage1.get("primary", "C")
    label   = STAGE1_LABEL.get(primary)

    if primary in ("A", "B"):
        # Prepend a screening note so Stage 2 knows what Stage 1 found
        hint = (
            f"[SCREENING NOTE: Initial analysis suggests this paper is about {label}. "
            f"Reasoning: {stage1.get('reasoning', '')}. "
            f"Confirm or correct based on your own reading of the paper below.]\n\n"
        )
        return f"{hint}{paper_input}\n\n{STAGE2A_PROMPT}"
    else:
        return f"{paper_input}\n\n{STAGE2B_PROMPT}"


# ── Saving ────────────────────────────────────────────────────────────────────
def save_paper(data: dict) -> None:
    """Write the updated paper data (including classification) back to its JSON file."""
    file_path = os.path.join(OUTPUT_DIR, f"{data['paper_id']}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# ── Main entry point ──────────────────────────────────────────────────────────
def process_paper(paper_id: str) -> tuple[dict, bool]:
    """Run the full two-stage classification pipeline for one paper.

    Stage 1 — fast screen: is this Efficiency, Scaling, or Other?
    Stage 2 — deep classification: confirm/correct with full reasoning.

    Returns (data, skipped) where skipped=True means it was already classified.
    """
    data = load_chunks(paper_id)

    if is_already_processed(data):
        return data, True  # already classified, skip

    paper_input = build_input(data)

    # ── Stage 1: quick efficiency / scaling / other screen ────────────────────
    # Default to C (Other) in case Stage 1 fails, so Stage 2 still runs safely.
    stage1 = {"primary": "C", "reasoning": "stage 1 skipped due to error"}
    try:
        stage1_raw = call_llm(build_stage1_prompt(paper_input))
        stage1     = parse_stage1(stage1_raw)
    except RuntimeError as e:
        print(f"\n  [WARN] Stage 1 failed: {e}. Proceeding to Stage 2 without hint.")

    # ── Stage 2: full classification ──────────────────────────────────────────
    try:
        stage2_raw = call_llm(build_stage2_prompt(paper_input, stage1))
    except RuntimeError as e:
        # If Stage 2 also fails after all retries, mark the paper as ERROR
        # so it gets retried on the next run instead of being silently skipped.
        print(f"\n  [ERROR] {e}")
        data["classification"] = {
            "reasoning"    : "",
            "category"     : "ERROR",
            "confidence"   : "0",
            "justification": str(e),
        }
        save_paper(data)
        return data, False

    data["classification"] = parse_classification(stage2_raw)
    save_paper(data)
    return data, False
