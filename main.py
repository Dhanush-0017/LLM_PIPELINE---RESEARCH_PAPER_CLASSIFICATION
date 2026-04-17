import os
import sys
import csv
import json
import glob
import textwrap

import importlib

from pipeline.pdf_extractor import process_pdf
from pipeline.classifier    import process_paper as summarize_paper

try:
    _fetcher = importlib.import_module("1_paper_download_onef")
    get_filtered_papers_exact_count = _fetcher.get_filtered_papers_exact_count
    download_pdf_for_paper          = _fetcher.download_pdf_for_paper
    FETCHER_AVAILABLE = True
except ModuleNotFoundError:
    FETCHER_AVAILABLE = False

PAPERS_DIR     = "papers/pdf"
OUTPUT_DIR     = "output"
CLEAN_TEXT_DIR = "papers/clean_text"
RESULTS_DIR    = "results"
HEADERS        = ["paper_id", "title", "classification", "accuracy", "justification"]

# ── Colours ───────────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
GRAY   = "\033[90m"
WHITE  = "\033[97m"


def c(text, *codes):
    return "".join(codes) + str(text) + RESET


def conf_bar(confidence):
    try:
        val = int(confidence)
    except (ValueError, TypeError):
        val = 0
    filled = round(val / 100 * 16)
    colour = GREEN if val >= 80 else YELLOW if val >= 50 else RED
    return c("█" * filled + "░" * (16 - filled), colour) + c(f"  {val}%", BOLD)


# ── Fetch papers from arXiv ───────────────────────────────────────────────────
def fetch_papers(query, count, year_min, year_max):
    print()
    print(f"  {c('fetching papers from arXiv', BOLD, WHITE)}")
    print(c("  " + "─" * 54, GRAY))
    print(f"  query      {c(query, CYAN)}")
    print(f"  count      {c(count, CYAN, BOLD)}")
    print(f"  years      {c(f'{year_min} – {year_max}', CYAN)}")
    print()

    papers = get_filtered_papers_exact_count(
        query=query,
        target_count=count,
        year_min=year_min,
        year_max=year_max,
        cite_min=0,
    )

    print(f"  {c(str(len(papers)), CYAN, BOLD)} paper(s) found")
    print()

    for i, paper in enumerate(papers, start=1):
        print(f"  {c(f'[{i}/{len(papers)}]', GRAY)}  {c(paper['title'], WHITE)}", end="  ", flush=True)
        try:
            download_pdf_for_paper(paper, output_dir=PAPERS_DIR)
            print(c("downloaded", GREEN))
        except (RuntimeError, ValueError) as e:
            print(c(f"failed — {e}", RED))

    print()


# ── File helpers ──────────────────────────────────────────────────────────────
def get_pdf_files():
    pdfs = glob.glob(os.path.join(PAPERS_DIR, "*.pdf"))
    if not pdfs:
        print(c(f"\n  no PDFs found in '{PAPERS_DIR}/'", RED))
        sys.exit(1)
    return pdfs



def save_results(data):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    csv_path  = os.path.join(RESULTS_DIR, "results.csv")
    json_path = os.path.join(RESULTS_DIR, "results.json")

    classification = data.get("classification", {})
    row = {
        "paper_id"      : data.get("paper_id", ""),
        "title"         : data["title"],
        "classification": classification.get("category", ""),
        "accuracy"      : classification.get("confidence", "") + "%",
        "justification" : classification.get("justification", ""),
    }

    file_exists = os.path.isfile(csv_path)
    try:
        with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=HEADERS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except PermissionError:
        print(c(f"  cannot write CSV — close the file first", RED))

    existing = []
    if os.path.isfile(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing = []

    existing.append({
        "title"         : data["title"],
        "num_chunks"    : data["num_chunks"],
        "classification": classification,
    })
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


# ── Per-paper ─────────────────────────────────────────────────────────────────
def process(pdf_path, index, total):
    paper_id = os.path.splitext(os.path.basename(pdf_path))[0]

    print()
    print(f"  {c(f'[{index}/{total}]', GRAY)}  {c(os.path.basename(pdf_path), BOLD, WHITE)}")
    print(c("  " + "─" * 54, GRAY))

    print(f"  {'chunking':<14}", end="", flush=True)
    chunks, title = process_pdf(pdf_path, paper_id=paper_id)
    print(c(f"  {len(chunks)} chunk(s)", DIM))

    print(f"  {'classifying':<14}", end="", flush=True)
    data, skipped = summarize_paper(paper_id)
    print(c("  done", GREEN))

    clf        = data.get("classification", {})
    category   = clf.get("category", "N/A")
    confidence = clf.get("confidence", "0")
    why        = clf.get("justification", "")
    try:
        flagged = int(clf.get("confidence", "0")) < 75
    except (ValueError, TypeError):
        flagged = False

    print()
    print(f"  {'title':<14}  {c(title, WHITE)}")
    print(f"  {'category':<14}  {c(category, CYAN, BOLD)}")
    flag_str = c("  ⚑ NEEDS REVIEW", YELLOW) if flagged else ""
    print(f"  {'confidence':<14}  {conf_bar(confidence)}{flag_str}")

    if why:
        lines = textwrap.wrap(why, width=54)
        print(f"  {'why':<14}  {c(lines[0], DIM)}")
        for line in lines[1:]:
            print(f"  {'':<14}  {c(line, DIM)}")

    save_results(data)
    return skipped, flagged, category == "ERROR"


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if sys.platform == "win32":
        os.system("")

    # ── Clear previous results ────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for f in [os.path.join(RESULTS_DIR, "results.csv"), os.path.join(RESULTS_DIR, "results.json")]:
        if os.path.isfile(f):
            os.remove(f)

    # ── Step 1: Fetch papers from arXiv ──────────────────────────────────────
    if FETCHER_AVAILABLE:
        fetch_papers(
            query    = "LLM efficiency",
            count    = 10,
            year_min = 2020,
            year_max = 2026,
        )
    else:
        print(c("  paper fetcher not available — classifying existing PDFs in papers/pdf/", YELLOW))

    # ── Step 2: Classify all downloaded PDFs ──────────────────────────────────
    print()
    print(f"  {c('research paper analysis', BOLD, WHITE)}")
    print(c("  " + "─" * 54, GRAY))

    pdfs  = get_pdf_files()
    total = len(pdfs)

    print(f"  {c(str(total), CYAN, BOLD)} paper(s)  ·  {c(PAPERS_DIR + '/', GRAY)}")

    n_done = n_errors = n_review = 0

    for i, pdf_path in enumerate(pdfs, start=1):
        _, flagged, errored = process(pdf_path, i, total)
        if errored:
            n_errors += 1
        else:
            n_done += 1
        if flagged:
            n_review += 1

    print()
    print(c("  " + "─" * 54, GRAY))
    parts = [
        f"{c(str(n_done),   GREEN,  BOLD)} classified",
        f"{c(str(n_errors), RED,    BOLD)} errors",
        f"{c(str(n_review), YELLOW, BOLD)} needs review",
    ]
    print("  " + "  ·  ".join(parts))
    print(f"  {c('saved', DIM)}  →  results/results.csv  &  results.json")
    print()


if __name__ == "__main__":
    main()
