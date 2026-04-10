import os
import re
import json
import fitz  # PyMuPDF — used to open and read PDF files


# Where to save intermediate and cleaned output files
OUTPUT_DIR     = "output"
CLEAN_TEXT_DIR = "papers/clean_text"

# A section must have at least this many characters to be considered valid.
# Prevents saving empty or near-empty sections (e.g. a heading with no body).
MIN_SECTION_LEN = 80


# ── Step 1: Read the PDF ──────────────────────────────────────────────────────
def read_pdf(pdf_path: str) -> tuple[str, fitz.Document]:
    """Open a PDF and extract all its text as one big string.
    Also returns the document object so we can read font sizes for the title.
    """
    doc       = fitz.open(pdf_path)
    full_text = "\n".join(page.get_text() for page in doc)
    return full_text, doc


# ── Step 2: Extract the Paper Title ──────────────────────────────────────────
def extract_title(doc: fitz.Document, metadata: dict) -> str:
    """Try to find the paper title using two strategies:
    1. Read from PDF metadata (fast, but often missing or wrong for arxiv papers).
    2. Find the largest font text on page 1 (reliable for most academic PDFs).
    """
    # Strategy 1: use embedded PDF metadata if it looks valid
    title = (metadata.get("title") or "").strip()
    if title and len(title) > 5 and not title.lower().startswith("arxiv"):
        return title

    # Strategy 2: scan page 1 for the line with the largest font size.
    # Skip lines that look like headers, dates, URLs, or arxiv stamps.
    page      = doc[0]
    blocks    = page.get_text("dict")["blocks"]
    best_size = 0

    skip_patterns = re.compile(
        r'(arxiv|preprint|\[cs\.|http|©|copyright|submission|proceedings|conference|\d{4}-\d{2}-\d{2})',
        re.IGNORECASE
    )

    all_lines = []
    for block in blocks:
        if block.get("type") != 0:  # skip non-text blocks (images, etc.)
            continue
        for line in block.get("lines", []):
            line_text = " ".join(s["text"] for s in line.get("spans", [])).strip()
            if not line_text or skip_patterns.search(line_text):
                continue
            line_max_size = max((s["size"] for s in line.get("spans", [])), default=0)
            if 8 < len(line_text) < 200:  # ignore very short or very long lines
                all_lines.append((line_max_size, line_text))
                if line_max_size > best_size:
                    best_size = line_max_size

    # Collect all lines at the largest font size — titles often span multiple lines
    title_lines = [text for size, text in all_lines if size >= best_size - 0.5]
    title = " ".join(title_lines) if title_lines else "Unknown Title"
    return fix_spaced_title(title)


def fix_spaced_title(title: str) -> str:
    """Fix OCR artifacts where letters in the title are spaced apart.
    e.g. "F LASH A TTENTION" → "FLASHATTENTION"
    """
    # Fix single uppercase letter separated from the rest: "F LASH" → "FLASH"
    title = re.sub(r'\b([A-Z])\s+([A-Z]{2,})\b', r'\1\2', title)

    # Fix short tokens before a colon: "Q L O R A: ..." → "QLORA: ..."
    parts  = title.split(":", 1)
    prefix = parts[0].strip()
    words  = prefix.split()
    if len(words) >= 2 and all(len(w) <= 3 for w in words):
        parts[0] = "".join(words)
        return ": ".join(parts)

    return title


# ── Step 3: Extract Abstract, Introduction, and Conclusion ────────────────────
def extract_sections(full_text: str) -> dict[str, str]:
    """Extract the three most important sections from the paper text.

    Uses multiple regex patterns per section to handle the many ways
    academic papers format their headings (e.g. "1. Introduction",
    "I. Introduction", "Concluding Remarks", "Summary", etc.).

    Falls back to position-based extraction if no heading is found.
    """
    sections = {"abstract": "", "introduction": "", "conclusion": ""}

    # ── Abstract ──────────────────────────────────────────────────────────────
    # Try patterns from most common to least common heading format.
    abstract_patterns = [
        # "Abstract" on its own line followed by the body
        r'(?i)\babstract\b\s*\n(.*?)(?=\n\s*(?:\d+[\.\s]|[IVX]+[\.\s]|introduction\b|keywords?\b|\Z))',
        # "Abstract—" or "Abstract:" with the text on the same line
        r'(?i)\babstract\b[\s:—–-]+(.*?)(?=\n\s*(?:\d+[\.\s]|introduction\b|keywords?\b|\Z))',
        # OCR artifact: "A B S T R A C T" with spaces between letters
        r'(?i)\bA\s*B\s*S\s*T\s*R\s*A\s*C\s*T\b\s*\n(.*?)(?=\n\s*(?:\d+[\.\s]|introduction\b|\Z))',
    ]
    for pattern in abstract_patterns:
        m = re.search(pattern, full_text, re.DOTALL)
        if m:
            text = m.group(1).strip()
            if len(text) >= MIN_SECTION_LEN:
                sections["abstract"] = text
                break

    # Fallback: find the first dense paragraph in the opening 3000 characters.
    # Most papers place the abstract at the very top, so this works well.
    if not sections["abstract"]:
        for m in re.finditer(r'\n\n(.{100,1500}?)\n\n', full_text[:3000], re.DOTALL):
            candidate = m.group(1).strip()
            lines     = candidate.splitlines()
            avg_len   = sum(len(l) for l in lines) / max(len(lines), 1)
            # Reject affiliation/address blocks which have many short lines
            if avg_len > 40:
                sections["abstract"] = candidate
                break

    # ── Introduction ──────────────────────────────────────────────────────────
    # Handles: "Introduction", "1. Introduction", "1 Introduction", "I. Introduction"
    intro_patterns = [
        r'(?i)\n\s*(?:\d+\.?\s+|[IVX]+\.?\s+)?introduction\s*\n'
        r'(.*?)'
        r'(?=\n\s*(?:\d+[\.\s]+\w|[IVX]+[\.\s]+\w|related work\b|background\b|'
        r'methodology\b|preliminaries\b|problem\s+formulation\b))',
    ]
    for pattern in intro_patterns:
        m = re.search(pattern, full_text, re.DOTALL)
        if m:
            text = m.group(1).strip()
            if len(text) >= MIN_SECTION_LEN:
                sections["introduction"] = text
                break

    # ── Conclusion ────────────────────────────────────────────────────────────
    # Handles many variations: "Conclusion", "Conclusions", "Concluding Remarks",
    # "Summary and Conclusions", "Discussion and Conclusions", "Summary", "Discussion"
    conclusion_patterns = [
        r'(?i)\n\s*(?:\d+[\.\s]+)?conclusions?\b[^\n]*\n'
        r'(.*?)'
        r'(?=\n\s*(?:references\b|bibliography\b|acknowledgem\b)|\Z)',

        r'(?i)\n\s*(?:\d+[\.\s]+)?concluding\s+remarks?\b[^\n]*\n'
        r'(.*?)'
        r'(?=\n\s*(?:references\b|bibliography\b|acknowledgem\b)|\Z)',

        r'(?i)\n\s*(?:\d+[\.\s]+)?(?:summary|discussion)\s+and\s+conclusions?\b[^\n]*\n'
        r'(.*?)'
        r'(?=\n\s*(?:references\b|bibliography\b|acknowledgem\b)|\Z)',

        # Last resort: standalone "Summary" or "Discussion" section
        r'(?i)\n\s*(?:\d+[\.\s]+)?(?:summary|discussion)\b[^\n]*\n'
        r'(.*?)'
        r'(?=\n\s*(?:references\b|bibliography\b|acknowledgem\b)|\Z)',
    ]
    for pattern in conclusion_patterns:
        m = re.search(pattern, full_text, re.DOTALL)
        if m:
            text = m.group(1).strip()
            if len(text) >= MIN_SECTION_LEN:
                sections["conclusion"] = text
                break

    # Fallback: grab the last paragraph before the References section.
    # Most papers end with the conclusion right before references.
    if not sections["conclusion"]:
        ref_match = re.search(r'(?i)\n\s*references\b', full_text)
        if ref_match:
            pre_refs  = full_text[:ref_match.start()].strip()
            last_para = re.split(r'\n{2,}', pre_refs)
            for para in reversed(last_para):
                para = para.strip()
                if len(para) >= MIN_SECTION_LEN:
                    sections["conclusion"] = para
                    break

    return sections


# ── Step 4: Warn if any section could not be found ───────────────────────────
def warn_missing_sections(paper_id: str, sections: dict[str, str]) -> None:
    """Print a warning for every section that came back empty.
    This is important because missing sections mean the LLM will classify
    with less information, which may lower accuracy.
    """
    missing = [name for name, text in sections.items() if not text.strip()]
    if missing:
        print(f"\n  [WARN] {paper_id}: could not extract → {', '.join(missing)}")
        print(f"         Classification will use only available sections.")


# ── Step 5: Turn sections into chunks ─────────────────────────────────────────
def chunk_by_section(sections: dict[str, str]) -> list[dict]:
    """Wrap each extracted section into a structured chunk dictionary.
    Each chunk records its index, section name, text, and character count.
    Only non-empty sections are included.
    """
    chunks = []
    for i, (section_name, text) in enumerate(sections.items()):
        text = text.strip()
        if text:
            chunks.append({
                "chunk_index": i,
                "section"    : section_name,
                "text"       : text,
                "char_count" : len(text),
            })
    return chunks


# ── Step 6: Save a human-readable text file ───────────────────────────────────
def save_clean_text(paper_id: str, title: str, sections: dict[str, str]) -> None:
    """Save the extracted sections as a clean .txt file for manual inspection.
    Useful for checking that extraction worked correctly before trusting classifications.
    """
    os.makedirs(CLEAN_TEXT_DIR, exist_ok=True)
    path = os.path.join(CLEAN_TEXT_DIR, f"{paper_id}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"TITLE\n{'=' * 60}\n{title}\n\n")
        for section_name in ("abstract", "introduction", "conclusion"):
            text = sections.get(section_name, "").strip()
            if text:
                f.write(f"{section_name.upper()}\n{'=' * 60}\n{text}\n\n")


# ── Step 7: Save structured data to JSON ──────────────────────────────────────
def save_chunks(paper_id: str, title: str, pdf_path: str, sections: dict[str, str], chunks: list[dict]) -> None:
    """Save all extracted data as a JSON file in the output/ directory.
    This JSON is the main working file — the summarizer reads it and writes
    the classification result back into it.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = {
        "paper_id"          : paper_id,
        "title"             : title,
        "pdf_path"          : pdf_path,
        "sections_extracted": [k for k, v in sections.items() if v.strip()],
        "sections_missing"  : [k for k, v in sections.items() if not v.strip()],
        "num_chunks"        : len(chunks),
        "chunks"            : chunks,
        "classification"    : {},  # filled in later by the summarizer
    }
    with open(os.path.join(OUTPUT_DIR, f"{paper_id}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# ── Main entry point ──────────────────────────────────────────────────────────
def process_pdf(pdf_path: str, paper_id: str | None = None, title: str | None = None) -> tuple[list[dict], str]:
    """Run the full chunking pipeline for a single PDF.
    Returns the list of chunks and the extracted title.
    Called by main.py for each paper.
    """
    if not paper_id:
        paper_id = os.path.splitext(os.path.basename(pdf_path))[0]

    full_text, doc = read_pdf(pdf_path)

    if not title:
        title = extract_title(doc, doc.metadata)

    sections = extract_sections(full_text)
    warn_missing_sections(paper_id, sections)

    chunks = chunk_by_section(sections)
    save_chunks(paper_id, title, pdf_path, sections, chunks)
    save_clean_text(paper_id, title, sections)

    return chunks, title
