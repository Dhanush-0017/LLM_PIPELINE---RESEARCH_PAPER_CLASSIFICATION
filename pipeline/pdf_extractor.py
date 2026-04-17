import os
import re
import json
import fitz  # PyMuPDF

OUTPUT_DIR     = "output"
CLEAN_TEXT_DIR = "papers/clean_text"
MIN_SECTION_LEN = 80
MAX_SECTION_LEN = 6000

# Match a line that IS a section heading (stripped, exact match)
# TOC entries like "Introduction . . . . 3" won't match because of the dots/numbers
SECTION_PATTERNS = {
    "abstract":     re.compile(r'^abstract$', re.IGNORECASE),
    "introduction": re.compile(r'^(?:\d+[\.\s]*)?\bintroduction\b$', re.IGNORECASE),
    "conclusion":   re.compile(r'^(?:\d+[\.\s]*)?\b(?:conclusions?|concluding\s+remarks?|discussion|summary(?:\s+and\s+conclusions?)?)$', re.IGNORECASE),
}

# Lines that signal end of section (don't start a new one we care about)
STOP_PATTERN = re.compile(
    r'^(?:\d+[\.\s]*)?(?:acknowledgem\w*|appendix|references|bibliography|related\s+work|background|preliminaries)\b',
    re.IGNORECASE
)

# Any numbered section heading like "2 Related Work" or "3. Experiments"
# Used to end the current section when it's not a target section
NUMBERED_SECTION = re.compile(r'^\d+\.?\s+[A-Z][a-zA-Z]')


# ── Step 1: Read the PDF ──────────────────────────────────────────────────────
def read_pdf(pdf_path: str) -> tuple[str, fitz.Document]:
    doc       = fitz.open(pdf_path)
    full_text = "\n".join(page.get_text() for page in doc)
    return full_text, doc


# ── Step 2: Extract the title ─────────────────────────────────────────────────
def extract_title(doc: fitz.Document) -> str:
    # Try PDF metadata first
    title = (doc.metadata.get("title") or "").strip()
    if title and len(title) > 5 and not title.lower().startswith("arxiv"):
        return title

    # Fall back to largest font on page 1
    best_size, best_text = 0, "Unknown Title"
    skip = re.compile(r'(arxiv|preprint|\[cs\.|http|©|copyright|\d{4}-\d{2}-\d{2})', re.IGNORECASE)

    for block in doc[0].get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = " ".join(s["text"] for s in line["spans"]).strip()
            size = max((s["size"] for s in line["spans"]), default=0)
            if text and not skip.search(text) and size > best_size and 8 < len(text) < 200:
                best_size, best_text = size, text

    return best_text


# ── Step 3: Extract Abstract, Introduction, Conclusion ───────────────────────
def extract_sections(full_text: str) -> dict[str, str]:
    """Simple line-by-line section extractor.

    Walks every line. When a line exactly matches a section heading name,
    start collecting text into that section. Stop when the next heading or
    a stop word (References, Appendix, etc.) is found.

    Why exact matching works: TOC entries look like "Introduction . . . 3"
    which won't match the pattern ^introduction$ — so TOC is skipped for free.

    For conclusion: take the LAST match (the real one, not a mid-paper subsection).
    For abstract/introduction: take the FIRST substantive match.
    """
    lines    = full_text.splitlines()
    buckets  = {"abstract": [], "introduction": [], "conclusion": []}
    current  = None
    buffer   = []

    def flush():
        if current and buffer:
            content = " ".join(buffer).strip()
            content = re.sub(r'  +', ' ', content)   # collapse extra spaces
            if len(content) >= MIN_SECTION_LEN:
                buckets[current].append(content)

    for line in lines:
        stripped = line.strip()

        # Check for a target section heading
        matched = next((name for name, pat in SECTION_PATTERNS.items() if pat.match(stripped)), None)
        if matched:
            flush()
            current, buffer = matched, []
            continue

        # Check for a stop word or any numbered section — end current section
        if STOP_PATTERN.match(stripped) or NUMBERED_SECTION.match(stripped):
            flush()
            current, buffer = None, []
            continue

        if current:
            buffer.append(stripped)

    flush()  # capture the last open section

    # Pick the best match from each bucket
    sections = {"abstract": "", "introduction": "", "conclusion": ""}

    for name in ("abstract", "introduction"):
        for candidate in buckets[name]:
            if len(candidate) >= MIN_SECTION_LEN:
                sections[name] = candidate
                break

    if buckets["conclusion"]:
        sections["conclusion"] = buckets["conclusion"][-1][:MAX_SECTION_LEN]

    return sections


# ── Step 4: Warn if any section is missing ────────────────────────────────────
def warn_missing_sections(paper_id: str, sections: dict) -> None:
    missing = [k for k, v in sections.items() if not v.strip()]
    if missing:
        print(f"\n  [WARN] {paper_id}: could not extract -> {', '.join(missing)}")


# ── Step 5: Turn sections into chunks ─────────────────────────────────────────
def chunk_by_section(sections: dict) -> list[dict]:
    return [
        {"chunk_index": i, "section": name, "text": text.strip(), "char_count": len(text)}
        for i, (name, text) in enumerate(sections.items())
        if text.strip()
    ]


# ── Step 6: Save human-readable text file ────────────────────────────────────
def save_clean_text(paper_id: str, title: str, sections: dict) -> None:
    os.makedirs(CLEAN_TEXT_DIR, exist_ok=True)
    with open(os.path.join(CLEAN_TEXT_DIR, f"{paper_id}.txt"), "w", encoding="utf-8") as f:
        f.write(f"TITLE\n{'=' * 60}\n{title}\n\n")
        for name in ("abstract", "introduction", "conclusion"):
            text = sections.get(name, "").strip()
            if text:
                f.write(f"{name.upper()}\n{'=' * 60}\n{text}\n\n")


# ── Step 7: Save structured JSON ──────────────────────────────────────────────
def save_chunks(paper_id: str, title: str, pdf_path: str, sections: dict, chunks: list) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = {
        "paper_id"          : paper_id,
        "title"             : title,
        "pdf_path"          : pdf_path,
        "sections_extracted": [k for k, v in sections.items() if v.strip()],
        "sections_missing"  : [k for k, v in sections.items() if not v.strip()],
        "num_chunks"        : len(chunks),
        "chunks"            : chunks,
        "classification"    : {},
    }
    with open(os.path.join(OUTPUT_DIR, f"{paper_id}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# ── Main entry point ──────────────────────────────────────────────────────────
def process_pdf(pdf_path: str, paper_id: str | None = None, title: str | None = None) -> tuple[list, str]:
    if not paper_id:
        paper_id = os.path.splitext(os.path.basename(pdf_path))[0]

    full_text, doc = read_pdf(pdf_path)

    if not title:
        title = extract_title(doc)

    sections = extract_sections(full_text)
    warn_missing_sections(paper_id, sections)

    chunks = chunk_by_section(sections)
    save_chunks(paper_id, title, pdf_path, sections, chunks)
    save_clean_text(paper_id, title, sections)

    return chunks, title
