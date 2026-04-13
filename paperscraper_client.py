from datetime import datetime
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET
import requests


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
DEFAULT_REQUEST_TIMEOUT = 60
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 3.0


def _build_search_query(query, *, title_only=False):
    field = "ti" if title_only else "all"
    return f'{field}:"{query}"'


def _clean_text(value):
    if not value:
        return None
    return " ".join(value.split())


def _extract_year(entry):
    published = _clean_text(entry.findtext("atom:published", default="", namespaces=ATOM_NS))
    if not published:
        return None

    try:
        return datetime.fromisoformat(published.replace("Z", "+00:00")).year
    except ValueError:
        return None


def _extract_authors(entry):
    authors = []
    for author in entry.findall("atom:author", ATOM_NS):
        name = _clean_text(author.findtext("atom:name", default="", namespaces=ATOM_NS))
        if name:
            authors.append(name)
    return authors


def _extract_arxiv_id(entry=None, paper=None):
    if paper:
        doi = paper.get("doi") or ""
        if "arXiv." in doi:
            return doi.split("arXiv.", 1)[1]

        link = paper.get("link") or ""
        if "/abs/" in link:
            return link.rsplit("/abs/", 1)[1]

    if entry is not None:
        entry_id = _clean_text(entry.findtext("atom:id", default="", namespaces=ATOM_NS)) or ""
        if "/abs/" in entry_id:
            return entry_id.rsplit("/abs/", 1)[1]

    return None


def _parse_entry(entry):
    arxiv_id = _extract_arxiv_id(entry=entry)
    if not arxiv_id:
        return None

    abstract = _clean_text(entry.findtext("atom:summary", default="", namespaces=ATOM_NS))
    if not abstract:
        return None

    return {
        "title": _clean_text(entry.findtext("atom:title", default="", namespaces=ATOM_NS)) or "Untitled",
        "abstract": abstract,
        "year": _extract_year(entry),
        "link": f"https://arxiv.org/abs/{arxiv_id}",
        "citation_count": 0,
        "fields": [],
        "authors": _extract_authors(entry),
        "journal": _clean_text(entry.findtext("arxiv:journal_ref", default="", namespaces=ATOM_NS)),
        "doi": f"10.48550/arXiv.{arxiv_id}",
    }


def _request_with_retries(
    url,
    *,
    params=None,
    timeout=DEFAULT_REQUEST_TIMEOUT,
    max_retries=DEFAULT_MAX_RETRIES,
    retry_backoff_seconds=DEFAULT_RETRY_BACKOFF_SECONDS,
):
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(retry_backoff_seconds * attempt)

    raise RuntimeError(
        f"arXiv request failed after {max_retries} attempts: {last_error}"
    ) from last_error


def _fetch_entries(
    query,
    max_results,
    *,
    title_only=False,
    timeout=DEFAULT_REQUEST_TIMEOUT,
    max_retries=DEFAULT_MAX_RETRIES,
    retry_backoff_seconds=DEFAULT_RETRY_BACKOFF_SECONDS,
):
    params = {
        "search_query": _build_search_query(query, title_only=title_only),
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    response = _request_with_retries(
        ARXIV_API_URL,
        params=params,
        timeout=timeout,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )

    root = ET.fromstring(response.text)
    return root.findall("atom:entry", ATOM_NS)


def get_papers(
    query,
    per_page,
    *,
    title_only=False,
    timeout=DEFAULT_REQUEST_TIMEOUT,
    max_retries=DEFAULT_MAX_RETRIES,
    retry_backoff_seconds=DEFAULT_RETRY_BACKOFF_SECONDS,
):
    entries = _fetch_entries(
        query,
        per_page,
        title_only=title_only,
        timeout=timeout,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    papers = []

    for entry in entries:
        paper = _parse_entry(entry)
        if paper:
            papers.append(paper)

    return papers


def apply_filters(
    papers,
    year_min,
    year_max,
    cite_min,
    author_query="",
    journal_query="",
):
    author_query = author_query.strip().lower()
    journal_query = journal_query.strip().lower()

    filtered = []

    for paper in papers:
        year = paper.get("year")
        citations = paper.get("citation_count", 0)
        authors = paper.get("authors", [])
        journal = (paper.get("journal") or "").lower()

        if year is None or year < year_min or year > year_max:
            continue
        if citations < cite_min:
            continue
        if author_query and not any(author_query in author.lower() for author in authors):
            continue
        if journal_query and journal_query not in journal:
            continue

        filtered.append(paper)

    return filtered


def get_filtered_papers_exact_count(
    query,
    target_count,
    year_min,
    year_max,
    cite_min,
    author_query="",
    journal_query="",
    max_pages=25,
    title_only=False,
    timeout=DEFAULT_REQUEST_TIMEOUT,
    max_retries=DEFAULT_MAX_RETRIES,
    retry_backoff_seconds=DEFAULT_RETRY_BACKOFF_SECONDS,
):
    if target_count <= 0:
        return []

    per_page = min(100, max(10, target_count))
    fetch_count = per_page * max_pages
    papers = get_papers(
        query,
        fetch_count,
        title_only=title_only,
        timeout=timeout,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    matched = apply_filters(
        papers,
        year_min,
        year_max,
        cite_min,
        author_query,
        journal_query,
    )
    return matched[:target_count]


def download_pdf_for_paper(
    paper,
    output_dir,
    *,
    timeout=DEFAULT_REQUEST_TIMEOUT,
    max_retries=DEFAULT_MAX_RETRIES,
    retry_backoff_seconds=DEFAULT_RETRY_BACKOFF_SECONDS,
):
    arxiv_id = _extract_arxiv_id(paper=paper)
    if not arxiv_id:
        raise ValueError("Paper is missing arXiv identifier")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / f"arxiv_{arxiv_id}.pdf"
    if final_path.exists() and final_path.stat().st_size > 0:
        return str(final_path)

    response = _request_with_retries(
        ARXIV_PDF_URL.format(arxiv_id=arxiv_id),
        timeout=timeout,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )

    if response.content[:4] != b"%PDF":
        raise RuntimeError(f"Expected PDF content for {arxiv_id}")

    with open(final_path, "wb") as handle:
        handle.write(response.content)

    return str(final_path)


def _get_pdf_reader_class():
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader
        except ImportError as exc:
            raise ImportError(
                "PDF extraction requires pypdf or PyPDF2. Install one in your active environment."
            ) from exc
    return PdfReader


def extract_pdf_text(pdf_path, max_chars=30000):
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    PdfReader = _get_pdf_reader_class()
    reader = PdfReader(str(pdf_path))

    parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text:
            parts.append(page_text)

    text = "\n\n".join(parts)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()

    if max_chars and len(text) > max_chars:
        text = text[:max_chars]

    return text


def build_fulltext_paper(paper, pdf_path, max_chars=30000):
    enriched = dict(paper)
    enriched["pdf_path"] = str(pdf_path)
    enriched["full_text"] = extract_pdf_text(pdf_path, max_chars=max_chars)
    return enriched
