import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import requests


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
ARXIV_MAX_RESULTS_PER_CALL = 2000
ARXIV_MAX_RESULTS_TOTAL = 30000
ARXIV_RECOMMENDED_DELAY_SECONDS = 3.0
DEFAULT_CONNECT_TIMEOUT = 10
DEFAULT_READ_TIMEOUT = 90
DEFAULT_REQUEST_TIMEOUT = (DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT)
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 3.0
DEFAULT_HEADERS = {"User-Agent": "GRA-arXiv-downloader/1.0"}


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch recent arXiv papers and download their PDFs")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--count", type=int, default=10, help="Number of papers to download")
    parser.add_argument("--year-min", type=int, default=2020, help="Minimum publication year")
    parser.add_argument("--year-max", type=int, default=2100, help="Maximum publication year")
    parser.add_argument("--cite-min", type=int, default=0, help="Minimum citation count; for arXiv this should stay 0")
    parser.add_argument("--author", default="", help="Author name contains")
    parser.add_argument("--journal", default="", help="Journal reference contains")
    parser.add_argument("--max-pages", type=int, default=2, help="Maximum result batches to scan")
    parser.add_argument("--title-only", action="store_true", help="Search only within paper titles instead of all indexed metadata")
    parser.add_argument("--timeout", type=int, default=DEFAULT_READ_TIMEOUT, help="Read timeout in seconds")
    parser.add_argument("--retries", type=int, default=DEFAULT_MAX_RETRIES, help="Number of retry attempts")
    parser.add_argument("--retry-backoff", type=float, default=DEFAULT_RETRY_BACKOFF_SECONDS, help="Base seconds to wait between retries; later retries wait longer")
    parser.add_argument("--output-dir", default="downloaded_pdfs", help="Folder where PDFs will be saved")
    return parser.parse_args()


def build_search_query(query, *, title_only=False):
    field = "ti" if title_only else "all"
    return f'{field}:"{query}"'


def normalize_timeout(timeout):
    if isinstance(timeout, tuple):
        return timeout
    return (DEFAULT_CONNECT_TIMEOUT, timeout)


def clean_text(value):
    if not value:
        return None
    return " ".join(value.split())


def extract_year(entry):
    published = clean_text(entry.findtext("atom:published", default="", namespaces=ATOM_NS))
    if not published:
        return None

    try:
        return datetime.fromisoformat(published.replace("Z", "+00:00")).year
    except ValueError:
        return None


def extract_authors(entry):
    authors = []
    for author in entry.findall("atom:author", ATOM_NS):
        name = clean_text(author.findtext("atom:name", default="", namespaces=ATOM_NS))
        if name:
            authors.append(name)
    return authors


def extract_arxiv_id(entry=None, paper=None):
    if paper:
        doi = paper.get("doi") or ""
        if "arXiv." in doi:
            return doi.split("arXiv.", 1)[1]

        link = paper.get("link") or ""
        if "/abs/" in link:
            return link.rsplit("/abs/", 1)[1]

    if entry is not None:
        entry_id = clean_text(entry.findtext("atom:id", default="", namespaces=ATOM_NS)) or ""
        if "/abs/" in entry_id:
            return entry_id.rsplit("/abs/", 1)[1]

    return None


def parse_entry(entry):
    arxiv_id = extract_arxiv_id(entry=entry)
    if not arxiv_id:
        return None

    abstract = clean_text(entry.findtext("atom:summary", default="", namespaces=ATOM_NS))
    if not abstract:
        return None

    return {
        "title": clean_text(entry.findtext("atom:title", default="", namespaces=ATOM_NS)) or "Untitled",
        "abstract": abstract,
        "year": extract_year(entry),
        "link": f"https://arxiv.org/abs/{arxiv_id}",
        "citation_count": 0,
        "fields": [],
        "authors": extract_authors(entry),
        "journal": clean_text(entry.findtext("arxiv:journal_ref", default="", namespaces=ATOM_NS)),
        "doi": f"10.48550/arXiv.{arxiv_id}",
    }


def request_with_retries(
    url,
    *,
    params=None,
    timeout=DEFAULT_REQUEST_TIMEOUT,
    max_retries=DEFAULT_MAX_RETRIES,
    retry_backoff_seconds=DEFAULT_RETRY_BACKOFF_SECONDS,
):
    last_error = None
    request_timeout = normalize_timeout(timeout)

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=request_timeout, headers=DEFAULT_HEADERS)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(retry_backoff_seconds * attempt)

    raise RuntimeError(f"arXiv request failed after {max_retries} attempts: {last_error}") from last_error


def fetch_entries(
    query,
    start,
    max_results,
    *,
    title_only=False,
    timeout=DEFAULT_REQUEST_TIMEOUT,
    max_retries=DEFAULT_MAX_RETRIES,
    retry_backoff_seconds=DEFAULT_RETRY_BACKOFF_SECONDS,
):
    params = {
        "search_query": build_search_query(query, title_only=title_only),
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    response = request_with_retries(
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
    fetch_count,
    *,
    title_only=False,
    timeout=DEFAULT_REQUEST_TIMEOUT,
    max_retries=DEFAULT_MAX_RETRIES,
    retry_backoff_seconds=DEFAULT_RETRY_BACKOFF_SECONDS,
):
    capped_fetch_count = min(fetch_count, ARXIV_MAX_RESULTS_TOTAL)
    papers = []
    start = 0

    while start < capped_fetch_count:
        batch_size = min(ARXIV_MAX_RESULTS_PER_CALL, capped_fetch_count - start)
        if start > 0:
            time.sleep(ARXIV_RECOMMENDED_DELAY_SECONDS)

        entries = fetch_entries(
            query,
            start,
            batch_size,
            title_only=title_only,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )

        if not entries:
            break

        for entry in entries:
            paper = parse_entry(entry)
            if paper:
                papers.append(paper)

        if len(entries) < batch_size:
            break

        start += batch_size

    return papers


def apply_filters(papers, year_min, year_max, cite_min, author_query="", journal_query=""):
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
    *,
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
    arxiv_id = extract_arxiv_id(paper=paper)
    if not arxiv_id:
        raise ValueError("Paper is missing arXiv identifier")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / f"arxiv_{arxiv_id}.pdf"

    if final_path.exists() and final_path.stat().st_size > 0:
        return str(final_path)

    response = request_with_retries(
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


def main():
    args = parse_args()
    print(
        "fetching papers... "
        f'query="{args.query}", count={args.count}, years={args.year_min}-{args.year_max}, '
        f"max_pages={args.max_pages}"
    )

    try:
        papers = get_filtered_papers_exact_count(
            query=args.query,
            target_count=args.count,
            year_min=args.year_min,
            year_max=args.year_max,
            cite_min=args.cite_min,
            author_query=args.author,
            journal_query=args.journal,
            max_pages=args.max_pages,
            title_only=args.title_only,
            timeout=args.timeout,
            max_retries=args.retries,
            retry_backoff_seconds=args.retry_backoff,
        )
    except RuntimeError as exc:
        print(f"fetch failed: {exc}", file=sys.stderr)
        print(
            "Try again in a bit, or lower --count / --max-pages, "
            "or raise --timeout / --retries.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    print(f"fetched={len(papers)}")

    for index, paper in enumerate(papers, start=1):
        print(f"[{index}/{len(papers)}] {paper['title']}")
        print("    downloading...")
        try:
            pdf_path = download_pdf_for_paper(
                paper,
                args.output_dir,
                timeout=args.timeout,
                max_retries=args.retries,
                retry_backoff_seconds=args.retry_backoff,
            )
        except RuntimeError as exc:
            print(f"    failed -> {exc}", file=sys.stderr)
            continue
        print(f"    saved -> {pdf_path}")


if __name__ == "__main__":
    main()
