from back_end.constants import (
    BASE_URL as BASE,
    HEADERS,
    extract_date_from_url,
    parse_date_str,
    iter_dates,
    build_catchup_url,
)

import os
import argparse
import time
import re
import sys
from typing import List, Dict, Tuple, Optional
from urllib.parse import urljoin, urlparse
import json
import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import Callable, Optional
from openpyxl.workbook import Workbook


OUTPUT_DIR = "./crawl_data"


# ---------------------- Constants & Common Configuration ----------------------

SECTION_TAGS = ("h2", "h3", "h4")
SECTION_NEW_KEYS = ["new submissions"]
SECTION_CROSS_KEYS = ["cross submissions", "cross-lists", "cross lists", "cross listings"]

ARXIV_ID_RX = re.compile(r"(\d{4}\.\d{5,})(v\d+)?")

# ---------------------- Utility Functions ----------------------
def clean_text(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()

def get_soup(url: str, session: Optional[requests.Session] = None,
             max_retry: int = 3, sleep: float = 1.0) -> BeautifulSoup:
    sess = session or requests.Session()
    last_err = None
    for attempt in range(1, max_retry + 1):
        try:
            resp = sess.get(url, headers=HEADERS, timeout=20)
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "lxml")
            last_err = RuntimeError(f"HTTP {resp.status_code}")
        except Exception as e:
            last_err = e
        time.sleep(sleep * attempt)
    raise RuntimeError(f"Failed to GET {url}: {last_err}")

def is_ar5iv(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
        return "ar5iv.org" in host
    except Exception:
        return False

# ---------------------- List Page Parsing (Section → Entry Blocks) ----------------------
def _is_target_section(h_tag_text: str) -> Optional[str]:
    t = clean_text(h_tag_text).lower()
    if any(k in t for k in SECTION_NEW_KEYS):
        return "New submissions"
    if any(k in t for k in SECTION_CROSS_KEYS):
        return "Cross submissions"
    return None

def _iter_section_blocks(soup: BeautifulSoup):
    headers = [h for h in soup.find_all(SECTION_TAGS)]
    for idx, h in enumerate(headers):
        sec = _is_target_section(h.get_text())
        if not sec:
            continue
        block_elems = []
        sib = h.find_next_sibling()
        while sib and sib.name not in SECTION_TAGS:
            block_elems.append(sib)
            sib = sib.find_next_sibling()
        yield sec, block_elems

def _chunk_items(block_elems: List[BeautifulSoup]) -> List[List[BeautifulSoup]]:
    chunks, cur = [], []

    def is_entry_head(tag) -> bool:
        if not hasattr(tag, "get_text"):
            return False
        txt = clean_text(tag.get_text())
        return bool(re.search(r"arXiv:\s*\d{4}\.\d{5,}", txt))

    for el in block_elems:
        if is_entry_head(el):
            if cur:
                chunks.append(cur); cur = []
        cur.append(el)
    if cur:
        chunks.append(cur)
    return chunks

def _extract_from_chunk(section: str, chunk: List[BeautifulSoup]) -> Dict:
    """
    Extract metadata from a single entry block: arxiv_id, title, abstract, html_url, authors_listing.
    """
    arxiv_id, html_url, title, abstract = "", "", "", ""
    authors_listing: List[str] = []
    head_text_all = []

    # 1) Header: arXiv id + various links (pdf/html/other)
    for el in chunk:
        txt = clean_text(el.get_text())
        if txt:
            head_text_all.append(txt)
        for a in el.find_all("a", href=True):
            href = a["href"]
            a_txt = clean_text(a.get_text())
            m = ARXIV_ID_RX.search(href + " " + a_txt)
            if m:
                arxiv_id = m.group(1)
            if "html" in a_txt.lower() or "/html/" in href:
                if "/abs/" in href and "html" in a_txt.lower():
                    href = href.replace("/abs/", "/html/")
                html_url = urljoin(BASE, href)

    if not html_url and arxiv_id:
        html_url = urljoin(BASE, f"/html/{arxiv_id}")

    head_text = "  ".join(head_text_all)

    # 2) Authors shown on the listing page (used to more robustly delimit title)
    for el in chunk:
        links = [clean_text(a.get_text()) for a in el.find_all("a", href=True)]
        cand = [x for x in links if re.search(r"[A-Za-z\-\.]\s+[A-Za-z\-\.]", x)]
        if len(cand) >= 1:
            authors_listing = cand
            break

    if authors_listing:
        for author in authors_listing:
            author_pos = head_text.find(author)
            if author_pos > 0 and "Title:" in head_text:
                title_section = head_text[head_text.find("Title:") + 6:author_pos]
                title = clean_text(title_section)
                break

    if not title:
        m_title = re.search(r"Title:\s*(.+)", head_text)
        if m_title:
            title = clean_text(m_title.group(1))

    # 3) Abstract (take the first long paragraph in the block; skip title/subjects/comments etc.)
    paras = []
    for el in chunk:
        txt = clean_text(el.get_text())
        if not txt:
            continue
        if txt.startswith("Title:") or "Subjects:" in txt or "Comments:" in txt:
            continue
        if re.match(r"^\[\d+\]\s*$", txt):
            continue
        paras.append(txt)

    for p in paras:
        if len(p) > 120:
            abstract = p
            break

    return {
        "section": section,
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "html_url": html_url,
        "authors_listing": authors_listing,
    }

def crawl_catchup(url: str) -> Tuple[List[Dict], Dict]:
    sess = requests.Session()
    soup = get_soup(url, sess)

    items: List[Dict] = []
    found_sections = []

    for section, block in _iter_section_blocks(soup):
        if section not in ("New submissions", "Cross submissions"):
            continue
        found_sections.append(section)
        chunks = _chunk_items(block)
        for ch in chunks:
            entry = _extract_from_chunk(section, ch)
            if entry.get("arxiv_id") and entry.get("title"):
                items.append(entry)

    if not items:
        raise RuntimeError(
            "No entries were extracted from the page. "
            "Please confirm the URL is a catch-up page (preferably including ?abs=True) "
            "and that the sections for the target date contain entries."
        )

    meta = {
        "source_url": url,
        "total_entries": len(items),
        "note": "sections parsed: " + ", ".join(sorted(set(found_sections))),
    }
    return items, meta

# ---------------------- Detail Page Extraction ----------------------
def extract_from_abs_meta(soup: BeautifulSoup) -> List[str]:
    authors = [m.get("content", "").strip()
               for m in soup.find_all("meta", attrs={"name": "citation_author"})]
    return [a for a in authors if a]

def extract_abstract_from_abs_or_html(soup: BeautifulSoup, prefer_ar5iv: bool = False) -> str:
    meta_abs = soup.find("meta", attrs={"name": "citation_abstract"})
    if meta_abs and meta_abs.get("content"):
        return clean_text(meta_abs["content"])

    if prefer_ar5iv:
        sec = soup.find(id="abstract") or soup.find("section", id="abstract")
        if sec:
            ps = [clean_text(p.get_text()) for p in sec.find_all("p")]
            ps = [p for p in ps if p]
            if ps:
                return " ".join(ps)
        for tag in soup.find_all(["h2", "h3", "h4"]):
            if clean_text(tag.get_text()).lower() == "abstract":
                text_parts, cursor, steps = [], tag.find_next_sibling(), 0
                while cursor and steps < 10:
                    if cursor.name in ["p", "div", "blockquote"]:
                        t = clean_text(cursor.get_text())
                        if t:
                            text_parts.append(t)
                    if cursor.name in ["h2", "h3", "h4"]:
                        break
                    cursor, steps = cursor.find_next_sibling(), steps + 1
                if text_parts:
                    return " ".join(text_parts)
    else:
        bq = soup.find("blockquote", class_=lambda c: c and "abstract" in c)
        if bq:
            t = clean_text(bq.get_text())
            t = re.sub(r"^\s*abstract\s*:\s*", "", t, flags=re.I)
            if t:
                return t
    return ""

def extract_authors_from_ar5iv_html(soup: BeautifulSoup) -> List[str]:
    authors: List[str] = []
    h1 = soup.find("h1")
    candidate_blocks = []
    if h1:
        cursor, steps = h1.find_next_sibling(), 0
        while cursor and steps < 15:
            candidate_blocks.append(cursor)
            cursor, steps = cursor.find_next_sibling(), steps + 1

    def _try_extract_authors(node) -> List[str]:
        if not node:
            return []
        links = [clean_text(a.get_text()) for a in node.find_all("a")]
        link_names = [n for n in links if re.search(r"[A-Za-zÀ-ÿ\.\-]+\s+[A-Za-zÀ-ÿ\.\-]+", n)]
        if link_names:
            return link_names
        text = clean_text(node.get_text())
        if text:
            parts = [clean_text(x) for x in re.split(r"[;,]| and ", text)]
            cand = [p for p in parts if re.search(r"[A-Za-zÀ-ÿ\.\-]+\s+[A-Za-zÀ-ÿ\.\-]+", p)]
            cand = [p for p in cand if 2 <= len(p.split()) <= 6 and len(p) <= 80]
            if cand:
                return cand
        return []

    for blk in candidate_blocks:
        if not authors:
            authors = _try_extract_authors(blk)

    seen, out = set(), []
    for s in authors:
        if s not in seen:
            out.append(s); seen.add(s)
    return out

# ---------------------- Progress Bar ----------------------
def _format_time(seconds: float) -> str:
    seconds = int(max(0, seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def _progress_bar(i: int, n: int, start_ts: float) -> None:
    width = 36
    ratio = 0 if n == 0 else min(1.0, (i + 1) / n)
    filled = int(width * ratio)
    bar = "█" * filled + "·" * (width - filled)
    elapsed = time.time() - start_ts
    speed = 0 if elapsed == 0 else (i + 1) / elapsed
    eta = 0 if speed == 0 else (n - (i + 1)) / speed
    msg = (
        f"\r[Detail crawl] |{bar}| {ratio:6.2%}  {i+1}/{n}  "
        f"Elapsed { _format_time(elapsed) }  ETA { _format_time(eta) }"
    )
    sys.stdout.write(msg); sys.stdout.flush()
    if i + 1 == n:
        sys.stdout.write("\n"); sys.stdout.flush()

# ---------------------- Detail Enrichment (with Rate Limit & Progress Bar) ----------------------
def enrich_with_detail(
    items: List[Dict],
    session: Optional[requests.Session] = None,
    max_per_minute: int = 30,
    progress_cb: Optional[Callable[[int, int, float, float], None]] = None,
) -> None:
    if not items:
        return

    sess = session or requests.Session()
    interval = max(2.0 / max_per_minute, 0.05)
    start_ts = time.time()
    total = len(items)

    for idx, it in enumerate(items):
        arxiv_id = it.get("arxiv_id") or ""
        html_url = it.get("html_url") or ""
        authors = it.get("authors", []) or []
        abstract = it.get("abstract", "") or ""

        abs_url = urljoin(BASE, f"/abs/{arxiv_id}") if arxiv_id else ""
        if abs_url:
            try:
                soup_abs = get_soup(abs_url, sess)
                a1 = extract_from_abs_meta(soup_abs)
                if a1:
                    authors = a1
                abs1 = extract_abstract_from_abs_or_html(soup_abs, prefer_ar5iv=False)
                if abs1:
                    abstract = abs1
            except Exception:
                pass

        time.sleep(interval)

        target = html_url or (urljoin(BASE, f"/html/{arxiv_id}") if arxiv_id else "")
        if target:
            try:
                soup_html = get_soup(target, sess)
                if is_ar5iv(target):
                    a2 = extract_authors_from_ar5iv_html(soup_html)
                    if a2 and not authors:
                        authors = a2
                    abs2 = extract_abstract_from_abs_or_html(soup_html, prefer_ar5iv=True)
                    if abs2 and (not abstract or len(abstract) < 50):
                        abstract = abs2
                else:
                    if not authors:
                        a3 = extract_from_abs_meta(soup_html)
                        if a3:
                            authors = a3
                    if (not abstract) or len(abstract) < 50:
                        abs3 = extract_abstract_from_abs_or_html(soup_html, prefer_ar5iv=False)
                        if abs3:
                            abstract = abs3
            except Exception:
                pass

        if not authors:
            authors = it.get("authors_listing", []) or []

        it["authors"] = authors
        it["abstract"] = abstract

        done = idx + 1
        elapsed = time.time() - start_ts
        speed = 0 if elapsed == 0 else done / elapsed
        eta = 0 if speed == 0 else (total - done) / speed

        if progress_cb is not None:
            progress_cb(done, total, elapsed, eta)
        else:
            _progress_bar(idx, total, start_ts)

        time.sleep(interval)
# ---------------------- Export ----------------------
def export_excel(items: List[Dict], meta: Dict, outfile: str) -> None:
    df = pd.DataFrame(items, columns=[
        "section", "arxiv_id", "title", "abstract", "html_url",
        "authors", "authors_listing"
    ])
    def join_if_list(x):
        if isinstance(x, list):
            return "; ".join([str(i) for i in x])
        return x
    for col in ["authors", "authors_listing"]:
        df[col] = df[col].apply(join_if_list)

    meta_df = pd.DataFrame([meta])

    with pd.ExcelWriter(outfile, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="papers", index=False)
        meta_df.to_excel(writer, sheet_name="metadata", index=False)

def export_json(items: List[Dict], meta: Dict, outfile: str, jsonl: bool = False) -> None:
    if jsonl:
        with open(outfile, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        return
    payload = {"metadata": meta, "papers": items}
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def crawl():
    parser = argparse.ArgumentParser(
        description="Crawl arXiv catch-up pages for one or multiple dates."
    )
    parser.add_argument("--url", help="Single catch-up URL")

    parser.add_argument(
        "--urls",
        action="append",
        help="Multiple URLs: comma-separated or repeated arguments",
    )
    parser.add_argument("--date", help="Single date YYYY-MM-DD")
    parser.add_argument(
        "--dates",
        action="append",
        help="Multiple dates: comma-separated or repeated arguments",
    )
    parser.add_argument(
        "--range",
        nargs=2,
        metavar=("START", "END"),
        help="Date range (inclusive), e.g. 2025-09-01 2025-09-30",
    )
    parser.add_argument(
        "--subject",
        default="cs.CV",
        help="Subject (default cs.CV), used to construct URLs from dates",
    )
    parser.add_argument(
        "--out",
        help="Output filename for a single target; ignored when multiple targets",
    )
    parser.add_argument(
        "--no-detail",
        action="store_true",
        help="Skip detail-page requests (faster)",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=30,
        help="Max request rate for detail pages (per minute)",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Export as JSON Lines (papers only)",
    )
    parser.add_argument(
        "--date-file",
        help=(
            "Read dates from file, one per line or comma-separated; "
            "if omitted, dates.txt will be used by default"
        ),
    )

    args = parser.parse_args()

    subject = args.subject.strip()
    url_list: List[str] = []
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # (1) Multiple URLs
    if args.urls:
        for group in args.urls:
            url_list.extend([u.strip() for u in group.split(",") if u.strip()])

    # (2) Single URL (compatibility)
    if args.url:
        url_list.append(args.url.strip())

    # (3) Multiple dates
    date_list: List[str] = []
    if args.dates:
        for group in args.dates:
            date_list.extend([parse_date_str(s) for s in group.split(",") if s.strip()])

    # (4) Single date
    if args.date:
        date_list.append(parse_date_str(args.date))

    # (5) Date range
    if args.range:
        start, end = args.range
        date_list.extend(list(iter_dates(start, end)))

    # (6) Read dates from file (only if no dates and no URLs are explicitly specified)
    if not date_list and not args.url and not args.urls:
        date_file = args.date_file or "crawl_data/dates.txt"
        file_dates = load_dates_from_file(date_file)
        if file_dates:
            print(f"[INFO] Read {len(file_dates)} dates from file {date_file}.")
            date_list.extend(file_dates)
        else:
            # File missing or no valid dates; do not exit immediately, let url_list check handle it
            print(f"[WARN] No valid dates found in file {date_file}.")

    # Build URLs from dates and merge
    if date_list:
        for d in date_list:
            url_list.append(build_catchup_url(subject, d, with_abs=True))

    # De-duplicate while preserving order
    seen = set(); merged = []
    for u in url_list:
        if u not in seen:
            seen.add(u); merged.append(u)
    url_list = merged

    if not url_list:
        raise SystemExit(
            "No crawl targets specified. Please provide at least one of "
            "--urls/--url/--date/--dates/--range."
        )

    multi_mode = len(url_list) > 1

    for u in url_list:
        # Extract date for metadata and filename
        try:
            catchup_date = extract_date_from_url(u)
        except Exception:
            catchup_date = "unknown"

        items, meta = crawl_catchup(u)
        meta["catchup_date"] = catchup_date
        meta["subject"] = subject

        if not args.no_detail:
            enrich_with_detail(items, max_per_minute=args.rate)

        # Output naming: multiple targets → per-date filenames; single target → respect --out if set
        if multi_mode or not args.out:
            base_name = f"arxiv_{catchup_date}.json"
        else:
            base_name = args.out  

        if os.path.isabs(base_name) or os.path.dirname(base_name):
            out_json = base_name      
        else:
            out_json = os.path.join(OUTPUT_DIR, base_name)

        export_json(items, meta, out_json, jsonl=args.jsonl)
        export_excel(items, meta, out_json.replace(".json", ".xlsx"))

        print(
            f"[OK] {catchup_date} → {out_json} | {len(items)} items | "
            f"{meta.get('note')} | {u}"
        )

def load_dates_from_file(path: str) -> List[str]:
    """
    Read dates from the specified file and return them as normalized date strings (YYYY-MM-DD).

    Supported formats:
      - One date per line: 2025-09-30
      - Multiple dates on one line: 2025-09-28,2025-09-29
      - Empty lines and lines starting with # are ignored.
    """
    dates: List[str] = []
    if not os.path.exists(path):
        return dates

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for part in line.split(","):
                part = part.strip()
                if not part:
                    continue
                # Reuse existing parse_date_str for normalization and validation
                dates.append(parse_date_str(part))
    return dates
