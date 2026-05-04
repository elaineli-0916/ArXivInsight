from __future__ import annotations
import re
from datetime import datetime, timedelta
from typing import Final, Iterable

BASE_URL: Final[str] = "https://arxiv.org"
USER_AGENT: Final[str] = (
    "Mozilla/5.0 (compatible; CourseProjectBot/1.0; +https://example.edu)"
)
HEADERS: Final[dict] = {"User-Agent": USER_AGENT}
DATE_FMT: Final[str] = "%Y-%m-%d"
DATE_RX = re.compile(r"/(\d{4}-\d{2}-\d{2})(?:[/?#]|$)")


def parse_date_str(s: str) -> str:
    """
    Normalize input to YYYY-MM-DD.
    Supports YYYY/MM/DD and YYYYMMDD.
    Raises ValueError for invalid formats.
    """
    s = s.strip().replace("/", "-")
    if re.fullmatch(r"\d{8}", s):
        s = f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    dt = datetime.strptime(s, DATE_FMT)
    return dt.strftime(DATE_FMT)


def extract_date_from_url(url: str) -> str:
    """
    Extract YYYY-MM-DD date from an arXiv catch-up URL.
    Raises ValueError if no valid date is found.
    """
    m = DATE_RX.search(url)
    if not m:
        raise ValueError(f"Failed to extract date from URL: {url}")
    return parse_date_str(m.group(1))


def iter_dates(start: str, end: str) -> Iterable[str]:
    """
    Generate a date sequence [start, end] (inclusive), in YYYY-MM-DD format.
    Raises ValueError if start is later than end.
    """
    s = datetime.strptime(parse_date_str(start), DATE_FMT)
    e = datetime.strptime(parse_date_str(end), DATE_FMT)
    if s > e:
        raise ValueError(f"Start date is later than end date: {start} > {end}")
    d = s
    one = timedelta(days=1)
    while d <= e:
        yield d.strftime(DATE_FMT)
        d += one


def build_catchup_url(subject: str, date_str: str, with_abs: bool = True) -> str:
    """
    Build an arXiv catch-up URL for a given subject and date.
    Example: /catchup/cs.CV/2025-11-20?abs=True
    """
    d = parse_date_str(date_str)
    tail = "?abs=True" if with_abs else ""
    return f"{BASE_URL}/catchup/{subject.strip()}/{d}{tail}"
