from __future__ import annotations

import asyncio
import io
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import unescape
from typing import Iterable
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI
from pydantic import BaseModel, Field
from pypdf import PdfReader


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
)
TIMEOUT = httpx.Timeout(15.0, connect=10.0)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?1[\s.\-]?)?(?:\(?(\d{3})\)?[\s.\-]?(\d{3})[\s.\-]?(\d{4}))(?!\d)"
)
CITY_STATE_ZIP_RE = re.compile(
    r"\b([A-Z][A-Za-z .'-]+),\s*([A-Z]{2})\s+(\d{5}(?:-\d{4})?)\b"
)
ADDRESS_ABBREVIATIONS = {
    " st ": " street ",
    " rd ": " road ",
    " ave ": " avenue ",
    " blvd ": " boulevard ",
    " dr ": " drive ",
    " ln ": " lane ",
    " ct ": " court ",
    " cir ": " circle ",
    " pkwy ": " parkway ",
    " hwy ": " highway ",
    " ste ": " suite ",
    " fl ": " floor ",
    " rm ": " room ",
    " bldg ": " building ",
    " mt ": " mount ",
}
COMMON_FIRM_WORDS = {
    "llp",
    "llc",
    "pllc",
    "pc",
    "p c",
    "law",
    "office",
    "offices",
    "group",
    "firm",
    "and",
    "of",
    "the",
}


class AttorneyInput(BaseModel):
    name: str
    firm: str


class ADTRow(BaseModel):
    name: str
    firm: str
    street_address: str
    city_state_zip: str
    phone: str
    email: str
    secondary_source_urls: list[str]
    evidence_tier: str = "T0"
    retrieval_attempts: int = 0


class QCRow(BaseModel):
    name: str
    firm: str
    bio_url: str
    bio_http: str
    final_url: str
    freshness_certification: str
    verification_status: str
    qc_notes: str
    same_turn_web_hits: int = 0
    retrieval_attempts: int = 0


class FindRequest(BaseModel):
    attorneys: list[AttorneyInput]
    max_attempts_per_attorney: int = 3
    tier1_only_first: bool = True
    blank_only_fill_phase: bool = True


class FindResponse(BaseModel):
    rows: list[ADTRow]
    retrieved_at_utc: str


class VerifyRequest(BaseModel):
    rows: list[ADTRow]
    max_attempts_per_attorney: int = 3
    require_same_turn_web_hit: bool = True


class VerifyResponse(BaseModel):
    rows: list[QCRow]
    verified_at_utc: str


@dataclass
class SearchResult:
    url: str
    title: str = ""
    snippet: str = ""


@dataclass
class FetchResult:
    url: str
    final_url: str
    status_code: int
    content_type: str
    text: str
    title: str = ""
    links: list[str] = field(default_factory=list)


@dataclass
class OfficeCandidate:
    street_address: str
    city_state_zip: str
    phone: str
    city: str
    url: str


@dataclass
class Resolution:
    adt_row: ADTRow
    bio_url: str = ""
    bio_http: str = ""
    final_url: str = ""
    freshness_tier: str = "T0"
    notes: list[str] = field(default_factory=list)
    web_hits: int = 0


app = FastAPI(title="Appearance Research Actions", version="1.0.0")


@app.get("/")
async def root() -> dict[str, str]:
    return {"ok": "true", "service": "appearance-actions"}


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/find", response_model=FindResponse)
async def find_attorneys(request: FindRequest) -> FindResponse:
    async with httpx.AsyncClient(timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
        rows = []
        for attorney in request.attorneys:
            resolution = await resolve_attorney(
                client=client,
                name=attorney.name,
                firm=attorney.firm,
                max_attempts=request.max_attempts_per_attorney,
                bio_hint="",
            )
            rows.append(resolution.adt_row)
    return FindResponse(rows=rows, retrieved_at_utc=utc_now())


@app.post("/verify", response_model=VerifyResponse)
async def verify_attorneys(request: VerifyRequest) -> VerifyResponse:
    async with httpx.AsyncClient(timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
        rows = []
        for row in request.rows:
            bio_hint = row.secondary_source_urls[0] if row.secondary_source_urls else ""
            resolution = await resolve_attorney(
                client=client,
                name=row.name,
                firm=row.firm,
                max_attempts=request.max_attempts_per_attorney,
                bio_hint=bio_hint,
            )
            qc_row = build_qc_row(seed=row, resolution=resolution)
            rows.append(qc_row)
    return VerifyResponse(rows=rows, verified_at_utc=utc_now())


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


async def resolve_attorney(
    *,
    client: httpx.AsyncClient,
    name: str,
    firm: str,
    max_attempts: int,
    bio_hint: str,
) -> Resolution:
    source_urls: list[str] = []
    notes: list[str] = []
    attempts = 0
    official_domain = ""
    bio_fetch: FetchResult | None = None
    location_fetches: list[FetchResult] = []
    web_hits = 0

    if bio_hint:
        attempts += 1
        bio_fetch = await fetch_url(client, bio_hint)
        if bio_fetch and bio_fetch.status_code < 400 and page_matches_attorney(bio_fetch, name):
            web_hits += 1
            official_domain = host_for_url(bio_fetch.final_url)
            source_urls.append(bio_fetch.final_url)
            notes.append("T1 bio match")

    if not bio_fetch:
        for query in build_bio_queries(name, firm)[:max_attempts]:
            attempts += 1
            search_results = await search_web(client, query)
            candidate = await pick_bio_candidate(client, search_results, name, firm)
            if candidate:
                bio_fetch = candidate
                web_hits += 1
                official_domain = host_for_url(candidate.final_url)
                source_urls.append(candidate.final_url)
                notes.append("T1 bio match")
                break

    if not official_domain and source_urls:
        official_domain = host_for_url(source_urls[0])

    if official_domain:
        location_fetches = await find_location_pages(client, firm, official_domain, bio_fetch)
        web_hits += len([page for page in location_fetches if page.status_code < 400])

    offices = [office for fetch in location_fetches for office in extract_office_candidates(fetch)]
    office_cities = sorted({office.city for office in offices if office.city})

    same_firm_pages: list[FetchResult] = []
    if official_domain:
        same_firm_pages = await find_same_firm_pages(client, name, official_domain)
        web_hits += len([page for page in same_firm_pages if page.status_code < 400])

    secondary_pages: list[FetchResult] = []
    if not office_cities or not any(extract_literal_email(page, name, official_domain) for page in [bio_fetch, *same_firm_pages] if page):
        secondary_pages = await find_secondary_pages(client, name, firm)
        web_hits += len([page for page in secondary_pages if page.status_code < 400])

    city_clue = ""
    clue_url = ""
    for page in filter(None, [bio_fetch, *same_firm_pages, *secondary_pages]):
        candidate_city = detect_city_clue(page, office_cities)
        if candidate_city:
            city_clue = candidate_city
            clue_url = page.final_url
            break

    matched_office = match_office(offices, city_clue)
    if matched_office and matched_office.url not in source_urls:
        source_urls.append(matched_office.url)
    if clue_url and clue_url not in source_urls and matched_office:
        source_urls.append(clue_url)
        notes.extend(tag_bridge(clue_url, official_domain))

    full_name = extract_display_name(bio_fetch, name) if bio_fetch else ""
    if not full_name and bio_fetch:
        full_name = name

    resolved_firm = firm if bio_fetch else ""

    email = ""
    email_tier = "T0"
    for tier, pages in (
        ("T1", [bio_fetch] if bio_fetch else []),
        ("T2", same_firm_pages),
        ("T3", secondary_pages),
    ):
        for page in pages:
            found = extract_literal_email(page, name, official_domain)
            if found:
                email = found
                email_tier = tier
                if page.final_url not in source_urls:
                    source_urls.append(page.final_url)
                if tier == "T2":
                    notes.append("same-firm file email")
                if tier == "T3":
                    notes.append("secondary email")
                break
        if email:
            break

    phone = ""
    phone_tier = "T0"
    if bio_fetch:
        phone = extract_direct_phone(bio_fetch, name)
        if phone:
            phone_tier = "T1"
    if not phone:
        for page in same_firm_pages:
            phone = extract_direct_phone(page, name)
            if phone:
                phone_tier = "T2"
                if page.final_url not in source_urls:
                    source_urls.append(page.final_url)
                break
    if not phone:
        for page in secondary_pages:
            phone = extract_direct_phone(page, name)
            if phone:
                phone_tier = "T3"
                if page.final_url not in source_urls:
                    source_urls.append(page.final_url)
                notes.append("direct phone match")
                break
    if not phone and matched_office and matched_office.phone:
        phone = normalize_phone(matched_office.phone)
        phone_tier = "T1"

    if phone and not matched_office and phone_tier == "T1" and bio_fetch and is_generic_firm_page(bio_fetch):
        phone = ""
        notes.append("headquarters rejected")

    street_address = ""
    city_state_zip = ""
    evidence_tier = "T0"
    if matched_office:
        street_address = expand_address_words(matched_office.street_address)
        city_state_zip = trim_zip(matched_office.city_state_zip)
        evidence_tier = "T1"
        notes.append("city to locations match")
    elif city_clue:
        notes.append("headquarters rejected")

    if email and evidence_tier == "T0":
        evidence_tier = email_tier
    if phone and evidence_tier == "T0":
        evidence_tier = phone_tier
    if bio_fetch and evidence_tier == "T0":
        evidence_tier = "T1"

    adt_row = ADTRow(
        name=full_name,
        firm=resolved_firm,
        street_address=street_address,
        city_state_zip=city_state_zip,
        phone=normalize_phone(phone) if phone else "",
        email=email,
        secondary_source_urls=dedupe_urls(source_urls),
        evidence_tier=evidence_tier,
        retrieval_attempts=attempts,
    )
    return Resolution(
        adt_row=adt_row,
        bio_url=bio_fetch.final_url if bio_fetch else "",
        bio_http=str(bio_fetch.status_code) if bio_fetch else "",
        final_url=bio_fetch.final_url if bio_fetch else "",
        freshness_tier=evidence_tier,
        notes=dedupe_strings(notes),
        web_hits=web_hits,
    )


def build_qc_row(seed: ADTRow, resolution: Resolution) -> QCRow:
    verification_status = "PASS"
    notes = list(resolution.notes)

    if not resolution.bio_url:
        verification_status = "FAIL"
        notes.append("bio missing")
    if seed.email and resolution.adt_row.email != seed.email:
        verification_status = "FAIL"
    if seed.phone and resolution.adt_row.phone != normalize_phone(seed.phone):
        verification_status = "FAIL"

    freshness_state = "PASS" if resolution.web_hits > 0 else "FAIL"
    freshness = (
        f"{freshness_state} | {utc_now()} | {resolution.freshness_tier} | {resolution.bio_http or ''}"
    )
    return QCRow(
        name=resolution.adt_row.name or seed.name,
        firm=resolution.adt_row.firm or seed.firm,
        bio_url=resolution.bio_url,
        bio_http=resolution.bio_http,
        final_url=resolution.final_url,
        freshness_certification=freshness,
        verification_status=verification_status,
        qc_notes="; ".join(dedupe_strings(notes)),
        same_turn_web_hits=resolution.web_hits,
        retrieval_attempts=resolution.adt_row.retrieval_attempts,
    )


async def pick_bio_candidate(
    client: httpx.AsyncClient, results: list[SearchResult], name: str, firm: str
) -> FetchResult | None:
    candidates: list[tuple[int, FetchResult]] = []
    for result in results[:8]:
        if looks_like_file(result.url):
            continue
        fetch = await fetch_url(client, result.url)
        if not fetch or fetch.status_code >= 400:
            continue
        score = score_bio_page(fetch, name, firm)
        if score > 0:
            candidates.append((score, fetch))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


async def find_location_pages(
    client: httpx.AsyncClient, firm: str, official_domain: str, bio_fetch: FetchResult | None
) -> list[FetchResult]:
    candidates: list[str] = []
    if bio_fetch:
        for link in bio_fetch.links:
            if is_same_domain(link, official_domain) and any(
                token in link.lower() for token in ("location", "office", "contact")
            ):
                candidates.append(link)
    for query in (
        f'site:{official_domain} "{firm}" locations',
        f'site:{official_domain} "{firm}" offices',
        f'site:{official_domain} "{firm}" contact',
    ):
        results = await search_web(client, query)
        candidates.extend(result.url for result in results[:5])

    fetched: list[FetchResult] = []
    seen: set[str] = set()
    for url in candidates:
        if url in seen:
            continue
        seen.add(url)
        fetch = await fetch_url(client, url)
        if fetch and fetch.status_code < 400 and is_same_domain(fetch.final_url, official_domain):
            fetched.append(fetch)
    return fetched


async def find_same_firm_pages(
    client: httpx.AsyncClient, name: str, official_domain: str
) -> list[FetchResult]:
    queries = [
        f'site:{official_domain} "{name}" email',
        f'site:{official_domain} "{name}" pdf',
        f'site:{official_domain} "{name}" phone',
    ]
    pages: list[FetchResult] = []
    seen: set[str] = set()
    for query in queries:
        for result in (await search_web(client, query))[:4]:
            if result.url in seen:
                continue
            seen.add(result.url)
            fetch = await fetch_url(client, result.url)
            if fetch and fetch.status_code < 400 and is_same_domain(fetch.final_url, official_domain):
                pages.append(fetch)
    return pages


async def find_secondary_pages(
    client: httpx.AsyncClient, name: str, firm: str
) -> list[FetchResult]:
    queries = [
        f'"{name}" "{firm}" email',
        f'"{name}" "{firm}" phone',
        f'"{name}" "{firm}" office',
        f'"{name}" "{firm}" city',
    ]
    pages: list[FetchResult] = []
    seen: set[str] = set()
    for query in queries:
        for result in (await search_web(client, query))[:4]:
            if result.url in seen:
                continue
            seen.add(result.url)
            fetch = await fetch_url(client, result.url)
            if fetch and fetch.status_code < 400:
                pages.append(fetch)
    return pages


def build_bio_queries(name: str, firm: str) -> list[str]:
    return [
        f'"{name}" "{firm}" attorney',
        f'"{name}" "{firm}" lawyer',
        f'"{name}" "{firm}" bio',
        f'"{name}" "{firm}" profile',
        f'"{name}" "{firm}" people',
    ]


async def search_web(client: httpx.AsyncClient, query: str) -> list[SearchResult]:
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    try:
        response = await client.get(url)
        response.raise_for_status()
    except httpx.HTTPError:
        return []

    soup = BeautifulSoup(response.text, "lxml")
    results: list[SearchResult] = []
    for anchor in soup.select("a.result__a, a[href]"):
        href = anchor.get("href") or ""
        url_value = decode_search_result_url(href)
        if not url_value.startswith("http"):
            continue
        title = clean_text(anchor.get_text(" ", strip=True))
        snippet_node = anchor.find_parent("div", class_="result")
        snippet = ""
        if snippet_node:
            snippet = clean_text(snippet_node.get_text(" ", strip=True))
        if url_value not in {item.url for item in results}:
            results.append(SearchResult(url=url_value, title=title, snippet=snippet))
        if len(results) >= 10:
            break
    return results


def decode_search_result_url(href: str) -> str:
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc:
        query = parse_qs(parsed.query)
        if "uddg" in query:
            return unquote(query["uddg"][0])
    return href


async def fetch_url(client: httpx.AsyncClient, url: str) -> FetchResult | None:
    try:
        response = await client.get(url, follow_redirects=True)
    except httpx.HTTPError:
        return None

    final_url = str(response.url)
    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    content = response.content

    text = ""
    title = ""
    links: list[str] = []
    if "html" in content_type or not content_type:
        soup = BeautifulSoup(content, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        title = clean_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
        text = clean_text(soup.get_text("\n", strip=True))
        links = [urljoin(final_url, a.get("href")) for a in soup.select("a[href]")]
    elif "pdf" in content_type or final_url.lower().endswith(".pdf"):
        text = extract_pdf_text(content)
    elif final_url.lower().endswith(".docx"):
        text = extract_docx_text(content)
    elif final_url.lower().endswith(".vcf"):
        text = extract_vcf_text(content)
    else:
        text = extract_plain_text(content)

    return FetchResult(
        url=url,
        final_url=final_url,
        status_code=response.status_code,
        content_type=content_type,
        text=text,
        title=title,
        links=links,
    )


def extract_pdf_text(content: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(content))
        return clean_text("\n".join(page.extract_text() or "" for page in reader.pages))
    except Exception:
        return ""


def extract_docx_text(content: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            data = archive.read("word/document.xml").decode("utf-8", "ignore")
        text = re.sub(r"<[^>]+>", " ", data)
        return clean_text(unescape(text))
    except Exception:
        return ""


def extract_vcf_text(content: bytes) -> str:
    return clean_text(content.decode("utf-8", "ignore"))


def extract_plain_text(content: bytes) -> str:
    return clean_text(content.decode("utf-8", "ignore"))


def score_bio_page(fetch: FetchResult, name: str, firm: str) -> int:
    haystack = " ".join([fetch.final_url, fetch.title, fetch.text[:3000]]).lower()
    score = 0
    if page_matches_attorney(fetch, name):
        score += 8
    if any(token in fetch.final_url.lower() for token in ("bio", "profile", "attorney", "people", "lawyer", "professionals")):
        score += 4
    if firm_key(firm) & set(tokenize(haystack)):
        score += 3
    if any(token in fetch.final_url.lower() for token in ("pdf", "doc", "vcf")):
        score -= 5
    return score


def page_matches_attorney(fetch: FetchResult, name: str) -> bool:
    tokens = [token for token in tokenize(name) if token not in {"esq", "esquire"}]
    if len(tokens) < 2:
        return False
    haystack = f"{fetch.title} {fetch.text[:4000]}".lower()
    return tokens[0] in haystack and tokens[-1] in haystack


def extract_display_name(fetch: FetchResult | None, fallback: str) -> str:
    if not fetch:
        return ""
    for line in lines(fetch.text[:3000]):
        if surname(fallback) in line.lower() and len(line.split()) <= 8:
            return line.strip()
    if fetch.title and surname(fallback) in fetch.title.lower():
        title = fetch.title.split("|")[0].split("-")[0].strip()
        if title:
            return title
    return fallback


def extract_literal_email(fetch: FetchResult, name: str, official_domain: str) -> str:
    text_lines = lines(fetch.text)
    surname_token = surname(name)
    for index, line in enumerate(text_lines):
        if surname_token and surname_token not in line.lower():
            continue
        nearby = " ".join(text_lines[max(0, index - 2) : index + 3])
        emails = EMAIL_RE.findall(nearby)
        if emails:
            return pick_best_email(emails, official_domain)

    if page_matches_attorney(fetch, name):
        emails = EMAIL_RE.findall(fetch.text)
        if len(emails) == 1:
            return pick_best_email(emails, official_domain)
    return ""


def pick_best_email(emails: Iterable[str], official_domain: str) -> str:
    normalized = dedupe_strings(email.lower() for email in emails)
    if official_domain:
        domain_matches = [email for email in normalized if email.split("@")[-1].endswith(official_domain)]
        if domain_matches:
            return domain_matches[0]
    return normalized[0] if normalized else ""


def extract_direct_phone(fetch: FetchResult, name: str) -> str:
    text_lines = lines(fetch.text)
    surname_token = surname(name)
    for index, line in enumerate(text_lines):
        if surname_token and surname_token not in line.lower():
            continue
        nearby = " ".join(text_lines[max(0, index - 2) : index + 3])
        match = PHONE_RE.search(nearby)
        if match:
            return normalize_phone(match.group(0))
    if page_matches_attorney(fetch, name):
        all_matches = PHONE_RE.findall(fetch.text)
        if len(all_matches) == 1:
            return normalize_phone("".join(all_matches[0]))
    return ""


def extract_office_candidates(fetch: FetchResult) -> list[OfficeCandidate]:
    raw_lines = lines(fetch.text)
    candidates: list[OfficeCandidate] = []
    for index, line in enumerate(raw_lines):
        match = CITY_STATE_ZIP_RE.search(line)
        if not match:
            continue
        city = clean_text(match.group(1))
        state = match.group(2)
        zip_code = trim_zip(match.group(3))
        street = ""
        phone = ""
        for prior in reversed(raw_lines[max(0, index - 2) : index]):
            if re.search(r"\d", prior) and any(
                token in prior.lower()
                for token in ("street", "st ", "avenue", "ave", "road", "rd", "suite", "boulevard", "drive", "lane", "court", "floor")
            ):
                street = expand_address_words(prior)
                break
        for later in raw_lines[index : index + 4]:
            phone_match = PHONE_RE.search(later)
            if phone_match:
                phone = normalize_phone(phone_match.group(0))
                break
        if street:
            candidates.append(
                OfficeCandidate(
                    street_address=street,
                    city_state_zip=f"{city}, {state} {zip_code}",
                    phone=phone,
                    city=city.lower(),
                    url=fetch.final_url,
                )
            )
    return dedupe_offices(candidates)


def detect_city_clue(fetch: FetchResult, known_cities: list[str]) -> str:
    lower_text = fetch.text.lower()
    for city in known_cities:
        if city and city.lower() in lower_text:
            return city

    for line in lines(fetch.text):
        match = CITY_STATE_ZIP_RE.search(line)
        if match:
            return clean_text(match.group(1))
    return ""


def match_office(offices: list[OfficeCandidate], city_clue: str) -> OfficeCandidate | None:
    city_key = city_clue.lower().strip()
    for office in offices:
        if office.city == city_key:
            return office
    return None


def tag_bridge(clue_url: str, official_domain: str) -> list[str]:
    if official_domain and official_domain not in host_for_url(clue_url):
        return ["tier bridge used"]
    return []


def host_for_url(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def is_same_domain(url: str, official_domain: str) -> bool:
    host = host_for_url(url)
    return host == official_domain or host.endswith(f".{official_domain}")


def looks_like_file(url: str) -> bool:
    lower = url.lower()
    return any(lower.endswith(ext) for ext in (".pdf", ".doc", ".docx", ".vcf"))


def is_generic_firm_page(fetch: FetchResult) -> bool:
    lower = f"{fetch.final_url} {fetch.title}".lower()
    return any(token in lower for token in ("contact", "locations", "office"))


def firm_key(firm: str) -> set[str]:
    return {token for token in tokenize(firm) if token not in COMMON_FIRM_WORDS}


def tokenize(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def surname(name: str) -> str:
    tokens = tokenize(name)
    return tokens[-1] if tokens else ""


def lines(text: str) -> list[str]:
    return [clean_text(line) for line in text.splitlines() if clean_text(line)]


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def expand_address_words(value: str) -> str:
    text = f" {value.strip()} "
    for short, full in ADDRESS_ABBREVIATIONS.items():
        text = re.sub(short, full, text, flags=re.I)
    return clean_text(text)


def trim_zip(value: str) -> str:
    return value[:5]


def normalize_phone(value: str) -> str:
    digits = re.sub(r"\D", "", value)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        return ""
    return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"


def dedupe_urls(urls: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for url in urls:
        if not url or url in seen:
            continue
        seen.add(url)
        output.append(url)
    return output


def dedupe_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def dedupe_offices(values: Iterable[OfficeCandidate]) -> list[OfficeCandidate]:
    seen: set[tuple[str, str, str]] = set()
    output: list[OfficeCandidate] = []
    for value in values:
        key = (value.street_address, value.city_state_zip, value.url)
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output
