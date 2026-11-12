#!/usr/bin/env python3
"""Build offline Wikipedia KB index for MultiCoNER retrieval (Stage A1).

Extracts entity mentions from training data, fetches intro extracts from the
Wikipedia API (batched, parallel, resumable), and builds a BM25 index saved
to disk.

Fetched descriptions are cached per language in <output>/cache/<LANG>.jsonl
as they arrive, so an interrupted build resumes where it stopped: just run the
same command again. Mentions whose fetch failed are not cached and are
retried on the next run (use --rebuild to regenerate entries.json).

Usage:
    python scripts/build_kb.py --languages EN-English BN-Bangla --output ./kb_index
    python scripts/build_kb.py --languages all --output ./kb_index --rebuild
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

# Ensure src/ is on sys.path when run as `python scripts/build_kb.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import requests

from multiconer_ner.data import load_split_sentences
from multiconer_ner.kb import KBEntry, KBIndex

WIKIPEDIA_API = "https://{lang}.wikipedia.org/w/api.php"
# Wikimedia rejects requests without a User-Agent (HTTP 403). Clients whose
# User-Agent has no contact info (email or URL) are "unidentified" and limited
# to 10 requests/min; with contact info the limit is 200 requests/min and at
# most 3 concurrent requests (https://www.mediawiki.org/wiki/Wikimedia_APIs/Rate_limits).
# Pass contact info with --contact or the WIKI_CONTACT environment variable.
USER_AGENT_BASE = "MultiCoNER-KB-builder/0.3"
USER_AGENT = f"{USER_AGENT_BASE} (academic NER research) python-requests/{requests.__version__}"
MAX_WORKERS = 3
MAX_TITLES_PER_REQUEST = 20  # API limit for prop=extracts with exintro
INVALID_TITLE_CHARS = set("#<>[]{}|")
DESCRIPTION_MAX_CHARS = 500

_thread_local = threading.local()


class _RateLimiter:
    """Process-wide request pacing shared by all worker threads."""

    def __init__(self, per_minute: float):
        self.interval = 60.0 / per_minute
        self.lock = threading.Lock()
        self.next_t = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_t)
            self.next_t = t + self.interval
        if t > now:
            time.sleep(t - now)

    def pause(self, seconds: float) -> None:
        """Delay every thread's next request (used on HTTP 429)."""
        with self.lock:
            self.next_t = max(self.next_t, time.monotonic() + seconds)


_LIMITER = _RateLimiter(9)


def configure_client(contact: Optional[str], per_minute: Optional[float]) -> None:
    global USER_AGENT, _LIMITER
    if contact:
        USER_AGENT = (f"{USER_AGENT_BASE} (academic NER research; {contact}) "
                      f"python-requests/{requests.__version__}")
    _LIMITER = _RateLimiter(per_minute or (180 if contact else 9))
    log(f"User-Agent: {USER_AGENT}")
    log(f"Rate limit: {60.0 / _LIMITER.interval:.0f} requests/min"
        + ("" if contact else "  (no --contact given: Wikimedia allows only 10/min for "
                              "unidentified clients, so this will be slow)"))


def log(msg: str) -> None:
    print(f"[build_kb {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def lang_code_of(lang: str) -> str:
    """Map folder name to Wikipedia language code (e.g. "EN-English" -> "en")."""
    return lang.split("-")[0].lower() if "-" in lang else lang.lower()


def extract_entity_mentions(sentences) -> Set[str]:
    """Extract unique entity mention texts from BIO-tagged sentences."""
    mentions: Set[str] = set()
    for sent in sentences:
        current_mention: List[str] = []
        for token, tag in zip(sent.tokens, sent.tags):
            if tag == "O":
                if current_mention:
                    mentions.add(" ".join(current_mention))
                    current_mention = []
            elif tag.startswith("B-"):
                if current_mention:
                    mentions.add(" ".join(current_mention))
                current_mention = [token]
            elif tag.startswith("I-"):
                if current_mention:
                    current_mention.append(token)
                else:
                    current_mention = [token]
        if current_mention:
            mentions.add(" ".join(current_mention))
    return mentions


# --------------------------------------------------------------------------- #
# Wikipedia API
# --------------------------------------------------------------------------- #
def _session() -> requests.Session:
    if getattr(_thread_local, "ua", None) != USER_AGENT:
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT})
        _thread_local.session, _thread_local.ua = s, USER_AGENT
    return _thread_local.session


def _api_post(lang_code: str, params: dict, max_retries: int = 8) -> dict:
    """POST to the Wikipedia API with retry/backoff. Raises after max_retries."""
    url = WIKIPEDIA_API.format(lang=lang_code)
    last_err: Optional[str] = None
    for attempt in range(max_retries):
        backoff = min(120.0, 5.0 * 2.0 ** attempt)
        _LIMITER.wait()
        try:
            resp = _session().post(url, data=params, timeout=60)
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = resp.headers.get("Retry-After", "")
                wait = max(float(retry_after), 5.0) if retry_after.isdigit() else backoff
                last_err = f"HTTP {resp.status_code}"
                if resp.status_code == 429:
                    _LIMITER.pause(wait)
                else:
                    time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                code = data["error"].get("code", "")
                last_err = f"API error {code}: {data['error'].get('info', '')}"
                if code in ("maxlag", "ratelimited", "internal_api_error_DBQueryError"):
                    time.sleep(backoff)
                    continue
                raise RuntimeError(last_err)
            return data
        except (requests.ConnectionError, requests.Timeout, ValueError) as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(backoff)
        except requests.HTTPError as e:
            raise RuntimeError(f"HTTP error: {e}") from e
    raise RuntimeError(f"giving up after {max_retries} attempts ({last_err})")


def is_valid_title(title: str) -> bool:
    return (
        bool(title.strip())
        and len(title.encode("utf-8")) <= 255
        and not any(c in INVALID_TITLE_CHARS for c in title)
    )


# Languages whose CoNLL tokens are single characters (no spaces between words).
CHAR_TOKEN_LANGS = {"zh"}


def _query_extracts(titles: Sequence[str], lang_code: str) -> Dict[str, str]:
    """One batched extracts query (<= 20 titles). Returns {title: extract} ("" = no page)."""
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": 1,
        "explaintext": 1,
        "exlimit": "max",
        "titles": "|".join(titles),
        "redirects": 1,
        "converttitles": 1,  # e.g. traditional <-> simplified Chinese
        "format": "json",
        "formatversion": 2,
    }
    mapping: Dict[str, str] = {}
    extracts: Dict[str, str] = {}
    while True:
        data = _api_post(lang_code, params)
        query = data.get("query", {})
        for key in ("normalized", "converted", "redirects"):
            for item in query.get(key, []):
                mapping[item["from"]] = item["to"]
        for page in query.get("pages", []):
            if page.get("extract"):
                extracts[page["title"]] = page["extract"]
        if "continue" not in data:
            break
        params = {**params, **data["continue"]}

    out: Dict[str, str] = {}
    for t in titles:
        cur, seen = t, set()
        while cur in mapping and cur not in seen:
            seen.add(cur)
            cur = mapping[cur]
        out[t] = extracts.get(cur, "")
    return out


def _title_case(s: str) -> str:
    return " ".join(w[:1].upper() + w[1:] for w in s.split(" "))


def fetch_wikipedia_descriptions(
    mentions: Sequence[str], lang_code: str, case_retry: bool = True
) -> Dict[str, str]:
    """Fetch intro extracts for up to 20 mentions.

    Returns {mention: extract[:500]} with "" for mentions that have no page.
    MultiCoNER text is lowercased, so mentions that miss are retried once in
    Title Case ("13 reasons why" -> "13 Reasons Why"). For character-tokenised
    languages (zh) the spaces between characters are removed first.
    Raises RuntimeError if a request ultimately fails (so the caller does not
    cache the failure).
    """
    out = {m: "" for m in mentions}
    lookup = {}
    for m in mentions:
        t = m.replace(" ", "") if lang_code in CHAR_TOKEN_LANGS else m
        if is_valid_title(t):
            lookup[m] = t
    if not lookup:
        return out

    first = _query_extracts(sorted(set(lookup.values())), lang_code)
    retry = {}
    for m, t in lookup.items():
        if first.get(t):
            out[m] = first[t][:DESCRIPTION_MAX_CHARS]
        else:
            tc = _title_case(t)
            if case_retry and tc != t and is_valid_title(tc):
                retry[m] = tc
    if retry:
        second = _query_extracts(sorted(set(retry.values())), lang_code)
        for m, tc in retry.items():
            out[m] = second.get(tc, "")[:DESCRIPTION_MAX_CHARS]
    return out


def fetch_wikipedia_description(title: str, lang: str = "en") -> str:
    """Fetch a single description (kept for backward compatibility)."""
    try:
        return fetch_wikipedia_descriptions([title], lang)[title]
    except RuntimeError:
        return ""


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
def load_cache(path: Path) -> Dict[str, str]:
    cache: Dict[str, str] = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # partial line from an interrupted write
                cache[rec["mention"]] = rec["description"]
    return cache


def search_text(mention: str, description: str, lang_code: str) -> str:
    """BM25 text for an entry. Queries are whitespace-tokenised sentences, so for
    character-tokenised languages the description is split into characters too."""
    if lang_code in CHAR_TOKEN_LANGS and description != mention:
        description = " ".join(c for c in description if not c.isspace())
    return f"{mention} {description}".strip()


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_kb_index(
    languages: List[str],
    data_root: str,
    output_dir: str | Path,
    train_splits: List[str] = None,
    rebuild: bool = False,
    workers: int = MAX_WORKERS,
    max_mentions: Optional[int] = None,
    contact: Optional[str] = None,
    requests_per_minute: Optional[float] = None,
    case_retry: bool = True,
) -> KBIndex:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    if not rebuild and (output_dir / "entries.json").exists():
        log(f"Loading existing index from {output_dir} (use --rebuild to regenerate)")
        return KBIndex.load(output_dir)

    if languages == ["all"]:
        languages = sorted(
            p.name for p in Path(data_root).iterdir()
            if p.is_dir() and "-" in p.name and not p.name.upper().startswith("MULTI")
        )
    log(f"Building KB index for languages: {languages}")
    configure_client(contact or os.environ.get("WIKI_CONTACT"), requests_per_minute)
    workers = max(1, min(workers, MAX_WORKERS))

    # 1. Collect mentions and work out what still needs fetching.
    mentions_by_lang: Dict[str, List[str]] = {}
    caches: Dict[str, Dict[str, str]] = {}
    jobs = []  # (lang, [titles])
    for lang in languages:
        try:
            train_sentences = []
            for split in (train_splits or ["train"]):
                train_sentences.extend(load_split_sentences(data_root, lang, split))
        except FileNotFoundError:
            log(f"No training data for {lang} under {data_root}, skipping.")
            continue
        mentions = sorted(extract_entity_mentions(train_sentences))
        if max_mentions:
            mentions = mentions[:max_mentions]
        mentions_by_lang[lang] = mentions
        caches[lang] = load_cache(cache_dir / f"{lang}.jsonl")
        todo = [m for m in mentions if m not in caches[lang]]
        log(f"{lang}: {len(mentions)} unique mentions, {len(mentions) - len(todo)} cached, "
            f"{len(todo)} to fetch")
        for i in range(0, len(todo), MAX_TITLES_PER_REQUEST):
            jobs.append((lang, todo[i:i + MAX_TITLES_PER_REQUEST]))

    if not mentions_by_lang:
        raise SystemExit(
            f"[build_kb] No training data found for {languages} under {data_root}. "
            "Use the folder names, e.g. EN-English BN-Bangla ..."
        )

    # 2. Fetch in parallel, appending each finished batch to the cache.
    n_failed = 0
    if jobs:
        total_titles = sum(len(t) for _, t in jobs)
        log(f"Fetching {total_titles} titles in {len(jobs)} requests with {workers} workers")
        locks = {lang: threading.Lock() for lang in mentions_by_lang}
        files = {lang: open(cache_dir / f"{lang}.jsonl", "a", encoding="utf-8")
                 for lang in mentions_by_lang}
        t0, done_titles, done_jobs = time.time(), 0, 0
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(fetch_wikipedia_descriptions, titles, lang_code_of(lang), case_retry):
                        (lang, titles)
                    for lang, titles in jobs
                }
                for fut in as_completed(futures):
                    lang, titles = futures[fut]
                    done_jobs += 1
                    done_titles += len(titles)
                    try:
                        result = fut.result()
                    except Exception as e:  # not cached -> retried next run
                        n_failed += len(titles)
                        log(f"WARNING {lang}: batch of {len(titles)} failed: {e}")
                        continue
                    with locks[lang]:
                        f = files[lang]
                        for t in titles:
                            caches[lang][t] = result[t]
                            f.write(json.dumps({"mention": t, "description": result[t]},
                                               ensure_ascii=False) + "\n")
                        f.flush()
                    if done_jobs % 50 == 0 or done_jobs == len(jobs):
                        elapsed = time.time() - t0
                        rate = done_titles / max(elapsed, 1e-6)
                        eta = (total_titles - done_titles) / max(rate, 1e-6)
                        log(f"progress {done_titles}/{total_titles} titles "
                            f"({100 * done_titles / total_titles:.1f}%), {rate:.1f} titles/s, "
                            f"ETA {eta / 60:.1f} min, failed {n_failed}")
        finally:
            for f in files.values():
                f.close()

    # 3. Build entries (fallback: mention text when no Wikipedia page).
    entries: List[KBEntry] = []
    seen_qids: Set[str] = set()
    for lang, mentions in mentions_by_lang.items():
        code = lang_code_of(lang)
        n_hit = 0
        for mention in mentions:
            description = caches[lang].get(mention, "")
            if description:
                n_hit += 1
            else:
                description = mention
            qid = f"wiki:{code}:{mention.replace(' ', '_')}"
            if qid in seen_qids:
                continue
            seen_qids.add(qid)
            entries.append(
                KBEntry(
                    qid=qid,
                    title=mention,
                    description=description,
                    aliases=[],
                    entity_type=None,
                    language=lang,
                    text=search_text(mention, description, code),
                )
            )
        log(f"{lang}: {n_hit}/{len(mentions)} mentions matched a Wikipedia page")

    index = KBIndex(entries)
    index.save(output_dir)
    log(f"Saved {len(entries)} entries to {output_dir / 'entries.json'}")
    if n_failed:
        log(f"WARNING: {n_failed} mentions failed to fetch and use the fallback text. "
            "Re-run the same command with --rebuild to retry only those.")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MultiCoNER KB index")
    parser.add_argument(
        "--languages", nargs="+", required=True,
        help="Language folders (e.g. EN-English BN-Bangla) or 'all'",
    )
    parser.add_argument(
        "--data-root", default="dataset/multiconer2023", help="Path to dataset root"
    )
    parser.add_argument(
        "--output", default="./kb_index", help="Output directory for KB index"
    )
    parser.add_argument(
        "--train-splits", nargs="+", default=["train"],
        help="Splits to extract entity mentions from",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Regenerate entries.json even if it exists (cached fetches are reused)",
    )
    parser.add_argument(
        "--workers", type=int, default=MAX_WORKERS,
        help=f"Parallel API requests (Wikimedia allows at most {MAX_WORKERS})",
    )
    parser.add_argument(
        "--contact", default=None,
        help="Email or URL put in the User-Agent (or set WIKI_CONTACT). Raises the "
             "Wikimedia limit from 10 to 200 requests/min.",
    )
    parser.add_argument(
        "--requests-per-minute", type=float, default=None,
        help="Override request pacing (default: 180 with --contact, else 9)",
    )
    parser.add_argument(
        "--no-case-retry", action="store_true",
        help="Do not retry missed (lowercased) mentions in Title Case",
    )
    parser.add_argument(
        "--max-mentions", type=int, default=None,
        help="Only use the first N mentions per language (for quick tests)",
    )
    args = parser.parse_args()

    build_kb_index(
        languages=args.languages,
        data_root=args.data_root,
        output_dir=args.output,
        train_splits=args.train_splits,
        rebuild=args.rebuild,
        workers=args.workers,
        max_mentions=args.max_mentions,
        contact=args.contact,
        requests_per_minute=args.requests_per_minute,
        case_retry=not args.no_case_retry,
    )


if __name__ == "__main__":
    main()
