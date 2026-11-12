#!/usr/bin/env python3
"""Build offline Wikipedia/Wikidata KB index for MultiCoNER retrieval (Stage A1).

Extracts entity mentions from training data, fetches descriptions from the
Wikipedia API, and builds a BM25 index saved to disk.

Usage:
    python scripts/build_kb.py --languages EN-English BN-Bangla --output ./kb_index
    python scripts/build_kb.py --languages all --data-root dataset/multiconer2023 --output ./kb_index --rebuild
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Set

# Ensure the project root is on sys.path when run as `python scripts/build_kb.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from multiconer_ner.data import load_split_sentences
from multiconer_ner.kb import KBEntry, KBIndex

WIKIPEDIA_API = "https://{lang}.wikipedia.org/w/api.php"


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


def fetch_wikipedia_description(title: str, lang: str = "en") -> str:
    """Fetch short description from Wikipedia API."""
    try:
        params = {
            "action": "query",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "titles": title,
            "format": "json",
            "redirects": 1,
        }
        resp = requests.get(WIKIPEDIA_API.format(lang=lang), params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract", "")
            return extract[:500]
    except Exception:
        return ""
    return ""


def build_kb_index(
    languages: List[str],
    data_root: str,
    output_dir: str | Path,
    train_splits: List[str] = None,
    rebuild: bool = False,
) -> KBIndex:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not rebuild and (output_dir / "entries.json").exists():
        print(f"[build_kb] Loading existing index from {output_dir}")
        return KBIndex.load(output_dir)

    print(f"[build_kb] Building new KB index for languages: {languages}")
    entries: List[KBEntry] = []
    seen_qids: Set[str] = set()

    for lang in languages:
        # Map folder name to Wikipedia language code (e.g. "EN-English" -> "en")
        lang_code = lang.split("-")[0].lower() if "-" in lang else lang.lower()

        # Load training sentences to extract entity mentions
        try:
            train_sentences = []
            for split in (train_splits or ["train"]):
                train_sentences.extend(
                    load_split_sentences(data_root, lang, split)
                )
        except FileNotFoundError:
            print(f"[build_kb] No training data for {lang}, skipping.")
            continue

        mentions = extract_entity_mentions(train_sentences)
        print(f"[build_kb] {lang}: {len(mentions)} unique entity mentions")

        for mention in mentions:
            description = fetch_wikipedia_description(mention, lang=lang_code)
            if not description:
                description = mention  # Fallback: use mention text itself

            qid = f"wiki:{lang_code}:{mention.replace(' ', '_')}"
            if qid in seen_qids:
                continue
            seen_qids.add(qid)

            text = f"{mention} {description}".strip()
            entries.append(
                KBEntry(
                    qid=qid,
                    title=mention,
                    description=description,
                    aliases=[],
                    entity_type=None,
                    language=lang,
                    text=text,
                )
            )

    index = KBIndex(entries)
    index.save(output_dir)
    print(f"[build_kb] Saved {len(entries)} entries to {output_dir}")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MultiCoNER KB index")
    parser.add_argument(
        "--languages", nargs="+", required=True,
        help="Language folders (e.g. EN-English BN-Bangla)",
    )
    parser.add_argument(
        "--data-root", default="dataset/multiconer2023", help="Path to dataset root"
    )
    parser.add_argument(
        "--output", default="./kb_index", help="Output directory for KB index"
    )
    parser.add_argument(
        "--train-splits",
        nargs="+",
        default=["train"],
        help="Splits to extract entity mentions from",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild the KB index even if it exists",
    )
    args = parser.parse_args()

    build_kb_index(
        languages=args.languages,
        data_root=args.data_root,
        output_dir=args.output,
        train_splits=args.train_splits,
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
