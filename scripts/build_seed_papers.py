#!/usr/bin/env python3
"""Build cache/seed_papers.json from real SciVerse meta-search across 6 directions.

Run once to bootstrap the seed corpus; commit the output to cache/.
Usage: python scripts/build_seed_papers.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import load_config  # noqa: E402
load_config()
from tools.clients.sciverse_client import SciVerseClient  # noqa: E402

DIRECTIONS = [
    {
        "category": "Game as AI Benchmark",
        "queries": [
            "DQN Atari deep reinforcement learning",
            "AlphaGo AlphaZero game AI",
            "AlphaStar StarCraft reinforcement learning",
            "OpenAI Five Dota multiplayer",
        ],
    },
    {
        "category": "Internal World Model",
        "queries": [
            "world model reinforcement learning latent dynamics",
            "DreamerV3 mastering diverse domains world model",
            "MuZero planning learned model",
            "EfficientZero sample efficient world model",
            "IRIS world model discrete tokens",
            "DIAMOND diffusion world model",
        ],
    },
    {
        "category": "Neural Game Engine",
        "queries": [
            "GameGAN neural game engine",
            "Genie generative interactive environment",
            "GameNGen neural game engine diffusion",
            "Oasis real-time interactive world model",
        ],
    },
    {
        "category": "Foundation Interactive Game World Model",
        "queries": [
            "GameCraft interactive game world generation",
            "Matrix-Game open-world game generation",
        ],
    },
    {
        "category": "LLM / MLLM Game Agent",
        "queries": [
            "Voyager LLM Minecraft agent",
            "MineDojo large-scale Minecraft benchmark",
            "Cradle LLM game agent general",
            "Generative Agents believable human behavior simulation",
        ],
    },
    {
        "category": "Benchmark / Evaluation / Toolkit",
        "queries": [
            "MineDojo benchmark evaluation Minecraft",
            "game AI benchmark evaluation toolkit",
        ],
    },
]

YEAR_FILTERS = [
    {"field": "publication_published_year", "operator": "FILTER_OP_GTE", "value": 2013},
    {"field": "publication_published_year", "operator": "FILTER_OP_LTE", "value": 2026},
]


def _bibtex_key(authors, year, title):
    first = (authors[0].split()[-1] if authors else "unknown").lower()
    word = "".join(c for c in title.split()[0].lower() if c.isalpha()) if title else "x"
    return f"{first}{year}{word}"


def _derive_url(hit):
    for url in hit.get("access_oa_url") or []:
        if "/pdf" in url.lower() or url.lower().endswith(".pdf"):
            return url
    for loc in hit.get("locations") or []:
        if isinstance(loc, dict) and loc.get("url"):
            u = loc["url"]
            if "/pdf" in u.lower() or u.lower().endswith(".pdf"):
                return u
    for url in hit.get("access_oa_url") or []:
        return url
    for loc in hit.get("locations") or []:
        if isinstance(loc, dict) and loc.get("url"):
            return loc["url"]
    doi = hit.get("doi")
    return f"https://doi.org/{doi}" if doi else ""


def build():
    sv = SciVerseClient()
    seen_ids = set()
    corpus = []

    for direction in DIRECTIONS:
        cat = direction["category"]
        for query in direction["queries"]:
            try:
                res = sv.meta_search(query, filters=YEAR_FILTERS, impact_boost="MILD", page_size=5)
            except Exception as e:
                print(f"  WARN: {query!r} failed: {e}", file=sys.stderr)
                continue

            for hit in res.get("results", []):
                uid = hit.get("unique_id", "")
                if uid in seen_ids:
                    continue
                seen_ids.add(uid)

                authors = [a["name"] for a in (hit.get("author") or []) if isinstance(a, dict) and a.get("name")]
                year = int(hit.get("publication_published_year") or 0)
                title = hit.get("title") or ""
                if not title or not year:
                    continue

                corpus.append({
                    "title": title,
                    "authors": authors[:5],
                    "year": year,
                    "venue": hit.get("publication_venue_name_unified") or "arXiv",
                    "url": _derive_url(hit),
                    "abstract": (hit.get("abstract") or "")[:500],
                    "keywords": (hit.get("keywords") or [])[:8],
                    "category": cat,
                    "bibtex_key": _bibtex_key(authors, year, title),
                })
        print(f"  {cat}: {sum(1 for p in corpus if p['category'] == cat)} papers")

    # dedup by bibtex_key
    final = []
    seen_keys = set()
    for p in corpus:
        if p["bibtex_key"] not in seen_keys:
            seen_keys.add(p["bibtex_key"])
            final.append(p)

    out_path = ROOT / "cache" / "seed_papers.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {len(final)} papers to {out_path}")
    cats = {p["category"] for p in final}
    missing = set(d["category"] for d in DIRECTIONS) - cats
    if missing:
        print(f"  WARNING: missing categories: {missing}", file=sys.stderr)
    return final


if __name__ == "__main__":
    build()
