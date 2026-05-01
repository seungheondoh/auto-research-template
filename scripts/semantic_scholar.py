#!/usr/bin/env python3
"""Query Semantic Scholar and print results in a format easy for LLM agents to parse."""

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request


API_BASE = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,year,abstract,citationCount,authors,externalIds,openAccessPdf"


def search(query: str, limit: int = 5, api_key: str | None = None) -> list[dict]:
    params = urllib.parse.urlencode({"query": query, "fields": FIELDS, "limit": limit})
    url = f"{API_BASE}/paper/search?{params}"

    for attempt in range(3):
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "auto-research-template/1.0")
        if api_key:
            req.add_header("x-api-key", api_key)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
                return data.get("data", [])
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 2:
                wait = 15 * (attempt + 1)
                print(f"[semantic_scholar] Rate limited — waiting {wait}s (attempt {attempt + 1}/3).")
                time.sleep(wait)
                continue
            print(f"[semantic_scholar] HTTP error {e.code}: {e.reason}", file=sys.stderr)
            return []
        except Exception as e:
            print(f"[semantic_scholar] Error: {e}", file=sys.stderr)
            return []
    return []


def format_paper(p: dict, idx: int) -> str:
    title = p.get("title", "Unknown title")
    year = p.get("year", "????")
    citations = p.get("citationCount", 0)
    authors = p.get("authors", [])
    first_author = authors[0]["name"].split()[-1] if authors else "Unknown"
    abstract = (p.get("abstract") or "No abstract available.")[:300].replace("\n", " ")
    pdf = (p.get("openAccessPdf") or {}).get("url", "")
    doi = (p.get("externalIds") or {}).get("DOI", "")
    s2_id = p.get("paperId", "")
    url = pdf or (f"https://doi.org/{doi}" if doi else f"https://www.semanticscholar.org/paper/{s2_id}")

    return (
        f"[{idx}] {first_author} et al. ({year}) — {citations} citations\n"
        f"    Title: {title}\n"
        f"    URL:   {url}\n"
        f"    Abstract (truncated): {abstract}...\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Search Semantic Scholar")
    parser.add_argument("query", help="Search query string")
    parser.add_argument("--limit", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    papers = search(args.query, limit=args.limit, api_key=api_key)

    if not papers:
        print("No results found.")
        return

    if args.json:
        print(json.dumps(papers, indent=2))
        return

    print(f"Semantic Scholar results for: \"{args.query}\"\n")
    print(f"Found {len(papers)} paper(s):\n")
    for i, paper in enumerate(papers, 1):
        print(format_paper(paper, i))


if __name__ == "__main__":
    main()
