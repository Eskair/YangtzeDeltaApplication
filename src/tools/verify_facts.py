# -*- coding: utf-8 -*-
"""
Stage 1.5 · Fact Verification (verify_facts.py)
------------------------------------------------
Input:
  - src/data/extracted/<proposal_id>/raw_facts.jsonl  (from extract_facts_by_chunk)
  - src/data/prepared/<proposal_id>/full_text.txt     (source text)

Output:
  - src/data/extracted/<proposal_id>/verified_facts.jsonl  (facts with verification metadata)

Responsibility:
  - For each extracted fact, fuzzy-match key entities/numbers against source text
  - Flag facts with low entity coverage as "unverified"
  - Remove facts where key numeric claims can't be found in source
  - This is the single highest-ROI change for hallucination reduction
"""

import json
import argparse
import re
from pathlib import Path
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple

BASE_DIR = Path(__file__).resolve().parents[2]
PREPARED_DIR = BASE_DIR / "src" / "data" / "prepared"
EXTRACTED_DIR = BASE_DIR / "src" / "data" / "extracted"
PROGRESS_FILE = BASE_DIR / "src" / "data" / "step_progress.json"

ENTITY_COVERAGE_THRESHOLD = 0.40
NUMERIC_VERIFICATION_THRESHOLD = 0.60
FUZZY_MATCH_THRESHOLD = 0.75


def _write_progress(done: int, total: int, pid: str = "") -> None:
    try:
        path = PROGRESS_FILE.parent / f"step_progress_{pid}.json" if pid else PROGRESS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"done": done, "total": total}), encoding="utf-8")
    except Exception:
        pass


def _extract_entities(text: str) -> List[str]:
    """Extract key entities from text: proper nouns, technical terms, numbers."""
    entities = []

    numbers = re.findall(r"\d+(?:\.\d+)?(?:%|％)?", text)
    entities.extend(numbers)

    cn_entities = re.findall(
        r"[\u4e00-\u9fff]{2,}(?:大学|医院|公司|集团|研究所|研究院|中心|实验室|平台|基金|协会|学会)",
        text,
    )
    entities.extend(cn_entities)

    en_entities = re.findall(r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]+)*\b", text)
    entities.extend(en_entities)

    acronyms = re.findall(r"\b[A-Z]{2,6}\b", text)
    entities.extend(acronyms)

    tech_terms = re.findall(
        r"\b(?:mRNA|siRNA|LNP|CRISPR|CAR-T|PD-[L1]|HER2|EGFR|VEGF|GLP-1|"
        r"FDA|EMA|NMPA|ICH|GMP|GCP|GLP|ISO|IEC|"
        r"Phase\s*[I1-3]+|I期|II期|III期|临床[一二三]期)\b",
        text,
        re.IGNORECASE,
    )
    entities.extend(tech_terms)

    return list(set(e.strip() for e in entities if e.strip()))


def _extract_numbers(text: str) -> List[str]:
    """Extract all numeric values from text, including currency and percentages."""
    patterns = [
        r"\d+(?:,\d{3})*(?:\.\d+)?(?:\s*%|％)?",
        r"\$\s*\d+(?:,\d{3})*(?:\.\d+)?",
        r"\d+(?:\.\d+)?\s*(?:万|亿|百万|千万|million|billion|trillion)",
        r"(?:CAGR|增长率|年增长)\s*(?:约|≈|~)?\s*\d+(?:\.\d+)?%",
    ]
    numbers = []
    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        numbers.extend(matches)
    return list(set(n.strip() for n in numbers if n.strip()))


def _fuzzy_find(needle: str, haystack: str, threshold: float = FUZZY_MATCH_THRESHOLD) -> bool:
    """Check if needle appears (exactly or fuzzily) in haystack."""
    needle_clean = needle.strip().lower()
    haystack_clean = haystack.lower()

    if needle_clean in haystack_clean:
        return True

    if len(needle_clean) <= 3:
        return needle_clean in haystack_clean

    window_size = min(len(haystack_clean), len(needle_clean) * 3)
    best_ratio = 0.0

    for i in range(0, max(1, len(haystack_clean) - window_size + 1), len(needle_clean) // 2 or 1):
        window = haystack_clean[i : i + window_size]
        ratio = SequenceMatcher(None, needle_clean, window).ratio()
        best_ratio = max(best_ratio, ratio)
        if best_ratio >= threshold:
            return True

    return best_ratio >= threshold


def _get_source_chunk(full_text: str, fact: Dict[str, Any], context_margin: int = 200) -> str:
    """Get the source text chunk that corresponds to this fact, with context margin."""
    meta = fact.get("meta", {})
    char_start = meta.get("char_start", 0)
    char_end = meta.get("char_end", len(full_text))

    start = max(0, char_start - context_margin)
    end = min(len(full_text), char_end + context_margin)
    return full_text[start:end]


def verify_single_fact(
    fact: Dict[str, Any], full_text: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Verify a single fact against the source text.

    Returns (fact, verification_result) where verification_result contains:
      - entity_coverage: float (0-1)
      - numeric_verified: bool
      - status: "verified" | "partially_verified" | "unverified" | "suspect"
      - details: dict with per-entity match info
    """
    fact_text = fact.get("text", "")
    if not fact_text:
        return fact, {
            "status": "unverified",
            "entity_coverage": 0.0,
            "numeric_verified": False,
            "reason": "empty_fact_text",
        }

    source_chunk = _get_source_chunk(full_text, fact)

    entities = _extract_entities(fact_text)
    numbers = _extract_numbers(fact_text)

    entity_matches = {}
    matched_count = 0
    for entity in entities:
        found = _fuzzy_find(entity, source_chunk)
        entity_matches[entity] = found
        if found:
            matched_count += 1

    entity_coverage = matched_count / max(1, len(entities)) if entities else 1.0

    numeric_results = {}
    numeric_verified = True
    if numbers:
        num_matched = 0
        for num in numbers:
            digits = re.sub(r"[^\d.]", "", num)
            if digits and digits in re.sub(r"[^\d.]", "", source_chunk):
                numeric_results[num] = True
                num_matched += 1
            elif _fuzzy_find(num, source_chunk, threshold=0.8):
                numeric_results[num] = True
                num_matched += 1
            else:
                numeric_results[num] = False

        numeric_coverage = num_matched / len(numbers) if numbers else 1.0
        numeric_verified = numeric_coverage >= NUMERIC_VERIFICATION_THRESHOLD

    existing_suspect = (fact.get("meta", {}) or {}).get("suspect_numeric", False)

    if entity_coverage >= 0.7 and numeric_verified and not existing_suspect:
        status = "verified"
    elif entity_coverage >= ENTITY_COVERAGE_THRESHOLD and (numeric_verified or not numbers):
        status = "partially_verified"
    elif existing_suspect or (numbers and not numeric_verified):
        status = "suspect"
    else:
        status = "unverified"

    verification = {
        "status": status,
        "entity_coverage": round(entity_coverage, 3),
        "numeric_verified": numeric_verified,
        "entities_checked": len(entities),
        "entities_matched": matched_count,
        "numbers_checked": len(numbers),
        "numbers_matched": sum(1 for v in numeric_results.values() if v),
    }

    return fact, verification


def run_verify(proposal_id: str):
    """Run fact verification for a proposal."""
    full_text_path = PREPARED_DIR / proposal_id / "full_text.txt"
    if not full_text_path.exists():
        raise FileNotFoundError(f"full_text.txt not found: {full_text_path}")

    facts_path = EXTRACTED_DIR / proposal_id / "raw_facts.jsonl"
    if not facts_path.exists():
        raise FileNotFoundError(f"raw_facts.jsonl not found: {facts_path}")

    full_text = full_text_path.read_text(encoding="utf-8")
    print(f"[INFO] Loaded source text: {len(full_text)} chars")

    facts = []
    with facts_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                facts.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"[INFO] Loaded {len(facts)} facts to verify")

    out_path = EXTRACTED_DIR / proposal_id / "verified_facts.jsonl"
    stats = {"verified": 0, "partially_verified": 0, "unverified": 0, "suspect": 0, "removed": 0}
    total_kept = 0

    _write_progress(0, len(facts), proposal_id)
    with out_path.open("w", encoding="utf-8") as f_out:
        for i, fact in enumerate(facts):
            fact_obj, verification = verify_single_fact(fact, full_text)

            meta = fact_obj.get("meta", {})
            if not isinstance(meta, dict):
                meta = {}
            meta["verification"] = verification
            fact_obj["meta"] = meta

            status = verification["status"]
            stats[status] = stats.get(status, 0) + 1

            if status == "suspect" and verification["entity_coverage"] < 0.2:
                stats["removed"] += 1
                print(
                    f"  [REMOVED] Fact {i+1}: coverage={verification['entity_coverage']:.2f}, "
                    f"text={fact_obj.get('text', '')[:80]}..."
                )
                continue

            f_out.write(json.dumps(fact_obj, ensure_ascii=False) + "\n")
            total_kept += 1

            if (i + 1) % 20 == 0:
                _write_progress(i + 1, len(facts), proposal_id)

    _write_progress(len(facts), len(facts), proposal_id)

    print(f"\n[SUMMARY] Fact Verification Results:")
    print(f"  Total input facts:       {len(facts)}")
    print(f"  Verified:                {stats['verified']}")
    print(f"  Partially verified:      {stats['partially_verified']}")
    print(f"  Unverified (kept):       {stats['unverified']}")
    print(f"  Suspect (kept):          {stats['suspect'] - stats['removed']}")
    print(f"  Removed (low coverage):  {stats['removed']}")
    print(f"  Total kept:              {total_kept}")
    print(f"\n[OK] Verified facts written to: {out_path}")

    summary_path = EXTRACTED_DIR / proposal_id / "verification_summary.json"
    summary = {
        "proposal_id": proposal_id,
        "total_input": len(facts),
        "total_kept": total_kept,
        "removed": stats["removed"],
        "by_status": {k: v for k, v in stats.items() if k != "removed"},
        "thresholds": {
            "entity_coverage": ENTITY_COVERAGE_THRESHOLD,
            "numeric_verification": NUMERIC_VERIFICATION_THRESHOLD,
            "fuzzy_match": FUZZY_MATCH_THRESHOLD,
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Verification summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1.5: Verify extracted facts against source text"
    )
    parser.add_argument(
        "--proposal_id",
        required=False,
        help="Proposal ID (corresponds to src/data/extracted/<proposal_id>)",
    )
    args = parser.parse_args()

    if args.proposal_id:
        pid = args.proposal_id
    else:
        if not EXTRACTED_DIR.exists():
            raise FileNotFoundError(f"Extracted directory not found: {EXTRACTED_DIR}")
        candidates = [
            (d.stat().st_mtime, d.name)
            for d in EXTRACTED_DIR.iterdir()
            if d.is_dir()
        ]
        if not candidates:
            raise FileNotFoundError(f"No proposal directories in: {EXTRACTED_DIR}")
        pid = max(candidates, key=lambda x: x[0])[1]
        print(f"[INFO] [auto] Selected latest proposal: {pid}")

    run_verify(pid)


if __name__ == "__main__":
    main()
