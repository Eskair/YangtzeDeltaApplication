# -*- coding: utf-8 -*-
"""
Stage 1.5 · Fact Verification (verify_facts.py)
------------------------------------------------
Input:
  - src/data/extracted/<proposal_id>/raw_facts.jsonl  (Stage 1 output)
  - src/data/prepared/<proposal_id>/full_text.txt     (Stage 0 output)

Chinese commercial PDFs: numerals may appear full-width, with Chinese comma thousands
separators, or as produced by OCR. Matching normalizes some whitespace/punctuation; facts
should still be phrased consistently with full_text when possible.

Output:
  - src/data/extracted/<proposal_id>/verified_facts.jsonl

Verification goals:
  - Detect whether key entities in a fact appear in the source chunk
  - Detect whether numeric claims in a fact are supported by the source chunk
  - Mark suspicious facts with verification status:
      verified / partially_verified / unverified
  - Assign a verification_score (0.0 - 1.0) for downstream ranking

This is a lightweight, deterministic verifier (no LLM calls).
It uses string overlap, numeric matching, and light fuzzy matching.
Heuristics favor recall for faithful extractions: long-fact fuzzy (head/tail),
substring containment boost, relaxed status thresholds, optional full-text
window when meta offsets are missing.

Optional: ``jieba`` (see requirements.txt) improves Chinese token overlap when
the fact is CJK-heavy; if not installed or VERIFY_USE_JIEBA=0, regex tokens are used.
Numeric checks normalize full-width digits/commas and expand common ``X万 / X亿 / X千万``
patterns to Arabic substrings for matching. Pure Chinese numerals (e.g. 三千万元) are
not expanded; keep those spellings aligned in extraction when possible.
"""

import json
import os
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from difflib import SequenceMatcher

BASE_DIR = Path(__file__).resolve().parents[2]
PREPARED_DIR = BASE_DIR / "src" / "data" / "prepared"
EXTRACTED_DIR = BASE_DIR / "src" / "data" / "extracted"
PROGRESS_FILE = BASE_DIR / "src" / "data" / "step_progress.json"

try:
    import jieba as _jieba_mod  # type: ignore
except ImportError:
    _jieba_mod = None


def _jieba_enabled() -> bool:
    if _jieba_mod is None:
        return False
    return os.getenv("VERIFY_USE_JIEBA", "1").strip().lower() not in ("0", "false", "no")


_FW_DIGIT_TRANS = str.maketrans(
    "０１２３４５６７８９，．",
    "0123456789,.",
)


def _flatten_for_numeric_match(s: str) -> str:
    """Normalize text for substring search of Arabic numerals (CN commercial PDFs)."""
    t = (s or "").translate(_FW_DIGIT_TRANS)
    for ch in (" ", ",", "，", "\u3000", "\xa0"):
        t = t.replace(ch, "")
    return t


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    n = len(text)
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return cjk / max(1, n)


def _jieba_token_overlap(fact_text: str, source_text: str) -> float:
    """Jaccard-style overlap on jieba tokens (CJK-friendly). Returns -1 if skipped."""
    if not _jieba_enabled() or not fact_text or not source_text:
        return -1.0

    def _tok_set(raw: str) -> set:
        out = set()
        for w in _jieba_mod.lcut(raw):
            w = w.strip().lower()
            if len(w) < 2:
                continue
            if w.isascii() and w.isalpha() and len(w) < 3:
                continue
            out.add(w)
        return out

    ft = _tok_set(fact_text)
    st = _tok_set(source_text)
    if not ft:
        return -1.0
    inter = len(ft & st)
    return inter / len(ft)


def _expand_cn_amount_tokens(text: str) -> List[str]:
    """
    Add Arabic strings for patterns like 3千万、0.3亿、1200万 so they match other spellings in source.
    """
    if not text:
        return []
    extra: List[str] = []
    t = text or ""

    def _push(n: float) -> None:
        if n == int(n) and n < 1e15:
            extra.append(str(int(n)))
        else:
            s = f"{n:.6f}".rstrip("0").rstrip(".")
            extra.append(s)

    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*千万", t):
        _push(float(m.group(1)) * 1e7)
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*百万", t):
        _push(float(m.group(1)) * 1e6)
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*万", t):
        _push(float(m.group(1)) * 1e4)
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*亿", t):
        _push(float(m.group(1)) * 1e8)
    return extra


def _write_progress(done: int, total: int, pid: str = "") -> None:
    try:
        path = PROGRESS_FILE.parent / f"step_progress_{pid}.json" if pid else PROGRESS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"done": done, "total": total}), encoding="utf-8")
    except Exception:
        pass


def find_latest_extracted_proposal() -> str:
    if not EXTRACTED_DIR.exists():
        raise FileNotFoundError(f"extracted directory not found: {EXTRACTED_DIR}")
    candidates = []
    for d in EXTRACTED_DIR.iterdir():
        if d.is_dir():
            candidates.append((d.stat().st_mtime, d.name))
    if not candidates:
        raise FileNotFoundError(f"No proposal directories in: {EXTRACTED_DIR}")
    proposal_id = max(candidates, key=lambda x: x[0])[1]
    print(f"[INFO] [auto] Selected latest proposal ID: {proposal_id}")
    return proposal_id


def load_full_text(proposal_id: str) -> str:
    path = PREPARED_DIR / proposal_id / "full_text.txt"
    if not path.exists():
        print(f"[WARN] full_text.txt not found: {path}, verification will use chunk text only")
        return ""
    return path.read_text(encoding="utf-8")


def load_raw_facts(proposal_id: str) -> List[Dict[str, Any]]:
    path = EXTRACTED_DIR / proposal_id / "raw_facts.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"raw_facts.jsonl not found: {path}")
    facts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                facts.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"[INFO] Loaded {len(facts)} facts from {path}")
    return facts


def extract_numbers(text: str) -> List[str]:
    """Extract all number-like tokens from text (integers, decimals, percentages)."""
    return re.findall(r"\d+(?:\.\d+)?%?", text or "")


def extract_key_entities(text: str) -> List[str]:
    """
    Extract key entity-like substrings: capitalized words, CJK name-like sequences,
    and technical terms (acronyms, alphanumeric identifiers).
    """
    entities = []

    # English: capitalized multi-word names (2+ words starting with uppercase)
    for m in re.finditer(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", text or ""):
        entities.append(m.group())

    # Acronyms and technical identifiers
    for m in re.finditer(r"\b[A-Z]{2,}(?:-?\d+)?\b", text or ""):
        entities.append(m.group())

    # CJK sequences (potential names/terms, 2-8 chars)
    for m in re.finditer(r"[\u4e00-\u9fff]{2,8}", text or ""):
        entities.append(m.group())

    return entities


def compute_text_overlap(fact_text: str, source_text: str) -> float:
    """
    Compute how much of the fact text overlaps with source text.
    Uses word-level token overlap normalized by fact length.
    When the fact is CJK-heavy and jieba is available, also scores jieba token overlap and takes the max.
    """
    if not fact_text or not source_text:
        return 0.0

    fact_lower = fact_text.lower()
    source_lower = source_text.lower()

    # Word-level tokens (regex; contiguous CJK runs — coarse for paraphrase / 我司 vs 全称)
    fact_words = set(re.findall(r"[a-z]+|\d+|[\u4e00-\u9fff]+", fact_lower))
    source_words = set(re.findall(r"[a-z]+|\d+|[\u4e00-\u9fff]+", source_lower))

    if not fact_words:
        return 0.0

    overlap = fact_words & source_words
    base = len(overlap) / len(fact_words)

    if _cjk_ratio(fact_text) >= 0.22:
        j = _jieba_token_overlap(fact_text, source_text)
        if j >= 0.0:
            return min(1.0, max(base, j))
    return base


def compute_entity_coverage(fact_text: str, source_text: str) -> Tuple[float, List[str]]:
    """
    Check what fraction of key entities in the fact appear in the source.
    Returns (coverage_ratio, list_of_missing_entities).
    """
    entities = extract_key_entities(fact_text)
    if not entities:
        return 1.0, []

    source_lower = source_text.lower()
    missing = []
    found = 0
    for ent in entities:
        if ent.lower() in source_lower:
            found += 1
        else:
            missing.append(ent)

    return found / len(entities), missing


def compute_numeric_accuracy(fact_text: str, source_text: str) -> Tuple[float, bool]:
    """
    Check whether numbers in the fact appear in the source text.
    Returns (accuracy_ratio, has_suspect_numbers).

    Normalizes full-width digits / Chinese commas; expands 万/亿/千万等写法为阿拉伯数字串再匹配，
    缓解「事实写 3000 万、原文写 3 千万」类 OCR/表述差异。
    """
    fact_nums = list(extract_numbers(fact_text))
    fact_nums.extend(_expand_cn_amount_tokens(fact_text))
    if not fact_nums:
        return 1.0, False

    source_flat = _flatten_for_numeric_match(source_text)
    seen = set()
    unique: List[str] = []
    for num in fact_nums:
        num_clean = num.replace(",", "").replace("，", "")
        if num_clean and num_clean not in seen:
            seen.add(num_clean)
            unique.append(num_clean)

    found = 0
    for num_clean in unique:
        pct = num_clean.endswith("%")
        core = num_clean[:-1] if pct else num_clean
        if core in source_flat:
            found += 1
            continue
        if pct and core in source_flat:
            found += 1
            continue
        # 百分比：原文可能写作「数字 + 百分比号」全角
        if pct and (core + "%" in source_text or core + "％" in source_text):
            found += 1

    accuracy = found / max(1, len(unique))
    return accuracy, accuracy < 1.0


def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").strip())


def _fact_likely_from_source(fact_text: str, source_text: str) -> bool:
    """True if fact (ignoring whitespace) appears as a contiguous substring of source."""
    a = _collapse_ws(fact_text)
    b = _collapse_ws(source_text)
    if len(a) < 6:
        return bool(a) and a in b
    return a in b


def fuzzy_substring_match(fact_text: str, source_text: str, threshold: float = 0.6) -> float:
    """
    Use SequenceMatcher to find the best fuzzy match ratio of the fact
    within the source text. Returns max similarity score.
    """
    if not fact_text or not source_text:
        return 0.0

    fact_clean = fact_text.strip()
    if len(fact_clean) < 5:
        return 1.0 if fact_clean in source_text else 0.0

    # For performance, only check a window around each potential start
    best = 0.0
    fact_len = len(fact_clean)
    source_len = len(source_text)

    # Slide a window of 2x fact length through source
    step = max(1, fact_len // 3)
    window = min(source_len, fact_len * 3)

    for start in range(0, max(1, source_len - fact_len + 1), step):
        end = min(source_len, start + window)
        chunk = source_text[start:end]
        ratio = SequenceMatcher(None, fact_clean, chunk).ratio()
        if ratio > best:
            best = ratio
        if best >= 0.9:
            break

    return best


def _fuzzy_score_for_fact_length(fact_text: str, chunk_text: str) -> float:
    """Head/tail windows so long facts are not penalized with fuzzy_score=0."""
    fact_clean = (fact_text or "").strip()
    if not fact_clean or not chunk_text:
        return 0.0
    if len(fact_clean) < 200:
        return fuzzy_substring_match(fact_clean, chunk_text)
    head = fact_clean[:200]
    tail = fact_clean[-200:]
    return max(
        fuzzy_substring_match(head, chunk_text),
        fuzzy_substring_match(tail, chunk_text),
    )


def _surrogate_chunk_from_fulltext(fact_text: str, full_text: str, margin: int = 500) -> str:
    """
    When meta offsets are missing, try to locate an early substring of the fact in full_text
    and return a surrounding window as verification context.
    """
    if not fact_text or not full_text:
        return ""
    needle = fact_text.strip()[: min(80, len(fact_text.strip()))]
    if len(needle) < 8:
        return ""
    idx = full_text.find(needle)
    if idx < 0:
        idx = full_text.find(needle.replace(" ", ""))
    if idx < 0:
        return ""
    lo = max(0, idx - margin)
    hi = min(len(full_text), idx + len(fact_text) + margin * 2)
    return full_text[lo:hi]


def get_chunk_text(fact: Dict[str, Any], full_text: str) -> str:
    """Extract the source chunk text for a given fact based on its meta offsets."""
    meta = fact.get("meta", {})
    char_start = meta.get("char_start")
    char_end = meta.get("char_end")
    if char_start is not None and char_end is not None and full_text:
        start = max(0, int(char_start))
        end = min(len(full_text), int(char_end))
        if start < end:
            return full_text[start:end]
    return ""


def verify_single_fact(fact: Dict[str, Any], full_text: str) -> Dict[str, Any]:
    """
    Verify a single fact against its source chunk.
    Returns the fact augmented with verification metadata.
    """
    fact_text = fact.get("text", "")
    if not fact_text:
        fact["verification"] = {
            "status": "unverified",
            "score": 0.0,
            "reason": "empty_fact_text",
        }
        return fact

    chunk_text = get_chunk_text(fact, full_text)
    chunk_reason = "meta_offsets"
    if not chunk_text and full_text:
        chunk_text = _surrogate_chunk_from_fulltext(fact_text, full_text)
        chunk_reason = "fulltext_surrogate" if chunk_text else "no_source_chunk"

    if not chunk_text:
        fact["verification"] = {
            "status": "unverified",
            "score": 0.3,
            "reason": "no_source_chunk",
        }
        return fact

    # 1. Word-level text overlap
    text_overlap = compute_text_overlap(fact_text, chunk_text)

    # 2. Entity coverage
    entity_coverage, missing_entities = compute_entity_coverage(fact_text, chunk_text)

    # 3. Numeric accuracy
    numeric_accuracy, has_suspect_nums = compute_numeric_accuracy(fact_text, chunk_text)

    # 4. Fuzzy match (head/tail for long facts — previously long facts always got 0 here)
    fuzzy_score = _fuzzy_score_for_fact_length(fact_text, chunk_text)

    # Composite: slightly less weight on entity_coverage (CJK 2–8 字碎片易误判缺失)
    score = (
        0.42 * text_overlap
        + 0.18 * entity_coverage
        + 0.25 * numeric_accuracy
        + 0.15 * fuzzy_score
    )

    # Strong signal: paraphrase-free alignment with chunk (typical of faithful extraction)
    if _fact_likely_from_source(fact_text, chunk_text):
        score = max(score, 0.58)

    # Determine status (thresholds relaxed vs 0.7 / 0.4 to reduce false unverified)
    if score >= 0.62:
        status = "verified"
    elif score >= 0.36:
        status = "partially_verified"
    else:
        status = "unverified"

    # Override: if numeric suspect was already flagged, downgrade
    if fact.get("meta", {}).get("suspect_numeric") and has_suspect_nums:
        if status == "verified":
            status = "partially_verified"
        score = min(score, 0.65)

    fact["verification"] = {
        "status": status,
        "score": round(score, 3),
        "chunk_source": chunk_reason,
        "text_overlap": round(text_overlap, 3),
        "entity_coverage": round(entity_coverage, 3),
        "numeric_accuracy": round(numeric_accuracy, 3),
        "fuzzy_score": round(fuzzy_score, 3),
        "missing_entities": missing_entities[:5],
        "has_suspect_numbers": has_suspect_nums,
    }

    return fact


def run_verification(proposal_id: str):
    """Run verification on all facts for a proposal."""
    full_text = load_full_text(proposal_id)
    facts = load_raw_facts(proposal_id)

    out_dir = EXTRACTED_DIR / proposal_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "verified_facts.jsonl"

    status_counts = {"verified": 0, "partially_verified": 0, "unverified": 0}
    total_score = 0.0

    _write_progress(0, len(facts), proposal_id)
    with out_path.open("w", encoding="utf-8") as f_out:
        for i, fact in enumerate(facts):
            verified_fact = verify_single_fact(fact, full_text)
            f_out.write(json.dumps(verified_fact, ensure_ascii=False) + "\n")

            v = verified_fact.get("verification", {})
            status = v.get("status", "unverified")
            status_counts[status] = status_counts.get(status, 0) + 1
            total_score += v.get("score", 0.0)

            if (i + 1) % 50 == 0:
                print(f"[INFO] Verified {i+1}/{len(facts)} facts...")
            _write_progress(i + 1, len(facts), proposal_id)

    avg_score = total_score / max(1, len(facts))

    print(f"\n[OK] Verified facts written to: {out_path}")
    print(f"\n[SUMMARY] Verification results:")
    print(f"  - Total facts: {len(facts)}")
    print(f"  - Verified: {status_counts['verified']}")
    print(f"  - Partially verified: {status_counts['partially_verified']}")
    print(f"  - Unverified: {status_counts['unverified']}")
    print(f"  - Average verification score: {avg_score:.3f}")

    if status_counts["unverified"] > len(facts) * 0.3:
        print(
            f"[WARN] Over 30% of facts are unverified ({status_counts['unverified']}/{len(facts)}). "
            "Consider reviewing extraction quality or chunk overlap settings."
        )


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
        pid = find_latest_extracted_proposal()

    run_verification(pid)


if __name__ == "__main__":
    main()
