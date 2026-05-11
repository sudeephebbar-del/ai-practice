"""
PDF QA with strikethrough awareness.

Plain text extraction cannot see deletion lines drawn as vector art. This script
scores horizontal ink that crosses glyph boxes, clusters words into approximate
reading lines, and proposes short phrases whose geometry looks crossed out.

This is heuristic: ornate PDF layouts (survey CTAs, nav underlines, table rules)
sometimes resemble strikes. Tune the constants near the StrikeConfig dataclass.

Install: pip install pymupdf openai python-dotenv
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import fitz  # pymupdf
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
client = OpenAI()


@dataclass(frozen=True)
class StrikeConfig:
    # Word boxes shorter than this are ignored – punctuation noise dominates.
    min_word_width: float = 9.5
    # Merge neighbors on the same visual line if the gap is within this many pt.
    max_merge_gap: float = 42.0
    # Line clustering tolerance in the vertical direction (points).
    line_y_tolerance: float = 3.25
    # When growing a seed horizontally, bridging tokens need at least this cover.
    bridge_min_cover: float = 0.50
    max_phrase_words: int = 4
    dashed_phrase_max_words: int = 2

    # ── Seeds (what we even try to assemble into a deletion phrase)
    long_seed_cover_min: float = 0.60
    long_seed_ink_pt_min: float = 19.0
    dashed_seed_cover_min: float = 0.575
    dashed_seed_long_max: float = 10.0
    dashed_seed_dash_lo: float = 0.76
    dashed_seed_dash_hi: float = 0.90

    # ── Final phrase acceptance
    glyph_cover_lo: float = 0.38
    # Long-ink strike path (mixed dashes + longer strokes, e.g. “Lazy river”)
    long_path_cover_lo: float = 0.572
    long_path_cover_hi: float = 0.675
    long_path_ink_lo: float = 38.0
    long_path_ink_width_frac: float = 0.603  # >= this * bbox width
    long_path_ink_cap_frac: float = 1.05  # ignore absurd totals from page rules
    # Pure dashed strike path (e.g. “Infinity” on some exports)
    dashed_path_cover_lo: float = 0.56
    dashed_path_cover_hi: float = 0.599
    dashed_path_dash_lo: float = 0.78
    dashed_path_dash_hi: float = 0.90
    dashed_path_long_max: float = 12.0


CFG = StrikeConfig()


@dataclass
class HorizontalSeg:
    xa: float
    xb: float
    y: float
    seglen: float


def _horizontal_segments(page: fitz.Page) -> list[HorizontalSeg]:
    out: list[HorizontalSeg] = []
    for drawing in page.get_drawings():
        for it in drawing.get("items", []) or []:
            if not it or it[0] != "l":
                continue
            p1, p2 = it[1], it[2]
            xa, xb = sorted((p1.x, p2.x))
            if abs(p2.y - p1.y) > 3:
                continue
            out.append(HorizontalSeg(xa, xb, (p1.y + p2.y) / 2.0, xb - xa))
    return out


@dataclass
class StrikeMetrics:
    cover_ratio: float
    short_ink_overlap: float
    long_ink_overlap: float
    rect_width: float


def _strike_metrics(segments: list[HorizontalSeg], rect: fitz.Rect) -> StrikeMetrics:
    h = rect.height
    y_lo = rect.y0 + 0.15 * h
    y_hi = rect.y1 - 0.20 * h
    if y_lo >= y_hi:
        return StrikeMetrics(0.0, 0.0, 0.0, rect.width)

    short_ink = 0.0
    long_ink = 0.0
    interval_union: list[tuple[float, float]] = []

    for s in segments:
        if not (y_lo <= s.y <= y_hi):
            continue
        ix0 = max(s.xa, rect.x0)
        ix1 = min(s.xb, rect.x1)
        if ix1 <= ix0:
            continue
        inter = ix1 - ix0
        if s.seglen < 4.0:
            short_ink += inter
        else:
            long_ink += inter
        interval_union.append((ix0, ix1))

    if not interval_union:
        return StrikeMetrics(0.0, short_ink, long_ink, rect.width)

    interval_union.sort()
    merged: list[list[float]] = []
    for a, b in interval_union:
        if not merged or a > merged[-1][1] + 0.5:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)
    cover = sum(r - l for l, r in merged) / max(rect.width, 1e-6)
    return StrikeMetrics(cover, short_ink, long_ink, rect.width)


def _cluster_visual_lines(
    words: list[tuple[float, float, float, float, str, int, int, int]],
    y_tol: float,
) -> list[list[tuple[float, float, float, float, str, int, int, int]]]:
    lines: list[list] = []
    bucket: list = []
    for w in sorted(words, key=lambda x: (float(x[1]), float(x[0]))):
        if not bucket:
            bucket = [w]
            continue
        med_y = sorted(float(x[1]) for x in bucket)[len(bucket) // 2]
        if abs(float(w[1]) - med_y) <= y_tol:
            bucket.append(w)
        else:
            lines.append(bucket)
            bucket = [w]
    if bucket:
        lines.append(bucket)
    return lines


def _word_rect(w: tuple[float, float, float, float, str, int, int, int]) -> fitz.Rect:
    return fitz.Rect(w[0], w[1], w[2], w[3])


def strikeout_annotation_text(page: fitz.Page) -> list[str]:
    found: list[str] = []
    for a in page.annots() or []:
        if a.type[1] != "StrikeOut":
            continue
        txt = (a.info or {}).get("content") or page.get_textbox(a.rect)
        txt = (txt or "").strip()
        if txt:
            found.append(txt)
    return found


def _seed_hit(m: StrikeMetrics, cfg: StrikeConfig) -> bool:
    dash = m.short_ink_overlap / max(m.rect_width, 1e-6)
    long_s = (
        m.cover_ratio >= cfg.long_seed_cover_min
        and m.long_ink_overlap >= cfg.long_seed_ink_pt_min
    )
    dash_s = (
        m.cover_ratio >= cfg.dashed_seed_cover_min
        and m.long_ink_overlap <= cfg.dashed_seed_long_max
        and cfg.dashed_seed_dash_lo <= dash <= cfg.dashed_seed_dash_hi
    )
    return long_s or dash_s


def _phrase_acceptable(
    m: StrikeMetrics,
    *,
    nw: int,
    phrase_text: str,
    cfg: StrikeConfig,
) -> bool:
    if re.search(r"\d", phrase_text):
        # Spec-style tokens (“420”, rankings) fooled the long path; crosses are
        # almost always letters on consumer PDFs—opt out digits here entirely.
        return False
    dash = m.short_ink_overlap / max(m.rect_width, 1e-6)
    if m.cover_ratio < cfg.glyph_cover_lo:
        return False

    long_floor = max(cfg.long_path_ink_lo, cfg.long_path_ink_width_frac * m.rect_width)
    long_path_ok = (
        nw <= cfg.max_phrase_words
        and cfg.long_path_cover_lo <= m.cover_ratio <= cfg.long_path_cover_hi
        and m.long_ink_overlap <= cfg.long_path_ink_cap_frac * m.rect_width
        and m.long_ink_overlap >= long_floor
    )
    dashed_path_ok = (
        nw <= cfg.dashed_phrase_max_words
        and cfg.dashed_path_cover_lo <= m.cover_ratio <= cfg.dashed_path_cover_hi
        and cfg.dashed_path_dash_lo <= dash <= cfg.dashed_path_dash_hi
        and m.long_ink_overlap <= cfg.dashed_path_long_max
    )

    return long_path_ok or dashed_path_ok


def collect_auto_stricken_phrases(doc: fitz.Document, cfg: StrikeConfig) -> list[str]:
    phrases: list[str] = []

    def add(p: str) -> None:
        p = " ".join(p.split())
        if p and p not in phrases:
            phrases.append(p)

    for pi in range(len(doc)):
        page = doc.load_page(pi)
        for t in strikeout_annotation_text(page):
            add(t)

        segments = _horizontal_segments(page)
        words = page.get_text("words")
        for line_words in _cluster_visual_lines(words, cfg.line_y_tolerance):
            line_words = [
                w for w in line_words if _word_rect(w).width >= cfg.min_word_width
            ]
            if not line_words:
                continue
            line_words.sort(key=lambda w: float(w[0]))

            metrics_list = [_strike_metrics(segments, _word_rect(w)) for w in line_words]
            seed_idx = [
                i for i, mm in enumerate(metrics_list) if _seed_hit(mm, cfg)
            ]
            if not seed_idx:
                continue

            seen_keys: set[tuple[int, int, int]] = set()

            for s in seed_idx:
                lo = hi = s
                while (
                    lo > 0
                    and float(line_words[lo][0]) - float(line_words[lo - 1][2])
                    < cfg.max_merge_gap
                    and metrics_list[lo - 1].cover_ratio >= cfg.bridge_min_cover
                ):
                    lo -= 1
                while (
                    hi < len(line_words) - 1
                    and float(line_words[hi + 1][0]) - float(line_words[hi][2])
                    < cfg.max_merge_gap
                    and metrics_list[hi + 1].cover_ratio >= cfg.bridge_min_cover
                ):
                    hi += 1
                if hi - lo + 1 > cfg.max_phrase_words:
                    lo = hi = s

                dedupe_key = (pi, lo, hi)
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)

                union_rect: fitz.Rect | None = None
                for idx in range(lo, hi + 1):
                    r = _word_rect(line_words[idx])
                    union_rect = r if union_rect is None else (union_rect | r)
                assert union_rect is not None

                phrase_txt = " ".join(line_words[i][4] for i in range(lo, hi + 1))
                um = _strike_metrics(segments, union_rect)

                if _phrase_acceptable(
                    um, nw=hi - lo + 1, phrase_text=phrase_txt, cfg=cfg
                ):
                    add(phrase_txt)

    return phrases


def build_document_bundle(pdf_path: str, cfg: StrikeConfig) -> tuple[str, list[str]]:
    doc = fitz.open(pdf_path)
    try:
        body: list[str] = []
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            body.append(f"\n--- Page {page_index + 1} ---\n")
            body.append(page.get_text())
        struck = collect_auto_stricken_phrases(doc, cfg)
    finally:
        doc.close()

    return "".join(body).strip(), struck


def count_tokens_approx(text: str) -> int:
    return len(text) // 4


def ask_about_document(
    doc_with_notes: str, question: str, stricken_known: list[str]
) -> str:
    strikes = (
        "; ".join(sorted(set(stricken_known)))
        if stricken_known
        else "(none detected programmatically)"
    )
    prompt = f"""You are a travel specialist.

Rules:
If the DOCUMENT shows an amenity or claim only inside material that has been visibly
crossed out / struck through (deleted lines), answer NO—or say clearly that it is
not offered as current fact—unless the SAME facility is plainly confirmed elsewhere
without strike-through.
Do not infer from marketing tone alone; stale crossed-out wording does not count.

AUTOMATED STRIKE / DELETION NOTES (approximate geometry + StrikeOut annotations):
Suspected visually struck wording (automated extraction; may omit edge cases): {strikes}

DOCUMENT:
{doc_with_notes}

QUESTION: {question}

Answer based only on the DOCUMENT and the rules above. If the answer is not in the
document, say "Not found in this document."
"""

    approx_tokens = count_tokens_approx(prompt)
    print(f"Approximate prompt tokens: {approx_tokens:,}")
    print(f"gpt-4o-mini context limit: 128,000 tokens")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content


PDF_PATH = os.environ.get(
    "PDF_QA_SAMPLE",
    r"C:\OneDrive\Personal\Career\Learnings\2026\Verizon\AI\files\Hilton Beachfront Resort and Spa Hilton Head Island Reviews & Prices _ U.S. News Travel.pdf",
)


if __name__ == "__main__":
    print("Extracting text + auto-detecting likely strikethrough phrases...")
    doc_text, struck = build_document_bundle(PDF_PATH, CFG)

    preamble = ""
    if struck:
        preamble = (
            "\n[Automated preprocessing] Phrases flagged as likely crossed-off in "
            "the PDF: "
            + ", ".join(struck)
            + "\n\n"
        )

    full_doc = preamble + doc_text
    print(f"Characters in document excerpt: {len(full_doc):,}")
    print(f"Likely struck phrases ({len(struck)}): {struck or '[]'}")

    questions = [
        "What is the main purpose of this document?",
        "What are the key features of this resort?",
        "List any limitations or constraints described.",
        "Does this resort have lazy river",
        "Does this resort have infinity pool",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {ask_about_document(full_doc, q, struck)}")
