"""Review and deduplicate PDF documents and slide streams, holding the first appearance alone."""

import os
import logging
import threading
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
import cv2
import numpy as np
import imagehash
from PIL import Image
import pypdfium2

from yt2pdf.detector import Slide, compute_opencv_ssim, has_meaningful_structural_change
from yt2pdf.human_detector import HumanNarratorDetector
from yt2pdf.downloader import clean_file_path

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationResult:
    """Detailed summary of the deduplication pass."""
    input_path: str
    output_path: str
    total_pages: int
    unique_pages_count: int
    duplicates_removed_count: int
    narrator_pages_removed_count: int = 0
    kept_indices: List[int] = field(default_factory=list)
    duplicate_details: List[Tuple[int, int, float]] = field(default_factory=list)
    narrator_details: List[Tuple[int, str]] = field(default_factory=list)


def are_slides_duplicate(
    gray1: np.ndarray,
    hash1: imagehash.ImageHash,
    gray2: np.ndarray,
    hash2: imagehash.ImageHash,
    similarity_threshold: float = 0.94
) -> Tuple[bool, float]:
    """Determines if two slide frames are identical/duplicate representations using
    dHash, connected-component structural gating, and SSIM.
    """
    # 1. Quick check with perceptual dHash
    dhash_dist = hash1 - hash2
    if dhash_dist > 10:
        return False, 0.0

    # 2. Compute SSIM
    ssim = compute_opencv_ssim(gray1, gray2)

    # 3. Morphological connected-component check to ignore blinking cursors / watch timers
    meaningful_diff = has_meaningful_structural_change(gray1, gray2)
    if not meaningful_diff and ssim >= (similarity_threshold - 0.03):
        return True, max(ssim, similarity_threshold)

    if meaningful_diff:
        return False, ssim

    if ssim >= similarity_threshold:
        return True, ssim

    return False, ssim


def deduplicate_slides_list(
    slides: List[Slide],
    similarity_threshold: float = 0.92,
    filter_narrator_frames: bool = True
) -> Tuple[List[Slide], List[Tuple[int, int, float]]]:
    """Reviews an in-memory list of Slide objects across the whole presentation."""
    if not slides:
        return [], []

    human_detector = HumanNarratorDetector() if filter_narrator_frames else None
    kept_slides: List[Slide] = []
    duplicate_records: List[Tuple[int, int, float]] = []

    for curr_slide in slides:
        if human_detector is not None:
            is_narrator, reason = human_detector.is_narrator_frame(curr_slide.frame_high_res)
            if is_narrator:
                logger.info(f"Slide #{curr_slide.index} is a narrator/gallery cut: {reason}. Truncating.")
                continue

        is_duplicate = False
        matched_orig_index = -1
        highest_sim = 0.0

        for kept in kept_slides:
            is_dup, sim_score = are_slides_duplicate(
                curr_slide.frame_analysis,
                curr_slide.dhash,
                kept.frame_analysis,
                kept.dhash,
                similarity_threshold=similarity_threshold
            )
            if is_dup:
                is_duplicate = True
                matched_orig_index = kept.index
                highest_sim = sim_score
                break

        if is_duplicate:
            logger.info(
                f"Slide #{curr_slide.index} (at {curr_slide.timestamp:.1f}s) is a repetitive duplicate "
                f"of Slide #{matched_orig_index} (SSIM: {highest_sim:.3f}). Truncating."
            )
            duplicate_records.append((curr_slide.index, matched_orig_index, highest_sim))
        else:
            kept_slides.append(curr_slide)

    # Re-index kept slides sequentially
    for new_idx, slide in enumerate(kept_slides, start=1):
        slide.index = new_idx

    return kept_slides, duplicate_records


def deduplicate_pdf(
    input_pdf_path: str,
    output_pdf_path: Optional[str] = None,
    similarity_threshold: float = 0.94,
    filter_narrator_frames: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cancel_event: Optional[threading.Event] = None
) -> DeduplicationResult:
    """Reviews an entire PDF, removing duplicate pages, narrator shots, and conference gallery screens."""
    input_pdf_path = clean_file_path(input_pdf_path)
    if output_pdf_path:
        output_pdf_path = clean_file_path(output_pdf_path)

    if not os.path.exists(input_pdf_path):
        raise FileNotFoundError(f"Input PDF file not found: {input_pdf_path}")

    if not output_pdf_path:
        base, ext = os.path.splitext(input_pdf_path)
        output_pdf_path = f"{base}_dedup{ext}"

    src_pdf = pypdfium2.PdfDocument(input_pdf_path)
    total_pages = len(src_pdf)

    if total_pages == 0:
        src_pdf.close()
        raise ValueError("The provided PDF file contains 0 pages.")

    human_detector = HumanNarratorDetector() if filter_narrator_frames else None
    kept_indices: List[int] = []
    kept_fingerprints: List[Tuple[int, np.ndarray, imagehash.ImageHash]] = []
    duplicate_details: List[Tuple[int, int, float]] = []
    narrator_details: List[Tuple[int, str]] = []

    try:
        for i in range(total_pages):
            if cancel_event and cancel_event.is_set():
                logger.info("PDF deduplication cancelled by user.")
                break

            if progress_callback:
                progress_callback(i + 1, total_pages, f"Analyzing page {i + 1} of {total_pages}...")

            page = src_pdf[i]
            pil_rendered = page.render(scale=0.5).to_pil()

            if human_detector is not None:
                bgr_rendered = cv2.cvtColor(np.array(pil_rendered), cv2.COLOR_RGB2BGR)
                is_narrator, n_reason = human_detector.is_narrator_frame(bgr_rendered)
                if is_narrator:
                    page_1based = i + 1
                    narrator_details.append((page_1based, n_reason))
                    logger.info(f"Page {page_1based} is a narrator/gallery frame ({n_reason}). Truncating from PDF.")
                    continue

            target_h = 360
            w, h = pil_rendered.size
            target_w = max(1, int(w * (target_h / float(h))))
            pil_resized = pil_rendered.resize((target_w, target_h), Image.Resampling.BILINEAR)

            gray = np.array(pil_resized.convert("L"))
            dhash = imagehash.dhash(pil_resized)

            is_dup = False
            matched_first_appearance = -1
            match_score = 0.0

            for kept_idx, kept_gray, kept_hash in kept_fingerprints:
                is_match, score = are_slides_duplicate(
                    gray, dhash, kept_gray, kept_hash, similarity_threshold=similarity_threshold
                )
                if is_match:
                    is_dup = True
                    matched_first_appearance = kept_idx
                    match_score = score
                    break

            if is_dup:
                dup_1based = i + 1
                orig_1based = matched_first_appearance + 1
                duplicate_details.append((dup_1based, orig_1based, match_score))
                logger.info(
                    f"Page {dup_1based} matches earlier Page {orig_1based} "
                    f"(Similarity: {match_score:.3f}). Truncating repetitive page."
                )
            else:
                kept_indices.append(i)
                kept_fingerprints.append((i, gray, dhash))

        out_dir = os.path.dirname(os.path.abspath(output_pdf_path))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        new_pdf = pypdfium2.PdfDocument.new()
        new_pdf.import_pages(src_pdf, kept_indices)
        new_pdf.save(output_pdf_path)
        new_pdf.close()

    finally:
        src_pdf.close()

    result = DeduplicationResult(
        input_path=input_pdf_path,
        output_path=os.path.abspath(output_pdf_path),
        total_pages=total_pages,
        unique_pages_count=len(kept_indices),
        duplicates_removed_count=len(duplicate_details),
        narrator_pages_removed_count=len(narrator_details),
        kept_indices=kept_indices,
        duplicate_details=duplicate_details,
        narrator_details=narrator_details
    )

    logger.info(
        f"PDF Review Complete: {result.total_pages} total pages -> "
        f"{result.unique_pages_count} unique kept, {result.duplicates_removed_count} duplicate pages truncated, "
        f"{result.narrator_pages_removed_count} narrator/gallery pages removed."
    )
    return result
