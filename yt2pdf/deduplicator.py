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

from yt2pdf.detector import Slide, compute_opencv_ssim
from yt2pdf.human_detector import HumanNarratorDetector

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
    kept_indices: List[int] = field(default_factory=list)  # 0-based page indices
    # Each entry: (duplicate_page_number_1based, original_page_number_1based, similarity_score)
    duplicate_details: List[Tuple[int, int, float]] = field(default_factory=list)
    # Each entry: (narrator_page_number_1based, reason_description)
    narrator_details: List[Tuple[int, str]] = field(default_factory=list)


def are_slides_duplicate(
    gray1: np.ndarray,
    hash1: imagehash.ImageHash,
    gray2: np.ndarray,
    hash2: imagehash.ImageHash,
    similarity_threshold: float = 0.94
) -> Tuple[bool, float]:
    """Determines if two slide frames are identical/duplicate representations."""
    # 1. Quick check with perceptual dHash
    dhash_dist = hash1 - hash2
    if dhash_dist > 8:
        return False, 0.0

    # 2. Check pixel difference to catch text modifications on uniform backgrounds
    diff = cv2.absdiff(gray1, gray2)
    significant_diff_count = np.count_nonzero(diff > 25)

    # Even a single character or bullet point change alters > 100 pixels.
    # Genuine duplicates (with slight video compression or antialiasing) have <= 60 pixels differing by > 25.
    if significant_diff_count > 60:
        return False, 0.0

    # 3. Calculate SSIM
    ssim = compute_opencv_ssim(gray1, gray2)
    if ssim >= similarity_threshold:
        return True, ssim

    return False, ssim


def deduplicate_slides_list(
    slides: List[Slide],
    similarity_threshold: float = 0.92,
    filter_narrator_frames: bool = True
) -> Tuple[List[Slide], List[Tuple[int, int, float]]]:
    """Reviews an in-memory list of Slide objects across the whole presentation.
    
    Finds any duplicate or recurring slides (even if separated by other slides)
    and truncates the repetitive ones, keeping only the first appearance.
    Also filters out any full-human narrator camera cuts.
    
    Returns:
        Tuple of (unique_slides, duplicate_records)
        where duplicate_records is a list of (dup_slide_idx, orig_slide_idx, similarity).
    """
    if not slides:
        return [], []

    human_detector = HumanNarratorDetector() if filter_narrator_frames else None
    kept_slides: List[Slide] = []
    duplicate_records: List[Tuple[int, int, float]] = []

    for curr_slide in slides:
        # Check if full-human narrator shot
        if human_detector is not None:
            is_narrator, reason = human_detector.is_narrator_frame(curr_slide.frame_high_res)
            if is_narrator:
                logger.info(f"Slide #{curr_slide.index} is a full-human narrator cut: {reason}. Truncating.")
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

    return kept_slides, duplicate_records


def deduplicate_pdf(
    input_pdf_path: str,
    output_pdf_path: Optional[str] = None,
    similarity_threshold: float = 0.94,
    filter_narrator_frames: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cancel_event: Optional[threading.Event] = None
) -> DeduplicationResult:
    """Reviews the entire PDF, identifies all duplicate/repetitive pages and
    full-human narrator shots, truncating them to preserve only unique presentation slides.
    
    Args:
        input_pdf_path: Path to the input PDF file.
        output_pdf_path: Destination path for the deduplicated PDF. If None,
                         appends '_dedup.pdf' to the input filename.
        similarity_threshold: SSIM threshold (0.80 to 0.99) for matching pages.
        filter_narrator_frames: Whether to detect and remove full-human narrator shots.
        progress_callback: Optional callback receiving (current_page, total_pages, status_text).
        cancel_event: Optional threading.Event to abort early.
        
    Returns:
        DeduplicationResult object with statistics and duplicate details.
    """
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
    # Tuples of (kept_index, 360p_gray_np, dhash)
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

            # Render page at low resolution (~360p height) for rapid fingerprinting
            page = src_pdf[i]
            # scale=0.5 provides ample resolution for 360p-720p slide hashing
            pil_rendered = page.render(scale=0.5).to_pil()

            # Check if this PDF page is a full-human / narrator shot
            if human_detector is not None:
                bgr_rendered = cv2.cvtColor(np.array(pil_rendered), cv2.COLOR_RGB2BGR)
                is_narrator, n_reason = human_detector.is_narrator_frame(bgr_rendered)
                if is_narrator:
                    page_1based = i + 1
                    narrator_details.append((page_1based, n_reason))
                    logger.info(f"Page {page_1based} is a full-human narrator frame ({n_reason}). Truncating from PDF.")
                    continue

            # Downscale to 360p height for normalized SSIM
            target_h = 360
            w, h = pil_rendered.size
            target_w = max(1, int(w * (target_h / float(h))))
            pil_resized = pil_rendered.resize((target_w, target_h), Image.Resampling.BILINEAR)

            # Grayscale numpy array
            gray = np.array(pil_resized.convert("L"))
            dhash = imagehash.dhash(pil_resized)

            # Compare against all previously kept pages (in chronological order)
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

        # Build clean deduplicated PDF by importing only the kept pages
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
        f"{result.narrator_pages_removed_count} narrator pages removed."
    )
    return result
