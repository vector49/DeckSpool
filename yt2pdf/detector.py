"""Two-tier frame sampling, perceptual difference hashing, SSIM, and debounce engine."""

import time
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
import numpy as np
import cv2
import imagehash
from PIL import Image

from yt2pdf.config import ExtractionConfig
from yt2pdf.roi import mask_player_chrome, mask_box, PresenterDetector
from yt2pdf.human_detector import HumanNarratorDetector

logger = logging.getLogger(__name__)


@dataclass
class Slide:
    """Represents an accepted, unique presentation slide."""
    index: int
    timestamp: float          # In seconds
    frame_high_res: np.ndarray # Original resolution BGR frame
    frame_analysis: np.ndarray # 360p analysis grayscale
    dhash: imagehash.ImageHash


@dataclass
class CandidateSlide:
    """Tracks a potential slide settling through bullet points or animations."""
    first_seen_time: float
    last_change_time: float
    frame_high_res: np.ndarray
    frame_analysis: np.ndarray
    dhash: imagehash.ImageHash


def compute_opencv_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes Structural Similarity Index (SSIM) between two grayscale images using OpenCV.
    
    Fast vectorized implementation running in < 2ms on 360p images.
    Returns float in range [-1.0, 1.0], where 1.0 is identical.
    """
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    c1 = 6.5025   # (0.01 * 255) ** 2
    c2 = 58.5225  # (0.03 * 255) ** 2

    i1 = img1.astype(np.float64)
    i2 = img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(i1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(i2, (11, 11), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(i1 * i1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(i2 * i2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(i1 * i2, (11, 11), 1.5) - mu1_mu2

    numerator = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

    ssim_map = numerator / (denominator + 1e-7)
    return float(np.mean(ssim_map))


def compute_frame_similarity(
    gray1: np.ndarray,
    hash1: imagehash.ImageHash,
    gray2: np.ndarray,
    hash2: imagehash.ImageHash
) -> float:
    """Calculates combined similarity using perceptual dHash and structural SSIM.
    
    Fast early-exit with dHash distance avoids heavy SSIM on identical or radically distinct frames.
    """
    # Hamming distance between 64-bit dHashes
    dhash_dist = hash1 - hash2
    if dhash_dist <= 1:
        return 1.0  # Visually identical

    # If hashes are vastly different, compute SSIM to confirm
    ssim = compute_opencv_ssim(gray1, gray2)
    return max(0.0, min(1.0, ssim))


class SlideDetector:
    """Two-tier slide detector with debouncing and PIP/chrome masking."""

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.presenter_detector = PresenterDetector()
        self.human_detector = HumanNarratorDetector()
        self.committed_slides: List[Slide] = []
        self.candidate: Optional[CandidateSlide] = None

    def _prepare_analysis_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, imagehash.ImageHash]:
        """Downscales frame to 360p height, applies ROI masks, and generates dHash."""
        h, w = frame_bgr.shape[:2]
        target_h = 360
        target_w = max(1, int(w * (target_h / float(h))))
        
        # Resize to 360p
        resized = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Update presenter motion statistics if enabled
        if self.config.mask_presenter:
            self.presenter_detector.update(gray)
            mask_box_coords = self.presenter_detector.get_mask_box(
                self.config.presenter_mode,
                self.config.custom_roi_box
            )
            if mask_box_coords:
                gray = mask_box(gray, mask_box_coords)

        # Apply player chrome mask (titles, scrubber)
        if self.config.ignore_chrome:
            gray = mask_player_chrome(gray, top_pct=0.03, bottom_pct=0.05)

        # Perceptual difference hash
        pil_img = Image.fromarray(gray)
        dhash = imagehash.dhash(pil_img)

        return gray, dhash

    def process_frame(
        self,
        timestamp: float,
        frame_bgr: np.ndarray,
        is_last_frame: bool = False
    ) -> Optional[Slide]:
        """Evaluates a sampled frame and returns a committed Slide if a transition settled.
        
        Handles progressive bullet points and animations via debounce settling.
        """
        # Filter full-human / narrator frames (camera cuts to speaker talking)
        if getattr(self.config, "filter_narrator_frames", True):
            is_narrator, reason = self.human_detector.is_narrator_frame(frame_bgr)
            if is_narrator:
                logger.debug(f"Skipping narrator cutaway at {timestamp:.1f}s: {reason}")
                return None

        gray, dhash = self._prepare_analysis_frame(frame_bgr)
        newly_committed_slide: Optional[Slide] = None

        if not self.committed_slides and self.candidate is None:
            # First frame of the presentation
            self.candidate = CandidateSlide(
                first_seen_time=timestamp,
                last_change_time=timestamp,
                frame_high_res=frame_bgr,
                frame_analysis=gray,
                dhash=dhash
            )
            return None

        if self.candidate is None and self.committed_slides:
            # Check difference against last committed slide
            last_slide = self.committed_slides[-1]
            sim = compute_frame_similarity(last_slide.frame_analysis, last_slide.dhash, gray, dhash)
            if sim < self.config.similarity_threshold:
                # Slide changed; start candidate tracking
                self.candidate = CandidateSlide(
                    first_seen_time=timestamp,
                    last_change_time=timestamp,
                    frame_high_res=frame_bgr,
                    frame_analysis=gray,
                    dhash=dhash
                )
            return None

        if self.candidate is not None:
            # Compare current frame to candidate
            candidate_sim = compute_frame_similarity(
                self.candidate.frame_analysis,
                self.candidate.dhash,
                gray,
                dhash
            )

            if candidate_sim >= self.config.similarity_threshold:
                # Frame is stable! Update high-res frame to latest build
                self.candidate.frame_high_res = frame_bgr
                self.candidate.frame_analysis = gray
                self.candidate.dhash = dhash

                settle_time = timestamp - self.candidate.last_change_time
                slide_duration = timestamp - self.candidate.first_seen_time

                # If settled long enough and meets minimum slide duration
                if settle_time >= self.config.debounce_settle_time and slide_duration >= self.config.min_slide_duration:
                    newly_committed_slide = self._try_commit_candidate(timestamp)
            else:
                # Frame is still changing (animation/bullet in progress)
                # If the candidate was visible for at least min_slide_duration and settled before,
                # we could commit previous state, or update last_change_time
                self.candidate.last_change_time = timestamp
                self.candidate.frame_high_res = frame_bgr
                self.candidate.frame_analysis = gray
                self.candidate.dhash = dhash

        # Video EOF flush
        if is_last_frame and self.candidate is not None:
            slide_duration = timestamp - self.candidate.first_seen_time
            if slide_duration >= (self.config.min_slide_duration * 0.5):
                final_commit = self._try_commit_candidate(timestamp)
                if final_commit:
                    newly_committed_slide = final_commit

        return newly_committed_slide

    def _try_commit_candidate(self, current_timestamp: float) -> Optional[Slide]:
        """Commits the candidate slide if it differs from the last committed slide."""
        if self.candidate is None:
            return None

        # Check if candidate is duplicate of the last committed slide (e.g. speaker cutaway)
        if self.committed_slides:
            last_slide = self.committed_slides[-1]
            sim = compute_frame_similarity(
                last_slide.frame_analysis,
                last_slide.dhash,
                self.candidate.frame_analysis,
                self.candidate.dhash
            )
            if sim >= self.config.similarity_threshold:
                # Duplicate of previous slide (e.g. cut back to presentation)
                self.candidate = None
                return None

        # Create new committed slide
        idx = len(self.committed_slides) + 1
        slide = Slide(
            index=idx,
            timestamp=self.candidate.first_seen_time,
            frame_high_res=self.candidate.frame_high_res.copy(),
            frame_analysis=self.candidate.frame_analysis,
            dhash=self.candidate.dhash
        )
        self.committed_slides.append(slide)
        self.candidate = None
        logger.info(f"Committed Slide #{slide.index} at {slide.timestamp:.1f}s")
        return slide
