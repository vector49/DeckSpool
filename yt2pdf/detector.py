"""Two-tier frame sampling, perceptual difference hashing, SSIM, and debounce engine."""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
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
    """Represents an accepted, unique presentation slide with compressed in-memory high-res storage."""
    index: int
    timestamp: float
    frame_analysis: np.ndarray
    dhash: imagehash.ImageHash
    _compressed_jpg: Optional[bytes] = field(default=None, repr=False)
    _raw_high_res: Optional[np.ndarray] = field(default=None, repr=False)

    def __init__(
        self,
        index: int,
        timestamp: float,
        frame_high_res: np.ndarray,
        frame_analysis: np.ndarray,
        dhash: imagehash.ImageHash,
        jpeg_quality: int = 95
    ):
        self.index = index
        self.timestamp = timestamp
        self.frame_analysis = frame_analysis
        self.dhash = dhash
        self._raw_high_res = None
        self._compressed_jpg = None

        if frame_high_res is not None:
            success, encoded = cv2.imencode(
                ".jpg", frame_high_res, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            )
            if success:
                self._compressed_jpg = encoded.tobytes()
            else:
                self._raw_high_res = frame_high_res.copy()

    @property
    def frame_high_res(self) -> np.ndarray:
        """Lazily decodes and returns the high-resolution BGR frame on demand."""
        if self._compressed_jpg is not None:
            arr = np.frombuffer(self._compressed_jpg, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return self._raw_high_res


@dataclass
class CandidateSlide:
    """Tracks a potential slide settling through bullet points or animations."""
    first_seen_time: float
    last_change_time: float
    last_stable_time: float
    frame_high_res: np.ndarray
    frame_analysis: np.ndarray
    dhash: imagehash.ImageHash


def compute_opencv_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes Structural Similarity Index (SSIM) between two grayscale images using OpenCV."""
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


def has_meaningful_structural_change(gray1: np.ndarray, gray2: np.ndarray) -> bool:
    """Uses morphological opening and connected components to distinguish real text/diagram changes
    from blinking cursors, watch timers, or video compression noise.
    """
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Remove 1-2px isolated compression artifacts and thin blinking cursors
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    active_pixels = int(np.count_nonzero(cleaned))
    if active_pixels > 350:
        return True

    if active_pixels < 45:
        return False

    # Check if there is at least one contiguous blob of new content (e.g. a new command or bullet)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    for label_idx in range(1, num_labels):
        blob_area = stats[label_idx, cv2.CC_STAT_AREA]
        if blob_area >= 140:
            return True

    return False


def compute_frame_similarity(
    gray1: np.ndarray,
    hash1: imagehash.ImageHash,
    gray2: np.ndarray,
    hash2: imagehash.ImageHash
) -> float:
    """Calculates combined similarity using perceptual dHash, connected-blob gating, and SSIM."""
    dhash_dist = hash1 - hash2
    if dhash_dist <= 1 and not has_meaningful_structural_change(gray1, gray2):
        return 1.0

    ssim = compute_opencv_ssim(gray1, gray2)
    # If SSIM is reasonably high and there is no meaningful contiguous blob change, boost stability
    if ssim >= 0.90 and not has_meaningful_structural_change(gray1, gray2):
        return max(ssim, 0.985)

    return max(0.0, min(1.0, ssim))


class SlideDetector:
    """Two-tier slide detector with debouncing, progressive build collapsing, and PIP/sidebar masking."""

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.presenter_detector = PresenterDetector()
        self.human_detector = HumanNarratorDetector()
        self.committed_slides: List[Slide] = []
        self.candidate: Optional[CandidateSlide] = None

    def _prepare_analysis_frame(
        self,
        frame_bgr: np.ndarray,
        detected_faces: Optional[List[Tuple[float, float, float, float, float]]] = None
    ) -> Tuple[np.ndarray, imagehash.ImageHash]:
        """Downscales frame to 360p height, applies ROI masks, and generates dHash."""
        h, w = frame_bgr.shape[:2]
        target_h = 360
        target_w = max(1, int(w * (target_h / float(h))))

        resized = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        if self.config.mask_presenter:
            self.presenter_detector.update(gray, detected_faces=detected_faces)
            mask_box_coords = self.presenter_detector.get_mask_box(
                self.config.presenter_mode,
                self.config.custom_roi_box
            )
            if mask_box_coords:
                gray = mask_box(gray, mask_box_coords)

        if self.config.ignore_chrome:
            gray = mask_player_chrome(gray, top_pct=0.03, bottom_pct=0.05)

        pil_img = Image.fromarray(gray)
        dhash = imagehash.dhash(pil_img)

        return gray, dhash

    def process_frame(
        self,
        timestamp: float,
        frame_bgr: np.ndarray,
        is_last_frame: bool = False
    ) -> Optional[Slide]:
        """Evaluates a sampled frame and returns a committed Slide if a transition settled."""
        detected_faces = []
        if self.config.mask_presenter or getattr(self.config, "filter_narrator_frames", True):
            detected_faces = self.human_detector.get_detected_faces(frame_bgr)

        if getattr(self.config, "filter_narrator_frames", True):
            is_narrator, reason = self.human_detector.is_narrator_frame(frame_bgr, detected_faces=detected_faces)
            if is_narrator:
                logger.debug(f"Skipping narrator/gallery cutaway at {timestamp:.1f}s: {reason}")
                return None

        gray, dhash = self._prepare_analysis_frame(frame_bgr, detected_faces=detected_faces)
        newly_committed_slide: Optional[Slide] = None

        if not self.committed_slides and self.candidate is None:
            self.candidate = CandidateSlide(
                first_seen_time=timestamp,
                last_change_time=timestamp,
                last_stable_time=timestamp,
                frame_high_res=frame_bgr,
                frame_analysis=gray,
                dhash=dhash
            )
            return None

        if self.candidate is None and self.committed_slides:
            last_slide = self.committed_slides[-1]
            sim = compute_frame_similarity(last_slide.frame_analysis, last_slide.dhash, gray, dhash)
            if sim < self.config.similarity_threshold:
                self.candidate = CandidateSlide(
                    first_seen_time=timestamp,
                    last_change_time=timestamp,
                    last_stable_time=timestamp,
                    frame_high_res=frame_bgr,
                    frame_analysis=gray,
                    dhash=dhash
                )
            return None

        if self.candidate is not None:
            candidate_sim = compute_frame_similarity(
                self.candidate.frame_analysis,
                self.candidate.dhash,
                gray,
                dhash
            )

            if candidate_sim >= self.config.similarity_threshold:
                # Frame is stable; update candidate to latest clean frame
                self.candidate.last_stable_time = timestamp
                self.candidate.frame_high_res = frame_bgr
                self.candidate.frame_analysis = gray
                self.candidate.dhash = dhash

                settle_time = timestamp - self.candidate.last_change_time
                slide_duration = timestamp - self.candidate.first_seen_time

                if settle_time >= self.config.debounce_settle_time and slide_duration >= self.config.min_slide_duration:
                    newly_committed_slide = self._try_commit_candidate(timestamp)
            else:
                # Abrupt change detected: check if the previous candidate had ALREADY settled before this change
                prior_settle = self.candidate.last_stable_time - self.candidate.last_change_time
                prior_duration = self.candidate.last_stable_time - self.candidate.first_seen_time

                if prior_settle >= self.config.debounce_settle_time and prior_duration >= self.config.min_slide_duration:
                    newly_committed_slide = self._try_commit_candidate(self.candidate.last_stable_time)
                    self.candidate = CandidateSlide(
                        first_seen_time=timestamp,
                        last_change_time=timestamp,
                        last_stable_time=timestamp,
                        frame_high_res=frame_bgr,
                        frame_analysis=gray,
                        dhash=dhash
                    )
                else:
                    # Animation or rapid scrolling still in progress; reset settle timer
                    self.candidate.last_change_time = timestamp
                    self.candidate.last_stable_time = timestamp
                    self.candidate.frame_high_res = frame_bgr
                    self.candidate.frame_analysis = gray
                    self.candidate.dhash = dhash

        if is_last_frame and self.candidate is not None:
            slide_duration = timestamp - self.candidate.first_seen_time
            if slide_duration >= (self.config.min_slide_duration * 0.5):
                final_commit = self._try_commit_candidate(timestamp)
                if final_commit:
                    newly_committed_slide = final_commit

        return newly_committed_slide

    def _is_progressive_slide_build(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> bool:
        """Checks if curr_gray is a progressive bullet-point build on top of prev_gray."""
        ssim = compute_opencv_ssim(prev_gray, curr_gray)
        if ssim < 0.84:
            return False

        edges_prev = cv2.Canny(prev_gray, 60, 180)
        edges_curr = cv2.Canny(curr_gray, 60, 180)

        prev_edge_count = int(np.count_nonzero(edges_prev))
        curr_edge_count = int(np.count_nonzero(edges_curr))

        if curr_edge_count <= prev_edge_count * 1.05:
            return False

        # Dilate current edges slightly to account for 1px compression shifts
        dilated_curr = cv2.dilate(edges_curr, np.ones((3, 3), np.uint8))
        retained_prev_edges = int(np.count_nonzero(cv2.bitwise_and(edges_prev, dilated_curr)))
        retention_ratio = retained_prev_edges / float(max(1, prev_edge_count))

        # If >= 88% of previous slide edges are still in place while new edges were added, it's a progressive build
        return retention_ratio >= 0.88

    def _try_commit_candidate(self, current_timestamp: float) -> Optional[Slide]:
        """Commits the candidate slide if it differs from the last committed slide, or upgrades a partial build."""
        if self.candidate is None:
            return None

        if self.committed_slides:
            last_slide = self.committed_slides[-1]
            sim = compute_frame_similarity(
                last_slide.frame_analysis,
                last_slide.dhash,
                self.candidate.frame_analysis,
                self.candidate.dhash
            )
            if sim >= self.config.similarity_threshold:
                self.candidate = None
                return None

            # Collapse progressive bullet-point builds into the completed slide (e.g. Page 2 -> Page 3)
            if getattr(self.config, "collapse_progressive_builds", True):
                if self._is_progressive_slide_build(last_slide.frame_analysis, self.candidate.frame_analysis):
                    high_res = self.candidate.frame_high_res
                    if self.config.mask_presenter:
                        mask_box_coords = self.presenter_detector.get_mask_box(
                            self.config.presenter_mode,
                            self.config.custom_roi_box
                        )
                        if mask_box_coords:
                            high_res = mask_box(high_res, mask_box_coords)
                    upgraded_slide = Slide(
                        index=last_slide.index,
                        timestamp=self.candidate.first_seen_time,
                        frame_high_res=high_res,
                        frame_analysis=self.candidate.frame_analysis,
                        dhash=self.candidate.dhash
                    )
                    self.committed_slides[-1] = upgraded_slide
                    self.candidate = None
                    logger.info(f"Upgraded progressive build on Slide #{upgraded_slide.index} at {upgraded_slide.timestamp:.1f}s")
                    return upgraded_slide

        idx = len(self.committed_slides) + 1
        high_res = self.candidate.frame_high_res
        if self.config.mask_presenter:
            mask_box_coords = self.presenter_detector.get_mask_box(
                self.config.presenter_mode,
                self.config.custom_roi_box
            )
            if mask_box_coords:
                high_res = mask_box(high_res, mask_box_coords)

        slide = Slide(
            index=idx,
            timestamp=self.candidate.first_seen_time,
            frame_high_res=high_res,
            frame_analysis=self.candidate.frame_analysis,
            dhash=self.candidate.dhash
        )
        self.committed_slides.append(slide)
        self.candidate = None
        logger.info(f"Committed Slide #{slide.index} at {slide.timestamp:.1f}s")
        return slide
