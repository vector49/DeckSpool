"""Region of Interest (ROI), Chrome Masking, and Canvas Auto-Cropping."""

import logging
from typing import Tuple, Optional, List
import numpy as np
import cv2

from yt2pdf.config import PresenterMaskMode

logger = logging.getLogger(__name__)

# Default corner presets (x, y, w, h) in normalized coordinates [0.0, 1.0]
PRESET_BOXES = {
    PresenterMaskMode.TOP_RIGHT: (0.72, 0.05, 0.26, 0.35),
    PresenterMaskMode.BOTTOM_RIGHT: (0.72, 0.60, 0.26, 0.35),
    PresenterMaskMode.BOTTOM_LEFT: (0.02, 0.60, 0.26, 0.35),
    PresenterMaskMode.TOP_LEFT: (0.02, 0.05, 0.26, 0.35),
}


def mask_player_chrome(frame: np.ndarray, top_pct: float = 0.03, bottom_pct: float = 0.05) -> np.ndarray:
    """Masks top and bottom player chrome (titles, seekbars, controls) with black.
    
    Operates in-place on a copy or input frame.
    """
    h, w = frame.shape[:2]
    top_h = int(h * top_pct)
    bottom_y = int(h * (1.0 - bottom_pct))
    
    masked = frame.copy()
    if top_h > 0:
        masked[:top_h, :] = 0
    if bottom_y < h:
        masked[bottom_y:, :] = 0
    return masked


def mask_box(frame: np.ndarray, box_norm: Tuple[float, float, float, float]) -> np.ndarray:
    """Masks out a normalized bounding box (x, y, w, h) in the frame with black."""
    h, w = frame.shape[:2]
    x, y, bw, bh = box_norm
    
    x1 = max(0, min(w, int(x * w)))
    y1 = max(0, min(h, int(y * h)))
    x2 = max(0, min(w, int((x + bw) * w)))
    y2 = max(0, min(h, int((y + bh) * h)))
    
    if x2 > x1 and y2 > y1:
        masked = frame.copy()
        masked[y1:y2, x1:x2] = 0
        return masked
    return frame


class PresenterDetector:
    """Detects presenter or PIP webcam box using motion variance grid and/or face detection."""

    def __init__(self, grid_size: int = 16):
        self.grid_size = grid_size
        self._prev_frame: Optional[np.ndarray] = None
        self._motion_accumulator: Optional[np.ndarray] = None
        self._samples_count: int = 0
        self.detected_box: Optional[Tuple[float, float, float, float]] = None

    def update(self, gray_360p: np.ndarray) -> None:
        """Update motion statistics with a new 360p grayscale frame."""
        h, w = gray_360p.shape[:2]
        if self._prev_frame is None or self._prev_frame.shape != gray_360p.shape:
            self._prev_frame = gray_360p.copy()
            self._motion_accumulator = np.zeros((h, w), dtype=np.float32)
            self._samples_count = 0
            return

        # Frame difference
        diff = cv2.absdiff(gray_360p, self._prev_frame).astype(np.float32)
        # Threshold small compression noise
        diff[diff < 10.0] = 0.0

        # Accumulate motion
        self._motion_accumulator += diff
        self._samples_count += 1
        self._prev_frame = gray_360p.copy()

        # Periodically resolve candidate PIP box
        if self._samples_count >= 5:
            self._resolve_candidate_box(h, w)

    def _resolve_candidate_box(self, h: int, w: int) -> None:
        """Check if one of the typical PIP regions exhibits high persistent motion."""
        if self._motion_accumulator is None or self._samples_count == 0:
            return

        avg_motion = self._motion_accumulator / float(self._samples_count)
        total_motion = np.sum(avg_motion)
        if total_motion < 1e-3:
            return

        # Check standard corner PIP areas (each ~28% width, ~35% height)
        candidate_boxes = [
            ("top_right", (0.70, 0.04, 0.28, 0.36)),
            ("bottom_right", (0.70, 0.60, 0.28, 0.36)),
            ("bottom_left", (0.02, 0.60, 0.28, 0.36)),
            ("top_left", (0.02, 0.04, 0.28, 0.36)),
        ]

        best_box = None
        highest_ratio = 0.0

        for name, box in candidate_boxes:
            bx, by, bw, bh = box
            x1 = int(bx * w)
            y1 = int(by * h)
            x2 = int((bx + bw) * w)
            y2 = int((by + bh) * h)

            region_motion = np.sum(avg_motion[y1:y2, x1:x2])
            region_area = (x2 - x1) * (y2 - y1)
            frame_area = w * h
            
            # Density of motion in region vs entire frame
            motion_fraction = region_motion / (total_motion + 1e-5)
            area_fraction = region_area / frame_area

            # If a corner contains > 40% of all motion while occupying < 15% of the frame
            if motion_fraction > 0.40 and (motion_fraction / area_fraction) > 2.5:
                if motion_fraction > highest_ratio:
                    highest_ratio = motion_fraction
                    best_box = box

        if best_box:
            self.detected_box = best_box

    def get_mask_box(self, mode: PresenterMaskMode, custom_box: Optional[Tuple[float, float, float, float]] = None) -> Optional[Tuple[float, float, float, float]]:
        """Returns the normalized box (x, y, w, h) to mask for the given mode."""
        if mode == PresenterMaskMode.AUTO:
            return self.detected_box or PRESET_BOXES[PresenterMaskMode.BOTTOM_RIGHT]
        if mode in PRESET_BOXES:
            return PRESET_BOXES[mode]
        if mode == PresenterMaskMode.CUSTOM and custom_box:
            return custom_box
        return None


def detect_canvas_crop_box(frames: List[np.ndarray], threshold: int = 18) -> Optional[Tuple[int, int, int, int]]:
    """Detects black bars / letterboxing around presentation slides.
    
    Returns (x1, y1, x2, y2) in pixel coordinates, or None if no significant cropping is needed.
    """
    if not frames:
        return None

    h, w = frames[0].shape[:2]
    crop_boxes = []

    # Sample up to 10 frames to avoid high overhead
    sample_frames = frames if len(frames) <= 10 else [frames[i] for i in np.linspace(0, len(frames) - 1, 10, dtype=int)]

    for f in sample_frames:
        if len(f.shape) == 3:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        else:
            gray = f

        # Row and col brightness profiles
        row_means = np.mean(gray, axis=1)
        col_means = np.mean(gray, axis=0)

        valid_rows = np.where(row_means > threshold)[0]
        valid_cols = np.where(col_means > threshold)[0]

        if len(valid_rows) > 0 and len(valid_cols) > 0:
            y1 = int(valid_rows[0])
            y2 = int(valid_rows[-1]) + 1
            x1 = int(valid_cols[0])
            x2 = int(valid_cols[-1]) + 1
            crop_boxes.append((x1, y1, x2, y2))

    if not crop_boxes:
        return None

    # Compute median crop box across sampled slides
    x1s = [b[0] for b in crop_boxes]
    y1s = [b[1] for b in crop_boxes]
    x2s = [b[2] for b in crop_boxes]
    y2s = [b[3] for b in crop_boxes]

    med_x1 = int(np.median(x1s))
    med_y1 = int(np.median(y1s))
    med_x2 = int(np.median(x2s))
    med_y2 = int(np.median(y2s))

    # Calculate coverage
    cropped_area = (med_x2 - med_x1) * (med_y2 - med_y1)
    total_area = w * h

    # Only crop if at least 2% of black borders exist and the crop region is valid
    if (cropped_area / total_area) < 0.98 and (med_x2 - med_x1) > 200 and (med_y2 - med_y1) > 150:
        return (med_x1, med_y1, med_x2, med_y2)

    return None
