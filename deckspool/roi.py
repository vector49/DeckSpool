# Copyright (c) 2026 Vignesh Rajan (vector49)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Region of Interest (ROI), Chrome Masking, Sidebar Detection, and Canvas Auto-Cropping."""

import logging
from typing import Tuple, Optional, List
import numpy as np
import cv2

from deckspool.config import PresenterMaskMode

logger = logging.getLogger(__name__)

# Default corner presets (x, y, w, h) in normalized coordinates [0.0, 1.0]
PRESET_BOXES = {
    PresenterMaskMode.TOP_RIGHT: (0.72, 0.05, 0.26, 0.35),
    PresenterMaskMode.BOTTOM_RIGHT: (0.72, 0.60, 0.26, 0.35),
    PresenterMaskMode.BOTTOM_LEFT: (0.02, 0.60, 0.26, 0.35),
    PresenterMaskMode.TOP_LEFT: (0.02, 0.05, 0.26, 0.35),
    PresenterMaskMode.SPLIT_RIGHT_50: (0.50, 0.00, 0.50, 1.00),
    PresenterMaskMode.SPLIT_LEFT_50: (0.00, 0.00, 0.50, 1.00),
}

# Extended candidate boxes including full-height Zoom/Teams right-hand participant strip and 50/50 splits
CANDIDATE_PIP_BOXES = [
    ("top_right", (0.70, 0.04, 0.28, 0.36)),
    ("bottom_right", (0.70, 0.60, 0.28, 0.36)),
    ("bottom_left", (0.02, 0.60, 0.28, 0.36)),
    ("top_left", (0.02, 0.04, 0.28, 0.36)),
    ("right_sidebar", (0.82, 0.00, 0.18, 1.00)),
    ("split_right_50", (0.50, 0.00, 0.50, 1.00)),
    ("split_left_50", (0.00, 0.00, 0.50, 1.00)),
]


def mask_player_chrome(frame: np.ndarray, top_pct: float = 0.03, bottom_pct: float = 0.05) -> np.ndarray:
    """Masks top and bottom player chrome (titles, seekbars, controls) with black."""
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
    """Detects presenter PIP webcam box, 50/50 split-screen, or Zoom/Teams participant sidebar using EMA motion variance and structural lines."""

    def __init__(self, grid_size: int = 16, ema_alpha: float = 0.15):
        self.grid_size = grid_size
        self.ema_alpha = ema_alpha
        self._prev_frame: Optional[np.ndarray] = None
        self._motion_ema: Optional[np.ndarray] = None
        self._samples_count: int = 0
        self._recent_faces: List[Tuple[float, float, float, float, float]] = []
        self.detected_box: Optional[Tuple[float, float, float, float]] = None

    def update(
        self,
        gray_360p: np.ndarray,
        detected_faces: Optional[List[Tuple[float, float, float, float, float]]] = None
    ) -> None:
        """Update Exponential Moving Average (EMA) motion statistics with a new 360p grayscale frame."""
        h, w = gray_360p.shape[:2]
        if detected_faces:
            self._recent_faces.extend(detected_faces)
            if len(self._recent_faces) > 30:
                self._recent_faces = self._recent_faces[-30:]

        if self._prev_frame is None or self._prev_frame.shape != gray_360p.shape:
            self._prev_frame = gray_360p.copy()
            self._motion_ema = np.zeros((h, w), dtype=np.float32)
            self._samples_count = 0
            return

        # Frame difference with compression noise floor
        diff = cv2.absdiff(gray_360p, self._prev_frame).astype(np.float32)
        diff[diff < 10.0] = 0.0

        # Exponential Moving Average so old terminal scrolling decays over time
        if self._samples_count == 0:
            self._motion_ema = diff.copy()
        else:
            cv2.accumulateWeighted(diff, self._motion_ema, self.ema_alpha)

        self._samples_count += 1
        self._prev_frame = gray_360p.copy()

        # 1. Check for 50/50 split-screen layout (e.g. left slide, right webcam)
        split_box = self._detect_split_screen(gray_360p)
        if split_box is not None:
            self.detected_box = split_box
            return

        # 2. Check for Zoom/Teams right-hand vertical participant strip boundary
        sidebar_box = self._detect_zoom_sidebar(gray_360p)
        if sidebar_box is not None:
            self.detected_box = sidebar_box
            return

        # 3. Check for corner PIP boxes
        if self._samples_count >= 4:
            self._resolve_candidate_box(h, w)

    def _detect_split_screen(self, gray_360p: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        """Detects 50/50 side-by-side presentation layout (e.g. left slide, right speaker webcam, or vice versa)."""
        h, w = gray_360p.shape[:2]
        mid = w // 2

        # Check center vertical column gradient (around 47% to 53% of width)
        x_start = int(w * 0.47)
        x_end = int(w * 0.53)
        col_diffs = np.mean(np.abs(np.diff(gray_360p[:, x_start:x_end].astype(np.float32), axis=1)), axis=0)
        max_center_grad = float(np.max(col_diffs)) if len(col_diffs) > 0 else 0.0

        # 1. Face-based ground truth (presenter webcam face cannot jump between presentation halves)
        if self._recent_faces:
            right_faces = sum(1 for f in self._recent_faces if (f[0] + f[2] / 2.0) >= 0.55)
            left_faces = sum(1 for f in self._recent_faces if (f[0] + f[2] / 2.0) <= 0.45)
            total_f = len(self._recent_faces)
            if total_f >= 2:
                if right_faces > left_faces and (right_faces / float(total_f)) >= 0.55:
                    if max_center_grad > 15.0 or (
                        self._motion_ema is not None and (
                            float(np.sum(self._motion_ema[:, mid:])) / max(1e-5, float(np.sum(self._motion_ema))) >= 0.30
                        )
                    ):
                        return (0.50, 0.00, 0.50, 1.00)

                if left_faces > right_faces and (left_faces / float(total_f)) >= 0.55:
                    if max_center_grad > 15.0 or (
                        self._motion_ema is not None and (
                            float(np.sum(self._motion_ema[:, :mid])) / max(1e-5, float(np.sum(self._motion_ema))) >= 0.30
                        )
                    ):
                        return (0.00, 0.00, 0.50, 1.00)

        # 2. Pure motion & boundary check (if no face detected or camera off)
        if self._motion_ema is not None and self._samples_count >= 4 and not self._recent_faces:
            total_motion = float(np.sum(self._motion_ema))
            if total_motion > 1e-3 and max_center_grad > 18.0:
                left_motion = float(np.sum(self._motion_ema[:, :mid])) / total_motion
                right_motion = float(np.sum(self._motion_ema[:, mid:])) / total_motion
                if right_motion >= 0.70:
                    return (0.50, 0.00, 0.50, 1.00)
                if left_motion >= 0.70:
                    return (0.00, 0.00, 0.50, 1.00)

        return None

        return None

    def _detect_zoom_sidebar(self, gray_360p: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        """Detects a persistent right-side participant sidebar (e.g. Zoom/Teams vertical strip)."""
        h, w = gray_360p.shape[:2]
        x_start = int(w * 0.82)
        x_end = int(w * 0.90)

        # Look for a sharp vertical transition / dark strip on the right 12-18% of the screen
        right_strip = gray_360p[:, int(w * 0.85):]
        main_canvas = gray_360p[:, :int(w * 0.80)]

        # In Zoom sidebars, the strip background is dark or split into stacked boxes with dark borders
        dark_border_ratio = np.count_nonzero(right_strip < 30) / float(max(1, right_strip.size))
        col_grad = np.mean(np.abs(np.diff(gray_360p[:, x_start:x_end].astype(np.float32), axis=1)), axis=0)

        if np.max(col_grad) > 18.0 and dark_border_ratio > 0.12 and np.mean(main_canvas) > 35.0:
            return (0.83, 0.0, 0.17, 1.0)
        return None

    def _resolve_candidate_box(self, h: int, w: int) -> None:
        """Check if one of the candidate PIP or sidebar regions exhibits concentrated recent motion."""
        if self._motion_ema is None or self._samples_count == 0:
            return

        total_motion = float(np.sum(self._motion_ema))
        if total_motion < 1e-3:
            return

        best_box = None
        highest_density_ratio = 0.0
        frame_area = float(w * h)

        for name, box in CANDIDATE_PIP_BOXES:
            bx, by, bw, bh = box
            x1, y1 = int(bx * w), int(by * h)
            x2, y2 = int((bx + bw) * w), int((by + bh) * h)

            region_motion = float(np.sum(self._motion_ema[y1:y2, x1:x2]))
            region_area = float(max(1, (x2 - x1) * (y2 - y1)))

            motion_fraction = region_motion / (total_motion + 1e-5)
            area_fraction = region_area / frame_area
            density_ratio = motion_fraction / area_fraction

            # Trigger if region holds >= 32% of recent motion with >= 2.2x average density
            if motion_fraction >= 0.32 and density_ratio >= 2.2:
                if density_ratio > highest_density_ratio:
                    highest_density_ratio = density_ratio
                    best_box = box

        if best_box:
            self.detected_box = best_box

    def get_mask_box(
        self,
        mode: PresenterMaskMode,
        custom_box: Optional[Tuple[float, float, float, float]] = None
    ) -> Optional[Tuple[float, float, float, float]]:
        """Returns the normalized box (x, y, w, h) to mask for the given mode."""
        if mode == PresenterMaskMode.AUTO:
            return self.detected_box or PRESET_BOXES[PresenterMaskMode.BOTTOM_RIGHT]
        if mode in PRESET_BOXES:
            return PRESET_BOXES[mode]
        if mode == PresenterMaskMode.CUSTOM and custom_box:
            return custom_box
        return None


def detect_canvas_crop_box(frames: List[np.ndarray], threshold: int = 18) -> Optional[Tuple[int, int, int, int]]:
    """Detects black bars / letterboxing around presentation slides or split-screen crops."""
    if not frames:
        return None

    h, w = frames[0].shape[:2]

    # 1. Quick split-screen mask check: if majority of frames have one half completely masked (< 2.0 mean)
    # while the other half has presentation content (> 5.0 mean), crop to that half
    sample_frames = (
        frames
        if len(frames) <= 10
        else [frames[i] for i in np.linspace(0, len(frames) - 1, 10, dtype=int)]
    )
    sample_grays = [
        cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f
        for f in sample_frames
    ]

    right_masked_count = sum(1 for g in sample_grays if np.mean(g[:, w // 2:]) < 2.0 and np.mean(g[:, :w // 2]) > 5.0)
    left_masked_count = sum(1 for g in sample_grays if np.mean(g[:, :w // 2]) < 2.0 and np.mean(g[:, w // 2:]) > 5.0)
    n_samples = len(sample_grays)

    if (right_masked_count / float(n_samples)) >= 0.50:
        return (0, 0, w // 2, h)
    if (left_masked_count / float(n_samples)) >= 0.50:
        return (w // 2, 0, w, h)

    crop_boxes = []

    for f in sample_frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f

        # Adaptive thresholding for dark presentations
        top_edge = float(np.mean(gray[:5, :]))
        bot_edge = float(np.mean(gray[-5:, :]))
        left_edge = float(np.mean(gray[:, :5]))
        right_edge = float(np.mean(gray[:, -5:]))
        border_lum = min(top_edge, bot_edge, left_edge, right_edge)
        eff_thresh = threshold if threshold != 18 else max(6.0, border_lum + 8.0)

        row_means = np.mean(gray, axis=1)
        col_means = np.mean(gray, axis=0)

        valid_rows = np.where(row_means > eff_thresh)[0]
        valid_cols = np.where(col_means > eff_thresh)[0]

        if len(valid_rows) > 0 and len(valid_cols) > 0:
            y1 = int(valid_rows[0])
            y2 = int(valid_rows[-1]) + 1
            x1 = int(valid_cols[0])
            x2 = int(valid_cols[-1]) + 1
            crop_boxes.append((x1, y1, x2, y2))

    if not crop_boxes:
        return None

    med_x1 = int(np.median([b[0] for b in crop_boxes]))
    med_y1 = int(np.median([b[1] for b in crop_boxes]))
    med_x2 = int(np.median([b[2] for b in crop_boxes]))
    med_y2 = int(np.median([b[3] for b in crop_boxes]))

    cropped_area = (med_x2 - med_x1) * (med_y2 - med_y1)
    total_area = w * h

    if (cropped_area / total_area) < 0.98 and (med_x2 - med_x1) > 200 and (med_y2 - med_y1) > 150:
        return (med_x1, med_y1, med_x2, med_y2)

    return None
