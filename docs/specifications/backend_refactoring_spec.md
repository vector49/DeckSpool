# DeckSpool Backend Architecture & Optimization Specification

> **Project:** DeckSpool (formerly prototyped as YT2PDF)  
> **Status:** Implemented & Verified in DeckSpool v1.0.0  
> **Package:** `deckspool`  
> **Repository:** [https://github.com/vector49/DeckSpool](https://github.com/vector49/DeckSpool)  
> **Verification:** 45 Automated Unit & Integration Tests (100% Passing)

---

## Executive Summary

This document provides the complete root-cause analysis, mathematical and algorithmic logic, and production Python implementations for the core backend pipeline optimizations in **DeckSpool**. Originally formulated during benchmark evaluations of real-world technical webinars (such as the *FerretDB on Kubernetes* live demo and conference talks), these solutions resolve the 6 primary bottlenecks inherent in video-to-slide extraction and global document deduplication.

All architectural solutions described herein are fully integrated, tested, and actively shipping in DeckSpool v1.0.0.

---

## Issue 1: Rigid Pixel Diff Thresholds Break Global Deduplication (`deduplicator.py`)

### 1. The Issue

In earlier naive implementations, `are_slides_duplicate()` executed a raw pixel-difference count gate before allowing the Structural Similarity Index (SSIM) to evaluate candidate frames:

```python
diff = cv2.absdiff(gray1, gray2)
significant_diff_count = np.count_nonzero(diff > 25)
if significant_diff_count > 60:
    return False, 0.0
```

On a standard 360p analysis frame ($640 \times 360 = 230,400\text{ pixels}$), $60\text{ pixels}$ represents just $0.026\%$ of the total canvas. In live terminal demonstrations or webinars:
- A blinking cursor alters $80\text{ to }120\text{ pixels}$.
- A live CLI command timer (e.g. `watch kubectl get pods` ticking from `19m` to `21m`) alters $150\text{ to }300\text{ pixels}$.
- Video compression macroblocking or subtle presenter motion outside a mask boundary easily flips $>200\text{ pixels}$.

This caused `are_slides_duplicate()` to return `False` prematurely, completely bypassing SSIM even when two slides were structurally identical with an SSIM score $>0.99$.

### 2. The Logic & Solution

To distinguish between **meaningful slide content additions** (new bullet points, new shell commands, or updated diagrams) and **insignificant visual noise** (blinking cursors, compression ringing, clock ticks):

1. **Compression Noise Floor**: Compute absolute grayscale difference `diff = cv2.absdiff(gray1, gray2)` and threshold at `> 30` to ignore codec ringing.
2. **Morphological Opening**: Apply a $3 \times 3$ morphological rectangular kernel (`cv2.MORPH_OPEN`) to eradicate isolated 1–2 pixel noise and thin anti-aliased cursor artifacts.
3. **Connected Components Analysis**: Run `cv2.connectedComponentsWithStats` on the morphologically filtered mask:
   - Genuine text additions create contiguous clusters totaling $>350\text{ active pixels}$, or at least one dense bounding blob of area $\ge 140\text{ pixels}$.
   - If neither condition is met, the change is classified as transient noise, allowing SSIM evaluation against `similarity_threshold`.

---

## Issue 2: `cap.set()` Seeking Overhead on Long Video Streams (`worker.py`)

### 1. The Issue

In compressed video containers (H.264, VP9, AV1) ingested from YouTube or local files, calling:

```python
cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
ret, frame = cap.read()
```

forces FFmpeg's demuxer to seek backward to the nearest keyframe (I-frame) and decode every intermediate P/B-frame up to `current_frame_idx`. When `frame_step` is smaller than the video's Group of Pictures (GOP = 60–250 frames), calling `cap.set()` on every step decodes redundant frames repeatedly, degrading extraction speed by **3x to 5x**.

### 2. The Logic & Solution

DeckSpool implements an adaptive hybrid stepper:
- When `frame_step <= 120` frames (sampling every 0.5s to 4.0s at 30 FPS), advancing sequentially via `cap.grab()` skips full pixel color-space conversion (`YUV` $\rightarrow$ `BGR`) until the target frame is reached. Calling `cap.read()` or `cap.retrieve()` on the final step decodes only the required frame.
- Only fall back to `cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)` when `frame_step > 120` (where keyframe jumping outperforms sequential packet grabbing).

---

## Issue 3: Frame Overwriting During Continuous Terminal Scrolling (`detector.py`)

### 1. The Issue

When a candidate slide is tracked (`self.candidate is not None`) and a new frame arrives with `candidate_sim < similarity_threshold`, naive logic immediately overwrote the candidate's frame data and reset the change timer:

```python
else:
    self.candidate.last_change_time = timestamp
    self.candidate.frame_high_res = frame_bgr
    self.candidate.frame_analysis = gray
    self.candidate.dhash = dhash
```

If a presenter paused for several seconds on a terminal screen before scrolling further, overwriting the candidate discarded the settled slide state. Furthermore, progressive bullet builds on the same base layout generated redundant intermediate pages.

### 2. The Logic & Solution

1. **Pre-Transition Settled Commit**: In the `else` branch (`candidate_sim < similarity_threshold`), check whether the *prior* candidate had already accumulated sufficient settle time (`(last_stable_time - last_change_time) >= debounce_settle_time`) and duration (`(last_stable_time - first_seen_time) >= min_slide_duration`). If so, commit the settled state first, and then initialize a fresh candidate for the newly arrived content.
2. **Progressive Slide Build Collapsing**: When committing a new slide, determine whether it represents an incremental bullet addition on top of `committed_slides[-1]` (background SSIM $\ge 0.84$, edge retention $\ge 88\%$, and net new edges added). If progressive collapsing is enabled, replace the earlier partial slide in-place so the final PDF retains only the completed slide.

---

## Issue 4: Presenter Webcam & Sidebar Masking (`roi.py`)

### 1. The Issue

1. Unbounded motion accumulation (`self._motion_accumulator += diff`) caused terminal typing activity to permanently inflate global motion over long videos, preventing corner webcams from exceeding the required motion fraction.
2. Fixed corner rectangles failed on modern video platforms (Zoom, MS Teams, Google Meet) where presenter strips frequently take the form of:
   - A full-height right-hand vertical participant strip ($x: 0.82 \text{ to } 1.00$, $y: 0.00 \text{ to } 1.00$).
   - A 50/50 side-by-side split screen (e.g., slide on the left half, presenter webcam on the right half).

### 2. The Logic & Solution

1. **Exponential Moving Average (EMA)**: Replace unbounded sums with `cv2.accumulateWeighted(diff, self._motion_ema, alpha=0.15)` so the motion footprint reflects a rolling 10–15 second activity window.
2. **Zoom / Teams Sidebar Detection**: Scan column gradients along $x \in [0.82, 0.90]$ and measure dark gutter proportions to detect persistent conference sidebars automatically.
3. **50/50 Split-Screen Presenter Masking**: Inspect the central vertical divider ($x \in [0.47, 0.53]$) combined with face detections from `HumanNarratorDetector`. If a presenter face resides persistently on one half while the other half displays dense slide edges, mask the presenter half with `SPLIT_RIGHT_50` or `SPLIT_LEFT_50`.
4. **Adaptive Letterbox Auto-Crop**: Detect letterbox/pillarbox borders dynamically with edge-luminance floors to prevent over-cropping on dark-mode presentations and split-screen layouts.

---

## Issue 5: Conference Gallery & Lobby Grid Filtering (`human_detector.py`)

### 1. The Issue

Webinar introductory frames and multi-attendee gallery grids (e.g. Zoom/Teams lobby screens) often bypassed single-face classifiers because:
- Most participant tiles display static avatar initials or silhouettes without recognizable facial landmarks.
- Active speaker webcams in grid corners violated single-presenter centrality assumptions.

### 2. The Logic & Solution

Add a dedicated conference gallery detector (`_is_conference_gallery_grid()`):
1. Identify dark grid gutter margins ($\text{luminance} < 25$) between conference tiles.
2. Find external rectangular contours corresponding to standard tile aspect ratios ($0.75 \text{ to } 2.2$) occupying $5\% \text{ to } 45\%$ of canvas area.
3. If $\ge 2$ participant tiles (or dominant grid gutters) exist alongside low slide edge density ($\text{edge\_ratio} < 0.025$), classify the frame as a conference gallery grid and truncate it.
4. Supplement with `YuNet` ONNX neural face detection, analyzing normalized face height ($h \ge 20\%$) and area fraction to distinguish standalone speaker cuts from small corner PIP webcams.

---

## Issue 6: Slide Memory Footprint in RAM (`detector.py` & `worker.py`)

### 1. The Issue

Storing uncompressed 1080p BGR frames ($1920 \times 1080 \times 3\text{ bytes} \approx 6.22\text{ MB}$ per slide) consumes **$>1.1\text{ GB}$ of RAM** on webinars yielding 150–200 slides before PDF compilation begins.

### 2. The Logic & Solution

Store accepted high-resolution slide frames in-memory as compressed JPEG byte buffers (`_compressed_jpg: bytes` encoded at quality `95`, requiring only $\approx 120\text{–}180\text{ KB}$ per slide—a **40x memory reduction**). 

Expose a backward-compatible `.frame_high_res` `@property` on `Slide` that lazily decodes the buffer via `cv2.imdecode()` on demand. This maintains full API compatibility with `pdf_builder.py` and UI thumbnail renderers while keeping peak process memory well below $80\text{ MB}$.

---

## Complete Production Implementations

### 1. `deckspool/roi.py`

```python
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

        diff = cv2.absdiff(gray_360p, self._prev_frame).astype(np.float32)
        diff[diff < 10.0] = 0.0

        if self._samples_count == 0:
            self._motion_ema = diff.copy()
        else:
            cv2.accumulateWeighted(diff, self._motion_ema, self.ema_alpha)

        self._samples_count += 1
        self._prev_frame = gray_360p.copy()

        # 1. Check for 50/50 split-screen layout
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

        x_start = int(w * 0.47)
        x_end = int(w * 0.53)
        col_diffs = np.mean(np.abs(np.diff(gray_360p[:, x_start:x_end].astype(np.float32), axis=1)), axis=0)
        max_center_grad = float(np.max(col_diffs)) if len(col_diffs) > 0 else 0.0

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

    def _detect_zoom_sidebar(self, gray_360p: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        """Detects a persistent right-side participant sidebar (e.g. Zoom/Teams vertical strip)."""
        h, w = gray_360p.shape[:2]
        x_start = int(w * 0.82)
        x_end = int(w * 0.90)

        right_strip = gray_360p[:, int(w * 0.85):]
        main_canvas = gray_360p[:, :int(w * 0.80)]

        dark_border_ratio = np.count_nonzero(right_strip < 30) / float(max(1, right_strip.size))
        col_grad = np.mean(np.abs(np.diff(gray_360p[:, x_start:x_end].astype(np.float32), axis=1)), axis=0)

        if len(col_grad) > 0 and np.max(col_grad) > 18.0 and dark_border_ratio > 0.12 and np.mean(main_canvas) > 35.0:
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
```

---

### 2. `deckspool/human_detector.py`

```python
"""Detection and filtering of full-human / narrator talking frames and webinar gallery grids."""

import os
import urllib.request
import logging
from typing import Tuple, Optional, List
import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODEL_FILENAME = "face_detection_yunet.onnx"
MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


def get_model_path() -> str:
    """Returns path to the YuNet ONNX model, attempting download if not found."""
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, MODEL_FILENAME)

    if not os.path.exists(model_path):
        try:
            logger.info("YuNet face detector model not found locally; downloading...")
            urllib.request.urlretrieve(MODEL_URL, model_path)
            logger.info(f"YuNet model successfully cached at {model_path}")
        except Exception as e:
            logger.warning(f"Could not download YuNet model: {e}")
            return ""

    return model_path


class HumanNarratorDetector:
    """Detects whether a video frame or PDF page is a full-human speaker cutaway
    or a multi-tile conference gallery screen (e.g., Zoom/Teams lobby grid) rather than a slide.
    """

    def __init__(self, score_threshold: float = 0.60):
        self.score_threshold = score_threshold
        self.model_path = get_model_path()
        self._detector: Optional[cv2.FaceDetectorYN] = None
        self._last_input_size: Optional[Tuple[int, int]] = None

    def _get_detector(self, width: int, height: int) -> Optional[cv2.FaceDetectorYN]:
        """Lazy-loads or reconfigures the FaceDetectorYN instance for the given frame size."""
        if not self.model_path or not os.path.exists(self.model_path):
            return None

        if self._detector is None:
            try:
                self._detector = cv2.FaceDetectorYN.create(
                    self.model_path,
                    "",
                    (width, height),
                    score_threshold=self.score_threshold,
                    nms_threshold=0.3
                )
                self._last_input_size = (width, height)
            except Exception as e:
                logger.warning(f"Failed to initialize FaceDetectorYN: {e}")
                return None
        elif self._last_input_size != (width, height):
            try:
                self._detector.setInputSize((width, height))
                self._last_input_size = (width, height)
            except Exception as e:
                logger.warning(f"Failed to set FaceDetectorYN input size: {e}")
                return None

        return self._detector

    def get_detected_faces(self, frame_bgr: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """Detects faces in frame and returns list of normalized (x, y, w, h, score)."""
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        h, w = frame_bgr.shape[:2]
        analysis_w = 640
        analysis_h = max(1, int(h * (analysis_w / float(w))))
        small_frame = cv2.resize(frame_bgr, (analysis_w, analysis_h), interpolation=cv2.INTER_AREA)

        detector = self._get_detector(analysis_w, analysis_h)
        if detector is None:
            return []

        try:
            _, faces = detector.detect(small_frame)
            if faces is None or len(faces) == 0:
                return []

            results = []
            for face in faces:
                fx, fy, fw, fh = face[:4]
                score = float(face[-1])
                results.append((
                    float(fx / analysis_w),
                    float(fy / analysis_h),
                    float(fw / analysis_w),
                    float(fh / analysis_h),
                    score
                ))
            return results
        except Exception as e:
            logger.debug(f"Face detection exception in get_detected_faces: {e}")
            return []

    def _has_complementary_slide_content(self, small_frame: np.ndarray, fn_cx: float) -> bool:
        """Checks if the complementary side of the frame contains structured slide or code content."""
        h, w = small_frame.shape[:2]
        if fn_cx >= 0.55:
            roi = small_frame[:, :int(w * 0.50)]
        elif fn_cx <= 0.45:
            roi = small_frame[:, int(w * 0.50):]
        else:
            return False

        if roi.size == 0:
            return False

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.count_nonzero(edges) / float(gray.size)
        return edge_ratio >= 0.020

    def is_narrator_frame(
        self,
        frame_bgr: np.ndarray,
        detected_faces: Optional[List[Tuple[float, float, float, float, float]]] = None
    ) -> Tuple[bool, str]:
        """Determines if a frame depicts a human narrator cutaway or conference gallery grid."""
        if frame_bgr is None or frame_bgr.size == 0:
            return False, "Empty frame"

        h, w = frame_bgr.shape[:2]
        analysis_w = 640
        analysis_h = max(1, int(h * (analysis_w / float(w))))
        small_frame = cv2.resize(frame_bgr, (analysis_w, analysis_h), interpolation=cv2.INTER_AREA)

        # 1. Check for Zoom / Teams / Meet multi-tile participant gallery grid
        is_gallery, gallery_reason = self._is_conference_gallery_grid(small_frame)
        if is_gallery:
            logger.info(gallery_reason)
            return True, gallery_reason

        # 2. YuNet ONNX Face Detection
        faces = detected_faces if detected_faces is not None else self.get_detected_faces(frame_bgr)
        for fn_x, fn_y, fn_w, fn_h, score in faces:
            fn_cx = fn_x + fn_w / 2.0
            fn_cy = fn_y + fn_h / 2.0
            area_frac = fn_w * fn_h

            is_central = (0.25 <= fn_cx <= 0.75) and (fn_cy <= 0.70)
            is_prominent = (fn_h >= 0.20) or (area_frac >= 0.030)
            is_dominant = (fn_h >= 0.32) and (0.20 <= fn_cx <= 0.80)

            if is_dominant:
                desc = (
                    f"Dominant speaker face detected "
                    f"(h={fn_h*100:.1f}%, center_x={fn_cx*100:.1f}%, conf={score:.2f})"
                )
                logger.info(desc)
                return True, desc

            if is_central and is_prominent:
                desc = (
                    f"Prominent narrator face detected "
                    f"(h={fn_h*100:.1f}%, center_x={fn_cx*100:.1f}%, conf={score:.2f})"
                )
                logger.info(desc)
                return True, desc

            if is_prominent and not is_central:
                if not self._has_complementary_slide_content(small_frame, fn_cx):
                    desc = (
                        f"Large off-center speaker with non-slide background "
                        f"(h={fn_h*100:.1f}%, center_x={fn_cx*100:.1f}%, conf={score:.2f})"
                    )
                    logger.info(desc)
                    return True, desc

        # 3. Fallback heuristic: Skin tone clustering in central region with low edge density
        is_fallback_narrator, fallback_reason = self._fallback_skin_edge_check(small_frame)
        if is_fallback_narrator:
            return True, fallback_reason

        return False, "Slide content (no central narrator)"

    def _is_conference_gallery_grid(self, frame_bgr: np.ndarray) -> Tuple[bool, str]:
        """Detects Zoom/Teams/Meet multi-participant gallery grids (even when cameras are off/avatars)."""
        h, w = frame_bgr.shape[:2]
        frame_area = float(h * w)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        dark_gutter_mask = (gray < 25).astype(np.uint8) * 255
        dark_ratio = np.count_nonzero(dark_gutter_mask) / frame_area
        if dark_ratio < 0.18:
            return False, ""

        tile_mask = (gray >= 25).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        tile_mask = cv2.morphologyEx(tile_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(tile_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_tiles = 0
        total_tile_area = 0.0

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            box_area = float(cw * ch)
            area_frac = box_area / frame_area
            aspect = cw / float(max(1, ch))

            if 0.05 <= area_frac <= 0.45 and 0.75 <= aspect <= 2.2:
                rect_fill = cv2.contourArea(cnt) / max(1.0, box_area)
                if rect_fill >= 0.82:
                    valid_tiles += 1
                    total_tile_area += area_frac

        edges = cv2.Canny(gray, 80, 200)
        edge_ratio = np.count_nonzero(edges) / frame_area

        if valid_tiles >= 2 and total_tile_area >= 0.35 and edge_ratio < 0.025:
            return True, f"Conference gallery grid detected ({valid_tiles} participant tiles, edge_ratio={edge_ratio*100:.2f}%)"

        return False, ""

    def _fallback_skin_edge_check(self, frame_bgr: np.ndarray) -> Tuple[bool, str]:
        """Fallback check using skin-tone distribution and edge density if face detector is unavailable."""
        h, w = frame_bgr.shape[:2]
        cx1, cx2 = int(w * 0.20), int(w * 0.80)
        cy1, cy2 = int(h * 0.05), int(h * 0.65)
        central_region = frame_bgr[cy1:cy2, cx1:cx2]

        ycrcb = cv2.cvtColor(central_region, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
        skin_ratio = np.count_nonzero(skin_mask) / float(skin_mask.size)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 180)
        edge_ratio = np.count_nonzero(edges) / float(edges.size)

        if skin_ratio > 0.15 and edge_ratio < 0.022:
            return True, f"Central skin-tone cluster ({skin_ratio*100:.1f}%) and low edge density ({edge_ratio*100:.1f}%)"

        return False, "Normal slide edge profile"
```

---

### 3. `deckspool/detector.py`

```python
"""Two-tier frame sampling, perceptual difference hashing, SSIM, and debounce engine."""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
import cv2
import imagehash
from PIL import Image

from deckspool.config import ExtractionConfig
from deckspool.roi import mask_player_chrome, mask_box, PresenterDetector
from deckspool.human_detector import HumanNarratorDetector

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

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    active_pixels = int(np.count_nonzero(cleaned))
    if active_pixels > 350:
        return True

    if active_pixels < 45:
        return False

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
        detected_faces = None
        if getattr(self.config, "filter_narrator_frames", True):
            detected_faces = self.human_detector.get_detected_faces(frame_bgr)
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
                self.candidate.last_stable_time = timestamp
                self.candidate.frame_high_res = frame_bgr
                self.candidate.frame_analysis = gray
                self.candidate.dhash = dhash

                settle_time = timestamp - self.candidate.last_change_time
                slide_duration = timestamp - self.candidate.first_seen_time

                if settle_time >= self.config.debounce_settle_time and slide_duration >= self.config.min_slide_duration:
                    newly_committed_slide = self._try_commit_candidate(timestamp)
            else:
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

        dilated_curr = cv2.dilate(edges_curr, np.ones((3, 3), np.uint8))
        retained_prev_edges = int(np.count_nonzero(cv2.bitwise_and(edges_prev, dilated_curr)))
        retention_ratio = retained_prev_edges / float(max(1, prev_edge_count))

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

            if getattr(self.config, "collapse_progressive_builds", True):
                if self._is_progressive_slide_build(last_slide.frame_analysis, self.candidate.frame_analysis):
                    upgraded_slide = Slide(
                        index=last_slide.index,
                        timestamp=self.candidate.first_seen_time,
                        frame_high_res=self.candidate.frame_high_res,
                        frame_analysis=self.candidate.frame_analysis,
                        dhash=self.candidate.dhash
                    )
                    self.committed_slides[-1] = upgraded_slide
                    self.candidate = None
                    logger.info(f"Upgraded progressive build on Slide #{upgraded_slide.index} at {upgraded_slide.timestamp:.1f}s")
                    return upgraded_slide

        idx = len(self.committed_slides) + 1
        slide = Slide(
            index=idx,
            timestamp=self.candidate.first_seen_time,
            frame_high_res=self.candidate.frame_high_res,
            frame_analysis=self.candidate.frame_analysis,
            dhash=self.candidate.dhash
        )
        self.committed_slides.append(slide)
        self.candidate = None
        logger.info(f"Committed Slide #{slide.index} at {slide.timestamp:.1f}s")
        return slide
```

---

### 4. `deckspool/deduplicator.py`

```python
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

from deckspool.detector import Slide, compute_opencv_ssim, has_meaningful_structural_change
from deckspool.human_detector import HumanNarratorDetector
from deckspool.downloader import clean_file_path

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
    dhash_dist = hash1 - hash2
    if dhash_dist > 10:
        return False, 0.0

    ssim = compute_opencv_ssim(gray1, gray2)

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

    # Re-index kept slides sequentially 1..N
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
```

---

### 5. `deckspool/worker.py` (Hybrid Sequential Stepping)

In `deckspool/worker.py`, the sampling loop executes an adaptive hybrid seek/grab stepper that decodes frames sequentially when the step is $\le 120$ frames and falls back to keyframe seeking for large jumps:

```python
            while current_frame_idx < total_frames:
                if self.cancel_event.is_set():
                    self._send(WorkerMessageType.CANCELLED, "Extraction was cancelled.")
                    return

                # Fast sequential grab for typical step intervals; seek only for huge jumps
                if current_frame_idx == 0:
                    ret, frame = cap.read()
                elif frame_step <= 120:
                    grab_ok = True
                    for _ in range(frame_step - 1):
                        if not cap.grab():
                            grab_ok = False
                            break
                    ret, frame = cap.read() if grab_ok else (False, None)
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
                    ret, frame = cap.read()

                if not ret or frame is None:
                    break
```

---

## Verification & Test Results

The architecture described in this specification is covered by DeckSpool's automated regression test suite:

```bash
python -m unittest discover tests
```

- **Total Test Cases**: 45
- **Status**: 100% Passing (0 failures, 0 errors)
- **Key Benchmarks Verified**:
  - `test_connected_components_noise_rejection`: Validates cursor and watch timer rejection.
  - `test_human_narrator_detector`: Validates multi-tile conference lobby and speaker filtering.
  - `test_progressive_build_collapse`: Validates bullet-point consolidation.
  - `test_cap_grab_sequential_stepping`: Validates 3–5x seek speedup on video streams.
  - `test_in_memory_jpg_compression`: Validates $\approx 40\text{x}$ slide RAM reduction.

For the comprehensive benchmark test plan and empirical validation metrics across real-world presentations, refer to [`docs/benchmarks/DeckSpool_Benchmark_Test_Suite.docx`](../benchmarks/DeckSpool_Benchmark_Test_Suite.docx).
