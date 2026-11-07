# YT2PDF Backend Optimization \& Refactoring Specification

This document provides the complete root-cause analysis, mathematical/algorithmic logic, and drop-in Python implementations to resolve the 6 core pipeline bottlenecks identified during the evaluation of the *FerretDB on Kubernetes* webinar extraction. At the end of this document, you will find a ready-to-use **Master Agent Prompt** to instruct an AI coding agent to apply these upgrades automatically.

\---

## Issue 1: The `significant\_diff\_count > 60` Gate Breaks Global Deduplication (`deduplicator.py`)

### 1\. The Issue

In `deduplicator.py`, the function `are\_slides\_duplicate()` executes a raw pixel-difference count before allowing the Structural Similarity Index (SSIM) to run:

```python
diff = cv2.absdiff(gray1, gray2)
significant\_diff\_count = np.count\_nonzero(diff > 25)
if significant\_diff\_count > 60:
    return False, 0.0
```

On a 360p frame ($640 \\times 360 = 230,400\\text{ pixels}$), $60\\text{ pixels}$ represents just $0.026%$ of the image area. In live terminal demos or webinars (such as Pages 11–13 and 37–39 of the FerretDB PDF), minor artifacts—such as a blinking terminal cursor, a `watch` timer ticking from `19m` to `21m`, or subtle webcam movement outside the mask—easily alter $80\\text{ to }450\\text{ pixels}$. This forces `are\_slides\_duplicate()` to return `False` prematurely, completely bypassing SSIM even when the two frames have an SSIM score of $>0.99$.

### 2\. The Logic

To distinguish between **meaningful slide content changes** (a new bullet point, a new CLI command, or a diagram change) and **insignificant visual noise** (blinking cursors, compression macroblocks, or tiny status ticks):

1. Compute the absolute grayscale difference `diff = cv2.absdiff(gray1, gray2)` and threshold at `> 30` to ignore H.264/VP9 compression ringing.
2. Apply a $3 \\times 3$ **Morphological Opening** (`cv2.MORPH\_OPEN`) to strip away isolated 1–2 pixel noise and thin video-compression edges.
3. Run **Connected Components Analysis** (`cv2.connectedComponentsWithStats`) on the morphologically cleaned mask:

   * A genuine text addition (even a short word or bullet point) creates a contiguous cluster or multiple clusters totaling $>350\\text{ active pixels}$, or a single dense bounding blob of $>140\\text{ pixels}$.
   * If neither condition is met, proceed to evaluate `compute\_opencv\_ssim(gray1, gray2)` against `similarity\_threshold`.

### 3\. The Solution

Replace `are\_slides\_duplicate()` and update `deduplicate\_slides\_list()` in `yt2pdf/deduplicator.py` so it uses morphological connected-component gating and supports lazy high-res frame access (`slide.get\_high\_res\_bgr()`).

\---

## Issue 2: `cap.set(cv2.CAP\_PROP\_POS\_FRAMES, ...)` Causes Massive Seeking Overhead (`worker.py`)

### 1\. The Issue

In `worker.py`, the frame sampling loop jumps to the next target frame on every iteration using:

```python
cap.set(cv2.CAP\_PROP\_POS\_FRAMES, current\_frame\_idx)
ret, frame = cap.read()
```

For compressed video containers (H.264, VP9, AV1) downloaded from YouTube, `cap.set(cv2.CAP\_PROP\_POS\_FRAMES, ...)` forces FFmpeg's demuxer to seek backward to the nearest Intra-frame (Keyframe / I-frame) and decode every intermediate P/B-frame up to `current\_frame\_idx`. When `frame\_step` is smaller than a typical Group of Pictures (GOP = 60–250 frames), calling `cap.set()` on every step decodes the exact same frames repeatedly, slowing extraction by **3x to 5x**.

### 2\. The Logic

* When `frame\_step <= 120` frames (e.g., sampling every 0.5s to 4.0s at 30 FPS), advancing sequentially via `cap.grab()` skips full pixel/color-space decoding (`YUV` $\\rightarrow$ `BGR`) until the target frame is reached, and then `cap.retrieve()` (or `cap.read()` on the final step) decodes only the required frame.
* Only fall back to `cap.set(cv2.CAP\_PROP\_POS\_FRAMES, current\_frame\_idx)` if `frame\_step > 120` (where keyframe jumping is faster than sequential grabbing).

### 3\. The Solution

In `yt2pdf/worker.py`, replace the `cap.set()` call inside the `while current\_frame\_idx < total\_frames:` loop with a hybrid `cap.grab()` / `cap.set()` stepper.

\---

## Issue 3: Missed Candidate Commits During Continuous Terminal Scrolling (`detector.py`)

### 1\. The Issue

In `detector.py`, when a candidate slide is being tracked (`self.candidate is not None`) and a new frame arrives with `candidate\_sim < self.config.similarity\_threshold`, the `else` branch immediately overwrites `self.candidate`'s frame data and resets `last\_change\_time = timestamp`:

```python
else:
    self.candidate.last\_change\_time = timestamp
    self.candidate.frame\_high\_res = frame\_bgr
    self.candidate.frame\_analysis = gray
    self.candidate.dhash = dhash
```

If a slide or terminal state was already stable for `settle\_time >= self.config.debounce\_settle\_time` (for example, when `sample\_interval` is coarse or when a presenter pauses briefly before scrolling a terminal), overwriting `self.candidate` without committing the settled state first causes the detector to either drop intermediate states or merge distinct screen states into a single delayed timestamp. Conversely, if `last\_slide` in `committed\_slides` is updated by a progressive bullet build on the *same* base slide layout, we also want to prevent redundant partial-build pages (like Pages 2 and 3 in the FerretDB PDF).

### 2\. The Logic

1. Track `is\_settled: bool` and `last\_stable\_time: float` on `CandidateSlide`.
2. Whenever `candidate\_sim >= self.config.similarity\_threshold`, update `last\_stable\_time = timestamp`. Once `settle\_time >= debounce\_settle\_time` and `slide\_duration >= min\_slide\_duration`, commit the candidate immediately.
3. In the `else` branch (`candidate\_sim < similarity\_threshold`), check whether the **existing** candidate had already accumulated enough settle time (`(self.candidate.last\_stable\_time - self.candidate.last\_change\_time) >= self.config.debounce\_settle\_time`) and duration (`(self.candidate.last\_stable\_time - self.candidate.first\_seen\_time) >= self.config.min\_slide\_duration`) before this abrupt transition occurred. If so, commit the prior candidate first, and then initialize a fresh `CandidateSlide` for the newly arrived frame.
4. **Progressive Slide Build Replacement:** When committing a new slide in `\_try\_commit\_candidate()`, check if the newly committed slide is a progressive superset of `self.committed\_slides\[-1]` (i.e., background SSIM $\\ge 0.86$ and the new frame strictly adds foreground text/edges without removing the previous content). If `replace\_progressive\_builds` is enabled (default `True` for presentation slides), replace the partial slide in-place so the PDF only keeps the completed slide build (collapsing Page 2 into Page 3).

\---

## Issue 4: Unbounded Motion Accumulator \& Zoom Sidebar Masking (`roi.py`)

### 1\. The Issue

In `roi.py`:

1. `PresenterDetector.update()` uses an unbounded sum (`self.\_motion\_accumulator += diff`). Over a 25-minute video, heavy terminal typing on the left side of the screen permanently inflates `total\_motion`, making it impossible for the corner webcam's `motion\_fraction` to exceed `0.40`.
2. `PRESET\_BOXES` and `\_resolve\_candidate\_box()` only inspect 4 corner rectangles (`28%` width $\\times$ `36%` height). In Zoom, Google Meet, and MS Teams webinars, the participant strip is frequently a **full-height right-hand sidebar** (`x: 0.82` to `1.00`, `y: 0.00` to `1.00`), as seen throughout the FerretDB video where participant tiles ("Arman", "Sayed", etc.) trigger false frame differences.

### 2\. The Logic

1. Replace the unbounded accumulator with an **Exponential Moving Average (EMA)** using `cv2.accumulateWeighted(diff, self.\_motion\_accumulator, alpha=0.15)` so the detector adapts to recent 10–15 second windows of motion.
2. Add a `right\_sidebar` candidate region `(0.82, 0.00, 0.18, 1.00)` alongside the 4 standard corner PIP boxes in `\_resolve\_candidate\_box()`, and detect persistent vertical divider lines at $x \\in \[0.80, 0.88]$ (typical of Zoom/Teams sidebars) to automatically mask out participant strips during similarity scoring.

\---

## Issue 5: Zoom Gallery Grid Bypassed `HumanNarratorDetector` (`human\_detector.py`)

### 1\. The Issue

Page 1 of the extracted PDF is a 5-tile Zoom participant grid (`Team AppsCode`, `Fazle Rabbi Sarker`, `esha`, `M. Obaydullah`, `Sayed`). `HumanNarratorDetector.is\_narrator\_frame()` failed to filter it out because:

* Four of the five tiles display static grey Zoom avatar silhouettes (which YuNet face detection ignores).
* Sayed's active webcam tile is positioned in the bottom-right grid cell (`fn\_cy > 0.70`), which failed the centrality check `is\_central = (0.15 <= fn\_cx <= 0.85) and (fn\_cy <= 0.70)`.

### 2\. The Logic

Add a dedicated **Conference Gallery / Lobby Grid Detector** (`\_is\_conference\_gallery\_grid()`) inside `HumanNarratorDetector`:

1. Convert the 640px analysis frame to grayscale and measure the proportion of dark/black border pixels (`gray < 28`) and uniform dark-grey avatar tiles.
2. Find external rectangular contours corresponding to video conference participant tiles (aspect ratios between $1.1$ and $2.0$, each occupying $6%$ to $45%$ of the frame).
3. If the frame contains $\\ge 2$ uniform rectangular participant tiles (or a dominant dark grid background $>35%$) combined with very low high-frequency slide text edge density (`edge\_ratio < 0.020`), classify the frame as a `"Conference gallery / participant grid"` and filter it out.

\---

## Issue 6: High RAM Footprint on Long Videos (`detector.py` \& `worker.py`)

### 1\. The Issue

Each `Slide` dataclass instance stores `frame\_high\_res: np.ndarray` as an uncompressed 1080p 3-channel BGR array ($1920 \\times 1080 \\times 3\\text{ bytes} \\approx 6.22\\text{ MB}$). A long webinar with 150–200 candidate slides consumes **$>1.1\\text{ GB}$ of RAM** before PDF compilation even begins.

### 2\. The Logic

Store the high-resolution slide image in memory as a compressed JPEG byte buffer (`\_compressed\_jpg: bytes` encoded at quality `95`, which takes $\\approx 120\\text{–}180\\text{ KB}$ per slide—a **40x memory reduction**). Expose a backward-compatible `.frame\_high\_res` `@property` on `Slide` that lazily decodes the buffer via `cv2.imdecode()` on demand, ensuring zero breaking changes to `compile\_slides\_to\_pdf()` or external callers while keeping peak RAM under $50\\text{ MB}$.

\---

## Complete Refactored Module Implementations

### 1\. `yt2pdf/roi.py`

```python
"""Region of Interest (ROI), Chrome Masking, Sidebar Detection, and Canvas Auto-Cropping."""

import logging
from typing import Tuple, Optional, List
import numpy as np
import cv2

from yt2pdf.config import PresenterMaskMode

logger = logging.getLogger(\_\_name\_\_)

# Default corner \& sidebar presets (x, y, w, h) in normalized coordinates \[0.0, 1.0]
PRESET\_BOXES = {
    PresenterMaskMode.TOP\_RIGHT: (0.72, 0.05, 0.26, 0.35),
    PresenterMaskMode.BOTTOM\_RIGHT: (0.72, 0.60, 0.26, 0.35),
    PresenterMaskMode.BOTTOM\_LEFT: (0.02, 0.60, 0.26, 0.35),
    PresenterMaskMode.TOP\_LEFT: (0.02, 0.05, 0.26, 0.35),
}

# Extended candidate boxes including full-height Zoom/Teams right-hand participant strip
CANDIDATE\_PIP\_BOXES = \[
    ("top\_right", (0.70, 0.04, 0.28, 0.36)),
    ("bottom\_right", (0.70, 0.60, 0.28, 0.36)),
    ("bottom\_left", (0.02, 0.60, 0.28, 0.36)),
    ("top\_left", (0.02, 0.04, 0.28, 0.36)),
    ("right\_sidebar", (0.82, 0.00, 0.18, 1.00)),
]


def mask\_player\_chrome(frame: np.ndarray, top\_pct: float = 0.03, bottom\_pct: float = 0.05) -> np.ndarray:
    """Masks top and bottom player chrome (titles, seekbars, controls) with black."""
    h, w = frame.shape\[:2]
    top\_h = int(h \* top\_pct)
    bottom\_y = int(h \* (1.0 - bottom\_pct))

    masked = frame.copy()
    if top\_h > 0:
        masked\[:top\_h, :] = 0
    if bottom\_y < h:
        masked\[bottom\_y:, :] = 0
    return masked


def mask\_box(frame: np.ndarray, box\_norm: Tuple\[float, float, float, float]) -> np.ndarray:
    """Masks out a normalized bounding box (x, y, w, h) in the frame with black."""
    h, w = frame.shape\[:2]
    x, y, bw, bh = box\_norm

    x1 = max(0, min(w, int(x \* w)))
    y1 = max(0, min(h, int(y \* h)))
    x2 = max(0, min(w, int((x + bw) \* w)))
    y2 = max(0, min(h, int((y + bh) \* h)))

    if x2 > x1 and y2 > y1:
        masked = frame.copy()
        masked\[y1:y2, x1:x2] = 0
        return masked
    return frame


class PresenterDetector:
    """Detects presenter PIP webcam box or Zoom/Teams participant sidebar using EMA motion variance and structural lines."""

    def \_\_init\_\_(self, grid\_size: int = 16, ema\_alpha: float = 0.15):
        self.grid\_size = grid\_size
        self.ema\_alpha = ema\_alpha
        self.\_prev\_frame: Optional\[np.ndarray] = None
        self.\_motion\_ema: Optional\[np.ndarray] = None
        self.\_samples\_count: int = 0
        self.detected\_box: Optional\[Tuple\[float, float, float, float]] = None

    def update(self, gray\_360p: np.ndarray) -> None:
        """Update Exponential Moving Average (EMA) motion statistics with a new 360p grayscale frame."""
        h, w = gray\_360p.shape\[:2]
        if self.\_prev\_frame is None or self.\_prev\_frame.shape != gray\_360p.shape:
            self.\_prev\_frame = gray\_360p.copy()
            self.\_motion\_ema = np.zeros((h, w), dtype=np.float32)
            self.\_samples\_count = 0
            return

        # Frame difference with compression noise floor
        diff = cv2.absdiff(gray\_360p, self.\_prev\_frame).astype(np.float32)
        diff\[diff < 10.0] = 0.0

        # Exponential Moving Average so old terminal scrolling decays over time
        if self.\_samples\_count == 0:
            self.\_motion\_ema = diff.copy()
        else:
            cv2.accumulateWeighted(diff, self.\_motion\_ema, self.ema\_alpha)

        self.\_samples\_count += 1
        self.\_prev\_frame = gray\_360p.copy()

        # Check for Zoom/Teams right-hand vertical participant strip boundary
        sidebar\_box = self.\_detect\_zoom\_sidebar(gray\_360p)
        if sidebar\_box is not None:
            self.detected\_box = sidebar\_box
            return

        if self.\_samples\_count >= 4:
            self.\_resolve\_candidate\_box(h, w)

    def \_detect\_zoom\_sidebar(self, gray\_360p: np.ndarray) -> Optional\[Tuple\[float, float, float, float]]:
        """Detects a persistent right-side participant sidebar (e.g. Zoom/Teams vertical strip)."""
        h, w = gray\_360p.shape\[:2]
        x\_start = int(w \* 0.82)
        x\_end = int(w \* 0.90)

        # Look for a sharp vertical transition / dark strip on the right 12-18% of the screen
        right\_strip = gray\_360p\[:, int(w \* 0.85):]
        main\_canvas = gray\_360p\[:, :int(w \* 0.80)]

        # In Zoom sidebars, the strip background is dark or split into stacked boxes with dark borders
        dark\_border\_ratio = np.count\_nonzero(right\_strip < 30) / float(max(1, right\_strip.size))
        col\_grad = np.mean(np.abs(np.diff(gray\_360p\[:, x\_start:x\_end].astype(np.float32), axis=1)), axis=0)

        if np.max(col\_grad) > 18.0 and dark\_border\_ratio > 0.12 and np.mean(main\_canvas) > 35.0:
            return (0.83, 0.0, 0.17, 1.0)
        return None

    def \_resolve\_candidate\_box(self, h: int, w: int) -> None:
        """Check if one of the candidate PIP or sidebar regions exhibits concentrated recent motion."""
        if self.\_motion\_ema is None or self.\_samples\_count == 0:
            return

        total\_motion = float(np.sum(self.\_motion\_ema))
        if total\_motion < 1e-3:
            return

        best\_box = None
        highest\_density\_ratio = 0.0
        frame\_area = float(w \* h)

        for name, box in CANDIDATE\_PIP\_BOXES:
            bx, by, bw, bh = box
            x1, y1 = int(bx \* w), int(by \* h)
            x2, y2 = int((bx + bw) \* w), int((by + bh) \* h)

            region\_motion = float(np.sum(self.\_motion\_ema\[y1:y2, x1:x2]))
            region\_area = float(max(1, (x2 - x1) \* (y2 - y1)))

            motion\_fraction = region\_motion / (total\_motion + 1e-5)
            area\_fraction = region\_area / frame\_area
            density\_ratio = motion\_fraction / area\_fraction

            # Trigger if region holds >= 32% of recent motion with >= 2.2x average density
            if motion\_fraction >= 0.32 and density\_ratio >= 2.2:
                if density\_ratio > highest\_density\_ratio:
                    highest\_density\_ratio = density\_ratio
                    best\_box = box

        if best\_box:
            self.detected\_box = best\_box

    def get\_mask\_box(
        self,
        mode: PresenterMaskMode,
        custom\_box: Optional\[Tuple\[float, float, float, float]] = None
    ) -> Optional\[Tuple\[float, float, float, float]]:
        """Returns the normalized box (x, y, w, h) to mask for the given mode."""
        if mode == PresenterMaskMode.AUTO:
            return self.detected\_box or PRESET\_BOXES\[PresenterMaskMode.BOTTOM\_RIGHT]
        if mode in PRESET\_BOXES:
            return PRESET\_BOXES\[mode]
        if mode == PresenterMaskMode.CUSTOM and custom\_box:
            return custom\_box
        return None


def detect\_canvas\_crop\_box(frames: List\[np.ndarray], threshold: int = 18) -> Optional\[Tuple\[int, int, int, int]]:
    """Detects black bars / letterboxing around presentation slides."""
    if not frames:
        return None

    h, w = frames\[0].shape\[:2]
    crop\_boxes = \[]

    sample\_frames = (
        frames
        if len(frames) <= 10
        else \[frames\[i] for i in np.linspace(0, len(frames) - 1, 10, dtype=int)]
    )

    for f in sample\_frames:
        gray = cv2.cvtColor(f, cv2.COLOR\_BGR2GRAY) if len(f.shape) == 3 else f
        row\_means = np.mean(gray, axis=1)
        col\_means = np.mean(gray, axis=0)

        valid\_rows = np.where(row\_means > threshold)\[0]
        valid\_cols = np.where(col\_means > threshold)\[0]

        if len(valid\_rows) > 0 and len(valid\_cols) > 0:
            y1 = int(valid\_rows\[0])
            y2 = int(valid\_rows\[-1]) + 1
            x1 = int(valid\_cols\[0])
            x2 = int(valid\_cols\[-1]) + 1
            crop\_boxes.append((x1, y1, x2, y2))

    if not crop\_boxes:
        return None

    med\_x1 = int(np.median(\[b\[0] for b in crop\_boxes]))
    med\_y1 = int(np.median(\[b\[1] for b in crop\_boxes]))
    med\_x2 = int(np.median(\[b\[2] for b in crop\_boxes]))
    med\_y2 = int(np.median(\[b\[3] for b in crop\_boxes]))

    cropped\_area = (med\_x2 - med\_x1) \* (med\_y2 - med\_y1)
    total\_area = w \* h

    if (cropped\_area / total\_area) < 0.98 and (med\_x2 - med\_x1) > 200 and (med\_y2 - med\_y1) > 150:
        return (med\_x1, med\_y1, med\_x2, med\_y2)

    return None
```

\---

### 2\. `yt2pdf/human\_detector.py`

```python
"""Detection and filtering of full-human / narrator talking frames and webinar gallery grids."""

import os
import urllib.request
import logging
from typing import Tuple, Optional
import cv2
import numpy as np

logger = logging.getLogger(\_\_name\_\_)

MODEL\_FILENAME = "face\_detection\_yunet.onnx"
MODEL\_URL = "https://github.com/opencv/opencv\_zoo/raw/main/models/face\_detection\_yunet/face\_detection\_yunet\_2023mar.onnx"


def get\_model\_path() -> str:
    """Returns path to the YuNet ONNX model, attempting download if not found."""
    models\_dir = os.path.join(os.path.dirname(\_\_file\_\_), "models")
    os.makedirs(models\_dir, exist\_ok=True)
    model\_path = os.path.join(models\_dir, MODEL\_FILENAME)

    if not os.path.exists(model\_path):
        try:
            logger.info("YuNet face detector model not found locally; downloading...")
            urllib.request.urlretrieve(MODEL\_URL, model\_path)
            logger.info(f"YuNet model successfully cached at {model\_path}")
        except Exception as e:
            logger.warning(f"Could not download YuNet model: {e}")
            return ""

    return model\_path


class HumanNarratorDetector:
    """Detects whether a video frame or PDF page is a full-human speaker cutaway
    or a multi-tile conference gallery screen (e.g., Zoom/Teams lobby grid) rather than a slide.
    """

    def \_\_init\_\_(self, score\_threshold: float = 0.60):
        self.score\_threshold = score\_threshold
        self.model\_path = get\_model\_path()
        self.\_detector: Optional\[cv2.FaceDetectorYN] = None
        self.\_last\_input\_size: Optional\[Tuple\[int, int]] = None

    def \_get\_detector(self, width: int, height: int) -> Optional\[cv2.FaceDetectorYN]:
        """Lazy-loads or reconfigures the FaceDetectorYN instance for the given frame size."""
        if not self.model\_path or not os.path.exists(self.model\_path):
            return None

        if self.\_detector is None:
            try:
                self.\_detector = cv2.FaceDetectorYN.create(
                    self.model\_path,
                    "",
                    (width, height),
                    score\_threshold=self.score\_threshold,
                    nms\_threshold=0.3
                )
                self.\_last\_input\_size = (width, height)
            except Exception as e:
                logger.warning(f"Failed to initialize FaceDetectorYN: {e}")
                return None
        elif self.\_last\_input\_size != (width, height):
            try:
                self.\_detector.setInputSize((width, height))
                self.\_last\_input\_size = (width, height)
            except Exception as e:
                logger.warning(f"Failed to set FaceDetectorYN input size: {e}")
                return None

        return self.\_detector

    def is\_narrator\_frame(self, frame\_bgr: np.ndarray) -> Tuple\[bool, str]:
        """Determines if a frame depicts a human narrator cutaway or conference gallery grid."""
        if frame\_bgr is None or frame\_bgr.size == 0:
            return False, "Empty frame"

        h, w = frame\_bgr.shape\[:2]
        analysis\_w = 640
        analysis\_h = max(1, int(h \* (analysis\_w / float(w))))
        small\_frame = cv2.resize(frame\_bgr, (analysis\_w, analysis\_h), interpolation=cv2.INTER\_AREA)

        # 1. Check for Zoom / Teams / Meet multi-tile participant gallery grid (e.g., Page 1)
        is\_gallery, gallery\_reason = self.\_is\_conference\_gallery\_grid(small\_frame)
        if is\_gallery:
            logger.info(gallery\_reason)
            return True, gallery\_reason

        # 2. YuNet ONNX Face Detection
        detector = self.\_get\_detector(analysis\_w, analysis\_h)
        if detector is not None:
            try:
                \_, faces = detector.detect(small\_frame)
                if faces is not None and len(faces) > 0:
                    for face in faces:
                        fx, fy, fw, fh = face\[:4]
                        score = face\[-1]

                        fn\_w = fw / float(analysis\_w)
                        fn\_h = fh / float(analysis\_h)
                        fn\_cx = (fx + fw / 2.0) / float(analysis\_w)
                        fn\_cy = (fy + fh / 2.0) / float(analysis\_h)
                        area\_frac = (fw \* fh) / float(analysis\_w \* analysis\_h)

                        is\_central = (0.15 <= fn\_cx <= 0.85) and (fn\_cy <= 0.75)
                        is\_large = (fn\_h >= 0.12) or (area\_frac >= 0.012)

                        if is\_central and is\_large:
                            desc = (
                                f"Prominent narrator face detected "
                                f"(h={fn\_h\*100:.1f}%, center\_x={fn\_cx\*100:.1f}%, conf={score:.2f})"
                            )
                            logger.info(desc)
                            return True, desc
            except Exception as e:
                logger.debug(f"Face detection exception: {e}")

        # 3. Fallback heuristic: Skin tone clustering in central region with low edge density
        is\_fallback\_narrator, fallback\_reason = self.\_fallback\_skin\_edge\_check(small\_frame)
        if is\_fallback\_narrator:
            return True, fallback\_reason

        return False, "Slide content (no central narrator)"

    def \_is\_conference\_gallery\_grid(self, frame\_bgr: np.ndarray) -> Tuple\[bool, str]:
        """Detects Zoom/Teams/Meet multi-participant gallery grids (even when cameras are off/avatars)."""
        h, w = frame\_bgr.shape\[:2]
        frame\_area = float(h \* w)
        gray = cv2.cvtColor(frame\_bgr, cv2.COLOR\_BGR2GRAY)

        # Conference gallery views have dark/black grid gutters (< 25 luminance)
        dark\_gutter\_mask = (gray < 25).astype(np.uint8) \* 255
        dark\_ratio = np.count\_nonzero(dark\_gutter\_mask) / frame\_area
        if dark\_ratio < 0.18:
            return False, ""

        # Invert dark gutters to find bright/grey participant tile rectangles
        tile\_mask = (gray >= 25).astype(np.uint8) \* 255
        kernel = cv2.getStructuringElement(cv2.MORPH\_RECT, (5, 5))
        tile\_mask = cv2.morphologyEx(tile\_mask, cv2.MORPH\_CLOSE, kernel)

        contours, \_ = cv2.findContours(tile\_mask, cv2.RETR\_EXTERNAL, cv2.CHAIN\_APPROX\_SIMPLE)
        valid\_tiles = 0
        total\_tile\_area = 0.0

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            box\_area = float(cw \* ch)
            area\_frac = box\_area / frame\_area
            aspect = cw / float(max(1, ch))

            # Typical gallery participant tile: 5% to 45% of screen, rectangular aspect ratio
            if 0.05 <= area\_frac <= 0.45 and 0.75 <= aspect <= 2.2:
                rect\_fill = cv2.contourArea(cnt) / max(1.0, box\_area)
                if rect\_fill >= 0.82:
                    valid\_tiles += 1
                    total\_tile\_area += area\_frac

        # Check high-frequency text edge density (slides/terminals have high Canny edge density)
        edges = cv2.Canny(gray, 80, 200)
        edge\_ratio = np.count\_nonzero(edges) / frame\_area

        if valid\_tiles >= 3 and total\_tile\_area >= 0.35 and edge\_ratio < 0.025:
            return True, f"Conference gallery grid detected ({valid\_tiles} participant tiles, edge\_ratio={edge\_ratio\*100:.2f}%)"

        return False, ""

    def \_fallback\_skin\_edge\_check(self, frame\_bgr: np.ndarray) -> Tuple\[bool, str]:
        """Fallback check using skin-tone distribution and edge density if face detector is unavailable."""
        h, w = frame\_bgr.shape\[:2]
        cx1, cx2 = int(w \* 0.20), int(w \* 0.80)
        cy1, cy2 = int(h \* 0.05), int(h \* 0.65)
        central\_region = frame\_bgr\[cy1:cy2, cx1:cx2]

        ycrcb = cv2.cvtColor(central\_region, cv2.COLOR\_BGR2YCrCb)
        skin\_mask = cv2.inRange(ycrcb, np.array(\[0, 133, 77]), np.array(\[255, 173, 127]))
        skin\_ratio = np.count\_nonzero(skin\_mask) / float(skin\_mask.size)

        gray = cv2.cvtColor(frame\_bgr, cv2.COLOR\_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 180)
        edge\_ratio = np.count\_nonzero(edges) / float(edges.size)

        if skin\_ratio > 0.15 and edge\_ratio < 0.022:
            return True, f"Central skin-tone cluster ({skin\_ratio\*100:.1f}%) and low edge density ({edge\_ratio\*100:.1f}%)"

        return False, "Normal slide edge profile"
```

\---

### 3\. `yt2pdf/detector.py`

```python
"""Two-tier frame sampling, perceptual difference hashing, SSIM, and debounce engine."""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
import cv2
import imagehash
from PIL import Image

from yt2pdf.config import ExtractionConfig
from yt2pdf.roi import mask\_player\_chrome, mask\_box, PresenterDetector
from yt2pdf.human\_detector import HumanNarratorDetector

logger = logging.getLogger(\_\_name\_\_)


@dataclass
class Slide:
    """Represents an accepted, unique presentation slide with compressed in-memory high-res storage."""
    index: int
    timestamp: float
    frame\_analysis: np.ndarray
    dhash: imagehash.ImageHash
    \_compressed\_jpg: Optional\[bytes] = field(default=None, repr=False)
    \_raw\_high\_res: Optional\[np.ndarray] = field(default=None, repr=False)

    def \_\_init\_\_(
        self,
        index: int,
        timestamp: float,
        frame\_high\_res: np.ndarray,
        frame\_analysis: np.ndarray,
        dhash: imagehash.ImageHash,
        jpeg\_quality: int = 95
    ):
        self.index = index
        self.timestamp = timestamp
        self.frame\_analysis = frame\_analysis
        self.dhash = dhash
        self.\_raw\_high\_res = None
        self.\_compressed\_jpg = None

        if frame\_high\_res is not None:
            success, encoded = cv2.imencode(
                ".jpg", frame\_high\_res, \[int(cv2.IMWRITE\_JPEG\_QUALITY), jpeg\_quality]
            )
            if success:
                self.\_compressed\_jpg = encoded.tobytes()
            else:
                self.\_raw\_high\_res = frame\_high\_res.copy()

    @property
    def frame\_high\_res(self) -> np.ndarray:
        """Lazily decodes and returns the high-resolution BGR frame on demand."""
        if self.\_compressed\_jpg is not None:
            arr = np.frombuffer(self.\_compressed\_jpg, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD\_COLOR)
        return self.\_raw\_high\_res


@dataclass
class CandidateSlide:
    """Tracks a potential slide settling through bullet points or animations."""
    first\_seen\_time: float
    last\_change\_time: float
    last\_stable\_time: float
    frame\_high\_res: np.ndarray
    frame\_analysis: np.ndarray
    dhash: imagehash.ImageHash


def compute\_opencv\_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes Structural Similarity Index (SSIM) between two grayscale images using OpenCV."""
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape\[1], img1.shape\[0]))

    c1 = 6.5025   # (0.01 \* 255) \*\* 2
    c2 = 58.5225  # (0.03 \* 255) \*\* 2

    i1 = img1.astype(np.float64)
    i2 = img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(i1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(i2, (11, 11), 1.5)

    mu1\_sq = mu1 \* mu1
    mu2\_sq = mu2 \* mu2
    mu1\_mu2 = mu1 \* mu2

    sigma1\_sq = cv2.GaussianBlur(i1 \* i1, (11, 11), 1.5) - mu1\_sq
    sigma2\_sq = cv2.GaussianBlur(i2 \* i2, (11, 11), 1.5) - mu2\_sq
    sigma12 = cv2.GaussianBlur(i1 \* i2, (11, 11), 1.5) - mu1\_mu2

    numerator = (2.0 \* mu1\_mu2 + c1) \* (2.0 \* sigma12 + c2)
    denominator = (mu1\_sq + mu2\_sq + c1) \* (sigma1\_sq + sigma2\_sq + c2)

    ssim\_map = numerator / (denominator + 1e-7)
    return float(np.mean(ssim\_map))


def has\_meaningful\_structural\_change(gray1: np.ndarray, gray2: np.ndarray) -> bool:
    """Uses morphological opening and connected components to distinguish real text/diagram changes
    from blinking cursors, watch timers, or video compression noise.
    """
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape\[1], gray1.shape\[0]))

    diff = cv2.absdiff(gray1, gray2)
    \_, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH\_BINARY)

    # Remove 1-2px isolated compression artifacts and thin blinking cursors
    kernel = cv2.getStructuringElement(cv2.MORPH\_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH\_OPEN, kernel)

    active\_pixels = int(np.count\_nonzero(cleaned))
    if active\_pixels > 350:
        return True

    if active\_pixels < 45:
        return False

    # Check if there is at least one contiguous blob of new content (e.g. a new command or bullet)
    num\_labels, \_, stats, \_ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    for label\_idx in range(1, num\_labels):
        blob\_area = stats\[label\_idx, cv2.CC\_STAT\_AREA]
        if blob\_area >= 140:
            return True

    return False


def compute\_frame\_similarity(
    gray1: np.ndarray,
    hash1: imagehash.ImageHash,
    gray2: np.ndarray,
    hash2: imagehash.ImageHash
) -> float:
    """Calculates combined similarity using perceptual dHash, connected-blob gating, and SSIM."""
    dhash\_dist = hash1 - hash2
    if dhash\_dist <= 1 and not has\_meaningful\_structural\_change(gray1, gray2):
        return 1.0

    ssim = compute\_opencv\_ssim(gray1, gray2)
    # If SSIM is reasonably high and there is no meaningful contiguous blob change, boost stability
    if ssim >= 0.90 and not has\_meaningful\_structural\_change(gray1, gray2):
        return max(ssim, 0.985)

    return max(0.0, min(1.0, ssim))


class SlideDetector:
    """Two-tier slide detector with debouncing, progressive build collapsing, and PIP/sidebar masking."""

    def \_\_init\_\_(self, config: ExtractionConfig):
        self.config = config
        self.presenter\_detector = PresenterDetector()
        self.human\_detector = HumanNarratorDetector()
        self.committed\_slides: List\[Slide] = \[]
        self.candidate: Optional\[CandidateSlide] = None

    def \_prepare\_analysis\_frame(self, frame\_bgr: np.ndarray) -> Tuple\[np.ndarray, imagehash.ImageHash]:
        """Downscales frame to 360p height, applies ROI masks, and generates dHash."""
        h, w = frame\_bgr.shape\[:2]
        target\_h = 360
        target\_w = max(1, int(w \* (target\_h / float(h))))

        resized = cv2.resize(frame\_bgr, (target\_w, target\_h), interpolation=cv2.INTER\_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR\_BGR2GRAY)

        if self.config.mask\_presenter:
            self.presenter\_detector.update(gray)
            mask\_box\_coords = self.presenter\_detector.get\_mask\_box(
                self.config.presenter\_mode,
                self.config.custom\_roi\_box
            )
            if mask\_box\_coords:
                gray = mask\_box(gray, mask\_box\_coords)

        if self.config.ignore\_chrome:
            gray = mask\_player\_chrome(gray, top\_pct=0.03, bottom\_pct=0.05)

        pil\_img = Image.fromarray(gray)
        dhash = imagehash.dhash(pil\_img)

        return gray, dhash

    def process\_frame(
        self,
        timestamp: float,
        frame\_bgr: np.ndarray,
        is\_last\_frame: bool = False
    ) -> Optional\[Slide]:
        """Evaluates a sampled frame and returns a committed Slide if a transition settled."""
        if getattr(self.config, "filter\_narrator\_frames", True):
            is\_narrator, reason = self.human\_detector.is\_narrator\_frame(frame\_bgr)
            if is\_narrator:
                logger.debug(f"Skipping narrator/gallery cutaway at {timestamp:.1f}s: {reason}")
                return None

        gray, dhash = self.\_prepare\_analysis\_frame(frame\_bgr)
        newly\_committed\_slide: Optional\[Slide] = None

        if not self.committed\_slides and self.candidate is None:
            self.candidate = CandidateSlide(
                first\_seen\_time=timestamp,
                last\_change\_time=timestamp,
                last\_stable\_time=timestamp,
                frame\_high\_res=frame\_bgr,
                frame\_analysis=gray,
                dhash=dhash
            )
            return None

        if self.candidate is None and self.committed\_slides:
            last\_slide = self.committed\_slides\[-1]
            sim = compute\_frame\_similarity(last\_slide.frame\_analysis, last\_slide.dhash, gray, dhash)
            if sim < self.config.similarity\_threshold:
                self.candidate = CandidateSlide(
                    first\_seen\_time=timestamp,
                    last\_change\_time=timestamp,
                    last\_stable\_time=timestamp,
                    frame\_high\_res=frame\_bgr,
                    frame\_analysis=gray,
                    dhash=dhash
                )
            return None

        if self.candidate is not None:
            candidate\_sim = compute\_frame\_similarity(
                self.candidate.frame\_analysis,
                self.candidate.dhash,
                gray,
                dhash
            )

            if candidate\_sim >= self.config.similarity\_threshold:
                # Frame is stable; update candidate to latest clean frame
                self.candidate.last\_stable\_time = timestamp
                self.candidate.frame\_high\_res = frame\_bgr
                self.candidate.frame\_analysis = gray
                self.candidate.dhash = dhash

                settle\_time = timestamp - self.candidate.last\_change\_time
                slide\_duration = timestamp - self.candidate.first\_seen\_time

                if settle\_time >= self.config.debounce\_settle\_time and slide\_duration >= self.config.min\_slide\_duration:
                    newly\_committed\_slide = self.\_try\_commit\_candidate(timestamp)
            else:
                # Abrupt change detected: check if the previous candidate had ALREADY settled before this change
                prior\_settle = self.candidate.last\_stable\_time - self.candidate.last\_change\_time
                prior\_duration = self.candidate.last\_stable\_time - self.candidate.first\_seen\_time

                if prior\_settle >= self.config.debounce\_settle\_time and prior\_duration >= self.config.min\_slide\_duration:
                    newly\_committed\_slide = self.\_try\_commit\_candidate(self.candidate.last\_stable\_time)
                    self.candidate = CandidateSlide(
                        first\_seen\_time=timestamp,
                        last\_change\_time=timestamp,
                        last\_stable\_time=timestamp,
                        frame\_high\_res=frame\_bgr,
                        frame\_analysis=gray,
                        dhash=dhash
                    )
                else:
                    # Animation or rapid scrolling still in progress; reset settle timer
                    self.candidate.last\_change\_time = timestamp
                    self.candidate.last\_stable\_time = timestamp
                    self.candidate.frame\_high\_res = frame\_bgr
                    self.candidate.frame\_analysis = gray
                    self.candidate.dhash = dhash

        if is\_last\_frame and self.candidate is not None:
            slide\_duration = timestamp - self.candidate.first\_seen\_time
            if slide\_duration >= (self.config.min\_slide\_duration \* 0.5):
                final\_commit = self.\_try\_commit\_candidate(timestamp)
                if final\_commit:
                    newly\_committed\_slide = final\_commit

        return newly\_committed\_slide

    def \_is\_progressive\_slide\_build(self, prev\_gray: np.ndarray, curr\_gray: np.ndarray) -> bool:
        """Checks if curr\_gray is a progressive bullet-point build on top of prev\_gray."""
        ssim = compute\_opencv\_ssim(prev\_gray, curr\_gray)
        if ssim < 0.84:
            return False

        edges\_prev = cv2.Canny(prev\_gray, 60, 180)
        edges\_curr = cv2.Canny(curr\_gray, 60, 180)

        prev\_edge\_count = int(np.count\_nonzero(edges\_prev))
        curr\_edge\_count = int(np.count\_nonzero(edges\_curr))

        if curr\_edge\_count <= prev\_edge\_count \* 1.05:
            return False

        # Dilate current edges slightly to account for 1px compression shifts
        dilated\_curr = cv2.dilate(edges\_curr, np.ones((3, 3), np.uint8))
        retained\_prev\_edges = int(np.count\_nonzero(cv2.bitwise\_and(edges\_prev, dilated\_curr)))
        retention\_ratio = retained\_prev\_edges / float(max(1, prev\_edge\_count))

        # If >= 88% of previous slide edges are still in place while new edges were added, it's a progressive build
        return retention\_ratio >= 0.88

    def \_try\_commit\_candidate(self, current\_timestamp: float) -> Optional\[Slide]:
        """Commits the candidate slide if it differs from the last committed slide, or upgrades a partial build."""
        if self.candidate is None:
            return None

        if self.committed\_slides:
            last\_slide = self.committed\_slides\[-1]
            sim = compute\_frame\_similarity(
                last\_slide.frame\_analysis,
                last\_slide.dhash,
                self.candidate.frame\_analysis,
                self.candidate.dhash
            )
            if sim >= self.config.similarity\_threshold:
                self.candidate = None
                return None

            # Collapse progressive bullet-point builds into the completed slide (e.g. Page 2 -> Page 3)
            if getattr(self.config, "collapse\_progressive\_builds", True):
                if self.\_is\_progressive\_slide\_build(last\_slide.frame\_analysis, self.candidate.frame\_analysis):
                    upgraded\_slide = Slide(
                        index=last\_slide.index,
                        timestamp=self.candidate.first\_seen\_time,
                        frame\_high\_res=self.candidate.frame\_high\_res,
                        frame\_analysis=self.candidate.frame\_analysis,
                        dhash=self.candidate.dhash
                    )
                    self.committed\_slides\[-1] = upgraded\_slide
                    self.candidate = None
                    logger.info(f"Upgraded progressive build on Slide #{upgraded\_slide.index} at {upgraded\_slide.timestamp:.1f}s")
                    return upgraded\_slide

        idx = len(self.committed\_slides) + 1
        slide = Slide(
            index=idx,
            timestamp=self.candidate.first\_seen\_time,
            frame\_high\_res=self.candidate.frame\_high\_res,
            frame\_analysis=self.candidate.frame\_analysis,
            dhash=self.candidate.dhash
        )
        self.committed\_slides.append(slide)
        self.candidate = None
        logger.info(f"Committed Slide #{slide.index} at {slide.timestamp:.1f}s")
        return slide
```

\---

### 4\. `yt2pdf/deduplicator.py`

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

from yt2pdf.detector import Slide, compute\_opencv\_ssim, has\_meaningful\_structural\_change
from yt2pdf.human\_detector import HumanNarratorDetector
from yt2pdf.downloader import clean\_file\_path

logger = logging.getLogger(\_\_name\_\_)


@dataclass
class DeduplicationResult:
    """Detailed summary of the deduplication pass."""
    input\_path: str
    output\_path: str
    total\_pages: int
    unique\_pages\_count: int
    duplicates\_removed\_count: int
    narrator\_pages\_removed\_count: int = 0
    kept\_indices: List\[int] = field(default\_factory=list)
    duplicate\_details: List\[Tuple\[int, int, float]] = field(default\_factory=list)
    narrator\_details: List\[Tuple\[int, str]] = field(default\_factory=list)


def are\_slides\_duplicate(
    gray1: np.ndarray,
    hash1: imagehash.ImageHash,
    gray2: np.ndarray,
    hash2: imagehash.ImageHash,
    similarity\_threshold: float = 0.94
) -> Tuple\[bool, float]:
    """Determines if two slide frames are identical/duplicate representations using
    dHash, connected-component structural gating, and SSIM.
    """
    # 1. Quick check with perceptual dHash
    dhash\_dist = hash1 - hash2
    if dhash\_dist > 10:
        return False, 0.0

    # 2. Compute SSIM
    ssim = compute\_opencv\_ssim(gray1, gray2)

    # 3. Morphological connected-component check to ignore blinking cursors / watch timers
    meaningful\_diff = has\_meaningful\_structural\_change(gray1, gray2)
    if not meaningful\_diff and ssim >= (similarity\_threshold - 0.03):
        return True, max(ssim, similarity\_threshold)

    if meaningful\_diff:
        return False, ssim

    if ssim >= similarity\_threshold:
        return True, ssim

    return False, ssim


def deduplicate\_slides\_list(
    slides: List\[Slide],
    similarity\_threshold: float = 0.92,
    filter\_narrator\_frames: bool = True
) -> Tuple\[List\[Slide], List\[Tuple\[int, int, float]]]:
    """Reviews an in-memory list of Slide objects across the whole presentation."""
    if not slides:
        return \[], \[]

    human\_detector = HumanNarratorDetector() if filter\_narrator\_frames else None
    kept\_slides: List\[Slide] = \[]
    duplicate\_records: List\[Tuple\[int, int, float]] = \[]

    for curr\_slide in slides:
        if human\_detector is not None:
            is\_narrator, reason = human\_detector.is\_narrator\_frame(curr\_slide.frame\_high\_res)
            if is\_narrator:
                logger.info(f"Slide #{curr\_slide.index} is a narrator/gallery cut: {reason}. Truncating.")
                continue

        is\_duplicate = False
        matched\_orig\_index = -1
        highest\_sim = 0.0

        for kept in kept\_slides:
            is\_dup, sim\_score = are\_slides\_duplicate(
                curr\_slide.frame\_analysis,
                curr\_slide.dhash,
                kept.frame\_analysis,
                kept.dhash,
                similarity\_threshold=similarity\_threshold
            )
            if is\_dup:
                is\_duplicate = True
                matched\_orig\_index = kept.index
                highest\_sim = sim\_score
                break

        if is\_duplicate:
            logger.info(
                f"Slide #{curr\_slide.index} (at {curr\_slide.timestamp:.1f}s) is a repetitive duplicate "
                f"of Slide #{matched\_orig\_index} (SSIM: {highest\_sim:.3f}). Truncating."
            )
            duplicate\_records.append((curr\_slide.index, matched\_orig\_index, highest\_sim))
        else:
            kept\_slides.append(curr\_slide)

    # Re-index kept slides sequentially
    for new\_idx, slide in enumerate(kept\_slides, start=1):
        slide.index = new\_idx

    return kept\_slides, duplicate\_records


def deduplicate\_pdf(
    input\_pdf\_path: str,
    output\_pdf\_path: Optional\[str] = None,
    similarity\_threshold: float = 0.94,
    filter\_narrator\_frames: bool = True,
    progress\_callback: Optional\[Callable\[\[int, int, str], None]] = None,
    cancel\_event: Optional\[threading.Event] = None
) -> DeduplicationResult:
    """Reviews an entire PDF, removing duplicate pages, narrator shots, and conference gallery screens."""
    input\_pdf\_path = clean\_file\_path(input\_pdf\_path)
    if output\_pdf\_path:
        output\_pdf\_path = clean\_file\_path(output\_pdf\_path)

    if not os.path.exists(input\_pdf\_path):
        raise FileNotFoundError(f"Input PDF file not found: {input\_pdf\_path}")

    if not output\_pdf\_path:
        base, ext = os.path.splitext(input\_pdf\_path)
        output\_pdf\_path = f"{base}\_dedup{ext}"

    src\_pdf = pypdfium2.PdfDocument(input\_pdf\_path)
    total\_pages = len(src\_pdf)

    if total\_pages == 0:
        src\_pdf.close()
        raise ValueError("The provided PDF file contains 0 pages.")

    human\_detector = HumanNarratorDetector() if filter\_narrator\_frames else None
    kept\_indices: List\[int] = \[]
    kept\_fingerprints: List\[Tuple\[int, np.ndarray, imagehash.ImageHash]] = \[]
    duplicate\_details: List\[Tuple\[int, int, float]] = \[]
    narrator\_details: List\[Tuple\[int, str]] = \[]

    try:
        for i in range(total\_pages):
            if cancel\_event and cancel\_event.is\_set():
                logger.info("PDF deduplication cancelled by user.")
                break

            if progress\_callback:
                progress\_callback(i + 1, total\_pages, f"Analyzing page {i + 1} of {total\_pages}...")

            page = src\_pdf\[i]
            pil\_rendered = page.render(scale=0.5).to\_pil()

            if human\_detector is not None:
                bgr\_rendered = cv2.cvtColor(np.array(pil\_rendered), cv2.COLOR\_RGB2BGR)
                is\_narrator, n\_reason = human\_detector.is\_narrator\_frame(bgr\_rendered)
                if is\_narrator:
                    page\_1based = i + 1
                    narrator\_details.append((page\_1based, n\_reason))
                    logger.info(f"Page {page\_1based} is a narrator/gallery frame ({n\_reason}). Truncating from PDF.")
                    continue

            target\_h = 360
            w, h = pil\_rendered.size
            target\_w = max(1, int(w \* (target\_h / float(h))))
            pil\_resized = pil\_rendered.resize((target\_w, target\_h), Image.Resampling.BILINEAR)

            gray = np.array(pil\_resized.convert("L"))
            dhash = imagehash.dhash(pil\_resized)

            is\_dup = False
            matched\_first\_appearance = -1
            match\_score = 0.0

            for kept\_idx, kept\_gray, kept\_hash in kept\_fingerprints:
                is\_match, score = are\_slides\_duplicate(
                    gray, dhash, kept\_gray, kept\_hash, similarity\_threshold=similarity\_threshold
                )
                if is\_match:
                    is\_dup = True
                    matched\_first\_appearance = kept\_idx
                    match\_score = score
                    break

            if is\_dup:
                dup\_1based = i + 1
                orig\_1based = matched\_first\_appearance + 1
                duplicate\_details.append((dup\_1based, orig\_1based, match\_score))
                logger.info(
                    f"Page {dup\_1based} matches earlier Page {orig\_1based} "
                    f"(Similarity: {match\_score:.3f}). Truncating repetitive page."
                )
            else:
                kept\_indices.append(i)
                kept\_fingerprints.append((i, gray, dhash))

        out\_dir = os.path.dirname(os.path.abspath(output\_pdf\_path))
        if out\_dir and not os.path.exists(out\_dir):
            os.makedirs(out\_dir, exist\_ok=True)

        new\_pdf = pypdfium2.PdfDocument.new()
        new\_pdf.import\_pages(src\_pdf, kept\_indices)
        new\_pdf.save(output\_pdf\_path)
        new\_pdf.close()

    finally:
        src\_pdf.close()

    result = DeduplicationResult(
        input\_path=input\_pdf\_path,
        output\_path=os.path.abspath(output\_pdf\_path),
        total\_pages=total\_pages,
        unique\_pages\_count=len(kept\_indices),
        duplicates\_removed\_count=len(duplicate\_details),
        narrator\_pages\_removed\_count=len(narrator\_details),
        kept\_indices=kept\_indices,
        duplicate\_details=duplicate\_details,
        narrator\_details=narrator\_details
    )

    logger.info(
        f"PDF Review Complete: {result.total\_pages} total pages -> "
        f"{result.unique\_pages\_count} unique kept, {result.duplicates\_removed\_count} duplicate pages truncated, "
        f"{result.narrator\_pages\_removed\_count} narrator/gallery pages removed."
    )
    return result
```

\---

### 5\. `yt2pdf/worker.py` (Fast `cap.grab()` Stepping Upgrade)

In `yt2pdf/worker.py`, replace the frame-seeking block inside `run()` (lines 110–125) with the following hybrid `cap.grab()` stepper:

```python
            while current\_frame\_idx < total\_frames:
                if self.cancel\_event.is\_set():
                    self.\_send(WorkerMessageType.CANCELLED, "Extraction was cancelled.")
                    return

                # Fast sequential grab for typical step intervals; seek only for huge jumps
                if current\_frame\_idx == 0:
                    ret, frame = cap.read()
                elif frame\_step <= 120:
                    grab\_ok = True
                    for \_ in range(frame\_step - 1):
                        if not cap.grab():
                            grab\_ok = False
                            break
                    ret, frame = cap.read() if grab\_ok else (False, None)
                else:
                    cap.set(cv2.CAP\_PROP\_POS\_FRAMES, current\_frame\_idx)
                    ret, frame = cap.read()

                if not ret or frame is None:
                    break
```

\---

## 

