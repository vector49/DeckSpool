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

        # 1. Check for Zoom / Teams / Meet multi-tile participant gallery grid (e.g., Page 1)
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

            if (is_central and is_prominent) or is_dominant:
                # If presenter is framed to one side (cx >= 0.68 or cx <= 0.32) and not dominating the screen,
                # check if the opposite side has slide content (split screen or PiP)
                if (fn_cx >= 0.68 or fn_cx <= 0.32) and (fn_h < 0.25):
                    if self._has_complementary_slide_content(small_frame, fn_cx):
                        continue

                desc = (
                    f"Prominent narrator face detected "
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

        # Conference gallery views have dark/black grid gutters (< 25 luminance)
        dark_gutter_mask = (gray < 25).astype(np.uint8) * 255
        dark_ratio = np.count_nonzero(dark_gutter_mask) / frame_area
        if dark_ratio < 0.18:
            return False, ""

        # Invert dark gutters to find bright/grey participant tile rectangles
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

            # Typical gallery participant tile: 5% to 45% of screen, rectangular aspect ratio
            if 0.05 <= area_frac <= 0.45 and 0.75 <= aspect <= 2.2:
                rect_fill = cv2.contourArea(cnt) / max(1.0, box_area)
                if rect_fill >= 0.82:
                    valid_tiles += 1
                    total_tile_area += area_frac

        # Check high-frequency text edge density (slides/terminals have high Canny edge density)
        edges = cv2.Canny(gray, 80, 200)
        edge_ratio = np.count_nonzero(edges) / frame_area

        if valid_tiles >= 3 and total_tile_area >= 0.35 and edge_ratio < 0.025:
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
