"""Detection and filtering of full-human / narrator talking frames."""

import os
import urllib.request
import logging
from typing import Tuple, Optional
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
    """Detects whether a video frame or PDF page is a full-human / speaker cutaway
    (narrator talking to the camera) rather than a presentation slide.
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

    def is_narrator_frame(self, frame_bgr: np.ndarray) -> Tuple[bool, str]:
        """Determines if a frame depicts a human narrator/speaker talking rather than a slide.
        
        Distinguishes between:
        1. Full-human / presenter camera cuts (large, central face & upper body) -> RETURNS TRUE
        2. Small corner PIP webcams on presentation slides -> RETURNS FALSE (slide preserved)
        3. Regular presentation slides -> RETURNS FALSE
        
        Returns:
            Tuple of (is_narrator, reason_description)
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return False, "Empty frame"

        h, w = frame_bgr.shape[:2]

        # Downscale to standardized analysis width (640px) for consistent geometry and ultra-fast inference (< 8ms)
        analysis_w = 640
        analysis_h = max(1, int(h * (analysis_w / float(w))))
        small_frame = cv2.resize(frame_bgr, (analysis_w, analysis_h), interpolation=cv2.INTER_AREA)

        detector = self._get_detector(analysis_w, analysis_h)

        if detector is not None:
            try:
                _, faces = detector.detect(small_frame)
                if faces is not None and len(faces) > 0:
                    for face in faces:
                        fx, fy, fw, fh = face[:4]
                        score = face[-1]

                        # Normalized coordinates
                        fn_w = fw / float(analysis_w)
                        fn_h = fh / float(analysis_h)
                        fn_cx = (fx + fw / 2.0) / float(analysis_w)
                        fn_cy = (fy + fh / 2.0) / float(analysis_h)
                        area_frac = (fw * fh) / float(analysis_w * analysis_h)

                        # Full-human / narrator characteristics:
                        # 1. Face is in the central region of the frame (not a corner PIP)
                        is_central = (0.15 <= fn_cx <= 0.85) and (fn_cy <= 0.70)
                        # 2. Face is prominent and large (face height >= 12% of frame or area >= 1.2%)
                        is_large = (fn_h >= 0.12) or (area_frac >= 0.012)

                        if is_central and is_large:
                            desc = f"Prominent narrator face detected (h={fn_h*100:.1f}%, center_x={fn_cx*100:.1f}%, conf={score:.2f})"
                            logger.info(desc)
                            return True, desc
            except Exception as e:
                logger.debug(f"Face detection exception: {e}")

        # Fallback heuristic: Skin tone clustering in central region with low edge density (text absence)
        is_fallback_narrator, fallback_reason = self._fallback_skin_edge_check(small_frame)
        if is_fallback_narrator:
            return True, fallback_reason

        return False, "Slide content (no central narrator)"

    def _fallback_skin_edge_check(self, frame_bgr: np.ndarray) -> Tuple[bool, str]:
        """Fallback check using skin-tone distribution and edge density if face detector is unavailable."""
        h, w = frame_bgr.shape[:2]

        # Central ROI: upper 60% of frame, central 60% horizontally
        cx1, cx2 = int(w * 0.20), int(w * 0.80)
        cy1, cy2 = int(h * 0.05), int(h * 0.65)
        central_region = frame_bgr[cy1:cy2, cx1:cx2]

        # Convert to YCrCb for skin tone detection
        ycrcb = cv2.cvtColor(central_region, cv2.COLOR_BGR2YCrCb)
        # Standard human skin tone range in YCrCb
        skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
        skin_ratio = np.count_nonzero(skin_mask) / float(skin_mask.size)

        # Check edge density on grayscale (slides have sharp text lines; camera shots have soft/blurred backgrounds)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 180)
        edge_ratio = np.count_nonzero(edges) / float(edges.size)

        # A narrator shot typically has high central skin presence (> 15%) and low sharp edge density (< 2.5%)
        if skin_ratio > 0.15 and edge_ratio < 0.022:
            return True, f"Central skin-tone cluster ({skin_ratio*100:.1f}%) and low edge density ({edge_ratio*100:.1f}%)"

        return False, "Normal slide edge profile"
