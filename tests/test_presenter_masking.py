import unittest
import numpy as np
import cv2
from deckspool.roi import PresenterDetector, CANDIDATE_PIP_BOXES
from deckspool.config import PresenterMaskMode


class TestPresenterMaskingDiagnosis(unittest.TestCase):
    """Deterministic reproducer tests for presenter masking bugs."""

    def test_no_blind_fallback_on_auto(self):
        """When detected_box is None, AUTO mode must return None (no blind masking of bottom-right)."""
        pd = PresenterDetector()
        self.assertIsNone(pd.detected_box)
        box = pd.get_mask_box(PresenterMaskMode.AUTO)
        self.assertIsNone(box, f"AUTO mode blindly fell back to masking {box} when no presenter was detected")

    def test_candidate_pip_boxes_excludes_5050_splits(self):
        """CANDIDATE_PIP_BOXES must represent PIP windows/sidebars only, not 50/50 splits."""
        box_names = [name for name, _ in CANDIDATE_PIP_BOXES]
        self.assertNotIn("split_right_50", box_names)
        self.assertNotIn("split_left_50", box_names)

    def test_corner_pip_not_misclassified_as_split_screen(self):
        """Corner PIP webcam (h < 0.18, cx <= 0.20) must never trigger 50/50 half-screen masking."""
        pd = PresenterDetector()
        h, w = 360, 640
        base_frame = np.ones((h, w), dtype=np.uint8) * 30
        cv2.putText(base_frame, "C++ Metaprogramming in Slides", (120, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)

        # Top-left corner PIP webcam face (cx=0.14, cy=0.25, w=0.038, h=0.088)
        corner_face = [(0.12, 0.21, 0.038, 0.088, 0.90)]

        # Simulate 8 frames with motion in top-left
        for i in range(8):
            frame = base_frame.copy()
            # Animate motion in top-left corner
            x1, x2 = int(0.04 * w), int(0.24 * w)
            y1, y2 = int(0.06 * h), int(0.36 * h)
            frame[y1:y2, x1:x2] = np.random.randint(0, 255, (y2 - y1, x2 - x1), dtype=np.uint8)
            pd.update(frame, detected_faces=corner_face)

        self.assertNotEqual(
            pd.detected_box,
            (0.0, 0.0, 0.5, 1.0),
            "Corner PIP webcam was misclassified as 50% split-screen blackout"
        )

    def test_temporal_latch_decay(self):
        """Detected box must decay to None when presenter motion stops."""
        pd = PresenterDetector()
        h, w = 360, 640
        static_slide = np.ones((h, w), dtype=np.uint8) * 180

        # Feed 8 frames of top-right motion to establish a detected box
        for _ in range(8):
            frame = static_slide.copy()
            x1, x2 = int(0.72 * w), int(0.96 * w)
            y1, y2 = int(0.06 * h), int(0.38 * h)
            frame[y1:y2, x1:x2] = np.random.randint(0, 255, (y2 - y1, x2 - x1), dtype=np.uint8)
            pd.update(frame)

        self.assertIsNotNone(pd.detected_box)

        # Feed 15 static frames with no motion
        for _ in range(15):
            pd.update(static_slide.copy())

        self.assertIsNone(pd.detected_box, "Detected box failed to decay after 15 static frames")


if __name__ == "__main__":
    unittest.main()
