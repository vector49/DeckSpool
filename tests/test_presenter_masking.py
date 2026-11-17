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


    def test_temporal_latch_decay_resolution_independent(self):
        """Latch decay must trigger at the same real-world inactivity duration regardless of frame resolution.

        A box established at 360p and a box established at 720p should both decay after the same
        number of inactive sample intervals. This fails if the threshold is a raw pixel-sum (500.0)
        rather than a normalized fraction of the box area.
        """
        def run_decay_at_resolution(height, width):
            pd = PresenterDetector()
            static = np.ones((height, width), dtype=np.uint8) * 180

            # Establish box with active motion in top-right corner
            for _ in range(8):
                frame = static.copy()
                x1, x2 = int(0.72 * width), int(0.96 * width)
                y1, y2 = int(0.06 * height), int(0.38 * height)
                frame[y1:y2, x1:x2] = np.random.randint(0, 255, (y2 - y1, x2 - x1), dtype=np.uint8)
                pd.update(frame)

            self.assertIsNotNone(pd.detected_box, f"Box not established at {height}p")

            # Feed static frames — decay must trigger within the same inactive interval count
            for _ in range(15):
                pd.update(static.copy())
            return pd.detected_box

        box_360p = run_decay_at_resolution(360, 640)
        box_720p = run_decay_at_resolution(720, 1280)

        self.assertIsNone(box_360p, "Box did not decay at 360p")
        self.assertIsNone(box_720p, "Box did not decay at 720p — decay threshold is resolution-dependent (bug)")

    def test_named_constants_exist(self):
        """Module-level named constants must be used for magic numbers in PresenterDetector decay logic."""
        import deckspool.roi as roi_module
        self.assertTrue(
            hasattr(roi_module, "BOX_MOTION_INACTIVE_THRESHOLD") or
            hasattr(roi_module, "_BOX_MOTION_INACTIVE_FRACTION") or
            hasattr(roi_module, "BOX_INACTIVE_SAMPLE_COUNT"),
            "roi.py must expose named constant(s) for the decay thresholds instead of bare magic numbers"
        )

    def test_split_screen_requires_slide_edges_on_non_presenter_half(self):
        """Split-screen masking must NOT trigger when the non-presenter half has no slide edges.

        Spec: 'the other half displays dense slide edges' — this is a required condition.
        A dark featureless right half beside a presenter face must NOT trigger SPLIT_LEFT_50.
        """
        pd = PresenterDetector()
        h, w = 360, 640

        # Left half: presenter face area (dark, low-edge background)
        # Right half: featureless dark area (no slide edges)
        frame = np.ones((h, w), dtype=np.uint8) * 20

        # Simulate large prominent face on the left (cx ~ 0.20, h ~ 0.18)
        big_left_face = [(0.10, 0.20, 0.20, 0.18, 0.95)]
        for _ in range(8):
            pd.update(frame.copy(), detected_faces=big_left_face)

        # Should NOT mask the right half — no slide content there
        self.assertNotEqual(
            pd.detected_box,
            (0.50, 0.00, 0.50, 1.00),
            "Split-screen masking triggered without slide edges on non-presenter half"
        )


class TestHumanDetectorEdgeSeamUnification(unittest.TestCase):
    """Tests verifying the unified edge-density seam in HumanNarratorDetector."""

    def test_off_center_stage_speaker_without_slides_is_narrator(self):
        """Off-center stage speaker on a dark background (no slides) must be flagged as narrator.

        Spec: 'if is_prominent and not is_central: if not _has_complementary_slide_content ...'
        A face at cx=0.30, h=14% on a dark stage must be classified as narrator.
        """
        from deckspool.human_detector import HumanNarratorDetector
        detector = HumanNarratorDetector()

        # Dark auditorium stage — no slide edges at all
        frame = np.full((720, 1280, 3), 30, dtype=np.uint8)
        # Off-center prominent face: cx ~ 0.30, h ~ 14%
        off_center_face = [(0.22, 0.20, 0.16, 0.14, 0.91)]

        is_narrator, reason = detector.is_narrator_frame(frame, detected_faces=off_center_face)
        self.assertTrue(
            is_narrator,
            f"Off-center stage speaker not classified as narrator. Reason: {reason}"
        )

    def test_off_center_presenter_with_slide_content_is_not_narrator(self):
        """Off-center face beside rich slide content must NOT be classified as narrator."""
        from deckspool.human_detector import HumanNarratorDetector
        detector = HumanNarratorDetector()

        # Slide frame: white bg with structured text, face on the left side
        frame = np.full((720, 1280, 3), 240, dtype=np.uint8)
        for y in range(100, 600, 60):
            cv2.line(frame, (400, y), (1200, y), (40, 40, 40), 2)
        cv2.putText(frame, "Deep Learning: Convolutional Networks", (400, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

        # Small off-center face on the left margin
        off_center_face = [(0.05, 0.25, 0.14, 0.13, 0.89)]

        is_narrator, reason = detector.is_narrator_frame(frame, detected_faces=off_center_face)
        self.assertFalse(
            is_narrator,
            f"Off-center face beside slide content incorrectly flagged as narrator. Reason: {reason}"
        )

    def test_has_slide_content_supersedes_has_complementary_slide_content(self):
        """_has_complementary_slide_content must delegate to _has_slide_content (no Canny duplication).

        After refactoring, _has_complementary_slide_content should call _has_slide_content
        rather than re-implementing Canny thresholding independently.
        """
        import inspect
        from deckspool.human_detector import HumanNarratorDetector
        src = inspect.getsource(HumanNarratorDetector._has_complementary_slide_content)
        # Must NOT contain its own cv2.Canny call — should delegate instead
        self.assertNotIn(
            "cv2.Canny",
            src,
            "_has_complementary_slide_content still contains its own Canny call (duplicated code not unified)"
        )


if __name__ == "__main__":
    unittest.main()
