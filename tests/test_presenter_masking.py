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
        """PresenterDetector must retain the latched box during sub-threshold inactivity intervals
        and decay to None once the full inactivity sample threshold is reached."""
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

        established_box = pd.detected_box
        self.assertIsNotNone(established_box, "Detected box failed to establish with active motion")

        # Feed 5 static frames (sub-threshold: 4 inactive transitions)
        for _ in range(5):
            pd.update(static_slide.copy())
        self.assertEqual(
            pd.detected_box,
            established_box,
            "Detected box prematurely decayed during sub-threshold inactivity"
        )

        # Feed 5 more static frames (total 10 static frames -> 9 inactive transitions, still sub-threshold)
        for _ in range(5):
            pd.update(static_slide.copy())
        self.assertEqual(
            pd.detected_box,
            established_box,
            "Detected box prematurely decayed before reaching 10 inactive samples"
        )

        # Feed the 11th static frame (reaches 10 consecutive inactive sample transitions)
        pd.update(static_slide.copy())
        self.assertIsNone(
            pd.detected_box,
            "Detected box failed to decay upon reaching the inactivity threshold (10 samples)"
        )

    def test_split_screen_requires_slide_edges_on_non_presenter_half(self):
        """Split-screen masking must NOT trigger when the non-presenter half has no slide edges.

        Spec line 108: 'If a presenter face resides persistently on one half while the other half
        displays dense slide edges, mask the presenter half with SPLIT_RIGHT_50 or SPLIT_LEFT_50.'
        Even with a sharp center vertical gradient (> 35.0) and prominent face, split-screen masking
        must NOT trigger when the non-presenter half lacks structured slide edges.
        """
        pd = PresenterDetector()
        h, w = 360, 640

        # Case 1: Presenter on left (20), featureless plain right half (120)
        # Center gradient |120 - 20| = 100 > 35.0, but zero slide edges on the non-presenter right side
        frame_no_edges = np.zeros((h, w), dtype=np.uint8)
        frame_no_edges[:, :w // 2] = 20
        frame_no_edges[:, w // 2:] = 120

        big_left_face = [(0.10, 0.20, 0.20, 0.18, 0.95)]
        for _ in range(8):
            pd.update(frame_no_edges.copy(), detected_faces=big_left_face)

        self.assertIsNone(
            pd.detected_box,
            f"Split-screen masking triggered ({pd.detected_box}) despite non-presenter right half having no slide edges"
        )

        # Case 2: Presenter on right (120), featureless plain left half (20)
        pd_right = PresenterDetector()
        big_right_face = [(0.68, 0.20, 0.08, 0.15, 0.95)]
        for _ in range(8):
            pd_right.update(frame_no_edges.copy(), detected_faces=big_right_face)

        self.assertIsNone(
            pd_right.detected_box,
            f"Split-screen masking triggered ({pd_right.detected_box}) despite non-presenter left half having no slide edges"
        )

        # Case 3: Positive verification - when non-presenter right half HAS structured slide edges,
        # presenter on left must trigger SPLIT_LEFT_50 (0.00, 0.00, 0.50, 1.00)
        pd_with_edges = PresenterDetector()
        frame_with_edges = frame_no_edges.copy()
        for y in range(60, 300, 40):
            cv2.putText(frame_with_edges, f"Slide Content Line {y}", (w // 2 + 20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, 240, 1)

        for _ in range(4):
            pd_with_edges.update(frame_with_edges.copy(), detected_faces=big_left_face)

        self.assertEqual(
            pd_with_edges.detected_box,
            (0.00, 0.00, 0.50, 1.00),
            "Split-screen masking failed to latch SPLIT_LEFT_50 when non-presenter half had slide edges"
        )


class TestHumanDetectorEdgeSeamUnification(unittest.TestCase):
    """Tests verifying the unified edge-density seam in HumanNarratorDetector."""

    def test_off_center_stage_speaker_without_slides_is_narrator(self):
        """Off-center stage speaker on a dark background (no slides) must be flagged as narrator.

        Spec: 'if is_prominent and not is_central: if not _has_complementary_slide_content ...'
        A genuinely off-center face (cx < 0.15 or cx > 0.85, e.g. fn_x=0.04, fn_w=0.08 -> cx=0.08,
        h=14% >= 10%) on a dark stage must exercise the off-center narrator branch.
        """
        from deckspool.human_detector import HumanNarratorDetector
        detector = HumanNarratorDetector()

        # Dark auditorium stage — no slide edges at all
        frame = np.full((720, 1280, 3), 30, dtype=np.uint8)

        # 1. Left off-center face: cx = 0.04 + 0.08 / 2 = 0.08 (< 0.15), h = 0.14 (>= 0.10)
        off_center_face_left = [(0.04, 0.20, 0.08, 0.14, 0.91)]
        is_narrator, reason = detector.is_narrator_frame(frame, detected_faces=off_center_face_left)
        self.assertTrue(
            is_narrator,
            f"Left off-center stage speaker not classified as narrator. Reason: {reason}"
        )
        self.assertIn(
            "Large off-center speaker",
            reason,
            f"Expected off-center narrator branch to be exercised, but got reason: {reason}"
        )

        # 2. Right off-center face: cx = 0.88 + 0.08 / 2 = 0.92 (> 0.85), h = 0.14 (>= 0.10)
        off_center_face_right = [(0.88, 0.20, 0.08, 0.14, 0.91)]
        is_narrator_r, reason_r = detector.is_narrator_frame(frame, detected_faces=off_center_face_right)
        self.assertTrue(
            is_narrator_r,
            f"Right off-center stage speaker not classified as narrator. Reason: {reason_r}"
        )
        self.assertIn(
            "Large off-center speaker",
            reason_r,
            f"Expected off-center narrator branch to be exercised, but got reason: {reason_r}"
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
        self.assertEqual(
            reason,
            "Slide content with presenter/photo",
            f"Expected reason 'Slide content with presenter/photo', got: {reason}"
        )

    def test_has_slide_content_supersedes_has_complementary_slide_content(self):
        """_has_complementary_slide_content observable behavior must match _has_slide_content
        on the complementary half-frame ROI."""
        from deckspool.human_detector import HumanNarratorDetector
        detector = HumanNarratorDetector()

        h, w = 360, 640
        # Create a frame with slide text on the left half and blank on the right half
        frame_slide_left = np.full((h, w, 3), 240, dtype=np.uint8)
        frame_slide_left[:, w // 2:] = 30  # right half is dark/blank
        for y in range(40, 320, 40):
            cv2.putText(frame_slide_left, f"Slide Content Line {y}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        left_half = frame_slide_left[:, :w // 2]
        right_half = frame_slide_left[:, w // 2:]

        # When presenter face is on the right (cx=0.75 >= 0.55):
        # complementary side is left half (which has slide text)
        self.assertTrue(detector._has_complementary_slide_content(frame_slide_left, 0.75))
        # Must agree with _has_slide_content on that exact ROI
        self.assertEqual(
            detector._has_complementary_slide_content(frame_slide_left, 0.75),
            detector._has_slide_content(left_half)
        )

        # When presenter face is on the left (cx=0.25 <= 0.45):
        # complementary side is right half (which is blank)
        self.assertFalse(detector._has_complementary_slide_content(frame_slide_left, 0.25))
        self.assertEqual(
            detector._has_complementary_slide_content(frame_slide_left, 0.25),
            detector._has_slide_content(right_half)
        )

        # When presenter face is centered (0.45 < cx < 0.55):
        # neither half is complementary, so it returns False
        self.assertFalse(detector._has_complementary_slide_content(frame_slide_left, 0.50))

        # Symmetric check: slide text on right half, blank on left half
        frame_slide_right = np.full((h, w, 3), 30, dtype=np.uint8)
        frame_slide_right[:, w // 2:] = 240
        for y in range(40, 320, 40):
            cv2.putText(frame_slide_right, f"Slide Line {y}", (w // 2 + 20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        self.assertTrue(detector._has_complementary_slide_content(frame_slide_right, 0.25))
        self.assertFalse(detector._has_complementary_slide_content(frame_slide_right, 0.75))

        # Robustness check: 2D grayscale frames passed to _has_slide_content
        gray_2d = cv2.cvtColor(frame_slide_left, cv2.COLOR_BGR2GRAY)
        self.assertTrue(detector._has_slide_content(gray_2d))

        # Robustness check: 3D BGR frames passed to PresenterDetector._has_slide_edges
        self.assertTrue(PresenterDetector._has_slide_edges(frame_slide_left[:, :w // 2]))
        self.assertFalse(PresenterDetector._has_slide_edges(frame_slide_left[:, w // 2:]))


if __name__ == "__main__":
    unittest.main()
