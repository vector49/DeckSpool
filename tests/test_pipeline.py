"""Automated integration and unit tests for DeckSpool pipeline."""

import os
import queue
import tempfile
import threading
import unittest
import numpy as np
import cv2

from deckspool.config import ExtractionConfig, SourceType, PresenterMaskMode, WorkerMessageType
from deckspool.roi import mask_player_chrome, mask_box, PresenterDetector, detect_canvas_crop_box
from deckspool.detector import SlideDetector, compute_opencv_ssim
from deckspool.pdf_builder import compile_slides_to_pdf
from deckspool.worker import ExtractionWorker


class TestDeckSpoolPipeline(unittest.TestCase):
    """Test suite for core components and synthetic end-to-end processing."""

    def test_ssim_calculation(self):
        """Verify OpenCV SSIM on identical, similar, and distinct frames."""
        img1 = np.ones((360, 640), dtype=np.uint8) * 128
        img2 = img1.copy()
        
        # Identical frames should yield SSIM ~ 1.0
        self.assertAlmostEqual(compute_opencv_ssim(img1, img2), 1.0, places=3)

        # Slightly noisy frames on actual slide content should yield high similarity (> 0.90)
        slide_base = np.zeros((360, 640), dtype=np.uint8)
        cv2.putText(slide_base, "Slide Title & Content", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 3)
        noise = np.random.normal(0, 2, slide_base.shape).astype(np.int16)
        img_noisy = np.clip(slide_base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        self.assertGreater(compute_opencv_ssim(slide_base, img_noisy), 0.90)

        # Radically different frames (black vs white)
        img_white = np.ones((360, 640), dtype=np.uint8) * 255
        img_black = np.zeros((360, 640), dtype=np.uint8)
        self.assertLess(compute_opencv_ssim(img_white, img_black), 0.05)

    def test_roi_and_chrome_masking(self):
        """Verify chrome and ROI box masking."""
        frame = np.ones((100, 200, 3), dtype=np.uint8) * 200
        masked_chrome = mask_player_chrome(frame, top_pct=0.10, bottom_pct=0.10)

        # Check top 10% is 0
        self.assertTrue(np.all(masked_chrome[:10, :] == 0))
        # Check bottom 10% is 0
        self.assertTrue(np.all(masked_chrome[90:, :] == 0))
        # Check middle is unchanged
        self.assertTrue(np.all(masked_chrome[20:80, :] == 200))

        # Check normalized box masking (bottom-right: x=0.5, y=0.5, w=0.5, h=0.5)
        masked_box = mask_box(frame, (0.5, 0.5, 0.5, 0.5))
        self.assertTrue(np.all(masked_box[50:, 100:] == 0))
        self.assertTrue(np.all(masked_box[:40, :80] == 200))

    def test_canvas_auto_crop(self):
        """Verify black border detection for letterboxed presentations."""
        # Frame with 1080x1920 with 100px letterboxing top/bottom
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Content region
        frame[100:980, 150:1770] = 220

        crop_box = detect_canvas_crop_box([frame, frame])
        self.assertIsNotNone(crop_box)
        x1, y1, x2, y2 = crop_box
        self.assertAlmostEqual(y1, 100, delta=10)
        self.assertAlmostEqual(y2, 980, delta=10)
        self.assertAlmostEqual(x1, 150, delta=10)
        self.assertAlmostEqual(x2, 1770, delta=10)

    def test_debounce_and_progressive_settling(self):
        """Verify that animated bullet points settle into a single final slide."""
        config = ExtractionConfig(
            sample_interval=1.0,
            similarity_threshold=0.92,
            debounce_settle_time=2.0,
            min_slide_duration=2.0
        )
        detector = SlideDetector(config)

        # Create 3 visual states:
        # State 1: Slide A (t = 0 to 3s)
        # State 2: Slide B bullet 1 (t = 4s)
        # State 3: Slide B bullet 2 (t = 5s)
        # State 4: Slide B bullet 3 - final (t = 6s to 10s)
        # State 5: Slide C (t = 11s to 15s)

        slide_a = np.ones((720, 1280, 3), dtype=np.uint8) * 50
        cv2.putText(slide_a, "Slide A Title", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        slide_b_1 = np.ones((720, 1280, 3), dtype=np.uint8) * 100
        cv2.putText(slide_b_1, "Slide B Title", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.putText(slide_b_1, "- Point 1", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

        slide_b_2 = slide_b_1.copy()
        cv2.putText(slide_b_2, "- Point 2", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

        slide_b_final = slide_b_2.copy()
        cv2.putText(slide_b_final, "- Point 3 (Final)", (100, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

        slide_c = np.ones((720, 1280, 3), dtype=np.uint8) * 160
        cv2.putText(slide_c, "Slide C Conclusion", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        timestamps_and_frames = [
            (0.0, slide_a),
            (1.0, slide_a),
            (2.0, slide_a),
            (3.0, slide_a),
            (4.0, slide_b_1),
            (5.0, slide_b_2),
            (6.0, slide_b_final),
            (7.0, slide_b_final),
            (8.0, slide_b_final),
            (9.0, slide_b_final),
            (10.0, slide_c),
            (11.0, slide_c),
            (12.0, slide_c),
            (13.0, slide_c),
        ]

        for i, (t, f) in enumerate(timestamps_and_frames):
            is_last = (i == len(timestamps_and_frames) - 1)
            detector.process_frame(t, f, is_last_frame=is_last)

        # There should be exactly 3 unique committed slides: Slide A, Slide B (final settled), and Slide C
        self.assertEqual(len(detector.committed_slides), 3)

    def test_synthetic_video_worker_pipeline(self):
        """Creates a synthetic video file, runs ExtractionWorker end-to-end, and verifies output PDF."""
        temp_dir = tempfile.mkdtemp(prefix="deckspool_test_")
        video_path = os.path.join(temp_dir, "test_presentation.mp4")
        pdf_path = os.path.join(temp_dir, "test_output.pdf")

        # Generate a 6-second 10fps video with 2 distinct slides
        fps = 10.0
        width, height = 640, 360
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        # Slide 1 (frames 0 to 29 = 3s)
        frame1 = np.ones((height, width, 3), dtype=np.uint8) * 40
        cv2.putText(frame1, "Introduction Slide", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # Slide 2 (frames 30 to 59 = 3s)
        frame2 = np.ones((height, width, 3), dtype=np.uint8) * 120
        cv2.putText(frame2, "Architecture Slide", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        for _ in range(30):
            writer.write(frame1)
        for _ in range(30):
            writer.write(frame2)
        writer.release()

        # Run extraction worker
        config = ExtractionConfig(
            source_type=SourceType.LOCAL_FILE,
            input_source=video_path,
            output_path=pdf_path,
            sample_interval=1.0,
            similarity_threshold=0.90,
            debounce_settle_time=1.5,
            min_slide_duration=1.5,
            canvas_auto_crop=True
        )

        msg_queue = queue.Queue()
        cancel_event = threading.Event()
        worker = ExtractionWorker(config, msg_queue, cancel_event)
        worker.start()
        worker.join(timeout=15.0)

        # Collect messages
        messages = []
        while not msg_queue.empty():
            messages.append(msg_queue.get())

        msg_types = [m.msg_type for m in messages]
        self.assertIn(WorkerMessageType.COMPLETE, msg_types)
        self.assertTrue(os.path.exists(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 500)

        # Verify PDF header
        with open(pdf_path, "rb") as f:
            header = f.read(5)
            self.assertEqual(header, b"%PDF-")

        # Cleanup
        try:
            os.remove(video_path)
            os.remove(pdf_path)
            os.rmdir(temp_dir)
        except Exception:
            pass

    def test_filename_sanitizer(self):
        """Verify filename sanitization removes forbidden OS characters."""
        from deckspool.downloader import sanitize_filename
        dirty_title = 'My Video: Lecture #1 / Part 2? <Best> *Slides* | "HD"'
        clean_title = sanitize_filename(dirty_title)
        self.assertNotIn(":", clean_title)
        self.assertNotIn("/", clean_title)
        self.assertNotIn("?", clean_title)
        self.assertNotIn("<", clean_title)
        self.assertNotIn(">", clean_title)
        self.assertNotIn("|", clean_title)
        self.assertNotIn('"', clean_title)
        self.assertNotIn("*", clean_title)

    def test_presenter_motion_detector(self):
        """Verify motion variance detector isolates corner webcam activity."""
        detector = PresenterDetector()
        h, w = 360, 640
        static_slide = np.ones((h, w), dtype=np.uint8) * 180

        # Simulate 8 frames where top-right corner has animated motion
        for frame_idx in range(8):
            frame = static_slide.copy()
            # Top-right PIP region (x=0.70 to 0.98, y=0.04 to 0.40)
            x1, x2 = int(0.72 * w), int(0.96 * w)
            y1, y2 = int(0.06 * h), int(0.38 * h)
            # Add dynamic high-variance noise in top-right
            frame[y1:y2, x1:x2] = np.random.randint(0, 255, (y2 - y1, x2 - x1), dtype=np.uint8)
            detector.update(frame)

        # Presenter detector should identify top-right candidate box
        self.assertIsNotNone(detector.detected_box)
        detected_x, detected_y, detected_w, detected_h = detector.detected_box
        self.assertAlmostEqual(detected_x, 0.70, delta=0.05)
        self.assertAlmostEqual(detected_y, 0.04, delta=0.05)

    def test_gui_instantiation(self):
        """Verify GUI loads without runtime or layout exceptions, and completion dialog works."""
        from deckspool.gui import DeckSpoolApp
        app = DeckSpoolApp()
        app.update_idletasks()
        self.assertIsNotNone(app.start_btn)
        self.assertIsNotNone(app.progress_bar)
        self.assertIsNotNone(app.status_label)
        self.assertIsNotNone(app.val_eta)
        self.assertEqual(app.source_segmented.get(), "YouTube URL")

        # Test slider value label binding
        self.assertEqual(app.interval_val_lbl.cget("text"), "1.0s")
        app.sample_interval_slider.set(2.5)
        # Verify the slider value updates dynamically
        app.sample_interval_slider._command(2.5)
        self.assertEqual(app.interval_val_lbl.cget("text"), "2.5s")

        # Test completion dialog modal rendering without Tkinter option errors
        app._show_completion_dialog("dummy_presentation.pdf", slides_count=10, elapsed=45.2)
        app.update_idletasks()

        # Test cancellation resets progress bar to 0.0
        from unittest.mock import patch
        app.progress_bar.set(0.45)
        self.assertAlmostEqual(app.progress_bar.get(), 0.45, places=2)
        with patch("tkinter.messagebox.showinfo"):
            app._handle_cancellation()
        self.assertAlmostEqual(app.progress_bar.get(), 0.0, places=2)
        self.assertIn("cancelled", app.status_label.cget("text").lower())

        # Test DeduplicatePDFDialog modal initialization
        from deckspool.gui import DeduplicatePDFDialog
        dialog = DeduplicatePDFDialog(app, default_pdf_path="dummy_presentation.pdf")
        app.update_idletasks()
        self.assertIsNotNone(dialog.run_btn)
        self.assertIsNotNone(dialog.progress_bar)
        self.assertEqual(dialog.threshold_label.cget("text"), "94%")
        dialog.destroy()

        app.destroy()

    def test_deduplicate_slides_list(self):
        """Verify that repetitive slides across presentation are truncated, holding 1st appearance."""
        from deckspool.detector import Slide
        from deckspool.deduplicator import deduplicate_slides_list
        from PIL import Image
        import imagehash

        def make_slide(idx, title, timestamp):
            frame = np.ones((720, 1280, 3), dtype=np.uint8) * 50
            cv2.putText(frame, title, (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            gray = cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_BGR2GRAY)
            dhash = imagehash.dhash(Image.fromarray(gray))
            return Slide(index=idx, timestamp=timestamp, frame_high_res=frame, frame_analysis=gray, dhash=dhash)

        # Presentation sequence with recurring Agenda and recurring Title
        s1 = make_slide(1, "Title Slide", 0.0)
        s2 = make_slide(2, "Agenda Slide", 10.0)
        s3 = make_slide(3, "Architecture Diagram Overview", 20.0)
        s4 = make_slide(4, "Agenda Slide", 30.0)       # Duplicate of s2
        s5 = make_slide(5, "Performance Benchmarks Report", 40.0)
        s6 = make_slide(6, "Title Slide", 50.0)        # Duplicate of s1

        all_slides = [s1, s2, s3, s4, s5, s6]
        kept_slides, dup_records = deduplicate_slides_list(all_slides, similarity_threshold=0.92)

        # Expected: only s1, s2, s3, s5 kept (4 unique slides, re-indexed 1..4)
        self.assertEqual(len(kept_slides), 4)
        kept_indices = [s.index for s in kept_slides]
        self.assertEqual(kept_indices, [1, 2, 3, 4])

        # Duplicates record should map slide 4 -> 2, slide 6 -> 1
        self.assertEqual(len(dup_records), 2)
        dup_map = {rec[0]: rec[1] for rec in dup_records}
        self.assertEqual(dup_map[4], 2)
        self.assertEqual(dup_map[6], 1)

    def test_deduplicate_pdf_end_to_end(self):
        """Verify that deduplicate_pdf parses a PDF, identifies all duplicates, and produces a clean PDF."""
        import img2pdf
        import pypdfium2
        from deckspool.deduplicator import deduplicate_pdf

        temp_dir = tempfile.mkdtemp(prefix="deckspool_dedup_test_")
        input_pdf_path = os.path.join(temp_dir, "raw_with_duplicates.pdf")
        output_pdf_path = os.path.join(temp_dir, "cleaned_unique.pdf")

        def make_encoded_image(text):
            img = np.ones((720, 1280, 3), dtype=np.uint8) * 80
            cv2.putText(img, text, (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            _, buf = cv2.imencode(".png", img)
            return buf.tobytes()

        # Create 5-page PDF: Page 1 (A), Page 2 (B), Page 3 (A - dup), Page 4 (C), Page 5 (B - dup)
        buf_a = make_encoded_image("Page A - Slide Intro")
        buf_b = make_encoded_image("Page B - Slide Details")
        buf_c = make_encoded_image("Page C - Slide Summary")

        pages = [buf_a, buf_b, buf_a, buf_c, buf_b]
        with open(input_pdf_path, "wb") as f:
            f.write(img2pdf.convert(pages))

        # Run deduplicate_pdf
        result = deduplicate_pdf(
            input_pdf_path=input_pdf_path,
            output_pdf_path=output_pdf_path,
            similarity_threshold=0.94
        )

        self.assertEqual(result.total_pages, 5)
        self.assertEqual(result.unique_pages_count, 3)
        self.assertEqual(result.duplicates_removed_count, 2)
        self.assertEqual(result.kept_indices, [0, 1, 3])  # Pages 1, 2, 4 in 0-based

        # Verify duplicate records: Page 3 matches Page 1, Page 5 matches Page 2
        dup_map = {dup_p: orig_p for dup_p, orig_p, _ in result.duplicate_details}
        self.assertEqual(dup_map[3], 1)
        self.assertEqual(dup_map[5], 2)

        # Verify output PDF on disk
        self.assertTrue(os.path.exists(output_pdf_path))
        out_pdf = pypdfium2.PdfDocument(output_pdf_path)
        self.assertEqual(len(out_pdf), 3)
        out_pdf.close()

        # Clean up
        try:
            os.remove(input_pdf_path)
            os.remove(output_pdf_path)
            os.rmdir(temp_dir)
        except Exception:
            pass

    def test_human_narrator_detector(self):
        """Verify that HumanNarratorDetector detects speaker camera cuts and ignores slides / corner PIP."""
        from deckspool.human_detector import HumanNarratorDetector

        detector = HumanNarratorDetector()
        narrator_img_path = os.path.join(os.path.dirname(__file__), "fixtures", "narrator_sample.png")

        if os.path.exists(narrator_img_path):
            narrator_img = cv2.imread(narrator_img_path)
            is_narrator, reason = detector.is_narrator_frame(narrator_img)
            self.assertTrue(is_narrator, f"Failed to identify narrator image: {reason}")
            self.assertIn("narrator", reason.lower())

        # Test on clean presentation slide
        slide_img = np.ones((720, 1280, 3), dtype=np.uint8) * 240
        cv2.putText(slide_img, "SLIDE: DATA STRUCTURES & ALGORITHMS", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        cv2.putText(slide_img, "- Binary Search Trees", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(slide_img, "- Hash Maps & Collisions", (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        is_narrator, _ = detector.is_narrator_frame(slide_img)
        self.assertFalse(is_narrator, "Slide falsely classified as narrator frame")

        # Test conference stage narrator cutaway (e.g. medium camera cut to speaker: cx=0.51, h=12.6%)
        mock_stage_cut = np.ones((720, 1280, 3), dtype=np.uint8) * 50  # dark stage backdrop
        mock_stage_faces = [(0.48, 0.21, 0.06, 0.126, 0.92)]
        is_narrator_stage, stage_reason = detector.is_narrator_frame(mock_stage_cut, detected_faces=mock_stage_faces)
        self.assertTrue(is_narrator_stage, f"Failed to identify stage narrator cut: {stage_reason}")
        self.assertIn("narrator", stage_reason.lower())

        # Test medium-wide auditorium cutaway (cx=0.50, h=10.2%, area=0.55%)
        mock_wide_stage = np.ones((720, 1280, 3), dtype=np.uint8) * 45
        mock_wide_faces = [(0.475, 0.22, 0.054, 0.102, 0.88)]
        is_narrator_wide, wide_reason = detector.is_narrator_frame(mock_wide_stage, detected_faces=mock_wide_faces)
        self.assertTrue(is_narrator_wide, f"Failed to identify wide stage narrator cut: {wide_reason}")

        # Test corner PIP / podium speaker on presentation slide (cx=0.13, h=9.3%)
        mock_podium_slide = slide_img.copy()
        mock_podium_faces = [(0.11, 0.21, 0.04, 0.093, 0.91)]
        is_narrator_podium, podium_reason = detector.is_narrator_frame(mock_podium_slide, detected_faces=mock_podium_faces)
        self.assertFalse(is_narrator_podium, f"Corner podium speaker on slide falsely rejected: {podium_reason}")

        # Test 50/50 split screen slide (left presenter cx=0.25 h=15%, right slide content with dense text)
        mock_split_slide = np.ones((720, 1280, 3), dtype=np.uint8) * 30
        for y_offset in range(100, 650, 40):
            cv2.putText(mock_split_slide, f"std::vector<int> item_{y_offset}; // C++ metaprogramming", (680, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        mock_split_faces = [(0.21, 0.20, 0.08, 0.15, 0.93)]
        is_narrator_split, split_reason = detector.is_narrator_frame(mock_split_slide, detected_faces=mock_split_faces)
        self.assertFalse(is_narrator_split, f"Split-screen slide falsely classified as narrator: {split_reason}")

        # Test 50/50 split screen slide (right presenter cx=0.75 h=15%, left slide content with dense text)
        mock_split_right = np.ones((720, 1280, 3), dtype=np.uint8) * 30
        for y_offset in range(100, 650, 40):
            cv2.putText(mock_split_right, f"std::vector<int> item_{y_offset}; // C++ metaprogramming", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        mock_split_rfaces = [(0.71, 0.20, 0.08, 0.15, 0.93)]
        is_narrator_split_r, split_r_reason = detector.is_narrator_frame(mock_split_right, detected_faces=mock_split_rfaces)
        self.assertFalse(is_narrator_split_r, f"Right split-screen slide falsely classified as narrator: {split_r_reason}")

        # Test slide with centered author/speaker photo (cx=0.50, h=15%) surrounded by slide text
        mock_author_slide = np.ones((720, 1280, 3), dtype=np.uint8) * 240
        cv2.putText(mock_author_slide, "Keynote: Modern Software Architecture 2026", (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
        for i in range(6):
            cv2.putText(mock_author_slide, f"- Design Principle #{i+1}: Zero Overhead Encapsulation", (100, 180 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
        for i in range(4):
            cv2.putText(mock_author_slide, f"template <typename T> constexpr auto eval_{i}(T&& val);", (100, 520 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 0), 2)
        mock_author_faces = [(0.46, 0.32, 0.08, 0.15, 0.94)]
        is_narrator_author, author_reason = detector.is_narrator_frame(mock_author_slide, detected_faces=mock_author_faces)
        self.assertFalse(is_narrator_author, f"Slide with centered photo falsely classified as narrator: {author_reason}")

        # Test dark reddish-brown slide with low edge density: when neural detector found 0 faces,
        # it must NOT fall back to skin heuristics or falsely classify as narrator.
        dark_brown_slide = np.zeros((720, 1280, 3), dtype=np.uint8)
        dark_brown_slide[:] = (44, 53, 90)  # Presentation slide with reddish-brown background
        cv2.putText(dark_brown_slide, "Thank You! Questions?", (300, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        is_narrator_brown, brown_reason = detector.is_narrator_frame(dark_brown_slide, detected_faces=[])
        self.assertFalse(is_narrator_brown, f"Dark brown slide falsely classified as narrator: {brown_reason}")

        # Test fallback heuristic: only runs if detector is unavailable
        fallback_detector = HumanNarratorDetector()
        fallback_detector.model_path = ""
        fallback_detector._detector = None
        # In fallback mode, dark slides are rejected by minimum luminance threshold (Y >= 60)
        dark_slide_low_y = np.zeros((720, 1280, 3), dtype=np.uint8)
        dark_slide_low_y[:] = (20, 30, 50)  # BGR dark brown (Y=35 < 60)
        cv2.putText(dark_slide_low_y, "Thank You! Questions?", (300, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        is_n_fb, fb_reason = fallback_detector.is_narrator_frame(dark_slide_low_y)
        self.assertFalse(is_n_fb, f"Fallback falsely flagged dark brown slide: {fb_reason}")

        # Also test the actual CppCon talk final slide color (BGR: 90, 53, 44)
        cppcon_slide = np.zeros((720, 1280, 3), dtype=np.uint8)
        cppcon_slide[:] = (90, 53, 44)
        cv2.putText(cppcon_slide, "Final Slide", (300, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        is_n_cppcon, cppcon_reason = fallback_detector.is_narrator_frame(cppcon_slide)
        self.assertFalse(is_n_cppcon, f"Fallback falsely flagged CppCon slide: {cppcon_reason}")

        # When detected_faces=[] is explicitly passed even with fallback_detector (detector is None),
        # it must respect that 0 faces were found and NEVER fall back to skin heuristics.
        skin_slide = np.zeros((720, 1280, 3), dtype=np.uint8)
        skin_slide[:] = (65, 80, 140)  # Skin-matching color in YCrCb
        cv2.putText(skin_slide, "Slide with 0 detected faces", (200, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        is_n_explicit_empty, reason_explicit_empty = fallback_detector.is_narrator_frame(skin_slide, detected_faces=[])
        self.assertFalse(is_n_explicit_empty, f"Passing detected_faces=[] falsely triggered fallback skin detector: {reason_explicit_empty}")
        self.assertEqual(reason_explicit_empty, "Slide content (no central narrator)")

        # Degenerate frame geometry must not cause cv2 or ZeroDivisionError crashes in fallback check
        self.assertFalse(fallback_detector._fallback_skin_edge_check(np.zeros((1, 1, 3), dtype=np.uint8))[0])
        self.assertFalse(fallback_detector._fallback_skin_edge_check(np.zeros((0, 0, 3), dtype=np.uint8))[0])

    def test_lightning_talk_benchmark_pdfs(self):
        """Verify narrator detection.

        Primary path: when benchmark PDFs are present (developer workstation), checks exact page
        numbers that are narrator frames.
        Fallback path: synthetic 2-page PDF (slide + narrator cutaway), always runs in CI.
        """
        import pypdfium2
        import img2pdf
        from deckspool.human_detector import HumanNarratorDetector

        v1_path = r"C:\Users\Vignesh\Desk.Vignesh\Project DeckSpool\Test Cases\Lightning Talk - V1-1.pdf"
        pyside_path = r"C:\Users\Vignesh\Desk.Vignesh\Project DeckSpool\Test Cases\Lightning Talk - PySide6 Branch.pdf"

        detector = HumanNarratorDetector()

        if os.path.exists(v1_path) and os.path.exists(pyside_path):
            # Primary: verify exact known narrator pages on benchmark PDFs
            doc_v1 = pypdfium2.PdfDocument(v1_path)
            v1_narrator_pages = []
            for i in range(len(doc_v1)):
                pil_p = doc_v1[i].render(scale=0.5).to_pil()
                bgr = cv2.cvtColor(np.array(pil_p), cv2.COLOR_RGB2BGR)
                is_n, _ = detector.is_narrator_frame(bgr)
                if is_n:
                    v1_narrator_pages.append(i + 1)
            doc_v1.close()
            self.assertEqual(v1_narrator_pages, [2, 10], f"V1-1 narrator pages mismatch: {v1_narrator_pages}")

            doc_pyside = pypdfium2.PdfDocument(pyside_path)
            pyside_narrator_pages = []
            for i in range(len(doc_pyside)):
                pil_p = doc_pyside[i].render(scale=0.5).to_pil()
                bgr = cv2.cvtColor(np.array(pil_p), cv2.COLOR_RGB2BGR)
                is_n, _ = detector.is_narrator_frame(bgr)
                if is_n:
                    pyside_narrator_pages.append(i + 1)
            doc_pyside.close()
            self.assertEqual(pyside_narrator_pages, [], f"PySide6 Branch narrator pages mismatch: {pyside_narrator_pages}")
        else:
            # Fallback: synthetic 2-page PDF – slide then close-up narrator face.
            # This always runs in CI and other workstations without benchmark files.
            temp_dir = tempfile.mkdtemp(prefix="deckspool_benchmark_fallback_")
            try:
                # Page 1: plain slide (white bg, black text, high edge density)
                slide = np.full((720, 1280, 3), 240, dtype=np.uint8)
                cv2.putText(slide, "Introduction to Metaprogramming", (80, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 4)
                cv2.putText(slide, "Bullet point 1: Templates", (80, 320),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (40, 40, 40), 2)
                _, slide_png = cv2.imencode(".png", slide)

                # Page 2: close-up narrator face (skin-toned centre, low edge density)
                narrator = np.full((720, 1280, 3), 50, dtype=np.uint8)  # dark stage bg
                # Synthetic prominent face: central oval filled with skin tone
                cx, cy, rx, ry = 640, 300, 180, 220
                skin_bgr = (110, 160, 190)
                cv2.ellipse(narrator, (cx, cy), (rx, ry), 0, 0, 360, skin_bgr, -1)
                _, narrator_png = cv2.imencode(".png", narrator)

                pdf_path = os.path.join(temp_dir, "synthetic_benchmark.pdf")
                with open(pdf_path, "wb") as f:
                    f.write(img2pdf.convert([slide_png.tobytes(), narrator_png.tobytes()]))

                doc = pypdfium2.PdfDocument(pdf_path)
                flagged_pages = []
                for i in range(len(doc)):
                    pil_p = doc[i].render(scale=1.0).to_pil()
                    bgr = cv2.cvtColor(np.array(pil_p), cv2.COLOR_RGB2BGR)
                    is_n, _ = detector.is_narrator_frame(bgr)
                    if is_n:
                        flagged_pages.append(i + 1)
                doc.close()

                # Slide (page 1) must NOT be flagged; narrator cutaway (page 2) must be flagged.
                self.assertNotIn(1, flagged_pages, "Slide page 1 was incorrectly classified as narrator frame")
                self.assertIn(2, flagged_pages, "Narrator cutaway page 2 was not flagged by detector")
            finally:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)

    def test_deduplicate_pdf_with_narrator_filtering(self):
        """Verify that deduplicate_pdf truncates full-human narrator pages from a PDF."""
        import img2pdf
        import pypdfium2
        from deckspool.deduplicator import deduplicate_pdf

        temp_dir = tempfile.mkdtemp(prefix="deckspool_narrator_test_")
        input_pdf_path = os.path.join(temp_dir, "presentation_with_speaker_cuts.pdf")
        output_pdf_path = os.path.join(temp_dir, "presentation_cleaned.pdf")

        narrator_img_path = os.path.join(os.path.dirname(__file__), "fixtures", "narrator_sample.png")
        if not os.path.exists(narrator_img_path):
            return

        with open(narrator_img_path, "rb") as f:
            narrator_bytes = f.read()

        def make_slide_bytes(title):
            img = np.ones((720, 1280, 3), dtype=np.uint8) * 245
            cv2.putText(img, title, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
            _, buf = cv2.imencode(".png", img)
            return buf.tobytes()

        slide1_bytes = make_slide_bytes("Slide 1: Overview")
        slide2_bytes = make_slide_bytes("Slide 2: Implementation")

        # 4-page PDF: Slide 1, Narrator Cutaway, Slide 2, Narrator Cutaway
        pages = [slide1_bytes, narrator_bytes, slide2_bytes, narrator_bytes]
        with open(input_pdf_path, "wb") as f:
            f.write(img2pdf.convert(pages))

        result = deduplicate_pdf(
            input_pdf_path=input_pdf_path,
            output_pdf_path=output_pdf_path,
            filter_narrator_frames=True
        )

        self.assertEqual(result.total_pages, 4)
        self.assertEqual(result.unique_pages_count, 2)
        self.assertEqual(result.narrator_pages_removed_count, 2)
        # Pages 2 and 4 should be truncated as narrator shots
        narrator_page_nums = [p for p, _ in result.narrator_details]
        self.assertEqual(narrator_page_nums, [2, 4])

        # Verify output PDF only has 2 pages
        self.assertTrue(os.path.exists(output_pdf_path))
        out_pdf = pypdfium2.PdfDocument(output_pdf_path)
        self.assertEqual(len(out_pdf), 2)
        out_pdf.close()

        # Clean up
        try:
            os.remove(input_pdf_path)
            os.remove(output_pdf_path)
            os.rmdir(temp_dir)
        except Exception:
            pass

    def test_presenter_split_screen_detection(self):
        """Verify that PresenterDetector detects 50/50 split screen layouts."""
        from deckspool.roi import PresenterDetector, detect_canvas_crop_box, mask_box
        from deckspool.config import PresenterMaskMode

        pd = PresenterDetector()
        # Mock 360p frame with 50/50 split (left slide, right speaker)
        h, w = 360, 640
        frame_360 = np.zeros((h, w), dtype=np.uint8)
        frame_360[:, :w//2] = 20   # Dark slide background
        frame_360[:, w//2:] = 120  # Speaker webcam background

        # Feed updates with detected face on the right half (cx ~ 0.72)
        face_right = [(0.68, 0.20, 0.08, 0.15, 0.95)]
        for _ in range(4):
            pd.update(frame_360, detected_faces=face_right)

        self.assertEqual(pd.detected_box, (0.50, 0.00, 0.50, 1.00))
        box = pd.get_mask_box(PresenterMaskMode.AUTO)
        self.assertEqual(box, (0.50, 0.00, 0.50, 1.00))

        # Test canvas auto-crop on 50/50 split masked frame
        high_res = np.zeros((720, 1280, 3), dtype=np.uint8)
        high_res[:, :640] = 30  # slide content
        high_res[:, 640:] = 150  # webcam
        masked = mask_box(high_res, box)
        crop = detect_canvas_crop_box([masked, masked])
        self.assertEqual(crop, (0, 0, 640, 720))

    def test_committed_slide_preserves_pristine_high_res_without_mask_box(self):
        """Verify that enabling mask_presenter masks analysis frame only, not the high-res slide output."""
        from deckspool.detector import SlideDetector
        from deckspool.config import ExtractionConfig, PresenterMaskMode

        config = ExtractionConfig(
            mask_presenter=True,
            presenter_mode=PresenterMaskMode.BOTTOM_RIGHT,
            debounce_settle_time=1.0,
            min_slide_duration=1.0
        )
        detector = SlideDetector(config)

        # Non-zero white/content frame everywhere
        frame = np.full((720, 1280, 3), 200, dtype=np.uint8)
        # Feed frames to settle a slide
        detector.process_frame(0.0, frame)
        detector.process_frame(1.5, frame)
        detector.process_frame(3.0, frame, is_last_frame=True)

        self.assertGreater(len(detector.committed_slides), 0)
        slide = detector.committed_slides[0]
        high_res = slide.frame_high_res
        # Verify that bottom-right presenter area is NOT blacked out in high_res
        # Bottom-right box is (0.72, 0.60, 0.26, 0.35)
        br_region = high_res[int(0.60 * 720):int((0.60 + 0.35) * 720), int(0.72 * 1280):int((0.72 + 0.26) * 1280)]
        self.assertTrue(np.all(br_region == 200), "High-res output must preserve original slide content without black mask box")

    def test_detector_null_and_empty_frame_handling(self):
        """Verify that SlideDetector handles None and 0-size frames gracefully without crashing."""
        from deckspool.detector import SlideDetector
        from deckspool.config import ExtractionConfig

        detector = SlideDetector(ExtractionConfig(min_slide_duration=1.0, debounce_settle_time=1.0))
        # 1. Normal frame to establish candidate
        frame = np.full((720, 1280, 3), 150, dtype=np.uint8)
        cv2.putText(frame, "Slide 1 Content", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        detector.process_frame(0.0, frame)
        detector.process_frame(1.5, frame)
        self.assertEqual(len(detector.committed_slides), 1)

        # 2. None frame mid-stream should not raise AttributeError
        res = detector.process_frame(2.0, None)
        self.assertIsNone(res)

        # 3. Empty frame mid-stream should not raise OpenCV error
        res_empty = detector.process_frame(2.5, np.empty((0, 0, 3), dtype=np.uint8))
        self.assertIsNone(res_empty)

        # 4. Introduce a new candidate slide before EOF
        frame2 = np.full((720, 1280, 3), 50, dtype=np.uint8)
        cv2.putText(frame2, "Slide 2 Content", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        detector.process_frame(3.0, frame2)

        # 5. None frame at EOF should cleanly flush candidate slide
        final = detector.process_frame(4.0, None, is_last_frame=True)
        self.assertIsNotNone(final)
        self.assertEqual(len(detector.committed_slides), 2)



class TestPathAutoformatting(unittest.TestCase):
    """Test suite for path cleaning, quote stripping, and Explorer copy autoformatting."""

    def test_file_explorer_copied_path_autoformatting(self):
        from deckspool.downloader import clean_file_path, is_local_file_path

        # 1. Exact case from user prompt: Windows File Explorer "Copy as path"
        raw_explorer = r'"C:\Users\TestUser\Videos\2026-10-30 10-34-09.mp4"'
        cleaned = clean_file_path(raw_explorer)
        expected = r"C:\Users\TestUser\Videos\2026-10-30 10-34-09.mp4"
        self.assertEqual(cleaned, expected)
        self.assertTrue(is_local_file_path(cleaned))

        # 2. Single quotes
        raw_single = r"'C:\Users\TestUser\Videos\2026-10-30 10-34-09.mp4'"
        self.assertEqual(clean_file_path(raw_single), expected)

        # 3. Surrounding whitespace with quotes
        raw_padded = r'   "C:\Users\TestUser\Videos\2026-10-30 10-34-09.mp4"   '
        self.assertEqual(clean_file_path(raw_padded), expected)

        # 4. Forward slashes with quotes
        raw_fslash = r'"C:/Users/TestUser/Videos/2026-10-30 10-34-09.mp4"'
        self.assertEqual(clean_file_path(raw_fslash), expected)

        # 5. file:/// URI (with encoded spaces %20)
        raw_uri = "file:///C:/Users/TestUser/Videos/2026-10-30%2010-34-09.mp4"
        self.assertEqual(clean_file_path(raw_uri), expected)

        # 6. PowerShell '&' call syntax
        raw_ps = r'& "C:\Users\TestUser\Videos\2026-10-30 10-34-09.mp4"'
        self.assertEqual(clean_file_path(raw_ps), expected)

        # 7. Git Bash format on Windows
        raw_git = "/c/Users/TestUser/Videos/2026-10-30 10-34-09.mp4"
        self.assertEqual(clean_file_path(raw_git), expected)

        # 8. WSL format on Windows
        raw_wsl = "/mnt/c/Users/TestUser/Videos/2026-10-30 10-34-09.mp4"
        self.assertEqual(clean_file_path(raw_wsl), expected)

        # 9. Normal YouTube URL - should preserve URL structure
        yt_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        self.assertEqual(clean_file_path(yt_url), yt_url)
        self.assertFalse(is_local_file_path(yt_url))

        # 10. Quoted YouTube URL - strips quotes while preserving URL
        quoted_yt = '"https://youtu.be/dQw4w9WgXcQ"'
        self.assertEqual(clean_file_path(quoted_yt), "https://youtu.be/dQw4w9WgXcQ")
        self.assertFalse(is_local_file_path(quoted_yt))

    def test_gui_path_autoformatting(self):
        from deckspool.gui import DeckSpoolApp, DeduplicatePDFDialog
        from deckspool.config import SourceType

        app = DeckSpoolApp()
        app.update_idletasks()

        # Simulate user copying from File Explorer with double quotes
        raw_explorer = r'"C:\Users\TestUser\Videos\2026-10-30 10-34-09.mp4"'
        app._set_input_value(raw_explorer)

        # 1. Strips quotes
        self.assertEqual(app.input_entry.get(), r"C:\Users\TestUser\Videos\2026-10-30 10-34-09.mp4")
        # 2. Auto-switches mode to Local File
        self.assertEqual(app.source_segmented.get(), SourceType.LOCAL_FILE.value)
        # 3. Auto-suggests destination PDF filename next to the video
        self.assertEqual(app.output_entry.get(), r"C:\Users\TestUser\Videos\2026-10-30 10-34-09_slides.pdf")

        # Test deduplicate dialog with quoted path
        raw_pdf = r'"C:\Users\TestUser\Videos\2026-10-30 10-34-09_slides.pdf"'
        dialog = DeduplicatePDFDialog(app, default_pdf_path=raw_pdf)
        dialog.update_idletasks()

        self.assertEqual(dialog.input_entry.get(), r"C:\Users\TestUser\Videos\2026-10-30 10-34-09_slides.pdf")
        self.assertEqual(dialog.output_entry.get(), r"C:\Users\TestUser\Videos\2026-10-30 10-34-09_slides_dedup.pdf")

        dialog.destroy()
        app.destroy()


if __name__ == "__main__":
    unittest.main()
