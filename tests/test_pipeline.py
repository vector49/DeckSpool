"""Automated integration and unit tests for YT2PDF pipeline."""

import os
import queue
import tempfile
import threading
import unittest
import numpy as np
import cv2

from yt2pdf.config import ExtractionConfig, SourceType, PresenterMaskMode, WorkerMessageType
from yt2pdf.roi import mask_player_chrome, mask_box, PresenterDetector, detect_canvas_crop_box
from yt2pdf.detector import SlideDetector, compute_opencv_ssim
from yt2pdf.pdf_builder import compile_slides_to_pdf
from yt2pdf.worker import ExtractionWorker


class TestYT2PDFPipeline(unittest.TestCase):
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
        temp_dir = tempfile.mkdtemp(prefix="yt2pdf_test_")
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
        from yt2pdf.downloader import sanitize_filename
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
        from yt2pdf.gui import YT2PDFApp
        app = YT2PDFApp()
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
        from yt2pdf.gui import DeduplicatePDFDialog
        dialog = DeduplicatePDFDialog(app, default_pdf_path="dummy_presentation.pdf")
        app.update_idletasks()
        self.assertIsNotNone(dialog.run_btn)
        self.assertIsNotNone(dialog.progress_bar)
        self.assertEqual(dialog.threshold_label.cget("text"), "94%")
        dialog.destroy()

        app.destroy()

    def test_deduplicate_slides_list(self):
        """Verify that repetitive slides across presentation are truncated, holding 1st appearance."""
        from yt2pdf.detector import Slide
        from yt2pdf.deduplicator import deduplicate_slides_list
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
        from yt2pdf.deduplicator import deduplicate_pdf

        temp_dir = tempfile.mkdtemp(prefix="yt2pdf_dedup_test_")
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
        from yt2pdf.human_detector import HumanNarratorDetector

        detector = HumanNarratorDetector()
        narrator_img_path = r"C:/Users/Vignesh/.gemini/antigravity/brain/d1ba7920-44b2-4dfa-8534-9c4c4c5c56dc/.user_uploaded/media_1792502282083_adde35e6.png"

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

    def test_deduplicate_pdf_with_narrator_filtering(self):
        """Verify that deduplicate_pdf truncates full-human narrator pages from a PDF."""
        import img2pdf
        import pypdfium2
        from yt2pdf.deduplicator import deduplicate_pdf

        temp_dir = tempfile.mkdtemp(prefix="yt2pdf_narrator_test_")
        input_pdf_path = os.path.join(temp_dir, "presentation_with_speaker_cuts.pdf")
        output_pdf_path = os.path.join(temp_dir, "presentation_cleaned.pdf")

        narrator_img_path = r"C:/Users/Vignesh/.gemini/antigravity/brain/d1ba7920-44b2-4dfa-8534-9c4c4c5c56dc/.user_uploaded/media_1792502282083_adde35e6.png"
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


class TestPathAutoformatting(unittest.TestCase):
    """Test suite for path cleaning, quote stripping, and Explorer copy autoformatting."""

    def test_file_explorer_copied_path_autoformatting(self):
        from yt2pdf.downloader import clean_file_path, is_local_file_path

        # 1. Exact case from user prompt: Windows File Explorer "Copy as path"
        raw_explorer = r'"C:\Users\Vignesh\Videos\2026-10-30 10-34-09.mp4"'
        cleaned = clean_file_path(raw_explorer)
        expected = r"C:\Users\Vignesh\Videos\2026-10-30 10-34-09.mp4"
        self.assertEqual(cleaned, expected)
        self.assertTrue(is_local_file_path(cleaned))

        # 2. Single quotes
        raw_single = r"'C:\Users\Vignesh\Videos\2026-10-30 10-34-09.mp4'"
        self.assertEqual(clean_file_path(raw_single), expected)

        # 3. Surrounding whitespace with quotes
        raw_padded = r'   "C:\Users\Vignesh\Videos\2026-10-30 10-34-09.mp4"   '
        self.assertEqual(clean_file_path(raw_padded), expected)

        # 4. Forward slashes with quotes
        raw_fslash = r'"C:/Users/Vignesh/Videos/2026-10-30 10-34-09.mp4"'
        self.assertEqual(clean_file_path(raw_fslash), expected)

        # 5. file:/// URI (with encoded spaces %20)
        raw_uri = "file:///C:/Users/Vignesh/Videos/2026-10-30%2010-34-09.mp4"
        self.assertEqual(clean_file_path(raw_uri), expected)

        # 6. PowerShell '&' call syntax
        raw_ps = r'& "C:\Users\Vignesh\Videos\2026-10-30 10-34-09.mp4"'
        self.assertEqual(clean_file_path(raw_ps), expected)

        # 7. Git Bash format on Windows
        raw_git = "/c/Users/Vignesh/Videos/2026-10-30 10-34-09.mp4"
        self.assertEqual(clean_file_path(raw_git), expected)

        # 8. WSL format on Windows
        raw_wsl = "/mnt/c/Users/Vignesh/Videos/2026-10-30 10-34-09.mp4"
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
        from yt2pdf.gui import YT2PDFApp, DeduplicatePDFDialog
        from yt2pdf.config import SourceType

        app = YT2PDFApp()
        app.update_idletasks()

        # Simulate user copying from File Explorer with double quotes
        raw_explorer = r'"C:\Users\Vignesh\Videos\2026-10-30 10-34-09.mp4"'
        app._set_input_value(raw_explorer)

        # 1. Strips quotes
        self.assertEqual(app.input_entry.get(), r"C:\Users\Vignesh\Videos\2026-10-30 10-34-09.mp4")
        # 2. Auto-switches mode to Local File
        self.assertEqual(app.source_segmented.get(), SourceType.LOCAL_FILE.value)
        # 3. Auto-suggests destination PDF filename next to the video
        self.assertEqual(app.output_entry.get(), r"C:\Users\Vignesh\Videos\2026-10-30 10-34-09_slides.pdf")

        # Test deduplicate dialog with quoted path
        raw_pdf = r'"C:\Users\Vignesh\Videos\2026-10-30 10-34-09_slides.pdf"'
        dialog = DeduplicatePDFDialog(app, default_pdf_path=raw_pdf)
        dialog.update_idletasks()

        self.assertEqual(dialog.input_entry.get(), r"C:\Users\Vignesh\Videos\2026-10-30 10-34-09_slides.pdf")
        self.assertEqual(dialog.output_entry.get(), r"C:\Users\Vignesh\Videos\2026-10-30 10-34-09_slides_dedup.pdf")

        dialog.destroy()
        app.destroy()


if __name__ == "__main__":
    unittest.main()
