"""Unit and integration tests for PySide6 Qt6 GUI in YT2PDF."""

import os
import sys
import queue
import tempfile
import threading
import unittest
import numpy as np

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from yt2pdf.gui import YT2PDFApp, DeduplicatePDFDialog, CompletionDialog
from yt2pdf.config import SourceType, PresenterMaskMode, WorkerMessage, WorkerMessageType
from yt2pdf.deduplicator import DeduplicationResult


class TestPySide6GUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv if sys.argv else [""])

    def setUp(self):
        self.gui = YT2PDFApp()
        self.gui.update_idletasks()

    def tearDown(self):
        self.gui.destroy()
        self.gui.update_idletasks()

    def test_window_geometry_and_title(self):
        """Test root container minimum and initial sizes and window title."""
        self.assertIn("YT2PDF", self.gui.windowTitle())
        self.assertGreaterEqual(self.gui.minimumWidth(), 980)
        self.assertGreaterEqual(self.gui.minimumHeight(), 560)

    def test_header_components(self):
        """Test header buttons and titles."""
        self.assertIsNotNone(self.gui.header_dedup_btn)
        self.assertIsNotNone(self.gui.header_view_pdf_btn)
        self.assertEqual(self.gui.header_view_pdf_btn.text(), "View PDF")
        self.assertEqual(self.gui.header_dedup_btn.text(), "Deduplicate PDF")

    def test_scroll_area_configuration(self):
        """Verify left column QScrollArea settings for smooth scrolling."""
        self.assertTrue(self.gui.scroll_area.widgetResizable())
        self.assertEqual(self.gui.scroll_area.verticalScrollBar().singleStep(), 16)

    def test_source_type_toggle(self):
        """Verify switching between YouTube URL and Local File updates inputs, keeps both buttons visible, and applies yellow/grey styles."""
        self.assertEqual(self.gui.source_segmented.get(), SourceType.YOUTUBE.value)
        self.assertTrue(self.gui.input_browse_btn.isHidden())
        self.assertEqual(self.gui.input_label.text(), "YouTube URL:")

        # Both buttons must be visible at all times (neither should disappear)
        btn_yt = self.gui.source_segmented._buttons[SourceType.YOUTUBE.value]
        btn_local = self.gui.source_segmented._buttons[SourceType.LOCAL_FILE.value]
        self.assertFalse(btn_yt.isHidden())
        self.assertFalse(btn_local.isHidden())

        # When YouTube is selected: YouTube is yellow (#ffd166), Local File is greyed out (#e5e7eb)
        self.assertIn("#ffd166", btn_yt.styleSheet())
        self.assertIn("#e5e7eb", btn_local.styleSheet())

        # Switch to Local File
        self.gui.source_segmented._on_btn_click(SourceType.LOCAL_FILE.value)
        self.gui.update_idletasks()
        self.assertEqual(self.gui.source_segmented.get(), SourceType.LOCAL_FILE.value)
        self.assertFalse(self.gui.input_browse_btn.isHidden())
        self.assertEqual(self.gui.input_label.text(), "Local Video File:")

        # Neither button disappears, styles flip: Local File is yellow, YouTube is greyed out
        self.assertFalse(btn_yt.isHidden())
        self.assertFalse(btn_local.isHidden())
        self.assertIn("#ffd166", btn_local.styleSheet())
        self.assertIn("#e5e7eb", btn_yt.styleSheet())

        # Switch back to YouTube
        self.gui.source_segmented._on_btn_click(SourceType.YOUTUBE.value)
        self.gui.update_idletasks()
        self.assertEqual(self.gui.source_segmented.get(), SourceType.YOUTUBE.value)
        self.assertTrue(self.gui.input_browse_btn.isHidden())
        self.assertEqual(self.gui.input_label.text(), "YouTube URL:")
        self.assertIn("#ffd166", btn_yt.styleSheet())
        self.assertIn("#e5e7eb", btn_local.styleSheet())

    def test_source_buttons_equi_spaced(self):
        """Verify that the two buttons are equi-spaced between them and the outer carrier box."""
        self.gui.show()
        self.gui.update_idletasks()
        btn_yt = self.gui.source_segmented._buttons[SourceType.YOUTUBE.value]
        btn_local = self.gui.source_segmented._buttons[SourceType.LOCAL_FILE.value]
        card = self.gui.findChild(object, "card_01")

        # Map button positions to card coordinates
        pos_yt = self.gui.source_segmented.mapTo(card, btn_yt.geometry().topLeft())
        pos_local = self.gui.source_segmented.mapTo(card, btn_local.geometry().topLeft())

        card_border = int(round(self.gui.theme.border_width))
        card_inner_left = card_border  # card border width
        card_inner_right = card.width() - card_border

        left_gap = pos_yt.x() - card_inner_left
        center_gap = pos_local.x() - (pos_yt.x() + btn_yt.width())
        right_gap = card_inner_right - (pos_local.x() + btn_local.width())

        # Left, center, and right gaps must be equal (16px)
        self.assertEqual(left_gap, 16)
        self.assertEqual(center_gap, 16)
        self.assertEqual(right_gap, 16)
        # Both buttons must have equal width and height
        self.assertEqual(btn_yt.width(), btn_local.width())
        self.assertEqual(btn_yt.height(), 36)
        self.assertEqual(btn_local.height(), 36)

    def test_input_path_sanitization_and_auto_switching(self):
        """Verify path cleaning, quote stripping, and automatic mode switching."""
        # 1. Local file path with quotes
        raw_local = r'"C:\Users\Test\Videos\presentation.mp4"'
        self.gui._set_input_value(raw_local)
        self.assertEqual(self.gui.input_entry.get(), r"C:\Users\Test\Videos\presentation.mp4")
        self.assertEqual(self.gui.source_segmented.get(), SourceType.LOCAL_FILE.value)
        self.assertEqual(self.gui.output_entry.get(), r"C:\Users\Test\Videos\presentation_slides.pdf")

        # 2. YouTube URL
        raw_yt = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        self.gui._set_input_value(raw_yt)
        self.assertEqual(self.gui.input_entry.get(), raw_yt)
        self.assertEqual(self.gui.source_segmented.get(), SourceType.YOUTUBE.value)

    def test_sliders_and_pill_badge_sync(self):
        """Verify sliders correctly update float values and dynamic pill readouts."""
        # Sample Interval (0.5 to 5.0)
        self.assertEqual(self.gui.interval_val_lbl.cget("text"), "1.0s")
        self.gui.sample_interval_slider.set(3.5)
        self.assertAlmostEqual(self.gui.sample_interval_slider.get(), 3.5, places=1)
        self.assertEqual(self.gui.interval_val_lbl.cget("text"), "3.5s")

        # Similarity Threshold (80% to 99%)
        self.assertEqual(self.gui.threshold_val_lbl.cget("text"), "92%")
        self.gui.similarity_slider.set(0.95)
        self.assertAlmostEqual(self.gui.similarity_slider.get(), 0.95, places=2)
        self.assertEqual(self.gui.threshold_val_lbl.cget("text"), "95%")

        # Debounce (1.0 to 10.0)
        self.assertEqual(self.gui.debounce_val_lbl.cget("text"), "2.0s")
        self.gui.debounce_slider.set(4.0)
        self.assertAlmostEqual(self.gui.debounce_slider.get(), 4.0, places=1)
        self.assertEqual(self.gui.debounce_val_lbl.cget("text"), "4.0s")

        # Min Slide Duration (1.0 to 15.0)
        self.assertEqual(self.gui.min_dur_val_lbl.cget("text"), "2.0s")
        self.gui.min_duration_slider.set(5.5)
        self.assertAlmostEqual(self.gui.min_duration_slider.get(), 5.5, places=1)
        self.assertEqual(self.gui.min_dur_val_lbl.cget("text"), "5.5s")

    def test_slider_wheel_ignored(self):
        """Verify wheel events on sliders are ignored so window scroll is not hijacked."""
        from PySide6.QtGui import QWheelEvent
        from PySide6.QtCore import QPointF, QPoint
        initial_val = self.gui.sample_interval_slider.get()
        event = QWheelEvent(QPointF(10, 10), QPointF(10, 10), QPoint(0, 0), QPoint(0, 120), Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False)
        self.gui.sample_interval_slider.wheelEvent(event)
        self.assertFalse(event.isAccepted())
        self.assertEqual(self.gui.sample_interval_slider.get(), initial_val)

    def test_roi_and_preprocessing_toggles(self):
        """Verify presenter mask toggle, dependent controls, greyed-out frame styling, and alignment."""
        # 1. Check switch text does not truncate
        sw = self.gui.mask_presenter_switch
        self.assertEqual(sw.text(), "Mask Presenter / PIP Webcam")
        self.assertGreaterEqual(sw.sizeHint().width(), 240)

        # 2. Check alignment: 16px margins on both sides relative to card_03
        self.gui.show()
        self.gui.update_idletasks()
        card3 = self.gui.findChild(object, "card_03")
        pos_in_card = self.gui.pip_mode_frame.mapTo(card3, self.gui.pip_mode_frame.rect().topLeft())
        card_border = int(round(self.gui.theme.border_width))
        card3_inner_left = card_border
        card3_inner_right = card3.width() - card_border
        left_gap = pos_in_card.x() - card3_inner_left
        right_gap = card3_inner_right - (pos_in_card.x() + self.gui.pip_mode_frame.width())
        self.assertEqual(left_gap, 16)
        self.assertEqual(right_gap, 16)

        # 3. Initially OFF: Frame is greyed out (#f3f4f6) and controls are disabled
        self.assertFalse(self.gui.mask_presenter_switch.isChecked())
        self.assertFalse(self.gui.pip_mode_dropdown.isEnabled())
        self.assertFalse(self.gui.custom_box_entry.isEnabled())
        self.assertIn("#f3f4f6", self.gui.pip_mode_frame.styleSheet())

        # 4. Toggle Presenter ON: Frame turns Buttercream (#fefce8)
        self.gui.mask_presenter_switch.setChecked(True)
        self.gui.update_idletasks()
        self.assertTrue(self.gui.pip_mode_dropdown.isEnabled())
        self.assertIn("#fefce8", self.gui.pip_mode_frame.styleSheet())
        # Default mode is AUTO, so custom box should remain disabled
        self.assertFalse(self.gui.custom_box_entry.isEnabled())

        # 5. Switch to CUSTOM mode
        self.gui.pip_mode_dropdown.set(PresenterMaskMode.CUSTOM.value)
        self.gui.update_idletasks()
        self.assertTrue(self.gui.custom_box_entry.isEnabled())

        # 6. Toggle Presenter OFF -> disables both and greys out frame again
        self.gui.mask_presenter_switch.setChecked(False)
        self.gui.update_idletasks()
        self.assertFalse(self.gui.pip_mode_dropdown.isEnabled())
        self.assertFalse(self.gui.custom_box_entry.isEnabled())
        self.assertIn("#f3f4f6", self.gui.pip_mode_frame.styleSheet())

    def test_worker_message_processing(self):
        """Verify queue draining updates status, progress, metrics, and thumbnails."""
        # Status message
        self.gui.msg_queue.put(WorkerMessage(WorkerMessageType.STATUS, "Downloading video..."))
        self.gui._process_queue()
        self.assertEqual(self.gui.status_label.text(), "Downloading video...")

        # Progress message
        self.gui.msg_queue.put(WorkerMessage(WorkerMessageType.PROGRESS, 0.65))
        self.gui._process_queue()
        self.assertAlmostEqual(self.gui.progress_bar.get(), 0.65, places=2)

        # Metrics message
        metrics_data = {
            "time_str": "01:23 / 05:00",
            "speed_str": "45 FPS (2.3x)",
            "slides_count": 8,
            "eta_str": "~1m 15s (02:30 PM)",
        }
        self.gui.msg_queue.put(WorkerMessage(WorkerMessageType.METRICS, metrics_data))
        self.gui._process_queue()
        self.assertEqual(self.gui.val_time.text(), "01:23 / 05:00")
        self.assertEqual(self.gui.val_speed.text(), "45 FPS (2.3x)")
        self.assertEqual(self.gui.val_slides.text(), "8 slides")
        self.assertEqual(self.gui.val_eta.text(), "~1m 15s (02:30 PM)")

        # Thumbnail message
        fake_rgb = np.ones((120, 160, 3), dtype=np.uint8) * 180
        self.gui.msg_queue.put(WorkerMessage(WorkerMessageType.THUMBNAIL, {"image_rgb": fake_rgb}))
        self.gui._process_queue()
        self.assertIsNotNone(self.gui.preview_label.pixmap())

    def test_completion_dialog_modal(self):
        """Verify completion dialog opens, displays metrics, and closes cleanly."""
        dialog = CompletionDialog(self.gui, "C:/test_path/presentation.pdf", slides_count=12, elapsed=75.4)
        self.gui.update_idletasks()
        self.assertIsNotNone(dialog)
        self.assertEqual(dialog.windowTitle(), "Extraction Complete")
        dialog.destroy()
        self.gui.update_idletasks()

    def test_deduplicate_dialog_lifecycle(self):
        """Verify DeduplicatePDFDialog initializes, updates threshold, and destroys cleanly."""
        dialog = DeduplicatePDFDialog(self.gui, default_pdf_path="C:/test/file.pdf")
        self.gui.update_idletasks()

        expected_in = os.path.normpath("C:/test/file.pdf")
        expected_out = os.path.normpath("C:/test/file_dedup.pdf")
        self.assertEqual(dialog.input_entry.get(), expected_in)
        self.assertEqual(dialog.output_entry.get(), expected_out)
        self.assertEqual(dialog.threshold_label.cget("text"), "94%")

        # Test threshold slider change
        dialog.similarity_slider.set(0.88)
        self.assertEqual(dialog.threshold_label.cget("text"), "88%")

        # Test queue progress update
        dialog.msg_queue.put(("PROGRESS", (0.5, 5, 10, "Analyzing page 5...")))
        dialog._process_queue()
        self.assertAlmostEqual(dialog.progress_bar.get(), 0.5, places=2)
        self.assertEqual(dialog.lbl_total.text(), "10")

        # Test completion result
        fake_result = DeduplicationResult(
            input_path="C:/test/file.pdf",
            output_path="C:/test/file_dedup.pdf",
            total_pages=10,
            unique_pages_count=8,
            duplicates_removed_count=1,
            narrator_pages_removed_count=1,
            duplicate_details=[(5, 2, 0.96)],
            narrator_details=[(3, "Speaker Face Detected")],
        )
        dialog.msg_queue.put(("COMPLETE", fake_result))
        dialog._process_queue()

        self.assertEqual(dialog.lbl_kept.text(), "8")
        self.assertEqual(dialog.lbl_pruned.text(), "1")
        self.assertEqual(dialog.lbl_narrator.text(), "1")
        self.assertTrue(dialog.open_pdf_btn.isEnabled())
        self.assertTrue(dialog.open_folder_btn.isEnabled())

        dialog.destroy()
        self.gui.update_idletasks()

    def test_dynamic_resizing_and_no_horizontal_scrollbar(self):
        """Verify left column cards dynamically resize with window without horizontal scrollbars or clipping."""
        self.gui.show()
        self.gui.update_idletasks()

        # Check horizontal scroll bar is permanently off
        self.assertEqual(self.gui.scroll_area.horizontalScrollBarPolicy(), Qt.ScrollBarAlwaysOff)

        for test_w, test_h in [(980, 560), (1080, 660), (1400, 800)]:
            self.gui.resize(test_w, test_h)
            self.gui.update_idletasks()

            vp_w = self.gui.scroll_area.viewport().width()
            col_w = self.gui.left_col_widget.width()

            # Left column widget must match viewport width (no overflow)
            self.assertEqual(col_w, vp_w)
            self.assertFalse(self.gui.scroll_area.horizontalScrollBar().isVisible())

            # Cards must stretch to fit viewport width with exactly 10px margin
            for card_id in ["card_01", "card_02", "card_03"]:
                card = self.gui.findChild(object, card_id)
                self.assertIsNotNone(card)
                self.assertEqual(card.width(), vp_w - 10)

    def test_deduplicate_dialog_scrollbar_policy_and_alignment(self):
        """Verify DeduplicatePDFDialog scroll policy, status box centering, and button label."""
        dialog = DeduplicatePDFDialog(self.gui)
        self.gui.update_idletasks()

        # Check scrollbar policy
        self.assertEqual(dialog.scroll.horizontalScrollBarPolicy(), Qt.ScrollBarAlwaysOff)
        self.assertFalse(dialog.scroll.horizontalScrollBar().isVisible())

        # Check run button text
        self.assertEqual(dialog.run_btn.text(), "Review and Truncate Duplicates")

        # Check status box alignment and centering
        self.assertEqual(dialog.status_label.alignment(), Qt.AlignCenter)
        card3 = dialog.card_results
        pos_box = dialog.dialog_status_box.mapTo(card3, dialog.dialog_status_box.rect().topLeft())
        pos_pb = dialog.progress_bar.mapTo(card3, dialog.progress_bar.rect().topLeft())
        self.assertEqual(pos_box.x(), pos_pb.x())
        self.assertEqual(dialog.dialog_status_box.width(), dialog.progress_bar.width())

        dialog.destroy()
        self.gui.update_idletasks()

    def test_view_pdf_button_handler(self):
        """Verify View PDF button invokes file opener when file exists or safely prompts when none."""
        # Case 1: When no output file exists and no last_generated_pdf, it safely exits without crashing
        self.gui.output_entry.setText("")
        self.gui.last_generated_pdf = ""
        # Mock QFileDialog.getOpenFileName to cancel
        with unittest.mock.patch("PySide6.QtWidgets.QFileDialog.getOpenFileName", return_value=("", "")):
            self.gui._on_view_pdf()

    def test_view_pdf_button_disabled_state(self):
        """Verify View PDF button starts disabled and is only enabled when output PDF is ready."""
        # 1. Initial launch: button must be disabled because no output is ready
        self.assertFalse(self.gui.header_view_pdf_btn.isEnabled())

        # 2. During processing: button must remain disabled
        self.gui._set_processing_ui_state(True)
        self.assertFalse(self.gui.header_view_pdf_btn.isEnabled())

        # 3. On completion with valid output file: button becomes enabled
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
            temp_pdf = tf.name

        try:
            with unittest.mock.patch.object(self.gui, "_show_completion_dialog"):
                self.gui._handle_completion({
                    "output_path": temp_pdf,
                    "slides_count": 5,
                    "elapsed_seconds": 12.0,
                })
            self.assertTrue(self.gui.header_view_pdf_btn.isEnabled())

            # 4. Starting a new extraction clears previous output and disables View PDF again
            self.gui.input_entry.setText("dummy_video.mp4")
            with unittest.mock.patch.object(self.gui, "worker", create=True):
                with unittest.mock.patch("yt2pdf.gui.ExtractionWorker"):
                    self.gui._on_start_clicked()
            self.assertFalse(self.gui.header_view_pdf_btn.isEnabled())
        finally:
            if os.path.exists(temp_pdf):
                os.remove(temp_pdf)

    def test_theme_toggle_button_placement_and_initial_state(self):
        """Verify sun/moon theme toggle button exists, is positioned to the left of DeDuplicate PDF button, and has custom Neo-Brutalist icon."""
        self.gui.show()
        self.gui.update_idletasks()

        self.assertIsNotNone(self.gui.theme_toggle_btn)
        self.assertIsNotNone(self.gui.header_dedup_btn)

        # Check layout order in header
        header_layout = self.gui.header_frame.layout()
        idx_toggle = header_layout.indexOf(self.gui.theme_toggle_btn)
        idx_dedup = header_layout.indexOf(self.gui.header_dedup_btn)
        self.assertLess(idx_toggle, idx_dedup, "Theme toggle button must be to the left of DeDuplicate PDF button in layout")

        # Check visual x coordinate: theme toggle button is strictly to the left of DeDuplicate PDF button
        pos_toggle = self.gui.theme_toggle_btn.mapTo(self.gui, self.gui.theme_toggle_btn.rect().topLeft())
        pos_dedup = self.gui.header_dedup_btn.mapTo(self.gui, self.gui.header_dedup_btn.rect().topLeft())
        self.assertLess(pos_toggle.x(), pos_dedup.x(), "Visual x position of theme toggle must be to the left of DeDuplicate PDF button")

        # Check initial Light Mode state
        self.assertFalse(self.gui.is_dark_mode())
        self.assertFalse(self.gui.theme_toggle_btn.icon().isNull(), "Theme toggle button should load the moon icon")
        self.assertEqual(self.gui.theme_toggle_btn.toolTip(), "Switch to Dark Theme")

    def test_theme_toggle_switches_to_dark_and_back(self):
        """Verify clicking the sun/moon button toggles between Light and Dark Neo-Brutalist themes and updates icons."""
        self.gui.show()
        self.gui.update_idletasks()

        # 1. Click toggle to switch to Dark Theme
        self.gui.theme_toggle_btn.click()
        self.gui.update_idletasks()

        self.assertTrue(self.gui.is_dark_mode())
        self.assertFalse(self.gui.theme_toggle_btn.icon().isNull(), "Theme toggle button should load the sun icon in dark mode")
        self.assertEqual(self.gui.theme_toggle_btn.toolTip(), "Switch to Light Theme")

        # Verify dark mode color tokens are applied
        self.assertIn("#121216", self.gui.styleSheet())
        self.assertIn("#121216", self.gui.root_widget.styleSheet())

        card1 = self.gui.findChild(object, "card_01")
        self.assertIn("#1a1a24", card1.styleSheet())
        self.assertIn("#ffffff", card1.styleSheet())

        # Verify dark mode input entry styling
        self.assertIn("#101014", self.gui.input_entry.styleSheet())
        self.assertIn("#ffffff", self.gui.input_entry.styleSheet())

        # 2. Click toggle again to switch back to Light Theme
        self.gui.theme_toggle_btn.click()
        self.gui.update_idletasks()

        self.assertFalse(self.gui.is_dark_mode())
        self.assertFalse(self.gui.theme_toggle_btn.icon().isNull(), "Theme toggle button should load the moon icon in light mode")
        self.assertEqual(self.gui.theme_toggle_btn.toolTip(), "Switch to Dark Theme")

        self.assertIn("#ede9f6", self.gui.styleSheet())
        self.assertIn("#ede9f6", self.gui.root_widget.styleSheet())
        self.assertIn("#ffffff", card1.styleSheet())
        self.assertIn("#000000", card1.styleSheet())

    def test_scrollbar_and_border_hierarchy(self):
        """Verify slim pastel scrollbar and border hierarchy (outer 2px vs inner 1.5px)."""
        self.gui.show()
        self.gui.update_idletasks()

        # 1. Check Scrollbar in Light Mode
        sb_ss = self.gui.scroll_area.styleSheet()
        self.assertIn("#c084fc", sb_ss, "Scrollbar handle should use pastel lilac color")
        self.assertIn("width: 12px", sb_ss, "Scrollbar should be a slim 12px carrier")
        self.assertIn("margin: 2px 2px 2px 2px", sb_ss, "Scrollbar handle should be equally spaced inside carrier")

        # 2. Check Border Hierarchy in Light Mode
        card2 = self.gui.findChild(object, "card_02")
        self.assertIn("border: 2px solid", card2.styleSheet(), "Outer card should have 2px border")
        self.assertIn("border: 1.5px solid", self.gui.interval_val_lbl.styleSheet(), "Inner value badge should have 1.5px border")

        # 3. Toggle to Dark Mode and Verify
        self.gui.toggle_theme()
        self.gui.update_idletasks()

        sb_ss_dark = self.gui.scroll_area.styleSheet()
        self.assertIn("#c084fc", sb_ss_dark, "Scrollbar handle should remain pastel in dark mode")
        self.assertIn("#1a1a24", sb_ss_dark, "Scrollbar carrier background should be dark in dark mode")
        self.assertIn("border: 1.5px solid #ffffff", sb_ss_dark, "Scrollbar carrier border should be white in dark mode")

        self.assertIn("border: 2px solid #ffffff", card2.styleSheet(), "Outer card should have 2px white border in dark mode")
        self.assertIn("border: 1.5px solid #ffffff", self.gui.interval_val_lbl.styleSheet(), "Inner badge should have 1.5px white border in dark mode")

        # Switch back to light
        self.gui.toggle_theme()

    def test_theme_applied_to_dialogs(self):
        """Verify modal dialogs adopt the active theme."""
        # Switch to dark theme
        self.gui.apply_theme(self.gui.theme.__class__(
            **{**self.gui.theme.__dict__, "name": "dark", "canvas_bg": "#121216", "card_bg": "#1a1a24", "border": "#ffffff"}
        ))
        from yt2pdf.gui import DARK_THEME
        self.gui.apply_theme(DARK_THEME)
        self.assertTrue(self.gui.is_dark_mode())

        # DeduplicatePDFDialog
        dialog = DeduplicatePDFDialog(self.gui, default_pdf_path="C:/dummy.pdf")
        self.assertIn("#121216", dialog.styleSheet())
        self.assertIn("#1a1a24", dialog.card_results.styleSheet())
        self.assertIn("#ffffff", dialog.card_results.styleSheet())
        dialog.destroy()

        # CompletionDialog
        comp_dialog = CompletionDialog(self.gui, "C:/dummy.pdf", slides_count=10, elapsed=30.0)
        self.assertIn("#121216", comp_dialog.styleSheet())
        comp_card = comp_dialog.findChild(object, "comp_card")
        self.assertIn("#1a1a24", comp_card.styleSheet())
        self.assertIn("#ffffff", comp_card.styleSheet())
        comp_dialog.destroy()


if __name__ == "__main__":
    unittest.main()
