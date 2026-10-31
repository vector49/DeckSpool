"""Modern CustomTkinter GUI for YT2PDF desktop application."""

import os
import sys
import queue
import logging
import threading
import subprocess
from typing import Optional
from PIL import Image

import customtkinter as ctk
from tkinter import filedialog, messagebox

from yt2pdf.config import (
    ExtractionConfig,
    SourceType,
    PresenterMaskMode,
    WorkerMessage,
    WorkerMessageType,
)
from yt2pdf.downloader import VideoDownloader, sanitize_filename, clean_file_path, is_local_file_path
from yt2pdf.worker import ExtractionWorker
from yt2pdf.deduplicator import deduplicate_pdf, DeduplicationResult

logger = logging.getLogger(__name__)

# Appearance & Color Palette
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

COLOR_CARD_BG = "#1e222b"
COLOR_INNER_BG = "#16191f"
COLOR_BORDER = "#2d333f"
COLOR_GREEN_ACCENT = "#238636"
COLOR_GREEN_HOVER = "#2ea043"
COLOR_RED_ACCENT = "#da3633"
COLOR_RED_HOVER = "#f85149"
COLOR_TEXT_DIM = "#8b949e"
COLOR_TEXT_BRIGHT = "#f0f6fc"


class YT2PDFApp(ctk.CTk):
    """Main application window for YT2PDF."""

    def __init__(self):
        super().__init__()

        self.title("YT2PDF — Smart Presentation Slide Extractor & Lossless PDF Builder")
        self.geometry("1120x860")
        self.minsize(980, 750)

        # Worker state
        self.worker: Optional[ExtractionWorker] = None
        self.msg_queue = queue.Queue()
        self.cancel_event = threading.Event()
        self.is_processing = False
        self.last_generated_pdf = ""

        # Build UI Layout
        self._create_header()
        self._create_main_content()
        self._create_footer()

        # Start background event loop for thread safety
        self.after(50, self._process_queue)

    def _create_header(self):
        """Top banner with application title and subtitle."""
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=24, pady=(16, 8))

        title_label = ctk.CTkLabel(
            header_frame,
            text="YT2PDF",
            font=ctk.CTkFont(family="Segoe UI", size=26, weight="bold"),
            text_color="#58a6ff"
        )
        title_label.pack(side="left", padx=(0, 12))

        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="|   Extract Clean, High-Resolution Slides into Lossless PDFs",
            font=ctk.CTkFont(family="Segoe UI", size=14),
            text_color=COLOR_TEXT_DIM
        )
        subtitle_label.pack(side="left", pady=(4, 0))

        self.header_view_pdf_btn = ctk.CTkButton(
            header_frame,
            text="View PDF",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#1f6feb",
            hover_color="#388bfd",
            height=32,
            width=90,
            command=self._on_view_pdf
        )
        self.header_view_pdf_btn.pack(side="right", padx=(8, 0))

        self.header_dedup_btn = ctk.CTkButton(
            header_frame,
            text="Review & Deduplicate PDF...",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#21262d",
            hover_color="#30363d",
            border_width=1,
            border_color="#30363d",
            height=32,
            command=lambda: self._open_deduplicate_dialog()
        )
        self.header_dedup_btn.pack(side="right")

    def _create_main_content(self):
        """Scrollable or dual-column content layout."""
        # Use two columns: Left for configuration, Right for live progress and preview
        self.container = ctk.CTkFrame(self, fg_color="transparent")
        self.container.pack(fill="both", expand=True, padx=20, pady=8)

        # Left Column (Config panels)
        self.left_col = ctk.CTkScrollableFrame(
            self.container,
            fg_color="transparent",
            width=540,
            scrollbar_button_color="#2d333f",
            scrollbar_button_hover_color="#404859"
        )
        self.left_col.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Right Column (Live metrics, preview, and controls)
        self.right_col = ctk.CTkFrame(
            self.container,
            fg_color=COLOR_CARD_BG,
            corner_radius=12,
            border_width=1,
            border_color=COLOR_BORDER
        )
        self.right_col.pack(side="right", fill="both", expand=True, padx=(10, 0))

        # Build individual cards
        self._build_input_output_card()
        self._build_detection_settings_card()
        self._build_roi_card()
        self._build_right_preview_panel()

    def _build_input_output_card(self):
        """Panel 1: Input source selection and output destination."""
        card = ctk.CTkFrame(self.left_col, fg_color=COLOR_CARD_BG, corner_radius=12, border_width=1, border_color=COLOR_BORDER)
        card.pack(fill="x", pady=(0, 14), padx=2)

        header = ctk.CTkLabel(card, text="1. Input & Output Destination", font=ctk.CTkFont(size=15, weight="bold"))
        header.pack(anchor="w", padx=16, pady=(14, 10))

        # Source type selector
        src_row = ctk.CTkFrame(card, fg_color="transparent")
        src_row.pack(fill="x", padx=16, pady=(0, 8))

        ctk.CTkLabel(src_row, text="Source Type:", width=90, anchor="w", text_color=COLOR_TEXT_DIM).pack(side="left")
        self.source_segmented = ctk.CTkSegmentedButton(
            src_row,
            values=[SourceType.YOUTUBE.value, SourceType.LOCAL_FILE.value],
            command=self._on_source_type_changed
        )
        self.source_segmented.set(SourceType.YOUTUBE.value)
        self.source_segmented.pack(side="left", fill="x", expand=True)

        # Input source row
        self.input_label = ctk.CTkLabel(card, text="YouTube URL:", font=ctk.CTkFont(size=12, weight="bold"))
        self.input_label.pack(anchor="w", padx=16, pady=(6, 2))

        input_entry_row = ctk.CTkFrame(card, fg_color="transparent")
        input_entry_row.pack(fill="x", padx=16, pady=(0, 10))

        self.input_entry = ctk.CTkEntry(
            input_entry_row,
            placeholder_text="https://www.youtube.com/watch?v=...",
            height=36,
            corner_radius=8
        )
        self.input_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self.input_browse_btn = ctk.CTkButton(
            input_entry_row,
            text="Browse...",
            width=75,
            height=36,
            command=self._on_browse_input_file
        )
        # Not packed initially (default mode is YouTube URL)

        self.input_action_btn = ctk.CTkButton(
            input_entry_row,
            text="Paste",
            width=75,
            height=36,
            command=self._on_paste_input
        )
        self.input_action_btn.pack(side="right")

        # Autoformat bindings for input entry
        self.input_entry.bind("<Control-v>", self._on_input_entry_paste)
        self.input_entry.bind("<Control-V>", self._on_input_entry_paste)
        self.input_entry.bind("<<Paste>>", self._on_input_entry_paste)
        self.input_entry.bind("<FocusOut>", self._on_input_entry_blur_or_change)
        self.input_entry.bind("<Return>", self._on_input_entry_blur_or_change)

        # Output PDF destination row
        ctk.CTkLabel(card, text="Output PDF File:", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=16, pady=(4, 2))

        output_entry_row = ctk.CTkFrame(card, fg_color="transparent")
        output_entry_row.pack(fill="x", padx=16, pady=(0, 16))

        self.output_view_btn = ctk.CTkButton(
            output_entry_row,
            text="View PDF",
            width=80,
            height=36,
            fg_color="#1f6feb",
            hover_color="#388bfd",
            command=self._on_view_pdf
        )
        self.output_view_btn.pack(side="right")

        self.output_browse_btn = ctk.CTkButton(
            output_entry_row,
            text="Browse...",
            width=80,
            height=36,
            command=self._on_browse_output
        )
        self.output_browse_btn.pack(side="right", padx=(0, 6))

        self.output_entry = ctk.CTkEntry(
            output_entry_row,
            placeholder_text="Path to save output .pdf (auto-suggested)",
            height=36,
            corner_radius=8
        )
        self.output_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))

        # Autoformat bindings for output entry
        self.output_entry.bind("<Control-v>", self._on_output_entry_paste)
        self.output_entry.bind("<Control-V>", self._on_output_entry_paste)
        self.output_entry.bind("<<Paste>>", self._on_output_entry_paste)
        self.output_entry.bind("<FocusOut>", self._on_output_entry_blur_or_change)
        self.output_entry.bind("<Return>", self._on_output_entry_blur_or_change)

    def _build_detection_settings_card(self):
        """Panel 2: Sampling and Debounce Sliders."""
        card = ctk.CTkFrame(self.left_col, fg_color=COLOR_CARD_BG, corner_radius=12, border_width=1, border_color=COLOR_BORDER)
        card.pack(fill="x", pady=(0, 14), padx=2)

        header = ctk.CTkLabel(card, text="2. Detection & Sampling Configuration", font=ctk.CTkFont(size=15, weight="bold"))
        header.pack(anchor="w", padx=16, pady=(14, 10))

        # 1. Sample Interval
        self.sample_interval_slider, self.interval_val_lbl = self._add_slider_row(
            parent=card,
            title="Sample Interval:",
            subtitle="Frequency of frame sampling (lower is more thorough, higher is faster)",
            initial_val_str="1.0s",
            slider_var=ctk.DoubleVar(value=1.0),
            from_=0.5, to_=5.0, steps=9,
            format_fn=lambda v: f"{v:.1f}s"
        )

        # 2. Similarity Threshold
        self.similarity_slider, self.threshold_val_lbl = self._add_slider_row(
            parent=card,
            title="Similarity Threshold:",
            subtitle="SSIM sensitivity required to trigger a slide transition",
            initial_val_str="92%",
            slider_var=ctk.DoubleVar(value=0.92),
            from_=0.80, to_=0.99, steps=19,
            format_fn=lambda v: f"{int(v * 100)}%"
        )

        # 3. Debounce / Animation Settle Time
        self.debounce_slider, self.debounce_val_lbl = self._add_slider_row(
            parent=card,
            title="Debounce / Settle Time:",
            subtitle="Wait duration for bullet points and animations to settle into completed state",
            initial_val_str="2.0s",
            slider_var=ctk.DoubleVar(value=2.0),
            from_=1.0, to_=10.0, steps=18,
            format_fn=lambda v: f"{v:.1f}s"
        )

        # 4. Min Slide Duration
        self.min_duration_slider, self.min_dur_val_lbl = self._add_slider_row(
            parent=card,
            title="Min Slide Duration:",
            subtitle="Filters out momentary camera cuts back to speaker",
            initial_val_str="2.0s",
            slider_var=ctk.DoubleVar(value=2.0),
            from_=1.0, to_=15.0, steps=28,
            format_fn=lambda v: f"{v:.1f}s"
        )

    def _add_slider_row(self, parent, title, subtitle, initial_val_str, slider_var, from_, to_, steps, format_fn):
        """Helper to create slider with title and dynamic value readout."""
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=(4, 8))

        top_line = ctk.CTkFrame(row, fg_color="transparent")
        top_line.pack(fill="x")

        ctk.CTkLabel(top_line, text=title, font=ctk.CTkFont(weight="bold", size=13)).pack(side="left")

        # Pill badge for current value
        badge = ctk.CTkFrame(top_line, fg_color=COLOR_INNER_BG, corner_radius=6, border_width=1, border_color=COLOR_BORDER)
        badge.pack(side="right")

        val_label = ctk.CTkLabel(
            badge,
            text=initial_val_str,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#58a6ff",
            width=54,
            height=24
        )
        val_label.pack(padx=6, pady=2)

        ctk.CTkLabel(row, text=subtitle, font=ctk.CTkFont(size=11), text_color=COLOR_TEXT_DIM).pack(anchor="w", pady=(2, 4))

        def on_slider_move(val):
            val_label.configure(text=format_fn(val))

        slider = ctk.CTkSlider(
            row,
            from_=from_,
            to=to_,
            number_of_steps=steps,
            variable=slider_var,
            command=on_slider_move,
            height=16
        )
        slider.pack(fill="x", pady=(2, 0))
        return slider, val_label

    def _build_roi_card(self):
        """Panel 3: Region of Interest (ROI) & Preprocessing Toggles."""
        card = ctk.CTkFrame(self.left_col, fg_color=COLOR_CARD_BG, corner_radius=12, border_width=1, border_color=COLOR_BORDER)
        card.pack(fill="x", pady=(0, 14), padx=2)

        header = ctk.CTkLabel(card, text="3. Region of Interest (ROI) & Preprocessing", font=ctk.CTkFont(size=15, weight="bold"))
        header.pack(anchor="w", padx=16, pady=(14, 10))

        # Mask Presenter / PIP Toggle
        pip_toggle_row = ctk.CTkFrame(card, fg_color="transparent")
        pip_toggle_row.pack(fill="x", padx=16, pady=4)

        self.mask_presenter_switch = ctk.CTkSwitch(
            pip_toggle_row,
            text="Mask Presenter / PIP Webcam",
            font=ctk.CTkFont(weight="bold"),
            command=self._on_toggle_presenter_mask
        )
        self.mask_presenter_switch.pack(side="left")

        # PIP Mode Selector
        self.pip_mode_frame = ctk.CTkFrame(card, fg_color=COLOR_INNER_BG, corner_radius=8)
        self.pip_mode_frame.pack(fill="x", padx=16, pady=(6, 10))

        mode_select_row = ctk.CTkFrame(self.pip_mode_frame, fg_color="transparent")
        mode_select_row.pack(fill="x", padx=12, pady=(8, 4))

        ctk.CTkLabel(mode_select_row, text="PIP Position:", font=ctk.CTkFont(size=12)).pack(side="left", padx=(0, 8))
        self.pip_mode_dropdown = ctk.CTkOptionMenu(
            mode_select_row,
            values=[m.value for m in PresenterMaskMode],
            command=self._on_pip_mode_changed,
            width=200
        )
        self.pip_mode_dropdown.set(PresenterMaskMode.AUTO.value)
        self.pip_mode_dropdown.pack(side="left")

        # Custom Normalized Box Entry
        self.custom_box_row = ctk.CTkFrame(self.pip_mode_frame, fg_color="transparent")
        self.custom_box_row.pack(fill="x", padx=12, pady=(4, 8))

        ctk.CTkLabel(self.custom_box_row, text="Custom Box (x,y,w,h):", font=ctk.CTkFont(size=11), text_color=COLOR_TEXT_DIM).pack(side="left", padx=(0, 6))
        self.custom_box_entry = ctk.CTkEntry(self.custom_box_row, height=28, width=170)
        self.custom_box_entry.insert(0, "0.75,0.70,0.25,0.30")
        self.custom_box_entry.pack(side="left")

        # Initially disable PIP options until switch is toggled
        self._on_toggle_presenter_mask()

        # Ignore Player Chrome Toggle
        chrome_row = ctk.CTkFrame(card, fg_color="transparent")
        chrome_row.pack(fill="x", padx=16, pady=(6, 4))
        self.ignore_chrome_switch = ctk.CTkSwitch(
            chrome_row,
            text="Ignore Player Chrome (Mask top 3% & bottom 5% titles/seekbars)",
            font=ctk.CTkFont(weight="bold")
        )
        self.ignore_chrome_switch.select()  # Default: On
        self.ignore_chrome_switch.pack(side="left")

        # Canvas Auto-Crop Toggle
        crop_row = ctk.CTkFrame(card, fg_color="transparent")
        crop_row.pack(fill="x", padx=16, pady=(4, 6))
        self.auto_crop_switch = ctk.CTkSwitch(
            crop_row,
            text="Canvas Auto-Crop (Detect and remove black letterbox/pillarbox borders)",
            font=ctk.CTkFont(weight="bold")
        )
        self.auto_crop_switch.pack(side="left")

        # Global Duplicate Review & Truncation Toggle
        dedup_row = ctk.CTkFrame(card, fg_color="transparent")
        dedup_row.pack(fill="x", padx=16, pady=(4, 6))
        self.prune_duplicates_switch = ctk.CTkSwitch(
            dedup_row,
            text="Global Duplicate Truncation (Prune recurring slides, keep 1st appearance)",
            font=ctk.CTkFont(weight="bold")
        )
        self.prune_duplicates_switch.select()  # Default: On
        self.prune_duplicates_switch.pack(side="left")

        # Filter Full-Human / Narrator Frames Toggle
        narrator_row = ctk.CTkFrame(card, fg_color="transparent")
        narrator_row.pack(fill="x", padx=16, pady=(4, 16))
        self.filter_narrator_switch = ctk.CTkSwitch(
            narrator_row,
            text="Filter Full-Human Narrator Frames (Remove camera cuts to speaker)",
            font=ctk.CTkFont(weight="bold")
        )
        self.filter_narrator_switch.select()  # Default: On
        self.filter_narrator_switch.pack(side="left")

    def _build_right_preview_panel(self):
        """Right column: Run Controls, Live Metrics, Progress, and Thumbnail Preview."""
        right_container = ctk.CTkFrame(self.right_col, fg_color="transparent")
        right_container.pack(fill="both", expand=True, padx=16, pady=16)

        # Header
        ctk.CTkLabel(
            right_container,
            text="Live Extraction & Preview",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", pady=(0, 8))

        # Status Label
        self.status_label = ctk.CTkLabel(
            right_container,
            text="Ready to extract slides",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#58a6ff",
            anchor="w"
        )
        self.status_label.pack(fill="x", pady=(0, 6))

        # Progress Bar
        self.progress_bar = ctk.CTkProgressBar(right_container, height=12, corner_radius=6)
        self.progress_bar.set(0.0)
        self.progress_bar.pack(fill="x", pady=(0, 14))

        # KPI Metrics Cards (2x2 grid layout)
        metrics_frame = ctk.CTkFrame(right_container, fg_color="transparent")
        metrics_frame.pack(fill="x", pady=(0, 14))

        m_row1 = ctk.CTkFrame(metrics_frame, fg_color="transparent")
        m_row1.pack(fill="x", pady=(0, 6))

        self.card_time, self.val_time = self._create_kpi_badge(m_row1, "Timestamp / Total", "00:00 / 00:00")
        self.card_speed, self.val_speed = self._create_kpi_badge(m_row1, "Processing Speed", "0 FPS (0.0x)")
        self.card_time.pack(side="left", fill="both", expand=True, padx=(0, 4))
        self.card_speed.pack(side="right", fill="both", expand=True, padx=(4, 0))

        m_row2 = ctk.CTkFrame(metrics_frame, fg_color="transparent")
        m_row2.pack(fill="x")

        self.card_slides, self.val_slides = self._create_kpi_badge(m_row2, "Slides Captured", "0 slides")
        self.card_eta, self.val_eta = self._create_kpi_badge(m_row2, "Est. Remaining (Finishes at)", "--:--")
        self.card_slides.pack(side="left", fill="both", expand=True, padx=(0, 4))
        self.card_eta.pack(side="right", fill="both", expand=True, padx=(4, 0))

        # Live Slide Thumbnail Card
        ctk.CTkLabel(right_container, text="Latest Unique Slide:", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLOR_TEXT_DIM).pack(anchor="w", pady=(4, 4))
        
        self.preview_frame = ctk.CTkFrame(
            right_container,
            fg_color=COLOR_INNER_BG,
            corner_radius=10,
            border_width=1,
            border_color=COLOR_BORDER
        )
        self.preview_frame.pack(fill="both", expand=True, pady=(0, 14))

        self.preview_label = ctk.CTkLabel(
            self.preview_frame,
            text="No slides captured yet\nPreview will appear here in real time",
            text_color=COLOR_TEXT_DIM,
            font=ctk.CTkFont(size=12)
        )
        self.preview_label.pack(fill="both", expand=True, padx=8, pady=8)

        # Run Controls (Prominent Green Start & Cancel Buttons)
        btn_frame = ctk.CTkFrame(right_container, fg_color="transparent")
        btn_frame.pack(fill="x")

        self.start_btn = ctk.CTkButton(
            btn_frame,
            text="Start Extraction",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=COLOR_GREEN_ACCENT,
            hover_color=COLOR_GREEN_HOVER,
            height=44,
            corner_radius=8,
            command=self._on_start_clicked
        )
        self.start_btn.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self.cancel_btn = ctk.CTkButton(
            btn_frame,
            text="Cancel / Stop",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=COLOR_RED_ACCENT,
            hover_color=COLOR_RED_HOVER,
            width=120,
            height=44,
            corner_radius=8,
            state="disabled",
            command=self._on_cancel_clicked
        )
        self.cancel_btn.pack(side="right")

    def _create_kpi_badge(self, parent, title, initial_val):
        """Creates a stylized KPI badge card."""
        badge = ctk.CTkFrame(parent, fg_color=COLOR_INNER_BG, corner_radius=8, border_width=1, border_color=COLOR_BORDER)
        t_lbl = ctk.CTkLabel(badge, text=title, font=ctk.CTkFont(size=10), text_color=COLOR_TEXT_DIM)
        t_lbl.pack(anchor="w", padx=10, pady=(6, 2))
        v_lbl = ctk.CTkLabel(badge, text=initial_val, font=ctk.CTkFont(size=13, weight="bold"), text_color=COLOR_TEXT_BRIGHT)
        v_lbl.pack(anchor="w", padx=10, pady=(0, 6))
        return badge, v_lbl

    def _create_footer(self):
        """Bottom bar."""
        footer = ctk.CTkFrame(self, fg_color="transparent", height=24)
        footer.pack(fill="x", padx=24, pady=(0, 10))
        ctk.CTkLabel(
            footer,
            text="YT2PDF v1.0 • Lossless img2pdf engine • Motion & SSIM debounced",
            font=ctk.CTkFont(size=11),
            text_color=COLOR_TEXT_DIM
        ).pack(side="left")

    # ---------------- UI Event Handlers ----------------

    def _on_source_type_changed(self, value):
        self._update_source_type_ui(value, clear_input=True)

    def _update_source_type_ui(self, value, clear_input=True):
        if value == SourceType.YOUTUBE.value:
            self.input_label.configure(text="YouTube URL:")
            if clear_input:
                self.input_entry.delete(0, "end")
            self.input_entry.configure(placeholder_text="https://www.youtube.com/watch?v=...")
            if hasattr(self, "input_browse_btn"):
                self.input_browse_btn.pack_forget()
        else:
            self.input_label.configure(text="Local Video File:")
            if clear_input:
                self.input_entry.delete(0, "end")
            self.input_entry.configure(placeholder_text="Select MP4, MKV, MOV, or AVI file...")
            if hasattr(self, "input_browse_btn"):
                self.input_browse_btn.pack(side="right", padx=(6, 0))

    def _set_input_value(self, raw_value: str):
        """Intelligently cleans, auto-formats, and routes input paths and URLs."""
        if not raw_value:
            return

        cleaned = clean_file_path(raw_value)
        if not cleaned:
            return

        if is_local_file_path(cleaned):
            # Auto-switch to Local File mode if currently in YouTube mode
            if self.source_segmented.get() != SourceType.LOCAL_FILE.value:
                self.source_segmented.set(SourceType.LOCAL_FILE.value)
                self._update_source_type_ui(SourceType.LOCAL_FILE.value, clear_input=False)

            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, cleaned)

            # Auto-suggest output PDF path next to video
            base, _ = os.path.splitext(cleaned)
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, f"{base}_slides.pdf")
        else:
            # YouTube URL mode
            if self.source_segmented.get() != SourceType.YOUTUBE.value:
                self.source_segmented.set(SourceType.YOUTUBE.value)
                self._update_source_type_ui(SourceType.YOUTUBE.value, clear_input=False)

            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, cleaned)
            self._auto_propose_youtube_title(cleaned)

    def _on_paste_input(self):
        """Retrieves clipboard content and auto-formats it."""
        try:
            clipboard = self.clipboard_get()
            if clipboard:
                self._set_input_value(clipboard)
        except Exception as e:
            logger.debug(f"Clipboard paste error: {e}")

    def _on_paste_url(self):
        """Backward-compatible alias for _on_paste_input."""
        self._on_paste_input()

    def _on_input_entry_paste(self, event=None):
        try:
            clipboard = self.clipboard_get()
            if clipboard:
                self._set_input_value(clipboard)
                return "break"
        except Exception:
            pass

    def _on_input_entry_blur_or_change(self, event=None):
        curr = self.input_entry.get()
        if curr:
            cleaned = clean_file_path(curr)
            if cleaned != curr:
                self._set_input_value(cleaned)

    def _on_output_entry_paste(self, event=None):
        try:
            clipboard = self.clipboard_get()
            if clipboard:
                cleaned = clean_file_path(clipboard)
                self.output_entry.delete(0, "end")
                self.output_entry.insert(0, cleaned)
                return "break"
        except Exception:
            pass

    def _on_output_entry_blur_or_change(self, event=None):
        curr = self.output_entry.get()
        if curr:
            cleaned = clean_file_path(curr)
            if cleaned != curr:
                self.output_entry.delete(0, "end")
                self.output_entry.insert(0, cleaned)

    def _on_browse_input_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Presentation Video",
            filetypes=[
                ("Video Files", "*.mp4 *.mkv *.mov *.avi *.webm"),
                ("All Files", "*.*")
            ]
        )
        if file_path:
            self._set_input_value(file_path)

    def _auto_propose_youtube_title(self, url: str):
        """Fetches YouTube title in background to suggest destination PDF filename."""
        def fetch():
            try:
                downloader = VideoDownloader()
                meta = downloader.fetch_youtube_metadata(url)
                default_name = meta["default_pdf_name"]
                
                # Check current output entry
                curr = self.output_entry.get().strip()
                if not curr or curr.endswith(".pdf"):
                    desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")
                    target_dir = desktop_dir if os.path.exists(desktop_dir) else os.getcwd()
                    proposed_path = os.path.join(target_dir, default_name)
                    self.output_entry.delete(0, "end")
                    self.output_entry.insert(0, proposed_path)
            except Exception as e:
                logger.debug(f"Could not auto-fetch title: {e}")

        threading.Thread(target=fetch, daemon=True).start()

    def _on_browse_output(self):
        initial_file = "presentation_slides.pdf"
        curr = self.output_entry.get().strip()
        initial_dir = os.getcwd()
        if curr:
            initial_dir = os.path.dirname(curr) or os.getcwd()
            initial_file = os.path.basename(curr) or initial_file

        path = filedialog.asksaveasfilename(
            title="Select Destination PDF",
            defaultextension=".pdf",
            initialdir=initial_dir,
            initialfile=initial_file,
            filetypes=[("PDF Document", "*.pdf")]
        )
        if path:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, path)

    def _open_file(self, file_path: str):
        """Cross-platform system file opener."""
        try:
            if sys.platform == "win32":
                os.startfile(file_path)
            elif sys.platform == "darwin":
                subprocess.run(["open", file_path])
            else:
                subprocess.run(["xdg-open", file_path])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open file:\n{e}")

    def _on_view_pdf(self):
        """Action handler to view the currently configured or selected output PDF."""
        # 1. Check current output entry
        pdf_path = clean_file_path(self.output_entry.get())
        if pdf_path and os.path.isfile(pdf_path):
            self._open_file(pdf_path)
            return

        # 2. Check last generated PDF
        if hasattr(self, "last_generated_pdf") and self.last_generated_pdf:
            last_clean = clean_file_path(self.last_generated_pdf)
            if os.path.isfile(last_clean):
                self._open_file(last_clean)
                return

        # 3. Prompt user to select an existing PDF
        selected = filedialog.askopenfilename(
            title="Select PDF to View",
            filetypes=[("PDF Documents", "*.pdf"), ("All Files", "*.*")]
        )
        if selected:
            selected_clean = clean_file_path(selected)
            if os.path.isfile(selected_clean):
                if not self.output_entry.get().strip():
                    self.output_entry.delete(0, "end")
                    self.output_entry.insert(0, selected_clean)
                self._open_file(selected_clean)

    def _on_toggle_presenter_mask(self):
        enabled = self.mask_presenter_switch.get() == 1
        state = "normal" if enabled else "disabled"
        self.pip_mode_dropdown.configure(state=state)
        self.custom_box_entry.configure(state=state)

    def _on_pip_mode_changed(self, choice):
        if choice == PresenterMaskMode.CUSTOM.value:
            self.custom_box_entry.configure(state="normal")
        else:
            if self.mask_presenter_switch.get() == 1:
                self.custom_box_entry.configure(state="disabled")

    # ---------------- Start & Cancellation ----------------

    def _on_start_clicked(self):
        input_val = clean_file_path(self.input_entry.get())
        if not input_val:
            messagebox.showwarning("Missing Input", "Please provide a YouTube URL or select a local video file.")
            return

        # Ensure UI entry reflects clean formatted path
        if input_val != self.input_entry.get().strip():
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, input_val)

        output_val = clean_file_path(self.output_entry.get())
        if not output_val:
            # Generate default output name
            default_out = os.path.join(os.getcwd(), "presentation_slides.pdf")
            self.output_entry.insert(0, default_out)
            output_val = default_out

        if not output_val.lower().endswith(".pdf"):
            output_val += ".pdf"

        if output_val != self.output_entry.get().strip():
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, output_val)

        # Parse custom ROI box if selected
        custom_box = (0.75, 0.70, 0.25, 0.30)
        pip_mode = PresenterMaskMode(self.pip_mode_dropdown.get())
        if pip_mode == PresenterMaskMode.CUSTOM:
            try:
                parts = [float(p.strip()) for p in self.custom_box_entry.get().split(",")]
                if len(parts) == 4 and all(0.0 <= p <= 1.0 for p in parts):
                    custom_box = tuple(parts)
                else:
                    raise ValueError
            except Exception:
                messagebox.showerror("Invalid ROI Box", "Custom box must be four normalized numbers: x,y,w,h (between 0.0 and 1.0)")
                return

        # Prepare configuration
        source_type = SourceType.YOUTUBE if self.source_segmented.get() == SourceType.YOUTUBE.value else SourceType.LOCAL_FILE
        config = ExtractionConfig(
            source_type=source_type,
            input_source=input_val,
            output_path=output_val,
            sample_interval=float(self.sample_interval_slider.get()),
            similarity_threshold=float(self.similarity_slider.get()),
            debounce_settle_time=float(self.debounce_slider.get()),
            min_slide_duration=float(self.min_duration_slider.get()),
            mask_presenter=(self.mask_presenter_switch.get() == 1),
            presenter_mode=pip_mode,
            custom_roi_box=custom_box,
            ignore_chrome=(self.ignore_chrome_switch.get() == 1),
            canvas_auto_crop=(self.auto_crop_switch.get() == 1),
            prune_global_duplicates=(self.prune_duplicates_switch.get() == 1),
            filter_narrator_frames=(self.filter_narrator_switch.get() == 1)
        )

        # Update UI state to processing
        self._set_processing_ui_state(True)
        self.cancel_event.clear()
        self.status_label.configure(text="Starting extraction pipeline...", text_color="#58a6ff")
        self.progress_bar.set(0.0)
        self.val_slides.configure(text="0 slides")
        self.val_time.configure(text="00:00 / 00:00")
        self.val_speed.configure(text="0 FPS (0.0x)")
        self.val_eta.configure(text="Calculating...")

        # Launch worker thread
        self.worker = ExtractionWorker(config, self.msg_queue, self.cancel_event)
        self.worker.start()

    def _on_cancel_clicked(self):
        if self.worker and self.worker.is_alive():
            confirm = messagebox.askyesno("Confirm Stop", "Do you want to stop the extraction in progress?")
            if confirm:
                self.cancel_event.set()
                self.status_label.configure(text="Cancelling extraction...", text_color=COLOR_RED_HOVER)
                self.cancel_btn.configure(state="disabled")

    def _set_processing_ui_state(self, processing: bool):
        self.is_processing = processing
        state = "disabled" if processing else "normal"
        self.start_btn.configure(state=state)
        self.source_segmented.configure(state=state)
        self.input_entry.configure(state=state)
        self.input_action_btn.configure(state=state)
        if hasattr(self, "input_browse_btn"):
            self.input_browse_btn.configure(state=state)
        self.output_entry.configure(state=state)
        self.output_browse_btn.configure(state=state)
        self.output_view_btn.configure(state=state)
        self.header_view_pdf_btn.configure(state=state)
        self.header_dedup_btn.configure(state=state)
        self.cancel_btn.configure(state="normal" if processing else "disabled")

    # ---------------- Message Queue Polling ----------------

    def _process_queue(self):
        """Polls messages sent from the worker thread."""
        try:
            while not self.msg_queue.empty():
                msg: WorkerMessage = self.msg_queue.get_nowait()
                self._handle_worker_message(msg)
        except Exception as e:
            logger.error(f"Error handling worker queue message: {e}")
        finally:
            self.after(50, self._process_queue)

    def _handle_worker_message(self, msg: WorkerMessage):
        if msg.msg_type == WorkerMessageType.STATUS:
            self.status_label.configure(text=str(msg.payload), text_color="#58a6ff")

        elif msg.msg_type == WorkerMessageType.PROGRESS:
            val = max(0.0, min(1.0, float(msg.payload)))
            self.progress_bar.set(val)

        elif msg.msg_type == WorkerMessageType.METRICS:
            data = msg.payload
            self.val_time.configure(text=data.get("time_str", "00:00 / 00:00"))
            self.val_speed.configure(text=data.get("speed_str", "0 FPS"))
            self.val_slides.configure(text=f"{data.get('slides_count', 0)} slides")
            self.val_eta.configure(text=data.get("eta_str", "--:--"))

        elif msg.msg_type == WorkerMessageType.THUMBNAIL:
            self._update_thumbnail_preview(msg.payload)

        elif msg.msg_type == WorkerMessageType.COMPLETE:
            self._handle_completion(msg.payload)

        elif msg.msg_type == WorkerMessageType.ERROR:
            self._handle_error(str(msg.payload))

        elif msg.msg_type == WorkerMessageType.CANCELLED:
            self._handle_cancellation()

    def _update_thumbnail_preview(self, data: dict):
        """Displays the latest accepted slide in the preview card with aspect ratio preservation."""
        try:
            rgb_arr = data["image_rgb"]
            pil_img = Image.fromarray(rgb_arr)
            
            # Target size: fits neatly inside the card
            card_w = max(260, self.preview_frame.winfo_width() - 20)
            card_h = max(180, self.preview_frame.winfo_height() - 20)
            pil_img.thumbnail((card_w, card_h), Image.Resampling.LANCZOS)

            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
            self.preview_label.configure(image=ctk_img, text="")
            self.preview_label.image = ctk_img
        except Exception as e:
            logger.debug(f"Preview update failed: {e}")

    def _handle_completion(self, payload: dict):
        self._set_processing_ui_state(False)
        out_path = payload.get("output_path", "")
        slides_count = payload.get("slides_count", 0)
        elapsed = payload.get("elapsed_seconds", 0)
        self.last_generated_pdf = out_path
        
        self.val_eta.configure(text="Finished")
        self.status_label.configure(
            text=f"Complete — {slides_count} unique slides compiled into PDF!",
            text_color="#3fb950"
        )
        self.progress_bar.set(1.0)

        # Completion Dialog with Open PDF, Review Duplicates, Open Folder, and Close actions
        self._show_completion_dialog(out_path, slides_count, elapsed)

    def _show_completion_dialog(self, pdf_path: str, slides_count: int, elapsed: float):
        """Custom modal dialog celebrating completion with direct action buttons in a 2x2 matrix."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Extraction Complete")
        dialog_w = 520
        dialog_h = 290
        dialog.geometry(f"{dialog_w}x{dialog_h}")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()

        # Center on parent
        x = self.winfo_x() + (self.winfo_width() // 2) - (dialog_w // 2)
        y = self.winfo_y() + (self.winfo_height() // 2) - (dialog_h // 2)
        dialog.geometry(f"+{x}+{y}")

        ctk.CTkLabel(
            dialog,
            text="Slides Extracted Successfully!",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#3fb950"
        ).pack(pady=(20, 6))

        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        time_text = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"

        info_text = f"Compiled {slides_count} unique presentation slides in {time_text}.\nSaved to:\n{pdf_path}"
        ctk.CTkLabel(
            dialog,
            text=info_text,
            font=ctk.CTkFont(size=12),
            text_color=COLOR_TEXT_DIM,
            wraplength=480,
            justify="center"
        ).pack(pady=(0, 16), padx=20)

        matrix_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        matrix_frame.pack(fill="x", padx=24, pady=(0, 16))

        def open_pdf():
            self._open_file(pdf_path)
            dialog.destroy()

        def review_duplicates():
            dialog.destroy()
            self._open_deduplicate_dialog(pdf_path)

        def open_folder():
            folder = os.path.dirname(os.path.abspath(pdf_path))
            try:
                if sys.platform == "win32":
                    subprocess.run(["explorer", folder])
                elif sys.platform == "darwin":
                    subprocess.run(["open", folder])
                else:
                    subprocess.run(["xdg-open", folder])
            except Exception as e:
                messagebox.showerror("Error", f"Could not open folder: {e}")
            dialog.destroy()

        # Row 1: [Open PDF] [Review Duplicates]
        row1 = ctk.CTkFrame(matrix_frame, fg_color="transparent")
        row1.pack(fill="x", pady=(0, 8))

        btn_open = ctk.CTkButton(
            row1,
            text="Open PDF",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=COLOR_GREEN_ACCENT,
            hover_color=COLOR_GREEN_HOVER,
            height=38,
            command=open_pdf
        )
        btn_open.pack(side="left", fill="x", expand=True, padx=(0, 4))

        btn_review = ctk.CTkButton(
            row1,
            text="Review Duplicates",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#1f6feb",
            hover_color="#388bfd",
            height=38,
            command=review_duplicates
        )
        btn_review.pack(side="left", fill="x", expand=True, padx=(4, 0))

        # Row 2: [Open Folder] [Close]
        row2 = ctk.CTkFrame(matrix_frame, fg_color="transparent")
        row2.pack(fill="x")

        btn_folder = ctk.CTkButton(
            row2,
            text="Open Folder",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#30363d",
            hover_color="#404859",
            height=38,
            command=open_folder
        )
        btn_folder.pack(side="left", fill="x", expand=True, padx=(0, 4))

        btn_close = ctk.CTkButton(
            row2,
            text="Close",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=COLOR_RED_ACCENT,
            hover_color=COLOR_RED_HOVER,
            height=38,
            command=dialog.destroy
        )
        btn_close.pack(side="left", fill="x", expand=True, padx=(4, 0))

    def _open_deduplicate_dialog(self, default_pdf: str = ""):
        """Opens modal to review and deduplicate any existing PDF."""
        if not default_pdf:
            curr_out = clean_file_path(self.output_entry.get())
            if curr_out and os.path.exists(curr_out):
                default_pdf = curr_out
        DeduplicatePDFDialog(self, default_pdf_path=default_pdf)

    def _handle_error(self, error_message: str):
        self._set_processing_ui_state(False)
        self.progress_bar.set(0.0)
        self.val_eta.configure(text="--:--")
        self.status_label.configure(text="Error occurred during extraction.", text_color=COLOR_RED_HOVER)
        messagebox.showerror("Extraction Error", f"An error occurred during extraction:\n\n{error_message}")

    def _handle_cancellation(self):
        self._set_processing_ui_state(False)
        self.progress_bar.set(0.0)
        self.val_eta.configure(text="--:--")
        self.status_label.configure(text="Extraction cancelled by user.", text_color=COLOR_TEXT_DIM)
        messagebox.showinfo("Cancelled", "Video extraction was stopped.")


class DeduplicatePDFDialog(ctk.CTkToplevel):
    """Interactive modal window to review an entire PDF and truncate repetitive pages."""

    def __init__(self, parent, default_pdf_path: str = ""):
        super().__init__(parent)

        self.title("Review & Deduplicate PDF")
        self.geometry("680x640")
        self.minsize(580, 520)
        self.transient(parent)
        self.grab_set()

        self.cancel_event = threading.Event()
        self.is_processing = False
        self.latest_result: Optional[DeduplicationResult] = None

        self._build_ui(default_pdf_path)

        # Center on parent window
        try:
            x = parent.winfo_x() + (parent.winfo_width() // 2) - 340
            y = parent.winfo_y() + (parent.winfo_height() // 2) - 320
            self.geometry(f"+{max(0, x)}+{max(0, y)}")
        except Exception:
            pass

    def _build_ui(self, default_pdf_path: str):
        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=20, pady=16)

        # Header
        header_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(
            header_frame,
            text="Review & Truncate PDF Duplicates",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color="#58a6ff"
        ).pack(anchor="w")

        ctk.CTkLabel(
            header_frame,
            text="Scans all pages, detects repetitive slides, and keeps only the first appearance alone.",
            font=ctk.CTkFont(size=12),
            text_color=COLOR_TEXT_DIM
        ).pack(anchor="w", pady=(2, 0))

        # Card 1: File Selection
        card_files = ctk.CTkFrame(scroll, fg_color=COLOR_CARD_BG, corner_radius=10, border_width=1, border_color=COLOR_BORDER)
        card_files.pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(card_files, text="1. Select Document", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=16, pady=(12, 6))

        # Input PDF Row
        ctk.CTkLabel(card_files, text="Input PDF:", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=16, pady=(2, 2))
        in_row = ctk.CTkFrame(card_files, fg_color="transparent")
        in_row.pack(fill="x", padx=16, pady=(0, 8))

        self.input_entry = ctk.CTkEntry(in_row, placeholder_text="Path to PDF file...", height=34, corner_radius=6)
        self.input_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        if default_pdf_path:
            self.input_entry.insert(0, clean_file_path(default_pdf_path))

        ctk.CTkButton(in_row, text="Browse...", width=75, height=34, command=self._on_browse_input).pack(side="right", padx=(6, 0))
        ctk.CTkButton(in_row, text="Paste", width=70, height=34, command=self._on_paste_input).pack(side="right")

        self.input_entry.bind("<Control-v>", self._on_input_entry_paste)
        self.input_entry.bind("<Control-V>", self._on_input_entry_paste)
        self.input_entry.bind("<<Paste>>", self._on_input_entry_paste)
        self.input_entry.bind("<FocusOut>", self._on_input_entry_blur_or_change)
        self.input_entry.bind("<Return>", self._on_input_entry_blur_or_change)

        # Output PDF Row
        ctk.CTkLabel(card_files, text="Save Cleaned PDF to:", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=16, pady=(2, 2))
        out_row = ctk.CTkFrame(card_files, fg_color="transparent")
        out_row.pack(fill="x", padx=16, pady=(0, 14))

        self.output_entry = ctk.CTkEntry(out_row, placeholder_text="Path to save deduplicated PDF...", height=34, corner_radius=6)
        self.output_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        if default_pdf_path:
            base, ext = os.path.splitext(clean_file_path(default_pdf_path))
            self.output_entry.insert(0, f"{base}_dedup{ext}")

        ctk.CTkButton(out_row, text="Browse...", width=75, height=34, command=self._on_browse_output).pack(side="right", padx=(6, 0))
        ctk.CTkButton(out_row, text="Paste", width=70, height=34, command=self._on_paste_output).pack(side="right")

        self.output_entry.bind("<Control-v>", self._on_output_entry_paste)
        self.output_entry.bind("<Control-V>", self._on_output_entry_paste)
        self.output_entry.bind("<<Paste>>", self._on_output_entry_paste)
        self.output_entry.bind("<FocusOut>", self._on_output_entry_blur_or_change)
        self.output_entry.bind("<Return>", self._on_output_entry_blur_or_change)

        # Card 2: Sensitivity & Controls
        card_controls = ctk.CTkFrame(scroll, fg_color=COLOR_CARD_BG, corner_radius=10, border_width=1, border_color=COLOR_BORDER)
        card_controls.pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(card_controls, text="2. Detection Sensitivity", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=16, pady=(12, 6))

        # Similarity Threshold Slider
        thresh_top = ctk.CTkFrame(card_controls, fg_color="transparent")
        thresh_top.pack(fill="x", padx=16)

        ctk.CTkLabel(thresh_top, text="Similarity Threshold:", font=ctk.CTkFont(weight="bold", size=13)).pack(side="left")
        badge = ctk.CTkFrame(thresh_top, fg_color=COLOR_INNER_BG, corner_radius=6, border_width=1, border_color=COLOR_BORDER)
        badge.pack(side="right")

        self.threshold_label = ctk.CTkLabel(badge, text="94%", font=ctk.CTkFont(size=12, weight="bold"), text_color="#58a6ff", width=54, height=24)
        self.threshold_label.pack(padx=6, pady=2)

        ctk.CTkLabel(
            card_controls,
            text="Sensitivity for declaring pages identical (higher requires stricter visual match)",
            font=ctk.CTkFont(size=11),
            text_color=COLOR_TEXT_DIM
        ).pack(anchor="w", padx=16, pady=(2, 4))

        self.similarity_slider = ctk.CTkSlider(
            card_controls,
            from_=0.80,
            to=0.99,
            number_of_steps=19,
            command=lambda v: self.threshold_label.configure(text=f"{int(v * 100)}%"),
            height=16
        )
        self.similarity_slider.set(0.94)
        self.similarity_slider.pack(fill="x", padx=16, pady=(2, 10))

        # Narrator Filter Switch
        narr_row = ctk.CTkFrame(card_controls, fg_color="transparent")
        narr_row.pack(fill="x", padx=16, pady=(0, 12))
        self.dialog_filter_narrator_switch = ctk.CTkSwitch(
            narr_row,
            text="Filter Full-Human / Narrator Frames (Remove camera cuts to speaker)",
            font=ctk.CTkFont(weight="bold")
        )
        self.dialog_filter_narrator_switch.select()
        self.dialog_filter_narrator_switch.pack(side="left")

        # Action Buttons
        btn_action_row = ctk.CTkFrame(card_controls, fg_color="transparent")
        btn_action_row.pack(fill="x", padx=16, pady=(0, 14))

        self.run_btn = ctk.CTkButton(
            btn_action_row,
            text="Review & Truncate Duplicates",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=COLOR_GREEN_ACCENT,
            hover_color=COLOR_GREEN_HOVER,
            height=38,
            corner_radius=6,
            command=self._on_start_dedup
        )
        self.run_btn.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self.cancel_btn = ctk.CTkButton(
            btn_action_row,
            text="Cancel",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=COLOR_RED_ACCENT,
            hover_color=COLOR_RED_HOVER,
            width=90,
            height=38,
            corner_radius=6,
            state="disabled",
            command=self._on_cancel_dedup
        )
        self.cancel_btn.pack(side="right")

        # Card 3: Live Progress & Results
        self.card_results = ctk.CTkFrame(scroll, fg_color=COLOR_CARD_BG, corner_radius=10, border_width=1, border_color=COLOR_BORDER)
        self.card_results.pack(fill="x", pady=(0, 8))

        ctk.CTkLabel(self.card_results, text="3. Analysis Progress & Results", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=16, pady=(12, 6))

        # Status & Progress bar
        self.status_label = ctk.CTkLabel(
            self.card_results,
            text="Ready to review PDF",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#58a6ff",
            anchor="w"
        )
        self.status_label.pack(fill="x", padx=16, pady=(0, 6))

        self.progress_bar = ctk.CTkProgressBar(self.card_results, height=10, corner_radius=5)
        self.progress_bar.set(0.0)
        self.progress_bar.pack(fill="x", padx=16, pady=(0, 12))

        # Summary Metrics
        metrics_row = ctk.CTkFrame(self.card_results, fg_color="transparent")
        metrics_row.pack(fill="x", padx=16, pady=(0, 10))

        self.b_total, self.lbl_total = self._create_mini_badge(metrics_row, "Total Pages", "0")
        self.b_kept, self.lbl_kept = self._create_mini_badge(metrics_row, "Unique Kept", "0")
        self.b_pruned, self.lbl_pruned = self._create_mini_badge(metrics_row, "Duplicates Pruned", "0")
        self.b_narrator, self.lbl_narrator = self._create_mini_badge(metrics_row, "Narrator Cuts Pruned", "0")

        self.b_total.pack(side="left", fill="both", expand=True, padx=(0, 4))
        self.b_kept.pack(side="left", fill="both", expand=True, padx=4)
        self.b_pruned.pack(side="left", fill="both", expand=True, padx=4)
        self.b_narrator.pack(side="left", fill="both", expand=True, padx=(4, 0))

        # Duplicate Details Log
        ctk.CTkLabel(self.card_results, text="Repetitive Pages Found:", font=ctk.CTkFont(size=11, weight="bold"), text_color=COLOR_TEXT_DIM).pack(anchor="w", padx=16, pady=(4, 2))

        self.details_textbox = ctk.CTkTextbox(self.card_results, height=120, corner_radius=6, fg_color=COLOR_INNER_BG, font=ctk.CTkFont(family="Consolas", size=11))
        self.details_textbox.pack(fill="x", padx=16, pady=(0, 14))
        self.details_textbox.insert("end", "Click 'Review & Truncate Duplicates' to scan the PDF...")
        self.details_textbox.configure(state="disabled")

        # Result Action Buttons
        self.res_btn_row = ctk.CTkFrame(self.card_results, fg_color="transparent")
        self.res_btn_row.pack(fill="x", padx=16, pady=(0, 14))

        self.open_pdf_btn = ctk.CTkButton(
            self.res_btn_row,
            text="Open Cleaned PDF",
            fg_color=COLOR_GREEN_ACCENT,
            hover_color=COLOR_GREEN_HOVER,
            state="disabled",
            command=self._on_open_pdf
        )
        self.open_pdf_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.open_folder_btn = ctk.CTkButton(
            self.res_btn_row,
            text="Open Folder",
            fg_color="#30363d",
            hover_color="#404859",
            state="disabled",
            command=self._on_open_folder
        )
        self.open_folder_btn.pack(side="left", fill="x", expand=True, padx=6)

        ctk.CTkButton(
            self.res_btn_row,
            text="Close",
            fg_color="#21262d",
            hover_color="#30363d",
            width=80,
            command=self.destroy
        ).pack(side="right", padx=(6, 0))

    def _create_mini_badge(self, parent, title, val):
        badge = ctk.CTkFrame(parent, fg_color=COLOR_INNER_BG, corner_radius=6, border_width=1, border_color=COLOR_BORDER)
        t_lbl = ctk.CTkLabel(badge, text=title, font=ctk.CTkFont(size=10), text_color=COLOR_TEXT_DIM)
        t_lbl.pack(anchor="w", padx=8, pady=(4, 1))
        v_lbl = ctk.CTkLabel(badge, text=val, font=ctk.CTkFont(size=13, weight="bold"), text_color=COLOR_TEXT_BRIGHT)
        v_lbl.pack(anchor="w", padx=8, pady=(0, 4))
        return badge, v_lbl

    def _set_input_value(self, raw_path: str):
        cleaned = clean_file_path(raw_path)
        if not cleaned:
            return
        self.input_entry.delete(0, "end")
        self.input_entry.insert(0, cleaned)
        base, ext = os.path.splitext(cleaned)
        if ext.lower() == ".pdf":
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, f"{base}_dedup{ext}")

    def _on_paste_input(self):
        try:
            clipboard = self.clipboard_get()
            if clipboard:
                self._set_input_value(clipboard)
        except Exception:
            pass

    def _on_paste_output(self):
        try:
            clipboard = self.clipboard_get()
            if clipboard:
                cleaned = clean_file_path(clipboard)
                self.output_entry.delete(0, "end")
                self.output_entry.insert(0, cleaned)
        except Exception:
            pass

    def _on_input_entry_paste(self, event=None):
        try:
            clipboard = self.clipboard_get()
            if clipboard:
                self._set_input_value(clipboard)
                return "break"
        except Exception:
            pass

    def _on_input_entry_blur_or_change(self, event=None):
        curr = self.input_entry.get()
        if curr:
            cleaned = clean_file_path(curr)
            if cleaned != curr:
                self._set_input_value(cleaned)

    def _on_output_entry_paste(self, event=None):
        try:
            clipboard = self.clipboard_get()
            if clipboard:
                cleaned = clean_file_path(clipboard)
                self.output_entry.delete(0, "end")
                self.output_entry.insert(0, cleaned)
                return "break"
        except Exception:
            pass

    def _on_output_entry_blur_or_change(self, event=None):
        curr = self.output_entry.get()
        if curr:
            cleaned = clean_file_path(curr)
            if cleaned != curr:
                self.output_entry.delete(0, "end")
                self.output_entry.insert(0, cleaned)

    def _on_browse_input(self):
        path = filedialog.askopenfilename(
            title="Select PDF Document to Deduplicate",
            filetypes=[("PDF Document", "*.pdf"), ("All Files", "*.*")]
        )
        if path:
            self._set_input_value(path)

    def _on_browse_output(self):
        curr = clean_file_path(self.output_entry.get())
        initial_dir = os.path.dirname(curr) if curr else os.getcwd()
        initial_file = os.path.basename(curr) if curr else "presentation_dedup.pdf"
        path = filedialog.asksaveasfilename(
            title="Save Deduplicated PDF As",
            defaultextension=".pdf",
            initialdir=initial_dir,
            initialfile=initial_file,
            filetypes=[("PDF Document", "*.pdf")]
        )
        if path:
            cleaned = clean_file_path(path)
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, cleaned)

    def _on_start_dedup(self):
        in_path = clean_file_path(self.input_entry.get())
        if not in_path or not os.path.exists(in_path):
            messagebox.showwarning("Missing File", "Please select an existing PDF file.")
            return

        if in_path != self.input_entry.get().strip():
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, in_path)

        out_path = clean_file_path(self.output_entry.get())
        if not out_path:
            base, ext = os.path.splitext(in_path)
            out_path = f"{base}_dedup{ext}"
            self.output_entry.insert(0, out_path)

        if not out_path.lower().endswith(".pdf"):
            out_path += ".pdf"

        if out_path != self.output_entry.get().strip():
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, out_path)

        # UI state
        self.is_processing = True
        self.cancel_event.clear()
        self.run_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.open_pdf_btn.configure(state="disabled")
        self.open_folder_btn.configure(state="disabled")

        self.status_label.configure(text="Reading document pages...", text_color="#58a6ff")
        self.progress_bar.set(0.0)
        self.lbl_total.configure(text="0")
        self.lbl_kept.configure(text="0")
        self.lbl_pruned.configure(text="0")
        self.lbl_narrator.configure(text="0")

        self.details_textbox.configure(state="normal")
        self.details_textbox.delete("1.0", "end")
        self.details_textbox.insert("end", "Analyzing pages for visual duplicates and narrator camera cuts...\n")
        self.details_textbox.configure(state="disabled")

        thresh = float(self.similarity_slider.get())
        filter_narrator = (self.dialog_filter_narrator_switch.get() == 1)

        def run_worker():
            try:
                def prog(curr, total, txt):
                    if not self.cancel_event.is_set():
                        pct = curr / float(max(1, total))
                        self.after(0, lambda: self._update_worker_progress(pct, curr, total, txt))

                result = deduplicate_pdf(
                    input_pdf_path=in_path,
                    output_pdf_path=out_path,
                    similarity_threshold=thresh,
                    filter_narrator_frames=filter_narrator,
                    progress_callback=prog,
                    cancel_event=self.cancel_event
                )
                self.after(0, lambda: self._on_worker_completed(result))
            except Exception as e:
                self.after(0, lambda: self._on_worker_error(str(e)))

        threading.Thread(target=run_worker, daemon=True).start()

    def _update_worker_progress(self, pct: float, curr: int, total: int, txt: str):
        self.progress_bar.set(pct)
        self.status_label.configure(text=f"Analyzing page {curr} of {total}...")
        self.lbl_total.configure(text=str(total))

    def _on_cancel_dedup(self):
        self.cancel_event.set()
        self.status_label.configure(text="Cancelling duplicate review...", text_color=COLOR_RED_HOVER)
        self.cancel_btn.configure(state="disabled")

    def _on_worker_completed(self, result: DeduplicationResult):
        self.latest_result = result
        self.is_processing = False
        self.run_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        self.progress_bar.set(1.0)

        # Update metrics
        self.lbl_total.configure(text=str(result.total_pages))
        self.lbl_kept.configure(text=str(result.unique_pages_count))
        self.lbl_pruned.configure(text=str(result.duplicates_removed_count))
        self.lbl_narrator.configure(text=str(result.narrator_pages_removed_count))

        # Populate details text box
        self.details_textbox.configure(state="normal")
        self.details_textbox.delete("1.0", "end")

        total_pruned = result.duplicates_removed_count + result.narrator_pages_removed_count

        if total_pruned > 0:
            self.status_label.configure(
                text=f"Done! Pruned {total_pruned} pages ({result.unique_pages_count} unique slides kept).",
                text_color="#3fb950"
            )
            self.details_textbox.insert(
                "end",
                f"SUMMARY: {result.unique_pages_count} unique slides retained out of {result.total_pages} total pages.\n"
                f"• {result.narrator_pages_removed_count} full-human narrator camera cuts removed.\n"
                f"• {result.duplicates_removed_count} repetitive duplicate slides truncated.\n\n"
            )
            if result.narrator_details:
                self.details_textbox.insert("end", "-- Narrator Camera Cuts Filtered --\n")
                for narr_p, reason in result.narrator_details:
                    self.details_textbox.insert("end", f"• Page {narr_p:>2}: Full-Human Narrator ({reason}) -> Truncated\n")
                self.details_textbox.insert("end", "\n")

            if result.duplicate_details:
                self.details_textbox.insert("end", "-- Duplicate Slides Truncated --\n")
                for dup_p, orig_p, score in result.duplicate_details:
                    self.details_textbox.insert("end", f"• Page {dup_p:>2}: Matches Page {orig_p:>2} ({score * 100:.1f}% match) -> Truncated\n")

            self.open_pdf_btn.configure(state="normal")
            self.open_folder_btn.configure(state="normal")
        else:
            self.status_label.configure(
                text="Document clean! No duplicate or narrator pages were found.",
                text_color="#58a6ff"
            )
            self.details_textbox.insert(
                "end",
                "All pages in this PDF are unique presentation slides.\n"
                "No repetitive pages or narrator camera cuts were detected."
            )
            self.open_pdf_btn.configure(state="normal")
            self.open_folder_btn.configure(state="normal")

        self.details_textbox.configure(state="disabled")

    def _on_worker_error(self, err_msg: str):
        self.is_processing = False
        self.run_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        self.progress_bar.set(0.0)
        self.status_label.configure(text="Error occurred during review.", text_color=COLOR_RED_HOVER)
        messagebox.showerror("Deduplication Error", f"An error occurred while reviewing the PDF:\n\n{err_msg}")

    def _on_open_pdf(self):
        if not self.latest_result or not os.path.exists(self.latest_result.output_path):
            return
        path = self.latest_result.output_path
        try:
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.run(["open", path])
            else:
                subprocess.run(["xdg-open", path])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open PDF: {e}")

    def _on_open_folder(self):
        if not self.latest_result or not os.path.exists(self.latest_result.output_path):
            return
        folder = os.path.dirname(os.path.abspath(self.latest_result.output_path))
        try:
            if sys.platform == "win32":
                subprocess.run(["explorer", folder])
            elif sys.platform == "darwin":
                subprocess.run(["open", folder])
            else:
                subprocess.run(["xdg-open", folder])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open folder: {e}")
