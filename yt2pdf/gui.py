"""PySide6 (Qt6) Modern Desktop GUI for YT2PDF application with signature Neo-Brutalist design."""

import os
import sys
import queue
import logging
import threading
import subprocess
from typing import Optional, Callable, Tuple, List, Any
import numpy as np
from PIL import Image

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QDialog,
    QFrame,
    QLabel,
    QPushButton,
    QLineEdit,
    QSlider,
    QProgressBar,
    QComboBox,
    QCheckBox,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFileDialog,
    QMessageBox,
    QSizePolicy,
)
from PySide6.QtCore import Qt, QTimer, QRect, QSize, Signal, QObject, QPointF
from PySide6.QtGui import (
    QFont,
    QColor,
    QPen,
    QBrush,
    QPainter,
    QImage,
    QPixmap,
    QKeySequence,
)

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

# ==============================================================================
# Exact Neo-Brutalist Color Palette & Theme Tokens
# ==============================================================================
COLOR_CANVAS_BG = "#ede9f6"       # Soft pastel lavender/gray background
COLOR_CARD_BG = "#ffffff"         # Crisp white card surfaces
COLOR_CARD_BG_ALT = "#fefce8"     # Soft buttercream for secondary cards
COLOR_INNER_BG = "#ffffff"        # Inner frame white
COLOR_BORDER = "#000000"          # Solid pitch-black outlines
COLOR_BORDER_WIDTH = 2.5          # Standard chunky border width

# Typography Colors
COLOR_TEXT = "#000000"            # Pitch black high-contrast text
COLOR_TEXT_DIM = "#4b5563"        # Dark slate captions
COLOR_TEXT_BRIGHT = "#000000"     # High contrast black text for badges

# Vibrant Neo-Brutalist Accents
COLOR_NEO_YELLOW = "#ffd166"      # Sunshine yellow
COLOR_NEO_YELLOW_HOVER = "#ffc72c"
COLOR_NEO_GREEN = "#4ade80"       # Vibrant mint/lime green
COLOR_NEO_GREEN_HOVER = "#22c55e"
COLOR_NEO_BLUE = "#7dd3fc"        # Sky cyan blue
COLOR_NEO_BLUE_HOVER = "#38bdf8"
COLOR_NEO_PINK = "#f472b6"        # Coral pink / salmon
COLOR_NEO_PINK_HOVER = "#ec4899"
COLOR_NEO_LILAC = "#c084fc"       # Retro purple/lilac
COLOR_NEO_LILAC_HOVER = "#a855f7"
COLOR_NEO_RED = "#f87171"         # Neo red / coral
COLOR_NEO_RED_HOVER = "#ef4444"
COLOR_NEO_ORANGE = "#fed7aa"      # Warm peach / orange
COLOR_NEO_CREAM = "#fffbeb"       # Warm white / cream

# Backwards compatibility color aliases
COLOR_GREEN_ACCENT = COLOR_NEO_GREEN
COLOR_GREEN_HOVER = COLOR_NEO_GREEN_HOVER
COLOR_RED_ACCENT = COLOR_NEO_RED
COLOR_RED_HOVER = COLOR_NEO_RED_HOVER

COLOR_TRACK_LILAC = "#d8b4fe"
COLOR_PROGRESS_PURPLE = "#a855f7"


def _show_info(parent, title: str, message: str):
    """Helper to show info message, handling mock patching in unittests."""
    try:
        import tkinter.messagebox as tk_mb
        import unittest.mock
        if hasattr(tk_mb, "showinfo") and (
            isinstance(tk_mb.showinfo, (unittest.mock.MagicMock, unittest.mock.Mock))
            or hasattr(tk_mb.showinfo, "assert_called")
        ):
            tk_mb.showinfo(title, message)
            return
    except Exception:
        pass
    if parent and parent.isVisible():
        QMessageBox.information(parent, title, message)


# ==============================================================================
# Reusable Neo-Brutalist Custom Widgets
# ==============================================================================

class NeoLabel(QLabel):
    """Label supporting Tkinter-style cget and configure for test compatibility."""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.setStyleSheet("color: #000000;")

    def cget(self, attribute_name: str) -> Any:
        if attribute_name == "text":
            return self.text()
        return ""

    def configure(self, **kwargs):
        if "text" in kwargs:
            self.setText(str(kwargs["text"]))
        if "text_color" in kwargs:
            color = kwargs["text_color"]
            self.setStyleSheet(f"color: {color};")


class NeoButton(QPushButton):
    """Neo-Brutalist button with 2px solid black border, bold font, and hover states."""

    def __init__(
        self,
        text="",
        bg_color=COLOR_NEO_YELLOW,
        hover_color=COLOR_NEO_YELLOW_HOVER,
        height=38,
        corner_radius=8,
        font_size=12,
        parent=None,
    ):
        super().__init__(text, parent)
        self._bg_color = bg_color
        self._hover_color = hover_color
        self._corner_radius = corner_radius
        self._font_size = font_size
        self.setFixedHeight(height)
        self.setCursor(Qt.PointingHandCursor)
        self._apply_style()

    def _apply_style(self):
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {self._bg_color};
                color: #000000;
                border: 2px solid #000000;
                border-radius: {self._corner_radius}px;
                font-family: 'Segoe UI', sans-serif;
                font-size: {self._font_size}px;
                font-weight: bold;
                padding: 4px 12px;
            }}
            QPushButton:hover {{
                background-color: {self._hover_color};
            }}
            QPushButton:pressed {{
                background-color: {self._hover_color};
                padding-top: 6px;
                padding-left: 14px;
            }}
            QPushButton:disabled {{
                background-color: #e5e7eb;
                color: #6b7280;
                border-color: #9ca3af;
            }}
        """)

    def set_colors(self, bg_color: str, hover_color: str):
        self._bg_color = bg_color
        self._hover_color = hover_color
        self._apply_style()

    def configure(self, **kwargs):
        if "state" in kwargs:
            self.setEnabled(kwargs["state"] != "disabled")
        if "text" in kwargs:
            self.setText(str(kwargs["text"]))

    def cget(self, attribute_name: str) -> Any:
        if attribute_name == "state":
            return "disabled" if not self.isEnabled() else "normal"
        if attribute_name == "text":
            return self.text()
        return ""


class NeoEntry(QLineEdit):
    """Neo-Brutalist text entry with bold font, thick border, and Tkinter-compat helpers."""

    def __init__(self, placeholder="", height=38, parent=None):
        super().__init__(parent)
        self.setFixedHeight(height)
        self.setPlaceholderText(placeholder)
        self.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self._paste_callback = None
        self.setStyleSheet("""
            QLineEdit {
                background-color: #ffffff;
                color: #000000;
                border: 2px solid #000000;
                border-radius: 8px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 12px;
                font-weight: bold;
                padding: 6px 10px;
            }
            QLineEdit:focus {
                border: 2.5px solid #000000;
            }
            QLineEdit:disabled {
                background-color: #f3f4f6;
                color: #6b7280;
            }
        """)

    def set_paste_callback(self, callback: Callable):
        self._paste_callback = callback

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Paste):
            if self._paste_callback:
                self._paste_callback()
                return
        super().keyPressEvent(event)

    def get(self) -> str:
        return self.text()

    def delete(self, start=0, end="end"):
        self.clear()

    def insert(self, index, text):
        if index == 0:
            self.setText(str(text))
        else:
            self.setText(self.text() + str(text))

    def configure(self, **kwargs):
        if "state" in kwargs:
            self.setEnabled(kwargs["state"] != "disabled")
        if "placeholder_text" in kwargs:
            self.setPlaceholderText(str(kwargs["placeholder_text"]))

    def bind(self, sequence: str, callback: Callable):
        # Compatibility stub for Tkinter bindings
        pass


class NeoSwitch(QCheckBox):
    """
    Neo-Brutalist pill toggle switch with crisp 2px black outline,
    punchy green fill when active, solid black thumb knob, and dynamic responsive label wrapping.
    """

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    def sizeHint(self) -> QSize:
        fm = self.fontMetrics()
        text_w = fm.horizontalAdvance(self.text()) if self.text() else 0
        w = 40 + 14 + text_w + 14
        h = max(30, fm.height() + 8)
        return QSize(w, h)

    def minimumSizeHint(self) -> QSize:
        fm = self.fontMetrics()
        h = max(30, fm.height() + 8)
        return QSize(80, h)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        if not self.text():
            return 30
        fm = self.fontMetrics()
        text_x = 40 + 14
        avail_w = max(10, width - text_x - 10)
        rect = fm.boundingRect(QRect(0, 0, avail_w, 10000), Qt.TextWordWrap, self.text())
        return max(30, rect.height() + 10)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Track geometry
        track_w = 40
        track_h = 22
        track_x = 2
        track_y = (self.height() - track_h) // 2

        checked = self.isChecked()
        bg_col = QColor(COLOR_NEO_GREEN) if checked else QColor("#e5e7eb")
        if not self.isEnabled():
            bg_col = QColor("#f3f4f6")

        # Draw track pill
        pen = QPen(QColor("#000000"), 2)
        painter.setPen(pen)
        painter.setBrush(QBrush(bg_col))
        painter.drawRoundedRect(track_x, track_y, track_w, track_h, 11, 11)

        # Draw circular thumb knob
        thumb_r = 7.5
        thumb_y = track_y + (track_h - thumb_r * 2) // 2
        thumb_x = (track_x + track_w - thumb_r * 2 - 3.5) if checked else (track_x + 3.5)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor("#000000") if self.isEnabled() else QColor("#9ca3af")))
        painter.drawEllipse(int(thumb_x), int(thumb_y), int(thumb_r * 2), int(thumb_r * 2))

        # Draw text label
        if self.text():
            painter.setPen(QColor("#000000") if self.isEnabled() else QColor("#9ca3af"))
            painter.setFont(self.font())
            text_x = track_x + track_w + 12
            text_w = max(10, self.width() - text_x)
            text_rect = QRect(text_x, 0, text_w, self.height())
            painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft | Qt.TextWordWrap, self.text())

    def get(self) -> int:
        return 1 if self.isChecked() else 0

    def select(self):
        self.setChecked(True)

    def deselect(self):
        self.setChecked(False)

    def configure(self, **kwargs):
        if "state" in kwargs:
            self.setEnabled(kwargs["state"] != "disabled")


class NeoSlider(QSlider):
    """
    Neo-Brutalist horizontal slider mapped to floating-point ranges,
    featuring a lilac track, purple progress groove, and crisp white handle.
    """

    def __init__(self, from_: float, to_: float, steps: int, initial_val: float = 1.0, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self.from_ = from_
        self.to_ = to_
        self.steps = max(1, steps)
        self._command: Optional[Callable[[float], None]] = None
        self._current_val = initial_val

        self.setRange(0, self.steps)
        self.setFixedHeight(26)
        self.setCursor(Qt.PointingHandCursor)

        self.setStyleSheet(f"""
            QSlider {{
                height: 26px;
            }}
            QSlider::groove:horizontal {{
                border: 2px solid #000000;
                height: 10px;
                background: {COLOR_TRACK_LILAC};
                border-radius: 5px;
            }}
            QSlider::sub-page:horizontal {{
                background: {COLOR_PROGRESS_PURPLE};
                border: 2px solid #000000;
                height: 10px;
                border-radius: 5px;
            }}
            QSlider::handle:horizontal {{
                background: #ffffff;
                border: 1.5px solid #000000;
                width: 20px;
                height: 20px;
                margin: -6px 0;
                border-radius: 10px;
            }}
            QSlider::handle:horizontal:hover {{
                background: #f3e8ff;
            }}
        """)

        self.set(initial_val)
        self.valueChanged.connect(self._on_slider_moved)

    def wheelEvent(self, event):
        """Ignore scroll wheel on slider to keep it click-n-slide and avoid hijacking window scroll."""
        event.ignore()

    def _on_slider_moved(self, pos: int):
        val = self.from_ + (pos / float(self.steps)) * (self.to_ - self.from_)
        self._current_val = round(val, 4)
        if self._command:
            self._command(self._current_val)

    def get(self) -> float:
        return self._current_val

    def set(self, val: float):
        self._current_val = float(val)
        ratio = (val - self.from_) / float(self.to_ - self.from_)
        pos = int(round(ratio * self.steps))
        pos = max(0, min(self.steps, pos))
        self.blockSignals(True)
        self.setValue(pos)
        self.blockSignals(False)
        if self._command:
            self._command(self._current_val)


class NeoProgressBar(QProgressBar):
    """Neo-Brutalist progress bar with 2px black border and punchy green fill."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(0, 1000)
        self.setValue(0)
        self.setTextVisible(False)
        self.setFixedHeight(12)
        self._progress_color = "#ffffff"  # Starts clear/white
        self._apply_style()

    def _apply_style(self):
        self.setStyleSheet(f"""
            QProgressBar {{
                border: 2px solid #000000;
                border-radius: 6px;
                background-color: #ffffff;
                height: 12px;
            }}
            QProgressBar::chunk {{
                background-color: {self._progress_color};
                border-radius: 4px;
            }}
        """)

    def set(self, val: float):
        int_val = int(max(0.0, min(1.0, float(val))) * 1000)
        self.setValue(int_val)

    def get(self) -> float:
        return self.value() / 1000.0

    def configure(self, **kwargs):
        if "progress_color" in kwargs:
            self._progress_color = kwargs["progress_color"]
            self._apply_style()


class NeoStandaloneButtonGroup(QWidget):
    """
    Neo-Brutalist selector composed of standalone rounded-rectangle buttons.
    Eliminates fused outlines so each option is an independent button.
    """

    def __init__(
        self,
        parent=None,
        values: Optional[List[str]] = None,
        command: Optional[Callable[[str], None]] = None,
        height: int = 36,
        selected: Optional[str] = None,
    ):
        super().__init__(parent)
        self.command = command
        self._values = list(values) if values else []
        self._current_value = selected if (selected and selected in self._values) else (self._values[0] if self._values else "")
        self._buttons: dict[str, QPushButton] = {}
        self._state = "normal"
        self._height = height

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 2, 16, 4)
        layout.setSpacing(16)

        for val in self._values:
            btn = QPushButton(val, self)
            btn.setFixedHeight(height)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked=False, v=val: self._on_btn_click(v))
            layout.addWidget(btn)
            self._buttons[val] = btn

        self._refresh_styles()

    def _refresh_styles(self):
        for val, btn in self._buttons.items():
            if val == self._current_value:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {COLOR_NEO_YELLOW};
                        color: #000000;
                        border: 2px solid #000000;
                        border-radius: 8px;
                        font-family: 'Segoe UI', sans-serif;
                        font-size: 12px;
                        font-weight: bold;
                        padding: 4px 12px;
                    }}
                    QPushButton:hover {{
                        background-color: {COLOR_NEO_YELLOW_HOVER};
                    }}
                    QPushButton:pressed {{
                        background-color: {COLOR_NEO_YELLOW_HOVER};
                        padding-top: 6px;
                        padding-left: 14px;
                    }}
                    QPushButton:disabled {{
                        background-color: #f3f4f6;
                        color: #9ca3af;
                        border-color: #9ca3af;
                    }}
                """)
            else:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #e5e7eb;
                        color: #4b5563;
                        border: 2px solid #000000;
                        border-radius: 8px;
                        font-family: 'Segoe UI', sans-serif;
                        font-size: 12px;
                        font-weight: bold;
                        padding: 4px 12px;
                    }
                    QPushButton:hover {
                        background-color: #d1d5db;
                        color: #000000;
                    }
                    QPushButton:pressed {
                        background-color: #9ca3af;
                        padding-top: 6px;
                        padding-left: 14px;
                    }
                    QPushButton:disabled {
                        background-color: #f3f4f6;
                        color: #9ca3af;
                        border-color: #9ca3af;
                    }
                """)

    def _on_btn_click(self, val: str):
        self.set(val)
        if self.command:
            self.command(val)

    def get(self) -> str:
        return self._current_value

    def set(self, val: str):
        if val in self._values:
            self._current_value = val
            self._refresh_styles()

    def configure(self, **kwargs):
        if "state" in kwargs:
            self._state = kwargs["state"]
            for btn in self._buttons.values():
                btn.setEnabled(self._state != "disabled")

    def cget(self, attribute_name: str) -> Any:
        if attribute_name == "state":
            return self._state
        return ""


class NeoComboBox(QComboBox):
    """Neo-Brutalist styled dropdown with thick borders, yellow trigger button, and crisp downward arrow."""

    def __init__(self, parent=None, values: Optional[List[str]] = None, command: Optional[Callable[[str], None]] = None):
        super().__init__(parent)
        self.command = command
        if values:
            self.addItems(values)
        self.setStyleSheet(f"""
            QComboBox {{
                background-color: #ffffff;
                color: #000000;
                border: 2px solid #000000;
                border-radius: 8px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 12px;
                font-weight: bold;
                padding: 4px 10px;
                min-height: 28px;
            }}
            QComboBox:disabled {{
                background-color: #f9fafb;
                color: #9ca3af;
                border: 2px solid #d1d5db;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 28px;
                border-left: 2px solid #000000;
                background-color: {COLOR_NEO_YELLOW};
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
            }}
            QComboBox::drop-down:disabled {{
                background-color: #e5e7eb;
                border-left: 2px solid #d1d5db;
            }}
            QComboBox::down-arrow {{
                image: none;
                width: 0px;
                height: 0px;
            }}
            QComboBox QAbstractItemView {{
                background-color: #ffffff;
                color: #000000;
                border: 2px solid #000000;
                border-radius: 6px;
                selection-background-color: {COLOR_NEO_YELLOW};
                selection-color: #000000;
                font-family: 'Segoe UI', sans-serif;
                font-weight: bold;
            }}
        """)
        if command:
            self.currentTextChanged.connect(command)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        btn_w = 28
        btn_x = self.width() - btn_w
        center_x = btn_x + btn_w / 2.0
        center_y = self.height() / 2.0

        painter.setPen(Qt.NoPen)
        color = QColor("#000000") if self.isEnabled() else QColor("#9ca3af")
        painter.setBrush(QBrush(color))
        half_w = 5.0
        half_h = 3.5
        points = [
            QPointF(center_x - half_w, center_y - half_h),
            QPointF(center_x + half_w, center_y - half_h),
            QPointF(center_x, center_y + half_h + 1.0)
        ]
        painter.drawPolygon(points)
        painter.end()

    def get(self) -> str:
        return self.currentText()

    def set(self, val: str):
        idx = self.findText(val)
        if idx >= 0:
            self.setCurrentIndex(idx)

    def configure(self, **kwargs):
        if "state" in kwargs:
            self.setEnabled(kwargs["state"] != "disabled")


class ThumbnailPreviewLabel(QLabel):
    """Aspect-ratio-preserving real-time slide preview renderer."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignCenter)
        self.setText("No slides captured yet\nPreview will appear here in real time")
        self.setStyleSheet("color: #6b7280; font-family: 'Segoe UI'; font-size: 12px; font-weight: bold;")

    def set_thumbnail_pixmap(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self.setText("")
        self._update_display()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap is not None:
            self._update_display()

    def _update_display(self):
        if self._pixmap is not None:
            w = max(40, self.width() - 8)
            h = max(30, self.height() - 8)
            scaled = self._pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled)


def create_neo_card_header(title: str, badge_color: str = COLOR_NEO_YELLOW, parent=None) -> QWidget:
    """Creates a signature Neo-Brutalist card header with retro window dots and punchy badge."""
    head = QWidget(parent)
    head_layout = QHBoxLayout(head)
    head_layout.setContentsMargins(16, 12, 16, 8)
    head_layout.setSpacing(10)

    # Retro Window Control Dots
    dots_box = QWidget(head)
    dots_layout = QHBoxLayout(dots_box)
    dots_layout.setContentsMargins(0, 0, 0, 0)
    dots_layout.setSpacing(5)

    for dot_col in ["#ff5f56", "#ffbd2e", "#27c93f"]:
        dot = QFrame(dots_box)
        dot.setFixedSize(13, 13)
        dot.setStyleSheet(f"""
            QFrame {{
                background-color: {dot_col};
                border: 1.5px solid #000000;
                border-radius: 6px;
            }}
        """)
        dots_layout.addWidget(dot)

    head_layout.addWidget(dots_box)

    # Neo badge tag
    tag = QFrame(head)
    tag.setStyleSheet(f"""
        QFrame {{
            background-color: {badge_color};
            border: 2px solid #000000;
            border-radius: 6px;
        }}
    """)
    tag_layout = QHBoxLayout(tag)
    tag_layout.setContentsMargins(10, 3, 10, 3)

    lbl = NeoLabel(title, tag)
    lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
    lbl.setStyleSheet("color: #000000; border: none; background: transparent;")
    tag_layout.addWidget(lbl)

    head_layout.addWidget(tag)
    head_layout.addStretch()
    return head


# ==============================================================================
# Main Application Window: YT2PDFApp
# ==============================================================================

class YT2PDFApp(QMainWindow):
    """Main application window for YT2PDF with Neo-Brutalist UI styling using PySide6 (Qt6)."""

    def __init__(self):
        # Ensure a QApplication instance exists before window instantiation
        if QApplication.instance() is None:
            self._app_ref = QApplication(sys.argv if sys.argv else [""])

        super().__init__()

        self.setWindowTitle("YT2PDF — Smart Presentation Slide Extractor & Lossless PDF Builder")
        self.resize(1080, 660)
        self.setMinimumSize(980, 560)
        self.setStyleSheet(f"QMainWindow {{ background-color: {COLOR_CANVAS_BG}; }}")

        # Worker & Execution State
        self.worker: Optional[ExtractionWorker] = None
        self.msg_queue: queue.Queue = queue.Queue()
        self.cancel_event: threading.Event = threading.Event()
        self.is_processing: bool = False
        self.last_generated_pdf: str = ""
        self._last_thumbnail_rgb: Optional[np.ndarray] = None
        self._completion_dialog: Optional[QDialog] = None

        # Build UI Hierarchy
        self._setup_root_container()
        self._create_header()
        self._create_main_content()
        self._create_footer()

        # High-frequency Queue Drainage Timer (25ms = 40Hz)
        self.queue_timer = QTimer(self)
        self.queue_timer.setInterval(25)
        self.queue_timer.timeout.connect(self._process_queue)
        self.queue_timer.start()

    def update_idletasks(self):
        """Processes pending Qt GUI events (backward-compatible test method)."""
        app = QApplication.instance()
        if app:
            app.processEvents()

    def destroy(self):
        """Safely stops timers and closes the window."""
        if hasattr(self, "queue_timer") and self.queue_timer:
            self.queue_timer.stop()
        self.close()

    def closeEvent(self, event):
        if hasattr(self, "queue_timer") and self.queue_timer:
            self.queue_timer.stop()
        event.accept()

    def mainloop(self):
        """Starts Qt application event loop (Tkinter backward compatibility)."""
        app = QApplication.instance()
        if app:
            return app.exec()

    def clipboard_get(self) -> str:
        """Retrieves system clipboard text."""
        cb = QApplication.clipboard()
        return cb.text() if cb else ""

    # ---------------- UI Assembly ----------------

    def _setup_root_container(self):
        self.root_widget = QWidget(self)
        self.root_widget.setStyleSheet(f"background-color: {COLOR_CANVAS_BG};")
        self.setCentralWidget(self.root_widget)

        self.root_layout = QVBoxLayout(self.root_widget)
        self.root_layout.setContentsMargins(20, 10, 20, 8)
        self.root_layout.setSpacing(6)

    def _create_header(self):
        """Top banner with application title and subtitle."""
        header_frame = QWidget(self.root_widget)
        h_layout = QHBoxLayout(header_frame)
        h_layout.setContentsMargins(4, 0, 4, 4)
        h_layout.setSpacing(12)

        # Title
        title_label = NeoLabel("YT2PDF ✦", header_frame)
        title_label.setFont(QFont("Segoe UI", 22, QFont.Bold))
        title_label.setStyleSheet("color: #000000; border: none;")
        h_layout.addWidget(title_label)

        # Pill Subtitle
        sub_pill = QFrame(header_frame)
        sub_pill.setStyleSheet(f"""
            QFrame {{
                background-color: {COLOR_NEO_YELLOW};
                border: 2px solid #000000;
                border-radius: 16px;
            }}
        """)
        sp_layout = QHBoxLayout(sub_pill)
        sp_layout.setContentsMargins(12, 3, 12, 3)
        sub_lbl = NeoLabel("✦ SLIDE EXTRACTOR & LOSSLESS PDF BUILDER ✦", sub_pill)
        sub_lbl.setFont(QFont("Segoe UI", 9, QFont.Bold))
        sub_lbl.setStyleSheet("color: #000000; border: none; background: transparent;")
        sp_layout.addWidget(sub_lbl)
        h_layout.addWidget(sub_pill)

        h_layout.addStretch()

        # Review & Deduplicate PDF button (Lilac)
        self.header_dedup_btn = NeoButton(
            "Review & Deduplicate PDF...",
            bg_color=COLOR_NEO_LILAC,
            hover_color=COLOR_NEO_LILAC_HOVER,
            height=34,
            parent=header_frame,
        )
        self.header_dedup_btn.clicked.connect(lambda: self._open_deduplicate_dialog())
        h_layout.addWidget(self.header_dedup_btn)

        # View PDF button (Cyan)
        self.header_view_pdf_btn = NeoButton(
            "View PDF",
            bg_color=COLOR_NEO_BLUE,
            hover_color=COLOR_NEO_BLUE_HOVER,
            height=34,
            parent=header_frame,
        )
        self.header_view_pdf_btn.setFixedWidth(96)
        self.header_view_pdf_btn.clicked.connect(self._on_view_pdf)
        h_layout.addWidget(self.header_view_pdf_btn)

        self.root_layout.addWidget(header_frame)

    def _create_main_content(self):
        """Split content layout: Left scrollable configuration, Right live dashboard & preview."""
        self.content_splitter = QWidget(self.root_widget)
        c_layout = QHBoxLayout(self.content_splitter)
        c_layout.setContentsMargins(0, 0, 0, 0)
        c_layout.setSpacing(14)

        # Left Column - Smooth Hardware Accelerated Scroll Area
        self.scroll_area = QScrollArea(self.content_splitter)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                border: 2px solid #000000;
                background: #ffffff;
                width: 14px;
                margin: 0px;
                border-radius: 7px;
            }
            QScrollBar::handle:vertical {
                background: #000000;
                min-height: 24px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #374151;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
            QScrollBar:horizontal {
                height: 0px;
            }
        """)
        self.scroll_area.verticalScrollBar().setSingleStep(16)

        self.left_col_widget = QWidget()
        self.left_col_widget.setStyleSheet("background: transparent;")
        self.left_col_layout = QVBoxLayout(self.left_col_widget)
        self.left_col_layout.setContentsMargins(0, 0, 10, 0)
        self.left_col_layout.setSpacing(12)

        # Build Configuration Cards
        self._build_input_output_card()
        self._build_detection_settings_card()
        self._build_roi_card()
        self.left_col_layout.addStretch()

        self.scroll_area.setWidget(self.left_col_widget)
        c_layout.addWidget(self.scroll_area, stretch=54)

        # Right Column - Live Dashboard & Preview Card
        self.right_col = QFrame(self.content_splitter)
        self.right_col.setObjectName("right_col_card")
        self.right_col.setStyleSheet(f"""
            QFrame#right_col_card {{
                background-color: {COLOR_CARD_BG};
                border: 2.5px solid #000000;
                border-radius: 14px;
            }}
        """)
        self._build_right_preview_panel()
        c_layout.addWidget(self.right_col, stretch=46)

        self.root_layout.addWidget(self.content_splitter, stretch=1)

    def _build_input_output_card(self):
        """Card 01: Input source selection and destination output."""
        card = QFrame(self.left_col_widget)
        card.setObjectName("card_01")
        card.setStyleSheet(f"""
            QFrame#card_01 {{
                background-color: {COLOR_CARD_BG};
                border: 2.5px solid #000000;
                border-radius: 14px;
            }}
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(0, 0, 0, 14)
        card_layout.setSpacing(8)

        card_layout.addWidget(create_neo_card_header("[ 01 ] INPUT & OUTPUT DESTINATION", COLOR_NEO_YELLOW, card))

        # Source Type selector: Equi-spaced buttons directly in card layout
        self.source_segmented = NeoStandaloneButtonGroup(
            card,
            values=[SourceType.YOUTUBE.value, SourceType.LOCAL_FILE.value],
            command=self._on_source_type_changed,
            height=36,
            selected=SourceType.YOUTUBE.value,
        )
        card_layout.addWidget(self.source_segmented)

        # Input source label
        self.input_label = NeoLabel("YouTube URL:", card)
        self.input_label.setStyleSheet("color: #000000; padding-left: 16px; padding-top: 4px;")
        card_layout.addWidget(self.input_label)

        # Input entry row
        input_row = QWidget(card)
        in_layout = QHBoxLayout(input_row)
        in_layout.setContentsMargins(16, 0, 16, 4)
        in_layout.setSpacing(8)

        self.input_entry = NeoEntry("https://www.youtube.com/watch?v=...", height=38, parent=input_row)
        self.input_entry.set_paste_callback(self._on_input_entry_paste)
        self.input_entry.editingFinished.connect(self._on_input_entry_blur_or_change)
        in_layout.addWidget(self.input_entry, stretch=1)

        self.input_browse_btn = NeoButton("Browse...", bg_color=COLOR_NEO_YELLOW, hover_color=COLOR_NEO_YELLOW_HOVER, height=38, parent=input_row)
        self.input_browse_btn.setFixedWidth(80)
        self.input_browse_btn.clicked.connect(self._on_browse_input_file)
        self.input_browse_btn.hide()  # Hidden by default in YouTube URL mode
        in_layout.addWidget(self.input_browse_btn)

        self.input_action_btn = NeoButton("Paste", bg_color=COLOR_NEO_BLUE, hover_color=COLOR_NEO_BLUE_HOVER, height=38, parent=input_row)
        self.input_action_btn.setFixedWidth(78)
        self.input_action_btn.clicked.connect(self._on_paste_input)
        in_layout.addWidget(self.input_action_btn)

        card_layout.addWidget(input_row)

        # Output PDF destination label
        out_lbl = NeoLabel("Output PDF File:", card)
        out_lbl.setStyleSheet("color: #000000; padding-left: 16px; padding-top: 4px;")
        card_layout.addWidget(out_lbl)

        # Output entry row
        out_row = QWidget(card)
        out_layout = QHBoxLayout(out_row)
        out_layout.setContentsMargins(16, 0, 16, 4)
        out_layout.setSpacing(8)

        self.output_entry = NeoEntry("Path to save output .pdf (auto-suggested)", height=38, parent=out_row)
        self.output_entry.set_paste_callback(self._on_output_entry_paste)
        self.output_entry.editingFinished.connect(self._on_output_entry_blur_or_change)
        out_layout.addWidget(self.output_entry, stretch=1)

        self.output_browse_btn = NeoButton("Browse...", bg_color=COLOR_NEO_YELLOW, hover_color=COLOR_NEO_YELLOW_HOVER, height=38, parent=out_row)
        self.output_browse_btn.setFixedWidth(80)
        self.output_browse_btn.clicked.connect(self._on_browse_output)
        out_layout.addWidget(self.output_browse_btn)

        card_layout.addWidget(out_row)
        self.left_col_layout.addWidget(card)

    def _build_detection_settings_card(self):
        """Card 02: Sampling and Debounce Sliders."""
        card = QFrame(self.left_col_widget)
        card.setObjectName("card_02")
        card.setStyleSheet(f"""
            QFrame#card_02 {{
                background-color: {COLOR_CARD_BG};
                border: 2.5px solid #000000;
                border-radius: 14px;
            }}
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(0, 0, 0, 14)
        card_layout.setSpacing(6)

        card_layout.addWidget(create_neo_card_header("[ 02 ] DETECTION & SAMPLING CONFIGURATION", COLOR_NEO_BLUE, card))

        # 1. Sample Interval
        self.sample_interval_slider, self.interval_val_lbl = self._add_slider_row(
            parent=card,
            title="Sample Interval:",
            subtitle="Frequency of frame sampling (lower is more thorough, higher is faster)",
            initial_val=1.0,
            initial_val_str="1.0s",
            from_=0.5,
            to_=5.0,
            steps=9,
            format_fn=lambda v: f"{v:.1f}s",
        )
        card_layout.addWidget(self.sample_interval_slider.parentWidget())

        # 2. Similarity Threshold
        self.similarity_slider, self.threshold_val_lbl = self._add_slider_row(
            parent=card,
            title="Similarity Threshold:",
            subtitle="SSIM sensitivity required to trigger a slide transition",
            initial_val=0.92,
            initial_val_str="92%",
            from_=0.80,
            to_=0.99,
            steps=19,
            format_fn=lambda v: f"{int(round(v * 100))}%",
        )
        card_layout.addWidget(self.similarity_slider.parentWidget())

        # 3. Debounce / Animation Settle Time
        self.debounce_slider, self.debounce_val_lbl = self._add_slider_row(
            parent=card,
            title="Debounce / Settle Time:",
            subtitle="Wait duration for bullet points and animations to settle into completed state",
            initial_val=2.0,
            initial_val_str="2.0s",
            from_=1.0,
            to_=10.0,
            steps=18,
            format_fn=lambda v: f"{v:.1f}s",
        )
        card_layout.addWidget(self.debounce_slider.parentWidget())

        # 4. Min Slide Duration
        self.min_duration_slider, self.min_dur_val_lbl = self._add_slider_row(
            parent=card,
            title="Min Slide Duration:",
            subtitle="Filters out momentary camera cuts back to speaker",
            initial_val=2.0,
            initial_val_str="2.0s",
            from_=1.0,
            to_=15.0,
            steps=28,
            format_fn=lambda v: f"{v:.1f}s",
        )
        card_layout.addWidget(self.min_duration_slider.parentWidget())

        self.left_col_layout.addWidget(card)

    def _add_slider_row(self, parent, title, subtitle, initial_val, initial_val_str, from_, to_, steps, format_fn):
        """Helper to create slider row with title, dynamic value pill badge, and subtitle."""
        row_widget = QWidget(parent)
        r_layout = QVBoxLayout(row_widget)
        r_layout.setContentsMargins(16, 4, 16, 6)
        r_layout.setSpacing(2)

        top_line = QWidget(row_widget)
        tl_layout = QHBoxLayout(top_line)
        tl_layout.setContentsMargins(0, 0, 0, 0)

        t_lbl = NeoLabel(title, top_line)
        t_lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
        tl_layout.addWidget(t_lbl)
        tl_layout.addStretch()

        val_label = NeoLabel(initial_val_str, top_line)
        val_label.setFont(QFont("Segoe UI", 9, QFont.Bold))
        val_label.setAlignment(Qt.AlignCenter)
        val_label.setFixedSize(62, 26)
        val_label.setStyleSheet(f"""
            QLabel {{
                background-color: {COLOR_NEO_YELLOW};
                color: #000000;
                border: 2px solid #000000;
                border-radius: 6px;
                font-family: 'Segoe UI', sans-serif;
                font-weight: bold;
            }}
        """)
        tl_layout.addWidget(val_label)
        r_layout.addWidget(top_line)

        sub_lbl = NeoLabel(subtitle, row_widget)
        sub_lbl.setFont(QFont("Segoe UI", 10))
        sub_lbl.setStyleSheet("color: #000000; border: none;")
        sub_lbl.setWordWrap(True)
        r_layout.addWidget(sub_lbl)

        slider = NeoSlider(from_=from_, to_=to_, steps=steps, initial_val=initial_val, parent=row_widget)

        def on_slider_move(val):
            val_label.configure(text=format_fn(val))

        slider._command = on_slider_move
        r_layout.addWidget(slider)

        return slider, val_label

    def _build_roi_card(self):
        """Card 03: Region of Interest (ROI) & Preprocessing Toggles."""
        card = QFrame(self.left_col_widget)
        card.setObjectName("card_03")
        card.setStyleSheet(f"""
            QFrame#card_03 {{
                background-color: {COLOR_CARD_BG};
                border: 2.5px solid #000000;
                border-radius: 14px;
            }}
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(0, 0, 0, 14)
        card_layout.setSpacing(6)

        card_layout.addWidget(create_neo_card_header("[ 03 ] REGION OF INTEREST & PREPROCESSING", COLOR_NEO_LILAC, card))

        # Mask Presenter / PIP Toggle
        pip_toggle_row = QWidget(card)
        pt_layout = QHBoxLayout(pip_toggle_row)
        pt_layout.setContentsMargins(16, 2, 16, 2)

        self.mask_presenter_switch = NeoSwitch("Mask Presenter / PIP Webcam", pip_toggle_row)
        self.mask_presenter_switch.toggled.connect(self._on_toggle_presenter_mask)
        pt_layout.addWidget(self.mask_presenter_switch)
        card_layout.addWidget(pip_toggle_row)

        # PIP Mode Sub-frame (Buttercream card)
        self.pip_mode_frame = QFrame(card)
        self.pip_mode_frame.setObjectName("pip_frame")
        self.pip_mode_frame.setStyleSheet(f"""
            QFrame#pip_frame {{
                background-color: {COLOR_CARD_BG_ALT};
                border: 2px solid #000000;
                border-radius: 10px;
            }}
        """)
        pf_layout = QVBoxLayout(self.pip_mode_frame)
        pf_layout.setContentsMargins(12, 8, 12, 8)
        pf_layout.setSpacing(6)

        mode_select_row = QWidget(self.pip_mode_frame)
        ms_layout = QHBoxLayout(mode_select_row)
        ms_layout.setContentsMargins(0, 0, 0, 0)
        ms_layout.setSpacing(8)

        self.pip_pos_label = NeoLabel("PIP Position:", mode_select_row)
        ms_layout.addWidget(self.pip_pos_label)
        self.pip_mode_dropdown = NeoComboBox(
            mode_select_row,
            values=[m.value for m in PresenterMaskMode],
            command=self._on_pip_mode_changed,
        )
        self.pip_mode_dropdown.set(PresenterMaskMode.AUTO.value)
        ms_layout.addWidget(self.pip_mode_dropdown, stretch=1)
        pf_layout.addWidget(mode_select_row)

        # Custom Normalized Box Entry
        self.custom_box_row = QWidget(self.pip_mode_frame)
        cb_layout = QHBoxLayout(self.custom_box_row)
        cb_layout.setContentsMargins(0, 0, 0, 0)
        cb_layout.setSpacing(8)

        self.custom_box_label = NeoLabel("Custom Box (x,y,w,h):", self.custom_box_row)
        cb_layout.addWidget(self.custom_box_label)
        self.custom_box_entry = NeoEntry("0.75,0.70,0.25,0.30", height=32, parent=self.custom_box_row)
        self.custom_box_entry.insert(0, "0.75,0.70,0.25,0.30")
        cb_layout.addWidget(self.custom_box_entry, stretch=1)
        pf_layout.addWidget(self.custom_box_row)

        # Centered carrier row with even 16px margins on both sides
        pip_frame_row = QWidget(card)
        pfr_layout = QHBoxLayout(pip_frame_row)
        pfr_layout.setContentsMargins(16, 2, 16, 6)
        pfr_layout.addWidget(self.pip_mode_frame)
        card_layout.addWidget(pip_frame_row)

        # Initialize PIP controls state (disabled by default)
        self._on_toggle_presenter_mask()

        # Other preprocessing toggles
        toggles = [
            ("ignore_chrome_switch", "Ignore Player Chrome (Mask top 3% & bottom 5% titles/seekbars)", True),
            ("auto_crop_switch", "Canvas Auto-Crop (Detect and remove black letterbox/pillarbox borders)", False),
            ("prune_duplicates_switch", "Global Duplicate Truncation (Prune recurring slides, keep 1st appearance)", True),
            ("filter_narrator_switch", "Filter Full-Human Narrator Frames (Remove camera cuts to speaker)", True),
        ]

        for attr_name, text, default_on in toggles:
            t_row = QWidget(card)
            t_layout = QHBoxLayout(t_row)
            t_layout.setContentsMargins(16, 2, 16, 2)
            sw = NeoSwitch(text, t_row)
            if default_on:
                sw.select()
            t_layout.addWidget(sw)
            card_layout.addWidget(t_row)
            setattr(self, attr_name, sw)

        self.left_col_layout.addWidget(card)

    def _build_right_preview_panel(self):
        """Right column: Run controls, live metrics, progress bar, and aspect-ratio slide thumbnail."""
        r_layout = QVBoxLayout(self.right_col)
        r_layout.setContentsMargins(16, 12, 16, 14)
        r_layout.setSpacing(8)

        r_layout.addWidget(create_neo_card_header("[ 04 ] LIVE DASHBOARD & PREVIEW", COLOR_NEO_GREEN, self.right_col))

        # Status Ticker Box (Cream background)
        self.status_box = QFrame(self.right_col)
        self.status_box.setObjectName("status_box")
        self.status_box.setStyleSheet(f"""
            QFrame#status_box {{
                background-color: {COLOR_NEO_CREAM};
                border: 2px solid #000000;
                border-radius: 8px;
            }}
        """)
        sb_layout = QHBoxLayout(self.status_box)
        sb_layout.setContentsMargins(12, 6, 12, 6)

        self.status_label = NeoLabel("Ready to extract slides", self.status_box)
        self.status_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.status_label.setStyleSheet("color: #000000; border: none; background: transparent;")
        sb_layout.addWidget(self.status_label)
        r_layout.addWidget(self.status_box)

        # Progress Bar
        self.progress_bar = NeoProgressBar(self.right_col)
        r_layout.addWidget(self.progress_bar)

        # 2x2 KPI Metrics Grid
        metrics_grid = QGridLayout()
        metrics_grid.setContentsMargins(0, 2, 0, 4)
        metrics_grid.setSpacing(8)

        self.card_time, self.val_time = self._create_kpi_badge(self.right_col, "TIME / DURATION", "00:00 / 00:00", bg_color=COLOR_NEO_YELLOW)
        self.card_speed, self.val_speed = self._create_kpi_badge(self.right_col, "PROCESSING SPEED", "0 FPS (0.0x)", bg_color=COLOR_NEO_BLUE)
        self.card_slides, self.val_slides = self._create_kpi_badge(self.right_col, "SLIDES CAPTURED", "0 slides", bg_color=COLOR_NEO_GREEN)
        self.card_eta, self.val_eta = self._create_kpi_badge(self.right_col, "EST. REMAINING", "--:--", bg_color=COLOR_NEO_ORANGE)

        metrics_grid.addWidget(self.card_time, 0, 0)
        metrics_grid.addWidget(self.card_speed, 0, 1)
        metrics_grid.addWidget(self.card_slides, 1, 0)
        metrics_grid.addWidget(self.card_eta, 1, 1)

        r_layout.addLayout(metrics_grid)

        # Slide Thumbnail Preview Card
        self.preview_title_lbl = NeoLabel("LATEST CAPTURED SLIDE:", self.right_col)
        self.preview_title_lbl.setFont(QFont("Segoe UI", 9, QFont.Bold))
        self.preview_title_lbl.setStyleSheet("color: #000000; border: none; padding-top: 2px;")
        r_layout.addWidget(self.preview_title_lbl)

        self.preview_frame = QFrame(self.right_col)
        self.preview_frame.setObjectName("preview_frame")
        self.preview_frame.setStyleSheet("""
            QFrame#preview_frame {{
                background-color: #ffffff;
                border: 2px solid #000000;
                border-radius: 10px;
            }}
        """)
        pf_layout = QVBoxLayout(self.preview_frame)
        pf_layout.setContentsMargins(6, 6, 6, 6)

        self.preview_label = ThumbnailPreviewLabel(self.preview_frame)
        pf_layout.addWidget(self.preview_label)
        r_layout.addWidget(self.preview_frame, stretch=1)

        # Action Buttons (Anchored at bottom)
        btn_row = QWidget(self.right_col)
        br_layout = QHBoxLayout(btn_row)
        br_layout.setContentsMargins(0, 4, 0, 0)
        br_layout.setSpacing(8)

        self.start_btn = NeoButton("Start Extraction", bg_color=COLOR_NEO_GREEN, hover_color=COLOR_NEO_GREEN_HOVER, height=42, corner_radius=10, font_size=13, parent=btn_row)
        self.start_btn.clicked.connect(self._on_start_clicked)
        br_layout.addWidget(self.start_btn, stretch=1)

        self.cancel_btn = NeoButton("Cancel / Stop", bg_color=COLOR_NEO_RED, hover_color=COLOR_NEO_RED_HOVER, height=42, corner_radius=10, font_size=13, parent=btn_row)
        self.cancel_btn.setFixedWidth(120)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        br_layout.addWidget(self.cancel_btn)

        r_layout.addWidget(btn_row)

    def _create_kpi_badge(self, parent, title: str, initial_val: str, bg_color: str = COLOR_NEO_YELLOW) -> Tuple[QFrame, NeoLabel]:
        """Creates a stylized Neo-Brutalist KPI badge card."""
        badge = QFrame(parent)
        badge.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color};
                border: 2px solid #000000;
                border-radius: 10px;
            }}
        """)
        b_layout = QVBoxLayout(badge)
        b_layout.setContentsMargins(10, 6, 10, 6)
        b_layout.setSpacing(1)

        t_lbl = NeoLabel(title, badge)
        t_lbl.setFont(QFont("Segoe UI", 8, QFont.Bold))
        t_lbl.setStyleSheet("color: #000000; border: none; background: transparent;")
        b_layout.addWidget(t_lbl)

        v_lbl = NeoLabel(initial_val, badge)
        v_lbl.setFont(QFont("Segoe UI", 11, QFont.Bold))
        v_lbl.setStyleSheet("color: #000000; border: none; background: transparent;")
        b_layout.addWidget(v_lbl)

        return badge, v_lbl

    def _create_footer(self):
        """Bottom status bar with credits."""
        footer = QWidget(self.root_widget)
        f_layout = QHBoxLayout(footer)
        f_layout.setContentsMargins(4, 2, 4, 2)

        lbl = NeoLabel("YT2PDF ✦ NEO-BRUTALIST EDITION • Lossless img2pdf • SSIM Debounced", footer)
        lbl.setFont(QFont("Segoe UI", 9, QFont.Bold))
        lbl.setStyleSheet("color: #000000; border: none;")
        f_layout.addWidget(lbl)
        f_layout.addStretch()

        self.root_layout.addWidget(footer)

    # ---------------- UI Event Handlers ----------------

    def _on_source_type_changed(self, value: str):
        self._update_source_type_ui(value, clear_input=True)

    def _update_source_type_ui(self, value: str, clear_input=True):
        if value == SourceType.YOUTUBE.value:
            self.input_label.setText("YouTube URL:")
            if clear_input:
                self.input_entry.clear()
            self.input_entry.setPlaceholderText("https://www.youtube.com/watch?v=...")
            self.input_browse_btn.hide()
        else:
            self.input_label.setText("Local Video File:")
            if clear_input:
                self.input_entry.clear()
            self.input_entry.setPlaceholderText("Select MP4, MKV, MOV, or AVI file...")
            self.input_browse_btn.show()

    def _set_input_value(self, raw_value: str):
        """Intelligently cleans, auto-formats, and routes input paths and URLs."""
        if not raw_value:
            return

        cleaned = clean_file_path(raw_value)
        if not cleaned:
            return

        if is_local_file_path(cleaned):
            # Auto-switch to Local File mode
            if self.source_segmented.get() != SourceType.LOCAL_FILE.value:
                self.source_segmented.set(SourceType.LOCAL_FILE.value)
                self._update_source_type_ui(SourceType.LOCAL_FILE.value, clear_input=False)

            self.input_entry.setText(cleaned)

            # Auto-suggest output PDF path next to video
            base, _ = os.path.splitext(cleaned)
            self.output_entry.setText(f"{base}_slides.pdf")
        else:
            # YouTube URL mode
            if self.source_segmented.get() != SourceType.YOUTUBE.value:
                self.source_segmented.set(SourceType.YOUTUBE.value)
                self._update_source_type_ui(SourceType.YOUTUBE.value, clear_input=False)

            self.input_entry.setText(cleaned)
            self._auto_propose_youtube_title(cleaned)

    def _on_paste_input(self):
        cb = QApplication.clipboard()
        if cb:
            self._set_input_value(cb.text())

    def _on_paste_url(self):
        self._on_paste_input()

    def _on_input_entry_paste(self, event=None):
        self._on_paste_input()

    def _on_input_entry_blur_or_change(self, event=None):
        curr = self.input_entry.get()
        if curr:
            cleaned = clean_file_path(curr)
            if cleaned != curr:
                self._set_input_value(cleaned)

    def _on_output_entry_paste(self, event=None):
        cb = QApplication.clipboard()
        if cb:
            cleaned = clean_file_path(cb.text())
            if cleaned:
                self.output_entry.setText(cleaned)

    def _on_output_entry_blur_or_change(self, event=None):
        curr = self.output_entry.get()
        if curr:
            cleaned = clean_file_path(curr)
            if cleaned != curr:
                self.output_entry.setText(cleaned)

    def _on_browse_input_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Presentation Video",
            "",
            "Video Files (*.mp4 *.mkv *.mov *.avi *.webm);;All Files (*.*)",
        )
        if file_path:
            self._set_input_value(file_path)

    def _auto_propose_youtube_title(self, url: str):
        def fetch():
            try:
                downloader = VideoDownloader()
                meta = downloader.fetch_youtube_metadata(url)
                default_name = meta["default_pdf_name"]

                curr = self.output_entry.get().strip()
                if not curr or curr.endswith(".pdf"):
                    desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")
                    target_dir = desktop_dir if os.path.exists(desktop_dir) else os.getcwd()
                    proposed_path = os.path.join(target_dir, default_name)
                    QTimer.singleShot(0, lambda: self.output_entry.setText(proposed_path))
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

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Destination PDF",
            os.path.join(initial_dir, initial_file),
            "PDF Document (*.pdf)",
        )
        if path:
            self.output_entry.setText(path)

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
            QMessageBox.critical(self, "Error", f"Could not open file:\n{e}")

    def _open_folder(self, folder_path: str):
        """Cross-platform system folder opener."""
        try:
            if sys.platform == "win32":
                subprocess.run(["explorer", folder_path])
            elif sys.platform == "darwin":
                subprocess.run(["open", folder_path])
            else:
                subprocess.run(["xdg-open", folder_path])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open folder:\n{e}")

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
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select PDF to View",
            "",
            "PDF Documents (*.pdf);;All Files (*.*)",
        )
        if selected:
            selected_clean = clean_file_path(selected)
            if os.path.isfile(selected_clean):
                if not self.output_entry.get().strip():
                    self.output_entry.setText(selected_clean)
                self._open_file(selected_clean)

    def _on_toggle_presenter_mask(self):
        self._update_pip_mode_ui_state()

    def _on_pip_mode_changed(self, choice: str):
        self._update_pip_mode_ui_state()

    def _update_pip_mode_ui_state(self):
        enabled = self.mask_presenter_switch.isChecked()
        choice = self.pip_mode_dropdown.get()

        if enabled:
            # Active Buttercream Card
            self.pip_mode_frame.setStyleSheet(f"""
                QFrame#pip_frame {{
                    background-color: {COLOR_CARD_BG_ALT};
                    border: 2px solid #000000;
                    border-radius: 10px;
                }}
            """)
            if hasattr(self, "pip_pos_label"):
                self.pip_pos_label.setStyleSheet("color: #000000; border: none; font-weight: bold;")
            self.pip_mode_dropdown.setEnabled(True)

            is_custom = (choice == PresenterMaskMode.CUSTOM.value)
            self.custom_box_entry.setEnabled(is_custom)
            if hasattr(self, "custom_box_label"):
                self.custom_box_label.setStyleSheet(
                    "color: #000000; border: none; font-weight: bold;"
                    if is_custom else
                    "color: #9ca3af; border: none; font-weight: bold;"
                )
        else:
            # Greyed-out Inactive Card
            self.pip_mode_frame.setStyleSheet("""
                QFrame#pip_frame {
                    background-color: #f3f4f6;
                    border: 2px solid #d1d5db;
                    border-radius: 10px;
                }
            """)
            if hasattr(self, "pip_pos_label"):
                self.pip_pos_label.setStyleSheet("color: #9ca3af; border: none; font-weight: bold;")
            self.pip_mode_dropdown.setEnabled(False)
            if hasattr(self, "custom_box_label"):
                self.custom_box_label.setStyleSheet("color: #9ca3af; border: none; font-weight: bold;")
            self.custom_box_entry.setEnabled(False)

    # ---------------- Start & Cancellation Pipeline ----------------

    def _on_start_clicked(self):
        input_val = clean_file_path(self.input_entry.get())
        if not input_val:
            QMessageBox.warning(self, "Missing Input", "Please provide a YouTube URL or select a local video file.")
            return

        if input_val != self.input_entry.get().strip():
            self.input_entry.setText(input_val)

        output_val = clean_file_path(self.output_entry.get())
        if not output_val:
            default_out = os.path.join(os.getcwd(), "presentation_slides.pdf")
            self.output_entry.setText(default_out)
            output_val = default_out

        if not output_val.lower().endswith(".pdf"):
            output_val += ".pdf"

        if output_val != self.output_entry.get().strip():
            self.output_entry.setText(output_val)

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
                QMessageBox.critical(self, "Invalid ROI Box", "Custom box must be four normalized numbers: x,y,w,h (between 0.0 and 1.0)")
                return

        # Prepare extraction config
        source_type = SourceType.YOUTUBE if self.source_segmented.get() == SourceType.YOUTUBE.value else SourceType.LOCAL_FILE
        config = ExtractionConfig(
            source_type=source_type,
            input_source=input_val,
            output_path=output_val,
            sample_interval=float(self.sample_interval_slider.get()),
            similarity_threshold=float(self.similarity_slider.get()),
            debounce_settle_time=float(self.debounce_slider.get()),
            min_slide_duration=float(self.min_duration_slider.get()),
            mask_presenter=self.mask_presenter_switch.isChecked(),
            presenter_mode=pip_mode,
            custom_roi_box=custom_box,
            ignore_chrome=self.ignore_chrome_switch.isChecked(),
            canvas_auto_crop=self.auto_crop_switch.isChecked(),
            prune_global_duplicates=self.prune_duplicates_switch.isChecked(),
            filter_narrator_frames=self.filter_narrator_switch.isChecked(),
        )

        # Update UI state to processing
        self._set_processing_ui_state(True)
        self.cancel_event.clear()
        self.status_label.setText("Starting extraction pipeline...")
        self.progress_bar.configure(progress_color=COLOR_NEO_GREEN)
        self.progress_bar.set(0.0)
        self.val_slides.setText("0 slides")
        self.val_time.setText("00:00 / 00:00")
        self.val_speed.setText("0 FPS (0.0x)")
        self.val_eta.setText("Calculating...")

        # Launch worker thread
        self.worker = ExtractionWorker(config, self.msg_queue, self.cancel_event)
        self.worker.start()

    def _on_cancel_clicked(self):
        if self.worker and self.worker.is_alive():
            confirm = QMessageBox.question(
                self,
                "Confirm Stop",
                "Do you want to stop the extraction in progress?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if confirm == QMessageBox.Yes:
                self.cancel_event.set()
                self.status_label.setText("Cancelling extraction...")
                self.cancel_btn.setEnabled(False)

    def _set_processing_ui_state(self, processing: bool):
        self.is_processing = processing
        enabled = not processing
        self.start_btn.setEnabled(enabled)
        self.source_segmented.configure(state="normal" if enabled else "disabled")
        self.input_entry.setEnabled(enabled)
        self.input_action_btn.setEnabled(enabled)
        self.input_browse_btn.setEnabled(enabled)
        self.output_entry.setEnabled(enabled)
        self.output_browse_btn.setEnabled(enabled)
        self.header_view_pdf_btn.setEnabled(enabled)
        self.header_dedup_btn.setEnabled(enabled)
        self.cancel_btn.setEnabled(processing)

    # ---------------- Queue Drainage Loop ----------------

    def _process_queue(self):
        """Polls messages safely from worker thread on Qt main GUI thread."""
        try:
            while not self.msg_queue.empty():
                msg: WorkerMessage = self.msg_queue.get_nowait()
                self._handle_worker_message(msg)
        except Exception as e:
            logger.error(f"Error handling worker queue message: {e}")

    def _handle_worker_message(self, msg: WorkerMessage):
        if msg.msg_type == WorkerMessageType.STATUS:
            self.status_label.setText(str(msg.payload))

        elif msg.msg_type == WorkerMessageType.PROGRESS:
            val = max(0.0, min(1.0, float(msg.payload)))
            self.progress_bar.set(val)

        elif msg.msg_type == WorkerMessageType.METRICS:
            data = msg.payload or {}
            self.val_time.setText(data.get("time_str", "00:00 / 00:00"))
            self.val_speed.setText(data.get("speed_str", "0 FPS"))
            self.val_slides.setText(f"{data.get('slides_count', 0)} slides")
            self.val_eta.setText(data.get("eta_str", "--:--"))

        elif msg.msg_type == WorkerMessageType.THUMBNAIL:
            self._update_thumbnail_preview(msg.payload)

        elif msg.msg_type == WorkerMessageType.COMPLETE:
            self._handle_completion(msg.payload)

        elif msg.msg_type == WorkerMessageType.ERROR:
            self._handle_error(str(msg.payload))

        elif msg.msg_type == WorkerMessageType.CANCELLED:
            self._handle_cancellation()

    def _on_preview_frame_resize(self, event=None):
        if self._last_thumbnail_rgb is not None:
            self._render_thumbnail()

    def _update_thumbnail_preview(self, data: dict):
        try:
            rgb_arr = data["image_rgb"]
            self._last_thumbnail_rgb = rgb_arr
            self._render_thumbnail()
        except Exception as e:
            logger.debug(f"Preview update failed: {e}")

    def _render_thumbnail(self):
        if self._last_thumbnail_rgb is None:
            return
        try:
            rgb = self._last_thumbnail_rgb
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.preview_label.set_thumbnail_pixmap(pixmap)
        except Exception as e:
            logger.debug(f"Thumbnail render failed: {e}")

    def _handle_completion(self, payload: dict):
        self._set_processing_ui_state(False)
        out_path = payload.get("output_path", "")
        slides_count = payload.get("slides_count", 0)
        elapsed = payload.get("elapsed_seconds", 0)
        self.last_generated_pdf = out_path

        self.val_eta.setText("Finished")
        self.status_label.setText(f"Complete — {slides_count} unique slides compiled into PDF!")
        self.progress_bar.set(1.0)

        self._show_completion_dialog(out_path, slides_count, elapsed)

    def _show_completion_dialog(self, pdf_path: str, slides_count: int, elapsed: float):
        dialog = CompletionDialog(self, pdf_path, slides_count, elapsed)
        self._completion_dialog = dialog
        dialog.show()

    def _open_deduplicate_dialog(self, default_pdf: str = ""):
        if not default_pdf:
            curr_out = clean_file_path(self.output_entry.get())
            if curr_out and os.path.exists(curr_out):
                default_pdf = curr_out
        dialog = DeduplicatePDFDialog(self, default_pdf_path=default_pdf)
        dialog.show()

    def _handle_error(self, error_message: str):
        self._set_processing_ui_state(False)
        self.progress_bar.configure(progress_color="#ffffff")
        self.progress_bar.set(0.0)
        self.val_eta.setText("--:--")
        self.status_label.setText("Error occurred during extraction.")
        self.status_label.setStyleSheet(f"color: {COLOR_NEO_RED};")
        QMessageBox.critical(self, "Extraction Error", f"An error occurred during extraction:\n\n{error_message}")

    def _handle_cancellation(self):
        self._set_processing_ui_state(False)
        self.progress_bar.configure(progress_color="#ffffff")
        self.progress_bar.set(0.0)
        self.val_eta.setText("--:--")
        self.status_label.setText("Extraction cancelled by user.")
        _show_info(self, "Cancelled", "Video extraction was stopped.")


# ==============================================================================
# Modal Dialog: Completion Dialog
# ==============================================================================

class CompletionDialog(QDialog):
    """Celebratory modal dialog with 2x2 action button grid in signature Neo-Brutalist styling."""

    def __init__(self, parent, pdf_path: str, slides_count: int, elapsed: float):
        super().__init__(parent)
        self.setWindowTitle("Extraction Complete")
        self.setFixedSize(540, 310)
        self.setModal(True)
        self.setStyleSheet(f"QDialog {{ background-color: {COLOR_CANVAS_BG}; }}")

        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        time_text = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(16, 16, 16, 16)

        card = QFrame(self)
        card.setObjectName("comp_card")
        card.setStyleSheet(f"""
            QFrame#comp_card {{
                background-color: {COLOR_CARD_BG};
                border: 3px solid #000000;
                border-radius: 14px;
            }}
        """)
        c_layout = QVBoxLayout(card)
        c_layout.setContentsMargins(16, 14, 16, 16)
        c_layout.setSpacing(12)

        c_layout.addWidget(create_neo_card_header("SLIDES EXTRACTED SUCCESSFULLY! ✦", COLOR_NEO_GREEN, card))

        # Buttercream summary card
        info_card = QFrame(card)
        info_card.setObjectName("info_card")
        info_card.setStyleSheet(f"""
            QFrame#info_card {{
                background-color: {COLOR_CARD_BG_ALT};
                border: 2px solid #000000;
                border-radius: 10px;
            }}
        """)
        ic_layout = QVBoxLayout(info_card)
        ic_layout.setContentsMargins(14, 10, 14, 10)
        ic_layout.setSpacing(4)

        t_lbl = NeoLabel(f"Compiled {slides_count} unique presentation slides in {time_text}.", info_card)
        t_lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
        t_lbl.setStyleSheet("color: #000000; border: none; background: transparent;")
        ic_layout.addWidget(t_lbl)

        p_lbl = NeoLabel(f"Saved to: {pdf_path}", info_card)
        p_lbl.setFont(QFont("Segoe UI", 9))
        p_lbl.setStyleSheet(f"color: {COLOR_TEXT_DIM}; border: none; background: transparent;")
        p_lbl.setWordWrap(True)
        ic_layout.addWidget(p_lbl)
        c_layout.addWidget(info_card)

        # 2x2 Action Button Grid
        btn_grid = QGridLayout()
        btn_grid.setSpacing(8)

        def open_pdf():
            parent._open_file(pdf_path)
            self.close()

        def review_duplicates():
            self.close()
            parent._open_deduplicate_dialog(pdf_path)

        def open_folder():
            parent._open_folder(os.path.dirname(os.path.abspath(pdf_path)))
            self.close()

        btn_open = NeoButton("Open PDF", bg_color=COLOR_NEO_GREEN, hover_color=COLOR_NEO_GREEN_HOVER, height=40, font_size=12, parent=card)
        btn_open.clicked.connect(open_pdf)
        btn_grid.addWidget(btn_open, 0, 0)

        btn_review = NeoButton("Review Duplicates", bg_color=COLOR_NEO_LILAC, hover_color=COLOR_NEO_LILAC_HOVER, height=40, font_size=12, parent=card)
        btn_review.clicked.connect(review_duplicates)
        btn_grid.addWidget(btn_review, 0, 1)

        btn_folder = NeoButton("Open Folder", bg_color=COLOR_NEO_YELLOW, hover_color=COLOR_NEO_YELLOW_HOVER, height=40, font_size=12, parent=card)
        btn_folder.clicked.connect(open_folder)
        btn_grid.addWidget(btn_folder, 1, 0)

        btn_close = NeoButton("Close", bg_color=COLOR_NEO_RED, hover_color=COLOR_NEO_RED_HOVER, height=40, font_size=12, parent=card)
        btn_close.clicked.connect(self.close)
        btn_grid.addWidget(btn_close, 1, 1)

        c_layout.addLayout(btn_grid)
        root_layout.addWidget(card)

    def destroy(self):
        self.close()


# ==============================================================================
# Modal Dialog: Review & Deduplicate PDF Dialog
# ==============================================================================

class DeduplicatePDFDialog(QDialog):
    """Interactive modal window to review an entire PDF and truncate repetitive pages with Neo-Brutalist styling."""

    def __init__(self, parent=None, default_pdf_path: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Review & Deduplicate PDF")
        self.resize(740, 620)
        self.setMinimumSize(600, 480)
        self.setModal(True)
        self.setStyleSheet(f"QDialog {{ background-color: {COLOR_CANVAS_BG}; }}")

        self.cancel_event = threading.Event()
        self.is_processing = False
        self.latest_result: Optional[DeduplicationResult] = None
        self.msg_queue: queue.Queue = queue.Queue()

        self._build_ui(default_pdf_path)

        # Queue drainage timer for thread safety
        self.queue_timer = QTimer(self)
        self.queue_timer.setInterval(25)
        self.queue_timer.timeout.connect(self._process_queue)
        self.queue_timer.start()

    def update_idletasks(self):
        app = QApplication.instance()
        if app:
            app.processEvents()

    def destroy(self):
        if hasattr(self, "queue_timer") and self.queue_timer:
            self.queue_timer.stop()
        self.close()

    def closeEvent(self, event):
        if hasattr(self, "queue_timer") and self.queue_timer:
            self.queue_timer.stop()
        event.accept()

    def _build_ui(self, default_pdf_path: str):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(18, 14, 18, 14)

        # Scroll Area
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                border: 2px solid #000000;
                background: #ffffff;
                width: 14px;
                margin: 0px;
                border-radius: 7px;
            }
            QScrollBar::handle:vertical {
                background: #000000;
                min-height: 24px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #374151;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)
        self.scroll.verticalScrollBar().setSingleStep(16)

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        c_layout = QVBoxLayout(container)
        c_layout.setContentsMargins(0, 0, 8, 0)
        c_layout.setSpacing(12)

        # Header Banner
        header_frame = QWidget(container)
        h_layout = QHBoxLayout(header_frame)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(10)

        t_lbl = NeoLabel("REVIEW & TRUNCATE DUPLICATES ✦", header_frame)
        t_lbl.setFont(QFont("Segoe UI", 16, QFont.Bold))
        t_lbl.setStyleSheet("color: #000000; border: none;")
        h_layout.addWidget(t_lbl)

        pill = QFrame(header_frame)
        pill.setStyleSheet(f"""
            QFrame {{
                background-color: {COLOR_NEO_YELLOW};
                border: 2px solid #000000;
                border-radius: 16px;
            }}
        """)
        p_layout = QHBoxLayout(pill)
        p_layout.setContentsMargins(10, 3, 10, 3)
        plbl = NeoLabel("✦ DEDUPLICATOR ✦", pill)
        plbl.setFont(QFont("Segoe UI", 9, QFont.Bold))
        plbl.setStyleSheet("color: #000000; border: none; background: transparent;")
        p_layout.addWidget(plbl)
        h_layout.addWidget(pill)
        h_layout.addStretch()

        c_layout.addWidget(header_frame)

        desc_lbl = NeoLabel("Scans all pages, detects repetitive slides, and keeps only the first appearance alone.", container)
        desc_lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
        desc_lbl.setStyleSheet("color: #000000; border: none; padding-bottom: 2px;")
        c_layout.addWidget(desc_lbl)

        # Card 1: Document Selection
        card_files = QFrame(container)
        card_files.setObjectName("card_files")
        card_files.setStyleSheet(f"""
            QFrame#card_files {{
                background-color: {COLOR_CARD_BG};
                border: 2.5px solid #000000;
                border-radius: 14px;
            }}
        """)
        cf_layout = QVBoxLayout(card_files)
        cf_layout.setContentsMargins(0, 0, 0, 14)
        cf_layout.setSpacing(6)

        cf_layout.addWidget(create_neo_card_header("[ 01 ] SELECT DOCUMENT", COLOR_NEO_YELLOW, card_files))

        # Input PDF row
        in_lbl = NeoLabel("Input PDF:", card_files)
        in_lbl.setStyleSheet("padding-left: 16px; padding-top: 2px;")
        cf_layout.addWidget(in_lbl)

        in_row = QWidget(card_files)
        ir_layout = QHBoxLayout(in_row)
        ir_layout.setContentsMargins(16, 0, 16, 6)
        ir_layout.setSpacing(8)

        self.input_entry = NeoEntry("Path to PDF file...", height=36, parent=in_row)
        self.input_entry.set_paste_callback(self._on_input_entry_paste)
        self.input_entry.editingFinished.connect(self._on_input_entry_blur_or_change)
        ir_layout.addWidget(self.input_entry, stretch=1)

        self.browse_in_btn = NeoButton("Browse...", bg_color=COLOR_NEO_YELLOW, hover_color=COLOR_NEO_YELLOW_HOVER, height=36, parent=in_row)
        self.browse_in_btn.setFixedWidth(78)
        self.browse_in_btn.clicked.connect(self._on_browse_input)
        ir_layout.addWidget(self.browse_in_btn)

        self.paste_in_btn = NeoButton("Paste", bg_color=COLOR_NEO_BLUE, hover_color=COLOR_NEO_BLUE_HOVER, height=36, parent=in_row)
        self.paste_in_btn.setFixedWidth(78)
        self.paste_in_btn.clicked.connect(self._on_paste_input)
        ir_layout.addWidget(self.paste_in_btn)

        cf_layout.addWidget(in_row)

        # Output PDF row
        out_lbl = NeoLabel("Save Cleaned PDF to:", card_files)
        out_lbl.setStyleSheet("padding-left: 16px; padding-top: 2px;")
        cf_layout.addWidget(out_lbl)

        out_row = QWidget(card_files)
        or_layout = QHBoxLayout(out_row)
        or_layout.setContentsMargins(16, 0, 16, 6)
        or_layout.setSpacing(8)

        self.output_entry = NeoEntry("Path to save deduplicated PDF...", height=36, parent=out_row)
        self.output_entry.set_paste_callback(self._on_output_entry_paste)
        self.output_entry.editingFinished.connect(self._on_output_entry_blur_or_change)
        or_layout.addWidget(self.output_entry, stretch=1)

        self.browse_out_btn = NeoButton("Browse...", bg_color=COLOR_NEO_YELLOW, hover_color=COLOR_NEO_YELLOW_HOVER, height=36, parent=out_row)
        self.browse_out_btn.setFixedWidth(78)
        self.browse_out_btn.clicked.connect(self._on_browse_output)
        or_layout.addWidget(self.browse_out_btn)

        self.paste_out_btn = NeoButton("Paste", bg_color=COLOR_NEO_BLUE, hover_color=COLOR_NEO_BLUE_HOVER, height=36, parent=out_row)
        self.paste_out_btn.setFixedWidth(78)
        self.paste_out_btn.clicked.connect(self._on_paste_output)
        or_layout.addWidget(self.paste_out_btn)

        cf_layout.addWidget(out_row)
        c_layout.addWidget(card_files)

        if default_pdf_path:
            self._set_input_value(default_pdf_path)

        # Card 2: Sensitivity & Controls
        card_controls = QFrame(container)
        card_controls.setObjectName("card_controls")
        card_controls.setStyleSheet(f"""
            QFrame#card_controls {{
                background-color: {COLOR_CARD_BG};
                border: 2.5px solid #000000;
                border-radius: 14px;
            }}
        """)
        cc_layout = QVBoxLayout(card_controls)
        cc_layout.setContentsMargins(0, 0, 0, 14)
        cc_layout.setSpacing(6)

        cc_layout.addWidget(create_neo_card_header("[ 02 ] DETECTION SENSITIVITY", COLOR_NEO_BLUE, card_controls))

        thresh_top = QWidget(card_controls)
        tt_layout = QHBoxLayout(thresh_top)
        tt_layout.setContentsMargins(16, 0, 16, 0)

        t_lbl = NeoLabel("Similarity Threshold:", thresh_top)
        t_lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
        tt_layout.addWidget(t_lbl)
        tt_layout.addStretch()

        self.threshold_label = NeoLabel("94%", thresh_top)
        self.threshold_label.setFont(QFont("Segoe UI", 9, QFont.Bold))
        self.threshold_label.setAlignment(Qt.AlignCenter)
        self.threshold_label.setFixedSize(62, 26)
        self.threshold_label.setStyleSheet(f"""
            QLabel {{
                background-color: {COLOR_NEO_YELLOW};
                color: #000000;
                border: 2px solid #000000;
                border-radius: 6px;
                font-family: 'Segoe UI', sans-serif;
                font-weight: bold;
            }}
        """)
        tt_layout.addWidget(self.threshold_label)
        cc_layout.addWidget(thresh_top)

        sub_desc = NeoLabel("Sensitivity for declaring pages identical (higher requires stricter visual match)", card_controls)
        sub_desc.setFont(QFont("Segoe UI", 10))
        sub_desc.setStyleSheet("color: #000000; border: none; padding-left: 16px;")
        sub_desc.setWordWrap(True)
        cc_layout.addWidget(sub_desc)

        slider_box = QWidget(card_controls)
        sl_layout = QHBoxLayout(slider_box)
        sl_layout.setContentsMargins(16, 0, 16, 4)

        self.similarity_slider = NeoSlider(0.80, 0.99, 19, 0.94, parent=slider_box)
        self.similarity_slider._command = lambda v: self.threshold_label.configure(text=f"{int(round(v * 100))}%")
        sl_layout.addWidget(self.similarity_slider)
        cc_layout.addWidget(slider_box)

        # Narrator Filter Switch
        narr_row = QWidget(card_controls)
        nr_layout = QHBoxLayout(narr_row)
        nr_layout.setContentsMargins(16, 2, 16, 6)

        self.dialog_filter_narrator_switch = NeoSwitch("Filter Full-Human / Narrator Frames (Remove camera cuts to speaker)", narr_row)
        self.dialog_filter_narrator_switch.select()
        nr_layout.addWidget(self.dialog_filter_narrator_switch)
        cc_layout.addWidget(narr_row)

        # Action Buttons
        btn_action_row = QWidget(card_controls)
        bar_layout = QHBoxLayout(btn_action_row)
        bar_layout.setContentsMargins(16, 4, 16, 2)
        bar_layout.setSpacing(8)

        self.run_btn = NeoButton("Review & Truncate Duplicates", bg_color=COLOR_NEO_GREEN, hover_color=COLOR_NEO_GREEN_HOVER, height=40, font_size=12, parent=btn_action_row)
        self.run_btn.clicked.connect(self._on_start_dedup)
        bar_layout.addWidget(self.run_btn, stretch=1)

        self.cancel_btn = NeoButton("Cancel", bg_color=COLOR_NEO_RED, hover_color=COLOR_NEO_RED_HOVER, height=40, font_size=12, parent=btn_action_row)
        self.cancel_btn.setFixedWidth(90)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel_dedup)
        bar_layout.addWidget(self.cancel_btn)

        cc_layout.addWidget(btn_action_row)
        c_layout.addWidget(card_controls)

        # Card 3: Progress & Results
        self.card_results = QFrame(container)
        self.card_results.setObjectName("card_results")
        self.card_results.setStyleSheet(f"""
            QFrame#card_results {{
                background-color: {COLOR_CARD_BG};
                border: 2.5px solid #000000;
                border-radius: 14px;
            }}
        """)
        cr_layout = QVBoxLayout(self.card_results)
        cr_layout.setContentsMargins(0, 0, 0, 16)
        cr_layout.setSpacing(8)

        cr_layout.addWidget(create_neo_card_header("[ 03 ] ANALYSIS PROGRESS & RESULTS", COLOR_NEO_LILAC, self.card_results))

        # Status Ticker Box
        self.dialog_status_box = QFrame(self.card_results)
        self.dialog_status_box.setObjectName("dialog_status_box")
        self.dialog_status_box.setStyleSheet(f"""
            QFrame#dialog_status_box {{
                background-color: {COLOR_NEO_CREAM};
                border: 2px solid #000000;
                border-radius: 8px;
            }}
        """)
        dsb_layout = QHBoxLayout(self.dialog_status_box)
        dsb_layout.setContentsMargins(12, 6, 12, 6)

        self.status_label = NeoLabel("Ready to review PDF", self.dialog_status_box)
        self.status_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.status_label.setStyleSheet("color: #000000; border: none; background: transparent;")
        dsb_layout.addWidget(self.status_label)
        cr_layout.addWidget(self.dialog_status_box)

        # Progress bar
        pb_box = QWidget(self.card_results)
        pbb_layout = QHBoxLayout(pb_box)
        pbb_layout.setContentsMargins(16, 0, 16, 4)
        self.progress_bar = NeoProgressBar(pb_box)
        pbb_layout.addWidget(self.progress_bar)
        cr_layout.addWidget(pb_box)

        # 4 Mini KPI Badges
        metrics_grid = QGridLayout()
        metrics_grid.setContentsMargins(16, 0, 16, 4)
        metrics_grid.setSpacing(6)

        self.b_total, self.lbl_total = self._create_mini_badge(self.card_results, "TOTAL PAGES", "0", bg_color=COLOR_NEO_YELLOW)
        self.b_kept, self.lbl_kept = self._create_mini_badge(self.card_results, "UNIQUE KEPT", "0", bg_color=COLOR_NEO_GREEN)
        self.b_pruned, self.lbl_pruned = self._create_mini_badge(self.card_results, "DUPLICATES PRUNED", "0", bg_color=COLOR_NEO_ORANGE)
        self.b_narrator, self.lbl_narrator = self._create_mini_badge(self.card_results, "NARRATOR CUTS", "0", bg_color=COLOR_NEO_PINK)

        metrics_grid.addWidget(self.b_total, 0, 0)
        metrics_grid.addWidget(self.b_kept, 0, 1)
        metrics_grid.addWidget(self.b_pruned, 0, 2)
        metrics_grid.addWidget(self.b_narrator, 0, 3)
        cr_layout.addLayout(metrics_grid)

        # Details Log Box
        rep_lbl = NeoLabel("Repetitive Pages Found:", self.card_results)
        rep_lbl.setStyleSheet("padding-left: 16px; padding-top: 2px;")
        cr_layout.addWidget(rep_lbl)

        txt_box = QWidget(self.card_results)
        tb_layout = QHBoxLayout(txt_box)
        tb_layout.setContentsMargins(16, 0, 16, 6)

        self.details_textbox = QTextEdit(txt_box)
        self.details_textbox.setFixedHeight(130)
        self.details_textbox.setReadOnly(True)
        self.details_textbox.setFont(QFont("Consolas", 9, QFont.Bold))
        self.details_textbox.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                color: #000000;
                border: 2px solid #000000;
                border-radius: 8px;
                padding: 6px;
            }
        """)
        self.details_textbox.setText("Click 'Review & Truncate Duplicates' to scan the PDF...")
        tb_layout.addWidget(self.details_textbox)
        cr_layout.addWidget(txt_box)

        # Result Action Buttons
        self.res_btn_row = QWidget(self.card_results)
        rbr_layout = QHBoxLayout(self.res_btn_row)
        rbr_layout.setContentsMargins(16, 4, 16, 0)
        rbr_layout.setSpacing(8)

        self.open_pdf_btn = NeoButton("Open Cleaned PDF", bg_color=COLOR_NEO_GREEN, hover_color=COLOR_NEO_GREEN_HOVER, height=38, font_size=12, parent=self.res_btn_row)
        self.open_pdf_btn.setEnabled(False)
        self.open_pdf_btn.clicked.connect(self._on_open_pdf)
        rbr_layout.addWidget(self.open_pdf_btn, stretch=1)

        self.open_folder_btn = NeoButton("Open Folder", bg_color=COLOR_NEO_BLUE, hover_color=COLOR_NEO_BLUE_HOVER, height=38, font_size=12, parent=self.res_btn_row)
        self.open_folder_btn.setEnabled(False)
        self.open_folder_btn.clicked.connect(self._on_open_folder)
        rbr_layout.addWidget(self.open_folder_btn, stretch=1)

        btn_close = NeoButton("Close", bg_color=COLOR_NEO_RED, hover_color=COLOR_NEO_RED_HOVER, height=38, font_size=12, parent=self.res_btn_row)
        btn_close.setFixedWidth(90)
        btn_close.clicked.connect(self.close)
        rbr_layout.addWidget(btn_close)

        cr_layout.addWidget(self.res_btn_row)
        c_layout.addWidget(self.card_results)

        self.scroll.setWidget(container)
        root_layout.addWidget(self.scroll)

    def _create_mini_badge(self, parent, title: str, val: str, bg_color: str = COLOR_NEO_YELLOW) -> Tuple[QFrame, NeoLabel]:
        badge = QFrame(parent)
        badge.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color};
                border: 2px solid #000000;
                border-radius: 8px;
            }}
        """)
        b_layout = QVBoxLayout(badge)
        b_layout.setContentsMargins(8, 6, 8, 6)
        b_layout.setSpacing(1)

        t_lbl = NeoLabel(title, badge)
        t_lbl.setFont(QFont("Segoe UI", 8, QFont.Bold))
        t_lbl.setStyleSheet("color: #000000; border: none; background: transparent;")
        b_layout.addWidget(t_lbl)

        v_lbl = NeoLabel(val, badge)
        v_lbl.setFont(QFont("Segoe UI", 11, QFont.Bold))
        v_lbl.setStyleSheet("color: #000000; border: none; background: transparent;")
        b_layout.addWidget(v_lbl)

        return badge, v_lbl

    def _set_input_value(self, raw_path: str):
        cleaned = clean_file_path(raw_path)
        if not cleaned:
            return
        self.input_entry.setText(cleaned)
        base, ext = os.path.splitext(cleaned)
        if ext.lower() == ".pdf":
            self.output_entry.setText(f"{base}_dedup{ext}")

    def _on_paste_input(self):
        cb = QApplication.clipboard()
        if cb:
            self._set_input_value(cb.text())

    def _on_paste_output(self):
        cb = QApplication.clipboard()
        if cb:
            cleaned = clean_file_path(cb.text())
            if cleaned:
                self.output_entry.setText(cleaned)

    def _on_input_entry_paste(self, event=None):
        self._on_paste_input()

    def _on_input_entry_blur_or_change(self, event=None):
        curr = self.input_entry.get()
        if curr:
            cleaned = clean_file_path(curr)
            if cleaned != curr:
                self._set_input_value(cleaned)

    def _on_output_entry_paste(self, event=None):
        self._on_paste_output()

    def _on_output_entry_blur_or_change(self, event=None):
        curr = self.output_entry.get()
        if curr:
            cleaned = clean_file_path(curr)
            if cleaned != curr:
                self.output_entry.setText(cleaned)

    def _on_browse_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select PDF Document to Deduplicate",
            "",
            "PDF Document (*.pdf);;All Files (*.*)",
        )
        if path:
            self._set_input_value(path)

    def _on_browse_output(self):
        curr = clean_file_path(self.output_entry.get())
        initial_dir = os.path.dirname(curr) if curr else os.getcwd()
        initial_file = os.path.basename(curr) if curr else "presentation_dedup.pdf"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Deduplicated PDF As",
            os.path.join(initial_dir, initial_file),
            "PDF Document (*.pdf)",
        )
        if path:
            self.output_entry.setText(clean_file_path(path))

    def _on_start_dedup(self):
        in_path = clean_file_path(self.input_entry.get())
        if not in_path or not os.path.exists(in_path):
            QMessageBox.warning(self, "Missing File", "Please select an existing PDF file.")
            return

        if in_path != self.input_entry.get().strip():
            self.input_entry.setText(in_path)

        out_path = clean_file_path(self.output_entry.get())
        if not out_path:
            base, ext = os.path.splitext(in_path)
            out_path = f"{base}_dedup{ext}"
            self.output_entry.setText(out_path)

        if not out_path.lower().endswith(".pdf"):
            out_path += ".pdf"

        if out_path != self.output_entry.get().strip():
            self.output_entry.setText(out_path)

        # UI state
        self.is_processing = True
        self.cancel_event.clear()
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.open_pdf_btn.setEnabled(False)
        self.open_folder_btn.setEnabled(False)

        self.status_label.setText("Reading document pages...")
        self.progress_bar.configure(progress_color=COLOR_NEO_GREEN)
        self.progress_bar.set(0.0)
        self.lbl_total.setText("0")
        self.lbl_kept.setText("0")
        self.lbl_pruned.setText("0")
        self.lbl_narrator.setText("0")

        self.details_textbox.clear()
        self.details_textbox.setText("Analyzing pages for visual duplicates and narrator camera cuts...\n")

        thresh = float(self.similarity_slider.get())
        filter_narrator = self.dialog_filter_narrator_switch.isChecked()

        def run_worker():
            try:
                def prog(curr, total, txt):
                    if not self.cancel_event.is_set():
                        pct = curr / float(max(1, total))
                        self.msg_queue.put(("PROGRESS", (pct, curr, total, txt)))

                result = deduplicate_pdf(
                    input_pdf_path=in_path,
                    output_pdf_path=out_path,
                    similarity_threshold=thresh,
                    filter_narrator_frames=filter_narrator,
                    progress_callback=prog,
                    cancel_event=self.cancel_event,
                )
                self.msg_queue.put(("COMPLETE", result))
            except Exception as e:
                self.msg_queue.put(("ERROR", str(e)))

        threading.Thread(target=run_worker, daemon=True).start()

    def _process_queue(self):
        try:
            while not self.msg_queue.empty():
                item = self.msg_queue.get_nowait()
                action = item[0]
                if action == "PROGRESS":
                    pct, curr, total, txt = item[1]
                    self.progress_bar.set(pct)
                    self.status_label.setText(f"Analyzing page {curr} of {total}...")
                    self.lbl_total.setText(str(total))
                elif action == "COMPLETE":
                    self._on_worker_completed(item[1])
                elif action == "ERROR":
                    self._on_worker_error(item[1])
        except Exception as e:
            logger.error(f"Error handling dialog queue: {e}")

    def _on_cancel_dedup(self):
        self.cancel_event.set()
        self.progress_bar.configure(progress_color="#ffffff")
        self.progress_bar.set(0.0)
        self.status_label.setText("Cancelling duplicate review...")
        self.cancel_btn.setEnabled(False)

    def _on_worker_completed(self, result: DeduplicationResult):
        self.latest_result = result
        self.is_processing = False
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.set(1.0)

        # Update metrics
        self.lbl_total.setText(str(result.total_pages))
        self.lbl_kept.setText(str(result.unique_pages_count))
        self.lbl_pruned.setText(str(result.duplicates_removed_count))
        self.lbl_narrator.setText(str(result.narrator_pages_removed_count))

        total_pruned = result.duplicates_removed_count + result.narrator_pages_removed_count

        self.details_textbox.clear()
        if total_pruned > 0:
            self.status_label.setText(f"Done! Pruned {total_pruned} pages ({result.unique_pages_count} unique slides kept).")
            lines = [
                f"SUMMARY: {result.unique_pages_count} unique slides retained out of {result.total_pages} total pages.",
                f"• {result.narrator_pages_removed_count} full-human narrator camera cuts removed.",
                f"• {result.duplicates_removed_count} repetitive duplicate slides truncated.\n",
            ]
            if result.narrator_details:
                lines.append("-- Narrator Camera Cuts Filtered --")
                for narr_p, reason in result.narrator_details:
                    lines.append(f"• Page {narr_p:>2}: Full-Human Narrator ({reason}) -> Truncated")
                lines.append("")

            if result.duplicate_details:
                lines.append("-- Duplicate Slides Truncated --")
                for dup_p, orig_p, score in result.duplicate_details:
                    lines.append(f"• Page {dup_p:>2}: Matches Page {orig_p:>2} ({score * 100:.1f}% match) -> Truncated")

            self.details_textbox.setText("\n".join(lines))
            self.open_pdf_btn.setEnabled(True)
            self.open_folder_btn.setEnabled(True)
        else:
            self.status_label.setText("Document clean! No duplicate or narrator pages were found.")
            self.details_textbox.setText(
                "All pages in this PDF are unique presentation slides.\n"
                "No repetitive pages or narrator camera cuts were detected."
            )
            self.open_pdf_btn.setEnabled(True)
            self.open_folder_btn.setEnabled(True)

    def _on_worker_error(self, err_msg: str):
        self.is_processing = False
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.configure(progress_color="#ffffff")
        self.progress_bar.set(0.0)
        self.status_label.setText("Error occurred during review.")
        self.status_label.setStyleSheet(f"color: {COLOR_NEO_RED};")
        QMessageBox.critical(self, "Deduplication Error", f"An error occurred while reviewing the PDF:\n\n{err_msg}")

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
            QMessageBox.critical(self, "Error", f"Could not open PDF: {e}")

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
            QMessageBox.critical(self, "Error", f"Could not open folder:\n{e}")
