"""In-memory Back-End Console and structured logging manager for DeckSpool.

Provides an OS-style terminal console window displaying live, structured telemetry
and session history (log levels, detection events, timestamps) with temporary in-memory
retention that erases upon session termination.
"""

from __future__ import annotations

import sys
import time
import html
import logging
import threading
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QPushButton,
    QLabel,
    QCheckBox,
    QApplication,
)
from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtGui import QFont, QTextCursor, QIcon, QKeySequence

from deckspool.font_manager import font_manager


# ==============================================================================
# In-Memory Session Log Storage
# ==============================================================================

@dataclass
class LogEntry:
    timestamp: str
    level: str
    source: str
    message: str
    is_raw: bool = False


class SessionLogBuffer:
    """Thread-safe temporary in-memory buffer storing log entries for the active session."""

    def __init__(self, max_entries: int = 5000):
        self._max_entries = max_entries
        self._entries: List[LogEntry] = []
        self._lock = threading.Lock()

    def add_entry(self, entry: LogEntry):
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max_entries:
                self._entries.pop(0)

    def get_entries(self) -> List[LogEntry]:
        with self._lock:
            return list(self._entries)

    def clear(self):
        with self._lock:
            self._entries.clear()


# ==============================================================================
# Qt Log Handler and Stream Interceptor
# ==============================================================================

class ConsoleSignaler(QObject):
    """Qt Object bridging thread-safe logging signals to the GUI thread."""
    new_entry = Signal(object)  # Emits LogEntry


class QtConsoleLogHandler(logging.Handler):
    """Logging handler pushing formatted records into the session buffer and GUI."""

    def __init__(self, buffer: SessionLogBuffer, signaler: ConsoleSignaler):
        super().__init__()
        self.buffer = buffer
        self.signaler = signaler
        self.setFormatter(logging.Formatter("%(asctime)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord):
        try:
            timestamp = self.format(record)
            level = record.levelname.upper()
            source = record.name
            msg = record.getMessage()

            entry = LogEntry(
                timestamp=timestamp,
                level=level,
                source=source,
                message=msg,
                is_raw=False,
            )
            self.buffer.add_entry(entry)
            self.signaler.new_entry.emit(entry)

            # Mirror to raw OS console if present
            if sys.__stdout__:
                try:
                    sys.__stdout__.write(f"{timestamp} [{level}] {source}: {msg}\n")
                    sys.__stdout__.flush()
                except Exception:
                    pass
        except Exception:
            self.handleError(record)


class _BinaryStreamBuffer:
    """Binary I/O buffer interface compatible with TextIO.buffer."""

    def __init__(self, parent: "StdStreamInterceptor"):
        self._parent = parent

    def write(self, data) -> int:
        if isinstance(data, (bytes, bytearray, memoryview)):
            enc = getattr(self._parent, "encoding", "utf-8") or "utf-8"
            text = bytes(data).decode(enc, errors="replace")
        else:
            text = str(data)
        self._parent.write(text)
        return len(data)

    def flush(self):
        self._parent.flush()

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    def writelines(self, lines):
        for line in lines:
            self.write(line)


class StdStreamInterceptor:
    """Redirects stdout and stderr to capture terminal messages and forward them."""

    def __init__(self, original_stream, stream_name: str, session_buffer: SessionLogBuffer, signaler: ConsoleSignaler):
        self.original_stream = original_stream
        self.stream_name = stream_name
        self.session_buffer = session_buffer
        self.signaler = signaler
        self._line_buffer = ""
        self._lock = threading.Lock()
        self.buffer = _BinaryStreamBuffer(self)

    @property
    def encoding(self) -> str:
        return getattr(self.original_stream, "encoding", "utf-8") or "utf-8"

    @property
    def errors(self) -> str:
        return getattr(self.original_stream, "errors", "replace") or "replace"

    @property
    def closed(self) -> bool:
        return getattr(self.original_stream, "closed", False)

    def isatty(self) -> bool:
        if self.original_stream and hasattr(self.original_stream, "isatty"):
            try:
                return self.original_stream.isatty()
            except Exception:
                return False
        return False

    def fileno(self):
        if self.original_stream and hasattr(self.original_stream, "fileno"):
            return self.original_stream.fileno()
        raise OSError("fileno not supported")

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def write(self, text: str) -> int:
        if not text:
            return 0

        if self.original_stream:
            try:
                self.original_stream.write(text)
                self.original_stream.flush()
            except UnicodeEncodeError:
                try:
                    enc = getattr(self.original_stream, "encoding", "utf-8") or "utf-8"
                    if hasattr(self.original_stream, "buffer"):
                        self.original_stream.buffer.write(text.encode(enc, errors="replace"))
                        self.original_stream.buffer.flush()
                except Exception:
                    pass
            except Exception:
                pass

        with self._lock:
            self._line_buffer += text
            while "\n" in self._line_buffer:
                line, self._line_buffer = self._line_buffer.split("\n", 1)
                line = line.strip("\r")
                if line.strip():
                    timestamp = time.strftime("%H:%M:%S")
                    entry = LogEntry(
                        timestamp=timestamp,
                        level=self.stream_name,
                        source="sys",
                        message=line,
                        is_raw=True,
                    )
                    self.session_buffer.add_entry(entry)
                    self.signaler.new_entry.emit(entry)

        return len(text)

    def flush(self):
        with self._lock:
            if self._line_buffer.strip("\r\n "):
                timestamp = time.strftime("%H:%M:%S")
                entry = LogEntry(
                    timestamp=timestamp,
                    level=self.stream_name,
                    source="sys",
                    message=self._line_buffer.strip("\r"),
                    is_raw=True,
                )
                self.session_buffer.add_entry(entry)
                self.signaler.new_entry.emit(entry)
            self._line_buffer = ""

        if self.original_stream:
            try:
                self.original_stream.flush()
            except Exception:
                pass


class BackendConsoleManager:
    """Singleton managing session logging, interception, and console window state."""

    _instance: Optional[BackendConsoleManager] = None

    def __new__(cls) -> BackendConsoleManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.buffer = SessionLogBuffer()
        self.signaler = ConsoleSignaler()
        self._window: Optional[BackEndConsoleWindow] = None
        self._setup_logging()

    def _setup_logging(self):
        # Attach Qt log handler to root logger
        self.handler = QtConsoleLogHandler(self.buffer, self.signaler)
        self.handler.setLevel(logging.DEBUG)
        root = logging.getLogger()
        if root.level > logging.DEBUG:
            root.setLevel(logging.DEBUG)
        root.addHandler(self.handler)

        # Intercept stdout and stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.stdout_interceptor = StdStreamInterceptor(self.original_stdout, "STDOUT", self.buffer, self.signaler)
        self.stderr_interceptor = StdStreamInterceptor(self.original_stderr, "STDERR", self.buffer, self.signaler)
        sys.stdout = self.stdout_interceptor
        sys.stderr = self.stderr_interceptor

    def get_window(self, parent=None) -> BackEndConsoleWindow:
        if self._window is None:
            self._window = BackEndConsoleWindow(self.buffer, self.signaler, parent=parent)
        return self._window

    def cleanup(self):
        """Restores original streams and terminates log handler."""
        if hasattr(self, "original_stdout") and self.original_stdout:
            sys.stdout = self.original_stdout
        if hasattr(self, "original_stderr") and self.original_stderr:
            sys.stderr = self.original_stderr

        root = logging.getLogger()
        if hasattr(self, "handler") and self.handler in root.handlers:
            root.removeHandler(self.handler)

        if self._window:
            self._window.close()
            self._window = None


# Global console manager singleton
backend_console_manager = BackendConsoleManager()


# ==============================================================================
# Plain OS-Style Terminal Console Window
# ==============================================================================

class BackEndConsoleWindow(QMainWindow):
    """Plain OS-Style Terminal Console Window displaying structured backend execution logs."""

    def __init__(self, buffer: SessionLogBuffer, signaler: ConsoleSignaler, parent=None):
        super().__init__(parent, Qt.Window)
        self.buffer = buffer
        self.signaler = signaler
        self._visibility_callback: Optional[Callable[[bool], None]] = None
        self._auto_scroll = True

        self.setWindowTitle("DeckSpool — Back-End Terminal Console")
        self.resize(720, 450)
        self.setMinimumSize(520, 320)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0c0c0e;
            }
        """)

        self._build_ui()
        self._connect_signals()
        self._populate_history()

    def set_visibility_callback(self, callback: Callable[[bool], None]):
        self._visibility_callback = callback

    def _build_ui(self):
        root = QWidget(self)
        root.setStyleSheet("background-color: #0c0c0e;")
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        # Top Control Bar (Clean, Minimal, OS Terminal style)
        bar = QWidget(root)
        b_layout = QHBoxLayout(bar)
        b_layout.setContentsMargins(0, 0, 0, 0)
        b_layout.setSpacing(8)

        # Terminal status dot & title
        self.status_lbl = QLabel("● SESSION CONSOLE", bar)
        if font_manager.is_loaded:
            self.status_lbl.setFont(font_manager.create_qfont("code", size=9, weight=QFont.Bold))
        else:
            self.status_lbl.setFont(QFont("Consolas", 9, QFont.Bold))
        self.status_lbl.setStyleSheet("color: #4ade80; background: transparent; border: none;")
        b_layout.addWidget(self.status_lbl)

        b_layout.addStretch()

        # Auto-scroll checkbox
        self.autoscroll_cb = QCheckBox("Auto-scroll", bar)
        self.autoscroll_cb.setChecked(True)
        if font_manager.is_loaded:
            self.autoscroll_cb.setFont(font_manager.create_qfont("body", size=9, weight=QFont.Normal))
        else:
            self.autoscroll_cb.setFont(QFont("Segoe UI", 9))
        self.autoscroll_cb.setStyleSheet("""
            QCheckBox {
                color: #94a3b8;
                spacing: 4px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #475569;
                background-color: #1e293b;
            }
            QCheckBox::indicator:checked {
                background-color: #38bdf8;
                border: 1px solid #38bdf8;
            }
        """)
        self.autoscroll_cb.toggled.connect(self._on_autoscroll_toggled)
        b_layout.addWidget(self.autoscroll_cb)

        # Copy Output Button
        self.copy_btn = QPushButton("Copy All", bar)
        self.copy_btn.setFixedHeight(24)
        if font_manager.is_loaded:
            self.copy_btn.setFont(font_manager.create_qfont("code", size=8, weight=QFont.Bold))
        else:
            self.copy_btn.setFont(QFont("Consolas", 8, QFont.Bold))
        self.copy_btn.setStyleSheet("""
            QPushButton {
                background-color: #1e293b;
                color: #e2e8f0;
                border: 1px solid #334155;
                padding: 2px 10px;
                border-radius: 2px;
            }
            QPushButton:hover {
                background-color: #334155;
                color: #ffffff;
            }
        """)
        self.copy_btn.clicked.connect(self._on_copy_all)
        b_layout.addWidget(self.copy_btn)

        # Clear Session Button
        self.clear_btn = QPushButton("Clear", bar)
        self.clear_btn.setFixedHeight(24)
        if font_manager.is_loaded:
            self.clear_btn.setFont(font_manager.create_qfont("code", size=8, weight=QFont.Bold))
        else:
            self.clear_btn.setFont(QFont("Consolas", 8, QFont.Bold))
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #1e293b;
                color: #f87171;
                border: 1px solid #334155;
                padding: 2px 10px;
                border-radius: 2px;
            }
            QPushButton:hover {
                background-color: #334155;
                color: #ef4444;
            }
        """)
        self.clear_btn.clicked.connect(self._on_clear)
        b_layout.addWidget(self.clear_btn)

        layout.addWidget(bar)

        # Raw OS Console Log Text Area
        self.console_text = QTextEdit(root)
        self.console_text.setReadOnly(True)
        if font_manager.is_loaded:
            self.console_text.setFont(font_manager.create_qfont("code", size=10, weight=QFont.Medium))
        else:
            self.console_text.setFont(QFont("Consolas", 10))

        self.console_text.setStyleSheet("""
            QTextEdit {
                background-color: #0c0c0e;
                color: #e2e8f0;
                border: 1.5px solid #1e293b;
                border-radius: 0px;
                padding: 8px;
                selection-background-color: #3b82f6;
                selection-color: #ffffff;
            }
            QScrollBar:vertical {
                border: none;
                background: #0c0c0e;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #334155;
                min-height: 20px;
                border-radius: 2px;
            }
            QScrollBar::handle:vertical:hover {
                background: #475569;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        layout.addWidget(self.console_text)

    def _connect_signals(self):
        self.signaler.new_entry.connect(self._append_entry)

    def _format_entry_html(self, entry: LogEntry) -> str:
        """Formats a structured log record into color-coded terminal HTML."""
        t_str = html.escape(entry.timestamp)
        src_str = html.escape(entry.source)
        msg_str = html.escape(entry.message)

        # Structured level badge colors
        lvl = entry.level.upper()
        if lvl in ("INFO",):
            badge = f'<b style="color: #4ade80;">[{lvl}]</b>'
        elif lvl in ("WARNING", "WARN"):
            badge = f'<b style="color: #facc15;">[{lvl}]</b>'
        elif lvl in ("ERROR", "CRITICAL", "STDERR"):
            badge = f'<b style="color: #f87171;">[{lvl}]</b>'
        elif lvl in ("DEBUG",):
            badge = f'<b style="color: #c084fc;">[{lvl}]</b>'
        elif lvl in ("STDOUT",):
            badge = f'<b style="color: #38bdf8;">[{lvl}]</b>'
        else:
            badge = f'<b style="color: #94a3b8;">[{lvl}]</b>'

        # Highlight detection keywords
        if "Committed Slide" in msg_str:
            msg_str = f'<span style="color: #67e8f9; font-weight: bold;">{msg_str}</span>'
        elif "narrator" in msg_str.lower() or "human" in msg_str.lower():
            msg_str = f'<span style="color: #f472b6;">{msg_str}</span>'
        elif "complete" in msg_str.lower() or "successfully" in msg_str.lower():
            msg_str = f'<span style="color: #86efac; font-weight: bold;">{msg_str}</span>'

        return (
            f'<div style="line-height: 130%; font-family: monospace;">'
            f'<span style="color: #64748b;">{t_str}</span> '
            f'{badge} '
            f'<span style="color: #64748b;">{src_str}:</span> '
            f'<span style="color: #e2e8f0;">{msg_str}</span>'
            f'</div>'
        )

    def _populate_history(self):
        entries = self.buffer.get_entries()
        if not entries:
            self.console_text.setHtml(
                '<div style="color: #64748b; font-family: monospace;">'
                '[DeckSpool Back-End Console Initialized — Ready for Extraction Session]'
                '</div>'
            )
            return

        html_lines = [self._format_entry_html(e) for e in entries]
        self.console_text.setHtml("".join(html_lines))
        self._scroll_to_bottom()

    def _append_entry(self, entry: LogEntry):
        html_line = self._format_entry_html(entry)
        self.console_text.append(html_line)
        if self._auto_scroll:
            self._scroll_to_bottom()

    def _scroll_to_bottom(self):
        sb = self.console_text.verticalScrollBar()
        if sb:
            sb.setValue(sb.maximum())

    def _on_autoscroll_toggled(self, checked: bool):
        self._auto_scroll = checked
        if checked:
            self._scroll_to_bottom()

    def _on_copy_all(self):
        cb = QApplication.clipboard()
        if cb:
            cb.setText(self.console_text.toPlainText())

    def _on_clear(self):
        self.buffer.clear()
        self.console_text.setHtml(
            '<div style="color: #64748b; font-family: monospace;">'
            '[Console cleared by user — new logs will appear below]'
            '</div>'
        )

    def toggle_visibility(self) -> bool:
        """Toggles the window visibility and returns True if now visible, False if hidden."""
        if self.isVisible():
            self.hide()
            if self._visibility_callback:
                self._visibility_callback(False)
            return False
        else:
            self.show()
            self.raise_()
            self.activateWindow()
            self._scroll_to_bottom()
            if self._visibility_callback:
                self._visibility_callback(True)
            return True

    def closeEvent(self, event):
        """Intercepts native window close [X] to hide instead of destroy, preserving state."""
        event.ignore()
        self.hide()
        if self._visibility_callback:
            self._visibility_callback(False)

    def close_and_cleanup(self):
        """Real destruction method invoked when the parent application exits."""
        self.hide()
        self.deleteLater()
