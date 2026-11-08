# Copyright (c) 2026 Vignesh Rajan (vector49)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Video ingestion and metadata extraction using yt-dlp and OpenCV."""

import os
import re
import tempfile
import logging
import threading
from typing import Optional, Dict, Any, Callable
import cv2
import yt_dlp

logger = logging.getLogger(__name__)


class VideoIngestionError(Exception):
    """Raised when video ingestion or download fails."""
    pass


class DownloadCancelledError(Exception):
    """Raised when user cancels download."""
    pass


def sanitize_filename(name: str) -> str:
    """Sanitizes a string to make it safe for filesystem filenames."""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = name.strip()
    return name if name else "presentation_slides"


def clean_file_path(path_str: str) -> str:
    """Auto-formats and cleans file paths or URLs copied from File Explorer, browsers, or shells.

    Handles:
    - Surrounding double or single quotes (e.g. '"C:\\Videos\\demo.mp4"')
    - Leading/trailing whitespace
    - PowerShell call syntax (e.g. "& 'C:\\...'")
    - Smart/curly quotes (e.g. “C:\\...”)
    - file:/// URIs (e.g. file:///C:/path/to/video.mp4)
    - Git Bash / WSL paths on Windows (e.g. /c/Users/... -> C:\\Users\\...)
    - Slash normalization on Windows
    """
    if not path_str:
        return ""

    s = str(path_str).strip()
    if not s:
        return ""

    # Strip PowerShell execution operator '&'
    if s.startswith("&"):
        s = s.lstrip("&").strip()

    # Strip matching quotes in a loop (handles doubly wrapped quotes)
    quote_chars = ('"', "'", '`', '“', '”', '‘', '’')
    changed = True
    while changed and len(s) >= 2:
        changed = False
        if s[0] in quote_chars and s[-1] in quote_chars:
            s = s[1:-1].strip()
            changed = True

    # Strip any dangling quotes on boundaries
    s = s.strip('"\'`“”‘’')

    # Handle file:// URIs (e.g. file:///C:/... or file://...)
    if s.lower().startswith("file:///"):
        from urllib.parse import unquote
        s = unquote(s[8:])
    elif s.lower().startswith("file://"):
        from urllib.parse import unquote
        s = unquote(s[7:])

    # Git Bash / MSYS / Cygwin on Windows: /c/Users/... -> C:/Users/...
    if os.name == "nt":
        m_git = re.match(r"^/([a-zA-Z])/(.*)$", s)
        if m_git:
            s = f"{m_git.group(1).upper()}:/{m_git.group(2)}"
        else:
            # WSL: /mnt/c/Users/... -> C:/Users/...
            m_wsl = re.match(r"^/mnt/([a-zA-Z])/(.*)$", s)
            if m_wsl:
                s = f"{m_wsl.group(1).upper()}:/{m_wsl.group(2)}"

    # If it's a Windows drive path (e.g., C:/... or C:\...) or UNC path, normalize slashes
    if re.match(r"^[a-zA-Z]:[/\\]", s) or s.startswith(r"\\"):
        s = os.path.normpath(s)

    return s.strip()


def is_local_file_path(path_or_url: str) -> bool:
    """Returns True if the given string represents a local filesystem path rather than a URL."""
    cleaned = clean_file_path(path_or_url)
    if not cleaned:
        return False
    if cleaned.lower().startswith(("http://", "https://", "www.", "youtube.com", "youtu.be")):
        return False
    if os.path.exists(cleaned):
        return True
    if re.match(r"^[a-zA-Z]:[/\\]", cleaned) or cleaned.startswith(("\\", "/")):
        return True
    video_exts = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v", ".pdf")
    if cleaned.lower().endswith(video_exts):
        return True
    return False


class VideoDownloader:
    """Manages YouTube stream downloading and local file verification."""

    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self._downloaded_file: Optional[str] = None

    def fetch_youtube_metadata(self, url: str) -> Dict[str, Any]:
        """Extracts video metadata (title, duration, thumbnail) without downloading."""
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "skip_download": True,
            "extract_flat": False,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if not info:
                    raise VideoIngestionError("Could not retrieve video information.")
                title = info.get("title", "YouTube Presentation")
                duration = info.get("duration", 0)  # in seconds
                thumbnail = info.get("thumbnail", "")
                sanitized_title = sanitize_filename(title)
                return {
                    "title": title,
                    "sanitized_title": sanitized_title,
                    "duration": duration,
                    "thumbnail": thumbnail,
                    "default_pdf_name": f"{sanitized_title}.pdf"
                }
        except Exception as e:
            if isinstance(e, VideoIngestionError):
                raise
            raise VideoIngestionError(f"Failed to fetch YouTube metadata: {str(e)}")

    def download_youtube_video(
        self,
        url: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        cancel_event: Optional[threading.Event] = None
    ) -> str:
        """Downloads a video-only 1080p stream to a temporary mp4 file.
        
        Using video-only stream (no audio) makes downloads significantly faster
        and enables reliable random-access frame seeking in OpenCV.
        """
        temp_filename = f"deckspool_{os.getpid()}_{threading.get_ident()}"
        outtmpl = os.path.join(self.temp_dir, f"{temp_filename}.%(ext)s")

        def hook(d):
            if cancel_event and cancel_event.is_set():
                raise DownloadCancelledError("Download cancelled by user.")
            if d.get("status") == "downloading" and progress_callback:
                total_bytes = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                downloaded_bytes = d.get("downloaded_bytes") or 0
                pct = (downloaded_bytes / total_bytes) if total_bytes > 0 else 0.0
                speed = d.get("speed") or 0
                speed_str = f"{speed / (1024 * 1024):.1f} MB/s" if speed else "N/A"
                progress_callback(pct, f"Downloading stream ({pct*100:.1f}% at {speed_str})...")

        ydl_opts = {
            # Prefer 1080p mp4 video-only, fallback to <=1080p best video
            "format": "bestvideo[height<=1080][ext=mp4]/bestvideo[height<=1080]/best[height<=1080]/best",
            "outtmpl": outtmpl,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "progress_hooks": [hook],
            "nocheckcertificate": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                downloaded_path = ydl.prepare_filename(info)
                # In case ext changed or merge occurred:
                if not os.path.exists(downloaded_path):
                    for ext in [".mp4", ".mkv", ".webm"]:
                        candidate = os.path.join(self.temp_dir, f"{temp_filename}{ext}")
                        if os.path.exists(candidate):
                            downloaded_path = candidate
                            break

                if not os.path.exists(downloaded_path):
                    raise VideoIngestionError(f"Downloaded file not found: {downloaded_path}")

                self._downloaded_file = downloaded_path
                return downloaded_path
        except Exception as e:
            if isinstance(e, (DownloadCancelledError, VideoIngestionError)):
                raise
            raise VideoIngestionError(f"Download failed: {str(e)}")

    def cleanup(self) -> None:
        """Removes downloaded temporary video files."""
        if self._downloaded_file and os.path.exists(self._downloaded_file):
            try:
                os.remove(self._downloaded_file)
                logger.info(f"Cleaned up temp video: {self._downloaded_file}")
            except Exception as e:
                logger.warning(f"Failed to remove temp file {self._downloaded_file}: {e}")
            self._downloaded_file = None
