"""Configuration and shared data structures for YT2PDF."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Optional, Any


class SourceType(str, Enum):
    YOUTUBE = "YouTube URL"
    LOCAL_FILE = "Local File"


class PresenterMaskMode(str, Enum):
    AUTO = "Auto-Detect (Motion/Face)"
    TOP_RIGHT = "Top-Right"
    BOTTOM_RIGHT = "Bottom-Right"
    BOTTOM_LEFT = "Bottom-Left"
    TOP_LEFT = "Top-Left"
    SPLIT_RIGHT_50 = "Right-Half Split (50%)"
    SPLIT_LEFT_50 = "Left-Half Split (50%)"
    CUSTOM = "Custom Normalized Box"


@dataclass
class ExtractionConfig:
    """Extraction parameters and options."""
    source_type: SourceType = SourceType.YOUTUBE
    input_source: str = ""
    output_path: str = ""
    
    # Detection parameters
    sample_interval: float = 1.0        # Seconds between sampled frames (0.5 - 5.0)
    similarity_threshold: float = 0.92   # SSIM threshold for slide change (0.80 - 0.99)
    debounce_settle_time: float = 2.0   # Seconds to wait for animations/bullet points to settle
    min_slide_duration: float = 2.0     # Minimum duration in seconds to accept as a unique slide
    
    # Preprocessing & ROI
    mask_presenter: bool = False
    presenter_mode: PresenterMaskMode = PresenterMaskMode.AUTO
    custom_roi_box: Tuple[float, float, float, float] = (0.75, 0.70, 0.25, 0.30)  # (x, y, w, h) normalized
    ignore_chrome: bool = True          # Mask top 3% and bottom 5%
    canvas_auto_crop: bool = False      # Auto-detect black bars/letterboxing and crop PDF canvas
    prune_global_duplicates: bool = True # Review whole presentation and truncate repetitive slides (keep 1st appearance)
    filter_narrator_frames: bool = True  # Automatically filter out full-human camera cuts to speaker


class WorkerMessageType(str, Enum):
    STATUS = "STATUS"
    PROGRESS = "PROGRESS"
    METRICS = "METRICS"
    THUMBNAIL = "THUMBNAIL"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"


@dataclass
class WorkerMessage:
    """Message sent from background processing worker to UI thread."""
    msg_type: WorkerMessageType
    payload: Any = None
