"""Background worker thread orchestrating the video processing pipeline."""

import os
import time
import queue
import logging
import datetime
import threading
from typing import Optional
import cv2

from deckspool.config import (
    ExtractionConfig,
    SourceType,
    WorkerMessage,
    WorkerMessageType,
)
from deckspool.downloader import VideoDownloader, DownloadCancelledError, VideoIngestionError, clean_file_path
from deckspool.detector import SlideDetector
from deckspool.pdf_builder import compile_slides_to_pdf, PDFBuilderError
from deckspool.deduplicator import deduplicate_slides_list

logger = logging.getLogger(__name__)


def format_time_seconds(seconds: float) -> str:
    """Formats seconds into MM:SS or HH:MM:SS string."""
    seconds = max(0, int(seconds))
    hrs = seconds // 3600
    mins = (seconds % 3600) // 60
    secs = seconds % 60
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


class ExtractionWorker(threading.Thread):
    """Worker thread running video ingestion, sampling, detection, and PDF generation."""

    def __init__(
        self,
        config: ExtractionConfig,
        msg_queue: queue.Queue,
        cancel_event: threading.Event
    ):
        super().__init__(daemon=True)
        self.config = config
        self.msg_queue = msg_queue
        self.cancel_event = cancel_event
        self.downloader = VideoDownloader()

    def _send(self, msg_type: WorkerMessageType, payload=None):
        """Pushes a message to the UI event queue."""
        self.msg_queue.put(WorkerMessage(msg_type, payload))

    def run(self):
        video_path = None
        cap = None
        start_time = time.time()

        try:
            # 1. Ingestion / Video Resolution
            if self.config.source_type == SourceType.YOUTUBE:
                url = clean_file_path(self.config.input_source)
                if not url:
                    raise VideoIngestionError("Please enter a valid YouTube URL.")

                self._send(WorkerMessageType.STATUS, "Fetching YouTube stream info...")
                self._send(WorkerMessageType.PROGRESS, 0.05)

                def dl_hook(pct, status_text):
                    self._send(WorkerMessageType.STATUS, status_text)
                    self._send(WorkerMessageType.PROGRESS, 0.05 + (pct * 0.25))

                video_path = self.downloader.download_youtube_video(
                    url,
                    progress_callback=dl_hook,
                    cancel_event=self.cancel_event
                )
            else:
                video_path = clean_file_path(self.config.input_source)
                if not os.path.exists(video_path):
                    raise VideoIngestionError(f"Local file does not exist: {video_path}")

            if self.cancel_event.is_set():
                self._send(WorkerMessageType.CANCELLED, "Extraction was cancelled.")
                return

            # 2. Open Video Capture
            self._send(WorkerMessageType.STATUS, "Opening video capture stream...")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise VideoIngestionError(f"Could not open video file: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps > 240:
                fps = 30.0

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = (total_frames / fps) if total_frames > 0 else 0.0

            # 3. Two-Tier Frame Sampling & Detection
            detector = SlideDetector(self.config)
            frame_step = max(1, int(round(self.config.sample_interval * fps)))
            current_frame_idx = 0
            analyzed_count = 0
            analysis_start = time.time()
            total_time_str = format_time_seconds(duration_sec)

            logger.info(f"Video opened: {total_frames} frames @ {fps:.1f} FPS, step={frame_step} frames")

            while current_frame_idx < total_frames:
                if self.cancel_event.is_set():
                    self._send(WorkerMessageType.CANCELLED, "Extraction was cancelled.")
                    return

                # Fast sequential grab for typical step intervals; seek only for huge jumps
                if current_frame_idx == 0:
                    ret, frame = cap.read()
                elif frame_step <= 120:
                    grab_ok = True
                    for _ in range(frame_step - 1):
                        if not cap.grab():
                            grab_ok = False
                            break
                    ret, frame = cap.read() if grab_ok else (False, None)
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
                    ret, frame = cap.read()

                if not ret or frame is None:
                    break

                curr_time_sec = current_frame_idx / fps
                is_last = (current_frame_idx + frame_step >= total_frames)

                # Process frame through two-tier engine
                committed_slide = detector.process_frame(
                    timestamp=curr_time_sec,
                    frame_bgr=frame,
                    is_last_frame=is_last
                )

                if committed_slide:
                    # Send thumbnail of the newly accepted slide
                    # Resize preview to compact resolution for UI responsiveness
                    preview_h = 240
                    preview_frame = committed_slide.frame_high_res
                    preview_w = max(1, int(preview_frame.shape[1] * (preview_h / float(preview_frame.shape[0]))))
                    preview_rgb = cv2.cvtColor(
                        cv2.resize(preview_frame, (preview_w, preview_h), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2RGB
                    )
                    self._send(WorkerMessageType.THUMBNAIL, {
                        "image_rgb": preview_rgb,
                        "slide_index": committed_slide.index,
                        "timestamp": committed_slide.timestamp
                    })

                analyzed_count += 1
                elapsed = time.time() - analysis_start

                # Metrics calculations
                progress_pct = min(1.0, current_frame_idx / float(max(1, total_frames)))
                speed_mult = (curr_time_sec / elapsed) if elapsed > 0.5 else 1.0
                fps_rate = (analyzed_count / elapsed) if elapsed > 0.5 else 0.0

                # Estimated Time Remaining & Finishing Clock Time based on processing speed
                remaining_video_sec = max(0.0, duration_sec - curr_time_sec)
                eta_wall_sec = (remaining_video_sec / speed_mult) if speed_mult > 0.1 else 0.0

                if eta_wall_sec >= 3600:
                    hrs = int(eta_wall_sec // 3600)
                    mins = int((eta_wall_sec % 3600) // 60)
                    eta_mins_str = f"~{hrs}h {mins}m"
                elif eta_wall_sec >= 60:
                    mins = int(eta_wall_sec // 60)
                    secs = int(eta_wall_sec % 60)
                    eta_mins_str = f"~{mins}m {secs:02d}s"
                else:
                    secs = int(eta_wall_sec)
                    eta_mins_str = f"~{secs}s"

                finish_dt = datetime.datetime.now() + datetime.timedelta(seconds=eta_wall_sec)
                finish_time_str = finish_dt.strftime("%I:%M %p").lstrip("0")
                eta_full_str = f"{eta_mins_str} ({finish_time_str})"

                curr_time_str = format_time_seconds(curr_time_sec)
                metrics_data = {
                    "time_str": f"{curr_time_str} / {total_time_str}",
                    "speed_str": f"{int(fps_rate)} FPS ({speed_mult:.1f}x)",
                    "slides_count": len(detector.committed_slides),
                    "eta_str": eta_full_str,
                    "eta_mins": eta_mins_str,
                    "finish_time": finish_time_str,
                }

                # Update UI every few iterations
                if analyzed_count % 3 == 0 or is_last:
                    self._send(WorkerMessageType.PROGRESS, 0.30 + (progress_pct * 0.55))
                    self._send(WorkerMessageType.METRICS, metrics_data)
                    self._send(
                        WorkerMessageType.STATUS,
                        f"Analyzing frame {curr_time_str} / {total_time_str}..."
                    )

                current_frame_idx += frame_step

            # Release video capture before compilation
            cap.release()
            cap = None

            # 4. Check Extracted Slides
            slides = detector.committed_slides
            if not slides:
                raise VideoIngestionError(
                    "No unique slides detected. Try adjusting the similarity threshold or sampling interval."
                )

            # Global Duplicate Review & Narrator Truncation (hold first appearance alone)
            if self.config.prune_global_duplicates or getattr(self.config, "filter_narrator_frames", True):
                self._send(WorkerMessageType.STATUS, f"Reviewing {len(slides)} slides for duplicates and narrator frames...")
                initial_count = len(slides)
                slides, dup_records = deduplicate_slides_list(
                    slides,
                    self.config.similarity_threshold,
                    filter_narrator_frames=getattr(self.config, "filter_narrator_frames", True)
                )
                pruned_count = initial_count - len(slides)
                if pruned_count > 0:
                    logger.info(f"Truncated {pruned_count} repetitive/narrator slide(s), holding first appearance alone.")
                    self._send(
                        WorkerMessageType.STATUS,
                        f"Truncated {pruned_count} duplicate/narrator slide(s)..."
                    )

            # 5. Compile PDF
            self._send(WorkerMessageType.PROGRESS, 0.90)
            self._send(WorkerMessageType.STATUS, f"Compiling {len(slides)} unique slides into PDF...")

            def pdf_status(msg):
                self._send(WorkerMessageType.STATUS, msg)

            output_file = compile_slides_to_pdf(
                slides=slides,
                output_path=clean_file_path(self.config.output_path),
                canvas_auto_crop=self.config.canvas_auto_crop,
                progress_callback=pdf_status
            )

            total_elapsed = time.time() - start_time
            self._send(WorkerMessageType.PROGRESS, 1.0)
            self._send(WorkerMessageType.STATUS, "Extraction completed successfully!")
            self._send(WorkerMessageType.COMPLETE, {
                "output_path": output_file,
                "slides_count": len(slides),
                "elapsed_seconds": total_elapsed,
            })

        except DownloadCancelledError:
            self._send(WorkerMessageType.CANCELLED, "Extraction was cancelled.")
        except Exception as e:
            logger.error(f"Worker execution failed: {e}", exc_info=True)
            self._send(WorkerMessageType.ERROR, str(e))
        finally:
            if cap is not None:
                cap.release()
            self.downloader.cleanup()
