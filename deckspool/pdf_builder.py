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

"""Lossless PDF assembly and Canvas Auto-Cropping using img2pdf."""

import os
import tempfile
import logging
from typing import List, Optional, Callable
import cv2
import img2pdf

from deckspool.detector import Slide
from deckspool.roi import detect_canvas_crop_box

logger = logging.getLogger(__name__)


class PDFBuilderError(Exception):
    """Raised when PDF compilation fails."""
    pass


def compile_slides_to_pdf(
    slides: List[Slide],
    output_path: str,
    canvas_auto_crop: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None
) -> str:
    """Compiles unique high-resolution slides into a clean, lossless PDF.
    
    Args:
        slides: List of Slide objects with high-res BGR frames.
        output_path: Path to write the output PDF.
        canvas_auto_crop: Whether to automatically detect and crop black borders.
        progress_callback: Callback for status updates.
        
    Returns:
        Absolute path to the compiled PDF file.
    """
    if not slides:
        raise PDFBuilderError("No slides were extracted to compile into a PDF.")

    temp_image_paths: List[str] = []
    temp_dir = tempfile.mkdtemp(prefix="deckspool_slides_")

    try:
        # Detect global canvas crop box across all slides if enabled
        crop_box = None
        if canvas_auto_crop:
            if progress_callback:
                progress_callback("Detecting canvas dimensions & cropping borders...")
            raw_frames = [s.frame_high_res for s in slides]
            crop_box = detect_canvas_crop_box(raw_frames)
            if crop_box:
                logger.info(f"Auto-crop box detected: {crop_box}")

        total_slides = len(slides)
        for i, slide in enumerate(slides):
            if progress_callback and (i % max(1, total_slides // 5) == 0 or i == total_slides - 1):
                progress_callback(f"Preparing slide {i + 1} of {total_slides}...")

            frame = slide.frame_high_res

            # Apply auto-crop if detected
            if crop_box:
                x1, y1, x2, y2 = crop_box
                frame = frame[y1:y2, x1:x2]

            temp_img_path = os.path.join(temp_dir, f"slide_{i:04d}.png")
            # Write lossless PNG
            success = cv2.imwrite(temp_img_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 4])
            if not success or not os.path.exists(temp_img_path):
                raise PDFBuilderError(f"Failed to encode temporary slide frame #{i + 1}")
            temp_image_paths.append(temp_img_path)

        if progress_callback:
            progress_callback("Assembling PDF document with img2pdf...")

        # Ensure parent output directory exists
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # Lossless compilation using img2pdf
        with open(output_path, "wb") as f:
            f.write(img2pdf.convert(temp_image_paths))

        logger.info(f"Successfully generated PDF with {total_slides} slides: {output_path}")
        return os.path.abspath(output_path)

    except Exception as e:
        logger.error(f"PDF compilation error: {e}", exc_info=True)
        if isinstance(e, PDFBuilderError):
            raise
        raise PDFBuilderError(f"Failed to compile PDF: {str(e)}")

    finally:
        # Clean up temporary frame images and directory
        for p in temp_image_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        try:
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception:
            pass
