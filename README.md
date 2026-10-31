# YT2PDF — Presentation Slide Extractor & Lossless PDF Builder

A high-performance desktop application built with Python and CustomTkinter that automatically detects and extracts unique presentation slides from YouTube videos or local video files and compiles them into a clean, lossless PDF.

---

## Key Features

- **Modern Dark-Mode Native GUI**: Built with CustomTkinter, featuring responsive layouts, customizable cards, and dynamic KPI badges.
- **Two-Tier Frame Sampling**:
  - Ultra-fast 360p grayscale analysis tier using combined 64-bit difference hashing (`dHash`) and vectorized OpenCV Structural Similarity Index (`SSIM`).
  - High-resolution source-native capture tier (up to 1080p) stored in memory buffers only when candidate transitions occur.
- **Debounced Slide Transition & Progressive Build Settle Engine**:
  - Handles animated bullet points, progressive handwriting, and slide animations by buffering candidate frames until visual content stabilizes for the configured settle time.
  - Commits only the final completed state of each slide, eliminating redundant duplicate slides.
- **Full-Human / Narrator Frame Filter (Speaker Cutaway Removal)**:
  - Automatically identifies camera shots of the teacher or speaker talking (full-human / narrator frames) using lightweight onboard neural face detection (`YuNet`) and edge density heuristics.
  - Distinguishes between full-frame presenter shots (which are filtered out) and small corner PIP webcams on presentation slides (which are preserved and masked).
  - Works during live video extraction and in the standalone PDF deduplication review tool to purge speaker-only shots from your final PDF.
- **Global PDF Review & Duplicate Truncation (Keep First Appearance Alone)**:
  - Scans the entire document or slide stream to find recurring duplicates (e.g. repeated agenda slides, title slides, or speaker cutbacks), truncating repetitive occurrences while preserving the first appearance.
  - Dedicated interactive modal tool (**"Review & Deduplicate PDF..."**) to inspect and clean any existing PDF file with live duplicate logs and custom thresholding.
  - Integrated into the video extraction pipeline with the **"Truncate Global Duplicate Slides"** toggle.
- **Presenter & PIP Masking**:
  - Exclude speaker webcam overlays to avoid false transition triggers.
  - Supports Auto-Detection (via motion variance analysis), corner presets (Top-Right, Bottom-Right, Bottom-Left, Top-Left), and custom normalized bounding boxes (`x,y,w,h`).
- **Player Chrome Masking**: Automatically masks the top 3% and bottom 5% of frames during analysis to ignore video titles, YouTube scrubber bars, and playback controls.
- **Canvas Auto-Crop**: Detects and eliminates black letterbox and pillarbox bars (e.g. 4:3 presentation within 16:9 canvas), exporting clean, edge-to-edge slide pages.
- **Lossless PDF Compilation**: Direct byte embedding using `img2pdf` to avoid lossy re-encoding artifacts.
- **Multi-Threaded Architecture**: Non-blocking background worker with thread-safe UI message queues and full cancellation support.

---

## Installation

Ensure Python 3.10+ is installed, then install dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies
- `customtkinter`: Modern, dark-mode GUI widgets.
- `yt-dlp`: Fast stream resolution and video-only ingestion without re-encoding.
- `opencv-python-headless`: Fast frame seeking, image transformation, and vectorized SSIM.
- `imagehash`: 64-bit perceptual difference hashing (`dHash`).
- `img2pdf`: Lossless image-to-PDF compiler without re-compression.
- `pypdfium2`: High-speed PDF page rasterization and lossless page pruning.
- `Pillow`: Image processing and thumbnail preview rendering.
- `scipy` & `numpy`: High-performance array operations.

---

## Usage

Launch the desktop GUI:

```bash
python main.py
```

### Workflow
1. **Choose Source**: Select **YouTube URL** or **Local File**.
   - For YouTube: Paste the video link (e.g., `https://www.youtube.com/watch?v=...`). The video title is automatically queried in the background to name your PDF.
   - For Local Files: Click **Browse...** to pick any `.mp4`, `.mkv`, `.mov`, `.avi`, or `.webm` presentation video.
2. **Configure Detection (Optional)**:
   - **Sample Interval**: 0.5s to 5.0s (Default: `1.0s`).
   - **Similarity Threshold**: 80% to 99% (Default: `92%`).
   - **Debounce / Settle Time**: 1.0s to 10.0s (Default: `2.0s`).
   - **Min Slide Duration**: 1.0s to 15.0s (Default: `2.0s`).
3. **Configure Preprocessing**:
   - Toggle **Mask Presenter / PIP** if the speaker webcam is in the corner.
   - Toggle **Ignore Player Chrome** (default: On) to mask player controls.
   - Toggle **Canvas Auto-Crop** to eliminate black borders around slides.
4. **Start Extraction**: Click the green **Start Extraction** button. Monitor real-time progress, FPS, slide count, and live preview thumbnails.
5. **Open Document**: Once finished, click **Open PDF** or **Open Folder** directly from the completion modal.

---

## Automated Tests

Run the test suite:

```bash
python -m unittest tests/test_pipeline.py
```
