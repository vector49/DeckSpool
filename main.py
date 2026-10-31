"""Entry point for the YT2PDF application."""

import sys
import logging
from yt2pdf import __version__
from yt2pdf.gui import YT2PDFApp


def setup_logging():
    """Sets up console logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    """Starts the YT2PDF desktop GUI application."""
    setup_logging()
    logger = logging.getLogger("yt2pdf")
    logger.info(f"Starting YT2PDF v{__version__}...")

    app = YT2PDFApp()
    try:
        app.mainloop()
    except KeyboardInterrupt:
        logger.info("Application closed by user.")
    except Exception as e:
        logger.error(f"Fatal application error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
