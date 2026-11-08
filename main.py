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

"""Entry point for the DeckSpool application."""

import sys
import logging
from PySide6.QtWidgets import QApplication
from deckspool import __version__
from deckspool.gui import DeckSpoolApp
from deckspool.font_manager import setup_windows_app_id, get_app_icon


def setup_logging():
    """Sets up console logging for the application."""
    root = logging.getLogger()
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)


def main():
    """Starts the DeckSpool desktop GUI application."""
    setup_logging()
    setup_windows_app_id()
    logger = logging.getLogger("deckspool")
    logger.info(f"Starting DeckSpool v{__version__}...")

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    app_icon = get_app_icon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)

    window = DeckSpoolApp()
    window.show()

    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        logger.info("Application closed by user.")
    except Exception as e:
        logger.error(f"Fatal application error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
