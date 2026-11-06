"""Unit tests for yt2pdf.font_manager and font assets using standard unittest."""

import os
import unittest
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QFontDatabase

from yt2pdf.font_manager import (
    FontManager,
    font_manager,
    get_asset_path,
    get_fonts_dir,
    PRESET_TECH_BRUTALIST,
    PRESET_CREATIVE_GROTESQUE,
    PRESET_INDUSTRIAL_BOLD,
    PRESET_CLEAN_CODE,
)


class TestFontManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication([])

    def test_fonts_directory_exists(self):
        fonts_dir = get_fonts_dir()
        self.assertTrue(fonts_dir.exists(), f"Fonts directory not found: {fonts_dir}")
        ttf_files = list(fonts_dir.glob("*.ttf"))
        self.assertGreaterEqual(len(ttf_files), 10, f"Expected at least 10 TTF files, found {len(ttf_files)}")

    def test_font_manager_loading(self):
        mgr = FontManager()
        registered = mgr.load_all_fonts()
        self.assertTrue(mgr.is_loaded)
        self.assertGreater(len(registered), 0)

        available_families = QFontDatabase.families()
        self.assertTrue(any("Space Grotesk" in f for f in available_families))
        self.assertTrue(any("Space Mono" in f for f in available_families))
        self.assertTrue(any("Plus Jakarta Sans" in f for f in available_families))

    def test_font_presets(self):
        mgr = FontManager()

        for preset in [
            PRESET_TECH_BRUTALIST,
            PRESET_CREATIVE_GROTESQUE,
            PRESET_INDUSTRIAL_BOLD,
            PRESET_CLEAN_CODE,
        ]:
            families = mgr.get_font_families(preset=preset)
            self.assertIn("display", families)
            self.assertIn("body", families)
            self.assertIn("mono", families)

    def test_create_qfont(self):
        mgr = FontManager()
        font_display = mgr.create_qfont("display", size=14, weight=QFont.Weight.Bold)
        self.assertEqual(font_display.pointSize(), 14)
        self.assertEqual(font_display.weight(), QFont.Weight.Bold)

        font_mono = mgr.create_qfont("mono", size=11, weight=QFont.Weight.Bold)
        self.assertEqual(font_mono.pointSize(), 11)

        font_body = mgr.create_qfont("body", size=10, weight=QFont.Weight.Normal)
        self.assertEqual(font_body.pointSize(), 10)


if __name__ == "__main__":
    unittest.main()
