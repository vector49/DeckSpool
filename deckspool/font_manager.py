"""Font management and asset resolution module for DeckSpool.

Handles dynamic registration of custom fonts into Qt's QFontDatabase and provides
bulletproof path resolution for both local development and PyInstaller bundled executables.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication


def get_asset_path(relative_path: str = "") -> Path:
    """Resolve an asset path reliably across dev environments and PyInstaller bundles.

    When compiled into a single-file or directory executable with PyInstaller,
    bundled assets are unpacked into `sys._MEIPASS`. In development mode,
    assets are resolved relative to the package directory.

    Args:
        relative_path: Subpath relative to the `assets/` directory or root.

    Returns:
        Path: Absolute resolved path.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        # Running inside PyInstaller bundled executable
        base_dir = Path(sys._MEIPASS) / "deckspool" / "assets"
        if not base_dir.exists():
            # Alternative PyInstaller mapping (e.g. --add-data "deckspool/assets;assets")
            base_dir = Path(sys._MEIPASS) / "assets"
    else:
        # Running in local development mode
        base_dir = Path(__file__).resolve().parent / "assets"

    if relative_path:
        return base_dir / relative_path
    return base_dir


def get_fonts_dir() -> Path:
    """Return the absolute path to the fonts directory."""
    return get_asset_path("fonts")


# Available Neo-Brutalist Font Presets
PRESET_TECH_BRUTALIST = "tech_brutalist"       # Space Grotesk + Plus Jakarta Sans + Space Mono
PRESET_CREATIVE_GROTESQUE = "creative_grotesque" # Bricolage Grotesque + Plus Jakarta Sans + Space Mono
PRESET_INDUSTRIAL_BOLD = "industrial_bold"     # Archivo Black + Plus Jakarta Sans + JetBrains Mono
PRESET_CLEAN_CODE = "clean_code"               # Space Grotesk + Plus Jakarta Sans + JetBrains Mono

DEFAULT_PRESET = PRESET_TECH_BRUTALIST


class FontManager:
    """Manages application font registration, fallbacks, and stylesheet integration."""

    _instance: Optional[FontManager] = None
    _loaded: bool = False

    def __new__(cls) -> FontManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registered_families = {}
            cls._instance._active_preset = DEFAULT_PRESET
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def active_preset(self) -> str:
        return self._active_preset

    def load_all_fonts(self) -> Dict[str, list[str]]:
        """Load all available TTF fonts from the assets/fonts directory into QFontDatabase.

        Returns:
            Dict mapping font filename stem to list of registered family names.
        """
        if QApplication.instance() is None:
            return {}

        fonts_dir = get_fonts_dir()
        if not fonts_dir.exists():
            print(f"[FontManager] Warning: Fonts directory not found at {fonts_dir}")
            return {}

        registered = {}
        for font_path in sorted(fonts_dir.glob("*.ttf")):
            font_id = QFontDatabase.addApplicationFont(str(font_path))
            if font_id == -1:
                print(f"[FontManager] Failed to load font: {font_path.name}")
                continue
            families = QFontDatabase.applicationFontFamilies(font_id)
            registered[font_path.stem] = families

        self._registered_families = registered
        self._loaded = True
        return registered

    def get_font_families(self, preset: Optional[str] = None) -> Dict[str, str]:
        """Return the family names for 'display', 'body', and 'mono' based on selected preset.

        Presets:
            - tech_brutalist: Space Grotesk (Display), Plus Jakarta Sans (Body), Space Mono (Mono)
            - creative_grotesque: Bricolage Grotesque (Display), Plus Jakarta Sans (Body), Space Mono (Mono)
            - industrial_bold: Archivo Black (Display), Plus Jakarta Sans (Body), JetBrains Mono (Mono)
            - clean_code: Space Grotesk (Display), Plus Jakarta Sans (Body), JetBrains Mono (Mono)
        """
        if not self._loaded:
            self.load_all_fonts()

        chosen_preset = preset or self._active_preset
        system_sans = "Segoe UI, sans-serif"
        system_mono = "Consolas, monospace"

        if chosen_preset == PRESET_CREATIVE_GROTESQUE:
            return {
                "display": "Bricolage Grotesque, sans-serif",
                "body": "Plus Jakarta Sans, " + system_sans,
                "mono": "Space Mono, " + system_mono,
                "code": "JetBrains Mono, " + system_mono,
            }
        elif chosen_preset == PRESET_INDUSTRIAL_BOLD:
            return {
                "display": "Archivo Black, sans-serif",
                "body": "Plus Jakarta Sans, " + system_sans,
                "mono": "JetBrains Mono, " + system_mono,
                "code": "JetBrains Mono, " + system_mono,
            }
        elif chosen_preset == PRESET_CLEAN_CODE:
            return {
                "display": "Space Grotesk, sans-serif",
                "body": "Plus Jakarta Sans, " + system_sans,
                "mono": "JetBrains Mono, " + system_mono,
                "code": "JetBrains Mono, " + system_mono,
            }
        else:  # PRESET_TECH_BRUTALIST (default)
            return {
                "display": "Space Grotesk, sans-serif",
                "body": "Plus Jakarta Sans, " + system_sans,
                "mono": "Space Mono, " + system_mono,
                "code": "JetBrains Mono, " + system_mono,
            }

    def set_preset(self, preset_name: str) -> None:
        """Set the active font preset."""
        self._active_preset = preset_name

    def create_qfont(
        self,
        role: str = "body",
        size: int = 10,
        weight: QFont.Weight = QFont.Weight.Normal,
        preset: Optional[str] = None,
    ) -> QFont:
        """Create a QFont instance using the registered family for a specific role.

        Args:
            role: 'display', 'body', or 'mono'
            size: Font point size
            weight: QFont.Weight enum (e.g. QFont.Weight.Bold)
            preset: Optional preset override
        """
        families = self.get_font_families(preset)
        primary_family = families.get(role, "sans-serif").split(",")[0].strip()
        font = QFont(primary_family, size)
        font.setWeight(weight)
        return font


# Global singleton helper
font_manager = FontManager()
