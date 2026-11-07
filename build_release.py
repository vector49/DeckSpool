#!/usr/bin/env python3
"""Build and packaging script for YT2PDF.

Compiles the YT2PDF application into:
  1. A standalone Portable ZIP archive (extract & run, zero installation required).
  2. A self-contained Windows GUI Setup Installer (.exe) with desktop/start menu shortcuts,
     registry registration, and uninstaller.
  3. An Inno Setup script (.iss) with automatic compilation if Inno Setup is installed.
  4. Cryptographic SHA-256 checksums (SHA256SUMS.txt).

Usage:
  python build_release.py [options]

Options:
  --clean           Remove previous build/dist artifacts before compiling.
  --skip-portable   Skip generating the portable ZIP archive.
  --skip-installer  Skip generating the standalone installer executable.
  --output-dir DIR  Custom destination directory for release artifacts (default: dist/release).
  --debug           Build with console window enabled for troubleshooting.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

# -----------------------------------------------------------------------------
# Project Metadata & Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from yt2pdf import __author__, __version__
except ImportError:
    __version__ = "1.0.0"
    __author__ = "YT2PDF Team"

APP_NAME = "YT2PDF"
APP_TITLE = "YT2PDF - Presentation Slide Extractor"
APP_DESCRIPTION = "Extract unique presentation slides from YouTube videos and local files into clean, searchable PDFs."
PUBLISHER = "DeckSpool"

BUILD_DIR = PROJECT_ROOT / "build"
DIST_DIR = PROJECT_ROOT / "dist"
DEFAULT_RELEASE_DIR = DIST_DIR / "release"


# -----------------------------------------------------------------------------
# Terminal Styling & Logging
# -----------------------------------------------------------------------------
class Log:
    """Formatted terminal output with ANSI colors and timestamps."""

    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    @classmethod
    def _now(cls) -> str:
        return time.strftime("%H:%M:%S")

    @classmethod
    def header(cls, text: str) -> None:
        print(f"\n{cls.BOLD}{cls.CYAN}=== {text} ==={cls.RESET}")

    @classmethod
    def info(cls, text: str) -> None:
        print(f"[{cls._now()}] {cls.BLUE}INFO{cls.RESET}  {text}")

    @classmethod
    def success(cls, text: str) -> None:
        print(f"[{cls._now()}] {cls.GREEN}SUCCESS{cls.RESET} {text}")

    @classmethod
    def warn(cls, text: str) -> None:
        print(f"[{cls._now()}] {cls.YELLOW}WARN{cls.RESET}  {text}")

    @classmethod
    def error(cls, text: str) -> None:
        print(f"[{cls._now()}] {cls.RED}ERROR{cls.RESET} {text}")


# -----------------------------------------------------------------------------
# Asset Preparation (Icon Generation)
# -----------------------------------------------------------------------------
def resolve_logo_path() -> Optional[Path]:
    """Finds the source branding logo in standard project locations."""
    candidates = [
        PROJECT_ROOT / "docs" / "images" / "deckspool_logo.png",
        PROJECT_ROOT / "Icon" / "DeckSpool Logo.png",
        PROJECT_ROOT / "docs" / "images" / "logo.png",
        PROJECT_ROOT / "yt2pdf" / "assets" / "logo.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def generate_application_icon(output_path: Path) -> Path:
    """Creates a multi-resolution Windows .ico file from the source PNG logo."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return output_path

    logo_path = resolve_logo_path()
    sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]

    if logo_path and logo_path.exists():
        Log.info(f"Generating icon from {logo_path.relative_to(PROJECT_ROOT)}...")
        img = Image.open(logo_path).convert("RGBA")
    else:
        Log.warn("No logo image found; generating default fallback icon...")
        # Create a clean high-res canvas with a geometric accent
        img = Image.new("RGBA", (256, 256), color=(26, 26, 26, 255))

    img.save(output_path, format="ICO", sizes=sizes)
    Log.success(f"Application icon generated at {output_path.relative_to(PROJECT_ROOT)}")
    return output_path


# -----------------------------------------------------------------------------
# Core Application Compilation (PyInstaller onedir)
# -----------------------------------------------------------------------------
def compile_core_application(icon_path: Path, debug: bool = False, force: bool = False) -> Path:
    """Invokes PyInstaller to build the standalone onedir distribution."""
    app_dist_dir = DIST_DIR / APP_NAME
    exe_path = app_dist_dir / f"{APP_NAME}.exe"

    if exe_path.exists() and not force:
        Log.info(f"Using existing compiled application core at {exe_path.relative_to(PROJECT_ROOT)}")
        return app_dist_dir

    Log.header("Compiling Application Core with PyInstaller")
    main_script = PROJECT_ROOT / "main.py"

    if not main_script.exists():
        raise FileNotFoundError(f"Entry script not found at {main_script}")

    # Build data inclusion mappings (Source;Destination)
    datas = [
        f"{PROJECT_ROOT / 'yt2pdf' / 'assets'};yt2pdf/assets",
        f"{PROJECT_ROOT / 'yt2pdf' / 'models'};yt2pdf/models",
    ]
    if (PROJECT_ROOT / "docs" / "images").exists():
        datas.append(f"{PROJECT_ROOT / 'docs' / 'images'};docs/images")

    pyinstaller_args = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name",
        APP_NAME,
        "--noconfirm",
        "--onedir",
        "--clean",
        "--distpath",
        str(DIST_DIR),
        "--workpath",
        str(BUILD_DIR / "work_app"),
        "--specpath",
        str(BUILD_DIR),
        f"--icon={icon_path}",
    ]

    if not debug:
        pyinstaller_args.append("--windowed")
    else:
        pyinstaller_args.append("--console")

    for d in datas:
        pyinstaller_args.extend(["--add-data", d])

    # Dynamic imports and binary collections
    pyinstaller_args.extend([
        "--collect-all",
        "pypdfium2",
        "--collect-all",
        "cv2",
        "--collect-all",
        "yt_dlp",
        "--collect-submodules",
        "yt2pdf",
        "--hidden-import",
        "img2pdf",
        "--hidden-import",
        "imagehash",
        "--hidden-import",
        "PIL",
        "--hidden-import",
        "scipy",
        "--hidden-import",
        "PySide6",
        str(main_script),
    ])

    Log.info(f"Running PyInstaller onedir build for {APP_NAME} v{__version__}...")
    start_time = time.time()

    process = subprocess.Popen(
        pyinstaller_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(PROJECT_ROOT),
    )

    # Stream output lines cleanly
    if process.stdout:
        for line in iter(process.stdout.readline, ""):
            line_str = line.strip()
            if any(k in line_str for k in ("Building PKG", "Building EXE", "Building COLLECT", "completed successfully")):
                Log.info(line_str)

    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"PyInstaller build failed with exit code {process.returncode}")

    exe_path = app_dist_dir / f"{APP_NAME}.exe"
    if not exe_path.exists():
        raise FileNotFoundError(f"Expected compiled executable not found at {exe_path}")

    elapsed = time.time() - start_time
    Log.success(f"Application compiled successfully in {elapsed:.1f}s: {exe_path}")
    return app_dist_dir


# -----------------------------------------------------------------------------
# Variant 1: Portable ZIP Archive Builder
# -----------------------------------------------------------------------------
def build_portable_zip(app_dir: Path, release_dir: Path, force: bool = False) -> Path:
    """Packages the compiled onedir application into a portable ZIP archive."""
    Log.header("Packaging Variant 1: Portable ZIP Archive")

    release_dir.mkdir(parents=True, exist_ok=True)
    zip_filename = f"{APP_NAME}-v{__version__}-windows-x64-portable.zip"
    zip_path = release_dir / zip_filename

    if zip_path.exists() and not force:
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        Log.info(f"Using existing portable ZIP archive ({size_mb:.1f} MB) at {zip_path.name}")
        return zip_path

    if zip_path.exists():
        zip_path.unlink()

    # Create a quick-start README for portable users
    readme_text = f"""========================================================================
{APP_NAME} v{__version__} - Portable Release (Windows x64)
{APP_DESCRIPTION}
Publisher: {PUBLISHER}
========================================================================

QUICK START:
1. Extract the contents of this ZIP file to any folder or USB drive.
2. Double-click '{APP_NAME}.exe' to launch the application.
3. No installation or administrator permissions are required.

SYSTEM REQUIREMENTS:
- Windows 10 (64-bit) or Windows 11
- Internet connection (for YouTube ingestion and streaming)

SUPPORT & DOCUMENTATION:
- Project Repository: https://github.com/vignesh/YT2PDF
========================================================================
"""
    readme_path = app_dir / "PORTABLE_README.txt"
    readme_path.write_text(readme_text, encoding="utf-8")

    Log.info(f"Compressing {app_dir.name} into {zip_path.name}...")
    start_time = time.time()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        total_files = sum(len(files) for _, _, files in os.walk(app_dir))
        count = 0

        for root, _, files in os.walk(app_dir):
            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(app_dir.parent)
                zf.write(file_path, arcname=str(rel_path))
                count += 1
                if count % 200 == 0 or count == total_files:
                    pct = (count / max(1, total_files)) * 100.0
                    print(f"\r  Compressing portable archive: {count}/{total_files} files ({pct:.0f}%)...", end="")

    print()
    elapsed = time.time() - start_time
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    Log.success(f"Portable ZIP created ({size_mb:.1f} MB, {elapsed:.1f}s): {zip_path}")
    return zip_path


# -----------------------------------------------------------------------------
# Variant 2: Windows Setup Installer Generator
# -----------------------------------------------------------------------------
WIZARD_TEMPLATE = r'''# -*- coding: utf-8 -*-
"""Standalone Windows Setup Installer Wizard for __APP_NAME__."""

import os
import sys
import shutil
import zipfile
import threading
import subprocess
import winreg
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

APP_NAME = "__APP_NAME__"
APP_VERSION = "__APP_VERSION__"
APP_PUBLISHER = "__APP_PUBLISHER__"
APP_DESC = "__APP_DESC__"

DEFAULT_INSTALL_DIR = os.path.expandvars(rf"%LOCALAPPDATA%\Programs\{APP_NAME}")


class SetupWizard(tk.Tk):
    """Graphical Setup Wizard for installing __APP_NAME__."""

    def __init__(self, silent: bool = False, target_dir: str = None, no_launch: bool = False):
        super().__init__()
        self.silent = silent
        self.install_dir = tk.StringVar(value=target_dir if target_dir else DEFAULT_INSTALL_DIR)
        self.create_desktop_shortcut = tk.BooleanVar(value=True)
        self.create_startmenu_shortcut = tk.BooleanVar(value=True)
        self.launch_after = tk.BooleanVar(value=not no_launch)
        self.is_installing = False

        if self.silent:
            self.withdraw()
            self.after(100, self._run_silent_install)
            return

        self.title(f"{APP_NAME} v{APP_VERSION} Setup")
        self.geometry("540x380")
        self.resizable(False, False)
        self._center_window()
        self._apply_styling()
        self._build_ui()

    def _center_window(self):
        self.update_idletasks()
        w = self.winfo_width()
        h = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (w // 2)
        y = (self.winfo_screenheight() // 2) - (h // 2)
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _apply_styling(self):
        self.configure(bg="#F8FAFC")
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("vista")
        except Exception:
            pass

    def _build_ui(self):
        header_frame = tk.Frame(self, bg="#0F172A", height=70)
        header_frame.pack(fill="x")

        title_lbl = tk.Label(
            header_frame,
            text=f"Install {APP_NAME} v{APP_VERSION}",
            font=("Segoe UI", 14, "bold"),
            fg="#FFFFFF",
            bg="#0F172A"
        )
        title_lbl.pack(anchor="w", padx=20, pady=(12, 2))

        sub_lbl = tk.Label(
            header_frame,
            text=APP_DESC,
            font=("Segoe UI", 8),
            fg="#94A3B8",
            bg="#0F172A"
        )
        sub_lbl.pack(anchor="w", padx=20, pady=(0, 10))

        self.body_frame = tk.Frame(self, bg="#F8FAFC", padx=24, pady=16)
        self.body_frame.pack(fill="both", expand=True)

        dest_lbl = tk.Label(
            self.body_frame,
            text="Destination Folder:",
            font=("Segoe UI", 9, "bold"),
            bg="#F8FAFC",
            fg="#1E293B"
        )
        dest_lbl.pack(anchor="w", pady=(0, 4))

        path_row = tk.Frame(self.body_frame, bg="#F8FAFC")
        path_row.pack(fill="x", pady=(0, 12))

        self.path_entry = ttk.Entry(path_row, textvariable=self.install_dir)
        self.path_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))

        browse_btn = ttk.Button(path_row, text="Browse...", command=self._on_browse)
        browse_btn.pack(side="right")

        opts_frame = tk.LabelFrame(
            self.body_frame,
            text=" Shortcuts & Integration ",
            bg="#F8FAFC",
            fg="#334155",
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=8
        )
        opts_frame.pack(fill="x", pady=(0, 12))

        cb_desktop = tk.Checkbutton(
            opts_frame,
            text="Create a Desktop Shortcut",
            variable=self.create_desktop_shortcut,
            bg="#F8FAFC",
            font=("Segoe UI", 9)
        )
        cb_desktop.pack(anchor="w", pady=2)

        cb_start = tk.Checkbutton(
            opts_frame,
            text="Create a Start Menu Shortcut",
            variable=self.create_startmenu_shortcut,
            bg="#F8FAFC",
            font=("Segoe UI", 9)
        )
        cb_start.pack(anchor="w", pady=2)

        cb_launch = tk.Checkbutton(
            opts_frame,
            text="Launch application when setup completes",
            variable=self.launch_after,
            bg="#F8FAFC",
            font=("Segoe UI", 9)
        )
        cb_launch.pack(anchor="w", pady=2)

        self.progress_bar = ttk.Progressbar(self.body_frame, mode="determinate")
        self.progress_bar.pack(fill="x", pady=(4, 4))

        self.status_lbl = tk.Label(
            self.body_frame,
            text="Ready to install.",
            font=("Segoe UI", 8),
            bg="#F8FAFC",
            fg="#64748B"
        )
        self.status_lbl.pack(anchor="w")

        btn_frame = tk.Frame(self, bg="#E2E8F0", height=48)
        btn_frame.pack(fill="x", side="bottom")

        self.cancel_btn = ttk.Button(btn_frame, text="Cancel", command=self.destroy)
        self.cancel_btn.pack(side="right", padx=16, pady=10)

        self.install_btn = ttk.Button(btn_frame, text="Install Now", command=self._on_start_install)
        self.install_btn.pack(side="right", padx=4, pady=10)

    def _on_browse(self):
        chosen = filedialog.askdirectory(initialdir=self.install_dir.get(), title="Select Installation Folder")
        if chosen:
            self.install_dir.set(os.path.normpath(chosen))

    def _on_start_install(self):
        target = self.install_dir.get().strip()
        if not target:
            messagebox.showwarning("Invalid Path", "Please specify a valid installation directory.")
            return

        self.install_btn.configure(state="disabled")
        self.path_entry.configure(state="disabled")
        self.cancel_btn.configure(state="disabled")
        self.is_installing = True

        threading.Thread(target=self._perform_installation, daemon=True).start()

    def _perform_installation(self):
        try:
            target_dir = Path(self.install_dir.get().strip())
            target_dir.mkdir(parents=True, exist_ok=True)

            base_meipass = getattr(sys, "_MEIPASS", os.path.dirname(__file__))
            payload_path = os.path.join(base_meipass, "app_payload.zip")

            if not os.path.exists(payload_path):
                raise FileNotFoundError(f"Embedded payload not found at {payload_path}")

            self._update_status("Extracting application files...", 0.1)

            with zipfile.ZipFile(payload_path, "r") as zf:
                all_members = zf.infolist()
                total = len(all_members)
                for i, member in enumerate(all_members):
                    zf.extract(member, target_dir)
                    if i % 100 == 0 or i == total - 1:
                        self._update_status(f"Extracting: {member.filename}", 0.1 + (0.7 * (i / max(1, total))))

            exe_path = target_dir / f"{APP_NAME}.exe"

            self._update_status("Creating desktop and start menu shortcuts...", 0.85)
            if self.create_desktop_shortcut.get():
                self._create_shortcut(exe_path, "Desktop")
            if self.create_startmenu_shortcut.get():
                self._create_shortcut(exe_path, "Programs")

            self._update_status("Registering application in Windows...", 0.92)
            self._register_uninstall(target_dir, exe_path)

            self._update_status("Installation complete!", 1.0)
            self.after(200, self._on_install_success)

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Installation Error", f"Installation failed:\n{str(e)}"))
            self.after(0, lambda: self.cancel_btn.configure(state="normal"))

    def _update_status(self, text: str, progress: float):
        if not self.silent:
            self.after(0, lambda: self.status_lbl.configure(text=text))
            self.after(0, lambda: self.progress_bar.configure(value=progress * 100))

    def _create_shortcut(self, target_exe: Path, location_type: str):
        try:
            tgt = str(target_exe).replace("\\", "\\\\")
            wdir = str(target_exe.parent).replace("\\", "\\\\")
            ps_script = (
                f"$Wsh = New-Object -ComObject WScript.Shell; "
                f"$f = [Environment]::GetFolderPath('{location_type}'); "
                f"if ('{location_type}' -eq 'Programs') {{ $f = Join-Path $f '{APP_NAME}'; if (-not (Test-Path $f)) {{ New-Item -ItemType Directory -Path $f | Out-Null }} }}; "
                f"$s = $Wsh.CreateShortcut((Join-Path $f '{APP_NAME}.lnk')); "
                f"$s.TargetPath = '{tgt}'; "
                f"$s.WorkingDirectory = '{wdir}'; "
                f"$s.Description = '{APP_DESC}'; "
                f"$s.Save();"
            )
            subprocess.run(["powershell", "-NoProfile", "-Command", ps_script], check=True, capture_output=True)
        except Exception:
            pass

    def _register_uninstall(self, install_dir: Path, exe_path: Path):
        try:
            uninst_bat = install_dir / "uninstall.bat"
            uninst_script = f"""@echo off
title {APP_NAME} Uninstaller
echo ========================================================================
echo Uninstalling {APP_NAME}...
echo ========================================================================
powershell -NoProfile -Command "Remove-Item -Path \\"$([Environment]::GetFolderPath('Desktop'))\\{APP_NAME}.lnk\\" -Force -ErrorAction SilentlyContinue; Remove-Item -Path \\"$([Environment]::GetFolderPath('Programs'))\\{APP_NAME}\\" -Recurse -Force -ErrorAction SilentlyContinue; Remove-Item -Path 'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{APP_NAME}\\' -Recurse -Force -ErrorAction SilentlyContinue;"
echo Shortcuts and registry entries removed.
echo Cleaning up application files...
cd ..
timeout /t 1 /nobreak >nul
rmdir /S /Q "%~dp0"
echo {APP_NAME} has been completely uninstalled.
pause
"""
            uninst_bat.write_text(uninst_script, encoding="utf-8")

            key_path = fr"Software\Microsoft\Windows\CurrentVersion\Uninstall\{APP_NAME}"
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                winreg.SetValueEx(key, "DisplayName", 0, winreg.REG_SZ, f"{APP_NAME} - {APP_DESC}")
                winreg.SetValueEx(key, "DisplayVersion", 0, winreg.REG_SZ, APP_VERSION)
                winreg.SetValueEx(key, "Publisher", 0, winreg.REG_SZ, APP_PUBLISHER)
                winreg.SetValueEx(key, "DisplayIcon", 0, winreg.REG_SZ, str(exe_path))
                winreg.SetValueEx(key, "InstallLocation", 0, winreg.REG_SZ, str(install_dir))
                winreg.SetValueEx(key, "UninstallString", 0, winreg.REG_SZ, str(uninst_bat))
                winreg.SetValueEx(key, "NoModify", 0, winreg.REG_DWORD, 1)
                winreg.SetValueEx(key, "NoRepair", 0, winreg.REG_DWORD, 1)
        except Exception:
            pass

    def _on_install_success(self):
        exe_path = Path(self.install_dir.get().strip()) / f"{APP_NAME}.exe"
        if self.launch_after.get() and exe_path.exists():
            try:
                subprocess.Popen([str(exe_path)], cwd=str(exe_path.parent))
            except Exception:
                pass

        if not self.silent:
            messagebox.showinfo("Installation Complete", f"{APP_NAME} v{APP_VERSION} was installed successfully!")
            self.destroy()

    def _run_silent_install(self):
        try:
            self._perform_installation()
        finally:
            self.destroy()


def main():
    silent = ("--silent" in sys.argv) or ("/S" in sys.argv)
    no_launch = ("--no-launch" in sys.argv)
    target_dir = None
    for i, arg in enumerate(sys.argv):
        if arg in ("--dir", "-d", "/D") and i + 1 < len(sys.argv):
            target_dir = sys.argv[i + 1]
    wizard = SetupWizard(silent=silent, target_dir=target_dir, no_launch=no_launch)
    wizard.mainloop()


if __name__ == "__main__":
    main()
'''


def generate_installer_wizard_script(
    installer_src_dir: Path,
    app_name: str,
    version: str,
    publisher: str,
    description: str,
) -> Path:
    """Generates the Python script for the standalone Windows graphical setup wizard."""
    installer_src_dir.mkdir(parents=True, exist_ok=True)
    script_path = installer_src_dir / "setup_wizard.py"

    code = (
        WIZARD_TEMPLATE
        .replace("__APP_NAME__", app_name)
        .replace("__APP_VERSION__", version)
        .replace("__APP_PUBLISHER__", publisher)
        .replace("__APP_DESC__", description)
    )
    script_path.write_text(code, encoding="utf-8")
    return script_path


def build_installer_executable(
    app_dir: Path,
    release_dir: Path,
    icon_path: Path,
) -> Path:
    """Creates a single-file Windows setup installer executable."""
    Log.header("Packaging Variant 2: Standalone Setup Installer (.exe)")

    installer_work_dir = BUILD_DIR / "installer_src"
    installer_work_dir.mkdir(parents=True, exist_ok=True)

    # 1. Compress app into internal payload
    payload_zip = installer_work_dir / "app_payload.zip"
    if not payload_zip.exists() or payload_zip.stat().st_size == 0:
        Log.info("Preparing internal application payload for installer...")
        start_payload = time.time()
        with zipfile.ZipFile(payload_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for root, _, files in os.walk(app_dir):
                for file in files:
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(app_dir)
                    zf.write(file_path, arcname=str(rel_path))
        Log.info(f"Payload prepared ({payload_zip.stat().st_size / (1024 * 1024):.1f} MB in {time.time() - start_payload:.1f}s)")
    else:
        Log.info(f"Reusing existing installer payload archive ({payload_zip.stat().st_size / (1024 * 1024):.1f} MB)")

    # 2. Generate setup wizard script
    wizard_script = generate_installer_wizard_script(
        installer_work_dir,
        APP_NAME,
        __version__,
        PUBLISHER,
        APP_DESCRIPTION,
    )

    # 3. Compile standalone installer with PyInstaller
    installer_exe_name = f"{APP_NAME}-v{__version__}-windows-x64-installer"
    installer_dist_dir = BUILD_DIR / "installer_dist"

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name",
        installer_exe_name,
        "--onefile",
        "--windowed",
        "--noconfirm",
        "--clean",
        f"--icon={icon_path}",
        "--distpath",
        str(installer_dist_dir),
        "--workpath",
        str(BUILD_DIR / "work_installer"),
        "--specpath",
        str(BUILD_DIR),
        "--add-data",
        f"{payload_zip};.",
        str(wizard_script),
    ]

    Log.info("Compiling standalone setup wizard executable...")
    start_time = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))

    if res.returncode != 0:
        raise RuntimeError(f"Failed to build installer executable:\n{res.stderr[-2000:]}")

    compiled_installer = installer_dist_dir / f"{installer_exe_name}.exe"
    target_installer = release_dir / f"{installer_exe_name}.exe"

    shutil.copy2(compiled_installer, target_installer)
    elapsed = time.time() - start_time
    size_mb = target_installer.stat().st_size / (1024 * 1024)
    Log.success(f"Setup Installer created ({size_mb:.1f} MB, {elapsed:.1f}s): {target_installer}")
    return target_installer


# -----------------------------------------------------------------------------
# Variant 2 (Bonus): Inno Setup Script Generator & Compiler
# -----------------------------------------------------------------------------
def generate_inno_setup_script(app_dir: Path, release_dir: Path, icon_path: Path) -> Path:
    """Generates an Inno Setup .iss compiler script and attempts compilation if ISCC is installed."""
    Log.header("Generating Inno Setup Script (.iss)")

    iss_path = release_dir / f"{APP_NAME}-v{__version__}.iss"
    inno_script_content = f"""[Setup]
AppId={{{{E8175E26-3C60-4FB4-B4A4-71289196E5CA}}}}
AppName={APP_NAME}
AppVersion={__version__}
AppVerName={APP_NAME} v{__version__}
AppPublisher={PUBLISHER}
DefaultDirName={{userappdata}}\\{APP_NAME}
DisableDirPage=no
DefaultGroupName={APP_NAME}
DisableProgramGroupPage=yes
OutputBaseFilename={APP_NAME}-v{__version__}-inno-setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
SetupIconFile={icon_path.resolve()}
UninstallDisplayIcon={{app}}\\{APP_NAME}.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{{cm:CreateDesktopIcon}}"; GroupDescription: "{{cm:AdditionalIcons}}"

[Files]
Source: "{app_dir.resolve()}\\*"; DestDir: "{{app}}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{{group}}\\{APP_NAME}"; Filename: "{{app}}\\{APP_NAME}.exe"
Name: "{{autodesktop}}\\{APP_NAME}"; Filename: "{{app}}\\{APP_NAME}.exe"; Tasks: desktopicon

[Run]
Filename: "{{app}}\\{APP_NAME}.exe"; Description: "{{cm:LaunchProgram,{APP_NAME}}}"; Flags: nowait postinstall skipifsilent
"""
    iss_path.write_text(inno_script_content, encoding="utf-8")
    Log.success(f"Inno Setup script written: {iss_path}")

    # Check for Inno Setup compiler
    iscc_candidates = [
        shutil.which("iscc"),
        r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        r"C:\Program Files\Inno Setup 6\ISCC.exe",
        r"C:\Program Files (x86)\Inno Setup 5\ISCC.exe",
    ]
    iscc_exe = next((p for p in iscc_candidates if p and os.path.exists(p)), None)

    if iscc_exe:
        Log.info(f"Found Inno Setup compiler at {iscc_exe}. Compiling Inno Setup EXE...")
        try:
            cmd = [str(iscc_exe), f"/O{release_dir.resolve()}", str(iss_path.resolve())]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                Log.success(f"Compiled Inno Setup executable into {release_dir}")
            else:
                Log.warn(f"Inno compilation failed:\n{res.stderr}")
        except Exception as e:
            Log.warn(f"Inno Setup execution error: {e}")
    else:
        Log.info("Inno Setup (iscc) not found on system; standalone Python installer was generated.")

    return iss_path


# -----------------------------------------------------------------------------
# Checksum Generation
# -----------------------------------------------------------------------------
def compute_sha256(file_path: Path) -> str:
    """Calculates SHA-256 hexadecimal digest for a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_checksums(release_dir: Path) -> Path:
    """Generates a SHA256SUMS.txt file for all release distribution artifacts."""
    checksums_file = release_dir / "SHA256SUMS.txt"
    entries = []

    for item in sorted(release_dir.glob("*")):
        if item.is_file() and item.name != "SHA256SUMS.txt" and not item.name.endswith(".iss"):
            digest = compute_sha256(item)
            entries.append(f"{digest}  {item.name}")

    if entries:
        checksums_file.write_text("\n".join(entries) + "\n", encoding="utf-8")
        Log.success(f"SHA-256 checksums recorded at {checksums_file.name}")

    return checksums_file


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Build and Package {APP_NAME} for Windows.")
    parser.add_argument("--clean", action="store_true", help="Clean build and dist directories before starting.")
    parser.add_argument("--recompile", action="store_true", help="Force recompilation of application core.")
    parser.add_argument("--skip-portable", action="store_true", help="Skip generating the portable ZIP archive.")
    parser.add_argument("--skip-installer", action="store_true", help="Skip generating the standalone installer executable.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RELEASE_DIR, help="Destination directory for release files.")
    parser.add_argument("--debug", action="store_true", help="Build with console window enabled for debugging.")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    start_total_time = time.time()

    Log.header(f"{APP_NAME} Release Builder v{__version__}")
    Log.info(f"Platform: {sys.platform} | Python: {sys.version.split()[0]}")
    Log.info(f"Target Output Directory: {args.output_dir.resolve()}")

    # 1. Clean previous builds if requested
    if args.clean:
        Log.info("Cleaning previous build and dist directories...")
        for p in (BUILD_DIR, DIST_DIR):
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)
        Log.success("Clean completed.")

    # 2. Asset Preparation
    icon_path = BUILD_DIR / "app_icon.ico"
    generate_application_icon(icon_path)

    # 3. Compile Core Application (Onedir)
    app_dir = compile_core_application(icon_path, debug=args.debug, force=args.recompile)

    # 4. Generate Variant 1: Portable ZIP
    portable_zip_path = None
    if not args.skip_portable:
        portable_zip_path = build_portable_zip(app_dir, args.output_dir, force=args.recompile)

    # 5. Generate Variant 2: Standalone Setup Installer (.exe)
    installer_exe_path = None
    if not args.skip_installer:
        installer_exe_path = build_installer_executable(app_dir, args.output_dir, icon_path)

    # 6. Generate Inno Setup Script (.iss)
    generate_inno_setup_script(app_dir, args.output_dir, icon_path)

    # 7. Generate Checksums
    generate_checksums(args.output_dir)

    # 8. Print Summary
    total_elapsed = time.time() - start_total_time
    Log.header("Release Build Summary")
    print(f"  Application:       {APP_NAME} v{__version__}")
    print(f"  Publisher:         {PUBLISHER}")
    print(f"  Total Build Time:  {total_elapsed:.1f}s")
    print(f"  Artifacts Location: {args.output_dir.resolve()}")
    print("-" * 75)

    for item in sorted(args.output_dir.glob("*")):
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  - {item.name:<48} ({size_mb:>6.2f} MB)")

    print("-" * 75)
    Log.success("All build artifacts generated successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
