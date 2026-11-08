#!/usr/bin/env python3
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

"""Build and packaging script for DeckSpool.

Compiles the DeckSpool application into two official release deliverables:
  1. Standalone Portable ZIP archive (extract & run, zero installation required,
     bundled with portable quick-start guide).
  2. Native Windows Installer (.msi) with interactive destination folder selection
     ("Change..." directory browser), Start Menu integration (Windows Search indexing),
     Desktop shortcut option, post-install launch ("Run DeckSpool"), and brief application description.
  3. Cryptographic SHA-256 checksums (SHA256SUMS.txt).

Usage:
  python build_release.py [options]

Options:
  --clean           Remove previous build/dist artifacts before compiling.
  --recompile       Force recompilation of application core.
  --skip-portable   Skip generating the portable ZIP archive.
  --skip-msi        Skip generating the native Windows Installer (.msi).
  --per-machine     Install MSI to Program Files instead of user profile.
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
    from deckspool import __author__, __version__
except ImportError:
    __version__ = "1.0.0"
    __author__ = "DeckSpool Team"

APP_NAME = "DeckSpool"
APP_TITLE = "DeckSpool - Presentation Slide Extractor"
APP_DESCRIPTION = "Extract unique presentation slides from YouTube videos and local files into clean, searchable PDFs."
PUBLISHER = "Vignesh Rajan (vector49)"

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
        PROJECT_ROOT / "deckspool" / "assets" / "deckspool_logo.png",
        PROJECT_ROOT / "docs" / "images" / "deckspool_logo.png",
        PROJECT_ROOT / "deckspool" / "assets" / "app_icon.png",
        PROJECT_ROOT / "Icon" / "DeckSpool Logo.png",
        PROJECT_ROOT / "docs" / "images" / "logo.png",
        PROJECT_ROOT / "deckspool" / "assets" / "logo.png",
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

    bundled_ico = PROJECT_ROOT / "deckspool" / "assets" / "app_icon.ico"
    if bundled_ico.exists() and output_path != bundled_ico:
        shutil.copy2(bundled_ico, output_path)
        Log.info(f"Using bundled icon from {bundled_ico.relative_to(PROJECT_ROOT)}")
        return output_path

    logo_path = resolve_logo_path()
    sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]

    if logo_path and logo_path.exists():
        Log.info(f"Generating icon from {logo_path.relative_to(PROJECT_ROOT)}...")
        img = Image.open(logo_path).convert("RGBA")
    else:
        Log.warn("No logo image found; generating default fallback icon...")
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
        f"{PROJECT_ROOT / 'deckspool' / 'assets'};deckspool/assets",
        f"{PROJECT_ROOT / 'deckspool' / 'models'};deckspool/models",
    ]
    if (PROJECT_ROOT / "docs" / "images").exists():
        datas.append(f"{PROJECT_ROOT / 'docs' / 'images'};docs/images")

    pyinstaller_args = [
        sys.executable,
        "-u",
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
        "deckspool",
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
    ])

    # Exclude heavy unrelated packages to keep build time fast and release footprint compact
    excludes = [
        "torch",
        "torchvision",
        "torchaudio",
        "pandas",
        "matplotlib",
        "lxml",
        "pikepdf",
        "IPython",
        "jupyter",
        "pytest",
        "unittest",
        "tkinter",
        "setuptools",
        "pip",
    ]
    for ex in excludes:
        pyinstaller_args.extend(["--exclude-module", ex])

    pyinstaller_args.append(str(main_script))

    Log.info(f"Running PyInstaller onedir build for {APP_NAME} v{__version__}...")
    start_time = time.time()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        pyinstaller_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(PROJECT_ROOT),
        env=env,
    )

    last_heartbeat = time.time()
    if process.stdout:
        for line in iter(process.stdout.readline, ""):
            line_str = line.strip()
            if any(k in line_str for k in (
                "Analyzing",
                "Processing module hooks",
                "Building PKG",
                "Building EXE",
                "Building COLLECT",
                "completed successfully",
                "checking Analysis",
            )):
                Log.info(line_str)
                last_heartbeat = time.time()
            elif time.time() - last_heartbeat > 5.0 and line_str:
                Log.info(f"Compiling... {line_str[:65]}")
                last_heartbeat = time.time()

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
# Deliverable 1: Portable ZIP Archive Builder
# -----------------------------------------------------------------------------
def build_portable_zip(app_dir: Path, release_dir: Path, force: bool = False) -> Path:
    """Packages the compiled onedir application into a portable ZIP archive."""
    Log.header("Packaging Deliverable 1: Portable ZIP Archive")

    release_dir.mkdir(parents=True, exist_ok=True)
    zip_filename = f"{APP_NAME}-v{__version__}-windows-x64-portable.zip"
    zip_path = release_dir / zip_filename

    if zip_path.exists() and not force:
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        Log.info(f"Using existing portable ZIP archive ({size_mb:.1f} MB) at {zip_path.name}")
        return zip_path

    if zip_path.exists():
        zip_path.unlink()

    # Quick-start instructions and application overview for portable users
    readme_text = f"""========================================================================
{APP_NAME} v{__version__} - Portable Release (Windows x64)
{APP_DESCRIPTION}
Publisher: {PUBLISHER}
========================================================================

PORTABLE USAGE & QUICK START:
1. Extract the contents of this ZIP archive to any folder or USB drive.
2. Double-click '{APP_NAME}.exe' to launch the application immediately.
3. Completely self-contained: zero installation, no registry modifications,
   and no administrator privileges required.
4. To remove, simply delete the extracted folder.

SYSTEM REQUIREMENTS:
- Windows 10 (64-bit) or Windows 11
- Active Internet connection (for YouTube video stream downloading)

ABOUT {APP_NAME.upper()}:
{APP_NAME} is an automated presentation slide extractor that converts video
lectures, workshops, and webinars into clean, searchable PDF documents.

KEY CAPABILITIES:
- Automated scene and slide transition detection
- Computer vision deduplication (SSIM and perceptual hashing)
- Intelligent presenter/speaker occlusion detection and removal
- YouTube video stream ingestion and local video file processing
- Direct PDF compilation with vector text and multi-page indexing

PROJECT REPOSITORY & SUPPORT:
- Project Homepage: https://github.com/vector49/DeckSpool
- Issue Tracker:    https://github.com/vector49/DeckSpool/issues

Copyright (c) 2026 {PUBLISHER}. All rights reserved.
Distributed under the Apache License, Version 2.0.
========================================================================
"""
    # Remove any lingering PORTABLE_README.txt on disk so it never leaks into other builds
    portable_readme_on_disk = app_dir / "PORTABLE_README.txt"
    if portable_readme_on_disk.exists():
        portable_readme_on_disk.unlink()

    Log.info(f"Compressing {app_dir.name} into {zip_path.name}...")
    start_time = time.time()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Embed PORTABLE_README.txt exclusively inside the portable zip archive
        zf.writestr(f"{app_dir.name}/PORTABLE_README.txt", readme_text)

        # Bundle LICENSE and NOTICE in portable archive
        for doc_name in ("LICENSE", "NOTICE"):
            doc_file = PROJECT_ROOT / doc_name
            if doc_file.exists():
                zf.write(doc_file, arcname=f"{app_dir.name}/{doc_name}")

        total_files = sum(len(files) for _, _, files in os.walk(app_dir))
        count = 0

        for root, _, files in os.walk(app_dir):
            for file in files:
                if file == "PORTABLE_README.txt":
                    continue  # Ensure portable readme is never duplicated from disk
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
# Deliverable 2: Native Windows Installer (.msi) via WiX Toolset
# -----------------------------------------------------------------------------
def get_wix_tools() -> Dict[str, Path]:
    """Locates or automatically prepares the WiX Toolset binaries (candle, light, heat)."""
    # 1. Check system PATH or standard installations
    tools = {}
    for name in ("candle", "light", "heat"):
        found = shutil.which(name)
        if found:
            tools[name] = Path(found)

    if len(tools) == 3:
        return tools

    # 2. Check local application cache (%LOCALAPPDATA%\WiXToolset\wix311)
    local_app_data = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    wix_cache = Path(local_app_data) / "WiXToolset" / "wix311"
    candle_exe = wix_cache / "candle.exe"
    light_exe = wix_cache / "light.exe"
    heat_exe = wix_cache / "heat.exe"

    if candle_exe.exists() and light_exe.exists() and heat_exe.exists():
        return {"candle": candle_exe, "light": light_exe, "heat": heat_exe}

    # 3. Automatically download portable WiX binaries if not present
    Log.info("WiX toolset not detected; downloading portable WiX 3.11 binaries (~34 MB)...")
    wix_cache.mkdir(parents=True, exist_ok=True)
    zip_path = wix_cache / "wix_binaries.zip"
    url = "https://github.com/wixtoolset/wix3/releases/download/wix3112rtm/wix311-binaries.zip"

    import urllib.request
    urllib.request.urlretrieve(url, zip_path)
    Log.info("Extracting WiX compiler tools...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(wix_cache)
    if zip_path.exists():
        zip_path.unlink()

    Log.success("WiX compiler tools ready.")
    return {"candle": candle_exe, "light": light_exe, "heat": heat_exe}


def build_msi_installer(
    app_dir: Path,
    release_dir: Path,
    icon_path: Path,
    per_machine: bool = False,
) -> Path:
    """Builds a standard native Windows Installer (.msi) package for DeckSpool with custom directory selection & shortcut controls."""
    Log.header("Packaging Deliverable 2: Native Windows Installer (.msi)")
    release_dir.mkdir(parents=True, exist_ok=True)

    tools = get_wix_tools()
    wix_work_dir = BUILD_DIR / "wix_work"
    wix_work_dir.mkdir(parents=True, exist_ok=True)

    msi_name = f"{APP_NAME}-v{__version__}-windows-x64-installer.msi"
    msi_path = release_dir / msi_name

    # 1. Clean portable readme and synthesize brief installer application description
    portable_readme_on_disk = app_dir / "PORTABLE_README.txt"
    if portable_readme_on_disk.exists():
        portable_readme_on_disk.unlink()

    app_readme = app_dir / "README.txt"
    app_readme_text = f"""========================================================================
{APP_NAME} v{__version__}
{APP_DESCRIPTION}
Publisher: {PUBLISHER}
========================================================================

ABOUT {APP_NAME.upper()}:
{APP_NAME} is an automated presentation slide extractor that converts video
lectures, workshops, and webinars into clean, searchable PDF documents.

KEY CAPABILITIES:
- Automated scene and slide transition detection
- Computer vision deduplication (SSIM and perceptual hashing)
- Intelligent presenter/speaker occlusion detection and removal
- YouTube video stream ingestion and local video file processing
- Direct PDF compilation with vector text and multi-page indexing

QUICK LAUNCH:
- Press the Windows key, type "{APP_NAME}", and press Enter.
- Launch from the Desktop shortcut (if enabled during installation).
- Run directly from the installation directory:
  {APP_NAME}.exe

DOCUMENTATION & SUPPORT:
- Project Homepage: https://github.com/vector49/DeckSpool
- Issue Tracker:    https://github.com/vector49/DeckSpool/issues

Copyright (c) 2026 {PUBLISHER}. All rights reserved.
Distributed under the Apache License, Version 2.0.
========================================================================
"""
    app_readme.write_text(app_readme_text, encoding="utf-8")

    # Copy official LICENSE and NOTICE files into app_dir for installer harvest
    for doc_name in ("LICENSE", "NOTICE"):
        doc_src = PROJECT_ROOT / doc_name
        if doc_src.exists():
            shutil.copy2(doc_src, app_dir / doc_name)

    # 2. Harvest application directory tree using heat.exe
    files_wxs = wix_work_dir / "files.wxs"
    Log.info(f"Harvesting directory tree for {app_dir.name}...")
    cmd_heat = [
        str(tools["heat"]),
        "dir", str(app_dir.resolve()),
        "-cg", "DeckSpoolComponents",
        "-dr", "INSTALLFOLDER",
        "-scom", "-sreg", "-sfrag", "-srd",
        "-gg",
        "-var", "var.SourceDir",
        "-out", str(files_wxs),
    ]
    res_heat = subprocess.run(cmd_heat, capture_output=True, text=True)
    if res_heat.returncode != 0:
        raise RuntimeError(f"WiX heat.exe failed:\n{res_heat.stderr}")

    # 3. Parse harvested files.wxs to locate the File ID of DeckSpool.exe for launch
    exe_file_id = None
    import xml.etree.ElementTree as ET
    wix_ns = {"wix": "http://schemas.microsoft.com/wix/2006/wi"}
    try:
        tree = ET.parse(files_wxs)
        for file_elem in tree.findall(".//wix:File", wix_ns):
            src = file_elem.attrib.get("Source", "")
            if src.endswith(f"\\{APP_NAME}.exe") or src.endswith(f"/{APP_NAME}.exe"):
                exe_file_id = file_elem.attrib.get("Id")
                break
    except Exception as e:
        Log.warn(f"Could not parse files.wxs for executable File ID ({e})")

    launch_target = f"[#{exe_file_id}]" if exe_file_id else f"[INSTALLFOLDER]{APP_NAME}.exe"

    # 4. Generate License RTF for installer
    license_rtf = wix_work_dir / "License.rtf"
    license_rtf.write_text(
        r"{\rtf1\ansi\deff0 {\fonttbl {\f0 Segoe UI;}}\f0\fs18 "
        rf"\b {APP_NAME} v{__version__} - Apache License, Version 2.0\b0\par\par"
        rf"Copyright 2026 {PUBLISHER}\par"
        rf"https://github.com/vector49/DeckSpool\par\par"
        r"Licensed under the Apache License, Version 2.0 (the \"License\"); "
        r"you may not use this file except in compliance with the License. "
        r"You may obtain a copy of the License at:\par\par"
        r"    http://www.apache.org/licenses/LICENSE-2.0\par\par"
        r"Unless required by applicable law or agreed to in writing, software "
        r"distributed under the License is distributed on an \"AS IS\" BASIS, "
        r"WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. "
        r"See the License for the specific language governing permissions and "
        r"limitations under the License.\par\par"
        r"\b TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION\b0\par\par"
        r"1. Definitions: \"License\" shall mean the terms and conditions for use, reproduction, and distribution as defined by Sections 1 through 9.\par\par"
        r"2. Grant of Copyright License: Each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare Derivative Works of, publicly display, publicly perform, sublicense, and distribute the Work.\par\par"
        r"3. Grant of Patent License: Each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable patent license to make, have made, use, sell, and transfer the Work.\par\par"
        r"4. Redistribution: You may reproduce and distribute copies provided that You give any other recipients a copy of this License, cause modified files to carry prominent notices, and retain all copyright, patent, trademark, and attribution notices.\par\par"
        r"6. Trademarks: This License does not grant permission to use the trade names, trademarks, service marks, or product names of the Licensor.\par\par"
        r"7. Disclaimer of Warranty: Licensor provides the Work on an \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.\par\par"
        r"Please refer to the LICENSE and NOTICE files installed with this software for complete legal terms and third-party attributions.\par}",
        encoding="utf-8"
    )

    # 5. Product.wxs definition with interactive directory selection, shortcut checkboxes & launch
    install_scope = "perMachine" if per_machine else "perUser"
    root_folder = "ProgramFiles64Folder" if per_machine else "LocalAppDataFolder"

    product_wxs = wix_work_dir / "Product.wxs"
    product_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Wix xmlns="http://schemas.microsoft.com/wix/2006/wi">
  <Product Id="*" 
           Name="{APP_NAME}" 
           Language="1033" 
           Version="{__version__}" 
           Manufacturer="{PUBLISHER}" 
           UpgradeCode="E8175E26-3C60-4FB4-B4A4-71289196E5CA">
    
    <Package InstallerVersion="500" 
             Compressed="yes" 
             InstallScope="{install_scope}" 
             Description="{APP_NAME} - {APP_DESCRIPTION}" 
             Comments="DeckSpool Windows Installer Package" />

    <MajorUpgrade DowngradeErrorMessage="A newer version of [ProductName] is already installed." />
    <MediaTemplate EmbedCab="yes" />

    <Icon Id="AppIcon.ico" SourceFile="{icon_path.resolve()}" />
    <Property Id="ARPPRODUCTICON" Value="AppIcon.ico" />
    <Property Id="ARPHELPLINK" Value="https://github.com/vector49/DeckSpool" />

    <!-- Configuration properties for user customization -->
    <Property Id="WIXUI_INSTALLDIR" Value="INSTALLFOLDER" />
    <Property Id="INSTALLDESKTOPSHORTCUT" Value="1" />
    <Property Id="INSTALLSTARTMENUSHORTCUT" Value="1" />

    <!-- Post-installation Launch Feature ("Run DeckSpool") -->
    <Property Id="WIXUI_EXITDIALOGOPTIONALCHECKBOXTEXT" Value="Run {APP_NAME}" />
    <Property Id="WIXUI_EXITDIALOGOPTIONALCHECKBOX" Value="1" />
    <Property Id="WixShellExecTarget" Value="{launch_target}" />
    <CustomAction Id="LaunchApplication" BinaryKey="WixCA" DllEntry="WixShellExec" Impersonate="yes" />

    <Directory Id="TARGETDIR" Name="SourceDir">
      <Directory Id="{root_folder}">
        <Directory Id="INSTALLFOLDER" Name="{APP_NAME}" />
      </Directory>
      
      <Directory Id="ProgramMenuFolder">
        <Directory Id="ApplicationProgramsFolder" Name="{APP_NAME}" />
      </Directory>
      
      <Directory Id="DesktopFolder" Name="Desktop" />
    </Directory>

    <!-- Start Menu Shortcut Component (conditioned on INSTALLSTARTMENUSHORTCUT) -->
    <DirectoryRef Id="ApplicationProgramsFolder">
      <Component Id="ApplicationStartMenuShortcutComponent" Guid="A762194D-9345-4277-B178-500DCA96338B">
        <Condition>INSTALLSTARTMENUSHORTCUT = 1</Condition>
        <Shortcut Id="ApplicationStartMenuShortcut"
                  Name="{APP_NAME}"
                  Description="{APP_DESCRIPTION}"
                  Target="[INSTALLFOLDER]{APP_NAME}.exe"
                  WorkingDirectory="INSTALLFOLDER"
                  Icon="AppIcon.ico" />
        <RemoveFolder Id="CleanUpShortcutsDir" On="uninstall" />
        <RegistryValue Root="HKCU" Key="Software\\{APP_NAME}" Name="start_menu_shortcut" Type="integer" Value="1" KeyPath="yes" />
      </Component>
    </DirectoryRef>

    <!-- Desktop Shortcut Component (conditioned on INSTALLDESKTOPSHORTCUT) -->
    <DirectoryRef Id="DesktopFolder">
      <Component Id="ApplicationDesktopShortcutComponent" Guid="B827194D-9345-4277-B178-500DCA96338C">
        <Condition>INSTALLDESKTOPSHORTCUT = 1</Condition>
        <Shortcut Id="AppDesktopShortcut"
                  Name="{APP_NAME}"
                  Description="{APP_DESCRIPTION}"
                  Target="[INSTALLFOLDER]{APP_NAME}.exe"
                  WorkingDirectory="INSTALLFOLDER"
                  Icon="AppIcon.ico" />
        <RegistryValue Root="HKCU" Key="Software\\{APP_NAME}" Name="desktop_shortcut" Type="integer" Value="1" KeyPath="yes" />
      </Component>
    </DirectoryRef>

    <Feature Id="MainApplication" Title="{APP_NAME}" Level="1">
      <ComponentGroupRef Id="DeckSpoolComponents" />
      <ComponentRef Id="ApplicationStartMenuShortcutComponent" />
      <ComponentRef Id="ApplicationDesktopShortcutComponent" />
    </Feature>

    <!-- Standard Windows Installer UI Sequence with Directory Selection and Shortcut Checkboxes -->
    <UI Id="DeckSpool_UI">
      <TextStyle Id="WixUI_Font_Normal" FaceName="Tahoma" Size="8" />
      <TextStyle Id="WixUI_Font_Bigger" FaceName="Tahoma" Size="12" />
      <TextStyle Id="WixUI_Font_Title" FaceName="Tahoma" Size="9" Bold="yes" />

      <Property Id="DefaultUIFont" Value="WixUI_Font_Normal" />
      <Property Id="WixUI_Mode" Value="InstallDir" />

      <DialogRef Id="BrowseDlg" />
      <DialogRef Id="DiskCostDlg" />
      <DialogRef Id="ErrorDlg" />
      <DialogRef Id="FatalError" />
      <DialogRef Id="FilesInUse" />
      <DialogRef Id="MsiRMFilesInUse" />
      <DialogRef Id="PrepareDlg" />
      <DialogRef Id="ProgressDlg" />
      <DialogRef Id="ResumeDlg" />
      <DialogRef Id="UserExit" />

      <!-- Custom Destination Folder & Shortcut Options Dialog -->
      <Dialog Id="CustomInstallDirDlg" Width="370" Height="270" Title="!(loc.InstallDirDlg_Title)">
        <Control Id="Next" Type="PushButton" X="236" Y="243" Width="56" Height="17" Default="yes" Text="!(loc.WixUINext)" />
        <Control Id="Back" Type="PushButton" X="180" Y="243" Width="56" Height="17" Text="!(loc.WixUIBack)" />
        <Control Id="Cancel" Type="PushButton" X="304" Y="243" Width="56" Height="17" Cancel="yes" Text="!(loc.WixUICancel)">
          <Publish Event="SpawnDialog" Value="CancelDlg">1</Publish>
        </Control>

        <Control Id="Description" Type="Text" X="25" Y="23" Width="280" Height="15" Transparent="yes" NoPrefix="yes" Text="Choose installation folder and shortcut preferences." />
        <Control Id="Title" Type="Text" X="15" Y="6" Width="200" Height="15" Transparent="yes" NoPrefix="yes" Text="{{\\WixUI_Font_Title}}Destination &amp; Options" />
        <Control Id="BannerBitmap" Type="Bitmap" X="0" Y="0" Width="370" Height="44" TabSkip="no" Text="!(loc.InstallDirDlgBannerBitmap)" />
        <Control Id="BannerLine" Type="Line" X="0" Y="44" Width="370" Height="0" />
        <Control Id="BottomLine" Type="Line" X="0" Y="234" Width="370" Height="0" />

        <Control Id="FolderLabel" Type="Text" X="20" Y="60" Width="290" Height="20" NoPrefix="yes" Text="Install [ProductName] to:" />
        <Control Id="Folder" Type="PathEdit" X="20" Y="82" Width="320" Height="18" Property="WIXUI_INSTALLDIR" Indirect="yes" />
        <Control Id="ChangeFolder" Type="PushButton" X="20" Y="105" Width="56" Height="17" Text="&amp;Change..." />

        <Control Id="ShortcutGroupLabel" Type="Text" X="20" Y="140" Width="320" Height="14" Text="Shortcut options:" />
        <Control Id="DesktopShortcutCheckBox" Type="CheckBox" X="20" Y="160" Width="320" Height="17" Property="INSTALLDESKTOPSHORTCUT" CheckBoxValue="1" Text="Create a shortcut on Desktop" />
        <Control Id="StartMenuShortcutCheckBox" Type="CheckBox" X="20" Y="180" Width="320" Height="17" Property="INSTALLSTARTMENUSHORTCUT" CheckBoxValue="1" Text="Create a shortcut in Start Menu (Windows Search)" />
      </Dialog>

      <Publish Dialog="BrowseDlg" Control="OK" Event="DoAction" Value="WixUIValidatePath" Order="3">1</Publish>
      <Publish Dialog="BrowseDlg" Control="OK" Event="SpawnDialog" Value="InvalidDirDlg" Order="4"><![CDATA[WIXUI_INSTALLDIR_VALID<>"1"]]></Publish>

      <Publish Dialog="ExitDialog" Control="Finish" Event="DoAction" Value="LaunchApplication" Order="1">WIXUI_EXITDIALOGOPTIONALCHECKBOX = 1 and NOT Installed</Publish>
      <Publish Dialog="ExitDialog" Control="Finish" Event="EndDialog" Value="Return" Order="999">1</Publish>

      <Publish Dialog="WelcomeDlg" Control="Next" Event="NewDialog" Value="LicenseAgreementDlg">NOT Installed</Publish>
      <Publish Dialog="WelcomeDlg" Control="Next" Event="NewDialog" Value="VerifyReadyDlg">Installed AND PATCH</Publish>

      <Publish Dialog="LicenseAgreementDlg" Control="Back" Event="NewDialog" Value="WelcomeDlg">1</Publish>
      <Publish Dialog="LicenseAgreementDlg" Control="Next" Event="NewDialog" Value="CustomInstallDirDlg">LicenseAccepted = "1"</Publish>

      <Publish Dialog="CustomInstallDirDlg" Control="Back" Event="NewDialog" Value="LicenseAgreementDlg">NOT Installed</Publish>
      <Publish Dialog="CustomInstallDirDlg" Control="Next" Event="SetTargetPath" Value="[WIXUI_INSTALLDIR]" Order="1">1</Publish>
      <Publish Dialog="CustomInstallDirDlg" Control="Next" Event="DoAction" Value="WixUIValidatePath" Order="2">NOT WIXUI_DONTVALIDATEPATH</Publish>
      <Publish Dialog="CustomInstallDirDlg" Control="Next" Event="SpawnDialog" Value="InvalidDirDlg" Order="3"><![CDATA[NOT WIXUI_DONTVALIDATEPATH AND WIXUI_INSTALLDIR_VALID<>"1"]]></Publish>
      <Publish Dialog="CustomInstallDirDlg" Control="Next" Event="NewDialog" Value="VerifyReadyDlg" Order="4">WIXUI_DONTVALIDATEPATH OR WIXUI_INSTALLDIR_VALID="1"</Publish>
      <Publish Dialog="CustomInstallDirDlg" Control="ChangeFolder" Property="_BrowseProperty" Value="[WIXUI_INSTALLDIR]" Order="1">1</Publish>
      <Publish Dialog="CustomInstallDirDlg" Control="ChangeFolder" Event="SpawnDialog" Value="BrowseDlg" Order="2">1</Publish>

      <Publish Dialog="VerifyReadyDlg" Control="Back" Event="NewDialog" Value="CustomInstallDirDlg" Order="1">NOT Installed AND NOT PATCH</Publish>
      <Publish Dialog="VerifyReadyDlg" Control="Back" Event="NewDialog" Value="WelcomeDlg" Order="2">Installed AND PATCH</Publish>

      <Publish Dialog="MaintenanceWelcomeDlg" Control="Next" Event="NewDialog" Value="MaintenanceTypeDlg">1</Publish>
      <Publish Dialog="MaintenanceTypeDlg" Control="RepairButton" Event="NewDialog" Value="VerifyReadyDlg">1</Publish>
      <Publish Dialog="MaintenanceTypeDlg" Control="RemoveButton" Event="NewDialog" Value="VerifyReadyDlg">1</Publish>
      <Publish Dialog="MaintenanceTypeDlg" Control="Back" Event="NewDialog" Value="MaintenanceWelcomeDlg">1</Publish>
    </UI>

    <UIRef Id="WixUI_Common" />
    <WixVariable Id="WixUILicenseRtf" Value="{license_rtf.resolve()}" />
  </Product>
</Wix>
"""
    product_wxs.write_text(product_xml, encoding="utf-8")

    # 4. Compile with candle.exe
    Log.info("Compiling WiX XML sources with candle...")
    cmd_candle = [
        str(tools["candle"]),
        f"-dSourceDir={app_dir.resolve()}",
        "-arch", "x64",
        "-ext", "WixUIExtension",
        "-ext", "WixUtilExtension",
        str(product_wxs),
        str(files_wxs),
        "-out", str(wix_work_dir) + "\\",
    ]
    res_candle = subprocess.run(cmd_candle, capture_output=True, text=True)
    if res_candle.returncode != 0:
        raise RuntimeError(f"WiX candle.exe failed:\n{res_candle.stderr or res_candle.stdout}")

    # 5. Link into single MSI with light.exe
    Log.info("Linking MSI package with light...")
    start_light = time.time()
    cmd_light = [
        str(tools["light"]),
        "-ext", "WixUIExtension",
        "-ext", "WixUtilExtension",
        "-spdb",
        "-sice:ICE69", "-sice:ICE91", "-sice:ICE38", "-sice:ICE64", "-sice:ICE60",
        str(wix_work_dir / "Product.wixobj"),
        str(wix_work_dir / "files.wixobj"),
        "-out", str(msi_path),
    ]
    res_light = subprocess.run(cmd_light, capture_output=True, text=True)
    if res_light.returncode != 0:
        raise RuntimeError(f"WiX light.exe failed:\n{res_light.stderr or res_light.stdout}")

    size_mb = msi_path.stat().st_size / (1024 * 1024)
    elapsed = time.time() - start_light
    Log.success(f"Windows MSI Installer created ({size_mb:.1f} MB, {elapsed:.1f}s): {msi_path}")
    return msi_path


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
        if item.is_file() and item.name != "SHA256SUMS.txt" and not item.name.endswith(".wixpdb"):
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
    parser.add_argument("--skip-msi", action="store_true", help="Skip generating the native Windows Installer (.msi).")
    parser.add_argument("--per-machine", action="store_true", help="Install MSI to Program Files instead of user profile.")
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

    # 4. Generate Deliverable 1: Portable ZIP Archive (.zip)
    portable_zip_path = None
    if not args.skip_portable:
        portable_zip_path = build_portable_zip(app_dir, args.output_dir, force=args.recompile)

    # 5. Generate Deliverable 2: Native Windows Installer (.msi)
    msi_path = None
    if not args.skip_msi:
        msi_path = build_msi_installer(app_dir, args.output_dir, icon_path, per_machine=args.per_machine)

    # 6. Generate Checksums
    generate_checksums(args.output_dir)

    # 7. Print Summary
    total_elapsed = time.time() - start_total_time
    Log.header("Release Build Summary")
    print(f"  Application:        {APP_NAME} v{__version__}")
    print(f"  Publisher:          {PUBLISHER}")
    print(f"  Total Build Time:   {total_elapsed:.1f}s")
    print(f"  Artifacts Location: {args.output_dir.resolve()}")
    print("-" * 75)

    for item in sorted(args.output_dir.glob("*")):
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  - {item.name:<48} ({size_mb:>6.2f} MB)")

    print("-" * 75)
    Log.success("All build deliverables generated successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
