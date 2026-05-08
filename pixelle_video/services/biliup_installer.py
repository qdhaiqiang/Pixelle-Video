# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Biliup-rs Auto-Installer

Automatically downloads and installs biliup-rs from GitHub Releases
if not already present in PATH or project tools directory.

Supported platforms:
- macOS (x86_64, arm64)
- Linux (x86_64, aarch64, arm)
- Windows (x86_64)
"""

import os
import platform
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger


# GitHub release info
GITHUB_REPO = "biliup/biliup-rs"
DEFAULT_VERSION = "0.2.4"


def get_tools_dir() -> Path:
    """Get the project-local tools directory."""
    project_root = Path(__file__).parent.parent.parent
    tools_dir = project_root / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    return tools_dir


def _get_release_filename(version: str) -> Optional[tuple[str, str]]:
    """
    Return (release_filename, archive_type) for this platform.

    Release filenames on GitHub:
      biliupR-v0.2.4-x86_64-macos.tar.xz
      biliupR-v0.2.4-aarch64-macos.tar.xz
      biliupR-v0.2.4-x86_64-linux.tar.xz
      biliupR-v0.2.4-x86_64-linux-musl.tar.xz
      biliupR-v0.2.4-aarch64-linux.tar.xz
      biliupR-v0.2.4-arm-linux.tar.xz
      biliupR-v0.2.4-x86_64-windows.zip
    """
    system = platform.system().lower()
    machine = platform.machine().lower()

    # Normalize architecture
    if machine in ("x86_64", "amd64"):
        arch = "x86_64"
    elif machine in ("arm64", "aarch64"):
        arch = "aarch64"
    elif machine in ("armv7l", "arm"):
        arch = "arm"
    else:
        arch = machine

    if system == "darwin":
        # macOS: biliupR-v0.2.4-x86_64-macos.tar.xz
        #        biliupR-v0.2.4-aarch64-macos.tar.xz
        return (f"biliupR-v{version}-{arch}-macos.tar.xz", "tar.xz")

    elif system == "linux":
        # Linux: prefer glibc version, fallback to musl
        #        biliupR-v0.2.4-x86_64-linux.tar.xz
        #        biliupR-v0.2.4-aarch64-linux.tar.xz
        #        biliupR-v0.2.4-arm-linux.tar.xz
        #        biliupR-v0.2.4-x86_64-linux-musl.tar.xz
        if arch == "x86_64":
            # Try glibc first, musl as fallback
            return (f"biliupR-v{version}-x86_64-linux.tar.xz", "tar.xz")
        return (f"biliupR-v{version}-{arch}-linux.tar.xz", "tar.xz")

    elif system == "windows":
        return (f"biliupR-v{version}-x86_64-windows.zip", "zip")

    return None


def find_biliup() -> Optional[str]:
    """Find biliup binary in PATH or project tools directory."""
    # Check system PATH
    biliup_path = shutil.which("biliup")
    if biliup_path:
        return biliup_path

    # Check project tools directory (also check nested dirs from tar extraction)
    tools_dir = get_tools_dir()
    candidates = [
        tools_dir / "biliup",
        tools_dir / "biliup.exe",
    ]
    # Also search one level deep (tar extracts into subdirs)
    for subdir in tools_dir.iterdir():
        if subdir.is_dir():
            candidates.append(subdir / "biliup")
            candidates.append(subdir / "biliup.exe")

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return str(candidate)

    return None


def install_biliup(version: str = DEFAULT_VERSION, force: bool = False) -> str:
    """
    Download and install biliup-rs to the project tools directory.

    Returns:
        Absolute path to the installed biliup binary
    """
    existing = find_biliup()
    if existing and not force:
        logger.info(f"biliup already installed: {existing}")
        return existing

    release_info = _get_release_filename(version)
    if not release_info:
        raise RuntimeError(f"Unsupported platform: {platform.system()} {platform.machine()}")

    filename, archive_type = release_info
    download_url = f"https://github.com/{GITHUB_REPO}/releases/download/v{version}/{filename}"
    tools_dir = get_tools_dir()
    archive_path = tools_dir / filename

    logger.info(f"Downloading biliup-rs v{version}: {download_url}")

    # Download
    try:
        with httpx.stream("GET", download_url, follow_redirects=True, timeout=120) as response:
            response.raise_for_status()
            with open(archive_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)
        logger.info(f"Downloaded: {archive_path}")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            # For Linux x86_64, try musl fallback
            if "linux.tar.xz" in filename and "-musl" not in filename:
                musl_filename = filename.replace("-linux.tar.xz", "-linux-musl.tar.xz")
                musl_url = f"https://github.com/{GITHUB_REPO}/releases/download/v{version}/{musl_filename}"
                logger.info(f"Glibc build not found, trying musl: {musl_url}")
                try:
                    archive_path = tools_dir / musl_filename
                    with httpx.stream("GET", musl_url, follow_redirects=True, timeout=120) as response:
                        response.raise_for_status()
                        with open(archive_path, "wb") as f:
                            for chunk in response.iter_bytes(chunk_size=8192):
                                f.write(chunk)
                    logger.info(f"Downloaded musl build: {archive_path}")
                except Exception as e2:
                    raise RuntimeError(
                        f"biliup-rs v{version} not found for this platform. "
                        f"Please install manually from https://github.com/{GITHUB_REPO}/releases"
                    )
            else:
                raise RuntimeError(
                    f"biliup-rs v{version} not found for this platform. "
                    f"Please install manually from https://github.com/{GITHUB_REPO}/releases"
                )
        raise RuntimeError(f"Download failed: {e}")
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}")

    # Extract
    try:
        if archive_type == "tar.xz":
            with tarfile.open(archive_path, "r:xz") as tf:
                tf.extractall(tools_dir)
        elif archive_type == "zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(tools_dir)
        logger.info(f"Extracted to: {tools_dir}")
    except Exception as e:
        raise RuntimeError(f"Extraction failed: {e}")
    finally:
        if archive_path.exists():
            archive_path.unlink()

    # Find binary
    binary = _find_binary_in_tools(tools_dir)
    if not binary:
        raise RuntimeError("biliup binary not found after extraction")

    # Make executable (Unix)
    if platform.system() != "Windows":
        binary.chmod(binary.stat().st_mode | 0o111)
        logger.info(f"Made executable: {binary}")
        # macOS Gatekeeper: remove quarantine
        try:
            subprocess.run(
                ["xattr", "-d", "com.apple.quarantine", str(binary)],
                capture_output=True,
                check=False,
            )
        except FileNotFoundError:
            pass

    logger.success(f"✅ biliup installed: {binary}")
    return str(binary)


def _find_binary_in_tools(tools_dir: Path) -> Optional[Path]:
    """Find the biliup binary in tools dir or its subdirectories."""
    # Direct
    for name in ("biliup", "biliup.exe"):
        p = tools_dir / name
        if p.exists() and p.is_file():
            return p

    # One level deep (tar extracts into subdir like biliupR-v0.2.4-aarch64-macos/)
    for subdir in tools_dir.iterdir():
        if not subdir.is_dir():
            continue
        for name in ("biliup", "biliup.exe"):
            p = subdir / name
            if p.exists() and p.is_file():
                return p

    return None


def ensure_biliup(version: str = DEFAULT_VERSION) -> str:
    """Ensure biliup is available. Install if missing."""
    existing = find_biliup()
    if existing:
        return existing
    return install_biliup(version=version)


def get_biliup_version(binary_path: str) -> Optional[str]:
    """Get installed biliup version."""
    try:
        result = subprocess.run(
            [binary_path, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None
