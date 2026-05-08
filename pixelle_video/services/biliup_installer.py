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
- Linux (x86_64, aarch64)
- Windows (x86_64)
"""

import os
import platform
import shutil
import subprocess
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
    # Project root: pixelle_video/services/ -> pixelle_video/ -> project root
    project_root = Path(__file__).parent.parent.parent
    tools_dir = project_root / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    return tools_dir


def get_platform_info() -> tuple[str, str]:
    """Return (os_name, arch) tuple."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    # Normalize OS name
    if system == "darwin":
        os_name = "apple-darwin"
    elif system == "linux":
        os_name = "unknown-linux-gnu"
    elif system == "windows":
        os_name = "pc-windows-msvc"
    else:
        os_name = system

    # Normalize architecture
    if machine in ("x86_64", "amd64"):
        arch = "x86_64"
    elif machine in ("arm64", "aarch64"):
        arch = "aarch64"
    else:
        arch = machine

    return os_name, arch


def get_download_url(version: str = DEFAULT_VERSION) -> Optional[str]:
    """Build the GitHub release download URL for this platform."""
    os_name, arch = get_platform_info()

    # macOS arm64 binaries may not exist; fall back to x86_64 (works via Rosetta)
    if os_name == "apple-darwin" and arch == "aarch64":
        arch = "x86_64"

    filename = f"biliup-{version}-{arch}-{os_name}"
    if os_name == "pc-windows-msvc":
        filename += ".zip"
    else:
        filename += ".zip"

    url = f"https://github.com/{GITHUB_REPO}/releases/download/v{version}/{filename}"
    return url


def find_biliup() -> Optional[str]:
    """Find biliup binary in PATH or project tools directory."""
    # Check system PATH
    biliup_path = shutil.which("biliup")
    if biliup_path:
        return biliup_path

    # Check project tools directory
    tools_dir = get_tools_dir()
    candidates = [
        tools_dir / "biliup",
        tools_dir / "biliup.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return None


def install_biliup(version: str = DEFAULT_VERSION, force: bool = False) -> str:
    """
    Download and install biliup-rs to the project tools directory.

    Args:
        version: biliup-rs version to install
        force: Re-install even if already present

    Returns:
        Absolute path to the installed biliup binary

    Raises:
        RuntimeError: If download or installation fails
    """
    existing = find_biliup()
    if existing and not force:
        logger.info(f"biliup already installed: {existing}")
        return existing

    tools_dir = get_tools_dir()
    download_url = get_download_url(version)
    if not download_url:
        raise RuntimeError(f"Unsupported platform: {platform.system()} {platform.machine()}")

    logger.info(f"Downloading biliup-rs v{version} from {download_url}")

    zip_path = tools_dir / f"biliup-v{version}.zip"

    # Download
    try:
        with httpx.stream("GET", download_url, follow_redirects=True, timeout=60) as response:
            response.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)
        logger.info(f"Downloaded: {zip_path}")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise RuntimeError(
                f"biliup-rs v{version} not found for this platform. "
                f"Please install manually from https://github.com/{GITHUB_REPO}/releases"
            )
        raise RuntimeError(f"Download failed: {e}")
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}")

    # Extract
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tools_dir)
        logger.info(f"Extracted to: {tools_dir}")
    except Exception as e:
        raise RuntimeError(f"Extraction failed: {e}")
    finally:
        # Clean up zip
        if zip_path.exists():
            zip_path.unlink()

    # Make executable (Unix)
    if platform.system() != "Windows":
        for binary in tools_dir.iterdir():
            if binary.name.startswith("biliup") and not binary.suffix:
                binary.chmod(binary.stat().st_mode | 0o111)
                logger.info(f"Made executable: {binary}")
                # macOS Gatekeeper: remove quarantine if possible
                try:
                    subprocess.run(
                        ["xattr", "-d", "com.apple.quarantine", str(binary)],
                        capture_output=True,
                        check=False,
                    )
                except FileNotFoundError:
                    pass
                return str(binary)

    # Windows
    for binary in tools_dir.iterdir():
        if binary.name == "biliup.exe":
            return str(binary)

    raise RuntimeError("biliup binary not found after extraction")


def ensure_biliup(version: str = DEFAULT_VERSION) -> str:
    """
    Ensure biliup is available. Install if missing.

    Returns:
        Path to biliup binary
    """
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
