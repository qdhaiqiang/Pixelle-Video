# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Install and detect local CosyVoice without requiring ComfyUI."""

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "tools" / "cosyvoice"
REPO_DIR = TOOLS_DIR / "CosyVoice"
VENV_DIR = TOOLS_DIR / ".venv"
RUNNER_PATH = PROJECT_ROOT / "tools" / "cosyvoice_runner.py"
RUNTIME_REQUIREMENTS = TOOLS_DIR / "requirements-runtime.txt"
INSTALL_MARKER = TOOLS_DIR / ".installed"
DEFAULT_REPO = "https://github.com/FunAudioLLM/CosyVoice.git"
SKIP_REQUIREMENT_PREFIXES = (
    "fastapi",
    "fastapi-cli",
    "gradio",
    "openai-whisper",
    "tensorboard",
    "uvicorn",
)


class CosyVoiceInstallError(RuntimeError):
    """Installation failed with captured command output."""


@dataclass
class CosyVoiceStatus:
    installed: bool
    repo_dir: str
    venv_dir: str
    runner_path: str
    message: str


def get_cosyvoice_python() -> Path:
    if (VENV_DIR / "bin" / "python").exists():
        return VENV_DIR / "bin" / "python"
    return VENV_DIR / "Scripts" / "python.exe"


def get_cosyvoice_status() -> CosyVoiceStatus:
    python_path = get_cosyvoice_python()
    installed = REPO_DIR.exists() and python_path.exists() and RUNNER_PATH.exists() and INSTALL_MARKER.exists()
    if installed:
        message = "CosyVoice local model is installed"
    elif not shutil.which("git"):
        message = "git is required to install CosyVoice"
    elif not shutil.which("uv"):
        message = "uv is required to create the isolated CosyVoice Python environment"
    else:
        message = "CosyVoice local model is not installed"
    return CosyVoiceStatus(
        installed=installed,
        repo_dir=str(REPO_DIR),
        venv_dir=str(VENV_DIR),
        runner_path=str(RUNNER_PATH),
        message=message,
    )


def install_cosyvoice(force: bool = False) -> CosyVoiceStatus:
    """Install CosyVoice into tools/cosyvoice with an isolated Python 3.10 env."""
    if force and TOOLS_DIR.exists():
        shutil.rmtree(TOOLS_DIR)

    TOOLS_DIR.mkdir(parents=True, exist_ok=True)

    if not REPO_DIR.exists():
        logger.info(f"Installing CosyVoice repo into {REPO_DIR}")
        _run_install_step(
            ["git", "clone", "--recursive", "--depth", "1", DEFAULT_REPO, str(REPO_DIR)],
            cwd=str(TOOLS_DIR),
        )
    else:
        logger.info(f"CosyVoice repo already exists: {REPO_DIR}")

    if not get_cosyvoice_python().exists():
        logger.info(f"Creating CosyVoice virtualenv: {VENV_DIR}")
        _run_install_step(
            ["uv", "venv", "--python", "3.10", str(VENV_DIR)],
            cwd=str(TOOLS_DIR),
        )

    python_path = get_cosyvoice_python()
    requirements = REPO_DIR / "requirements.txt"
    install_cmd = [
        "uv",
        "pip",
        "install",
        "--index-strategy",
        "unsafe-best-match",
        "--python",
        str(python_path),
    ]
    _run_install_step([*install_cmd, "setuptools<81", "wheel"], cwd=str(REPO_DIR))
    if requirements.exists():
        runtime_requirements = _write_runtime_requirements(requirements)
        _run_install_step(
            [*install_cmd, "-r", str(runtime_requirements)],
            cwd=str(REPO_DIR),
        )
    _run_install_step(
        [*install_cmd, "modelscope", "huggingface_hub", "soundfile"],
        cwd=str(REPO_DIR),
    )
    _run_install_step(
        [*install_cmd, "--no-build-isolation", "openai-whisper==20231117"],
        cwd=str(REPO_DIR),
    )
    INSTALL_MARKER.write_text("ok\n", encoding="utf-8")

    return get_cosyvoice_status()


def _run_install_step(cmd: list[str], cwd: str) -> None:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        output = "\n".join(
            part
            for part in [
                f"Command: {' '.join(cmd)}",
                f"Exit code: {result.returncode}",
                f"STDOUT:\n{result.stdout.strip()}" if result.stdout.strip() else "",
                f"STDERR:\n{result.stderr.strip()}" if result.stderr.strip() else "",
            ]
            if part
        )
        raise CosyVoiceInstallError(output)


def _write_runtime_requirements(source: Path) -> Path:
    lines: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if any(stripped.startswith(prefix) for prefix in SKIP_REQUIREMENT_PREFIXES):
            continue
        lines.append(line)
    RUNTIME_REQUIREMENTS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return RUNTIME_REQUIREMENTS
