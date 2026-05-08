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
Jianying (CapCut) Material Exporter

Exports commentary pipeline assets as a structured material package
for easy import into Jianying / CapCut.

Large files (video/audio) use symlinks to save disk space.
Small files (text, subtitles) are copied and converted.
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger


class JianyingMaterialExporter:
    """Export pipeline assets to a Jianying-friendly material package."""

    def __init__(self, task_dir: str):
        self.task_dir = Path(task_dir)
        self.work_dir = self.task_dir / "_work"
        self.outputs_dir = self.task_dir / "outputs"
        self.materials_dir = self.task_dir / "materials"

    def export(
        self,
        chunks: Optional[List] = None,
        final_video_path: Optional[str] = None,
    ) -> Path:
        """
        Export all available assets.

        Args:
            chunks: List of CommentaryChunk objects (for narration text)
            final_video_path: Override final video path

        Returns:
            Path to the materials directory
        """
        if self.materials_dir.exists():
            # Clean old materials
            for item in self.materials_dir.iterdir():
                if item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    for sub in item.iterdir():
                        if sub.is_symlink():
                            sub.unlink()
                        else:
                            sub.unlink()
                    item.rmdir()
                else:
                    item.unlink()
        else:
            self.materials_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"📦 Exporting Jianying materials to: {self.materials_dir}")

        # 1. Video clips (symlink)
        self._export_clips()

        # 2. Voiceover audio (symlink)
        self._export_voiceover()

        # 3. Subtitles (ASS -> SRT, copy)
        self._export_subtitles()

        # 4. Cover images (symlink)
        self._export_cover()

        # 5. Narration scripts (copy)
        self._export_scripts(chunks)

        # 6. Final video (symlink)
        self._export_final(final_video_path)

        logger.success(f"✅ Jianying materials exported: {self.materials_dir}")
        return self.materials_dir

    def _mkdir(self, name: str) -> Path:
        d = self.materials_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _symlink(self, src: Path, dst_dir: Path, dst_name: Optional[str] = None) -> bool:
        """Create symlink from src to dst_dir. Falls back to copy on Windows if needed."""
        if not src.exists():
            return False
        name = dst_name or src.name
        dst = dst_dir / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            os.symlink(src.resolve(), dst)
            return True
        except OSError:
            # Windows may need admin for symlinks; fallback to hardlink then copy
            try:
                os.link(src.resolve(), dst)
                return True
            except OSError:
                import shutil
                shutil.copy2(src, dst)
                return True

    def _write_text(self, dst: Path, content: str) -> None:
        dst.write_text(content, encoding="utf-8")

    def _export_clips(self) -> None:
        """Symlink all clip_*.mp4 files."""
        d = self._mkdir("01_video_clips")
        count = 0
        if self.work_dir.exists():
            for clip in sorted(self.work_dir.glob("clip_*.mp4")):
                if self._symlink(clip, d):
                    count += 1
        logger.info(f"   01_video_clips: {count} clips")

    def _export_voiceover(self) -> None:
        """Symlink voiceover audio files."""
        d = self._mkdir("02_voiceover")
        count = 0
        if self.work_dir.exists():
            # Full voiceover
            vo = self.work_dir / "voiceover.wav"
            if self._symlink(vo, d):
                count += 1
            # Per-chunk voiceover
            for wav in sorted(self.work_dir.glob("chunk_*.wav")):
                if self._symlink(wav, d):
                    count += 1
        logger.info(f"   02_voiceover: {count} audio files")

    def _export_subtitles(self) -> None:
        """Convert ASS to SRT and copy."""
        d = self._mkdir("03_subtitles")
        ass_path = self.work_dir / "commentary.ass"
        if ass_path.exists():
            srt_content = self._ass_to_srt(ass_path.read_text(encoding="utf-8"))
            srt_path = d / "commentary.srt"
            self._write_text(srt_path, srt_content)
            logger.info(f"   03_subtitles: commentary.srt")
        else:
            logger.info(f"   03_subtitles: no ASS file found")

    def _export_cover(self) -> None:
        """Symlink cover background images."""
        d = self._mkdir("04_cover")
        count = 0
        if self.work_dir.exists():
            for img in sorted(self.work_dir.glob("cover_bg_*.jpg")):
                if self._symlink(img, d):
                    count += 1
        logger.info(f"   04_cover: {count} images")

    def _export_scripts(self, chunks: Optional[List]) -> None:
        """Write narration text file."""
        d = self._mkdir("05_scripts")
        lines: List[str] = []
        if chunks:
            for i, chunk in enumerate(chunks, 1):
                lines.append(f"【段落 {i}】{chunk.start:.1f}s - {chunk.end:.1f}s")
                lines.append(chunk.text)
                lines.append("")
        else:
            lines.append("（无分镜文案）")
        self._write_text(d / "narration.txt", "\n".join(lines))
        logger.info(f"   05_scripts: narration.txt")

    def _export_final(self, final_video_path: Optional[str] = None) -> None:
        """Symlink final video."""
        d = self._mkdir("06_final")
        # Try provided path first
        if final_video_path:
            fp = Path(final_video_path)
            if self._symlink(fp, d, "final_video.mp4"):
                logger.info(f"   06_final: final_video.mp4")
                return
        # Try outputs dir
        candidates = [
            self.outputs_dir / "commentary_final.mp4",
            self.outputs_dir / "commentary_base.mp4",
        ]
        for cand in candidates:
            if cand.exists():
                if self._symlink(cand, d, "final_video.mp4"):
                    logger.info(f"   06_final: final_video.mp4")
                    return
        logger.info(f"   06_final: no final video found")

    @staticmethod
    def _ass_to_srt(ass_text: str) -> str:
        """Convert ASS subtitle text to SRT format."""
        dialogues: List[Tuple[int, str, str, str]] = []
        idx = 1
        for line in ass_text.splitlines():
            line = line.strip()
            if not line.startswith("Dialogue:"):
                continue
            # Format: Dialogue: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
            parts = line.split(",", 9)
            if len(parts) < 10:
                continue
            start = parts[1].strip()
            end = parts[2].strip()
            text = parts[9].strip()
            # Convert ASS \N to newline
            text = text.replace(r"\N", "\n")
            # Remove ASS override tags {\...}
            text = re.sub(r"\{[^}]*\}", "", text)
            if text:
                dialogues.append((idx, start, end, text))
                idx += 1

        srt_lines: List[str] = []
        for num, start, end, text in dialogues:
            srt_start = JianyingMaterialExporter._ass_time_to_srt(start)
            srt_end = JianyingMaterialExporter._ass_time_to_srt(end)
            srt_lines.append(str(num))
            srt_lines.append(f"{srt_start} --> {srt_end}")
            srt_lines.append(text)
            srt_lines.append("")

        return "\n".join(srt_lines)

    @staticmethod
    def _ass_time_to_srt(ass_time: str) -> str:
        """Convert ASS time (H:MM:SS.cc) to SRT time (HH:MM:SS,mmm)."""
        # ASS: 0:00:05.00 or 0:05:00.00
        match = re.match(r"(\d+):(\d{2}):(\d{2})\.(\d{2})", ass_time)
        if not match:
            return "00:00:00,000"
        h, m, s, cs = match.groups()
        # Convert centiseconds to milliseconds
        ms = int(cs) * 10
        return f"{int(h):02d}:{m}:{s},{ms:03d}"
