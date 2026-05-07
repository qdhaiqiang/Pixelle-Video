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
Subtitle Extraction Service

Automatically detects, extracts, and parses subtitles from video files.
Supports external subtitle files and embedded subtitle streams.

Priority:
1. External subtitle files in same directory (srt/ass/vtt/txt)
2. Embedded subtitle streams (ffmpeg extraction)
3. Returns raw text with optional timestamps
"""

import re
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

from loguru import logger


@dataclass
class SubtitleDetectionResult:
    """Result of subtitle detection (without extraction)"""
    has_external: bool = False
    external_files: List[str] = None
    has_embedded: bool = False
    embedded_count: int = 0
    embedded_codecs: List[str] = None
    can_extract: bool = False  # True if we have extractable subtitle
    hard_subtitle_warning: bool = True  # Always warn about hard subtitles (can't detect)


@dataclass
class SubtitleLine:
    """A single subtitle line with optional timestamps"""
    start: Optional[float] = None
    end: Optional[float] = None
    text: str = ""


@dataclass
class StoryResult:
    """Result of story extraction"""
    text: str
    lines: List[SubtitleLine]
    has_timestamps: bool
    source: str  # 'external_srt', 'external_ass', 'external_vtt', 'external_txt', 'embedded', 'none'
    inferred_start: Optional[float] = None  # Auto-detected content start (skip intro)
    inferred_end: Optional[float] = None    # Auto-detected content end (skip outro)


class SubtitleExtractor:
    """
    Extract story/subtitle text from video with automatic fallback.
    """

    SUBTITLE_EXTENSIONS = [".srt", ".ass", ".ssa", ".vtt", ".txt"]

    def extract_story(self, video_path: str) -> StoryResult:
        """
        Main entry point: extract story text from video.
        Auto-detects intro/outro bounds based on subtitle timestamps.

        Priority:
        1. External subtitle files
        2. Embedded subtitle streams
        3. Return empty result

        Args:
            video_path: Path to video file

        Returns:
            StoryResult with text, lines, timestamps, and inferred content bounds
        """
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        result = None

        # Priority 1: External subtitle files
        result = self._find_external_subtitle(video_path_obj)
        if result:
            logger.info(f"✅ Found external subtitle: {result.source}")

        # Priority 2: Embedded subtitle streams
        if not result:
            result = self._extract_embedded_subtitles(video_path_obj)
            if result:
                logger.info(f"✅ Found embedded subtitle: {result.source}")

        if result:
            # Auto-detect intro/outro from subtitle timestamps
            result = self._infer_content_bounds(result, video_path_obj)
            return result

        # Priority 3: Nothing found - still infer bounds for long videos
        logger.warning(f"⚠️ No subtitles found for: {video_path}")
        result = StoryResult(
            text="",
            lines=[],
            has_timestamps=False,
            source="none"
        )
        result = self._infer_content_bounds(result, video_path_obj)
        return result

    def _infer_content_bounds(self, result: StoryResult, video_path: Path) -> StoryResult:
        """
        Auto-detect content_start/content_end by analyzing subtitle density.
        Skips intro/outro where subtitle density is abnormally low.

        For videos without subtitles but long duration (>10min), applies
        conservative default bounds to skip typical movie intro/outro.
        """

        # Get video duration first (needed for both subtitle and no-subtitle paths)
        try:
            probe = subprocess.run(
                ["ffprobe", "-hide_banner", "-v", "error",
                 "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
                capture_output=True, text=True, timeout=10
            )
            if probe.returncode != 0:
                return result
            video_duration = float(probe.stdout.strip())
        except Exception:
            return result

        # ============================================================
        # No subtitles: use conservative defaults for long videos
        # ============================================================
        if not result.lines:
            if video_duration > 600:
                result.inferred_start = 90.0
                result.inferred_end = video_duration - 120.0
                logger.info(f"🎬 No subtitles found, using default bounds for long video: "
                           f"start={result.inferred_start:.1f}s, end={result.inferred_end:.1f}s "
                           f"(duration={video_duration:.1f}s)")
            return result

        # ============================================================
        # With subtitles: density-based detection
        # ============================================================
        if not result.has_timestamps:
            return result

        timed_lines = [l for l in result.lines if l.start is not None]
        if not timed_lines:
            return result

        timed_lines.sort(key=lambda x: x.start)
        first_start = timed_lines[0].start
        last_end = timed_lines[-1].end or timed_lines[-1].start

        # -----------------------------------------------------------
        # Strategy 1: Dialogue cluster detection (gap-based)
        # -----------------------------------------------------------
        # Intro/outro subtitles are sparse; dialogue is dense.
        # Find first cluster where 3 consecutive subtitles have gaps < 3s.

        DIALOGUE_GAP_THRESHOLD = 3.0  # seconds
        MIN_CLUSTER_SIZE = 3

        dense_start = None
        for i in range(len(timed_lines) - MIN_CLUSTER_SIZE + 1):
            gaps = []
            valid = True
            for j in range(MIN_CLUSTER_SIZE - 1):
                gap = timed_lines[i + j + 1].start - timed_lines[i + j].end
                if gap > DIALOGUE_GAP_THRESHOLD:
                    valid = False
                    break
            if valid:
                dense_start = timed_lines[i].start
                break

        if dense_start and dense_start > 10.0:
            result.inferred_start = max(0.0, dense_start - 3.0)
            logger.info(f"🎬 Auto-detected intro skip (dialogue cluster): start at {result.inferred_start:.1f}s "
                       f"(first dense dialogue at {dense_start:.1f}s)")

        # Fallback: if first subtitle starts > 30s
        if result.inferred_start is None and first_start > 30.0:
            result.inferred_start = max(0.0, first_start - 5.0)
            logger.info(f"🎬 Auto-detected intro skip (late first subtitle): start at {result.inferred_start:.1f}s")

        # Detect outro: find last dense dialogue cluster
        dense_end = None
        for i in range(len(timed_lines) - 1, MIN_CLUSTER_SIZE - 2, -1):
            valid = True
            for j in range(MIN_CLUSTER_SIZE - 1):
                gap = timed_lines[i - j].start - timed_lines[i - j - 1].end
                if gap > DIALOGUE_GAP_THRESHOLD:
                    valid = False
                    break
            if valid:
                dense_end = timed_lines[i].end or timed_lines[i].start
                break

        if dense_end and dense_end < video_duration - 10.0:
            result.inferred_end = min(video_duration, dense_end + 5.0)
            logger.info(f"🎬 Auto-detected outro skip (dialogue cluster): end at {result.inferred_end:.1f}s "
                       f"(last dense dialogue at {dense_end:.1f}s)")

        # Fallback: if last subtitle ends > 30s before video end
        if result.inferred_end is None and video_duration - last_end > 30.0:
            result.inferred_end = min(video_duration, last_end + 5.0)
            logger.info(f"🎬 Auto-detected outro skip (early last subtitle): end at {result.inferred_end:.1f}s")

        return result

    def _find_external_subtitle(self, video_path: Path) -> Optional[StoryResult]:
        """Look for subtitle files in the same directory with same base name."""
        base_name = video_path.stem
        parent = video_path.parent

        for ext in self.SUBTITLE_EXTENSIONS:
            candidate = parent / f"{base_name}{ext}"
            if candidate.exists():
                content = candidate.read_text(encoding="utf-8", errors="ignore")
                if ext == ".srt":
                    return self._parse_srt(content)
                elif ext in (".ass", ".ssa"):
                    return self._parse_ass(content)
                elif ext == ".vtt":
                    return self._parse_vtt(content)
                elif ext == ".txt":
                    return StoryResult(
                        text=content.strip(),
                        lines=[SubtitleLine(text=line.strip()) for line in content.splitlines() if line.strip()],
                        has_timestamps=False,
                        source=f"external_txt"
                    )

        # Also try common patterns: .zh.srt, .cn.srt, etc.
        for candidate in parent.glob(f"{base_name}.*"):
            if candidate.suffix.lower() in self.SUBTITLE_EXTENSIONS:
                content = candidate.read_text(encoding="utf-8", errors="ignore")
                if candidate.suffix.lower() == ".srt":
                    return self._parse_srt(content)
                elif candidate.suffix.lower() in (".ass", ".ssa"):
                    return self._parse_ass(content)
                elif candidate.suffix.lower() == ".vtt":
                    return self._parse_vtt(content)

        return None

    def _extract_embedded_subtitles(self, video_path: Path) -> Optional[StoryResult]:
        """Extract embedded subtitle stream using ffmpeg."""
        try:
            # Probe for subtitle streams
            probe = subprocess.run(
                ["ffprobe", "-hide_banner", "-v", "error",
                 "-show_entries", "stream=index,codec_name:stream_disposition=default",
                 "-select_streams", "s",
                 "-of", "csv=p=0", str(video_path)],
                capture_output=True, text=True, timeout=30
            )

            if probe.returncode != 0 or not probe.stdout.strip():
                return None

            # Find the first/default subtitle stream
            streams = [line.strip() for line in probe.stdout.strip().splitlines() if line.strip()]
            if not streams:
                return None

            # Try stream index 0 first, then others
            for stream_idx, _ in enumerate(streams):
                try:
                    # Extract to srt format
                    output_srt = video_path.with_suffix(".extracted.srt")
                    result = subprocess.run(
                        ["ffmpeg", "-hide_banner", "-y", "-i", str(video_path),
                         "-map", f"0:s:{stream_idx}", str(output_srt)],
                        capture_output=True, text=True, timeout=60
                    )

                    if result.returncode == 0 and output_srt.exists():
                        content = output_srt.read_text(encoding="utf-8", errors="ignore")
                        output_srt.unlink()  # Clean up temp file
                        if content.strip():
                            return self._parse_srt(content)
                except Exception as e:
                    logger.debug(f"Failed to extract subtitle stream {stream_idx}: {e}")
                    continue

            return None

        except Exception as e:
            logger.warning(f"Failed to extract embedded subtitles: {e}")
            return None

    def _parse_srt(self, content: str) -> StoryResult:
        """Parse SRT format subtitle."""
        lines: List[SubtitleLine] = []

        # Split by double newline (SRT blocks)
        blocks = re.split(r"\n\s*\n", content.strip())

        for block in blocks:
            block_lines = [l.strip() for l in block.splitlines() if l.strip()]
            if len(block_lines) < 2:
                continue

            # Find the timing line (contains -->)
            timing_line = None
            timing_idx = -1
            for i, line in enumerate(block_lines):
                if "-->" in line:
                    timing_line = line
                    timing_idx = i
                    break

            if timing_line is None:
                continue

            # Parse timing
            time_match = re.match(
                r"(\d{1,2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{3})",
                timing_line
            )
            if not time_match:
                continue

            start = self._parse_time(time_match.group(1))
            end = self._parse_time(time_match.group(2))

            # Text is everything after timing line
            text = " ".join(block_lines[timing_idx + 1:])
            text = re.sub(r"<[^>]+>", "", text)  # Strip HTML tags

            if text:
                lines.append(SubtitleLine(start=start, end=end, text=text))

        # Build combined text with timestamps for LLM
        text_parts = []
        for line in lines:
            text_parts.append(f"[{line.start:.1f}s-{line.end:.1f}s] {line.text}")

        return StoryResult(
            text="\n".join(text_parts),
            lines=lines,
            has_timestamps=True,
            source="external_srt"
        )

    def _parse_ass(self, content: str) -> StoryResult:
        """Parse ASS/SSA format subtitle."""
        lines: List[SubtitleLine] = []

        # Find the [Events] section
        in_events = False
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if line.startswith("[Events]"):
                in_events = True
                continue
            if in_events and line.startswith("["):
                in_events = False
                continue
            if not in_events:
                continue
            if not line.startswith("Dialogue:"):
                continue

            # Format: Dialogue: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
            parts = line.split(",", 9)
            if len(parts) < 10:
                continue

            start = self._parse_time(parts[1])
            end = self._parse_time(parts[2])
            text = parts[9]

            # Strip ASS override tags {\...}
            text = re.sub(r"\{[^}]*\}", "", text)
            # Replace \N with space
            text = text.replace("\\N", " ").replace("\\n", " ")

            if text.strip():
                lines.append(SubtitleLine(start=start, end=end, text=text.strip()))

        text_parts = []
        for line in lines:
            text_parts.append(f"[{line.start:.1f}s-{line.end:.1f}s] {line.text}")

        return StoryResult(
            text="\n".join(text_parts),
            lines=lines,
            has_timestamps=True,
            source="external_ass"
        )

    def _parse_vtt(self, content: str) -> StoryResult:
        """Parse WebVTT format subtitle."""
        lines: List[SubtitleLine] = []

        # Skip WEBVTT header and metadata
        content_lines = content.splitlines()
        start_parsing = False

        for line in content_lines:
            stripped = line.strip()
            if stripped == "WEBVTT":
                start_parsing = True
                continue
            if not start_parsing:
                continue
            if not stripped or stripped.startswith("NOTE") or "-->" not in stripped:
                continue

            # Timing line: 00:00:01.000 --> 00:00:04.000
            time_match = re.match(
                r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})",
                stripped
            )
            if not time_match:
                continue

            start = self._parse_time(time_match.group(1))
            end = self._parse_time(time_match.group(2))

            # Next non-empty line is the text
            # We'll collect text in a simple way
            text = ""
            # This is a simplified parser - we assume single-line cues
            # A more robust parser would read ahead

            if text.strip():
                lines.append(SubtitleLine(start=start, end=end, text=text.strip()))

        # For VTT, do a simpler approach: extract all text after timing lines
        text_parts = []
        i = 0
        vtt_lines = content.splitlines()
        while i < len(vtt_lines):
            line = vtt_lines[i].strip()
            time_match = re.match(
                r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})",
                line
            )
            if time_match:
                start = self._parse_time(time_match.group(1))
                end = self._parse_time(time_match.group(2))
                i += 1
                cue_text = []
                while i < len(vtt_lines) and vtt_lines[i].strip() and "-->" not in vtt_lines[i]:
                    cue_text.append(vtt_lines[i].strip())
                    i += 1
                text = " ".join(cue_text)
                text = re.sub(r"<[^>]+>", "", text)
                if text:
                    lines.append(SubtitleLine(start=start, end=end, text=text))
                    text_parts.append(f"[{start:.1f}s-{end:.1f}s] {text}")
            else:
                i += 1

        return StoryResult(
            text="\n".join(text_parts),
            lines=lines,
            has_timestamps=True,
            source="external_vtt"
        )

    def detect_subtitles(self, video_path: str) -> SubtitleDetectionResult:
        """
        Detect subtitles presence without extracting content.

        Returns detection info for UI display.
        """
        video_path_obj = Path(video_path)
        result = SubtitleDetectionResult()

        if not video_path_obj.exists():
            return result

        # Check external subtitles
        external_files = []
        base_name = video_path_obj.stem
        parent = video_path_obj.parent

        for ext in self.SUBTITLE_EXTENSIONS:
            candidate = parent / f"{base_name}{ext}"
            if candidate.exists():
                external_files.append(str(candidate.name))

        # Also check .zh.srt etc patterns
        for candidate in parent.glob(f"{base_name}.*"):
            if candidate.suffix.lower() in self.SUBTITLE_EXTENSIONS and str(candidate.name) not in external_files:
                external_files.append(str(candidate.name))

        result.has_external = len(external_files) > 0
        result.external_files = external_files

        # Check embedded subtitle streams
        embedded_codecs = []
        try:
            probe = subprocess.run(
                ["ffprobe", "-hide_banner", "-v", "error",
                 "-show_entries", "stream=codec_name",
                 "-select_streams", "s",
                 "-of", "csv=p=0", str(video_path_obj)],
                capture_output=True, text=True, timeout=10
            )
            if probe.returncode == 0 and probe.stdout.strip():
                streams = [l.strip() for l in probe.stdout.strip().splitlines() if l.strip()]
                embedded_codecs = streams
                result.has_embedded = True
                result.embedded_count = len(streams)
                result.embedded_codecs = embedded_codecs
        except Exception:
            pass

        result.can_extract = result.has_external or result.has_embedded
        return result

    @staticmethod
    def _parse_time(time_str: str) -> float:
        """Parse time string like '00:01:23,456' or '0:01:23.45' to seconds."""
        time_str = time_str.replace(",", ".")
        parts = time_str.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        else:
            return float(parts[0])
