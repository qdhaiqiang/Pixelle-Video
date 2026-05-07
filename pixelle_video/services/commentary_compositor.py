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
Commentary Video Compositor

Core video composition engine for commentary pipeline.
Handles: clip extraction, TTS + atempo, ASS captions, progress bar, cover intro, final mix.
"""

import asyncio
import json
import math
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

from loguru import logger

from pixelle_video.models.commentary import CommentaryChunk, CommentaryCover, CommentaryConfig


@dataclass
class CompositorPaths:
    """Paths for a single commentary compositor run"""
    task_dir: Path
    work_dir: Path
    outputs_dir: Path
    voiceover_path: Path
    assembled_path: Path
    base_path: Path
    progress_path: Path
    final_path: Path

    @classmethod
    def create(cls, task_dir: str) -> "CompositorPaths":
        td = Path(task_dir)
        work = td / "_work"
        outputs = td / "outputs"
        work.mkdir(parents=True, exist_ok=True)
        outputs.mkdir(parents=True, exist_ok=True)
        return cls(
            task_dir=td,
            work_dir=work,
            outputs_dir=outputs,
            voiceover_path=work / "voiceover.wav",
            assembled_path=work / "assembled.mp4",
            base_path=outputs / "commentary_base.mp4",
            progress_path=outputs / "commentary_progress.mp4",
            final_path=outputs / "commentary_final.mp4",
        )


class CommentaryCompositor:
    """
    Video compositor for commentary generation.

    All ffmpeg operations use subprocess for maximum control.
    """

    def __init__(self, core):
        self.core = core
        self.config = core.config if hasattr(core, "config") else {}

    # ==================== Utilities ====================

    @staticmethod
    def _run(cmd: List[str]) -> None:
        """Run a subprocess command, logging it."""
        logger.debug("+ " + " ".join(shlex.quote(c) for c in cmd))
        subprocess.run(cmd, check=True)

    @staticmethod
    def _ffprobe_json(path: Path) -> dict:
        out = subprocess.check_output([
            "ffprobe", "-hide_banner", "-v", "error",
            "-show_entries", "format=duration",
            "-show_entries", "stream=width,height",
            "-of", "json", str(path),
        ], text=True)
        return json.loads(out)

    @staticmethod
    def _ffprobe_duration(path: Path) -> float:
        data = CommentaryCompositor._ffprobe_json(path)
        return float(data["format"]["duration"])

    @staticmethod
    def _probe_size(path: Path) -> Tuple[int, int]:
        data = CommentaryCompositor._ffprobe_json(path)
        video = next(s for s in data["streams"] if "width" in s)
        return int(video["width"]), int(video["height"])

    @staticmethod
    def _atempo_chain(ratio: float) -> str:
        """Build atempo filter chain for speed adjustment."""
        ratio = max(0.25, min(4.0, ratio))
        parts: List[float] = []
        while ratio > 2.0:
            parts.append(2.0)
            ratio /= 2.0
        while ratio < 0.5:
            parts.append(0.5)
            ratio /= 0.5
        parts.append(ratio)
        return ",".join(f"atempo={part:.6f}" for part in parts)

    @staticmethod
    def _ass_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        cs = int(round((seconds - math.floor(seconds)) * 100))
        if cs == 100:
            s += 1
            cs = 0
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    @staticmethod
    def _ass_escape(text: str) -> str:
        return text.replace("{", "").replace("}", "")

    @staticmethod
    def _wrap_caption(text: str, max_chars: int = 26) -> str:
        text = re.sub(r"\s+", "", text)
        lines: List[str] = []
        while text:
            lines.append(text[:max_chars])
            text = text[max_chars:]
        return r"\N".join(lines[:2])

    # ==================== Step 1: Clip Extraction ====================

    def render_video_clips(self, video_path: Path, chunks: List[CommentaryChunk],
                           content_start: float, content_end: float, work_dir: Path,
                           mask_subtitles: bool = False) -> Path:
        """Extract clips from source video according to chunk source_windows."""
        clip_files: List[Path] = []
        index = 0

        # Pre-calc video size for mask filter
        video_width, video_height = self._probe_size(video_path) if mask_subtitles else (0, 0)

        for chunk in chunks:
            windows = self._bounded_windows(chunk, content_start, content_end)
            per = (chunk.end - chunk.start) / len(windows) if windows else (chunk.end - chunk.start)
            for src_start, _src_end in windows:
                index += 1
                out = work_dir / f"clip_{index:03d}.mp4"

                # Build video filter
                base_vf = "setsar=1,fps=30000/1001,format=yuv420p"
                if mask_subtitles and video_height > 0:
                    # Blur bottom 10% area to mask original hard subtitles
                    mask_h = max(int(video_height * 0.10), 30)
                    mask_y = video_height - mask_h
                    base_vf = f"{base_vf},boxblur=0:0:0:0:0:0,drawbox=x=0:y={mask_y}:w={video_width}:h={mask_h}:color=black@0.85:t=fill"
                    logger.debug(f"Masking subtitle area: y={mask_y}, h={mask_h}")

                self._run([
                    "ffmpeg", "-hide_banner", "-y",
                    "-ss", f"{src_start:.3f}", "-i", str(video_path),
                    "-t", f"{per:.3f}",
                    "-map", "0:v:0", "-map", "0:a:0?",
                    "-vf", base_vf,
                    "-af", "aresample=48000",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                    "-c:a", "aac", "-b:a", "96k", str(out),
                ])
                clip_files.append(out)

        # Concatenate clips
        concat_file = work_dir / "video_concat.txt"
        concat_file.write_text("\n".join(f"file '{c}'" for c in clip_files) + "\n", encoding="utf-8")
        assembled = work_dir / "assembled.mp4"
        self._run([
            "ffmpeg", "-hide_banner", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_file), "-c", "copy", str(assembled),
        ])
        return assembled

    def _bounded_windows(self, chunk: CommentaryChunk, content_start: float, content_end: float) -> List[Tuple[float, float]]:
        """Clamp source windows to content bounds, with fallback."""
        windows = [(w.start, w.end) for w in chunk.source_windows
                   if content_start <= w.start < w.end <= content_end]
        if windows:
            return windows
        midpoint = min(max((content_start + content_end) / 2, content_start), content_end - 6)
        return [(midpoint, midpoint + 6)]

    # ==================== Step 2: TTS + Voiceover ====================

    async def synthesize_voiceover(self, chunks: List[CommentaryChunk], cfg: CommentaryConfig,
                                   output_path: Path) -> Path:
        """Generate TTS voiceover with atempo adjustment to match chunk durations."""
        work_dir = output_path.parent
        work_dir.mkdir(parents=True, exist_ok=True)

        # Import edge_tts dynamically
        import edge_tts as etts

        concat_file = work_dir / "voice_concat.txt"
        entries: List[str] = []

        async def synthesize(text: str, out_path: Path) -> None:
            communicate = etts.Communicate(text, cfg.tts_voice, rate=cfg.tts_rate, volume="+0%")
            await communicate.save(str(out_path))

        for chunk in chunks:
            raw = work_dir / f"{chunk.chunk_id}_raw.mp3"
            normalized = work_dir / f"{chunk.chunk_id}_raw.wav"
            wav = work_dir / f"{chunk.chunk_id}.wav"

            # Retry TTS up to 4 times
            for attempt in range(1, 5):
                try:
                    logger.debug(f"TTS {chunk.chunk_id} attempt={attempt}")
                    await synthesize(chunk.text, raw)
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2 * attempt)

            # Normalize
            self._run(["ffmpeg", "-hide_banner", "-y", "-i", str(raw),
                       "-ac", "2", "-ar", "48000", str(normalized)])

            # Calculate atempo
            raw_dur = self._ffprobe_duration(normalized)
            slot = chunk.end - chunk.start
            target_dur = max(0.1, slot * max(0.55, min(1.0, cfg.narration_slot_ratio)))
            tempo = raw_dur / target_dur if target_dur > 0 else 1.0
            # Never slow down audio (tempo < 1); only speed up if too long
            if tempo < 1.0:
                tempo = 1.0

            filters = f"{self._atempo_chain(tempo)},apad,atrim=0:{slot},asetpts=N/SR/TB,aresample=48000"
            self._run(["ffmpeg", "-hide_banner", "-y", "-i", str(normalized),
                       "-af", filters, "-ac", "2", str(wav)])
            entries.append(f"file '{wav}'")

        concat_file.write_text("\n".join(entries) + "\n", encoding="utf-8")
        self._run([
            "ffmpeg", "-hide_banner", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_file), "-c:a", "pcm_s16le", str(output_path),
        ])
        return output_path

    # ==================== Step 3: ASS Captions ====================

    def generate_ass_captions(self, chunks: List[CommentaryChunk], width: int, height: int,
                              slot_ratio: float, output_path: Path) -> Path:
        """Generate ASS subtitle file for commentary captions."""
        font_size = max(24, int(height * 0.057))
        margin_v = max(12, int(height * 0.034))

        header = f"""[Script Info]
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: {width}
PlayResY: {height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Songti SC,{font_size},&H00FFFFFF,&H000000FF,&H00101010,&H99000000,1,0,0,0,100,100,0,0,1,2.5,0.6,2,38,38,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        events = []
        for chunk in chunks:
            text = self._ass_escape(self._wrap_caption(chunk.text))
            target_dur = max(0.1, (chunk.end - chunk.start) * max(0.55, min(1.0, slot_ratio)))
            caption_end = min(chunk.end, chunk.start + target_dur)
            events.append(
                f"Dialogue: 0,{self._ass_time(chunk.start)},{self._ass_time(caption_end)},Default,,0,0,0,,{text}"
            )

        output_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
        return output_path

    # ==================== Step 4: Final Composition ====================

    def compose_final(self, assembled: Path, voiceover: Path, captions: Path,
                      bgm_path: Optional[Path], target_duration: float, output_path: Path,
                      keep_original_audio: bool = True, original_audio_volume: float = 0.2) -> Path:
        """Mix assembled video + voiceover + original audio/BGM into final video.

        Audio priority: BGM > Original Audio > None (voiceover only).
        BGM and original audio are mutually exclusive.
        """
        vf = f"ass={shlex.quote(str(captions))}"
        has_bgm = bgm_path and bgm_path.exists()

        # BGM takes priority over original audio
        if has_bgm:
            logger.info(f"🎵 Using BGM ({bgm_path.name}), original audio disabled")
            cmd = [
                "ffmpeg", "-hide_banner", "-y",
                "-i", str(assembled),
                "-i", str(voiceover),
                "-stream_loop", "-1", "-i", str(bgm_path),
                "-filter_complex",
                f"[0:v]{vf}[v];"
                "[1:a]volume=1.70[a1];"
                f"[2:a]volume=0.055,atrim=0:{target_duration:.3f},asetpts=N/SR/TB[a2];"
                "[a1][a2]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[a]",
                "-map", "[v]", "-map", "[a]",
                "-t", f"{target_duration:.3f}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                str(output_path),
            ]
        elif keep_original_audio:
            logger.info(f"🔊 Keeping original audio (volume={original_audio_volume:.0%})")
            orig_vol = f"{original_audio_volume:.2f}"
            cmd = [
                "ffmpeg", "-hide_banner", "-y",
                "-i", str(assembled),
                "-i", str(voiceover),
                "-filter_complex",
                f"[0:v]{vf}[v];"
                f"[0:a]volume={orig_vol}[a0];"
                "[1:a]volume=1.70[a1];"
                "[a0][a1]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[a]",
                "-map", "[v]", "-map", "[a]",
                "-t", f"{target_duration:.3f}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                str(output_path),
            ]
        else:
            logger.info("🔇 Original audio removed, voiceover only")
            cmd = [
                "ffmpeg", "-hide_banner", "-y",
                "-i", str(assembled),
                "-i", str(voiceover),
                "-filter_complex",
                f"[0:v]{vf}[v];"
                "[1:a]volume=1.70[a1];"
                "[a1]alimiter=limit=0.95[a]",
                "-map", "[v]", "-map", "[a]",
                "-t", f"{target_duration:.3f}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                str(output_path),
            ]

        self._run(cmd)
        return output_path

    # ==================== Step 5: Progress Bar ====================

    def add_progress_bar(self, input_video: Path, segments: List[str], output_path: Path) -> Path:
        """Add top progress bar overlay to video."""
        if not segments:
            segments = ["铺垫", "冲突", "转折", "爆发", "余波"]

        duration = self._ffprobe_duration(input_video)
        width, height = self._probe_size(input_video)

        seg_count = len(segments)
        seg_len = duration / seg_count
        segment_data = [(i * seg_len, (i + 1) * seg_len, seg) for i, seg in enumerate(segments)]

        # Build static overlay filters
        work_dir = output_path.parent / "_work"
        work_dir.mkdir(parents=True, exist_ok=True)
        segment_w = width / seg_count
        filters: List[str] = []

        for idx in range(1, seg_count):
            x = int(idx * segment_w)
            filters.append(f"drawbox=x={x - 1}:y=6:w=3:h=27:color=0x171717@1:t=fill")

        for idx, (_s, _e, title) in enumerate(segment_data):
            title_file = work_dir / f"progress_title_{idx + 1}.txt"
            title_file.write_text(title, encoding="utf-8")
            center = int((idx + 0.5) * segment_w)
            filters.append(
                f"drawtext=fontfile=/System/Library/Fonts/Supplemental/Songti.ttc:"
                f"textfile={shlex.quote(str(title_file))}:"
                f"x={center}-(text_w/2):y=10:fontsize=17:fontcolor=0x222222"
            )

        static_filter = ",".join(filters)
        slide = f"-overlay_w+overlay_w*t/{duration:.3f}"

        filter_complex = (
            "[0:v][1:v]overlay=x=0:y=0[vp0];"
            f"[vp0][2:v]overlay=x='{slide}':y=0:eval=frame[vp1];"
            f"[vp1][3:v]overlay=x='{slide}':y=37:eval=frame[vp2];"
            f"[vp2]{static_filter}[v]"
        )

        self._run([
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
            "-i", str(input_video),
            "-f", "lavfi",
            "-i", f"color=c=0xF3F1EC@0.54:s={width}x40:r=30000/1001:d={duration:.3f},format=rgba",
            "-f", "lavfi",
            "-i", f"color=c=0xDDD7CD@0.86:s={width}x40:r=30000/1001:d={duration:.3f},format=rgba",
            "-f", "lavfi",
            "-i", f"color=c=0x111111@1:s={width}x3:r=30000/1001:d={duration:.3f},format=rgba",
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "0:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "copy", "-movflags", "+faststart", str(output_path),
        ])
        return output_path

    # ==================== Step 6: Cover Intro ====================

    def add_cover_intro(self, input_video: Path, source_video: Path, cover: CommentaryCover,
                        cover_bg_path: Optional[Path], output_path: Path) -> Path:
        """Prepend 3-second AI-style cover intro to video.

        Args:
            input_video: The processed video to prepend cover to
            source_video: The original source video (for fallback frame extraction)
            cover: Cover configuration
            cover_bg_path: Path to AI-generated cover background image (or None)
            output_path: Final output path
        """
        work_dir = output_path.parent / "_work"
        work_dir.mkdir(parents=True, exist_ok=True)
        cover_clip = work_dir / "cover_intro.mp4"

        # Fallback: if AI cover not generated or invalid, extract frame from source video
        if cover_bg_path is None or not cover_bg_path.is_file():
            cover_bg_path = work_dir / "cover_bg_fallback.jpg"
            self._run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{cover.background_time:.3f}", "-i", str(source_video),
                "-frames:v", "1", "-q:v", "2", str(cover_bg_path),
            ])

        width, height = self._probe_size(input_video)
        font = "/System/Library/Fonts/Supplemental/Songti.ttc"

        # Build headline (split if too long)
        headline_lines = self._split_headline(cover.headline or cover.title)
        headline_font = 124 if len(headline_lines) == 1 else 102
        headline_gap = int(headline_font * 1.08)
        headline_block_h = headline_font + (len(headline_lines) - 1) * headline_gap
        headline_y = int((height - headline_block_h) * 0.39)
        rule_y = headline_y + headline_block_h + 42
        question_y = rule_y + 38
        rule_x = int(width * 0.18)
        rule_w = int(width * 0.64)

        filters = [
            f"scale={width}:{height}:force_original_aspect_ratio=increase",
            f"crop={width}:{height}",
            "format=yuv420p",
            "eq=contrast=1.18:brightness=-0.12:saturation=0.72",
            "drawbox=x=0:y=0:w=iw:h=ih:color=0x000000@0.58:t=fill",
            "drawbox=x=0:y=0:w=iw:h=ih:color=0x161000@0.14:t=fill",
            "drawbox=x=96:y=80:w=iw-192:h=ih-160:color=0x000000@0.14:t=fill",
        ]

        if cover.title:
            filters.append(
                f"drawtext=fontfile={font}:text='{self._drawtext_escape(cover.title)}':"
                "x=136:y=94:fontsize=42:fontcolor=0xE5D5AA:"
                "shadowcolor=0x000000@0.80:shadowx=2:shadowy=2"
            )

        for idx, line in enumerate(headline_lines[:2]):
            filters.append(
                f"drawtext=fontfile={font}:text='{self._drawtext_escape(line)}':"
                f"x=(w-text_w)/2:y={headline_y + idx * headline_gap}:"
                f"fontsize={headline_font}:fontcolor=0xFFF8DF:"
                "shadowcolor=0x000000@0.95:shadowx=5:shadowy=5"
            )

        filters.append(f"drawbox=x={rule_x}:y={rule_y}:w={rule_w}:h=6:color=0xE5B84D@0.98:t=fill")

        if cover.question:
            filters.append(
                f"drawtext=fontfile={font}:text='{self._drawtext_escape(cover.question)}':"
                f"x=(w-text_w)/2:y={question_y}:fontsize=46:fontcolor=0xFFFFFF:"
                "shadowcolor=0x000000@0.85:shadowx=3:shadowy=3"
            )

        filters.append(
            f"drawtext=fontfile={font}:text='{self._drawtext_escape('本视频完全通过AI处理，一键生成')}':"
            "x=136:y=h-156:fontsize=30:fontcolor=0xEFE5D0:"
            "shadowcolor=0x000000@0.75:shadowx=2:shadowy=2"
        )

        # Generate cover clip (3 seconds)
        self._run([
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
            "-loop", "1", "-i", str(cover_bg_path),
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-t", "3.0",
            "-vf", ",".join(filters),
            "-r", "30000/1001",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k", "-shortest", str(cover_clip),
        ])

        # Concatenate cover + input video
        self._run([
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
            "-i", str(cover_clip), "-i", str(input_video),
            "-filter_complex",
            "[0:v]setsar=1[v0];[1:v]setsar=1[v1];[0:a]aresample=48000[a0];[1:a]aresample=48000[a1];"
            "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]",
            "-map", "[v]", "-map", "[a]",
            "-r", "30000/1001", "-s", f"{width}x{height}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output_path),
        ])
        return output_path

    @staticmethod
    def _drawtext_escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", r"\'")

    @staticmethod
    def _split_headline(text: str) -> List[str]:
        explicit = [l.strip() for l in text.split("|") if l.strip()]
        if explicit:
            return explicit[:2]
        clean = text.strip()
        if len(clean) <= 8:
            return [clean]
        midpoint = len(clean) // 2
        return [clean[:midpoint], clean[midpoint:]]

    # ==================== Main Orchestrator ====================

    async def compose_commentary(
        self,
        video_path: Path,
        chunks: List[CommentaryChunk],
        cover: CommentaryCover,
        progress_segments: List[str],
        cfg: CommentaryConfig,
        task_dir: str,
    ) -> Path:
        """
        Full commentary composition pipeline.

        Returns path to final video.
        """
        paths = CompositorPaths.create(task_dir)
        content_start = cfg.content_start or 0.0
        content_end = cfg.content_end or self._ffprobe_duration(video_path)
        target_duration = float(cfg.target_duration)

        # Step 1: Extract clips
        logger.info("🎬 Extracting video clips...")
        mask_subtitles = getattr(cfg, 'mask_subtitles', False)
        if mask_subtitles:
            logger.info("🎭 Subtitle masking enabled: blurring bottom subtitle area")
        assembled = self.render_video_clips(video_path, chunks, content_start, content_end, paths.work_dir, mask_subtitles)

        # Step 2: TTS voiceover
        logger.info("🎙️ Generating voiceover...")
        voiceover = await self.synthesize_voiceover(chunks, cfg, paths.voiceover_path)

        # Step 3: ASS captions
        logger.info("📝 Generating captions...")
        width, height = self._probe_size(assembled)
        ass_path = paths.work_dir / "commentary.ass"
        self.generate_ass_captions(chunks, width, height, cfg.narration_slot_ratio, ass_path)

        # Step 4: Compose base video
        logger.info("🎞️ Composing base video...")
        bgm_path = Path(cfg.bgm_path) if cfg.bgm_path else None
        keep_original = getattr(cfg, 'keep_original_audio', True)
        orig_vol = getattr(cfg, 'original_audio_volume', 0.2)
        self.compose_final(
            assembled, voiceover, ass_path, bgm_path, target_duration, paths.base_path,
            keep_original_audio=keep_original,
            original_audio_volume=orig_vol,
        )

        # Step 5: Add progress bar
        logger.info("📊 Adding progress bar...")
        self.add_progress_bar(paths.base_path, progress_segments, paths.progress_path)

        # Step 6: Generate AI cover background
        logger.info("🖼️ Generating AI cover background...")
        cover_bg_path: Optional[Path] = paths.work_dir / "cover_bg_ai.jpg"
        if cover.image_prompt and self.core.media:
            try:
                result = await self.core.media(
                    prompt=cover.image_prompt,
                    media_type="image",
                    width=width,
                    height=height,
                )
                # Download/copy result to cover_bg_path
                if hasattr(result, 'url') and result.url:
                    import httpx
                    async with httpx.AsyncClient() as client:
                        r = await client.get(result.url)
                        r.raise_for_status()
                        cover_bg_path.write_bytes(r.content)
                elif hasattr(result, 'path') and result.path:
                    import shutil
                    shutil.copy2(result.path, cover_bg_path)
            except Exception as e:
                logger.warning(f"AI cover generation failed: {e}, using fallback")
                cover_bg_path = None  # Will trigger fallback in add_cover_intro
        else:
            cover_bg_path = None

        # Step 7: Add cover intro
        logger.info("🎨 Adding cover intro...")
        final = self.add_cover_intro(
            input_video=paths.progress_path,
            source_video=video_path,
            cover=cover,
            cover_bg_path=cover_bg_path,
            output_path=paths.final_path,
        )

        logger.success(f"✅ Commentary video complete: {final}")
        return final
