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
    def _has_audio_stream(path: Path) -> bool:
        """Return True if the file contains at least one audio stream."""
        try:
            out = subprocess.check_output(
                [
                    "ffprobe", "-hide_banner", "-v", "error",
                    "-select_streams", "a",
                    "-show_entries", "stream=codec_type",
                    "-of", "json", str(path),
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            data = json.loads(out)
            return bool(data.get("streams"))
        except Exception:
            return False

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
        # Strip backslashes (ASS control-char prefix) and braces (override tags)
        return text.replace("\\", "").replace("{", "").replace("}", "")

    @staticmethod
    def _split_semantic(text: str, max_chars: int) -> List[str]:
        """Split text into semantic segments by punctuation.
        Each segment fits within max_chars characters.
        Subtitles appear phrase-by-phrase, synced with narration."""
        text = re.sub(r"\s+", "", text)
        if len(text) <= max_chars:
            return [text]

        # Split by major punctuation
        raw_parts = re.split(r'([。！？；])', text)
        phrases: List[str] = []
        i = 0
        while i < len(raw_parts):
            if i + 1 < len(raw_parts) and raw_parts[i + 1] in '。！？；':
                phrases.append(raw_parts[i] + raw_parts[i + 1])
                i += 2
            else:
                if raw_parts[i]:
                    phrases.append(raw_parts[i])
                i += 1

        # Merge short phrases into segments that fit max_chars
        segments: List[str] = []
        current = ""
        for p in phrases:
            if len(current) + len(p) <= max_chars:
                current += p
            else:
                if current:
                    segments.append(current)
                current = p
        if current:
            segments.append(current)

        # Hard-split any segment still too long
        final: List[str] = []
        for seg in segments:
            while len(seg) > max_chars:
                final.append(seg[:max_chars])
                seg = seg[max_chars:]
            if seg:
                final.append(seg)
        return final

    @staticmethod
    def _wrap_caption(text: str, max_chars: int = 26) -> str:
        """Wrap text into balanced lines, each not exceeding max_chars.
        Returns all lines (no truncation) joined by \\N."""
        text = re.sub(r"\s+", "", text)
        total = len(text)
        if total <= max_chars:
            return text

        import math
        lines_needed = math.ceil(total / max_chars)
        base_len = total // lines_needed
        remainder = total % lines_needed

        lines: List[str] = []
        pos = 0
        for i in range(lines_needed):
            line_len = base_len + (1 if i < remainder else 0)
            lines.append(text[pos:pos + line_len])
            pos += line_len

        return r"\N".join(lines)

    # ==================== Step 1: Clip Extraction ====================

    def render_video_clips(self, video_path: Path, chunks: List[CommentaryChunk],
                           content_start: float, content_end: float, work_dir: Path,
                           mask_subtitles: bool = False,
                           mask_subtitle_height_ratio: float = 0.10) -> Path:
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
                    # Blur bottom area to mask original hard subtitles (configurable height)
                    ratio = max(0.05, min(0.40, mask_subtitle_height_ratio))
                    mask_h = max(int(video_height * ratio), 30)
                    mask_y = video_height - mask_h
                    base_vf = f"{base_vf},boxblur=0:0:0:0:0:0,drawbox=x=0:y={mask_y}:w={video_width}:h={mask_h}:color=black@0.85:t=fill"
                    logger.info(f"Masking subtitle area: ratio={ratio:.0%}, y={mask_y}, h={mask_h}")

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

        # Ensure every clip has an audio stream so concat demuxer doesn't
        # produce broken output when some clips are missing audio.
        for i, clip in enumerate(clip_files):
            if not self._has_audio_stream(clip):
                logger.warning(f"Clip {clip.name} has no audio; injecting silent track")
                silent = work_dir / f"clip_{i+1:03d}_silent.mp4"
                self._run([
                    "ffmpeg", "-hide_banner", "-y",
                    "-i", str(clip),
                    "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
                    "-shortest",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "96k",
                    str(silent),
                ])
                clip_files[i] = silent

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
        import asyncio
        import random
        import edge_tts as etts
        from aiohttp import ClientConnectorError, WSServerHandshakeError, ClientResponseError
        from edge_tts.exceptions import NoAudioReceived

        work_dir = output_path.parent
        work_dir.mkdir(parents=True, exist_ok=True)

        concat_file = work_dir / "voice_concat.txt"
        entries: List[str] = []
        cosy_tts = None
        if cfg.tts_inference_mode == "cosyvoice":
            from pixelle_video.config import config_manager
            from pixelle_video.services.tts_service import TTSService
            cosy_tts = TTSService(config_manager.config.to_dict())

        async def synthesize_with_retry(text: str, out_path: Path) -> None:
            if cosy_tts is not None:
                await cosy_tts(
                    text=text,
                    inference_mode="cosyvoice",
                    voice=cfg.tts_voice,
                    speed=cfg.tts_speed,
                    output_path=str(out_path),
                    allow_instruct=False,
                )
                return

            max_attempts = 6
            base_delay = 1.0
            for attempt in range(1, max_attempts + 1):
                try:
                    communicate = etts.Communicate(
                        text,
                        cfg.tts_voice,
                        rate=cfg.tts_rate,
                        volume="+0%",
                        connect_timeout=15,
                        receive_timeout=60,
                    )
                    await communicate.save(str(out_path))
                    return
                except (ClientConnectorError, WSServerHandshakeError, ClientResponseError, NoAudioReceived) as e:
                    if attempt >= max_attempts:
                        logger.error(f"TTS failed after {max_attempts} attempts: {e}")
                        raise
                    delay = min(base_delay * (2 ** (attempt - 1)), 30.0) + random.uniform(0, 1.0)
                    logger.warning(
                        f"TTS connection error (attempt {attempt}/{max_attempts}): {type(e).__name__}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                except Exception:
                    if attempt >= max_attempts:
                        raise
                    await asyncio.sleep(2.0)

        for chunk in chunks:
            raw = work_dir / f"{chunk.chunk_id}_raw.mp3"
            normalized = work_dir / f"{chunk.chunk_id}_raw.wav"
            wav = work_dir / f"{chunk.chunk_id}.wav"

            await asyncio.sleep(random.uniform(0.3, 0.8))
            await synthesize_with_retry(chunk.text, raw)

            # Normalize
            self._run(["ffmpeg", "-hide_banner", "-y", "-i", str(raw),
                       "-ac", "2", "-ar", "48000", str(normalized)])

            # Calculate atempo
            raw_dur = self._ffprobe_duration(normalized)
            slot = chunk.end - chunk.start
            target_dur = max(0.1, slot * max(0.55, min(1.0, cfg.narration_slot_ratio)))
            tempo = raw_dur / target_dur if target_dur > 0 else 1.0
            # Never slow down — stretched audio sounds dragged and unnatural
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

    # ==================== Step 2.5: Continuous TTS (no padding) ====================

    async def _synthesize_chunks_continuous(self, chunks: List[CommentaryChunk], cfg: CommentaryConfig,
                                             work_dir: Path) -> Tuple[List[Path], List[float]]:
        """Generate TTS for all chunks with atempo but WITHOUT apad/atrim.

        Returns (list of wav paths, list of actual durations after tempo).
        Each audio file is exactly as long as the TTS reads (stretched by atempo),
        with no silence padding. Chunks are meant to be concatenated back-to-back.
        """
        import asyncio
        import random
        import edge_tts as etts
        from aiohttp import ClientConnectorError, WSServerHandshakeError, ClientResponseError
        from edge_tts.exceptions import NoAudioReceived

        wav_paths: List[Path] = []
        actual_durations: List[float] = []
        cosy_tts = None
        if cfg.tts_inference_mode == "cosyvoice":
            from pixelle_video.config import config_manager
            from pixelle_video.services.tts_service import TTSService
            cosy_tts = TTSService(config_manager.config.to_dict())

        async def synthesize_with_retry(text: str, out_path: Path) -> None:
            """Call edge-tts with robust retry/backoff and longer timeout."""
            if cosy_tts is not None:
                await cosy_tts(
                    text=text,
                    inference_mode="cosyvoice",
                    voice=cfg.tts_voice,
                    speed=cfg.tts_speed,
                    output_path=str(out_path),
                    allow_instruct=False,
                )
                return

            max_attempts = 6
            base_delay = 1.0
            for attempt in range(1, max_attempts + 1):
                try:
                    communicate = etts.Communicate(
                        text,
                        cfg.tts_voice,
                        rate=cfg.tts_rate,
                        volume="+0%",
                        connect_timeout=15,
                        receive_timeout=60,
                    )
                    await communicate.save(str(out_path))
                    return
                except (ClientConnectorError, WSServerHandshakeError, ClientResponseError, NoAudioReceived) as e:
                    if attempt >= max_attempts:
                        logger.error(f"TTS failed after {max_attempts} attempts: {e}")
                        raise
                    # Exponential backoff + jitter
                    delay = min(base_delay * (2 ** (attempt - 1)), 30.0)
                    delay += random.uniform(0, 1.0)
                    logger.warning(
                        f"TTS connection error (attempt {attempt}/{max_attempts}): {type(e).__name__}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                except Exception:
                    # Unknown error — one more quick retry then give up
                    if attempt >= max_attempts:
                        raise
                    await asyncio.sleep(2.0)

        for chunk in chunks:
            raw = work_dir / f"{chunk.chunk_id}_raw.mp3"
            normalized = work_dir / f"{chunk.chunk_id}_raw.wav"
            wav = work_dir / f"{chunk.chunk_id}.wav"

            # Rate-limit: small random delay between chunks to avoid hammering the server
            await asyncio.sleep(random.uniform(0.3, 0.8))
            await synthesize_with_retry(chunk.text, raw)

            # Normalize
            self._run(["ffmpeg", "-hide_banner", "-y", "-i", str(raw),
                       "-ac", "2", "-ar", "48000", str(normalized)])

            # Calculate atempo
            raw_dur = self._ffprobe_duration(normalized)
            slot = chunk.end - chunk.start
            target_dur = max(0.1, slot * max(0.55, min(1.0, cfg.narration_slot_ratio)))
            tempo = raw_dur / target_dur if target_dur > 0 else 1.0
            # In continuous mode: only speed up (tempo >= 1.0), never slow down.
            # Slowdown would stretch audio and make narration sound dragged.
            if tempo < 1.0:
                tempo = 1.0

            # Apply atempo ONLY — no apad, no atrim
            filters = f"{self._atempo_chain(tempo)},asetpts=N/SR/TB,aresample=48000"
            self._run(["ffmpeg", "-hide_banner", "-y", "-i", str(normalized),
                       "-af", filters, "-ac", "2", str(wav)])

            actual_dur = self._ffprobe_duration(wav)
            wav_paths.append(wav)
            actual_durations.append(actual_dur)

            logger.info(
                f"🎙️ Chunk {chunk.chunk_id}: raw={raw_dur:.2f}s, slot={slot:.2f}s, "
                f"tempo={tempo:.2f}, actual={actual_dur:.2f}s"
            )

        return wav_paths, actual_durations

    @staticmethod
    def _concat_wav_files(wav_paths: List[Path], output_path: Path) -> Path:
        """Concatenate multiple WAV files into one using ffmpeg concat demuxer."""
        concat_file = output_path.parent / "voice_concat.txt"
        concat_file.write_text(
            "\n".join(f"file '{p}'" for p in wav_paths) + "\n",
            encoding="utf-8",
        )
        subprocess.run([
            "ffmpeg", "-hide_banner", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_file), "-c:a", "pcm_s16le", str(output_path),
        ], check=True)
        return output_path

    # ==================== Step 3: ASS Captions ====================

    def generate_ass_captions(self, chunks: List[CommentaryChunk], width: int, height: int,
                              slot_ratio: float, output_path: Path) -> Path:
        """Generate ASS subtitle file for commentary captions."""
        font_size = max(24, int(height * 0.040))
        margin_v = max(12, int(height * 0.030))

        # Fix max_chars for cinematic subtitle readability.
        # Dynamic calc based on video width yields 45+ chars on 1920px screens,
        # making subtitles stretch edge-to-edge — ugly. Cap at 18 for clean lines.
        max_chars = 18
        MAX_SUBTITLE_LINES = 2  # 字幕始终最多2行，超出则拆成多个时间片段

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
        max_segment_chars = max_chars * MAX_SUBTITLE_LINES  # e.g. 18 * 2 = 36
        for chunk in chunks:
            escaped = self._ass_escape(chunk.text)
            segments = self._split_semantic(escaped, max_segment_chars)

            target_dur = max(0.1, (chunk.end - chunk.start) * max(0.55, min(1.0, slot_ratio)))
            caption_end = min(chunk.end, chunk.start + target_dur)
            total_duration = caption_end - chunk.start
            total_chars = sum(len(s) for s in segments)

            current_time = chunk.start
            for seg in segments:
                text = self._wrap_caption(seg, max_chars=max_chars)
                seg_duration = total_duration * (len(seg) / total_chars) if total_chars > 0 else total_duration
                seg_end = min(caption_end, current_time + seg_duration)
                events.append(
                    f"Dialogue: 0,{self._ass_time(current_time)},{self._ass_time(seg_end)},Default,,0,0,0,,{text}"
                )
                current_time = seg_end

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
        # In filter_complex the path is parsed by FFmpeg, not the shell.
        # Wrap in single quotes so FFmpeg treats the whole path literally.
        caption_path = str(captions).replace("'", "'\\''")
        vf = f"ass='{caption_path}'"
        has_bgm = bgm_path and bgm_path.exists()
        has_orig_audio = self._has_audio_stream(assembled)
        if has_orig_audio:
            logger.debug(f"Detected audio stream in assembled video: {assembled}")
        else:
            logger.warning(f"No audio stream in assembled video, skipping original audio mix: {assembled}")

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
        elif keep_original_audio and has_orig_audio:
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
        work_dir = cover_bg_path.parent if cover_bg_path is not None else output_path.parent / "_work"
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

        # Build headline (split if too long, auto-scale font to fit)
        headline_lines = self._split_headline(cover.headline or cover.title)
        safe_width = int(width * 0.88)  # 6% margin on each side
        headline_font = 124 if len(headline_lines) == 1 else 102
        headline_font = self._fit_font_size(headline_lines, safe_width, headline_font, min_font=52)
        headline_gap = int(headline_font * 1.08)
        headline_block_h = headline_font + (len(headline_lines) - 1) * headline_gap
        headline_y = int((height - headline_block_h) * 0.39)
        rule_y = headline_y + headline_block_h + 42
        question_y = rule_y + 38
        rule_x = int(width * 0.18)
        rule_w = int(width * 0.64)

        # Auto-scale title if too long
        title_font = 42
        if cover.title and self._estimate_text_width(cover.title, title_font) > safe_width:
            title_font = self._fit_font_size([cover.title], safe_width, title_font, min_font=28)

        # Auto-scale question if too long
        question_font = 46
        if cover.question and self._estimate_text_width(cover.question, question_font) > safe_width:
            question_font = self._fit_font_size([cover.question], safe_width, question_font, min_font=32)

        logger.info(
            f"🎨 Cover layout: width={width}, headline_font={headline_font}, "
            f"title_font={title_font}, question_font={question_font}"
        )

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
                f"x=136:y=94:fontsize={title_font}:fontcolor=0xE5D5AA:"
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
                f"x=(w-text_w)/2:y={question_y}:fontsize={question_font}:fontcolor=0xFFFFFF:"
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
    def _estimate_text_width(text: str, font_size: int) -> float:
        """Estimate pixel width of text for Songti SC at given font size.
        Chinese chars ~0.95*font_size, ASCII ~0.55*font_size, punctuation ~0.5*font_size.
        """
        width = 0.0
        for ch in text:
            o = ord(ch)
            if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0x3000 <= o <= 0x303F:
                width += font_size * 0.95
            elif ch.isascii():
                width += font_size * 0.55
            else:
                width += font_size * 0.90
        return width

    @staticmethod
    def _fit_font_size(lines: List[str], target_width: int, base_font: int, min_font: int = 48) -> int:
        """Auto-shrink font size so the longest line fits within target_width."""
        longest = max(lines, key=len)
        font = base_font
        while font > min_font:
            estimated = CommentaryCompositor._estimate_text_width(longest, font)
            if estimated <= target_width:
                break
            font -= 4
        return font

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
    ) -> Tuple[Path, Optional[Path]]:
        """
        Full commentary composition pipeline.

        Returns (path to final video, path to cover background image).
        """
        paths = CompositorPaths.create(task_dir)
        content_start = cfg.content_start or 0.0
        content_end = cfg.content_end or self._ffprobe_duration(video_path)
        target_duration = float(cfg.target_duration)

        # Step 0: Generate raw TTS chunks and measure actual durations
        logger.info("🎙️ Generating voiceover chunks (continuous mode)...")
        chunk_wavs, chunk_durations = await self._synthesize_chunks_continuous(chunks, cfg, paths.work_dir)

        # Re-calculate timeline: each chunk's start/end based on actual audio length
        logger.info("📐 Re-calculating timeline for continuous narration...")
        current_time = 0.0
        for i, chunk in enumerate(chunks):
            chunk.start = current_time
            chunk.end = current_time + chunk_durations[i]
            current_time = chunk.end

        actual_total_duration = current_time
        logger.info(f"📐 Actual total narration duration: {actual_total_duration:.2f}s (target was {target_duration:.2f}s)")

        # Step 1: Extract clips with updated timeline
        logger.info("🎬 Extracting video clips...")
        mask_subtitles = getattr(cfg, 'mask_subtitles', False)
        mask_ratio = getattr(cfg, 'mask_subtitle_height_ratio', 0.10)
        if mask_subtitles:
            logger.info(f"🎭 Subtitle masking enabled: blurring bottom {mask_ratio:.0%} of video")
        assembled = self.render_video_clips(
            video_path, chunks, content_start, content_end, paths.work_dir,
            mask_subtitles=mask_subtitles,
            mask_subtitle_height_ratio=mask_ratio,
        )

        # Step 2: Concatenate audio files back-to-back (no silence gaps)
        logger.info("🎙️ Concatenating voiceover...")
        self._concat_wav_files(chunk_wavs, paths.voiceover_path)

        # Step 3: ASS captions with updated timeline (full chunk duration)
        logger.info("📝 Generating captions...")
        width, height = self._probe_size(assembled)
        ass_path = paths.work_dir / "commentary.ass"
        # Use slot_ratio=1.0 so captions span the entire chunk (no early cut-off)
        self.generate_ass_captions(chunks, width, height, 1.0, ass_path)

        # Step 4: Compose base video
        logger.info("🎞️ Composing base video...")
        bgm_path = Path(cfg.bgm_path) if cfg.bgm_path else None
        keep_original = getattr(cfg, 'keep_original_audio', True)
        orig_vol = getattr(cfg, 'original_audio_volume', 0.2)
        self.compose_final(
            assembled, paths.voiceover_path, ass_path, bgm_path, actual_total_duration, paths.base_path,
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
                url_or_path = getattr(result, 'url', None) or getattr(result, 'path', None)
                if url_or_path:
                    if url_or_path.startswith('http://') or url_or_path.startswith('https://'):
                        import httpx
                        async with httpx.AsyncClient() as client:
                            r = await client.get(url_or_path, timeout=60)
                            r.raise_for_status()
                            cover_bg_path.write_bytes(r.content)
                            logger.info(f"✅ AI cover downloaded from URL: {url_or_path}")
                    else:
                        # Local file path (e.g. from self-hosted ComfyUI)
                        import shutil
                        src = Path(url_or_path)
                        if not src.exists():
                            # Try resolving from project root if relative
                            project_root = Path(__file__).resolve().parents[2]
                            alt_src = project_root / url_or_path
                            if alt_src.exists():
                                src = alt_src
                        if src.exists():
                            shutil.copy2(src, cover_bg_path)
                            logger.info(f"✅ AI cover copied from local path: {src}")
                        else:
                            logger.warning(f"AI cover local path does not exist: {src}")
                            cover_bg_path = None
                else:
                    logger.warning("AI cover result has no url or path")
                    cover_bg_path = None
            except Exception as e:
                logger.warning(f"AI cover generation failed: {e}, using fallback")
                cover_bg_path = None  # Will trigger fallback in add_cover_intro
        else:
            if not cover.image_prompt:
                logger.warning("No image_prompt in cover config, skipping AI cover generation")
            cover_bg_path = None

        # Step 7: Add cover intro
        logger.info("🎨 Adding cover intro...")
        cover_intro_bg_path = cover_bg_path or (paths.work_dir / "cover_bg_fallback.jpg")
        final = self.add_cover_intro(
            input_video=paths.progress_path,
            source_video=video_path,
            cover=cover,
            cover_bg_path=cover_intro_bg_path,
            output_path=paths.final_path,
        )

        # Determine actual cover path used (add_cover_intro may fallback)
        actual_cover_path = cover_intro_bg_path
        if actual_cover_path is None or not actual_cover_path.is_file():
            actual_cover_path = paths.work_dir / "cover_bg_fallback.jpg"
        if not actual_cover_path.is_file():
            actual_cover_path = None

        logger.success(f"✅ Commentary video complete: {final}")
        return final, actual_cover_path
