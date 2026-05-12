# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Screen recording subtitle and dubbing processor."""

import json
import re
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from pixelle_video.models.progress import ProgressEvent
from pixelle_video.utils.os_util import create_task_output_dir
from pixelle_video.services.video import VideoService


@dataclass
class SubtitleSegment:
    start: float
    end: float
    text: str


@dataclass
class ScreenRecordingResult:
    task_id: str
    video_path: str
    materials_dir: str
    materials_zip: str
    srt_path: str
    ass_path: str
    segments_json_path: str
    transcript_json_path: str
    dubbing_audio_path: Optional[str] = None


class ScreenRecordingProcessor:
    """Process recorded videos into burned-subtitle videos plus editable materials."""

    TERM_FIXES = {
        "公单": "工单",
        "以关闭": "已关闭",
        "已关必": "已关闭",
        "mark down": "Markdown",
        "markdown": "Markdown",
        "MAC当": "Markdown",
        "马可当": "Markdown",
    }

    FILLER_PATTERNS = [
        r"^\s*(嗯|呃|啊|额|就是|然后|那个|这个|就是说)[，,、\s]*",
        r"[，,、\s]*(嗯|呃|啊|额)[，,、\s]*",
        r"(然后){2,}",
        r"(就是){2,}",
    ]

    def __init__(self, core):
        self.core = core

    async def process(
        self,
        video_path: str,
        glossary_path: Optional[str] = None,
        correction_path: Optional[str] = None,
        industry_context: str = "",
        whisper_model: str = "small",
        language: str = "zh",
        ai_polish: bool = True,
        synthesize_dubbing: bool = False,
        tts_inference_mode: str = "local",
        tts_voice: str = "zh-CN-XiaoxiaoNeural",
        tts_speed: float = 1.2,
        tts_workflow: Optional[str] = None,
        ref_audio: Optional[str] = None,
        bgm_path: Optional[str] = None,
        bgm_volume: float = 0.10,
        pace_mode: str = "keep_original",
        silence_gap_threshold: float = 1.2,
        clip_padding: float = 0.45,
        progress_callback: Optional[Callable[[ProgressEvent], None]] = None,
    ) -> ScreenRecordingResult:
        self._report(progress_callback, "initializing", 0.02)

        source_video = Path(video_path).expanduser().resolve()
        if not source_video.exists():
            raise FileNotFoundError(f"Video not found: {source_video}")

        task_dir_str, task_id = create_task_output_dir()
        task_dir = Path(task_dir_str)
        work_dir = task_dir / "screen_recording_work"
        materials_dir = task_dir / "screen_recording_materials"
        work_dir.mkdir(parents=True, exist_ok=True)
        materials_dir.mkdir(parents=True, exist_ok=True)

        wav_path = work_dir / "source_16k.wav"
        transcript_json = work_dir / "transcript.json"
        segments_json = work_dir / "segments.json"
        srt_path = work_dir / "subtitles.srt"
        ass_path = work_dir / "subtitles.ass"
        dubbed_audio = work_dir / "dubbing_audio.mp3" if synthesize_dubbing else None
        processed_video = task_dir / "screen_recording_subtitled.mp4"
        source_duration = self._probe_duration(source_video)

        glossary_fixes, glossary_terms = self._parse_glossary(Path(glossary_path)) if glossary_path else ({}, [])
        correction_fixes = self._parse_corrections(Path(correction_path)) if correction_path else {}

        self._report(progress_callback, "extracting_audio", 0.08)
        self._extract_audio(source_video, wav_path)

        self._report(progress_callback, "transcribing_audio", 0.18)
        transcript = self._transcribe(
            wav_path,
            transcript_json,
            model=whisper_model,
            language=language,
            glossary_terms=glossary_terms,
        )

        self._report(progress_callback, "cleaning_subtitles", 0.45)
        segments = self._build_segments(transcript, glossary_fixes, correction_fixes)

        if ai_polish and segments:
            self._report(progress_callback, "polishing_subtitles", 0.56)
            segments = await self._polish_segments_with_ai(segments, industry_context)

        if source_duration <= 0 and segments:
            source_duration = max(segment.end for segment in segments)

        render_source_video = source_video
        render_duration = source_duration
        timeline_ranges = None
        dubbing_synthesized = False
        if synthesize_dubbing and pace_mode == "smart_compress" and source_duration > 0:
            self._report(progress_callback, "synthesizing_dubbing", 0.64)
            assert dubbed_audio is not None
            segments = self._merge_short_speech_segments(segments)
            segments, timeline_ranges, render_duration = await self._synthesize_dubbing_voice_paced(
                segments=segments,
                output_audio=dubbed_audio,
                source_duration=source_duration,
                tts_inference_mode=tts_inference_mode,
                tts_voice=tts_voice,
                tts_speed=tts_speed,
                tts_workflow=tts_workflow,
                ref_audio=ref_audio,
            )
            if timeline_ranges and render_duration < source_duration - 0.5:
                self._report(progress_callback, "compressing_timeline", 0.72)
                compact_video = work_dir / "compact_source.mp4"
                self._render_clip_timeline(source_video, timeline_ranges, compact_video)
                render_source_video = compact_video
            dubbing_synthesized = True
        elif pace_mode == "smart_compress" and source_duration > 0:
            timeline_ranges = self._build_compact_ranges(
                segments=segments,
                source_duration=source_duration,
                gap_threshold=silence_gap_threshold,
                padding=clip_padding,
            )
            compact_duration = sum(end - start for start, end in timeline_ranges)
            if timeline_ranges and compact_duration < source_duration - 0.5:
                self._report(progress_callback, "compressing_timeline", 0.60)
                compact_video = work_dir / "compact_source.mp4"
                self._render_clip_timeline(source_video, timeline_ranges, compact_video)
                segments = self._remap_segments_to_timeline(segments, timeline_ranges)
                render_source_video = compact_video
                render_duration = compact_duration

        segments_json.write_text(
            json.dumps([s.__dict__ for s in segments], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._write_srt(segments, srt_path)
        self._write_ass(segments, ass_path)

        if synthesize_dubbing and not dubbing_synthesized:
            self._report(progress_callback, "synthesizing_dubbing", 0.68)
            assert dubbed_audio is not None
            await self._synthesize_dubbing(
                segments,
                dubbed_audio,
                target_duration=render_duration,
                tts_inference_mode=tts_inference_mode,
                tts_voice=tts_voice,
                tts_speed=tts_speed,
                tts_workflow=tts_workflow,
                ref_audio=ref_audio,
            )

        self._report(progress_callback, "rendering_video", 0.82)
        self._render_video(
            source_video=render_source_video,
            subtitles_ass=ass_path,
            output_video=processed_video,
            dubbing_audio=dubbed_audio,
            bgm_path=bgm_path,
            bgm_volume=bgm_volume,
            target_duration=render_duration,
        )

        self._report(progress_callback, "exporting_materials", 0.92)
        materials_zip = self._export_materials(
            source_video=source_video,
            timeline_video=render_source_video,
            materials_dir=materials_dir,
            srt_path=srt_path,
            ass_path=ass_path,
            segments_json=segments_json,
            transcript_json=transcript_json,
            dubbing_audio=dubbed_audio,
            bgm_path=bgm_path,
            glossary_path=Path(glossary_path) if glossary_path else None,
            correction_path=Path(correction_path) if correction_path else None,
        )

        await self._persist_history_data(
            task_id=task_id,
            source_video=source_video,
            processed_video=processed_video,
            materials_dir=materials_dir,
            materials_zip=materials_zip,
            srt_path=srt_path,
            ass_path=ass_path,
            segments=segments,
            industry_context=industry_context,
            whisper_model=whisper_model,
            language=language,
            ai_polish=ai_polish,
            synthesize_dubbing=synthesize_dubbing,
            tts_inference_mode=tts_inference_mode,
            tts_voice=tts_voice,
            tts_speed=tts_speed,
            tts_workflow=tts_workflow,
            bgm_path=bgm_path,
            bgm_volume=bgm_volume,
            pace_mode=pace_mode,
            silence_gap_threshold=silence_gap_threshold,
            clip_padding=clip_padding,
            timeline_ranges=timeline_ranges,
            glossary_path=glossary_path,
            correction_path=correction_path,
        )

        self._report(progress_callback, "completed", 1.0)
        return ScreenRecordingResult(
            task_id=task_id,
            video_path=str(processed_video),
            materials_dir=str(materials_dir),
            materials_zip=str(materials_zip),
            srt_path=str(srt_path),
            ass_path=str(ass_path),
            segments_json_path=str(segments_json),
            transcript_json_path=str(transcript_json),
            dubbing_audio_path=str(dubbed_audio) if dubbed_audio else None,
        )

    def _transcribe(
        self,
        wav_path: Path,
        transcript_json: Path,
        model: str,
        language: str,
        glossary_terms: list[str],
    ) -> dict:
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError(
                "录屏处理需要 faster-whisper。请安装依赖：uv pip install faster-whisper"
            ) from e

        initial_prompt = None
        if glossary_terms:
            initial_prompt = "请准确识别以下专业术语：" + "、".join(glossary_terms[:80])

        whisper = WhisperModel(model, device="cpu", compute_type="int8")
        kwargs = {"language": language, "word_timestamps": False}
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt
        segments, info = whisper.transcribe(str(wav_path), **kwargs)

        output_segments = []
        for segment in segments:
            output_segments.append(
                {
                    "id": len(output_segments),
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": segment.text.strip(),
                }
            )
        result = {
            "text": "".join(s["text"] for s in output_segments),
            "segments": output_segments,
            "language": getattr(info, "language", language),
        }
        transcript_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def _build_segments(
        self,
        transcript: dict,
        glossary_fixes: dict[str, str],
        correction_fixes: dict[str, str],
    ) -> list[SubtitleSegment]:
        fixes = dict(self.TERM_FIXES)
        fixes.update(glossary_fixes)
        fixes.update(correction_fixes)

        result = []
        for item in transcript.get("segments", []):
            text = self._clean_text(item.get("text", ""), fixes)
            if not text:
                continue
            result.append(
                SubtitleSegment(
                    start=float(item["start"]),
                    end=float(item["end"]),
                    text=text,
                )
            )
        if not result:
            raise RuntimeError("No speech segments found in the input video")
        return result

    @staticmethod
    def _merge_short_speech_segments(
        segments: list[SubtitleSegment],
        max_gap: float = 1.4,
        min_chars: int = 26,
        max_chars: int = 92,
    ) -> list[SubtitleSegment]:
        if not segments:
            return []

        merged: list[SubtitleSegment] = []
        current = SubtitleSegment(segments[0].start, segments[0].end, segments[0].text)

        for segment in segments[1:]:
            gap = max(0.0, segment.start - current.end)
            combined_text = ScreenRecordingProcessor._join_narration_text(current.text, segment.text)
            should_merge = (
                gap <= max_gap
                and len(combined_text) <= max_chars
                and (
                    len(current.text) < min_chars
                    or len(segment.text) < min_chars
                    or len(combined_text) <= max_chars
                )
            )

            if should_merge:
                current = SubtitleSegment(current.start, segment.end, combined_text)
            else:
                merged.append(current)
                current = SubtitleSegment(segment.start, segment.end, segment.text)

        merged.append(current)
        return merged

    @staticmethod
    def _join_narration_text(left: str, right: str) -> str:
        left = left.strip()
        right = right.strip()
        if not left:
            return right
        if not right:
            return left
        if left[-1] in "。！？.!?；;":
            return left + right
        return left + "，" + right

    @staticmethod
    def _display_text(text: str) -> str:
        text = re.sub(r"[。．]+", "", text)
        text = re.sub(r"(?<![A-Za-z0-9])\.|\.(?![A-Za-z0-9])", "", text)
        return text.strip()

    async def _polish_segments_with_ai(
        self,
        segments: list[SubtitleSegment],
        industry_context: str,
    ) -> list[SubtitleSegment]:
        if not self.core.llm:
            return segments

        prompt_segments = [
            {"index": i + 1, "start": s.start, "end": s.end, "text": s.text}
            for i, s in enumerate(segments)
        ]
        prompt = (
            "你是录屏教程字幕编辑。请修正以下 ASR 字幕，使其适合视频所在行业。\n"
            "要求：\n"
            "1. 去掉口语填充词，例如 嗯、呃、啊、然后然后、这个那个 等。\n"
            "2. 保留原意和时间顺序，不要扩写，不要增加原文没有的信息。\n"
            "3. 保留专业术语、产品名、英文缩写、菜单名。\n"
            "4. 每条字幕尽量短，适合直接显示在视频底部。\n"
            "5. 只输出 JSON：{\"segments\":[{\"index\":1,\"text\":\"...\"}]}。\n\n"
            f"行业/业务背景：{industry_context or '未提供'}\n\n"
            f"字幕：{json.dumps(prompt_segments, ensure_ascii=False)}"
        )
        try:
            response = await self.core.llm(prompt=prompt, max_tokens=6000, temperature=0.1)
            match = re.search(r"\{[\s\S]*\}", str(response))
            if not match:
                return segments
            data = json.loads(match.group())
            corrected = {
                int(item["index"]): str(item["text"]).strip()
                for item in data.get("segments", [])
                if item.get("index") and item.get("text")
            }
            polished = []
            for i, seg in enumerate(segments, start=1):
                polished.append(SubtitleSegment(seg.start, seg.end, corrected.get(i, seg.text)))
            return polished
        except Exception as e:
            logger.warning(f"AI subtitle polish failed, using rule-cleaned subtitles: {e}")
            return segments

    async def _synthesize_dubbing(
        self,
        segments: list[SubtitleSegment],
        output_audio: Path,
        target_duration: float,
        tts_inference_mode: str,
        tts_voice: str,
        tts_speed: float,
        tts_workflow: Optional[str],
        ref_audio: Optional[str],
    ) -> None:
        tts_dir = output_audio.parent / "tts_segments"
        tts_dir.mkdir(parents=True, exist_ok=True)

        concat_lines: list[str] = []
        cursor = 0.0
        for idx, seg in enumerate(segments):
            gap = max(0.0, seg.start - cursor)
            if gap > 0.03:
                silence = tts_dir / f"silence_{idx:03d}.mp3"
                self._make_silence(silence, gap)
                concat_lines.append(f"file '{silence.resolve()}'")

            seg_audio = tts_dir / f"tts_{idx:03d}.mp3"
            tts_params = {
                "text": seg.text,
                "inference_mode": tts_inference_mode,
                "output_path": str(seg_audio),
            }
            if tts_inference_mode == "local":
                tts_params["voice"] = tts_voice
                tts_params["speed"] = tts_speed
            else:
                if tts_workflow:
                    tts_params["workflow"] = tts_workflow
                if ref_audio:
                    tts_params["ref_audio"] = ref_audio
            await self.core.tts(**tts_params)
            aligned_audio = tts_dir / f"aligned_{idx:03d}.mp3"
            self._fit_audio_to_duration(
                input_audio=seg_audio,
                output_audio=aligned_audio,
                target_duration=max(0.1, seg.end - seg.start),
            )
            concat_lines.append(f"file '{aligned_audio.resolve()}'")
            cursor = max(cursor, seg.end)

        tail_gap = max(0.0, target_duration - cursor)
        if tail_gap > 0.03:
            silence = tts_dir / "silence_tail.mp3"
            self._make_silence(silence, tail_gap)
            concat_lines.append(f"file '{silence.resolve()}'")

        concat_path = tts_dir / "concat.txt"
        concat_path.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-y", "-f", "concat", "-safe", "0",
                "-i", str(concat_path), "-t", f"{target_duration:.3f}",
                "-c:a", "libmp3lame", "-q:a", "4", str(output_audio),
            ],
            check=True,
        )

    async def _synthesize_dubbing_voice_paced(
        self,
        segments: list[SubtitleSegment],
        output_audio: Path,
        source_duration: float,
        tts_inference_mode: str,
        tts_voice: str,
        tts_speed: float,
        tts_workflow: Optional[str],
        ref_audio: Optional[str],
    ) -> tuple[list[SubtitleSegment], list[tuple[float, float]], float]:
        tts_dir = output_audio.parent / "tts_segments"
        tts_dir.mkdir(parents=True, exist_ok=True)

        concat_lines: list[str] = []
        compact_ranges: list[tuple[float, float]] = []
        remapped_segments: list[SubtitleSegment] = []
        cursor = 0.0
        inter_segment_gap = 0.22
        first_preroll = 1.0

        if first_preroll > 0.03:
            preroll = tts_dir / "gap_preroll.mp3"
            self._make_silence(preroll, first_preroll)
            concat_lines.append(f"file '{preroll.resolve()}'")
            cursor += first_preroll

        for idx, seg in enumerate(segments):
            gap = 0.0 if idx == 0 else inter_segment_gap
            if gap > 0.03:
                silence = tts_dir / f"gap_{idx:03d}.mp3"
                self._make_silence(silence, gap)
                concat_lines.append(f"file '{silence.resolve()}'")
                cursor += gap

            raw_audio = tts_dir / f"tts_{idx:03d}.mp3"
            tts_params = {
                "text": seg.text,
                "inference_mode": tts_inference_mode,
                "output_path": str(raw_audio),
            }
            if tts_inference_mode == "local":
                tts_params["voice"] = tts_voice
                tts_params["speed"] = tts_speed
            else:
                if tts_workflow:
                    tts_params["workflow"] = tts_workflow
                if ref_audio:
                    tts_params["ref_audio"] = ref_audio
            await self.core.tts(**tts_params)

            raw_duration = max(0.1, self._probe_duration(raw_audio))
            original_duration = max(0.1, seg.end - seg.start)
            voice_duration = min(raw_duration, original_duration)
            aligned_audio = tts_dir / f"aligned_{idx:03d}.mp3"
            self._fit_audio_to_duration(raw_audio, aligned_audio, voice_duration)
            concat_lines.append(f"file '{aligned_audio.resolve()}'")

            segment_start = cursor
            segment_end = cursor + voice_duration
            remapped_segments.append(SubtitleSegment(segment_start, segment_end, seg.text))

            range_duration = gap + voice_duration
            if idx == 0:
                range_duration += first_preroll
                desired_source_start = 0.0
            else:
                desired_source_start = max(0.0, seg.start - gap)
            source_start = min(desired_source_start, max(0.0, source_duration - range_duration))
            source_end = min(source_duration, source_start + range_duration)
            if source_end - source_start > 0.05:
                compact_ranges.append((source_start, source_end))

            cursor = segment_end

        tail_gap = 0.15
        if tail_gap > 0.03:
            silence = tts_dir / "gap_tail.mp3"
            self._make_silence(silence, tail_gap)
            concat_lines.append(f"file '{silence.resolve()}'")
            cursor += tail_gap

        concat_path = tts_dir / "concat.txt"
        concat_path.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-y", "-f", "concat", "-safe", "0",
                "-i", str(concat_path), "-t", f"{cursor:.3f}",
                "-c:a", "libmp3lame", "-q:a", "4", str(output_audio),
            ],
            check=True,
        )
        return remapped_segments, compact_ranges, cursor

    def _render_video(
        self,
        source_video: Path,
        subtitles_ass: Path,
        output_video: Path,
        dubbing_audio: Optional[Path],
        bgm_path: Optional[str],
        bgm_volume: float,
        target_duration: float,
    ) -> None:
        subtitles_filter = f"subtitles={self._ffmpeg_filter_path(subtitles_ass)}"
        if dubbing_audio:
            if bgm_path:
                resolved_bgm = VideoService()._resolve_bgm_path(bgm_path)
                filter_complex = (
                    f"[0:v]{subtitles_filter}[v];"
                    f"[1:a]volume=1.0,apad,atrim=0:{target_duration:.3f},asetpts=N/SR/TB[voice];"
                    f"[2:a]volume={bgm_volume},atrim=0:{target_duration:.3f},asetpts=N/SR/TB[bgm];"
                    "[voice][bgm]amix=inputs=2:duration=first:normalize=0[a]"
                )
                cmd = [
                    "ffmpeg", "-hide_banner", "-y",
                    "-i", str(source_video),
                    "-i", str(dubbing_audio),
                    "-stream_loop", "-1", "-i", resolved_bgm,
                    "-filter_complex", filter_complex,
                    "-map", "[v]", "-map", "[a]",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                    "-c:a", "aac", "-b:a", "192k", "-t", f"{target_duration:.3f}",
                    "-movflags", "+faststart",
                    str(output_video),
                ]
            else:
                filter_complex = (
                    f"[0:v]{subtitles_filter}[v];"
                    f"[1:a]apad,atrim=0:{target_duration:.3f},asetpts=N/SR/TB[a]"
                )
                cmd = [
                    "ffmpeg", "-hide_banner", "-y", "-i", str(source_video), "-i", str(dubbing_audio),
                    "-filter_complex", filter_complex,
                    "-map", "[v]", "-map", "[a]",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                    "-c:a", "aac", "-b:a", "192k", "-t", f"{target_duration:.3f}",
                    "-movflags", "+faststart",
                    str(output_video),
                ]
        else:
            cmd = [
                "ffmpeg", "-hide_banner", "-y", "-i", str(source_video),
                "-vf", subtitles_filter,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-c:a", "copy", "-movflags", "+faststart", str(output_video),
            ]
        subprocess.run(cmd, check=True)

    def _build_compact_ranges(
        self,
        segments: list[SubtitleSegment],
        source_duration: float,
        gap_threshold: float,
        padding: float,
    ) -> list[tuple[float, float]]:
        ranges: list[tuple[float, float]] = []
        for segment in segments:
            start = max(0.0, segment.start - padding)
            end = min(source_duration, segment.end + padding)
            if not ranges or start - ranges[-1][1] > gap_threshold:
                ranges.append((start, end))
            else:
                prev_start, prev_end = ranges[-1]
                ranges[-1] = (prev_start, max(prev_end, end))

        if not ranges:
            return [(0.0, source_duration)]
        return [(start, end) for start, end in ranges if end - start > 0.05]

    def _render_clip_timeline(
        self,
        source_video: Path,
        ranges: list[tuple[float, float]],
        output_video: Path,
    ) -> None:
        clips_dir = output_video.parent / "timeline_clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        clip_paths: list[Path] = []

        for idx, (start, end) in enumerate(ranges):
            clip_path = clips_dir / f"clip_{idx:03d}.mp4"
            duration = max(0.05, end - start)
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-y",
                    "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(source_video),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                    "-c:a", "aac", "-b:a", "160k",
                    "-avoid_negative_ts", "make_zero", str(clip_path),
                ],
                check=True,
            )
            clip_paths.append(clip_path)

        concat_path = clips_dir / "concat.txt"
        concat_path.write_text(
            "\n".join(f"file '{clip.resolve()}'" for clip in clip_paths) + "\n",
            encoding="utf-8",
        )
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-y", "-f", "concat", "-safe", "0",
                "-i", str(concat_path), "-c", "copy", str(output_video),
            ],
            check=True,
        )

    @staticmethod
    def _remap_segments_to_timeline(
        segments: list[SubtitleSegment],
        ranges: list[tuple[float, float]],
    ) -> list[SubtitleSegment]:
        offsets: list[tuple[float, float, float]] = []
        cursor = 0.0
        for start, end in ranges:
            offsets.append((start, end, cursor))
            cursor += end - start

        remapped: list[SubtitleSegment] = []
        for segment in segments:
            for start, end, out_start in offsets:
                if segment.end <= start or segment.start >= end:
                    continue
                new_start = out_start + max(0.0, segment.start - start)
                new_end = out_start + min(end, segment.end) - start
                if new_end - new_start > 0.05:
                    remapped.append(SubtitleSegment(new_start, new_end, segment.text))
                break
        return remapped

    def _export_materials(
        self,
        source_video: Path,
        timeline_video: Path,
        materials_dir: Path,
        srt_path: Path,
        ass_path: Path,
        segments_json: Path,
        transcript_json: Path,
        dubbing_audio: Optional[Path],
        bgm_path: Optional[str],
        glossary_path: Optional[Path],
        correction_path: Optional[Path],
    ) -> Path:
        self._copy_or_link(source_video, materials_dir / "01_original_video" / source_video.name)
        self._copy_or_link(timeline_video, materials_dir / "02_timeline_video" / timeline_video.name)
        self._copy_or_link(srt_path, materials_dir / "03_subtitles" / "subtitles.srt")
        self._copy_or_link(ass_path, materials_dir / "03_subtitles" / "subtitles.ass")
        self._copy_or_link(segments_json, materials_dir / "04_transcript" / "segments.json")
        self._copy_or_link(transcript_json, materials_dir / "04_transcript" / "transcript.json")
        if dubbing_audio and dubbing_audio.exists():
            self._copy_or_link(dubbing_audio, materials_dir / "05_dubbing" / "dubbing_audio.mp3")
        if bgm_path:
            resolved_bgm = Path(VideoService()._resolve_bgm_path(bgm_path))
            self._copy_or_link(resolved_bgm, materials_dir / "07_bgm" / resolved_bgm.name)
        if glossary_path and glossary_path.exists():
            self._copy_or_link(glossary_path, materials_dir / "06_rules" / glossary_path.name)
        if correction_path and correction_path.exists():
            self._copy_or_link(correction_path, materials_dir / "06_rules" / correction_path.name)

        readme = materials_dir / "README.md"
        readme.write_text(
            "# 录屏处理素材包\n\n"
            "- `01_original_video/`: 原始录屏视频\n"
            "- `02_timeline_video/`: 已按节奏处理但未烧录字幕、未混入配音和背景音乐的视频底板\n"
            "- `03_subtitles/`: 可导入剪映继续编辑的 SRT/ASS 字幕，请与 `02_timeline_video/` 中的视频对齐使用\n"
            "- `04_transcript/`: ASR 原始结果与清洗后的分段\n"
            "- `05_dubbing/`: 可选配音合成音频，请作为独立音轨导入剪映\n"
            "- `06_rules/`: 本次使用的术语/修正规则文件\n"
            "- `07_bgm/`: 本次选择的背景音乐，请作为独立音轨导入剪映\n",
            encoding="utf-8",
        )

        zip_path = materials_dir.parent / f"{materials_dir.name}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in materials_dir.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(materials_dir.parent))
        return zip_path

    async def _persist_history_data(
        self,
        task_id: str,
        source_video: Path,
        processed_video: Path,
        materials_dir: Path,
        materials_zip: Path,
        srt_path: Path,
        ass_path: Path,
        segments: list[SubtitleSegment],
        industry_context: str,
        whisper_model: str,
        language: str,
        ai_polish: bool,
        synthesize_dubbing: bool,
        tts_inference_mode: str,
        tts_voice: str,
        tts_speed: float,
        tts_workflow: Optional[str],
        bgm_path: Optional[str],
        bgm_volume: float,
        pace_mode: str,
        silence_gap_threshold: float,
        clip_padding: float,
        timeline_ranges: Optional[list[tuple[float, float]]],
        glossary_path: Optional[str],
        correction_path: Optional[str],
    ) -> None:
        if not getattr(self.core, "persistence", None):
            logger.warning("No persistence service available, skipping screen recording history")
            return

        try:
            duration = self._probe_duration(processed_video)
            file_size = processed_video.stat().st_size if processed_video.exists() else 0
            title = source_video.stem or "Screen Recording"
            now = datetime.now().isoformat()
            text_preview = " ".join(seg.text for seg in segments[:8])

            metadata = {
                "task_id": task_id,
                "title": title,
                "created_at": now,
                "completed_at": now,
                "status": "completed",
                "pipeline": "screen_recording",
                "input": {
                    "title": title,
                    "text": text_preview,
                    "mode": "screen_recording",
                    "n_scenes": len(segments),
                    "source_video": str(source_video),
                    "industry_context": industry_context,
                    "whisper_model": whisper_model,
                    "language": language,
                    "ai_polish": ai_polish,
                    "synthesize_dubbing": synthesize_dubbing,
                    "tts_inference_mode": tts_inference_mode,
                    "tts_voice": tts_voice,
                    "tts_speed": tts_speed,
                    "tts_workflow": tts_workflow,
                    "bgm_path": bgm_path,
                    "bgm_volume": bgm_volume,
                    "pace_mode": pace_mode,
                    "silence_gap_threshold": silence_gap_threshold,
                    "clip_padding": clip_padding,
                    "glossary_path": glossary_path,
                    "correction_path": correction_path,
                },
                "result": {
                    "video_path": str(processed_video),
                    "video_paths": [str(processed_video)],
                    "materials_dir": str(materials_dir),
                    "materials_zip": str(materials_zip),
                    "srt_path": str(srt_path),
                    "ass_path": str(ass_path),
                    "duration": duration,
                    "file_size": file_size,
                    "n_frames": len(segments),
                    "n_segments": len(segments),
                    "timeline_ranges": timeline_ranges,
                },
                "config": {
                    "llm_model": self.core.config.get("llm", {}).get("model", "unknown")
                    if getattr(self.core, "config", None)
                    else "unknown",
                },
            }
            await self.core.persistence.save_task_metadata(task_id, metadata)
            logger.info(f"💾 Saved screen recording task metadata: {task_id}")
        except Exception as e:
            logger.error(f"Failed to persist screen recording history: {e}")

    @classmethod
    def _parse_glossary(cls, path: Path) -> tuple[dict[str, str], list[str]]:
        if not path.exists():
            return {}, []
        fixes: dict[str, str] = {}
        terms: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped.startswith("|") or "---" in stripped:
                continue
            parts = [p.strip().strip("*") for p in stripped.split("|")]
            if len(parts) >= 4:
                correct = parts[1]
                aliases = [a.strip() for a in re.split(r"[,，、]", parts[3]) if a.strip()]
                if correct and correct not in ("术语", "正确文本"):
                    terms.append(correct)
                for alias in aliases:
                    if alias and correct and alias != correct:
                        fixes[alias] = correct
                        terms.append(alias)
        return fixes, list(dict.fromkeys(terms))

    @classmethod
    def _parse_corrections(cls, path: Path) -> dict[str, str]:
        if not path.exists():
            return {}
        content = path.read_text(encoding="utf-8")
        fixes: dict[str, str] = {}
        rows = re.findall(r"\|([^|]+)\|([^|]+)\|(?:[^|]*\|)?", content)
        for wrong, right in rows:
            wrong = wrong.strip()
            right = right.strip()
            if wrong and right and wrong not in ("错误识别", "错误文本", "---") and right not in ("正确文本", "---"):
                if wrong != right:
                    fixes[wrong] = right
        return fixes

    @classmethod
    def _clean_text(cls, text: str, fixes: dict[str, str]) -> str:
        text = text.strip()
        for src, dst in fixes.items():
            text = text.replace(src, dst)
        for pattern in cls.FILLER_PATTERNS:
            text = re.sub(pattern, "", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = text.strip("，,、 ")
        if text and text[-1] not in "。！？.!?":
            text += "。"
        return text

    @staticmethod
    def _write_srt(segments: list[SubtitleSegment], output: Path) -> None:
        lines = []
        for idx, seg in enumerate(ScreenRecordingProcessor._caption_events(segments), start=1):
            lines.extend([
                str(idx),
                f"{ScreenRecordingProcessor._timestamp_srt(seg.start)} --> {ScreenRecordingProcessor._timestamp_srt(seg.end)}",
                ScreenRecordingProcessor._wrap_caption_text(
                    ScreenRecordingProcessor._display_text(seg.text),
                    max_chars=22,
                    line_break="\n",
                ),
                "",
            ])
        output.write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _write_ass(segments: list[SubtitleSegment], output: Path) -> None:
        header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,42,&H00FFFFFF,&H00000000,&H00000000,&H70000000,0,0,0,0,100,100,0,0,1,3,1,2,120,120,76,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        lines = [header]
        for seg in ScreenRecordingProcessor._caption_events(segments):
            text = ScreenRecordingProcessor._wrap_caption_text(
                ScreenRecordingProcessor._display_text(seg.text).replace("\n", " ").replace(",", "，"),
                max_chars=22,
                line_break=r"\N",
            )
            lines.append(
                f"Dialogue: 0,{ScreenRecordingProcessor._timestamp_ass(seg.start)},"
                f"{ScreenRecordingProcessor._timestamp_ass(seg.end)},Default,,0,0,0,,{text}\n"
            )
        output.write_text("".join(lines), encoding="utf-8")

    @staticmethod
    def _caption_events(segments: list[SubtitleSegment], max_chars: int = 44) -> list[SubtitleSegment]:
        events: list[SubtitleSegment] = []
        for seg in segments:
            chunks = ScreenRecordingProcessor._split_caption_text(seg.text, max_chars=max_chars)
            if not chunks:
                continue
            total_chars = sum(max(1, len(chunk)) for chunk in chunks)
            duration = max(0.1, seg.end - seg.start)
            cursor = seg.start
            for idx, chunk in enumerate(chunks):
                if idx == len(chunks) - 1:
                    end = seg.end
                else:
                    ratio = max(1, len(chunk)) / total_chars
                    end = min(seg.end, cursor + duration * ratio)
                if end - cursor > 0.05:
                    events.append(SubtitleSegment(cursor, end, chunk))
                cursor = end
        return events

    @staticmethod
    def _split_caption_text(text: str, max_chars: int) -> list[str]:
        text = re.sub(r"\s+", " ", text.strip())
        if not text:
            return []

        parts = [part for part in re.split(r"(?<=[。！？!?；;，,、])", text) if part]
        chunks: list[str] = []
        current = ""
        for part in parts:
            if len(part) > max_chars:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(part[i:i + max_chars] for i in range(0, len(part), max_chars))
            elif current and len(current) + len(part) > max_chars:
                chunks.append(current)
                current = part
            else:
                current += part
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _wrap_caption_text(text: str, max_chars: int, line_break: str) -> str:
        text = re.sub(r"\s+", " ", text.strip())
        chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
        return line_break.join(chunks) if chunks else text

    @staticmethod
    def _timestamp_srt(seconds: float) -> str:
        ms_total = int(round(seconds * 1000))
        h, rem = divmod(ms_total, 3600_000)
        m, rem = divmod(rem, 60_000)
        s, ms = divmod(rem, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    @staticmethod
    def _timestamp_ass(seconds: float) -> str:
        cs_total = int(round(seconds * 100))
        h, rem = divmod(cs_total, 3600 * 100)
        m, rem = divmod(rem, 60 * 100)
        s, cs = divmod(rem, 100)
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    @staticmethod
    def _extract_audio(video: Path, output_wav: Path) -> None:
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-y", "-i", str(video),
                "-vn", "-ac", "1", "-ar", "16000", str(output_wav),
            ],
            check=True,
        )

    @staticmethod
    def _make_silence(output: Path, duration: float) -> None:
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-y", "-f", "lavfi", "-i",
                "anullsrc=r=44100:cl=stereo", "-t", f"{duration:.3f}",
                "-c:a", "libmp3lame", str(output),
            ],
            check=True,
        )

    def _fit_audio_to_duration(self, input_audio: Path, output_audio: Path, target_duration: float) -> None:
        raw_duration = self._probe_duration(input_audio)
        if raw_duration <= 0:
            self._make_silence(output_audio, target_duration)
            return

        filters = []
        # Keep the user's selected TTS speed. Only compress narration that would
        # overrun the original subtitle slot; shorter narration becomes silence.
        tempo = raw_duration / target_duration if target_duration > 0 else 1.0
        if tempo > 1.02:
            filters.extend(self._atempo_filters(tempo))
        filters.append("apad")
        filters.append(f"atrim=0:{target_duration:.3f}")
        filters.append("asetpts=N/SR/TB")

        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-y", "-i", str(input_audio),
                "-af", ",".join(filters),
                "-c:a", "libmp3lame", "-q:a", "4", str(output_audio),
            ],
            check=True,
        )

    @staticmethod
    def _atempo_filters(tempo: float) -> list[str]:
        filters = []
        remaining = tempo
        while remaining > 2.0:
            filters.append("atempo=2.0")
            remaining /= 2.0
        while remaining < 0.5:
            filters.append("atempo=0.5")
            remaining /= 0.5
        filters.append(f"atempo={remaining:.4f}")
        return filters

    @staticmethod
    def _probe_duration(path: Path) -> float:
        try:
            out = subprocess.check_output(
                [
                    "ffprobe", "-hide_banner", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                text=True,
            )
            return float(out.strip())
        except Exception:
            return 0.0

    @staticmethod
    def _has_audio_stream(path: Path) -> bool:
        try:
            out = subprocess.check_output(
                [
                    "ffprobe", "-hide_banner", "-v", "error",
                    "-select_streams", "a",
                    "-show_entries", "stream=codec_type",
                    "-of", "json", str(path),
                ],
                text=True,
            )
            data = json.loads(out)
            return bool(data.get("streams"))
        except Exception:
            return False

    @staticmethod
    def _ffmpeg_filter_path(path: Path) -> str:
        return str(path.resolve()).replace("\\", "\\\\").replace(":", "\\:")

    @staticmethod
    def _copy_or_link(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        try:
            dst.symlink_to(src.resolve())
        except Exception:
            shutil.copy2(src, dst)

    @staticmethod
    def _report(callback: Optional[Callable[[ProgressEvent], None]], event_type: str, progress: float) -> None:
        if callback:
            callback(ProgressEvent(event_type=event_type, progress=progress))
