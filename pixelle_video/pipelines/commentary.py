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
Commentary Video Generation Pipeline

End-to-end pipeline that turns a source video into an AI-commentary short video.
Supports multi-segment generation (e.g., 3 segments of 5 min each from a 15-min total).

Features:
- Auto subtitle detection (external + embedded)
- Multi-segment story splitting
- AI-generated commentary script per segment with structured output
- AI-generated cover image per segment
- Precise TTS with atempo adjustment
- ASS caption burn-in
- Top progress bar overlay
- 3-second cover intro per segment
- Final concatenation of all segments
"""

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, List, Tuple

from loguru import logger

from pixelle_video.pipelines.base import BasePipeline
from pixelle_video.models.progress import ProgressEvent
from pixelle_video.models.storyboard import (
    Storyboard,
    StoryboardFrame,
    StoryboardConfig,
    VideoGenerationResult,
)
from pixelle_video.models.commentary import (
    CommentaryConfig,
    CommentaryScript,
)
from pixelle_video.prompts.commentary_script import (
    build_commentary_prompt,
    build_segment_commentary_prompt,
    build_fallback_script_prompt,
)
from pixelle_video.services.subtitle_extractor import SubtitleExtractor, StoryResult
from pixelle_video.services.commentary_compositor import CommentaryCompositor
from pixelle_video.utils.os_util import (
    create_task_output_dir,
    get_task_final_video_path,
)


class CommentaryPipeline(BasePipeline):
    """
    Commentary video generation pipeline.

    Inherits BasePipeline (not LinearVideoPipeline) because commentary
    does not use HTML template / FrameProcessor rendering.
    """

    async def __call__(
        self,
        text: str,
        progress_callback: Optional[Callable[[ProgressEvent], None]] = None,
        **kwargs,
    ) -> VideoGenerationResult:
        """
        Execute commentary pipeline.

        Supports single-segment or multi-segment generation.
        When segment_count > 1, the story is split evenly and each segment
        gets its own script, cover, and video, then concatenated.
        """
        # ====== 1. Setup ======
        self._report_progress(progress_callback, "initializing", 0.01)

        source_video = kwargs.get("source_video")
        if not source_video:
            raise ValueError("source_video is required")

        video_path = Path(source_video)
        if not video_path.exists():
            raise FileNotFoundError(f"Source video not found: {source_video}")

        task_dir, task_id = create_task_output_dir()
        final_video_path = get_task_final_video_path(task_id)
        segment_count = kwargs.get("segment_count", 1)
        target_duration = kwargs.get("target_duration", 300)  # per-video duration
        segment_duration = target_duration
        total_duration = target_duration * segment_count

        logger.info(f"🎙️ CommentaryPipeline starting")
        logger.info(f"   Source: {video_path}")
        logger.info(f"   Task: {task_id}")
        logger.info(f"   Segments: {segment_count}, Duration per segment: {segment_duration}s, Total: {total_duration}s")

        cfg = CommentaryConfig(
            source_video=str(video_path),
            target_duration=target_duration,
            tts_voice=kwargs.get("tts_voice", "zh-CN-YunxiNeural"),
            tts_rate=kwargs.get("tts_rate", "+18%"),
            narration_slot_ratio=kwargs.get("narration_slot_ratio", 0.82),
            bgm_path=kwargs.get("bgm_path"),
            content_start=kwargs.get("content_start"),
            content_end=kwargs.get("content_end"),
            cover_headline=kwargs.get("cover_headline"),
            cover_question=kwargs.get("cover_question"),
            mask_subtitles=kwargs.get("mask_subtitles", False),
            keep_original_audio=kwargs.get("keep_original_audio", True),
            original_audio_volume=kwargs.get("original_audio_volume", 0.2),
            segment_count=segment_count,
        )

        # ====== 2. Extract Story ======
        self._report_progress(progress_callback, "extracting_story", 0.05)
        extractor = SubtitleExtractor()
        story = extractor.extract_story(str(video_path))

        logger.info(f"📖 Story source: {story.source}")
        logger.info(f"   Lines: {len(story.lines)}, Has timestamps: {story.has_timestamps}")

        # Auto-detect intro/outro from subtitle timestamps
        video_duration = self._ffprobe_duration(video_path)
        if story.inferred_start is not None and cfg.content_start is None:
            cfg.content_start = story.inferred_start
            logger.info(f"📍 Using auto-detected content_start: {cfg.content_start:.1f}s")
        if story.inferred_end is not None and cfg.content_end is None:
            cfg.content_end = story.inferred_end
            logger.info(f"📍 Using auto-detected content_end: {cfg.content_end:.1f}s")

        # ====== 3. Fallback: video analysis if no subtitles ======
        if not story.text:
            logger.warning("⚠️ No subtitles found, using video analysis fallback")
            try:
                first_frame_desc = await self.core.video_analysis(
                    str(video_path), source="llm"
                )
                story.text = first_frame_desc
                story.has_timestamps = False
                story.source = "video_analysis"
                logger.info(f"📹 Video analysis result: {first_frame_desc[:100]}...")
            except Exception as e:
                logger.error(f"Video analysis failed: {e}")
                raise RuntimeError(
                    f"无法获取视频剧情文本。请确保视频有字幕文件（同目录 .srt/.ass/.vtt/.txt），"
                    f"或视频包含内嵌字幕流。错误: {e}"
                )

        # ====== 4. Split story into segments ======
        video_duration = self._ffprobe_duration(video_path)
        content_start = cfg.content_start or 0.0
        content_end = cfg.content_end or video_duration
        total_content = content_end - content_start

        segment_stories = self._split_story_for_segments(
            story, content_start, content_end, segment_count
        )

        # ====== 5. Generate scripts + compose videos for each segment ======
        segment_videos: List[Path] = []
        all_scripts: List[CommentaryScript] = []
        compositor = CommentaryCompositor(self.core)

        for seg_idx, (seg_story_text, seg_content_start, seg_content_end) in enumerate(segment_stories):
            seg_num = seg_idx + 1
            seg_progress_start = 0.10 + seg_idx * (0.60 / segment_count)
            seg_progress_end = 0.10 + (seg_idx + 1) * (0.60 / segment_count)

            self._report_progress(
                progress_callback,
                f"generating_segment_{seg_num}",
                seg_progress_start,
                extra_info=f"segment {seg_num}/{segment_count}",
            )

            # Build segment prompt
            prompt = build_segment_commentary_prompt(
                story_text=seg_story_text,
                seg_idx=seg_idx,
                segment_count=segment_count,
                segment_duration=segment_duration,
                content_start=seg_content_start,
                content_end=seg_content_end,
                has_timestamps=story.has_timestamps,
                video_title=video_path.stem,
            )

            logger.info(f"🤖 Generating script for segment {seg_num}/{segment_count}...")
            script: CommentaryScript = await self.llm(
                prompt=prompt,
                response_type=CommentaryScript,
                temperature=0.7,
                max_tokens=8000,
            )

            # Set cover background to middle of this segment's time range (skip intro)
            seg_duration_actual = seg_content_end - seg_content_start
            script.cover.background_time = seg_content_start + seg_duration_actual * 0.5
            logger.info(f"🖼️ Segment {seg_num} cover background: {script.cover.background_time:.1f}s")

            # Apply manual overrides only to first segment's cover
            if seg_idx == 0:
                if cfg.cover_headline:
                    script.cover.headline = cfg.cover_headline
                if cfg.cover_question:
                    script.cover.question = cfg.cover_question
            elif segment_count > 1:
                # Add segment number to cover for subsequent segments
                script.cover.headline = f"第{seg_num}段：{script.cover.headline}"

            logger.info(f"✅ Segment {seg_num} script: {len(script.chunks)} chunks")
            all_scripts.append(script)

            # Compose segment video
            self._report_progress(
                progress_callback,
                f"composing_segment_{seg_num}",
                seg_progress_start + (seg_progress_end - seg_progress_start) * 0.3,
                extra_info=f"segment {seg_num}/{segment_count}",
            )

            seg_cfg = CommentaryConfig(
                source_video=cfg.source_video,
                target_duration=segment_duration,
                tts_voice=cfg.tts_voice,
                tts_rate=cfg.tts_rate,
                narration_slot_ratio=cfg.narration_slot_ratio,
                bgm_path=cfg.bgm_path,
                content_start=seg_content_start,
                content_end=seg_content_end,
                mask_subtitles=cfg.mask_subtitles,
                segment_count=1,
            )

            seg_task_dir = f"{task_dir}/segment_{seg_num:02d}"
            Path(seg_task_dir).mkdir(parents=True, exist_ok=True)

            seg_video = await compositor.compose_commentary(
                video_path=video_path,
                chunks=script.chunks,
                cover=script.cover,
                progress_segments=script.progress_segments,
                cfg=seg_cfg,
                task_dir=seg_task_dir,
            )
            segment_videos.append(seg_video)
            logger.info(f"✅ Segment {seg_num} video complete: {seg_video}")

        # ====== 6. Save segments as independent files ======
        self._report_progress(progress_callback, "saving_segments", 0.85)

        Path(final_video_path).parent.mkdir(parents=True, exist_ok=True)
        all_video_paths: List[str] = []

        if len(segment_videos) == 1:
            shutil.copy2(segment_videos[0], final_video_path)
            all_video_paths.append(final_video_path)
            logger.info(f"📹 Single segment copied to: {final_video_path}")
        else:
            # Save each segment as an independent file
            fp = Path(final_video_path)
            stem, ext = fp.stem, fp.suffix  # e.g. "final", ".mp4"
            for seg_idx, seg_video in enumerate(segment_videos):
                seg_num = seg_idx + 1
                if seg_idx == 0:
                    seg_output = str(fp)
                else:
                    seg_output = str(fp.with_name(f"{stem}_{seg_num:02d}{ext}"))
                shutil.copy2(seg_video, seg_output)
                all_video_paths.append(seg_output)
                logger.info(f"📹 Segment {seg_num} saved to: {seg_output}")

        final_video_path = all_video_paths[0]

        # ====== 7. Build Result ======
        self._report_progress(progress_callback, "completed", 1.0)

        video_size = Path(final_video_path).stat().st_size
        total_chunks = sum(len(s.chunks) for s in all_scripts)

        # Build a lightweight Storyboard for compatibility
        sb_config = StoryboardConfig(
            media_width=1920, media_height=1080,
            task_id=task_id,
            n_storyboard=total_chunks,
            frame_template="commentary",
        )
        storyboard = Storyboard(
            title=all_scripts[0].title if all_scripts else "Commentary",
            config=sb_config,
            frames=[
                StoryboardFrame(index=i, narration=c.text, image_prompt=None)
                for s in all_scripts
                for i, c in enumerate(s.chunks)
            ],
            final_video_path=final_video_path,
            total_duration=target_duration,
            created_at=datetime.now(),
            completed_at=datetime.now(),
        )

        result = VideoGenerationResult(
            video_path=final_video_path,
            storyboard=storyboard,
            duration=total_duration,
            file_size=video_size,
            additional_video_paths=all_video_paths[1:] if len(all_video_paths) > 1 else [],
        )

        logger.success(f"🎬 Commentary pipeline complete: {final_video_path}")
        logger.info(f"   Segments: {segment_count}, Total Duration: {target_duration}s")
        logger.info(f"   Size: {video_size / (1024*1024):.2f} MB")
        logger.info(f"   Total Chunks: {total_chunks}")

        # Persist metadata
        await self._persist_task_data(task_id, result, kwargs, all_scripts)

        return result

    # ==================== Helpers ====================

    def _split_story_for_segments(
        self,
        story: StoryResult,
        content_start: float,
        content_end: float,
        segment_count: int,
    ) -> List[Tuple[str, float, float]]:
        """
        Split story into segments by TIME RANGE (not by line count).

        Each segment covers a contiguous time slice of the content range.
        Subtitles within that slice are collected for the LLM prompt.

        Returns list of (story_text, seg_content_start, seg_content_end).
        """
        total_content = content_end - content_start
        seg_duration = total_content / segment_count
        results = []

        for seg_idx in range(segment_count):
            # Compute time slice for this segment
            seg_c_start = content_start + seg_idx * seg_duration
            seg_c_end = content_end if seg_idx == segment_count - 1 else content_start + (seg_idx + 1) * seg_duration

            # Collect subtitle lines within this time slice
            seg_lines = []
            for line in story.lines:
                if line.start is not None and seg_c_start <= line.start < seg_c_end:
                    seg_lines.append(line)

            # Build text from collected lines
            if story.has_timestamps:
                seg_text = "\n".join(
                    f"[{l.start:.1f}s-{l.end:.1f}s] {l.text}"
                    for l in seg_lines if l.text
                )
            else:
                # No timestamps: use proportional line slice
                text_lines = story.text.splitlines() if story.text else []
                lines_per_seg = max(1, len(text_lines) // segment_count)
                start_idx = seg_idx * lines_per_seg
                end_idx = len(text_lines) if seg_idx == segment_count - 1 else (seg_idx + 1) * lines_per_seg
                seg_text = "\n".join(text_lines[start_idx:end_idx])

            # Fallback: if segment has no text, grab the closest line(s) to the midpoint
            if not seg_text.strip():
                midpoint = (seg_c_start + seg_c_end) / 2
                if story.lines:
                    closest = min(
                        (l for l in story.lines if l.start is not None),
                        key=lambda l: abs(l.start - midpoint),
                        default=None,
                    )
                    if closest and closest.text:
                        seg_text = f"[{closest.start:.1f}s-{closest.end:.1f}s] {closest.text}"
                if not seg_text.strip():
                    seg_text = f"（第{seg_idx + 1}段剧情，时间范围 {seg_c_start:.1f}s - {seg_c_end:.1f}s）"

            results.append((seg_text, seg_c_start, seg_c_end))

        # Validate: must return exactly segment_count segments
        if len(results) != segment_count:
            logger.warning(f"Segment count mismatch: expected {segment_count}, got {len(results)}")

        logger.info(f"📦 Story split into {len(results)} segments")
        for i, (text, s, e) in enumerate(results):
            logger.info(f"   Segment {i+1}: {s:.1f}s-{e:.1f}s, text={len(text)} chars")

        return results

    @staticmethod
    def _concat_segments(segment_videos: List[Path], task_dir: Path) -> Path:
        """Concatenate multiple segment videos into one."""
        concat_file = task_dir / "segment_concat.txt"
        concat_file.write_text(
            "\n".join(f"file '{v}'" for v in segment_videos) + "\n",
            encoding="utf-8",
        )
        output = task_dir / "commentary_combined.mp4"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_file), "-c", "copy", str(output),
        ], check=True)
        return output

    @staticmethod
    def _ffprobe_duration(path: Path) -> float:
        import subprocess, json
        out = subprocess.check_output([
            "ffprobe", "-hide_banner", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", str(path),
        ], text=True)
        return float(json.loads(out)["format"]["duration"])

    async def _persist_task_data(
        self,
        task_id: str,
        result: VideoGenerationResult,
        input_params: dict,
        scripts: List[CommentaryScript],
    ):
        """Persist task metadata for history tracking."""
        try:
            # Derive title from source video filename
            source = input_params.get("source_video", "")
            video_title = Path(source).stem if source else "Commentary"

            metadata = {
                "task_id": task_id,
                "title": video_title,
                "created_at": datetime.now().isoformat(),
                "completed_at": datetime.now().isoformat(),
                "status": "completed",
                "pipeline": "commentary",
                "input": {
                    "title": video_title,
                    "text": input_params.get("text", ""),
                    "mode": "commentary",
                    "n_scenes": sum(len(s.chunks) for s in scripts),
                    "tts_inference_mode": input_params.get("tts_inference_mode", "edge"),
                    "tts_voice": input_params.get("tts_voice", "zh-CN-YunxiNeural"),
                    "tts_rate": input_params.get("tts_rate", "+18%"),
                    "source_video": source,
                    "target_duration": input_params.get("target_duration"),
                    "segment_count": input_params.get("segment_count", 1),
                    "bgm_path": input_params.get("bgm_path"),
                },
                "result": {
                    "video_path": result.video_path,
                    "video_paths": [result.video_path] + list(result.additional_video_paths),
                    "duration": result.duration,
                    "file_size": result.file_size,
                    "n_frames": sum(len(s.chunks) for s in scripts),
                    "n_segments": len(scripts),
                },
                "config": {
                    "llm_model": self.core.config.get("llm", {}).get("model", "unknown"),
                },
            }
            await self.core.persistence.save_task_metadata(task_id, metadata)
            # Save storyboard for History detail page
            if result.storyboard:
                await self.core.persistence.save_storyboard(task_id, result.storyboard)
                logger.info(f"💾 Saved storyboard: {task_id}")
            logger.info(f"💾 Saved task metadata: {task_id}")
        except Exception as e:
            logger.error(f"Failed to persist task data: {e}")
