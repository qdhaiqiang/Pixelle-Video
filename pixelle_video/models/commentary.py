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
Commentary (Video Commentary) data models

Structured output models for AI-generated commentary scripts.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class SourceWindow(BaseModel):
    """A time window within the source video for a chunk"""
    start: float = Field(description="Start time in seconds within source video")
    end: float = Field(description="End time in seconds within source video")


class CommentaryChunk(BaseModel):
    """A single commentary segment"""
    chunk_id: str = Field(description="Unique identifier (e.g., n001, n002)")
    start: float = Field(description="Start time in final video (seconds)")
    end: float = Field(description="End time in final video (seconds)")
    event: str = Field(description="Story event type (e.g., opening, conflict, twist, climax, resolution)")
    text: str = Field(description="Commentary narration text in spoken Chinese")
    source_windows: List[SourceWindow] = Field(description="Time windows from source video to use as visuals")


class CommentaryCover(BaseModel):
    """Cover intro configuration (3-second AI-style cover)"""
    title: str = Field(description="Episode title shown on cover")
    headline: str = Field(description="Large centered headline (sharp claim or question)")
    question: str = Field(description="Question-style subtitle below headline")
    image_prompt: str = Field(description="AI image generation prompt for cover background")
    background_time: float = Field(default=0.0, description="Fallback: timestamp in source video for background frame")


class CommentaryScript(BaseModel):
    """Complete commentary script with all segments"""
    title: str = Field(description="Video title")
    content_start: float = Field(description="Start of usable story range (skip intro/opening)")
    content_end: float = Field(description="End of usable story range (skip credits/preview)")
    target_duration: float = Field(default=300.0, description="Target duration in seconds")
    progress_segments: List[str] = Field(description="4-6 story stage labels for top progress bar")
    cover: CommentaryCover = Field(description="Cover configuration")
    chunks: List[CommentaryChunk] = Field(description="Commentary chunks in order")


class CommentaryConfig(BaseModel):
    """User-provided configuration for commentary generation"""
    source_video: str = Field(description="Absolute path to source video")
    target_duration: int = Field(default=300, ge=60, le=900, description="Target commentary duration in seconds (1-15 min)")
    tts_voice: str = Field(default="zh-CN-YunxiNeural", description="Edge TTS voice ID")
    tts_rate: str = Field(default="+18%", description="Edge TTS rate modifier")
    narration_slot_ratio: float = Field(default=0.82, ge=0.55, le=1.0, description="Ratio of chunk duration for narration")
    bgm_path: Optional[str] = Field(default=None, description="Optional BGM file path")
    content_start: Optional[float] = Field(default=None, description="Manual override for story start (auto-detected if None)")
    content_end: Optional[float] = Field(default=None, description="Manual override for story end (auto-detected if None)")
    cover_headline: Optional[str] = Field(default=None, description="Manual override for cover headline")
    cover_question: Optional[str] = Field(default=None, description="Manual override for cover question")
    mask_subtitles: bool = Field(default=False, description="Blur bottom area to mask original hard subtitles")
    mask_subtitle_height_ratio: float = Field(default=0.10, ge=0.05, le=0.40, description="Height ratio of bottom area to mask (0.05-0.40)")
    segment_count: int = Field(default=1, ge=1, le=10, description="Number of commentary segments to generate (total duration is split evenly)")
    keep_original_audio: bool = Field(default=True, description="Keep original video audio as background")
    original_audio_volume: float = Field(default=0.2, ge=0.0, le=1.0, description="Volume of original audio (0.0-1.0)")
