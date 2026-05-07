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
Commentary video generation API schemas
"""

from typing import Optional
from pydantic import BaseModel, Field


class CommentaryGenerateRequest(BaseModel):
    """Commentary video generation request"""

    source_video: str = Field(..., description="Absolute path to source video file (local)")
    target_duration: int = Field(300, ge=60, le=900, description="Target commentary duration in seconds (1-15 min)")
    segment_count: int = Field(1, ge=1, le=10, description="Number of segments to split the video into")

    # TTS parameters (same as standard pipeline)
    tts_inference_mode: str = Field("local", description="TTS inference mode: 'local' or 'comfyui'")
    tts_voice: Optional[str] = Field("zh-CN-YunxiNeural", description="Edge TTS voice ID (local mode)")
    tts_speed: float = Field(1.2, ge=0.5, le=2.0, description="Speech speed multiplier (local mode)")
    tts_rate: str = Field("+18%", description="Edge TTS rate modifier (overrides tts_speed)")
    tts_workflow: Optional[str] = Field(None, description="TTS workflow key (comfyui mode)")
    ref_audio: Optional[str] = Field(None, description="Reference audio for voice cloning")

    # Commentary-specific
    narration_slot_ratio: float = Field(0.82, ge=0.55, le=1.0, description="Narration duration ratio per chunk")
    content_start: Optional[float] = Field(None, description="Manual story start (auto-detected if None)")
    content_end: Optional[float] = Field(None, description="Manual story end (auto-detected if None)")
    cover_headline: Optional[str] = Field(None, description="Manual cover headline override")
    cover_question: Optional[str] = Field(None, description="Manual cover question override")
    mask_subtitles: bool = Field(False, description="Blur bottom area to mask original hard subtitles")
    keep_original_audio: bool = Field(True, description="Keep original video audio as background")
    original_audio_volume: float = Field(0.2, ge=0.0, le=1.0, description="Volume of original audio")

    # BGM
    bgm_path: Optional[str] = Field(None, description="Background music file path")

    class Config:
        json_schema_extra = {
            "example": {
                "source_video": "/path/to/movie.mp4",
                "target_duration": 300,
                "segment_count": 1,
                "tts_inference_mode": "local",
                "tts_voice": "zh-CN-YunxiNeural",
                "tts_rate": "+18%",
                "narration_slot_ratio": 0.82,
            }
        }


class CommentaryGenerateResponse(BaseModel):
    """Commentary video generation response (synchronous)"""
    success: bool = True
    message: str = "Success"
    video_url: str = Field(..., description="URL to access generated video (primary)")
    video_urls: list[str] = Field(default_factory=list, description="All generated video URLs (multi-segment)")
    duration: float = Field(..., description="Video duration in seconds")
    file_size: int = Field(..., description="File size in bytes")
    n_chunks: int = Field(..., description="Number of commentary chunks")


class CommentaryGenerateAsyncResponse(BaseModel):
    """Commentary video generation async response"""
    success: bool = True
    message: str = "Task created successfully"
    task_id: str = Field(..., description="Task ID for tracking progress")
