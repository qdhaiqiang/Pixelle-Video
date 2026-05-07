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
Commentary video generation endpoints

Supports both synchronous and asynchronous commentary video generation.
"""

import os
from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from api.dependencies import PixelleVideoDep
from api.schemas.commentary import (
    CommentaryGenerateRequest,
    CommentaryGenerateResponse,
    CommentaryGenerateAsyncResponse,
)
from api.tasks import task_manager, TaskType
from api.routers.video import path_to_url

router = APIRouter(prefix="/video", tags=["Video Commentary"])


@router.post("/commentary/generate/sync", response_model=CommentaryGenerateResponse)
async def generate_commentary_sync(
    request_body: CommentaryGenerateRequest,
    pixelle_video: PixelleVideoDep,
    request: Request,
):
    """
    Generate commentary video synchronously

    Converts a source video into a commentary video with AI-generated script,
    TTS narration, captions, progress bar, and cover intro.

    **Note**: source_video must be an absolute local path. The server checks
    file existence but does not upload the file.
    """
    try:
        logger.info(f"Sync commentary generation: {request_body.source_video}")

        # Build pipeline parameters
        pipeline_params = {
            "source_video": request_body.source_video,
            "target_duration": request_body.target_duration,
            "tts_voice": request_body.tts_voice,
            "tts_rate": request_body.tts_rate,
            "narration_slot_ratio": request_body.narration_slot_ratio,
            "bgm_path": request_body.bgm_path,
            "content_start": request_body.content_start,
            "content_end": request_body.content_end,
            "cover_headline": request_body.cover_headline,
            "cover_question": request_body.cover_question,
            "mask_subtitles": request_body.mask_subtitles,
            "keep_original_audio": request_body.keep_original_audio,
            "original_audio_volume": request_body.original_audio_volume,
            "segment_count": request_body.segment_count,
        }

        result = await pixelle_video.pipelines["commentary"](
            text="",  # Not used
            **pipeline_params,
        )

        all_paths = [result.video_path] + list(getattr(result, "additional_video_paths", []) or [])
        file_size = sum(os.path.getsize(p) for p in all_paths if os.path.exists(p))
        video_urls = [path_to_url(request, p) for p in all_paths]

        return CommentaryGenerateResponse(
            video_url=video_urls[0],
            video_urls=video_urls,
            duration=result.duration,
            file_size=file_size,
            n_chunks=len(result.storyboard.frames) if result.storyboard else 0,
        )

    except Exception as e:
        logger.error(f"Sync commentary generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/commentary/generate/async", response_model=CommentaryGenerateAsyncResponse)
async def generate_commentary_async(
    request_body: CommentaryGenerateRequest,
    pixelle_video: PixelleVideoDep,
    request: Request,
):
    """
    Generate commentary video asynchronously

    Creates a background task for commentary video generation.
    Returns immediately with a task_id for tracking progress.
    """
    try:
        logger.info(f"Async commentary generation: {request_body.source_video}")

        task = task_manager.create_task(
            task_type=TaskType.VIDEO_GENERATION,
            request_params=request_body.model_dump(),
        )

        async def execute_commentary_generation():
            pipeline_params = {
                "source_video": request_body.source_video,
                "target_duration": request_body.target_duration,
                "tts_voice": request_body.tts_voice,
                "tts_rate": request_body.tts_rate,
                "narration_slot_ratio": request_body.narration_slot_ratio,
                "bgm_path": request_body.bgm_path,
                "content_start": request_body.content_start,
                "content_end": request_body.content_end,
                "cover_headline": request_body.cover_headline,
                "cover_question": request_body.cover_question,
                "mask_subtitles": request_body.mask_subtitles,
                "segment_count": request_body.segment_count,
            }

            result = await pixelle_video.pipelines["commentary"](
                text="",
                **pipeline_params,
            )

            all_paths = [result.video_path] + list(getattr(result, "additional_video_paths", []) or [])
            file_size = sum(os.path.getsize(p) for p in all_paths if os.path.exists(p))
            video_urls = [path_to_url(request, p) for p in all_paths]

            return {
                "video_url": video_urls[0],
                "video_urls": video_urls,
                "duration": result.duration,
                "file_size": file_size,
                "n_chunks": len(result.storyboard.frames) if result.storyboard else 0,
            }

        await task_manager.execute_task(
            task_id=task.task_id,
            coro_func=execute_commentary_generation,
        )

        return CommentaryGenerateAsyncResponse(task_id=task.task_id)

    except Exception as e:
        logger.error(f"Async commentary generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
