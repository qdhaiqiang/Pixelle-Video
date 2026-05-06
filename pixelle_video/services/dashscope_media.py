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
DashScope / Alibaba Cloud Bailian media generation client.

Uses the HTTP async task protocol used by Tongyi Wanxiang image and video
generation APIs: create task, poll task status, then return the temporary URL.
"""

import asyncio
import base64
import math
import mimetypes
import os
from pathlib import Path
from typing import Any, Optional

import httpx
from loguru import logger

from pixelle_video.models.media import MediaResult


REGION_BASE_URLS = {
    "beijing": "https://dashscope.aliyuncs.com/api/v1",
    "cn-beijing": "https://dashscope.aliyuncs.com/api/v1",
    "singapore": "https://dashscope-intl.aliyuncs.com/api/v1",
    "intl": "https://dashscope-intl.aliyuncs.com/api/v1",
    "us": "https://dashscope-us.aliyuncs.com/api/v1",
    "virginia": "https://dashscope-us.aliyuncs.com/api/v1",
    "us-virginia": "https://dashscope-us.aliyuncs.com/api/v1",
}


class DashScopeMediaClient:
    """Minimal async HTTP client for DashScope media generation."""

    def __init__(self, config: dict):
        dashscope_config = config.get("dashscope", {})
        self.api_key = (
            dashscope_config.get("api_key")
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("DASHSCOPE_BAILIAN_API_KEY")
        )
        self.workspace = dashscope_config.get("workspace") or os.getenv("DASHSCOPE_WORKSPACE")
        self.region = (dashscope_config.get("region") or os.getenv("DASHSCOPE_REGION") or "beijing").lower()
        self.base_url = (
            dashscope_config.get("base_url")
            or os.getenv("DASHSCOPE_BASE_URL")
            or REGION_BASE_URLS.get(self.region)
            or self.region
        ).rstrip("/")
        self.poll_interval = float(dashscope_config.get("poll_interval", 10))
        self.timeout = float(dashscope_config.get("timeout", 600))

    async def generate(
        self,
        prompt: str,
        workflow_info: dict[str, Any],
        media_type: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        duration: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        **params: Any,
    ) -> MediaResult:
        if not self.api_key:
            raise ValueError(
                "DashScope API key is not configured. Set comfyui.dashscope.api_key "
                "in config.yaml or export DASHSCOPE_API_KEY."
            )

        if media_type == "video":
            return await self._generate_video(
                prompt=prompt,
                workflow_info=workflow_info,
                width=width,
                height=height,
                duration=duration,
                negative_prompt=negative_prompt,
                seed=seed,
                **params,
            )

        return await self._generate_image(
            prompt=prompt,
            workflow_info=workflow_info,
            width=width,
            height=height,
            negative_prompt=negative_prompt,
            seed=seed,
            **params,
        )

    async def _generate_image(
        self,
        prompt: str,
        workflow_info: dict[str, Any],
        width: Optional[int],
        height: Optional[int],
        negative_prompt: Optional[str],
        seed: Optional[int],
        **params: Any,
    ) -> MediaResult:
        model = workflow_info.get("model", "wan2.2-t2i-flash")
        size = params.get("size") or workflow_info.get("size") or self._image_size(width, height)
        prompt_extend = params.get("prompt_extend", workflow_info.get("prompt_extend", True))
        watermark = params.get("watermark", workflow_info.get("watermark", False))

        input_data = {"prompt": prompt}
        if negative_prompt:
            input_data["negative_prompt"] = negative_prompt

        parameters: dict[str, Any] = {
            "size": size,
            "n": int(params.get("n", workflow_info.get("n", 1))),
            "prompt_extend": bool(prompt_extend),
            "watermark": bool(watermark),
        }
        if seed is not None:
            parameters["seed"] = seed

        payload = {
            "model": model,
            "input": input_data,
            "parameters": parameters,
        }
        task = await self._create_task("/services/aigc/text2image/image-synthesis", payload)
        output, _usage = await self._poll_task(task["task_id"])

        results = output.get("results") or []
        image_url = next((item.get("url") for item in results if item.get("url")), None)
        if not image_url:
            image_url = self._extract_choice_image(output)
        if not image_url:
            raise RuntimeError(f"DashScope image task succeeded but returned no image URL: {output}")

        logger.info(f"✅ Generated image with DashScope: {image_url}")
        return MediaResult(media_type="image", url=image_url)

    async def _generate_video(
        self,
        prompt: str,
        workflow_info: dict[str, Any],
        width: Optional[int],
        height: Optional[int],
        duration: Optional[float],
        negative_prompt: Optional[str],
        seed: Optional[int],
        **params: Any,
    ) -> MediaResult:
        model = workflow_info.get("model", "wan2.6-t2v")
        image_input = params.get("image") or params.get("img_url") or params.get("first_frame")
        size = params.get("size") or workflow_info.get("size") or self._video_size(width, height)
        resolution = params.get("resolution") or workflow_info.get("resolution")
        requested_duration = params.get("duration") or duration or workflow_info.get("duration")
        prompt_extend = params.get("prompt_extend", workflow_info.get("prompt_extend", True))
        watermark = params.get("watermark", workflow_info.get("watermark", False))

        input_data = {"prompt": prompt}
        if image_input:
            input_data["img_url"] = self._image_input_to_url(image_input)
        if negative_prompt:
            input_data["negative_prompt"] = negative_prompt

        parameters: dict[str, Any] = {
            "prompt_extend": bool(prompt_extend),
            "watermark": bool(watermark),
        }
        if image_input:
            parameters["resolution"] = resolution or "720P"
        else:
            parameters["size"] = size
        if requested_duration is not None and self._supports_duration(model):
            parameters["duration"] = self._video_duration(requested_duration)
        if seed is not None:
            parameters["seed"] = seed
        if workflow_info.get("shot_type"):
            parameters["shot_type"] = workflow_info["shot_type"]

        payload = {
            "model": model,
            "input": input_data,
            "parameters": parameters,
        }
        task = await self._create_task("/services/aigc/video-generation/video-synthesis", payload)
        output, usage = await self._poll_task(task["task_id"], poll_interval=max(self.poll_interval, 15))

        video_url = output.get("video_url")
        if not video_url:
            raise RuntimeError(f"DashScope video task succeeded but returned no video URL: {output}")

        output_duration = (
            usage.get("output_video_duration")
            or usage.get("duration")
            or parameters.get("duration")
        )
        logger.info(f"✅ Generated video with DashScope: {video_url}")
        return MediaResult(media_type="video", url=video_url, duration=output_duration)

    async def _create_task(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = self._headers(async_task=True)
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(f"{self.base_url}{path}", headers=headers, json=payload)
        data = self._parse_response(response)
        output = data.get("output") or {}
        task_id = output.get("task_id")
        if not task_id:
            raise RuntimeError(f"DashScope task creation returned no task_id: {data}")
        logger.info(f"Submitted DashScope task: {task_id}")
        return output

    async def _poll_task(
        self,
        task_id: str,
        poll_interval: Optional[float] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        interval = poll_interval or self.poll_interval
        deadline = asyncio.get_running_loop().time() + self.timeout

        async with httpx.AsyncClient(timeout=60) as client:
            while True:
                if asyncio.get_running_loop().time() > deadline:
                    raise TimeoutError(f"DashScope task timed out after {self.timeout:.0f}s: {task_id}")

                response = await client.get(f"{self.base_url}/tasks/{task_id}", headers=self._headers())
                data = self._parse_response(response)
                output = data.get("output") or {}
                status = output.get("task_status")
                logger.info(f"DashScope task {task_id} status: {status}")

                if status == "SUCCEEDED":
                    return output, data.get("usage") or {}
                if status in {"FAILED", "CANCELED", "UNKNOWN"}:
                    code = output.get("code") or data.get("code") or status
                    message = output.get("message") or data.get("message") or "Unknown DashScope error"
                    raise RuntimeError(f"DashScope task failed: {code}: {message}")

                await asyncio.sleep(interval)

    def _headers(self, async_task: bool = False) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if async_task:
            headers["X-DashScope-Async"] = "enable"
        if self.workspace:
            headers["X-DashScope-WorkSpace"] = self.workspace
        return headers

    def _parse_response(self, response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(f"DashScope returned non-JSON response: {response.text}") from exc

        if response.status_code >= 400 or data.get("code"):
            code = data.get("code") or response.status_code
            message = data.get("message") or response.text
            raise RuntimeError(f"DashScope API error: {code}: {message}")
        return data

    def _extract_choice_image(self, output: dict[str, Any]) -> Optional[str]:
        for choice in output.get("choices") or []:
            message = choice.get("message") or {}
            for item in message.get("content") or []:
                if item.get("image"):
                    return item["image"]
        return None

    def _image_size(self, width: Optional[int], height: Optional[int]) -> str:
        if not width or not height:
            return "1024*1024"
        width = min(max(int(width), 512), 1440)
        height = min(max(int(height), 512), 1440)
        return f"{width}*{height}"

    def _video_size(self, width: Optional[int], height: Optional[int]) -> str:
        if not width or not height:
            return "1280*720"

        ratio = width / height
        candidates = [
            (1280, 720),
            (720, 1280),
            (960, 960),
            (1088, 832),
            (832, 1088),
            (832, 480),
            (480, 832),
            (624, 624),
        ]
        best = min(candidates, key=lambda item: abs((item[0] / item[1]) - ratio))
        return f"{best[0]}*{best[1]}"

    def _video_duration(self, duration: float) -> int:
        return min(max(math.ceil(float(duration)), 2), 15)

    def _supports_duration(self, model: str) -> bool:
        return not (
            model.startswith("wanx2.1-t2v")
            or model.startswith("wan2.2-t2v")
            or model.startswith("wanx2.1-i2v")
            or model.startswith("wan2.2-i2v")
        )

    def _image_input_to_url(self, image_input: str) -> str:
        if image_input.startswith(("http://", "https://", "data:image/")):
            return image_input

        image_path = Path(image_input)
        if not image_path.exists():
            raise FileNotFoundError(f"DashScope image-to-video input not found: {image_input}")

        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
        data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{data}"
