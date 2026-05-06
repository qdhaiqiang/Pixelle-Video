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
Image Analysis Service - ComfyUI Workflow-based implementation

Uses Florence-2 or other vision models to analyze images and generate descriptions.
"""

import base64
import mimetypes
from typing import Optional, Literal
from pathlib import Path

from comfykit import ComfyKit
from loguru import logger
from openai import BadRequestError

from pixelle_video.services.comfy_base_service import ComfyBaseService


class ImageAnalysisService(ComfyBaseService):
    """
    Image analysis service - Workflow-based
    
    Uses ComfyKit to execute image analysis workflows (e.g., Florence-2, BLIP, etc.).
    Returns detailed textual descriptions of images.
    
    Convention: workflows follow {source}/analyse_image.json pattern
    - runninghub/analyse_image.json (default, cloud-based)
    - selfhost/analyse_image.json (local ComfyUI)
    
    Usage:
        # Use default (runninghub cloud)
        description = await pixelle_video.image_analysis("path/to/image.jpg")
        
        # Use local ComfyUI
        description = await pixelle_video.image_analysis(
            "path/to/image.jpg",
            source="selfhost"
        )
        
        # List available workflows
        workflows = pixelle_video.image_analysis.list_workflows()
    """
    
    WORKFLOW_PREFIX = "analyse_"
    WORKFLOWS_DIR = "workflows"
    
    def __init__(self, config: dict, core=None):
        """
        Initialize image analysis service
        
        Args:
            config: Full application config dict
            core: PixelleVideoCore instance (for accessing shared ComfyKit)
        """
        super().__init__(config, service_name="image_analysis", core=core)
    
    async def __call__(
        self,
        image_path: str,
        # Workflow source selection
        source: Literal['runninghub', 'selfhost', 'llm', 'dashscope'] = 'runninghub',
        workflow: Optional[str] = None,
        # ComfyUI connection (optional overrides)
        comfyui_url: Optional[str] = None,
        runninghub_api_key: Optional[str] = None,
        # Additional workflow parameters
        **params
    ) -> str:
        """
        Analyze an image using workflow
        
        Args:
            image_path: Path to the image file (local or URL)
            source: Workflow source - 'runninghub' (cloud, default) or 'selfhost' (local ComfyUI)
            workflow: Workflow filename (optional, overrides source-based resolution)
            comfyui_url: ComfyUI URL (optional, overrides config)
            runninghub_api_key: RunningHub API key (optional, overrides config)
            **params: Additional workflow parameters
        
        Returns:
            str: Text description of the image
        
        Examples:
            # Simplest: use default (runninghub cloud)
            description = await pixelle_video.image_analysis("temp/06.JPG")
            
            # Use local ComfyUI
            description = await pixelle_video.image_analysis(
                "temp/06.JPG",
                source="selfhost"
            )
            
            # Use specific workflow (bypass source-based resolution)
            description = await pixelle_video.image_analysis(
                "temp/06.JPG",
                workflow="selfhost/custom_analysis.json"
            )
        """
        from pixelle_video.utils.workflow_util import resolve_workflow_path
        
        # 1. Validate image path
        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if source in ("llm", "dashscope"):
            return await self._analyze_with_llm(image_path_obj, params.get("prompt"))
        
        # 2. Resolve workflow path using convention
        if workflow is None:
            # Use standardized naming: {source}/analyse_image.json
            workflow = resolve_workflow_path("analyse_image", source)
            logger.info(f"Using {source} workflow: {workflow}")
        
        # 2. Resolve workflow (returns structured info)
        workflow_info = self._resolve_workflow(workflow=workflow)
        
        # 3. Build workflow parameters
        workflow_params = {
            "image": str(image_path)  # Pass image path to workflow
        }
        
        # Add any additional parameters
        workflow_params.update(params)
        
        logger.debug(f"Workflow parameters: {workflow_params}")
        
        # 4. Execute workflow using shared ComfyKit instance from core
        try:
            # Get shared ComfyKit instance (lazy initialization + config hot-reload)
            kit = await self.core._get_or_create_comfykit()
            
            # Determine what to pass to ComfyKit based on source
            if workflow_info["source"] == "runninghub" and "workflow_id" in workflow_info:
                # RunningHub: pass workflow_id
                workflow_input = workflow_info["workflow_id"]
                logger.info(f"Executing RunningHub workflow: {workflow_input}")
            else:
                # Selfhost: pass file path
                workflow_input = workflow_info["path"]
                logger.info(f"Executing selfhost workflow: {workflow_input}")
            
            result = await kit.execute(workflow_input, workflow_params)
            
            # 5. Extract description from result
            if result.status != "completed":
                error_msg = result.msg or "Unknown error"
                logger.error(f"Image analysis failed: {error_msg}")
                raise Exception(f"Image analysis failed: {error_msg}")
            
            # Extract text description from result (format varies by source)
            description = None
            
            # Try format 1: Selfhost outputs (direct text in outputs)
            # Format: {'6': {'text': ['description text']}}
            if result.outputs:
                for node_id, node_output in result.outputs.items():
                    if 'text' in node_output:
                        text_list = node_output['text']
                        if text_list and len(text_list) > 0:
                            description = text_list[0]
                            break
            
            # Try format 2: RunningHub raw_data (text file URL)
            # Format: {'raw_data': [{'fileUrl': 'https://...txt', 'fileType': 'txt', ...}]}
            if not description and result.outputs and 'raw_data' in result.outputs:
                raw_data = result.outputs['raw_data']
                if raw_data and len(raw_data) > 0:
                    # Find text file entry
                    for item in raw_data:
                        if item.get('fileType') == 'txt' and 'fileUrl' in item:
                            # Download text content from URL
                            import aiohttp
                            async with aiohttp.ClientSession() as session:
                                async with session.get(item['fileUrl']) as resp:
                                    if resp.status == 200:
                                        description = await resp.text()
                                        description = description.strip()
                                        break
            
            if not description:
                logger.error(f"No text found in outputs: {result.outputs}")
                raise Exception("No description generated")
            
            logger.info(f"✅ Image analyzed: {description[:100]}...")
            
            return description
        
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            raise

    async def _analyze_with_llm(self, image_path: Path, prompt: Optional[str] = None) -> str:
        """Analyze an image with the configured OpenAI-compatible vision LLM."""
        if not self.core or not self.core.llm:
            raise RuntimeError("LLM service is not initialized")

        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
        image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        user_prompt = prompt or (
            "请详细描述这张图片中的主体、场景、文字、商品、颜色、构图和可用于短视频创作的卖点。"
            "输出一段简洁中文描述，不要使用 Markdown。"
        )

        client = self.core.llm._create_client()
        model = self.core.llm._get_config_value("model")
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_data}"
                                },
                            },
                        ],
                    }
                ],
                temperature=0.2,
                max_tokens=800,
            )
        except BadRequestError as e:
            if self._looks_like_text_only_model_error(str(e)):
                logger.warning(
                    "Configured LLM rejected image_url content; falling back to filename-only asset analysis"
                )
                return await self._fallback_text_only_analysis(image_path)
            raise

        description = response.choices[0].message.content or ""
        if not description.strip():
            raise RuntimeError("No description generated by LLM vision analysis")
        logger.info(f"✅ Image analyzed by LLM: {description[:100]}...")
        return description.strip()

    def _looks_like_text_only_model_error(self, error_message: str) -> bool:
        lowered = error_message.lower()
        return (
            "unknown variant image_url" in lowered
            or "expected text" in lowered
            or "image_url" in lowered and "invalid_request_error" in lowered
        )

    async def _fallback_text_only_analysis(self, image_path: Path) -> str:
        prompt = (
            "当前配置的大模型不支持图片视觉输入，无法直接查看图片内容。"
            f"用户上传了一个图片素材，文件名是：{image_path.name}。"
            "请基于文件名生成一段保守、中性的短视频素材描述，明确说明这是基于文件名推断，"
            "不要编造具体画面细节。输出一段中文，不要使用 Markdown。"
        )
        description = await self.core.llm(prompt=prompt, temperature=0.2, max_tokens=300)
        return description.strip()
