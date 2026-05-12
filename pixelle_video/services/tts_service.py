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
TTS (Text-to-Speech) Service - Supports both local and ComfyUI inference
"""

import os
import subprocess
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any

from comfykit import ComfyKit
from loguru import logger

from pixelle_video.services.comfy_base_service import ComfyBaseService
from pixelle_video.services.cosyvoice_installer import (
    REPO_DIR,
    RUNNER_PATH,
    get_cosyvoice_python,
    get_cosyvoice_status,
)
from pixelle_video.utils.tts_util import edge_tts
from pixelle_video.tts_voices import speed_to_rate


class TTSService(ComfyBaseService):
    """
    TTS (Text-to-Speech) service - Workflow-based
    
    Uses ComfyKit to execute TTS workflows.
    
    Usage:
        # Use default workflow
        audio_path = await pixelle_video.tts(text="Hello, world!")
        
        # Use specific workflow
        audio_path = await pixelle_video.tts(
            text="你好，世界！",
            workflow="tts_edge.json"
        )
        
        # List available workflows
        workflows = pixelle_video.tts.list_workflows()
    """
    
    WORKFLOW_PREFIX = "tts_"
    DEFAULT_WORKFLOW = None  # No hardcoded default, must be configured
    WORKFLOWS_DIR = "workflows"
    COSYVOICE_INSTALL_HINT = (
        "CosyVoice local model is not installed. Select CosyVoice Local Model in the UI "
        "and click Install CosyVoice first."
    )
    COSYVOICE_MODEL_BY_MODE = {
        "sft": "iic/CosyVoice-300M-SFT",
        "instruct": "iic/CosyVoice-300M-Instruct",
        "zero_shot": "iic/CosyVoice-300M",
    }
    
    def __init__(self, config: dict, core=None):
        """
        Initialize TTS service
        
        Args:
            config: Full application config dict
            core: PixelleVideoCore instance (for accessing shared ComfyKit)
        """
        super().__init__(config, service_name="tts", core=core)
    
    
    async def __call__(
        self,
        text: str,
        workflow: Optional[str] = None,
        # ComfyUI connection (optional overrides)
        comfyui_url: Optional[str] = None,
        runninghub_api_key: Optional[str] = None,
        # TTS parameters
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        # Inference mode override
        inference_mode: Optional[str] = None,
        # Output path
        output_path: Optional[str] = None,
        **params
    ) -> str:
        """
        Generate speech using local Edge TTS or ComfyUI workflow
        
        Args:
            text: Text to convert to speech
            workflow: Workflow filename (for ComfyUI mode, default: from config)
            comfyui_url: ComfyUI URL (optional, overrides config)
            runninghub_api_key: RunningHub API key (optional, overrides config)
            voice: Voice ID (for local mode: Edge TTS voice ID; for ComfyUI: workflow-specific)
            speed: Speech speed multiplier (1.0 = normal, >1.0 = faster, <1.0 = slower)
            inference_mode: Override inference mode ("local" or "comfyui", default: from config)
            output_path: Custom output path (auto-generated if None)
            **params: Additional workflow parameters
        
        Returns:
            Generated audio file path
        
        Examples:
            # Local inference (Edge TTS)
            audio_path = await pixelle_video.tts(
                text="Hello, world!",
                inference_mode="local",
                voice="zh-CN-YunjianNeural",
                speed=1.2
            )
            
            # ComfyUI inference
            audio_path = await pixelle_video.tts(
                text="你好，世界！",
                inference_mode="comfyui",
                workflow="runninghub/tts_edge.json"
            )
        """
        # Determine inference mode (param > config)
        mode = inference_mode or self.config.get("inference_mode", "local")
        if mode == "cosyvoice":
            return await self._call_cosyvoice_tts(
                text=text,
                voice=voice,
                speed=speed,
                output_path=output_path,
                **params,
            )
        
        # Route to appropriate implementation
        if mode == "local":
            return await self._call_local_tts(
                text=text,
                voice=voice,
                speed=speed,
                output_path=output_path
            )
        else:  # comfyui
            # 1. Resolve workflow (returns structured info)
            workflow_info = self._resolve_workflow(workflow=workflow)
            
            # 2. Execute ComfyUI workflow
            return await self._call_comfyui_workflow(
                workflow_info=workflow_info,
                text=text,
                comfyui_url=comfyui_url,
                runninghub_api_key=runninghub_api_key,
                voice=voice,
                speed=speed,
                output_path=output_path,
                **params
            )

    def list_workflows(self) -> List[Dict[str, Any]]:
        """List TTS workflows with engine labels for the UI."""
        workflows = super().list_workflows()
        for workflow in workflows:
            engine = self._detect_workflow_engine(workflow)
            workflow["engine"] = engine
            if engine == "cosyvoice":
                workflow["display_name"] = f"CosyVoice - {workflow['source'].title()} ({workflow['name']})"
            elif engine == "index":
                workflow["display_name"] = f"IndexTTS - {workflow['source'].title()} ({workflow['name']})"
            elif engine == "spark":
                workflow["display_name"] = f"SparkTTS - {workflow['source'].title()} ({workflow['name']})"
            elif engine == "edge":
                workflow["display_name"] = f"Edge TTS - {workflow['source'].title()} ({workflow['name']})"
        return sorted(workflows, key=lambda wf: (0 if wf.get("engine") == "cosyvoice" else 1, wf["key"]))

    def _find_cosyvoice_workflow(self) -> str:
        """Return the first configured CosyVoice workflow key, preferring local selfhost."""
        workflows = self.list_workflows()
        for workflow in workflows:
            if workflow.get("engine") == "cosyvoice" and workflow.get("source") == "selfhost":
                return workflow["key"]
        for workflow in workflows:
            if workflow.get("engine") == "cosyvoice":
                return workflow["key"]
        raise ValueError(self.COSYVOICE_INSTALL_HINT)

    @staticmethod
    def _detect_workflow_engine(workflow: Dict[str, Any]) -> str:
        text = " ".join(
            str(workflow.get(key, ""))
            for key in ("name", "display_name", "key", "workflow_id", "engine", "provider")
        ).lower()
        if "cosy" in text or "cosyvoice" in text:
            return "cosyvoice"
        if "index" in text:
            return "index"
        if "spark" in text:
            return "spark"
        if "edge" in text:
            return "edge"
        return "workflow"
    
    async def _call_local_tts(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate speech using local Edge TTS
        
        Args:
            text: Text to convert to speech
            voice: Edge TTS voice ID (default: from config)
            speed: Speech speed multiplier (default: from config)
            output_path: Custom output path (auto-generated if None)
        
        Returns:
            Generated audio file path
        """
        # Get config defaults
        local_config = self.config.get("local", {})
        
        # Determine voice and speed (param > config)
        final_voice = voice or local_config.get("voice", "zh-CN-YunjianNeural")
        final_speed = speed if speed is not None else local_config.get("speed", 1.2)
        
        # Convert speed to rate parameter
        rate = speed_to_rate(final_speed)
        
        logger.info(f"🎙️  Using local Edge TTS: voice={final_voice}, speed={final_speed}x (rate={rate})")
        
        # Generate output path if not provided
        if not output_path:
            # Generate unique filename
            unique_id = uuid.uuid4().hex
            output_path = f"output/{unique_id}.mp3"
            
            # Ensure output directory exists
            Path("output").mkdir(parents=True, exist_ok=True)
        
        # Call Edge TTS
        try:
            audio_bytes = await edge_tts(
                text=text,
                voice=final_voice,
                rate=rate,
                output_path=output_path
            )
            
            logger.info(f"✅ Generated audio (local Edge TTS): {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Local TTS generation error: {e}")
            raise

    async def _call_cosyvoice_tts(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        output_path: Optional[str] = None,
        **params,
    ) -> str:
        status = get_cosyvoice_status()
        if not status.installed:
            raise RuntimeError(status.message + ". " + self.COSYVOICE_INSTALL_HINT)

        if not output_path:
            unique_id = uuid.uuid4().hex
            output_path = f"output/{unique_id}.mp3"
            Path("output").mkdir(parents=True, exist_ok=True)

        python_path = get_cosyvoice_python()
        cosyvoice_config = self.config.get("cosyvoice", {})
        model = (
            params.get("model")
            or params.get("cosyvoice_model")
            or cosyvoice_config.get("model")
            or self.COSYVOICE_MODEL_BY_MODE["sft"]
        )
        mode = params.get("mode") or params.get("cosyvoice_mode") or cosyvoice_config.get("mode") or "sft"
        if mode == "instruct" and params.get("allow_instruct", True) is False:
            logger.warning("CosyVoice instruct mode is disabled for this generation; falling back to SFT to avoid control text leakage.")
            mode = "sft"
            model = self.COSYVOICE_MODEL_BY_MODE["sft"]
        self._validate_cosyvoice_mode_model(mode, str(model))
        speaker = params.get("speaker") or voice or params.get("voice") or cosyvoice_config.get("speaker") or "中文女"
        final_speed = speed if speed is not None else params.get("tts_speed", 1.0)
        instruct = params.get("instruct") or cosyvoice_config.get("instruct") or ""
        if mode != "instruct":
            instruct = ""
        prompt_text = params.get("prompt_text") or cosyvoice_config.get("prompt_text") or ""
        prompt_audio = params.get("prompt_audio") or params.get("ref_audio") or cosyvoice_config.get("prompt_audio") or ""

        cmd = [
            str(python_path),
            str(RUNNER_PATH),
            "--repo-dir",
            str(REPO_DIR),
            "--text",
            text,
            "--output",
            output_path,
            "--model",
            str(model),
            "--mode",
            str(mode),
            "--speaker",
            str(speaker),
            "--speed",
            str(final_speed),
            "--instruct",
            str(instruct),
            "--prompt-text",
            str(prompt_text),
            "--prompt-audio",
            str(prompt_audio),
        ]
        logger.info(f"🎙️  Using local CosyVoice: mode={mode}, model={model}, speaker={speaker}, speed={final_speed}x")
        result = subprocess.run(cmd, text=True, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                "CosyVoice generation failed:\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT:\n{result.stdout.strip()}\n"
                f"STDERR:\n{result.stderr.strip()}"
            )
        logger.info(f"✅ Generated audio (local CosyVoice): {output_path}")
        return output_path

    @staticmethod
    def _validate_cosyvoice_mode_model(mode: str, model: str) -> None:
        model_lower = model.lower()
        if mode == "instruct" and "instruct" not in model_lower:
            raise ValueError(
                "CosyVoice 配置错误：指令语气必须使用 Instruct 模型，例如 iic/CosyVoice-300M-Instruct。"
                "当前模型不是 Instruct，已停止生成，避免把语气指令混入视频。"
            )
        if mode == "sft" and "sft" not in model_lower:
            raise ValueError(
                "CosyVoice 配置错误：预置音色必须使用 SFT 模型，例如 iic/CosyVoice-300M-SFT。"
            )
        if mode == "zero_shot" and ("sft" in model_lower or "instruct" in model_lower):
            raise ValueError(
                "CosyVoice 配置错误：参考音频复刻必须使用基础模型，不能使用 SFT/Instruct 模型。"
            )
    
    async def _call_comfyui_workflow(
        self,
        workflow_info: dict,
        text: str,
        comfyui_url: Optional[str] = None,
        runninghub_api_key: Optional[str] = None,
        voice: Optional[str] = None,
        speed: float = 1.0,
        output_path: Optional[str] = None,
        **params
    ) -> str:
        """
        Generate speech using ComfyUI workflow
        
        Args:
            workflow_info: Workflow info dict from _resolve_workflow()
            text: Text to convert to speech
            comfyui_url: ComfyUI URL
            runninghub_api_key: RunningHub API key
            voice: Voice ID (workflow-specific)
            speed: Speech speed multiplier (workflow-specific)
            output_path: Custom output path (downloads if URL returned)
            **params: Additional workflow parameters
        
        Returns:
            Generated audio file path (local if output_path provided, otherwise URL)
        """
        logger.info(f"🎙️  Using workflow: {workflow_info['key']}")
        
        # 1. Build workflow parameters (ComfyKit config is now managed by core)
        workflow_params = {"text": text}
        
        # Add optional TTS parameters (only if explicitly provided and not None)
        if voice is not None:
            workflow_params["voice"] = voice
        if speed is not None and speed != 1.0:
            workflow_params["speed"] = speed
        
        # Add any additional parameters
        workflow_params.update(params)
        
        logger.debug(f"Workflow parameters: {workflow_params}")
        
        # 3. Execute workflow using shared ComfyKit instance from core
        try:
            # Get shared ComfyKit instance (lazy initialization + config hot-reload)
            kit = await self.core._get_or_create_comfykit()
            
            # Determine what to pass to ComfyKit based on source
            if workflow_info["source"] == "runninghub" and "workflow_id" in workflow_info:
                # RunningHub: pass workflow_id
                workflow_input = workflow_info["workflow_id"]
                logger.info(f"Executing RunningHub TTS workflow: {workflow_input}")
            else:
                # Selfhost: pass file path
                workflow_input = workflow_info["path"]
                logger.info(f"Executing selfhost TTS workflow: {workflow_input}")
            
            result = await kit.execute(workflow_input, workflow_params)
            
            # 4. Handle result
            if result.status != "completed":
                error_msg = result.msg or "Unknown error"
                logger.error(f"TTS generation failed: {error_msg}")
                raise Exception(f"TTS generation failed: {error_msg}")
            
            # ComfyKit result can have audio files in different output types
            # Try to get audio file path from result
            audio_path = None
            
            # Check for audio files in result.audios (if available)
            if hasattr(result, 'audios') and result.audios:
                audio_path = result.audios[0]
                logger.debug(f"✅ Found audio in result.audios: {audio_path}")
            # Check for files in result.files
            elif hasattr(result, 'files') and result.files:
                audio_path = result.files[0]
                logger.debug(f"✅ Found audio in result.files: {audio_path}")
            # Check in outputs dictionary
            elif hasattr(result, 'outputs') and result.outputs:
                logger.debug(f"Searching for audio file in result.outputs: {result.outputs}")
                # Try to find audio file in outputs
                for key, value in result.outputs.items():
                    if isinstance(value, str) and any(value.endswith(ext) for ext in ['.mp3', '.wav', '.flac']):
                        audio_path = value
                        logger.debug(f"✅ Found audio in result.outputs[{key}]: {audio_path}")
                        break
            
            if not audio_path:
                logger.error("No audio file generated")
                logger.error(f"❌ Result analysis:")
                logger.error(f"   - result.audios: {getattr(result, 'audios', 'NOT_FOUND')}")
                logger.error(f"   - result.files: {getattr(result, 'files', 'NOT_FOUND')}")
                logger.error(f"   - result.outputs: {getattr(result, 'outputs', 'NOT_FOUND')}")
                logger.error(f"   - Full __dict__: {result.__dict__}")
                raise Exception("No audio file generated by workflow")
            
            # If output_path provided and audio_path is URL, download to local
            if output_path and audio_path.startswith(('http://', 'https://')):
                import httpx
                import os
                
                # Ensure parent directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                logger.info(f"Downloading audio from {audio_path} to {output_path}")
                async with httpx.AsyncClient() as client:
                    response = await client.get(audio_path)
                    response.raise_for_status()
                    
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                
                logger.info(f"✅ Generated audio (ComfyUI): {output_path}")
                return output_path
            
            logger.info(f"✅ Generated audio (ComfyUI): {audio_path}")
            return audio_path
        
        except Exception as e:
            logger.error(f"TTS generation error: {e}")
            raise
