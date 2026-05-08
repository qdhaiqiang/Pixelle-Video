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
Configuration Manager - Singleton pattern

Provides unified access to configuration with automatic validation.
"""
from pathlib import Path
from typing import Any, Optional
from loguru import logger
from .schema import PixelleVideoConfig
from .loader import load_config_dict, save_config_dict


class ConfigManager:
    """
    Configuration Manager (Singleton)
    
    Provides unified access to configuration with automatic validation.
    """
    _instance: Optional['ConfigManager'] = None
    
    def __new__(cls, config_path: str = "config.yaml"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: str = "config.yaml"):
        # Only initialize once
        if hasattr(self, '_initialized'):
            return
        
        self.config_path = Path(config_path)
        self.config: PixelleVideoConfig = self._load()
        self._initialized = True
    
    def _load(self) -> PixelleVideoConfig:
        """Load configuration from file"""
        data = load_config_dict(str(self.config_path))
        config = PixelleVideoConfig(**data)
        
        # Validate template path exists
        self._validate_template(config.template.default_template)
        
        return config
    
    def _validate_template(self, template_path: str):
        """Validate that the configured template exists"""
        from pixelle_video.utils.template_util import resolve_template_path
        
        try:
            # Try to resolve the template path
            resolved_path = resolve_template_path(template_path)
            logger.debug(f"Template validation passed: {template_path} -> {resolved_path}")
        except FileNotFoundError as e:
            logger.warning(
                f"Configured default template '{template_path}' not found. "
                f"Will fall back to '1080x1920/default.html' if needed. Error: {e}"
            )
    
    def reload(self):
        """Reload configuration from file"""
        self.config = self._load()
        logger.info("Configuration reloaded")
    
    def save(self):
        """Save current configuration to file"""
        save_config_dict(self.config.to_dict(), str(self.config_path))
    
    def update(self, updates: dict):
        """
        Update configuration with new values
        
        Args:
            updates: Dictionary of updates (e.g., {"llm": {"api_key": "xxx"}})
        """
        current = self.config.to_dict()
        
        # Deep merge
        def deep_merge(base: dict, updates: dict) -> dict:
            for key, value in updates.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
            return base
        
        merged = deep_merge(current, updates)
        self.config = PixelleVideoConfig(**merged)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like access (for backward compatibility)"""
        return self.config.to_dict().get(key, default)
    
    def validate(self) -> bool:
        """Validate configuration completeness"""
        return self.config.validate_required()
    
    def get_llm_config(self) -> dict:
        """Get LLM configuration as dict"""
        return {
            "api_key": self.config.llm.api_key,
            "base_url": self.config.llm.base_url,
            "model": self.config.llm.model,
        }
    
    def set_llm_config(self, api_key: str, base_url: str, model: str):
        """Set LLM configuration"""
        self.update({
            "llm": {
                "api_key": api_key,
                "base_url": base_url,
                "model": model,
            }
        })
    
    def get_comfyui_config(self) -> dict:
        """Get ComfyUI configuration as dict"""
        return {
            "comfyui_url": self.config.comfyui.comfyui_url,
            "comfyui_api_key": self.config.comfyui.comfyui_api_key,
            "runninghub_api_key": self.config.comfyui.runninghub_api_key,
            "runninghub_concurrent_limit": self.config.comfyui.runninghub_concurrent_limit,
            "runninghub_instance_type": self.config.comfyui.runninghub_instance_type,
            "dashscope": {
                "api_key": self.config.comfyui.dashscope.api_key,
                "region": self.config.comfyui.dashscope.region,
                "base_url": self.config.comfyui.dashscope.base_url,
                "workspace": self.config.comfyui.dashscope.workspace,
                "poll_interval": self.config.comfyui.dashscope.poll_interval,
                "timeout": self.config.comfyui.dashscope.timeout,
            },
            "tts": {
                "default_workflow": self.config.comfyui.tts.default_workflow,
            },
            "image": {
                "default_workflow": self.config.comfyui.image.default_workflow,
                "prompt_prefix": self.config.comfyui.image.prompt_prefix,
            },
            "video": {
                "default_workflow": self.config.comfyui.video.default_workflow,
                "prompt_prefix": self.config.comfyui.video.prompt_prefix,
            }
        }
    
    def set_comfyui_config(
        self,
        comfyui_url: Optional[str] = None,
        comfyui_api_key: Optional[str] = None,
        runninghub_api_key: Optional[str] = None,
        runninghub_concurrent_limit: Optional[int] = None,
        runninghub_instance_type: Optional[str] = None,
        dashscope_api_key: Optional[str] = None,
        dashscope_region: Optional[str] = None,
        dashscope_base_url: Optional[str] = None,
        dashscope_workspace: Optional[str] = None,
    ):
        """Set ComfyUI global configuration"""
        updates = {}
        if comfyui_url is not None:
            updates["comfyui_url"] = comfyui_url
        if comfyui_api_key is not None:
            updates["comfyui_api_key"] = comfyui_api_key
        if runninghub_api_key is not None:
            updates["runninghub_api_key"] = runninghub_api_key
        if runninghub_concurrent_limit is not None:
            updates["runninghub_concurrent_limit"] = runninghub_concurrent_limit
        if runninghub_instance_type is not None:
            # Empty string means disable (treat as None for storage)
            updates["runninghub_instance_type"] = runninghub_instance_type if runninghub_instance_type else None
        dashscope_updates = {}
        if dashscope_api_key is not None:
            dashscope_updates["api_key"] = dashscope_api_key if dashscope_api_key else None
        if dashscope_region is not None:
            dashscope_updates["region"] = dashscope_region
        if dashscope_base_url is not None:
            dashscope_updates["base_url"] = dashscope_base_url if dashscope_base_url else None
        if dashscope_workspace is not None:
            dashscope_updates["workspace"] = dashscope_workspace if dashscope_workspace else None
        if dashscope_updates:
            updates["dashscope"] = dashscope_updates

        if updates:
            self.update({"comfyui": updates})

    def get_bilibili_config(self) -> dict:
        """Get Bilibili default configuration as dict"""
        return {
            "default_tid": self.config.bilibili.default_tid,
            "default_copyright": self.config.bilibili.default_copyright,
        }

    def set_bilibili_config(
        self,
        default_tid: Optional[int] = None,
        default_copyright: Optional[int] = None,
    ):
        """Set Bilibili default configuration"""
        updates = {}
        if default_tid is not None:
            updates["default_tid"] = default_tid
        if default_copyright is not None:
            updates["default_copyright"] = default_copyright
        if updates:
            self.update({"bilibili": updates})
