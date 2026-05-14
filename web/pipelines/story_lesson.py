# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Story Lesson Pipeline UI

UI for generating children's story explanation videos from a title.
"""

import os
import time
from pathlib import Path
from typing import Any

import streamlit as st
from loguru import logger

from pixelle_video.config import config_manager
from pixelle_video.models.progress import ProgressEvent
from pixelle_video.services.cosyvoice_installer import get_cosyvoice_status
from web.components.content_input import render_bgm_section
from web.i18n import get_language, tr
from web.pipelines.base import PipelineUI, register_pipeline_ui
from web.utils.async_helpers import run_async
from web.utils.streamlit_helpers import check_and_warn_selfhost_workflow


def _get_template_preview_path(template_path: str, language: str = "zh_CN") -> str:
    path_parts = template_path.split("/")
    if len(path_parts) < 2:
        return ""

    size = path_parts[0]
    template_name = path_parts[1].replace(".html", "")
    suffix = "" if language == "zh_CN" else "_en"

    for ext in [".jpg", ".png"]:
        preview_path = f"docs/images/{size}/{template_name}{suffix}{ext}"
        if os.path.exists(preview_path):
            return preview_path

    for ext in [".jpg", ".png"]:
        preview_path = f"docs/images/{size}/{template_name}{ext}"
        if os.path.exists(preview_path):
            return preview_path

    return ""


class StoryLessonPipelineUI(PipelineUI):
    """UI for the story lesson video generation pipeline."""

    name = "story_lesson"
    icon = "📖"

    @property
    def display_name(self):
        return tr("pipeline.story_lesson.name")

    @property
    def description(self):
        return tr("pipeline.story_lesson.description")

    def render(self, pixelle_video: Any):
        left_col, middle_col, right_col = st.columns([1, 1, 1])

        with left_col:
            story_params = self._render_story_input()
            bgm_params = render_bgm_section(key_prefix="story_")

        with middle_col:
            style_params = self._render_story_style_config(pixelle_video)

        with right_col:
            video_params = {
                **story_params,
                **bgm_params,
                **style_params,
            }
            self._render_output(pixelle_video, video_params)

    def _render_story_input(self) -> dict:
        with st.container(border=True):
            st.markdown(f"**{tr('story_lesson.section.input')}**")

            story_title = st.text_input(
                tr("story_lesson.title"),
                placeholder=tr("story_lesson.title_placeholder"),
                help=tr("story_lesson.title_help"),
                key="story_lesson_title",
            )

            category_options = {
                "idiom": tr("story_lesson.category.idiom"),
                "proverb": tr("story_lesson.category.proverb"),
                "fable": tr("story_lesson.category.fable"),
                "myth": tr("story_lesson.category.myth"),
                "poetry": tr("story_lesson.category.poetry"),
                "textbook": tr("story_lesson.category.textbook"),
                "historical": tr("story_lesson.category.historical"),
                "culture": tr("story_lesson.category.culture"),
                "science": tr("story_lesson.category.science"),
                "safety": tr("story_lesson.category.safety"),
                "character": tr("story_lesson.category.character"),
                "picture_book": tr("story_lesson.category.picture_book"),
            }
            story_category = st.selectbox(
                tr("story_lesson.category"),
                options=list(category_options.keys()),
                format_func=lambda value: category_options[value],
                index=0,
                key="story_lesson_category",
            )

            all_audience_options = {
                "preschool": tr("story_lesson.audience.preschool"),
                "lower": tr("story_lesson.audience.lower"),
                "middle": tr("story_lesson.audience.middle"),
                "upper": tr("story_lesson.audience.upper"),
                "teen": tr("story_lesson.audience.teen"),
                "adult": tr("story_lesson.audience.adult"),
                "elderly": tr("story_lesson.audience.elderly"),
            }
            audience_by_category = {
                "picture_book": ["preschool", "lower"],
                "textbook": ["lower", "middle", "upper", "teen"],
                "safety": ["lower", "middle", "upper", "teen", "adult", "elderly"],
                "science": ["lower", "middle", "upper", "teen", "adult", "elderly"],
                "culture": ["lower", "middle", "upper", "teen", "adult", "elderly"],
                "historical": ["lower", "middle", "upper", "teen", "adult", "elderly"],
                "poetry": ["lower", "middle", "upper", "teen", "adult", "elderly"],
                "character": ["preschool", "lower", "middle", "upper", "teen", "adult"],
            }
            default_audience_by_category = {
                "picture_book": "preschool",
                "textbook": "middle",
                "safety": "adult",
                "science": "middle",
                "culture": "adult",
                "historical": "middle",
                "poetry": "middle",
                "character": "lower",
            }
            allowed_audience_keys = audience_by_category.get(
                story_category,
                ["lower", "middle", "upper", "teen", "adult"],
            )
            audience_options = {
                key: all_audience_options[key]
                for key in allowed_audience_keys
            }

            last_category = st.session_state.get("story_lesson_last_category")
            if last_category != story_category:
                st.session_state["story_lesson_audience"] = default_audience_by_category.get(story_category, "middle")
                st.session_state["story_lesson_last_category"] = story_category

            if st.session_state.get("story_lesson_audience") not in audience_options:
                st.session_state["story_lesson_audience"] = allowed_audience_keys[0]
            audience_key = st.selectbox(
                tr("story_lesson.audience"),
                options=list(audience_options.keys()),
                format_func=lambda value: audience_options[value],
                help=tr("story_lesson.audience_help"),
                key="story_lesson_audience",
            )
            audience = audience_options[audience_key]

            duration_limits_by_category = {
                "picture_book": (30, 150, 60, 15),
                "idiom": (30, 180, 75, 15),
                "proverb": (30, 180, 75, 15),
                "fable": (30, 180, 75, 15),
                "myth": (45, 240, 120, 15),
                "poetry": (45, 240, 120, 15),
                "textbook": (45, 240, 120, 15),
                "historical": (60, 300, 180, 30),
                "culture": (60, 300, 180, 30),
                "science": (45, 240, 120, 15),
                "safety": (30, 180, 90, 15),
                "character": (30, 180, 90, 15),
            }
            min_duration, max_duration, default_duration, duration_step = duration_limits_by_category.get(
                story_category,
                (30, 180, 75, 15),
            )
            duration_state_key = f"story_lesson_duration_{story_category}"
            duration_seconds = st.slider(
                tr("story_lesson.duration"),
                min_value=min_duration,
                max_value=max_duration,
                value=default_duration,
                step=duration_step,
                help=tr("story_lesson.duration_help"),
                key=duration_state_key,
            )

            scene_limits_by_category = {
                "picture_book": (4, 10, 6),
                "idiom": (4, 10, 6),
                "proverb": (4, 10, 6),
                "fable": (4, 10, 6),
                "myth": (5, 12, 8),
                "poetry": (5, 10, 6),
                "textbook": (5, 12, 8),
                "historical": (6, 16, 10),
                "culture": (6, 16, 10),
                "science": (5, 12, 8),
                "safety": (4, 10, 6),
                "character": (4, 10, 6),
            }
            min_scenes, max_scenes, default_scenes = scene_limits_by_category.get(story_category, (4, 10, 6))
            n_scenes = st.slider(
                tr("story_lesson.scenes"),
                min_value=min_scenes,
                max_value=max_scenes,
                value=default_scenes,
                step=1,
                key=f"story_lesson_scenes_{story_category}",
            )
            st.caption(tr("story_lesson.scenes_help"))

            teaching_option_key = story_category if story_category in {
                "poetry",
                "textbook",
                "historical",
                "culture",
                "science",
                "safety",
                "character",
            } else "story"
            include_moral = st.checkbox(
                tr(f"story_lesson.include_teaching.{teaching_option_key}"),
                value=True,
                key="story_lesson_include_moral",
            )
            include_life_example = st.checkbox(
                tr(f"story_lesson.include_example.{teaching_option_key}"),
                value=True,
                key="story_lesson_include_life_example",
            )

            extra_requirements = st.text_area(
                tr("story_lesson.extra_requirements"),
                placeholder=tr("story_lesson.extra_requirements_placeholder"),
                height=90,
                key="story_lesson_extra_requirements",
            )

        return {
            "text": story_title,
            "story_category": story_category,
            "audience": audience,
            "duration_seconds": duration_seconds,
            "n_scenes": n_scenes,
            "include_moral": include_moral,
            "include_life_example": include_life_example,
            "extra_requirements": extra_requirements,
        }

    def _render_story_style_config(self, pixelle_video: Any) -> dict:
        tts_params = self._render_tts_config(pixelle_video)
        visual_params = self._render_visual_config(pixelle_video)
        return {**tts_params, **visual_params}

    def _render_tts_config(self, pixelle_video: Any) -> dict:
        with st.container(border=True):
            st.markdown(f"**{tr('section.tts')}**")

            comfyui_config = config_manager.get_comfyui_config()
            tts_config = comfyui_config["tts"]
            tts_modes = ["local", "cosyvoice", "comfyui"]
            saved_mode = tts_config.get("inference_mode", "local")
            tts_mode = st.radio(
                tr("tts.inference_mode"),
                tts_modes,
                horizontal=True,
                format_func=lambda value: tr(f"tts.mode.{value}"),
                index=tts_modes.index(saved_mode) if saved_mode in tts_modes else 0,
                key="story_tts_inference_mode",
            )
            st.caption(tr(f"tts.mode.{tts_mode}_hint"))

            selected_voice = None
            tts_speed = None
            tts_workflow_key = None
            ref_audio_path = None

            if tts_mode == "local":
                from pixelle_video.tts_voices import EDGE_TTS_VOICES, get_voice_display_name

                local_config = tts_config.get("local", {})
                saved_voice = local_config.get("voice", "zh-CN-XiaoxiaoNeural")
                saved_speed = local_config.get("speed", 1.1)
                voice_options = []
                voice_ids = []
                default_voice_index = 0

                for idx, voice_config in enumerate(EDGE_TTS_VOICES):
                    voice_id = voice_config["id"]
                    voice_options.append(get_voice_display_name(voice_id, tr, get_language()))
                    voice_ids.append(voice_id)
                    if voice_id == saved_voice:
                        default_voice_index = idx

                voice_col, speed_col = st.columns([1, 1])
                with voice_col:
                    selected_voice_display = st.selectbox(
                        tr("tts.voice_selector"),
                        voice_options,
                        index=default_voice_index,
                        key="story_tts_local_voice",
                    )
                    selected_voice = voice_ids[voice_options.index(selected_voice_display)]
                with speed_col:
                    tts_speed = st.slider(
                        tr("tts.speed"),
                        min_value=0.5,
                        max_value=2.0,
                        value=saved_speed,
                        step=0.1,
                        format="%.1fx",
                        key="story_tts_local_speed",
                    )
            elif tts_mode == "cosyvoice":
                cosyvoice_config = tts_config.get("cosyvoice", {})
                status = get_cosyvoice_status()
                if not cosyvoice_config.get("enabled", False):
                    st.warning(tr("tts.cosyvoice.configure_in_settings"))
                elif status.installed:
                    st.success(tr("tts.cosyvoice.installed", path=status.repo_dir))
                else:
                    st.warning(tr("tts.cosyvoice.not_installed_settings", message=status.message))

                voice_col, speed_col = st.columns([1, 1])
                with voice_col:
                    cosyvoice_speakers = ["中文女", "中文男", "英文女", "英文男", "日语男", "粤语女", "韩语女"]
                    saved_speaker = cosyvoice_config.get("speaker", "中文女")
                    selected_voice = st.selectbox(
                        tr("tts.voice_selector"),
                        cosyvoice_speakers,
                        index=cosyvoice_speakers.index(saved_speaker) if saved_speaker in cosyvoice_speakers else 0,
                        key="story_tts_cosyvoice_voice",
                    )
                with speed_col:
                    saved_speed = tts_config.get("local", {}).get("speed", 1.1)
                    tts_speed = st.slider(
                        tr("tts.speed"),
                        min_value=0.5,
                        max_value=2.0,
                        value=saved_speed,
                        step=0.1,
                        format="%.1fx",
                        key="story_tts_cosyvoice_speed",
                    )
            else:
                tts_workflows = pixelle_video.tts.list_workflows()
                workflow_options = [wf["display_name"] for wf in tts_workflows]
                workflow_keys = [wf["key"] for wf in tts_workflows]
                saved_tts_workflow = tts_config.get("comfyui", {}).get("default_workflow")
                default_tts_index = workflow_keys.index(saved_tts_workflow) if saved_tts_workflow in workflow_keys else 0

                workflow_display = st.selectbox(
                    tr("tts.voice_selector"),
                    workflow_options if workflow_options else ["No TTS workflows found"],
                    index=default_tts_index,
                    key="story_tts_workflow_select",
                )
                tts_workflow_key = workflow_keys[workflow_options.index(workflow_display)] if workflow_options else "selfhost/tts_edge.json"
                check_and_warn_selfhost_workflow(tts_workflow_key)

                ref_audio_file = st.file_uploader(
                    tr("tts.ref_audio"),
                    type=["mp3", "wav", "flac", "m4a", "aac", "ogg"],
                    help=tr("tts.ref_audio_help"),
                    key="story_ref_audio_upload",
                )
                if ref_audio_file is not None:
                    st.audio(ref_audio_file)
                    temp_dir = Path("temp")
                    temp_dir.mkdir(exist_ok=True)
                    ref_audio_path = temp_dir / f"story_ref_audio_{ref_audio_file.name}"
                    with open(ref_audio_path, "wb") as f:
                        f.write(ref_audio_file.getbuffer())

        return {
            "tts_inference_mode": tts_mode,
            "tts_voice": selected_voice if tts_mode in {"local", "cosyvoice"} else None,
            "tts_speed": tts_speed if tts_mode in {"local", "cosyvoice"} else None,
            "tts_workflow": tts_workflow_key if tts_mode == "comfyui" else None,
            "ref_audio": str(ref_audio_path) if ref_audio_path else None,
        }

    def _render_visual_config(self, pixelle_video: Any) -> dict:
        with st.container(border=True):
            st.markdown(f"**{tr('story_lesson.section.visual')}**")

            visual_style_options = {
                "textbook": tr("story_lesson.visual_style.textbook"),
                "picture_book": tr("story_lesson.visual_style.picture_book"),
                "watercolor": tr("story_lesson.visual_style.watercolor"),
                "chinese": tr("story_lesson.visual_style.chinese"),
                "cartoon": tr("story_lesson.visual_style.cartoon"),
            }
            visual_style_key = st.selectbox(
                tr("story_lesson.visual_style"),
                options=list(visual_style_options.keys()),
                format_func=lambda value: visual_style_options[value],
                index=0,
                key="story_visual_style",
            )
            story_visual_style = visual_style_options[visual_style_key]

            template_type_options = {
                "image": tr("template.type.image"),
                "video": tr("template.type.video"),
            }
            selected_template_type = st.radio(
                tr("template.type_selector"),
                options=list(template_type_options.keys()),
                format_func=lambda value: template_type_options[value],
                index=0,
                horizontal=True,
                key="story_template_type_selector",
            )

            st.markdown(f"**{tr('section.template')}**")

            from pixelle_video.services.frame_html import HTMLFrameGenerator
            from pixelle_video.utils.template_util import (
                get_templates_grouped_by_size_and_type,
                resolve_template_path,
            )

            orientation_options = {
                "1080x1920": tr("story_lesson.aspect.portrait"),
                "1920x1080": tr("story_lesson.aspect.landscape"),
                "1080x1080": tr("story_lesson.aspect.square"),
            }
            grouped_templates = get_templates_grouped_by_size_and_type(selected_template_type)
            available_orientations = [
                size
                for size in orientation_options
                if any(
                    template.display_info.name.startswith(("image_", "video_"))
                    for template in grouped_templates.get(size, [])
                )
            ]
            if not available_orientations:
                available_orientations = ["1080x1920"]

            aspect_state_key = f"story_aspect_{selected_template_type}"
            if st.session_state.get(aspect_state_key) not in available_orientations:
                st.session_state[aspect_state_key] = available_orientations[0]

            selected_orientation = st.selectbox(
                tr("story_lesson.aspect"),
                options=available_orientations,
                format_func=lambda value: orientation_options[value],
                key=aspect_state_key,
            )

            template_options = []
            for size, templates in grouped_templates.items():
                if size != selected_orientation:
                    continue
                for template in templates:
                    name = template.display_info.name
                    if name.startswith(("image_", "video_")):
                        template_options.append((template.template_path, name))

            default_template = (
                "1080x1920/image_story_fullscreen.html"
                if selected_template_type == "image"
                else "1080x1920/video_live_fullscreen.html"
            )
            template_paths = [item[0] for item in template_options]
            default_index = template_paths.index(default_template) if default_template in template_paths else 0

            if not template_options:
                st.warning(tr("template.no_templates_with_preview"))
                frame_template = default_template
            else:
                state_key = f"story_selected_template_{selected_template_type}_{selected_orientation}"
                legacy_center_templates = {
                    "1080x1920/image_default.html",
                    "1080x1920/video_story_portrait.html",
                }
                selected_template = st.session_state.get(state_key)
                if selected_template not in template_paths or (
                    selected_template in legacy_center_templates and default_template in template_paths
                ):
                    st.session_state[state_key] = template_paths[default_index]

                with st.expander(tr("template.gallery_view"), expanded=True):
                    num_cols = 3
                    cols = st.columns(num_cols)
                    width, height = selected_orientation.split("x", maxsplit=1)
                    for idx, (template_path, template_name) in enumerate(template_options):
                        with cols[idx % num_cols]:
                            preview_path = _get_template_preview_path(template_path, get_language())
                            if preview_path:
                                st.image(preview_path, use_container_width=True)
                            else:
                                st.markdown(
                                    f"""
                                    <div style="
                                        aspect-ratio: {width} / {height};
                                        width: 100%;
                                        border-radius: 8px;
                                        background: linear-gradient(135deg, #f4f0e8 0%, #d7e7e1 55%, #b7d3df 100%);
                                        display: flex;
                                        align-items: center;
                                        justify-content: center;
                                        text-align: center;
                                        color: #1f2933;
                                        padding: 12px;
                                        margin-bottom: 12px;
                                        border: 1px solid rgba(31,41,51,.12);
                                    ">
                                        <div style="font-size: 14px; line-height: 1.35; word-break: break-word;">{template_name}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                            is_selected = st.session_state[state_key] == template_path
                            if st.button(
                                tr("template.selected") if is_selected else tr("template.select_button"),
                                key=f"story_template_{selected_template_type}_{selected_orientation}_{template_path}",
                                use_container_width=True,
                                type="primary" if is_selected else "secondary",
                            ):
                                st.session_state[state_key] = template_path
                                st.rerun()

                frame_template = st.session_state[state_key]
                selected_template_name = dict(template_options).get(frame_template)
                if selected_template_name:
                    st.info(f"📋 {tr('template.selected_template')}: **{selected_template_name}**")

            template_path_for_params = resolve_template_path(frame_template)
            generator_for_params = HTMLFrameGenerator(template_path_for_params)
            media_width, media_height = generator_for_params.get_media_size()
            size_info_key = "style.video_size_info" if selected_template_type == "video" else "style.image_size_info"
            st.info(tr(size_info_key, width=media_width, height=media_height))

            all_workflows = pixelle_video.media.list_workflows()
            if selected_template_type == "video":
                workflows = [wf for wf in all_workflows if "video_" in wf["key"].lower()]
                media_config_key = "video"
            else:
                workflows = [wf for wf in all_workflows if "image_" in wf["key"].lower()]
                media_config_key = "image"

            workflow_options = [wf["display_name"] for wf in workflows]
            workflow_keys = [wf["key"] for wf in workflows]
            comfyui_config = config_manager.get_comfyui_config()
            saved_workflow = comfyui_config.get(media_config_key, {}).get("default_workflow", "")
            default_workflow_index = workflow_keys.index(saved_workflow) if saved_workflow in workflow_keys else 0

            workflow_display = st.selectbox(
                "Workflow",
                workflow_options if workflow_options else ["No workflows found"],
                index=default_workflow_index,
                label_visibility="collapsed",
                key=f"story_media_workflow_select_{selected_template_type}",
            )
            media_workflow = workflow_keys[workflow_options.index(workflow_display)] if workflow_options else None
            if media_workflow:
                check_and_warn_selfhost_workflow(media_workflow)

            with st.expander(tr("story_lesson.prompt_prefix.advanced"), expanded=False):
                prompt_prefix = st.text_area(
                    tr("style.prompt_prefix"),
                    value="",
                    placeholder=tr("story_lesson.prompt_prefix_placeholder"),
                    height=80,
                    help=tr("story_lesson.prompt_prefix_help"),
                    key="story_prompt_prefix",
                )

            media_scale_mode = "cover"
            if selected_template_type == "video":
                scale_options = {
                    "cover": tr("story_lesson.scale.cover"),
                    "contain": tr("story_lesson.scale.contain"),
                    "stretch": tr("story_lesson.scale.stretch"),
                }
                media_scale_mode = st.radio(
                    tr("story_lesson.scale"),
                    options=list(scale_options.keys()),
                    format_func=lambda value: scale_options[value],
                    index=0,
                    horizontal=True,
                    key="story_media_scale_mode",
                )

        return {
            "story_visual_style": story_visual_style,
            "frame_template": frame_template,
            "media_workflow": media_workflow,
            "prompt_prefix": prompt_prefix if prompt_prefix else "",
            "media_width": media_width,
            "media_height": media_height,
            "media_scale_mode": media_scale_mode,
        }

    def _render_output(self, pixelle_video: Any, video_params: dict):
        with st.container(border=True):
            st.markdown(f"**{tr('section.video_generation')}**")

            if not config_manager.validate():
                st.warning(tr("settings.not_configured"))

            if not video_params.get("text"):
                st.info(tr("story_lesson.output.need_title"))
                st.button(tr("btn.generate"), type="primary", use_container_width=True, disabled=True, key="story_generate_disabled")
                self._render_stored_result()
                return

            st.info(tr("story_lesson.output.ready", scenes=video_params.get("n_scenes", 6)))

            if st.button(tr("btn.generate"), type="primary", use_container_width=True, key="story_generate"):
                if not config_manager.validate():
                    st.error(tr("settings.not_configured"))
                    st.stop()

                progress_bar = st.progress(0)
                status_text = st.empty()
                start_time = time.time()

                try:
                    def update_progress(event: ProgressEvent):
                        if event.event_type == "generating_storyboard":
                            message = tr("story_lesson.progress.generating_storyboard")
                        elif event.event_type == "frame_step":
                            action_text = tr(f"progress.step_{event.action}")
                            message = tr(
                                "progress.frame_step",
                                current=event.frame_current,
                                total=event.frame_total,
                                step=event.step,
                                action=action_text,
                            )
                        elif event.event_type == "processing_frame":
                            message = tr("progress.frame", current=event.frame_current, total=event.frame_total)
                        else:
                            message = tr(f"progress.{event.event_type}")
                        if event.extra_info:
                            message = f"{message} - {event.extra_info}"
                        status_text.text(message)
                        progress_bar.progress(min(int(event.progress * 100), 99))

                    result = run_async(pixelle_video.generate_video(
                        text=video_params["text"],
                        pipeline="story_lesson",
                        story_category=video_params.get("story_category"),
                        audience=video_params.get("audience"),
                        duration_seconds=video_params.get("duration_seconds"),
                        n_scenes=video_params.get("n_scenes"),
                        include_moral=video_params.get("include_moral"),
                        include_life_example=video_params.get("include_life_example"),
                        extra_requirements=video_params.get("extra_requirements"),
                        story_visual_style=video_params.get("story_visual_style"),
                        media_workflow=video_params.get("media_workflow"),
                        frame_template=video_params.get("frame_template"),
                        prompt_prefix=video_params.get("prompt_prefix"),
                        bgm_path=video_params.get("bgm_path"),
                        bgm_volume=video_params.get("bgm_volume", 0.2),
                        progress_callback=update_progress,
                        media_width=video_params.get("media_width"),
                        media_height=video_params.get("media_height"),
                        media_scale_mode=video_params.get("media_scale_mode", "contain"),
                        tts_inference_mode=video_params.get("tts_inference_mode", "local"),
                        tts_voice=video_params.get("tts_voice"),
                        tts_speed=video_params.get("tts_speed"),
                        tts_workflow=video_params.get("tts_workflow"),
                        ref_audio=video_params.get("ref_audio"),
                    ))

                    total_time = time.time() - start_time
                    progress_bar.progress(100)
                    status_text.text(tr("status.success"))
                    st.session_state["story_lesson_last_result"] = {
                        "video_path": result.video_path,
                        "file_size": result.file_size,
                        "n_frames": len(result.storyboard.frames),
                        "generation_time": total_time,
                    }
                except Exception as e:
                    status_text.text("")
                    progress_bar.empty()
                    st.error(tr("status.error", error=str(e)))
                    logger.exception(e)
                    st.stop()

            self._render_stored_result()

    def _render_stored_result(self):
        result = st.session_state.get("story_lesson_last_result")
        if not result:
            return

        video_path = result.get("video_path")
        st.success(tr("status.video_generated", path=video_path))
        st.markdown("---")

        file_size_mb = (result.get("file_size") or 0) / (1024 * 1024)
        st.caption(
            f"⏱️ {tr('info.generation_time')} {result.get('generation_time', 0):.1f}s   "
            f"📦 {file_size_mb:.2f}MB   "
            f"🎬 {result.get('n_frames', 0)}{tr('info.scenes_unit')}"
        )
        st.markdown("---")

        if video_path and os.path.exists(video_path):
            st.video(video_path)
            with open(video_path, "rb") as video_file:
                st.download_button(
                    label="⬇️ 下载视频" if get_language() == "zh_CN" else "⬇️ Download Video",
                    data=video_file.read(),
                    file_name=os.path.basename(video_path),
                    mime="video/mp4",
                    use_container_width=True,
                    key="story_lesson_download_last",
                )
        else:
            st.error(tr("status.video_not_found", path=video_path))


register_pipeline_ui(StoryLessonPipelineUI)
