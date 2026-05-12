# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Screen recording processing UI."""

import os
import time
from pathlib import Path
from typing import Any, Optional

import streamlit as st
from loguru import logger

from pixelle_video.config import config_manager
from pixelle_video.services.screen_recording_processor import ScreenRecordingProcessor
from pixelle_video.services.video import VideoService
from web.i18n import get_language, tr
from web.pipelines.base import PipelineUI, register_pipeline_ui
from web.utils.async_helpers import run_async
from web.utils.streamlit_helpers import check_and_warn_selfhost_workflow


class ScreenRecordingPipelineUI(PipelineUI):
    name = "screen_recording"
    icon = "🖥️"

    @property
    def display_name(self):
        return tr("screen_recording.display_name")

    @property
    def description(self):
        return tr("screen_recording.description")

    def render(self, pixelle_video: Any):
        left_col, middle_col, right_col = st.columns([1, 1, 1])

        with left_col:
            input_params = self._render_input()

        with middle_col:
            subtitle_params = self._render_subtitle_settings()
            dubbing_params = self._render_dubbing_settings(pixelle_video)

        with right_col:
            params = {**input_params, **subtitle_params, **dubbing_params}
            self._render_output(pixelle_video, params)

    def _render_input(self) -> dict:
        with st.container(border=True):
            st.markdown(f"**{tr('screen_recording.input.title')}**")
            st.caption(tr("screen_recording.input.caption"))

            video_path = st.text_input(
                tr("screen_recording.input.video_path"),
                placeholder="/Users/name/Videos/demo.mp4",
                help=tr("screen_recording.input.video_path_help"),
                key="screen_recording_video_path",
            )

            if video_path:
                preview_path = Path(video_path).expanduser()
                if preview_path.is_file():
                    st.success(tr("screen_recording.input.video_found"))
                    st.video(str(preview_path))
                elif preview_path.is_dir():
                    st.error(tr("screen_recording.input.path_is_directory", path=preview_path))
                else:
                    st.error(tr("screen_recording.input.video_missing", path=preview_path))

            glossary_file = st.file_uploader(
                tr("screen_recording.input.glossary_upload"),
                type=["md", "markdown"],
                help=tr("screen_recording.input.glossary_help"),
                key="screen_recording_glossary_upload",
            )

            correction_file = st.file_uploader(
                tr("screen_recording.input.correction_upload"),
                type=["md", "markdown"],
                help=tr("screen_recording.input.correction_help"),
                key="screen_recording_correction_upload",
            )

            temp_dir = Path("temp") / "screen_recording_rules"
            temp_dir.mkdir(parents=True, exist_ok=True)
            glossary_path = self._save_uploaded_file(glossary_file, temp_dir) if glossary_file else None
            correction_path = self._save_uploaded_file(correction_file, temp_dir) if correction_file else None

            if glossary_path:
                st.success(tr("screen_recording.input.glossary_uploaded", path=glossary_path))
            if correction_path:
                st.success(tr("screen_recording.input.correction_uploaded", path=correction_path))

        return {
            "video_path": video_path.strip(),
            "glossary_path": str(glossary_path) if glossary_path else None,
            "correction_path": str(correction_path) if correction_path else None,
        }

    def _render_subtitle_settings(self) -> dict:
        with st.container(border=True):
            st.markdown(f"**{tr('screen_recording.subtitle.title')}**")

            industry_context = st.text_area(
                tr("screen_recording.subtitle.industry_context"),
                placeholder=tr("screen_recording.subtitle.industry_context_placeholder"),
                height=110,
                help=tr("screen_recording.subtitle.industry_context_help"),
                key="screen_recording_industry_context",
            )

            col1, col2 = st.columns(2)
            with col1:
                whisper_model = st.selectbox(
                    tr("screen_recording.subtitle.whisper_model"),
                    ["tiny", "base", "small", "medium", "large-v3"],
                    index=2,
                    key="screen_recording_whisper_model",
                )
            with col2:
                language = st.selectbox(
                    tr("screen_recording.subtitle.language"),
                    ["zh", "en", "ja", "ko"],
                    index=0,
                    key="screen_recording_language",
                )

            ai_polish = st.checkbox(
                tr("screen_recording.subtitle.ai_polish"),
                value=True,
                help=tr("screen_recording.subtitle.ai_polish_help"),
                key="screen_recording_ai_polish",
            )

            pace_mode = st.radio(
                tr("screen_recording.pace.mode"),
                ["keep_original", "smart_compress"],
                horizontal=True,
                format_func=lambda x: tr(f"screen_recording.pace.mode.{x}"),
                key="screen_recording_pace_mode",
            )
            silence_gap_threshold = 1.2
            clip_padding = 0.45
            if pace_mode == "smart_compress":
                col_gap, col_pad = st.columns(2)
                with col_gap:
                    silence_gap_threshold = st.slider(
                        tr("screen_recording.pace.gap_threshold"),
                        min_value=0.5,
                        max_value=5.0,
                        value=1.2,
                        step=0.1,
                        help=tr("screen_recording.pace.gap_threshold_help"),
                        key="screen_recording_silence_gap_threshold",
                    )
                with col_pad:
                    clip_padding = st.slider(
                        tr("screen_recording.pace.clip_padding"),
                        min_value=0.1,
                        max_value=2.0,
                        value=0.45,
                        step=0.05,
                        help=tr("screen_recording.pace.clip_padding_help"),
                        key="screen_recording_clip_padding",
                    )

        return {
            "industry_context": industry_context.strip(),
            "whisper_model": whisper_model,
            "language": language,
            "ai_polish": ai_polish,
            "pace_mode": pace_mode,
            "silence_gap_threshold": silence_gap_threshold,
            "clip_padding": clip_padding,
        }

    def _render_dubbing_settings(self, pixelle_video: Any) -> dict:
        with st.container(border=True):
            st.markdown(f"**{tr('section.tts')}**")

            with st.expander(tr("help.feature_description"), expanded=False):
                st.markdown(f"**{tr('help.what')}**")
                st.markdown(tr("tts.what"))
                st.markdown(f"**{tr('help.how')}**")
                st.markdown(tr("tts.how"))

            synthesize_dubbing = st.checkbox(
                tr("screen_recording.tts.enable"),
                value=False,
                help=tr("screen_recording.tts.enable_help"),
                key="screen_recording_synthesize_dubbing",
            )

            tts_inference_mode = "local"
            tts_voice = "zh-CN-XiaoxiaoNeural"
            tts_speed = 1.2
            tts_workflow = None
            ref_audio_path: Optional[str] = None
            bgm_path = None
            bgm_volume = 0.10

            if synthesize_dubbing:
                comfyui_config = config_manager.get_comfyui_config()
                tts_config = comfyui_config["tts"]

                tts_inference_mode = st.radio(
                    tr("tts.inference_mode"),
                    ["local", "comfyui"],
                    horizontal=True,
                    format_func=lambda x: tr(f"tts.mode.{x}"),
                    index=0 if tts_config.get("inference_mode", "local") == "local" else 1,
                    key="screen_recording_tts_mode",
                )

                if tts_inference_mode == "local":
                    st.caption(tr("tts.mode.local_hint"))
                else:
                    st.caption(tr("tts.mode.comfyui_hint"))

                if tts_inference_mode == "local":
                    from pixelle_video.tts_voices import EDGE_TTS_VOICES, get_voice_display_name

                    local_config = tts_config.get("local", {})
                    saved_voice = local_config.get("voice", "zh-CN-YunjianNeural")
                    saved_speed = local_config.get("speed", 1.2)

                    voice_options = []
                    voice_ids = []
                    default_voice_index = 0

                    for idx, voice_config in enumerate(EDGE_TTS_VOICES):
                        voice_id = voice_config["id"]
                        display_name = get_voice_display_name(voice_id, tr, get_language())
                        voice_options.append(display_name)
                        voice_ids.append(voice_id)
                        if voice_id == saved_voice:
                            default_voice_index = idx

                    voice_col, speed_col = st.columns([1, 1])

                    with voice_col:
                        selected_voice_display = st.selectbox(
                            tr("tts.voice_selector"),
                            voice_options,
                            index=default_voice_index,
                            key="screen_recording_tts_voice",
                        )
                        selected_voice_index = voice_options.index(selected_voice_display)
                        tts_voice = voice_ids[selected_voice_index]

                    with speed_col:
                        tts_speed = st.slider(
                            tr("tts.speed"),
                            min_value=0.5,
                            max_value=2.0,
                            value=saved_speed,
                            step=0.1,
                            format="%.1fx",
                            key="screen_recording_tts_speed",
                        )
                        st.caption(tr("tts.speed_label", speed=f"{tts_speed:.1f}"))
                else:
                    workflows = pixelle_video.tts.list_workflows()
                    tts_workflow_options = [wf["display_name"] for wf in workflows]
                    tts_workflow_keys = [wf["key"] for wf in workflows]

                    default_tts_index = 0
                    saved_tts_workflow = tts_config.get("comfyui", {}).get("default_workflow")
                    if saved_tts_workflow and saved_tts_workflow in tts_workflow_keys:
                        default_tts_index = tts_workflow_keys.index(saved_tts_workflow)

                    tts_workflow_display = st.selectbox(
                        tr("tts.voice_selector"),
                        tts_workflow_options if tts_workflow_options else ["No TTS workflows found"],
                        index=default_tts_index,
                        key="screen_recording_tts_workflow",
                    )

                    if tts_workflow_options:
                        tts_selected_index = tts_workflow_options.index(tts_workflow_display)
                        tts_workflow = tts_workflow_keys[tts_selected_index]
                    else:
                        tts_workflow = "selfhost/tts_edge.json"

                    check_and_warn_selfhost_workflow(tts_workflow)

                    ref_audio = st.file_uploader(
                        tr("tts.ref_audio"),
                        type=["mp3", "wav", "flac", "m4a", "aac", "ogg"],
                        help=tr("tts.ref_audio_help"),
                        key="screen_recording_ref_audio",
                    )
                    if ref_audio is not None:
                        st.audio(ref_audio)
                        temp_dir = Path("temp")
                        temp_dir.mkdir(exist_ok=True)
                        path = temp_dir / f"ref_audio_{ref_audio.name}"
                        with open(path, "wb") as f:
                            f.write(ref_audio.getbuffer())
                        ref_audio_path = str(path)

                with st.expander(tr("tts.preview_title"), expanded=False):
                    preview_text = st.text_input(
                        tr("tts.preview_text"),
                        value=tr("screen_recording.tts.preview_default"),
                        placeholder=tr("tts.preview_text_placeholder"),
                        key="screen_recording_preview_text",
                    )

                    if st.button(
                        tr("tts.preview_button"),
                        key="screen_recording_preview_tts",
                        use_container_width=True,
                    ):
                        with st.spinner(tr("tts.previewing")):
                            try:
                                tts_params = {
                                    "text": preview_text,
                                    "inference_mode": tts_inference_mode,
                                }
                                if tts_inference_mode == "local":
                                    tts_params["voice"] = tts_voice
                                    tts_params["speed"] = tts_speed
                                else:
                                    tts_params["workflow"] = tts_workflow
                                    if ref_audio_path:
                                        tts_params["ref_audio"] = ref_audio_path

                                audio_path = run_async(pixelle_video.tts(**tts_params))

                                if audio_path:
                                    st.success(tr("tts.preview_success"))
                                    if os.path.exists(audio_path):
                                        st.audio(audio_path, format="audio/mp3")
                                    elif audio_path.startswith("http"):
                                        st.audio(audio_path)
                                    st.caption(f"📁 {audio_path}")
                                else:
                                    st.error(tr("screen_recording.tts.preview_empty"))
                            except Exception as e:
                                st.error(tr("tts.preview_failed", error=str(e)))
                                logger.exception(e)

                bgm_files = VideoService()._list_available_bgm()
                bgm_options = [tr("bgm.none")] + bgm_files
                bgm_display = st.selectbox(
                    tr("bgm.selector"),
                    bgm_options,
                    key="screen_recording_bgm",
                )
                if bgm_display != tr("bgm.none"):
                    bgm_path = bgm_display
                    bgm_pct = st.slider(
                        tr("bgm.volume"),
                        min_value=0,
                        max_value=60,
                        value=10,
                        step=1,
                        help=tr("bgm.volume_help"),
                        key="screen_recording_bgm_volume",
                    )
                    bgm_volume = bgm_pct / 100.0

        return {
            "synthesize_dubbing": synthesize_dubbing,
            "tts_inference_mode": tts_inference_mode,
            "tts_voice": tts_voice,
            "tts_speed": tts_speed,
            "tts_workflow": tts_workflow,
            "ref_audio": ref_audio_path,
            "bgm_path": bgm_path,
            "bgm_volume": bgm_volume,
        }

    def _render_output(self, pixelle_video: Any, params: dict):
        with st.container(border=True):
            st.markdown(f"**{tr('screen_recording.output.title')}**")

            if not params.get("video_path"):
                st.info(tr("screen_recording.output.need_video_path"))
                return

            video_path = Path(params["video_path"]).expanduser()
            if video_path.is_dir():
                st.warning(tr("screen_recording.input.path_is_directory", path=video_path))
                return
            if not video_path.is_file():
                st.warning(tr("screen_recording.input.video_missing", path=video_path))
                return

            start_clicked = st.button(
                tr("screen_recording.output.start"),
                type="primary",
                use_container_width=True,
                key="screen_recording_start",
            )

            if not start_clicked:
                last_result = st.session_state.get("screen_recording_last_result")
                if last_result and last_result.get("source_video") == str(video_path.resolve()):
                    self._render_result(last_result)
                return

            if start_clicked:
                progress_bar = st.progress(0)
                status_text = st.empty()
                start = time.time()

                def on_progress(event):
                    progress_bar.progress(min(100, int(event.progress * 100)))
                    status_text.text(self._progress_text(event.event_type))

                try:
                    processor = ScreenRecordingProcessor(pixelle_video)
                    result = run_async(processor.process(progress_callback=on_progress, **params))
                    elapsed = time.time() - start
                    progress_bar.progress(100)
                    status_text.text(tr("screen_recording.progress.completed"))

                    result_state = {
                        "source_video": str(video_path.resolve()),
                        "elapsed": elapsed,
                        "video_path": result.video_path,
                        "materials_dir": result.materials_dir,
                        "materials_zip": result.materials_zip,
                        "srt_path": result.srt_path,
                        "ass_path": result.ass_path,
                    }
                    st.session_state["screen_recording_last_result"] = result_state
                    self._render_result(result_state)
                except Exception as e:
                    logger.exception(e)
                    status_text.text("")
                    progress_bar.empty()
                    st.error(tr("screen_recording.output.failed", error=e))

    def _render_result(self, result: dict):
        elapsed = result.get("elapsed")
        if elapsed is not None:
            st.success(tr("screen_recording.output.completed", elapsed=f"{elapsed:.1f}"))

        video_path = result.get("video_path")
        if video_path:
            st.markdown(f"**{tr('screen_recording.output.processed_video')}**: `{video_path}`")
            if os.path.exists(video_path):
                st.video(video_path)
                with open(video_path, "rb") as f:
                    st.download_button(
                        tr("screen_recording.output.download_video"),
                        data=f.read(),
                        file_name=Path(video_path).name,
                        mime="video/mp4",
                        use_container_width=True,
                        key="screen_recording_download_video",
                    )
            else:
                st.warning(tr("screen_recording.input.video_missing", path=video_path))

        materials_dir = result.get("materials_dir")
        if materials_dir:
            st.markdown(f"**{tr('screen_recording.output.materials_dir')}**: `{materials_dir}`")

        materials_zip = result.get("materials_zip")
        if materials_zip and os.path.exists(materials_zip):
            with open(materials_zip, "rb") as f:
                st.download_button(
                    tr("screen_recording.output.download_materials"),
                    data=f.read(),
                    file_name=Path(materials_zip).name,
                    mime="application/zip",
                    use_container_width=True,
                    key="screen_recording_download_materials",
                )
        elif materials_zip:
            st.warning(tr("screen_recording.output.file_missing", path=materials_zip))

        if result.get("srt_path"):
            st.caption(f"SRT: {result['srt_path']}")
        if result.get("ass_path"):
            st.caption(f"ASS: {result['ass_path']}")

    @staticmethod
    def _save_uploaded_file(uploaded_file, directory: Path) -> Optional[Path]:
        if uploaded_file is None:
            return None
        path = directory / uploaded_file.name
        path.write_bytes(uploaded_file.getbuffer())
        return path

    @staticmethod
    def _progress_text(event_type: str) -> str:
        return {
            "initializing": tr("screen_recording.progress.initializing"),
            "extracting_audio": tr("screen_recording.progress.extracting_audio"),
            "transcribing_audio": tr("screen_recording.progress.transcribing_audio"),
            "cleaning_subtitles": tr("screen_recording.progress.cleaning_subtitles"),
            "polishing_subtitles": tr("screen_recording.progress.polishing_subtitles"),
            "compressing_timeline": tr("screen_recording.progress.compressing_timeline"),
            "synthesizing_dubbing": tr("screen_recording.progress.synthesizing_dubbing"),
            "rendering_video": tr("screen_recording.progress.rendering_video"),
            "exporting_materials": tr("screen_recording.progress.exporting_materials"),
            "completed": tr("screen_recording.progress.completed"),
        }.get(event_type, event_type)


register_pipeline_ui(ScreenRecordingPipelineUI)
