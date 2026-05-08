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
Commentary Pipeline UI

Generates commentary videos from source video with AI script, TTS, captions,
progress bar, and cover intro.
"""

import os
import time
from pathlib import Path
from typing import Any

import streamlit as st
from loguru import logger

from web.i18n import tr, get_language
from web.pipelines.base import PipelineUI, register_pipeline_ui
from web.utils.async_helpers import run_async
from web.utils.streamlit_helpers import check_and_warn_selfhost_workflow
from pixelle_video.config import config_manager
from pixelle_video.services.subtitle_extractor import SubtitleExtractor, SubtitleDetectionResult


class CommentaryPipelineUI(PipelineUI):
    """
    UI for the Commentary Video Generation Pipeline.
    """
    name = "video_commentary"
    icon = "🎙️"

    @property
    def display_name(self):
        return tr("pipeline.video_commentary.name")

    @property
    def description(self):
        return tr("pipeline.video_commentary.description")

    def render(self, pixelle_video: Any):
        # Three-column layout
        left_col, middle_col, right_col = st.columns([1, 1, 1])

        # ====================================================================
        # Left Column: Video Path, Duration, BGM
        # ====================================================================
        with left_col:
            left_params = self._render_left_column()

        # ====================================================================
        # Middle Column: TTS Configuration
        # ====================================================================
        with middle_col:
            tts_params = self._render_tts_config(pixelle_video)

        # ====================================================================
        # Right Column: Output Preview
        # ====================================================================
        with right_col:
            video_params = {**left_params, **tts_params}
            self._render_output_preview(pixelle_video, video_params)

    def _render_left_column(self) -> dict:
        """Render left column: video path, duration, BGM."""
        with st.container(border=True):
            st.markdown(f"**{tr('commentary.source_video')}**")

            with st.expander(tr("help.feature_description"), expanded=False):
                st.markdown(f"**{tr('help.what')}**")
                st.markdown(tr("commentary.what"))
                st.markdown(f"**{tr('help.how')}**")
                st.markdown(tr("commentary.how"))

            # Video path input (local file, not upload)
            source_video = st.text_input(
                tr("commentary.video_path"),
                placeholder="/path/to/video.mp4",
                help=tr("commentary.video_path_help"),
                key="commentary_video_path"
            )

            # Check if file exists + subtitle detection
            mask_subtitles = False
            if source_video:
                if Path(source_video).exists():
                    st.success(tr("commentary.video_found"))
                    # Show video preview
                    st.video(source_video)

                    # Subtitle detection
                    with st.spinner(tr("commentary.detecting_subtitles")):
                        extractor = SubtitleExtractor()
                        detection = extractor.detect_subtitles(source_video)

                    self._render_subtitle_detection(detection)

                    # Mask toggle if hard subtitles suspected
                    if detection.hard_subtitle_warning:
                        mask_subtitles = st.checkbox(
                            tr("commentary.mask_subtitles"),
                            value=False,
                            help=tr("commentary.mask_subtitles_help"),
                            key="commentary_mask_subtitles"
                        )
                else:
                    st.error(tr("commentary.video_not_found"))

            # Segment count
            segment_count = st.number_input(
                tr("commentary.segment_count"),
                min_value=1,
                max_value=10,
                value=1,
                step=1,
                help=tr("commentary.segment_count_help"),
                key="commentary_segment_count"
            )

            # Target duration per video
            target_duration = st.slider(
                tr("commentary.target_duration"),
                min_value=60,
                max_value=900,
                value=300,
                step=30,
                format="%d秒",
                key="commentary_duration"
            )
            minutes = target_duration // 60
            seconds = target_duration % 60

            # Show per-video and total duration
            total_dur = target_duration * segment_count
            total_min = total_dur // 60
            total_sec = total_dur % 60
            if segment_count > 1:
                st.caption(tr("commentary.duration_per_segment",
                              seg_count=segment_count,
                              per_min=minutes, per_sec=seconds,
                              total_min=total_min, total_sec=total_sec))
            else:
                st.caption(tr("commentary.duration_display", minutes=minutes, seconds=seconds))

            # Audio source selection (BGM vs Original Audio - mutually exclusive)
            st.markdown(f"**{tr('commentary.audio_source')}**")

            audio_mode = st.radio(
                tr("commentary.audio_mode"),
                options=["original_audio", "bgm"],
                format_func=lambda x: tr(f"commentary.audio_mode.{x}"),
                index=0,
                horizontal=True,
                key="commentary_audio_mode"
            )

            bgm_path = None
            keep_original_audio = True
            original_audio_volume = 0.2

            if audio_mode == "bgm":
                keep_original_audio = False
                bgm_path = self._render_bgm_section()
                if not bgm_path:
                    st.warning(tr("commentary.bgm_not_selected"))
            else:
                # Original audio mode
                with st.expander(tr("commentary.original_audio_settings"), expanded=False):
                    keep_original_audio = st.checkbox(
                        tr("commentary.keep_original_audio"),
                        value=True,
                        key="commentary_keep_original"
                    )
                    if keep_original_audio:
                        original_audio_volume = st.slider(
                            tr("commentary.original_audio_volume"),
                            min_value=0.0,
                            max_value=1.0,
                            value=0.2,
                            step=0.05,
                            format="%.0f%%",
                            key="commentary_orig_volume"
                        )
                        st.caption(tr("commentary.original_audio_hint", volume=f"{int(original_audio_volume*100)}%"))

            # Initialize advanced vars with defaults
            narration_slot_ratio = 0.82
            content_start = 0.0
            content_end = None
            cover_headline = None
            cover_question = None

            # Advanced options
            with st.expander(tr("commentary.advanced"), expanded=False):
                narration_slot_ratio = st.slider(
                    tr("commentary.slot_ratio"),
                    min_value=0.55,
                    max_value=1.0,
                    value=0.82,
                    step=0.01,
                    key="commentary_slot_ratio"
                )

                content_start = st.number_input(
                    tr("commentary.content_start"),
                    min_value=0.0,
                    value=0.0,
                    step=1.0,
                    key="commentary_content_start"
                )

                content_end = st.number_input(
                    tr("commentary.content_end"),
                    min_value=0.0,
                    value=0.0,
                    step=1.0,
                    key="commentary_content_end"
                )

                if content_end == 0.0:
                    content_end = None

                cover_headline = st.text_input(
                    tr("commentary.cover_headline"),
                    placeholder=tr("commentary.cover_headline_placeholder"),
                    key="commentary_cover_headline"
                )

                cover_question = st.text_input(
                    tr("commentary.cover_question"),
                    placeholder=tr("commentary.cover_question_placeholder"),
                    key="commentary_cover_question"
                )

        return {
            "source_video": source_video,
            "target_duration": target_duration,
            "segment_count": segment_count,
            "bgm_path": bgm_path,
            "narration_slot_ratio": narration_slot_ratio,
            "content_start": content_start if content_start > 0 else None,
            "content_end": content_end,
            "cover_headline": cover_headline or None,
            "cover_question": cover_question or None,
            "mask_subtitles": mask_subtitles,
            "keep_original_audio": keep_original_audio,
            "original_audio_volume": original_audio_volume,
        }

    def _render_subtitle_detection(self, detection: SubtitleDetectionResult):
        """Render subtitle detection results in UI."""
        with st.container():
            if detection.has_external:
                files_str = ", ".join(detection.external_files)
                st.success(tr("commentary.detected_external", files=files_str))

            if detection.has_embedded:
                codecs_str = ", ".join(detection.embedded_codecs or [])
                st.success(tr("commentary.detected_embedded", count=detection.embedded_count, codecs=codecs_str))

            if not detection.has_external and not detection.has_embedded:
                st.warning(tr("commentary.no_subtitle_detected"))

            if detection.hard_subtitle_warning:
                st.info(tr("commentary.hard_subtitle_warning"))

    def _render_bgm_section(self) -> str:
        """Render BGM selection."""
        from pixelle_video.services.video import VideoService

        video_service = VideoService()
        bgm_files = video_service._list_available_bgm()

        bgm_options = [tr("bgm.none")] + bgm_files
        bgm_display = st.selectbox(
            tr("bgm.selector"),
            bgm_options,
            key="commentary_bgm"
        )

        if bgm_display == tr("bgm.none"):
            return None
        return bgm_display

    def _render_tts_config(self, pixelle_video: Any) -> dict:
        """
        Render TTS configuration (copied from style_config.py TTS section).
        Returns TTS parameters dict.
        """
        with st.container(border=True):
            st.markdown(f"**{tr('section.tts')}**")

            with st.expander(tr("help.feature_description"), expanded=False):
                st.markdown(f"**{tr('help.what')}**")
                st.markdown(tr("tts.what"))
                st.markdown(f"**{tr('help.how')}**")
                st.markdown(tr("tts.how"))

            # Get TTS config
            comfyui_config = config_manager.get_comfyui_config()
            tts_config = comfyui_config["tts"]

            # Inference mode selection
            tts_mode = st.radio(
                tr("tts.inference_mode"),
                ["local", "comfyui"],
                horizontal=True,
                format_func=lambda x: tr(f"tts.mode.{x}"),
                index=0 if tts_config.get("inference_mode", "local") == "local" else 1,
                key="commentary_tts_mode"
            )

            if tts_mode == "local":
                st.caption(tr("tts.mode.local_hint"))
            else:
                st.caption(tr("tts.mode.comfyui_hint"))

            # ================================================================
            # Local Mode
            # ================================================================
            if tts_mode == "local":
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
                        key="commentary_tts_voice"
                    )
                    selected_voice_index = voice_options.index(selected_voice_display)
                    selected_voice = voice_ids[selected_voice_index]

                with speed_col:
                    tts_speed = st.slider(
                        tr("tts.speed"),
                        min_value=0.5,
                        max_value=2.0,
                        value=saved_speed,
                        step=0.1,
                        format="%.1fx",
                        key="commentary_tts_speed"
                    )
                    st.caption(tr("tts.speed_label", speed=f"{tts_speed:.1f}"))

                # Default commentary uses +18% rate for fast narration
                tts_rate = "+18%"
                tts_workflow_key = None
                ref_audio_path = None

            # ================================================================
            # ComfyUI Mode
            # ================================================================
            else:
                tts_workflows = pixelle_video.tts.list_workflows()
                tts_workflow_options = [wf["display_name"] for wf in tts_workflows]
                tts_workflow_keys = [wf["key"] for wf in tts_workflows]

                default_tts_index = 0
                saved_tts_workflow = tts_config.get("comfyui", {}).get("default_workflow")
                if saved_tts_workflow and saved_tts_workflow in tts_workflow_keys:
                    default_tts_index = tts_workflow_keys.index(saved_tts_workflow)

                tts_workflow_display = st.selectbox(
                    "TTS Workflow",
                    tts_workflow_options if tts_workflow_options else ["No TTS workflows found"],
                    index=default_tts_index,
                    label_visibility="collapsed",
                    key="commentary_tts_workflow"
                )

                if tts_workflow_options:
                    tts_selected_index = tts_workflow_options.index(tts_workflow_display)
                    tts_workflow_key = tts_workflow_keys[tts_selected_index]
                else:
                    tts_workflow_key = "selfhost/tts_edge.json"

                check_and_warn_selfhost_workflow(tts_workflow_key)

                ref_audio_file = st.file_uploader(
                    tr("tts.ref_audio"),
                    type=["mp3", "wav", "flac", "m4a", "aac", "ogg"],
                    help=tr("tts.ref_audio_help"),
                    key="commentary_ref_audio"
                )

                ref_audio_path = None
                if ref_audio_file is not None:
                    temp_dir = Path("temp")
                    temp_dir.mkdir(exist_ok=True)
                    ref_audio_path = temp_dir / f"ref_audio_{ref_audio_file.name}"
                    with open(ref_audio_path, "wb") as f:
                        f.write(ref_audio_file.getbuffer())

                selected_voice = None
                tts_speed = None
                tts_rate = "+0%"

            # ================================================================
            # Jianying Material Export
            # ================================================================
            st.markdown("---")
            st.markdown(f"**{tr('commentary.jianying.title')}**")
            jianying_export = st.checkbox(
                tr("commentary.jianying.enable"),
                value=False,
                help=tr("commentary.jianying.help"),
                key="commentary_jianying_export"
            )

            # ================================================================
            # Bilibili Upload Settings
            # ================================================================
            st.markdown("---")
            st.markdown(f"**{tr('commentary.bilibili.title')}**")

            bili_config = config_manager.get_bilibili_config()
            bili_upload = st.checkbox(
                tr("commentary.bilibili.enable"),
                value=False,
                key="commentary_bili_upload"
            )

            bili_video_title = ""
            bili_extra_tags = ""
            bili_tid = 228
            bili_copyright = 1
            bili_cookie_path = ""

            if bili_upload:
                st.markdown(f"**{tr('commentary.bilibili.cookie_step1')}**")
                st.caption(tr("commentary.bilibili.cookie_step1_desc"))

                # Cookie file uploader
                st.markdown(f"**{tr('commentary.bilibili.cookie_step2')}**")
                cookie_file = st.file_uploader(
                    tr("commentary.bilibili.cookie_upload"),
                    type=["json"],
                    help=tr("commentary.bilibili.cookie_upload_help"),
                    key="commentary_bili_cookie_upload"
                )

                bili_cookie_path = ""
                if cookie_file is not None:
                    temp_dir = Path("temp") / "bilibili_cookies"
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    cookie_path = temp_dir / f"cookies_{cookie_file.name}"
                    with open(cookie_path, "wb") as f:
                        f.write(cookie_file.getbuffer())
                    bili_cookie_path = str(cookie_path)
                    st.success(tr("commentary.bilibili.cookie_uploaded", path=bili_cookie_path))
                else:
                    # Fallback: text input for server-local path
                    bili_cookie_path = st.text_input(
                        tr("commentary.bilibili.cookie_path_local"),
                        value=st.session_state.get("bili_cookie_path", ""),
                        placeholder="/path/to/cookies.json",
                        help=tr("commentary.bilibili.cookie_path_local_help"),
                        key="commentary_bili_cookie_path"
                    )
                    if bili_cookie_path:
                        st.session_state["bili_cookie_path"] = bili_cookie_path

                bili_video_title = st.text_input(
                    tr("commentary.bilibili.video_title"),
                    placeholder=tr("commentary.bilibili.video_title_placeholder"),
                    key="commentary_bili_title"
                )
                bili_extra_tags = st.text_input(
                    tr("commentary.bilibili.extra_tags"),
                    placeholder=tr("commentary.bilibili.extra_tags_placeholder"),
                    help=tr("commentary.bilibili.extra_tags_help"),
                    key="commentary_bili_tags"
                )
                # TID options
                tid_options = {
                    228: "电影", 230: "电视剧", 231: "计算机技术",
                    232: "软件应用", 171: "电子竞技", 172: "单机游戏",
                    31: "音乐综合", 28: "原创音乐", 160: "搞笑"
                }
                tid_labels = [f"{tid} - {name}" for tid, name in tid_options.items()]
                tid_values = list(tid_options.keys())
                default_tid_index = tid_values.index(bili_config.get("default_tid", 228)) if bili_config.get("default_tid", 228) in tid_values else 0
                selected_tid_label = st.selectbox(
                    tr("commentary.bilibili.tid"),
                    tid_labels,
                    index=default_tid_index,
                    key="commentary_bili_tid"
                )
                bili_tid = tid_values[tid_labels.index(selected_tid_label)]

                bili_copyright = st.radio(
                    tr("commentary.bilibili.copyright"),
                    options=[1, 2],
                    format_func=lambda x: tr("commentary.bilibili.copyright_original") if x == 1 else tr("commentary.bilibili.copyright_reprint"),
                    index=0 if bili_config.get("default_copyright", 1) == 1 else 1,
                    horizontal=True,
                    key="commentary_bili_copyright"
                )

            # ================================================================
            # TTS Preview
            # ================================================================
            with st.expander(tr("tts.preview_title"), expanded=False):
                preview_text = st.text_input(
                    tr("tts.preview_text"),
                    value="大家好，这是一段测试语音。",
                    placeholder=tr("tts.preview_text_placeholder"),
                    key="commentary_preview_text"
                )

                if st.button(tr("tts.preview_button"), key="commentary_preview_tts", use_container_width=True):
                    with st.spinner(tr("tts.previewing")):
                        try:
                            tts_params = {
                                "text": preview_text,
                                "inference_mode": tts_mode
                            }
                            if tts_mode == "local":
                                tts_params["voice"] = selected_voice
                                tts_params["speed"] = tts_speed
                            else:
                                tts_params["workflow"] = tts_workflow_key
                                if ref_audio_path:
                                    tts_params["ref_audio"] = str(ref_audio_path)

                            audio_path = run_async(pixelle_video.tts(**tts_params))

                            if audio_path:
                                st.success(tr("tts.preview_success"))
                                if os.path.exists(audio_path):
                                    st.audio(audio_path, format="audio/mp3")
                                elif audio_path.startswith("http"):
                                    st.audio(audio_path)
                                st.caption(f"📁 {audio_path}")
                            else:
                                st.error("Failed to generate preview audio")
                        except Exception as e:
                            st.error(tr("tts.preview_failed", error=str(e)))
                            logger.exception(e)

        return {
            "tts_inference_mode": tts_mode,
            "tts_voice": selected_voice if tts_mode == "local" else None,
            "tts_speed": tts_speed if tts_mode == "local" else None,
            "tts_rate": tts_rate if tts_mode == "local" else "+0%",
            "tts_workflow": tts_workflow_key if tts_mode == "comfyui" else None,
            "ref_audio": str(ref_audio_path) if ref_audio_path else None,
            "export_jianying_materials": jianying_export,
            "bili_upload": bili_upload,
            "bili_cookie_path": bili_cookie_path,
            "bili_video_title": bili_video_title,
            "bili_extra_tags": bili_extra_tags,
            "bili_tid": bili_tid,
            "bili_copyright": bili_copyright,
        }

    def _render_output_preview(self, pixelle_video: Any, video_params: dict):
        """Render output preview section (right column)."""
        with st.container(border=True):
            st.markdown(f"**{tr('section.video_generation')}**")

            if not config_manager.validate():
                st.warning(tr("settings.not_configured"))

            source_video = video_params.get("source_video", "")
            target_duration = video_params.get("target_duration", 300)

            # Validation
            if not source_video:
                st.info(tr("commentary.video_path_hint"))
                st.button(
                    tr("btn.generate"),
                    type="primary",
                    use_container_width=True,
                    disabled=True,
                    key="commentary_generate_disabled_no_video"
                )
                return

            if not Path(source_video).exists():
                st.error(tr("commentary.video_not_found"))
                st.button(
                    tr("btn.generate"),
                    type="primary",
                    use_container_width=True,
                    disabled=True,
                    key="commentary_generate_disabled_not_found"
                )
                return

            # Show summary
            st.info(tr("commentary.summary",
                       path=source_video,
                       duration=target_duration))

            # Generate button
            if st.button(tr("btn.generate"), type="primary", use_container_width=True, key="commentary_generate"):
                if not config_manager.validate():
                    st.error(tr("settings.not_configured"))
                    st.stop()

                progress_bar = st.progress(0)
                status_text = st.empty()
                start_time = time.time()

                try:
                    async def generate_commentary_video():
                        status_text.text(tr("progress.generation"))
                        progress_bar.progress(5)

                        pipeline_params = {
                            "source_video": video_params["source_video"],
                            "target_duration": video_params["target_duration"],
                            "segment_count": video_params.get("segment_count", 1),
                            "tts_voice": video_params.get("tts_voice", "zh-CN-YunxiNeural"),
                            "tts_rate": video_params.get("tts_rate", "+18%"),
                            "narration_slot_ratio": video_params.get("narration_slot_ratio", 0.82),
                            "bgm_path": video_params.get("bgm_path"),
                            "content_start": video_params.get("content_start"),
                            "content_end": video_params.get("content_end"),
                            "cover_headline": video_params.get("cover_headline"),
                            "cover_question": video_params.get("cover_question"),
                            "mask_subtitles": video_params.get("mask_subtitles", False),
                            "keep_original_audio": video_params.get("keep_original_audio", True),
                            "original_audio_volume": video_params.get("original_audio_volume", 0.2),
                            "export_jianying_materials": video_params.get("export_jianying_materials", False),
                        }

                        def progress_callback(event):
                            progress_bar.progress(min(int(event.progress * 100), 99))
                            status_text.text(event.event_type)

                        result = await pixelle_video.pipelines["commentary"](
                            text="",
                            progress_callback=progress_callback,
                            **pipeline_params,
                        )

                        progress_bar.progress(100)
                        status_text.text(tr("status.success"))
                        return result

                    result = run_async(generate_commentary_video())
                    total_time = time.time() - start_time

                    # Display result
                    all_paths = [result.video_path] + list(getattr(result, "additional_video_paths", []) or [])
                    if len(all_paths) > 1:
                        st.success(f"✅ 已生成 {len(all_paths)} 个独立视频文件" if get_language() == "zh_CN" else f"✅ Generated {len(all_paths)} independent video files")
                    else:
                        st.success(tr("status.video_generated", path=result.video_path))
                    st.markdown("---")

                    # Show all videos with download buttons
                    for idx, vp in enumerate(all_paths):
                        if os.path.exists(vp):
                            file_size_mb = os.path.getsize(vp) / (1024 * 1024)
                            seg_label = f"第 {idx+1} 段" if get_language() == "zh_CN" else f"Segment {idx+1}"
                            st.markdown(f"**{seg_label}** — `{os.path.basename(vp)}` ({file_size_mb:.1f}MB)")
                            st.video(vp)
                            with open(vp, "rb") as video_file:
                                video_bytes = video_file.read()
                                video_filename = os.path.basename(vp)
                                st.download_button(
                                    label=f"⬇️ {seg_label}" if get_language() == "zh_CN" else f"⬇️ {seg_label}",
                                    data=video_bytes,
                                    file_name=video_filename,
                                    mime="video/mp4",
                                    use_container_width=True,
                                    key=f"commentary_download_{idx}"
                                )
                            st.markdown("---")
                        else:
                            st.error(tr("status.video_not_found", path=vp))

                    # Summary info
                    total_size_mb = sum(os.path.getsize(p) / (1024 * 1024) for p in all_paths if os.path.exists(p))
                    info_text = (
                        f"⏱️ {tr('info.generation_time')} {total_time:.1f}s   "
                        f"📦 {total_size_mb:.1f}MB ({len(all_paths)} files)"
                    )
                    st.caption(info_text)

                    # ================================================================
                    # Jianying Materials Export Info
                    # ================================================================
                    if video_params.get("export_jianying_materials"):
                        st.markdown("---")
                        st.success(tr("commentary.jianying.exported"))
                        st.markdown(tr("commentary.jianying.import_hint"))

                    # ================================================================
                    # Bilibili Upload
                    # ================================================================
                    if video_params.get("bili_upload"):
                        st.markdown("---")
                        st.markdown(f"**{tr('commentary.bilibili.title')}**")

                        cookie_path = video_params.get("bili_cookie_path", "")

                        if not cookie_path:
                            st.error(tr("commentary.bilibili.upload_failed", error="Bilibili cookie path not provided"))
                        elif not Path(cookie_path).exists():
                            st.error(tr("commentary.bilibili.upload_failed", error=f"Cookie file not found: {cookie_path}"))
                        else:
                            for idx, vp in enumerate(all_paths):
                                if not os.path.exists(vp):
                                    continue
                                seg_label = f"第 {idx+1} 段" if get_language() == "zh_CN" else f"Segment {idx+1}"
                                with st.spinner(tr("commentary.bilibili.uploading") + f" ({seg_label})"):
                                    try:
                                        from pixelle_video.services.bilibili_uploader import BilibiliUploader
                                        uploader = BilibiliUploader(cookie_path=cookie_path)

                                        title = video_params.get("bili_video_title", "") or Path(vp).stem
                                        extra_tags = video_params.get("bili_extra_tags", "")
                                        tid = video_params.get("bili_tid", 228)
                                        copyright_type = video_params.get("bili_copyright", 1)

                                        bvid = uploader.upload(
                                            video_path=vp,
                                            title=title,
                                            extra_tags=extra_tags,
                                            tid=tid,
                                            copyright=copyright_type,
                                        )
                                        st.success(tr("commentary.bilibili.upload_success", bvid=bvid))
                                    except Exception as e:
                                        logger.exception(e)
                                        st.error(tr("commentary.bilibili.upload_failed", error=str(e)))

                except Exception as e:
                    logger.exception(e)
                    status_text.text("")
                    progress_bar.empty()
                    st.error(tr("status.error", error=str(e)))
                    st.stop()


register_pipeline_ui(CommentaryPipelineUI)
