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
Image-to-Video Pipeline UI

Generates videos from user-provided images with optional TTS voiceover.
"""

import os
import time
import json
from pathlib import Path
from typing import Any

import streamlit as st
from loguru import logger
import httpx
from web.i18n import tr, get_language
from web.pipelines.base import PipelineUI, register_pipeline_ui
from web.utils.async_helpers import run_async
from web.utils.streamlit_helpers import check_and_warn_selfhost_workflow
from pixelle_video.config import config_manager
from pixelle_video.utils.os_util import create_task_output_dir


class ImageToVideoPipelineUI(PipelineUI):
    """UI for the Image To Video Generation Pipeline."""

    name = "image_to_video"
    icon = "🎥"

    @property
    def display_name(self):
        return tr("pipeline.i2v.name")

    @property
    def description(self):
        return tr("pipeline.i2v.description")

    # ========================================================================
    # Main render: 3-column layout
    # ========================================================================

    def render(self, pixelle_video: Any):
        left_col, middle_col, right_col = st.columns([1, 1, 1])

        with left_col:
            left_params = self._render_left_column()

        with middle_col:
            tts_params = self._render_tts_config(pixelle_video)

        with right_col:
            video_params = {**left_params, **tts_params}
            self._render_output_preview(pixelle_video, video_params)

    # ========================================================================
    # Left Column: Image upload + prompt + workflow + duration
    # ========================================================================

    def _render_left_column(self) -> dict:
        with st.container(border=True):
            st.markdown(f"**{tr('i2v.video_generation')}**")

            with st.expander(tr("help.feature_description"), expanded=False):
                st.markdown(f"**{tr('help.what')}**")
                st.markdown(tr("i2v.assets.image_what"))
                st.markdown(f"**{tr('help.how')}**")
                st.markdown(tr("i2v.assets.how"))

            # File uploader
            uploaded_files = st.file_uploader(
                tr("i2v.assets.upload"),
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True,
                help=tr("i2v.assets.upload_help"),
                key="material_files"
            )

            audio_asset_paths = []
            if uploaded_files:
                import uuid
                session_id = str(uuid.uuid4()).replace('-', '')[:12]
                temp_dir = Path(f"temp/assets_{session_id}")
                temp_dir.mkdir(parents=True, exist_ok=True)

                for uploaded_file in uploaded_files:
                    file_path = temp_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    audio_asset_paths.append(str(file_path.absolute()))

                st.success(tr("i2v.assets.character_sucess"))

                with st.expander(tr("i2v.assets.preview"), expanded=True):
                    cols = st.columns(3)
                    for i, (file, path) in enumerate(zip(uploaded_files, audio_asset_paths)):
                        with cols[i % 3]:
                            ext = Path(path).suffix.lower()
                            if ext in [".jpg", ".jpeg", ".png", ".webp"]:
                                st.image(file, caption=file.name, use_container_width=True)
            else:
                st.info(tr("i2v.assets.character_empty_hint"))

            # Prompt
            prompt_text = st.text_area(
                tr("i2v.input_text"),
                placeholder=tr("i2v.input.topic_placeholder"),
                height=150,
                help=tr("input.text_help_audio"),
                key="audio_box"
            )
            st.caption(tr("i2v.prompt_hint"))

            # Workflow selector
            i2v_workflows = _list_i2v_workflows()
            workflow_options = [wf["display_name"] for wf in i2v_workflows]
            workflow_keys = [wf["key"] for wf in i2v_workflows]
            default_workflow_index = 0

            workflow_display = st.selectbox(
                tr("i2v.workflow_select"),
                workflow_options if workflow_options else ["No workflow found"],
                index=default_workflow_index,
                label_visibility="collapsed",
                key="i2v_workflow_select"
            )

            if workflow_options:
                workflow_selected_index = workflow_options.index(workflow_display)
                workflow_key = workflow_keys[workflow_selected_index]
            else:
                workflow_key = None

            check_and_warn_selfhost_workflow(workflow_key)

            # Duration
            duration = st.slider(
                tr("i2v.duration"),
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                format="%d秒",
                help=tr("i2v.duration_help"),
                key="i2v_duration"
            )
            st.caption(tr("i2v.duration_caption", seconds=duration))

        return {
            "audio_assets": audio_asset_paths,
            "prompt_text": prompt_text,
            "workflow_key": workflow_key,
            "duration": duration,
        }

    # ========================================================================
    # Middle Column: TTS config (matching commentary style)
    # ========================================================================

    def _render_tts_config(self, pixelle_video: Any) -> dict:
        """Render TTS configuration, styled consistently with commentary pipeline."""
        with st.container(border=True):
            st.markdown(f"**{tr('section.tts')}**")

            with st.expander(tr("help.feature_description"), expanded=False):
                st.markdown(f"**{tr('help.what')}**")
                st.markdown(tr("tts.what"))
                st.markdown(f"**{tr('help.how')}**")
                st.markdown(tr("tts.how"))

            enable_tts = st.checkbox(
                tr("i2v.enable_tts"),
                value=False,
                help=tr("i2v.enable_tts_help"),
                key="i2v_enable_tts"
            )

            tts_text = ""
            tts_voice = "zh-CN-YunxiNeural"
            tts_rate = "+0%"

            if enable_tts:
                tts_text = st.text_area(
                    tr("i2v.tts_text"),
                    placeholder=tr("i2v.tts_text_placeholder"),
                    height=120,
                    key="i2v_tts_text"
                )
                if not tts_text.strip():
                    st.caption(tr("i2v.tts_auto_hint"))

                from pixelle_video.tts_voices import EDGE_TTS_VOICES, get_voice_display_name
                voice_options = []
                voice_ids = []
                for vc in EDGE_TTS_VOICES:
                    voice_options.append(get_voice_display_name(vc["id"], tr, get_language()))
                    voice_ids.append(vc["id"])

                default_idx = voice_ids.index("zh-CN-YunxiNeural") if "zh-CN-YunxiNeural" in voice_ids else 0

                col_v, col_r = st.columns([1, 1])
                with col_v:
                    selected_display = st.selectbox(
                        tr("i2v.tts_voice"),
                        voice_options,
                        index=default_idx,
                        key="i2v_tts_voice_select"
                    )
                    tts_voice = voice_ids[voice_options.index(selected_display)]

                with col_r:
                    tts_speed = st.slider(
                        tr("i2v.tts_rate"),
                        min_value=0.5,
                        max_value=2.0,
                        value=1.0,
                        step=0.1,
                        format="%.1fx",
                        key="i2v_tts_rate_slider"
                    )
                    tts_rate = f"{int((tts_speed - 1.0) * 100):+d}%"
                    st.caption(tr("tts.speed_label", speed=tts_speed))

                # TTS preview
                with st.expander(tr("tts.preview_title"), expanded=False):
                    preview_text = st.text_input(
                        tr("tts.preview_text"),
                        value="大家好，这是一段测试语音。",
                        key="i2v_preview_text"
                    )
                    if st.button(tr("tts.preview_button"), key="i2v_preview_tts", use_container_width=True):
                        with st.spinner(tr("tts.previewing")):
                            try:
                                import edge_tts
                                preview_path = Path("temp") / "i2v_tts_preview.mp3"
                                preview_path.parent.mkdir(exist_ok=True)
                                comm = edge_tts.Communicate(preview_text, tts_voice, rate=tts_rate)
                                run_async(comm.save(str(preview_path)))
                                st.audio(str(preview_path), format="audio/mp3")
                                st.caption(f"📁 {preview_path}")
                            except Exception as e:
                                st.error(str(e))

        return {
            "enable_tts": enable_tts,
            "tts_text": tts_text,
            "tts_voice": tts_voice,
            "tts_rate": tts_rate,
        }

    # ========================================================================
    # Right Column: Output Preview
    # ========================================================================

    def _render_output_preview(self, pixelle_video: Any, video_params: dict):
        with st.container(border=True):
            st.markdown(f"**{tr('section.video_generation')}**")

            if not config_manager.validate():
                st.warning(tr("settings.not_configured"))

            audio_assets = video_params.get("audio_assets", [])
            prompt_text = video_params.get("prompt_text", "")
            workflow_key = video_params.get("workflow_key")

            if not audio_assets:
                st.info(tr("i2v.assets.image_warning"))
                st.button(tr("btn.generate"), type="primary", use_container_width=True,
                          disabled=True, key="i2v_gen_disabled_img")
                return

            if not prompt_text:
                st.info(tr("i2v.assets.prompt_warning"))
                st.button(tr("btn.generate"), type="primary", use_container_width=True,
                          disabled=True, key="i2v_gen_disabled_prompt")
                return

            # Show summary
            duration = video_params.get("duration", 5)
            enable_tts = video_params.get("enable_tts", False)
            tts_text = video_params.get("tts_text", "")
            summary_parts = [f"🎞️ {len(audio_assets)} 张图片" if get_language() == "zh_CN" else f"🎞️ {len(audio_assets)} images",
                             f"⏱️ {duration}s"]
            if enable_tts and tts_text:
                summary_parts.append("🔊 配音" if get_language() == "zh_CN" else "🔊 Voiceover")
            st.info(" | ".join(summary_parts))

            # Generate button
            if st.button(tr("btn.generate"), type="primary", use_container_width=True, key="i2v_generate"):
                if not config_manager.validate():
                    st.error(tr("settings.not_configured"))
                    st.stop()

                progress_bar = st.progress(0)
                status_text = st.empty()
                start_time = time.time()

                try:
                    final_video_path, task_id = run_async(
                        _generate_i2v_video(pixelle_video, video_params, progress_bar, status_text)
                    )
                    total_time = time.time() - start_time

                    # Save to history
                    try:
                        async def _save_history():
                            metadata = {
                                "task_id": task_id,
                                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                "status": "completed",
                                "pipeline": "i2v",
                                "input": {
                                    "title": Path(audio_assets[0]).stem if audio_assets else "i2v",
                                    "text": prompt_text,
                                    "duration": duration,
                                    "enable_tts": enable_tts,
                                },
                                "result": {
                                    "video_path": final_video_path,
                                    "video_paths": [final_video_path],
                                    "duration": duration,
                                    "file_size": os.path.getsize(final_video_path),
                                },
                            }
                            await pixelle_video.persistence.save_task_metadata(task_id, metadata)
                        run_async(_save_history())
                        logger.info(f"💾 i2v task saved: {task_id}")
                    except Exception as e:
                        logger.warning(f"Failed to save i2v history: {e}")

                    progress_bar.progress(100)
                    status_text.text(tr("status.success"))
                    st.success(tr("status.video_generated", path=final_video_path))
                    st.markdown("---")

                    if os.path.exists(final_video_path):
                        file_size_mb = os.path.getsize(final_video_path) / (1024 * 1024)
                        st.caption(f"⏱️ {total_time:.1f}s | 📦 {file_size_mb:.2f}MB")
                        st.markdown("---")
                        st.video(final_video_path)
                        with open(final_video_path, "rb") as vf:
                            st.download_button(
                                label="⬇️ 下载视频" if get_language() == "zh_CN" else "⬇️ Download Video",
                                data=vf.read(),
                                file_name=os.path.basename(final_video_path),
                                mime="video/mp4",
                                use_container_width=True
                            )
                    else:
                        st.error(tr("status.video_not_found", path=final_video_path))

                except Exception as e:
                    logger.exception(e)
                    status_text.text("")
                    progress_bar.empty()
                    st.error(str(e))
                    st.stop()


# ========================================================================
# Async generator (module-level to avoid pickling issues)
# ========================================================================

async def _generate_i2v_video(pixelle_video, video_params, progress_bar, status_text):
    task_dir, task_id = create_task_output_dir()
    logger.info(f"[i2v] Task Directory: {task_dir}")

    status_text.text(tr("progress.generation"))
    progress_bar.progress(10)

    audio_assets = video_params.get("audio_assets", [])
    prompt = video_params.get("prompt_text", "")
    workflow_key = video_params.get("workflow_key")
    duration = video_params.get("duration", 5)
    image_path = audio_assets[0]

    workflow_path = Path("workflows") / workflow_key
    if not workflow_path.exists():
        raise Exception(f"Workflow not found: {workflow_path}")

    with open(workflow_path, 'r', encoding='utf-8') as f:
        workflow_config = json.load(f)

    # --- Dashscope path: pass duration directly ---
    if workflow_config.get("source") == "dashscope":
        media_result = await pixelle_video.media(
            prompt=prompt,
            workflow=workflow_key,
            media_type="video",
            image=image_path,
            duration=duration,
        )
        generated_video_url = media_result.url

    # --- RunningHub / Selfhost path ---
    else:
        kit = await pixelle_video._get_or_create_comfykit()
        workflow_params = {
            "image": image_path,
            "prompt": prompt,
            "duration": duration,
        }

        if workflow_config.get("source") == "runninghub" and "workflow_id" in workflow_config:
            workflow_input = workflow_config["workflow_id"]
        else:
            workflow_input = str(workflow_path)

        video_result = await kit.execute(workflow_input, workflow_params)

        generated_video_url = None
        if hasattr(video_result, 'videos') and video_result.videos:
            generated_video_url = video_result.videos[0]
        elif hasattr(video_result, 'outputs') and video_result.outputs:
            for node_id, node_output in video_result.outputs.items():
                if isinstance(node_output, dict) and 'videos' in node_output:
                    videos = node_output['videos']
                    if videos:
                        generated_video_url = videos[0]
                        break

    if not generated_video_url:
        raise Exception("The workflow did not return a video. Please check the workflow configuration.")

    # Download video
    final_video_path = os.path.join(task_dir, "final.mp4")
    timeout = httpx.Timeout(300.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(generated_video_url)
        response.raise_for_status()
        with open(final_video_path, 'wb') as f:
            f.write(response.content)

    progress_bar.progress(70)

    # ================================================================
    # Optional TTS voiceover (auto-generate narration if needed)
    # ================================================================
    enable_tts = video_params.get("enable_tts", False)
    tts_text = video_params.get("tts_text", "").strip()

    if enable_tts:
        # Auto-generate narration if user didn't provide one
        if not tts_text:
            status_text.text("🤖 AI generating narration...")
            progress_bar.progress(75)
            try:
                narration_prompt = (
                    f"根据以下视频提示词，生成一段适合 {duration} 秒视频的简短中文旁白（30-80字），"
                    f"语气自然口语化，适合 TTS 朗读。只输出旁白文本，不要其他内容。\n\n"
                    f"视频提示词：{prompt}"
                )
                tts_text = await pixelle_video.llm(
                    prompt=narration_prompt,
                    temperature=0.8,
                    max_tokens=200,
                )
                tts_text = tts_text.strip()
                logger.info(f"🤖 Auto-generated narration ({len(tts_text)} chars): {tts_text[:80]}...")
            except Exception as e:
                logger.warning(f"AI narration generation failed: {e}, skipping voiceover")
                tts_text = ""

    if enable_tts and tts_text:
        status_text.text("🎙️ Generating voiceover...")
        try:
            import edge_tts
            tts_voice = video_params.get("tts_voice", "zh-CN-YunxiNeural")
            tts_rate = video_params.get("tts_rate", "+0%")
            tts_path = os.path.join(task_dir, "voiceover.mp3")
            comm = edge_tts.Communicate(tts_text, tts_voice, rate=tts_rate)
            await comm.save(tts_path)

            # Get video duration
            import subprocess
            probe = subprocess.run([
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                final_video_path
            ], capture_output=True, text=True, check=True)
            video_dur = float(probe.stdout.strip())

            # Mix: replace audio with TTS
            mixed_path = os.path.join(task_dir, "final_mixed.mp4")
            subprocess.run([
                "ffmpeg", "-hide_banner", "-y",
                "-i", final_video_path,
                "-stream_loop", "-1", "-i", tts_path,
                "-t", str(video_dur),
                "-c:v", "copy",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                mixed_path
            ], check=True)
            os.replace(mixed_path, final_video_path)
            logger.info("✅ i2v voiceover mixed")
        except Exception as e:
            logger.warning(f"TTS voiceover failed, keeping silent video: {e}")

    progress_bar.progress(100)
    status_text.text(tr("status.success"))
    return final_video_path, task_id


# ========================================================================
# Helpers
# ========================================================================

def _list_i2v_workflows():
    result = []
    for source in ("dashscope", "runninghub", "selfhost"):
        dir_path = os.path.join("workflows", source)
        if not os.path.isdir(dir_path):
            continue
        for fname in sorted(os.listdir(dir_path)):
            if fname.startswith("i2v_") and fname.endswith(".json"):
                source_display = {
                    "dashscope": "阿里云百炼/通义万相" if get_language() == "zh_CN" else "Alibaba Bailian / Tongyi Wanxiang",
                    "runninghub": tr("asset_based.source.runninghub"),
                    "selfhost": tr("asset_based.source.selfhost"),
                }.get(source, source.title())
                result.append({
                    "key": f"{source}/{fname}",
                    "display_name": f"{fname} - {source_display}"
                })
    return sorted(result, key=lambda item: item["key"])


register_pipeline_ui(ImageToVideoPipelineUI)
