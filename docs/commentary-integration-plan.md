# Pixelle-Video「视频解说」功能开发方案（v2）

> 目标：将 `video-commentary-automation` 的影视解说能力集成到 Pixelle-Video 中，作为一个新的 Pipeline UI Tab。

---

## 一、需求概述

### 输入
- 原视频本地路径（不传输，只验证存在）
- 目标解说时长（默认 5 分钟，1-15 分钟可调）
- 配音合成配置（复用现有 TTS 选择器）
- BGM（可选）

### 输出
- 带解说、字幕、封面、进度条的解说视频（端到端全自动）

### 核心能力
1. **自动剧情理解**：检测同目录字幕 → 提取内嵌字幕流 → 视频帧分析兜底
2. **AI 生成解说脚本**：根据剧情+目标时长自动分 chunk，每个 chunk 分配对应原视频时间窗口
3. **AI 生成封面**：LLM 生成封面 prompt → AI 生图作为 3 秒封面背景
4. **精确语音合成**：TTS + atempo 调速链，精确匹配 chunk 时间槽
5. **视频合成**：原片抽 clip → 混音+ASS 字幕 → 进度条 → 封面 intro

---

## 二、全自动剧情理解策略

用户只提供原视频本地路径，后端按以下优先级全自动获取剧情文本：

```
Priority 1: 同目录同名字幕文件（.srt / .ass / .vtt / .txt）
Priority 2: ffprobe 检测内嵌字幕流 → ffmpeg 提取为文本+时间戳
Priority 3: VideoAnalysisService（抽首帧+LLM vision 分析，适合短视频兜底）
Priority 4: 全部失败 → 报错提示
```

- **有时间戳的字幕** → LLM 直接按时间戳分配 `source_windows`
- **纯文本无时间戳** → LLM 按剧情密度均匀分配 `source_windows`

---

## 三、技术架构

```
Web UI (Streamlit)
├─ 左栏：视频路径、目标时长、BGM、封面预览
├─ 中栏：TTS 配置（local/ComfyUI、音色、语速、预览）
└─ 右栏：生成按钮、进度、视频预览/下载
         │
         ▼
CommentaryPipeline (BasePipeline)
1. setup_environment      → 验证视频路径，创建 task_dir
2. extract_story          → SubtitleExtractor 自动字幕检测/提取
3. generate_script        → LLM 生成 CommentaryScript
4. generate_cover_image   → media_service AI 生图（封面背景）
5. synthesize_voiceovers  → tts_service + atempo 精确调速
6. render_clips           → ffmpeg 按 source_windows 抽 clip
7. compose_video          → 合成+混音+ASS 字幕
8. add_progress_bar       → 顶部进度条 overlay
9. add_cover_intro        → 3秒 AI 封面 intro
10. finalize              → 持久化 metadata，返回结果
```

---

## 四、低侵入性修改清单

| 文件 | 修改内容 | 行数 |
|------|----------|------|
| `pixelle_video/service.py` | `self.pipelines` 加 `"commentary": CommentaryPipeline(self)` | +1 |
| `web/pipelines/__init__.py` | 末尾加 `from web.pipelines import commentary` | +1 |
| `api/app.py` | import + `app.include_router(commentary_router)` | +2 |
| `web/i18n/locales/zh_CN.json` | 添加 commentary 翻译词条 | +~20 |
| `web/i18n/locales/en_US.json` | 添加 commentary 翻译词条 | +~20 |

其余全部是**新增文件**，不影响现有功能。

---

## 五、开发 TODO（按实施顺序）

### Phase 1：数据模型与 Prompt

- [ ] **TODO-1** `pixelle_video/models/commentary.py` — CommentaryChunk, CommentaryCover, CommentaryScript Pydantic 模型
- [ ] **TODO-2** `pixelle_video/prompts/commentary_script.py` — build_commentary_prompt()，LLM structured output 生成解说脚本

### Phase 2：核心服务

- [ ] **TODO-3** `pixelle_video/services/subtitle_extractor.py` — 自动字幕检测/提取/解析
- [ ] **TODO-4** `pixelle_video/services/commentary_compositor.py` — 视频合成器（抽clip、TTS+atempo、ASS字幕、进度条、封面intro）

### Phase 3：Pipeline

- [ ] **TODO-5** `pixelle_video/pipelines/commentary.py` — CommentaryPipeline 端到端流程

### Phase 4：API 层

- [ ] **TODO-6** `api/schemas/commentary.py` — Request/Response schemas
- [ ] **TODO-7** `api/routers/commentary.py` — sync/async endpoints
- [ ] **TODO-8** 修改 `api/app.py` — include_router

### Phase 5：Web UI

- [ ] **TODO-9** `web/pipelines/commentary.py` — CommentaryPipelineUI 三栏布局
- [ ] **TODO-10** 修改 `web/pipelines/__init__.py` — 导入 commentary
- [ ] **TODO-11** 修改 `pixelle_video/service.py` — 注册 pipeline
- [ ] **TODO-12** 修改 `web/i18n/locales/zh_CN.json` + `en_US.json` — 翻译词条

---

## 六、关键设计决策

### 1. 为什么 CommentaryPipeline 不继承 LinearVideoPipeline？
LinearVideoPipeline 的核心假设是**逐帧 HTML Template + FrameProcessor 渲染**。视频解说的生产方式是**剪辑合成（抽 clip + 混音 + overlay）**，没有逐帧 HTML 渲染。继承 BasePipeline 更轻量。

### 2. 封面 AI 生图如何接入？
在 `generate_script` 阶段，LLM 输出 `CommentaryCover` 包含 `image_prompt`。然后在 `generate_cover_image` 阶段调用 `self.core.media(prompt=cover.image_prompt, media_type="image", ...)` 生成封面背景图。

### 3. TTS 配置如何复用？
中栏 TTS UI 代码从 `style_config.py` 的 TTS section 完整复制，返回值中的 `tts_inference_mode`, `tts_voice`, `tts_speed`, `tts_workflow`, `ref_audio` 直接传给 Pipeline。

### 4. 如何保持原片分辨率？
在 `setup_environment` 阶段用 ffprobe 探测原片宽高，后续所有生成以此为准。

### 5. 字幕与画面如何对应？
- 带时间戳字幕：LLM prompt 要求 source_windows 取自字幕对应时间段附近
- 纯文本：按 content_start 到 content_end 均匀分布分配窗口

---

## 七、页面位置

Tab 顺序由 `web/pipelines/__init__.py` import 顺序决定：

1. ⚡ 快速创作 (standard)
2. 📁 自定义素材 (asset_based)
3. 💻 数字人口播 (digital_human)
4. 🎬 图生视频 (i2v)
5. 💃 动作迁移 (action_transfer)
6. 🎙️ **视频解说** (commentary) ← 新增

---

*方案创建日期：2026-05-06*
