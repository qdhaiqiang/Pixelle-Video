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
Commentary Script Generation Prompts

Builds prompts for LLM to generate structured commentary scripts from story material.
"""

from typing import List, Optional, Tuple
from pixelle_video.models.commentary import CommentaryScript


def build_commentary_prompt(
    story_text: str,
    target_duration: int = 300,
    content_start: float = 0.0,
    content_end: float = 0.0,
    has_timestamps: bool = False,
    video_title: str = "",
) -> str:
    """
    Build prompt for LLM to generate a structured commentary script.

    Args:
        story_text: The extracted story/subtitle text (with or without timestamps)
        target_duration: Target duration in seconds
        content_start: Usable story start time in source video
        content_end: Usable story end time in source video
        has_timestamps: Whether story_text contains timestamps
        video_title: Optional video title hint

    Returns:
        Prompt string for LLM
    """
    minutes = target_duration // 60
    seconds = target_duration % 60
    duration_str = f"{minutes}分{seconds}秒" if seconds else f"{minutes}分钟"

    timestamp_hint = """\n\n**时间戳可用**：story_text 中包含了时间戳。请确保每个 chunk 的 source_windows 尽量取自对应字幕的时间戳附近，确保解说内容与画面一致。""" if has_timestamps else """\n\n**无时间戳**：story_text 是纯文本。请按剧情密度均匀分配 source_windows，将可用视频范围 [content_start, content_end] 合理切分。"""

    prompt = f"""# 角色设定

你是一位专业的影视解说撰稿人。你的任务是根据提供的剧情文本，生成一段约 {duration_str} 的解说脚本。脚本将用于 AI 全自动生成解说视频。

# 输入信息

- **视频标题**：{video_title or "未指定"}
- **可用剧情范围**：{content_start:.1f}s ~ {content_end:.1f}s（已去除片头片尾）
- **目标解说时长**：{target_duration} 秒（约 {target_duration // 60 + (1 if target_duration % 60 >= 30 else 0)} 分钟）
{timestamp_hint}

# 剧情文本

```
{story_text[:8000]}
```

---

# 输出要求

请生成一个 CommentaryScript，包含以下要素：

## 1. 标题与范围
- `title`：解说视频的标题
- `content_start` / `content_end`：保持输入值
- `target_duration`：{target_duration}

## 2. 封面配置 (`cover`)
- `title`：剧集标题
- `headline`：醒目大标题，用尖锐的声明或疑问句，避免泛泛的剧集名称
- `question`：一个引发好奇心的疑问式副标题
- `image_prompt`：一段英文 AI 生图 prompt，描述封面场景（ cinematic, dramatic lighting, 适合作为视频封面背景）
- `background_time`：在 source video 中截取一帧作为封面备选的时间点

## 3. 进度条阶段 (`progress_segments`)
- 4-6 个故事阶段标签，例如：["矛盾引爆", "人物交锋", "局势反转", "代价显现", "结局伏笔"]
- 标签要具体、有悬念感，避免泛泛的"开始/中间/结束"

## 4. 解说分段 (`chunks`)

将 {target_duration} 秒解说分成多个 chunk（通常 8-15 个），每个 chunk 包含：

- `chunk_id`：编号如 n001, n002...
- `start` / `end`：在最终解说视频中的时间位置（0-based，单位秒）
- `event`：事件类型（opening / conflict / twist / climax / resolution / 自定义）
- `text`：解说词文本
  - 聚焦故事线、人物动机、前因后果
  - 不复述原台词，不做配音
  - 不机械强调"第几集展示了什么"
  - 口语化短句，适合快速 TTS
  - 信息密度要足够填满时间线，避免空白
- `source_windows`：2-4 个原视频时间窗口（每个 2-6 秒），作为该 chunk 对应的画面素材
  - 必须落在 [{content_start:.1f}, {content_end:.1f}] 范围内
  - 优先选择剧情对应的画面

## 硬性规则（不可违反）

1. 不复述、不翻译原片对白
2. 不重复使用"这一集展示的是""第几集的关键是"等报告式表达
3. 用短句，每句控制在 15-25 字以内
4. 解说词总字数需匹配目标时长（语速约 250-300 字/分钟）
5. 所有 source_windows 必须在可用范围内
6. 封面 headline 要有冲击力，不要只是剧集名称
7. image_prompt 必须是英文，适合 FLUX/SD 等文生图模型

请直接输出 JSON，格式严格匹配 CommentaryScript schema。
"""
    return prompt.strip()


def build_segment_commentary_prompt(
    story_text: str,
    seg_idx: int,
    segment_count: int,
    segment_duration: int,
    content_start: float,
    content_end: float,
    has_timestamps: bool = False,
    video_title: str = "",
) -> str:
    """
    Build prompt for LLM to generate a commentary script for a specific segment.

    Args:
        story_text: The story text for this segment
        seg_idx: Segment index (0-based)
        segment_count: Total number of segments
        segment_duration: Duration of this segment in seconds
        content_start: Story start time in source video for this segment
        content_end: Story end time in source video for this segment
        has_timestamps: Whether story_text contains timestamps
        video_title: Optional video title hint

    Returns:
        Prompt string for LLM
    """
    seg_num = seg_idx + 1
    minutes = segment_duration // 60
    seconds = segment_duration % 60
    duration_str = f"{minutes}分{seconds}秒" if seconds else f"{minutes}分钟"

    segment_hint = f"""
**分段信息**：这是全部 {segment_count} 段中的第 {seg_num} 段。
- 本段解说时长：{segment_duration} 秒（约 {segment_duration // 60 + (1 if segment_duration % 60 >= 30 else 0)} 分钟）
- 本段覆盖的剧情范围：{content_start:.1f}s ~ {content_end:.1f}s
"""

    timestamp_hint = """\n\n**时间戳可用**：story_text 中包含了时间戳。请确保每个 chunk 的 source_windows 尽量取自对应字幕的时间戳附近，确保解说内容与画面一致。""" if has_timestamps else """\n\n**无时间戳**：story_text 是纯文本。请按剧情密度均匀分配 source_windows，将可用视频范围合理切分。"""

    # Adjust headline for multi-segment
    headline_hint = ""
    if segment_count > 1:
        headline_hint = f"\n- 封面 headline 建议体现这是第 {seg_num}/{segment_count} 段，例如 '第{seg_num}段：...'"

    prompt = f"""# 角色设定

你是一位专业的影视解说撰稿人。你的任务是根据提供的剧情文本，生成一段约 {duration_str} 的解说脚本。脚本将用于 AI 全自动生成解说视频。

# 输入信息

- **视频标题**：{video_title or "未指定"}
{segment_hint}
- **目标解说时长**：{segment_duration} 秒
{timestamp_hint}

# 剧情文本（第 {seg_num}/{segment_count} 段）

```
{story_text[:8000]}
```

---

# 输出要求

请生成一个 CommentaryScript，包含以下要素：

## 1. 标题与范围
- `title`：解说视频的标题（可带段号标记）
- `content_start` / `content_end`：保持输入值
- `target_duration`：{segment_duration}

## 2. 封面配置 (`cover`)
- `title`：剧集标题
- `headline`：醒目大标题，用尖锐的声明或疑问句{headline_hint}
- `question`：一个引发好奇心的疑问式副标题
- `image_prompt`：一段英文 AI 生图 prompt，描述封面场景（cinematic, dramatic lighting, 适合作为视频封面背景）
- `background_time`：在 source video 中截取一帧作为封面备选的时间点（取本段剧情范围的中间附近）

## 3. 进度条阶段 (`progress_segments`)
- 4-6 个故事阶段标签，例如：["铺垫", "冲突", "转折", "爆发", "余波"]
- 标签要具体、有悬念感

## 4. 解说分段 (`chunks`)

将 {segment_duration} 秒解说分成多个 chunk（通常 {max(3, segment_duration // 30)}-{max(6, segment_duration // 20)} 个），每个 chunk 包含：

- `chunk_id`：编号如 n001, n002...
- `start` / `end`：在最终解说视频中的时间位置（0-based，单位秒）
- `event`：事件类型
- `text`：解说词文本
  - 聚焦故事线、人物动机、前因后果
  - 不复述原台词，不做配音
  - 口语化短句，适合快速 TTS
  - 信息密度要足够填满时间线
- `source_windows`：2-4 个原视频时间窗口（每个 2-6 秒）
  - 必须落在 [{content_start:.1f}, {content_end:.1f}] 范围内

## 硬性规则

1. 不复述、不翻译原片对白
2. 不机械强调"第几集展示了什么"
3. 用短句，每句 15-25 字以内
4. 解说词总字数匹配目标时长（约 250-300 字/分钟）
5. 所有 source_windows 必须在本段范围内
6. image_prompt 必须是英文

请直接输出 JSON，格式严格匹配 CommentaryScript schema。
"""
    return prompt.strip()


def build_fallback_script_prompt(
    video_path: str,
    target_duration: int = 300,
    first_frame_description: str = "",
) -> str:
    """
    Fallback prompt when no subtitle/story text is available.
    Uses video path + first frame description to generate commentary.
    """
    prompt = f"""# 角色设定

你是一位专业的影视解说撰稿人。现有一段视频需要生成解说脚本，但无法获取到字幕或详细剧情文本。

# 已知信息

- **视频路径**：{video_path}
- **目标解说时长**：{target_duration} 秒
- **首帧画面描述**：{first_frame_description or "（未提供）"}

# 任务

请根据视频路径和首帧描述，尽力推断这是一部什么类型的影视作品，然后生成一段通用的解说脚本。

如果信息不足，请生成一个**通用模板式**的解说脚本，包含：
- 一个吸引人的封面配置
- 合理的进度条阶段
- 与目标时长匹配的 chunks（用占位性解说词）

**注意**：source_windows 可均匀分布在视频全长范围内，假设视频时长约 {target_duration * 5} 秒。

输出 JSON，格式严格匹配 CommentaryScript schema。
"""
    return prompt.strip()
