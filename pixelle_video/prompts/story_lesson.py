# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Story lesson prompt

Generates structured educational storyboards for stories, poems,
traditional culture, safety lessons, science explainers, and picture books.
"""


CONTENT_TYPE_GUIDANCE = {
    "poetry": (
        "For classical poetry, include the poem background, line-by-line meaning, "
        "key imagery, emotion, and a child-friendly takeaway. Do not turn it into "
        "a fictional plot that changes the poem."
    ),
    "textbook": (
        "For textbook stories, explain the plot, key vocabulary, character actions, "
        "and what the target audience should understand from the text."
    ),
    "historical": (
        "For historical figure stories, keep facts age-appropriate, avoid unsupported "
        "details, and focus on the person's action, choice, and value."
    ),
    "culture": (
        "For traditional culture, explain origin, custom, symbolic meaning, and how "
        "the target audience can recognize it in daily life."
    ),
    "science": (
        "For science stories, explain the phenomenon with simple cause-and-effect, "
        "use accurate child-friendly language, and avoid pseudo-science."
    ),
    "safety": (
        "For safety education, show the risky situation, correct behavior, and clear "
        "safety rule. Keep the tone calm and non-frightening."
    ),
    "character": (
        "For character education, use a concrete child-friendly situation to explain "
        "the value, choice, consequence, and practical behavior."
    ),
}


TEACHING_SUMMARY_GUIDANCE = {
    "poetry": "Explain the poem's meaning, imagery, and emotion.",
    "textbook": "Explain the key plot, vocabulary, and reading focus.",
    "historical": "Summarize the historical figure's quality, choice, and value.",
    "culture": "Explain the cultural meaning, symbol, and custom.",
    "science": "Explain the core science principle accurately and simply.",
    "safety": "Explain the safety rule and the correct behavior.",
    "character": "Explain the character value and practical behavior.",
}


LIFE_EXAMPLE_GUIDANCE = {
    "poetry": "Connect the poem to the target audience's real feeling or everyday experience.",
    "textbook": "Add a classroom or daily-life example related to the text.",
    "historical": "Add an audience-appropriate action people can learn from the figure.",
    "culture": "Add an example of where the target audience may see this culture in daily life.",
    "science": "Add a daily observation or safe example.",
    "safety": "Add a realistic school, home, or street safety scenario.",
    "character": "Add a school or family scenario where the value can be practiced.",
}


STORY_LESSON_PROMPT = """You are a professional educational short-video scriptwriter and storyboard director.
Create a short educational story video plan from the title provided by the user.

## User Request
- Content title: {title}
- Content type: {category}
- Target audience: {audience}
- Visual style: {visual_style}
- Storyboard scenes: {n_scenes}
- Target duration: about {duration_seconds} seconds
- Include teaching summary: {include_moral} ({teaching_summary_guidance})
- Include audience-appropriate example/application: {include_life_example} ({life_example_guidance})
- Extra requirements: {extra_requirements}

## Category-Specific Guidance
{category_guidance}

## Content Requirements
1. The narration language must follow the story title and extra requirements. If the title is Chinese, narrations must be Chinese.
2. Adapt vocabulary, sentence length, explanation depth, pace, and examples to the target audience.
3. For story-based content, use a complete arc: opening, setting, key action/conflict, result, lesson.
4. For knowledge-based content, use a clear teaching flow: introduce topic, explain background, break down key points, give examples, summarize.
5. For idioms, proverbs, poems, textbook stories, culture, science, safety, and character education, explain the meaning clearly and accurately.
6. Do not invent harmful, frightening, violent, or adult-oriented details.
7. Each scene narration should be concise and natural for TTS.
8. The visual prompts must keep characters consistent across scenes.

## Visual Requirements
1. Every visual_prompt should describe one clear frame or clip.
2. Include subject, action, setting, mood, composition, and style.
3. Use the visual style exactly: {visual_style}.
4. Prefer clean educational composition suitable for the target audience.
5. Avoid text inside generated images unless explicitly requested.
6. Keep the main characters visually consistent by repeating the same character design.

## Output Requirements
Return exactly {n_scenes} scenes.
Each scene must include:
- scene_number: starts from 1
- narration: voiceover for this scene
- visual_prompt: prompt for image/video generation
- teaching_point: the small learning point of this scene

Now generate the structured story lesson storyboard."""


def build_story_lesson_prompt(
    title: str,
    category: str,
    audience: str,
    visual_style: str,
    n_scenes: int,
    duration_seconds: int,
    include_moral: bool,
    include_life_example: bool,
    extra_requirements: str = "",
) -> str:
    """Build a story lesson structured generation prompt."""
    category_guidance = CONTENT_TYPE_GUIDANCE.get(category, "Follow the selected category accurately and keep the lesson child-friendly.")
    teaching_summary_guidance = TEACHING_SUMMARY_GUIDANCE.get(category, "Explain the story meaning, lesson, or core takeaway clearly.")
    life_example_guidance = LIFE_EXAMPLE_GUIDANCE.get(category, "Add a child-friendly life example only if it naturally helps understanding.")
    return STORY_LESSON_PROMPT.format(
        title=title,
        category=category,
        audience=audience,
        visual_style=visual_style,
        n_scenes=n_scenes,
        duration_seconds=duration_seconds,
        include_moral="yes" if include_moral else "no",
        include_life_example="yes" if include_life_example else "no",
        teaching_summary_guidance=teaching_summary_guidance,
        life_example_guidance=life_example_guidance,
        extra_requirements=extra_requirements or "None",
        category_guidance=category_guidance,
    )
