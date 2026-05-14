# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Story Lesson Pipeline

Generates children's story explanation videos from a title. It is specialized
for idiom stories, proverb stories, fables, mythology, and picture-book style
content, while reusing the standard media/TTS/composition pipeline.
"""

from typing import List

from loguru import logger
from pydantic import BaseModel, Field

from pixelle_video.models.progress import ProgressEvent
from pixelle_video.models.storyboard import ContentMetadata
from pixelle_video.pipelines.standard import StandardPipeline
from pixelle_video.utils.prompt_helper import build_image_prompt


class StoryLessonScene(BaseModel):
    """A single story lesson scene."""

    scene_number: int = Field(description="Scene number starting from 1")
    narration: str = Field(description="Voiceover narration for the scene")
    visual_prompt: str = Field(description="Image or video generation prompt")
    teaching_point: str = Field(description="Small learning point for this scene")


class StoryLessonPlan(BaseModel):
    """Complete story lesson plan."""

    title: str = Field(description="Final video title")
    category: str = Field(description="Story category")
    summary: str = Field(description="Short story summary")
    moral: str = Field(description="Main lesson or moral")
    character_design: str = Field(description="Consistent character design for visual prompts")
    scenes: List[StoryLessonScene] = Field(description="Ordered story scenes")


class StoryLessonPipeline(StandardPipeline):
    """
    Story lesson video generation pipeline.

    Workflow:
    1. Generate a structured story lesson plan from a title
    2. Use scene narrations directly for TTS
    3. Use scene visual prompts directly for media generation
    4. Reuse StandardPipeline for storyboard initialization, asset production,
       post-production, and persistence
    """

    async def generate_content(self, ctx):
        """Generate structured story scenes and narrations."""
        from pixelle_video.prompts.story_lesson import build_story_lesson_prompt

        title = ctx.input_text.strip()
        if not title:
            raise ValueError("Story title is required")

        n_scenes = ctx.params.get("n_scenes", 6)
        category = ctx.params.get("story_category", "idiom")
        audience = ctx.params.get("audience", "小学三四年级")
        visual_style = ctx.params.get("story_visual_style", "小学生语文课本插图风格")
        duration_seconds = ctx.params.get("duration_seconds", 60)
        include_moral = ctx.params.get("include_moral", True)
        include_life_example = ctx.params.get("include_life_example", True)
        extra_requirements = ctx.params.get("extra_requirements", "")

        self._report_progress(ctx.progress_callback, "generating_storyboard", 0.05)

        prompt = build_story_lesson_prompt(
            title=title,
            category=category,
            audience=audience,
            visual_style=visual_style,
            n_scenes=n_scenes,
            duration_seconds=duration_seconds,
            include_moral=include_moral,
            include_life_example=include_life_example,
            extra_requirements=extra_requirements,
        )

        plan: StoryLessonPlan = await self.core.llm(
            prompt=prompt,
            response_type=StoryLessonPlan,
            temperature=0.75,
            max_tokens=5000,
        )

        if len(plan.scenes) > n_scenes:
            plan.scenes = plan.scenes[:n_scenes]
        elif len(plan.scenes) < n_scenes:
            raise ValueError(f"Expected {n_scenes} story scenes, got {len(plan.scenes)}")

        ctx.story_lesson_plan = plan
        ctx.narrations = [scene.narration for scene in plan.scenes]
        ctx.title = ctx.params.get("title") or plan.title or title
        ctx.params["mode"] = "story_lesson"
        ctx.params["pipeline"] = "story_lesson"
        ctx.params["title"] = ctx.title
        ctx.params["content_metadata"] = ContentMetadata(
            title=ctx.title,
            genre=plan.category,
            summary=plan.summary,
            subtitle=plan.moral,
        )

        logger.info(f"✅ Generated story lesson plan: {ctx.title} ({len(ctx.narrations)} scenes)")

    async def determine_title(self, ctx):
        """Use the generated story title from generate_content."""
        if not ctx.title:
            ctx.title = ctx.input_text.strip()
        logger.info(f"📝 Story lesson title: {ctx.title}")

    async def plan_visuals(self, ctx):
        """Use structured visual prompts from the story plan."""
        from pixelle_video.utils.template_util import get_template_type
        from pathlib import Path

        frame_template = ctx.params.get("frame_template") or "1080x1920/image_default.html"
        template_type = get_template_type(Path(frame_template).name)
        template_requires_media = template_type in {"image", "video"}

        if not template_requires_media:
            ctx.image_prompts = [None] * len(ctx.narrations)
            logger.info("⚡ Static template selected; skipped story visual prompt media generation")
            return

        self._report_progress(ctx.progress_callback, "generating_image_prompts", 0.15)

        plan: StoryLessonPlan = ctx.story_lesson_plan
        prompt_prefix = ctx.params.get("prompt_prefix", "")
        visual_style = ctx.params.get("story_visual_style", "")

        ctx.image_prompts = []
        for scene in plan.scenes:
            base_prompt = (
                f"{scene.visual_prompt}. "
                f"Consistent character design: {plan.character_design}. "
                f"Educational children's story illustration, {visual_style}. "
                "Clean composition, expressive characters, no text in image."
            )
            ctx.image_prompts.append(build_image_prompt(base_prompt, prompt_prefix))

        logger.info(f"✅ Prepared {len(ctx.image_prompts)} story visual prompts")
