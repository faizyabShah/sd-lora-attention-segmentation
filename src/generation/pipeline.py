"""Prompt generation utilities for image synthesis."""

from __future__ import annotations

import random


class PromptTemplate:
    """Template-based prompt generator mirroring notebook logic."""

    def __init__(self, components: dict[str, list[str]] | None = None):
        self.components = components or {
            "road_type": [
                "narrow street",
                "two-lane road",
                "urban intersection",
                "residential street",
            ],
            "density": ["empty", "moderate traffic", "busy traffic"],
            "weather": ["clear weather", "light rain", "heavy rain", "foggy conditions"],
            "lighting": ["bright daytime", "overcast daylight", "sunset lighting"],
            "road_elements": [
                "with lane markings and sidewalks",
                "with parked cars and crosswalks",
                "with trees and traffic signs",
                "with buildings and streetlights",
            ],
        }

    def generate_prompt(self) -> str:
        """Build one random prompt from component pools."""
        return (
            "Photorealistic dashcam view of a "
            f"{random.choice(self.components['density'])} "
            f"{random.choice(self.components['road_type'])}, "
            f"{random.choice(self.components['weather'])}, "
            f"{random.choice(self.components['lighting'])}, "
            f"{random.choice(self.components['road_elements'])}."
        )

    def generate_many(self, count: int) -> list[str]:
        """Generate many prompts with the same template."""
        return [self.generate_prompt() for _ in range(count)]
