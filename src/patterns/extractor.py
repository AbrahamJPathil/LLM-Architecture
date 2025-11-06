"""
PatternExtractor: Extract reusable structural patterns from a champion prompt.

Uses Tier-1 LLM by default; in mock mode falls back to a small static set.
Returns a list of {name, description} pattern dicts.
"""

from __future__ import annotations

import json
from typing import List, Dict, Optional

from src.core.llm_client import create_llm_client
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PatternExtractor:
    def __init__(self, tier: str = "tier1"):
        self.client = create_llm_client(tier)

    def extract(self, champion_prompt: str, domain: Optional[str] = None, task_description: Optional[str] = None) -> List[Dict]:
        """Extract pattern name/description pairs from a prompt."""
        prompt = self._build_prompt(champion_prompt, domain, task_description)
        try:
            resp = self.client.complete(prompt, temperature=0.2, max_tokens=600)
            patterns = self._parse(resp.content)
            if not patterns:
                logger.warning("PatternExtractor returned empty set; using minimal defaults")
                patterns = self._defaults(champion_prompt)
            return patterns
        except Exception as e:
            logger.error(f"Pattern extraction failed: {e}")
            return self._defaults(champion_prompt)

    def _build_prompt(self, prompt_text: str, domain: Optional[str], task: Optional[str]) -> str:
        parts = [
            "You are an expert prompt engineer.",
            "Analyze the following successful prompt and identify its reusable structural patterns.",
        ]
        if domain and domain != "general":
            parts.append(f"Domain: {domain}")
        if task:
            parts.append(f"Task: {task}")
        parts.extend([
            "\nPROMPT:\n" + prompt_text,
            "\nReturn ONLY JSON list where each item has: {\"pattern_name\": str, \"description\": str}.",
            "Example: [{\"pattern_name\": \"EXPERT_ROLE_SPECIFICATION\", \"description\": \"Assigns a qualified role...\"}]",
        ])
        return "\n".join(parts)

    def _parse(self, content: str) -> List[Dict]:
        text = content.strip()
        # Extract JSON array
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]
        # Ensure list
        data = json.loads(text)
        if isinstance(data, dict):
            data = [data]
        results = []
        for item in data:
            name = item.get("pattern_name") or item.get("name")
            desc = item.get("description") or ""
            if name:
                results.append({"pattern_name": name, "description": desc})
        return results

    def _defaults(self, prompt_text: str) -> List[Dict]:
        # Simple heuristics
        defaults = []
        if "json" in prompt_text.lower():
            defaults.append({"pattern_name": "JSON_SCHEMA_OUTPUT_FORMATTING", "description": "Require strict JSON output with a schema."})
        if any(k in prompt_text.lower() for k in ["step", "reason", "think"]):
            defaults.append({"pattern_name": "CHAIN_OF_THOUGHT_INSTRUCTION", "description": "Ask for step-by-step reasoning."})
        if "role" in prompt_text.lower() or "act as" in prompt_text.lower():
            defaults.append({"pattern_name": "EXPERT_ROLE_SPECIFICATION", "description": "Assign an expert role relevant to the task."})
        if not defaults:
            defaults.append({"pattern_name": "STRUCTURED_OUTPUT_SECTIONS", "description": "Use clear sections and bullet points."})
        return defaults
