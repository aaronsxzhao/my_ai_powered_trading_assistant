"""
LLM client for narrative enhancement.

Uses Claude via LiteLLM proxy (OpenAI-compatible API) to enhance 
rule-based analysis with natural language coaching.
"""

import logging
from typing import Optional

from app.config import settings, get_llm_api_key, get_llm_base_url, get_llm_model

logger = logging.getLogger(__name__)


class LLMClient:
    """
    LLM client for narrative coaching via LiteLLM proxy.

    The LLM only:
    - Converts computed findings into Brooks-style narrative
    - Never invents price data
    - Always cites the computed context
    """

    def __init__(self):
        """Initialize LLM client."""
        self.api_key = get_llm_api_key()
        self.base_url = get_llm_base_url()
        self.model = get_llm_model()
        self._client = None

    @property
    def is_available(self) -> bool:
        """Check if LLM is available and enabled."""
        return settings.llm_enabled and self.api_key is not None

    def _get_client(self):
        """Get or create OpenAI-compatible client for LiteLLM proxy."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                logger.warning("OpenAI package not installed. LLM features disabled.")
                return None
            except Exception as e:
                logger.error(f"Failed to initialize LLM client: {e}")
                return None

        return self._client

    def generate_coaching_narrative(
        self,
        context: dict,
        system_prompt: str,
        max_tokens: int = 1500,
    ) -> Optional[str]:
        """
        Generate coaching narrative from computed context.

        Args:
            context: Dictionary with computed analysis data
            system_prompt: System prompt for the LLM
            max_tokens: Maximum tokens in response

        Returns:
            Generated narrative or None if failed
        """
        if not self.is_available:
            logger.debug("LLM not available, skipping narrative generation")
            return None

        client = self._get_client()
        if client is None:
            return None

        try:
            # Build user message from context
            user_message = self._format_context_for_llm(context)

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return None

    def _format_context_for_llm(self, context: dict) -> str:
        """Format context dictionary for LLM input."""
        lines = ["## Computed Analysis Data", ""]

        for key, value in context.items():
            if isinstance(value, dict):
                lines.append(f"### {key}")
                for k, v in value.items():
                    lines.append(f"- {k}: {v}")
            elif isinstance(value, list):
                lines.append(f"### {key}")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(f"- {item}")
                    else:
                        lines.append(f"- {item}")
            else:
                lines.append(f"- **{key}**: {value}")

        return "\n".join(lines)

    def enhance_trade_review(self, review_data: dict) -> Optional[str]:
        """
        Enhance trade review with LLM narrative.

        Args:
            review_data: Trade review data

        Returns:
            Enhanced narrative or None
        """
        from app.llm.prompts import PromptBuilder

        prompt_builder = PromptBuilder()
        system_prompt = prompt_builder.get_trade_review_prompt()

        return self.generate_coaching_narrative(
            context=review_data,
            system_prompt=system_prompt,
        )

    def enhance_premarket_report(self, report_data: dict) -> Optional[str]:
        """
        Enhance premarket report with LLM narrative.

        Args:
            report_data: Premarket report data

        Returns:
            Enhanced narrative or None
        """
        from app.llm.prompts import PromptBuilder

        prompt_builder = PromptBuilder()
        system_prompt = prompt_builder.get_premarket_prompt()

        return self.generate_coaching_narrative(
            context=report_data,
            system_prompt=system_prompt,
            max_tokens=2000,
        )

    def ask_clarification(self, question: str, context: dict) -> Optional[str]:
        """
        Ask a clarifying question about trading context.

        Args:
            question: User's question
            context: Current analysis context

        Returns:
            LLM response or None
        """
        if not self.is_available:
            return None

        from app.llm.prompts import PromptBuilder

        prompt_builder = PromptBuilder()
        system_prompt = prompt_builder.get_qa_prompt()

        context["user_question"] = question

        return self.generate_coaching_narrative(
            context=context,
            system_prompt=system_prompt,
            max_tokens=500,
        )


# Singleton instance cache
_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Get LLM client instance (singleton)."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
