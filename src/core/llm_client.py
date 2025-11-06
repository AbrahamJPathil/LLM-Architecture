"""
Multi-provider LLM client with unified interface.
Supports OpenAI, Anthropic, Gemini, and Groq.
"""

import asyncio
import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional

import openai
try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    import groq
except ImportError:
    groq = None

try:
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )
except ImportError:
    # Simple retry decorator fallback
    def retry(**kwargs):
        def decorator(func):
            return func
        return decorator
    retry_if_exception_type = None
    stop_after_attempt = None
    wait_exponential = None

from src.utils.cost_tracker import TokenUsage
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """Unified response format across providers."""
    
    content: str
    model: str
    provider: str
    
    # Token usage
    input_tokens: int
    output_tokens: int
    
    # Metadata
    latency_ms: float
    finish_reason: Optional[str] = None
    raw_response: Optional[Any] = None
    
    @property
    def token_usage(self) -> TokenUsage:
        """Get TokenUsage object for cost tracking."""
        return TokenUsage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            model=self.model
        )


class BaseLLMClient(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 60
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.provider_name = self.__class__.__name__.replace("Client", "").lower()
    
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion for a prompt."""
        pass
    
    @abstractmethod
    async def complete_async(self, prompt: str, **kwargs) -> LLMResponse:
        """Async version of complete."""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API client with rate-limit-aware fallbacks."""
    
    def __init__(self, **kwargs):
        cooldown = kwargs.pop("rate_limit_cooldown_seconds", 30)
        super().__init__(**kwargs)
        self.client = openai.OpenAI(api_key=self.api_key)
        self.rate_limit_cooldown_seconds = cooldown
        self._rate_limit_cooldowns: Dict[str, float] = {}
    
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion using OpenAI with automatic fallback handling."""
        start_time = time.time()
        original_model = self.model

        # Define fallback chains ordered by preference (highest capacity first)
        fallback_chains: Dict[str, List[str]] = {
            "gpt-4-0125-preview": ["gpt-4-1106-preview", "gpt-4", "gpt-3.5-turbo-0125"],
            "gpt-4-1106-preview": ["gpt-4-0125-preview", "gpt-4", "gpt-3.5-turbo-0125"],
            "gpt-3.5-turbo-0125": ["gpt-3.5-turbo-1106", "gpt-3.5-turbo"],
            "gpt-3.5-turbo-1106": ["gpt-3.5-turbo", "gpt-3.5-turbo-0125"],
        }

        candidate_models = [original_model] + fallback_chains.get(original_model, [])

        # Skip models still in cooldown to avoid repeated rate-limit delays
        now = time.time()
        cooled_candidates = [
            model_name
            for model_name in candidate_models
            if self._rate_limit_cooldowns.get(model_name, 0) <= now
        ]
        if cooled_candidates:
            candidate_models = cooled_candidates

        tried: List[str] = []
        last_error: Optional[Exception] = None

        for current_model in candidate_models:
            tried.append(current_model)
            attempt_start = time.time()

            try:
                response = self.client.chat.completions.create(
                    model=current_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=kwargs.get("temperature", self.temperature),
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                    timeout=kwargs.get("timeout", self.timeout)
                )

                latency_ms = (time.time() - start_time) * 1000
                self._rate_limit_cooldowns.pop(current_model, None)

                if current_model != original_model:
                    logger.info(
                        "Successfully fell back to alternate model",
                        original_model=original_model,
                        fallback_model=current_model,
                        attempted_models=tried,
                        fallback_latency_ms=(time.time() - attempt_start) * 1000
                    )

                return LLMResponse(
                    content=response.choices[0].message.content,
                    model=current_model,
                    provider="openai",
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    latency_ms=latency_ms,
                    finish_reason=response.choices[0].finish_reason,
                    raw_response=response
                )

            except openai.RateLimitError as exc:
                last_error = exc
                self._rate_limit_cooldowns[current_model] = time.time() + self.rate_limit_cooldown_seconds
                logger.warning(
                    "Rate limit encountered, falling back",
                    failed_model=current_model
                )
                continue

            except openai.BadRequestError as exc:
                last_error = exc
                # Longer cooldown for models we do not have access to
                self._rate_limit_cooldowns[current_model] = time.time() + max(self.rate_limit_cooldown_seconds, 300)
                logger.warning(
                    "Model unavailable or unauthorized, falling back",
                    failed_model=current_model
                )
                continue

            except Exception as exc:
                last_error = exc
                logger.error(
                    "OpenAI completion error",
                    model=current_model,
                    error=str(exc)
                )
                break

        if last_error:
            logger.error(
                "All model attempts failed",
                attempted_models=tried,
                original_model=original_model
            )
            raise last_error

        raise RuntimeError("OpenAI completion failed without specific error")
    
    async def complete_async(self, prompt: str, **kwargs) -> LLMResponse:
        """Async completion using OpenAI with automatic fallback."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self.complete, prompt, **kwargs))


class AnthropicClient(BaseLLMClient):
    """Anthropic (Claude) API client."""
    
    def __init__(self, **kwargs):
        if anthropic is None:
            raise ImportError("anthropic package is not installed. Install with: pip install anthropic")
        super().__init__(**kwargs)
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion using Anthropic."""
        start_time = time.time()
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                messages=[{"role": "user", "content": prompt}],
                timeout=kwargs.get("timeout", self.timeout)
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            return LLMResponse(
                content=response.content[0].text,
                model=self.model,
                provider="anthropic",
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_ms=latency_ms,
                finish_reason=response.stop_reason,
                raw_response=response
            )
        
        except Exception as e:
            logger.error(f"Anthropic completion error: {e}", model=self.model)
            raise
    
    async def complete_async(self, prompt: str, **kwargs) -> LLMResponse:
        """Async completion using Anthropic."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.complete, prompt)


class GeminiClient(BaseLLMClient):
    """Google Gemini API client."""
    
    def __init__(self, **kwargs):
        if genai is None:
            raise ImportError("google-generativeai package is not installed. Install with: pip install google-generativeai")
        super().__init__(**kwargs)
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.model)
    
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion using Gemini."""
        start_time = time.time()
        
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=kwargs.get("temperature", self.temperature),
                max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
            )
            
            response = self.client.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Gemini doesn't provide detailed token counts in response
            # We'll estimate them
            from src.utils.cost_tracker import count_tokens
            input_tokens = count_tokens(prompt, self.model)
            output_tokens = count_tokens(response.text, self.model)
            
            return LLMResponse(
                content=response.text,
                model=self.model,
                provider="gemini",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                finish_reason=None,
                raw_response=response
            )
        
        except Exception as e:
            logger.error(f"Gemini completion error: {e}", model=self.model)
            raise
    
    async def complete_async(self, prompt: str, **kwargs) -> LLMResponse:
        """Async completion using Gemini."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.complete, prompt)


class GroqClient(BaseLLMClient):
    """Groq API client (OpenAI-compatible)."""
    
    def __init__(self, **kwargs):
        if groq is None:
            raise ImportError("groq package is not installed. Install with: pip install groq")
        super().__init__(**kwargs)
        self.client = groq.Groq(api_key=self.api_key)
    
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion using Groq."""
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=self.model,
                provider="groq",
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                latency_ms=latency_ms,
                finish_reason=response.choices[0].finish_reason,
                raw_response=response
            )
        
        except Exception as e:
            logger.error(f"Groq completion error: {e}", model=self.model)
            raise
    
    async def complete_async(self, prompt: str, **kwargs) -> LLMResponse:
        """Async completion using Groq."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.complete, prompt)


class XAIClient(BaseLLMClient):
    """XAI (X.AI) API client - OpenAI-compatible interface."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # XAI uses OpenAI-compatible API
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.x.ai/v1"
        )
    
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion using XAI."""
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=self.model,
                provider="xai",
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                latency_ms=latency_ms,
                finish_reason=response.choices[0].finish_reason,
                raw_response=response
            )
        
        except Exception as e:
            logger.error(f"XAI completion error: {e}", model=self.model)
            raise
    
    async def complete_async(self, prompt: str, **kwargs) -> LLMResponse:
        """Async completion using XAI."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(self.complete, **kwargs), prompt)


class MockLLMClient(BaseLLMClient):
    """Deterministic mock client used for fast local testing."""

    def __init__(self, **kwargs):
        kwargs.setdefault("api_key", "mock-api-key")
        kwargs.setdefault("model", kwargs.get("model", "mock-llm"))
        kwargs.setdefault("temperature", 0.0)
        kwargs.setdefault("max_tokens", 512)
        kwargs.setdefault("timeout", 1)
        super().__init__(**kwargs)

    def _synthesize_response(self, prompt: str) -> str:
        """Create a structured deterministic response for the prompt."""
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        stripped = prompt.strip()
        
        # Check if prompt expects JSON output
        if "**Output Format (JSON):**" in prompt or '"new_prompt":' in prompt:
            # Extract original prompt if this is a mutation request
            import re
            current_prompt_match = re.search(r'\*\*Current Prompt:\*\*\s*\n(.+?)(?:\n\*\*|$)', prompt, re.DOTALL)
            if current_prompt_match:
                original = current_prompt_match.group(1).strip()
                # Simple mutation: add a prefix
                mutated = f"Analyze and {original.lower()}" if not original.lower().startswith("analyze") else original
            else:
                mutated = "Extract key information from the provided text."
            
            return json.dumps({
                "new_prompt": mutated,
                "explanation": f"Mock mutation applied (hash: {digest})",
                "diff_summary": "Added clarity and structure",
                "expected_improvement": "Better accuracy and consistency"
            }, indent=2)
        
        # Check if this is a judge evaluation request
        if "overall_quality" in prompt.lower() or "evaluate the quality" in prompt.lower():
            return json.dumps({
                "overall_quality": 0.8,
                "accuracy": 0.85,
                "completeness": 0.8,
                "clarity": 0.75,
                "relevance": 0.85,
                "strengths": ["Clear structure", "Relevant information"],
                "weaknesses": ["Could be more detailed"],
                "suggestions": ["Add more examples"]
            }, indent=2)
        
        # Default text response for evaluation outputs
        first_line = stripped.splitlines()[0] if stripped else "(no input)"
        first_line = first_line.strip()[:160]
        return (
            f"- Mock summary [{digest}]\n"
            f"- First line: {first_line}\n"
            "- Recommendation: Provide a structured contract brief."
        )

    def _estimate_tokens(self, prompt: str, response: str) -> tuple[int, int]:
        """Rough token estimates for tracking."""
        input_tokens = max(10, len(prompt) // 4)
        output_tokens = max(10, len(response) // 4)
        return input_tokens, output_tokens

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        start = time.time()
        content = self._synthesize_response(prompt)
        latency_ms = (time.time() - start) * 1000
        input_tokens, output_tokens = self._estimate_tokens(prompt, content)

        return LLMResponse(
            content=content,
            model=self.model,
            provider="mock",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            finish_reason="stop",
            raw_response={
                "mock": True,
                "prompt_hash": hashlib.md5(prompt.encode("utf-8")).hexdigest(),
            },
        )

    async def complete_async(self, prompt: str, **kwargs) -> LLMResponse:
        await asyncio.sleep(0)
        return self.complete(prompt, **kwargs)


class LLMClient:
    """
    Unified LLM client that routes requests to appropriate providers.
    Supports tier-based configuration (Tier-1, Tier-3, Judge, Embeddings).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM client with configuration.
        
        Args:
            config: Dict with 'provider', 'model', 'api_key', and optional params
        """
        self.config = config
        self.provider = config['provider'].lower()
        self.model = config['model']
        
        # Create appropriate client
        client_class = {
            'openai': OpenAIClient,
            'anthropic': AnthropicClient,
            'gemini': GeminiClient,
            'groq': GroqClient,
            'xai': XAIClient,
            'mock': MockLLMClient,
        }.get(self.provider)
        
        if not client_class:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        client_kwargs = {
            'api_key': config['api_key'],
            'model': self.model,
            'temperature': config.get('temperature', 0.7),
            'max_tokens': config.get('max_tokens', 2048),
            'timeout': config.get('timeout', 60)
        }

        if client_class is OpenAIClient:
            client_kwargs['rate_limit_cooldown_seconds'] = config.get('rate_limit_cooldown_seconds', 30)
        
        self.client = client_class(**client_kwargs)
        
        # Setup fallback provider if configured
        self.fallback_client = None
        if config.get('fallback_provider') and config.get('fallback_model'):
            try:
                fallback_api_key = os.getenv(f"{config['fallback_provider'].upper()}_API_KEY")
                if fallback_api_key:
                    fallback_class = {
                        'openai': OpenAIClient,
                        'anthropic': AnthropicClient,
                        'gemini': GeminiClient,
                        'groq': GroqClient,
                        'xai': XAIClient,
                    }.get(config['fallback_provider'].lower())
                    
                    if fallback_class:
                        fallback_kwargs = {
                            'api_key': fallback_api_key,
                            'model': config['fallback_model'],
                            'temperature': config.get('temperature', 0.7),
                            'max_tokens': config.get('max_tokens', 2048),
                            'timeout': config.get('timeout', 60)
                        }
                        self.fallback_client = fallback_class(**fallback_kwargs)
                        logger.info(
                            f"Fallback provider configured",
                            primary_provider=self.provider,
                            fallback_provider=config['fallback_provider'],
                            fallback_model=config['fallback_model']
                        )
            except Exception as e:
                logger.warning(f"Failed to setup fallback provider: {e}")
        
        logger.info(
            f"LLM client initialized",
            provider=self.provider,
            model=self.model
        )
    
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion with fallback support."""
        try:
            return self.client.complete(prompt, **kwargs)
        except (openai.RateLimitError, Exception) as e:
            if self.fallback_client:
                logger.warning(
                    f"Primary provider failed, trying fallback",
                    primary_provider=self.provider,
                    fallback_provider=self.fallback_client.provider_name,
                    error=str(e)
                )
                try:
                    return self.fallback_client.complete(prompt, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Fallback provider also failed: {fallback_error}")
                    raise e  # Raise original error
            raise
    
    async def complete_async(self, prompt: str, **kwargs) -> LLMResponse:
        """Async completion with fallback support."""
        try:
            return await self.client.complete_async(prompt, **kwargs)
        except (openai.RateLimitError, Exception) as e:
            if self.fallback_client:
                logger.warning(
                    f"Primary provider failed, trying fallback",
                    primary_provider=self.provider,
                    fallback_provider=self.fallback_client.provider_name,
                    error=str(e)
                )
                try:
                    return await self.fallback_client.complete_async(prompt, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Fallback provider also failed: {fallback_error}")
                    raise e  # Raise original error
            raise
    
    async def complete_batch(self, prompts: List[str], **kwargs) -> List[LLMResponse]:
        """Generate completions for multiple prompts in parallel."""
        tasks = [self.complete_async(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)


def create_llm_client(tier: str, config_override: Optional[Dict] = None) -> LLMClient:
    """
    Factory function to create LLM client for a specific tier.
    
    Args:
        tier: 'tier1', 'tier3', 'judge', or 'embeddings'
        config_override: Optional config overrides
    
    Returns:
        LLMClient instance
    """
    from src.utils.config import get_config
    
    config = get_config()
    llm_config = dict(config.get_llm_config(tier))

    if config_override:
        llm_config.update(config_override)

    mock_mode = os.getenv("WARMSTART_MOCK_LLM", "0").lower() in {"1", "true", "yes", "on"}
    if mock_mode:
        llm_config['provider'] = 'mock'
        llm_config.setdefault('model', f"mock-{tier}")
        llm_config.setdefault('api_key', 'mock-api-key')

    if llm_config.get('provider', '').lower() != 'mock' and not llm_config.get('api_key'):
        raise ValueError(f"No API key found for {tier} provider: {llm_config.get('provider')}")

    return LLMClient(llm_config)
