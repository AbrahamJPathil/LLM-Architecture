"""
Cost tracking and token usage monitoring across LLM providers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import tiktoken

from src.utils.logging import get_logger

logger = get_logger(__name__)


# Token costs per 1K tokens (as of Nov 2024)
TOKEN_COSTS = {
    # OpenAI
    "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
    
    # Anthropic
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    
    # Gemini
    "gemini-pro": {"input": 0.00025, "output": 0.0005},
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.00375},
    
    # Groq (very cheap, but rate limited)
    "mixtral-8x7b-32768": {"input": 0.00024, "output": 0.00024},
    "llama2-70b-4096": {"input": 0.0007, "output": 0.0008},
}


@dataclass
class TokenUsage:
    """Token usage for a single LLM call."""
    
    input_tokens: int
    output_tokens: int
    model: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens
    
    @property
    def cost_usd(self) -> float:
        """Calculate cost in USD."""
        # Skip cost calculation for mock models
        if self.model.startswith("mock-") or self.model == "mock-llm":
            return 0.0
            
        if self.model not in TOKEN_COSTS:
            logger.warning(f"Unknown model for cost calculation: {self.model}")
            return 0.0
        
        costs = TOKEN_COSTS[self.model]
        input_cost = (self.input_tokens / 1000) * costs["input"]
        output_cost = (self.output_tokens / 1000) * costs["output"]
        
        return input_cost + output_cost


@dataclass
class CostTracker:
    """
    Tracks token usage and costs across an experiment or generation.
    """
    
    name: str
    usages: List[TokenUsage] = field(default_factory=list)
    max_cost_limit: Optional[float] = None
    
    def add_usage(self, usage: TokenUsage) -> None:
        """Add a token usage record."""
        self.usages.append(usage)
        
        # Check cost limit
        if self.max_cost_limit and self.total_cost > self.max_cost_limit:
            logger.warning(
                f"Cost limit exceeded",
                tracker_name=self.name,
                total_cost=self.total_cost,
                limit=self.max_cost_limit
            )
    
    @property
    def total_input_tokens(self) -> int:
        """Total input tokens across all calls."""
        return sum(u.input_tokens for u in self.usages)
    
    @property
    def total_output_tokens(self) -> int:
        """Total output tokens across all calls."""
        return sum(u.output_tokens for u in self.usages)
    
    @property
    def total_tokens(self) -> int:
        """Total tokens across all calls."""
        return self.total_input_tokens + self.total_output_tokens
    
    @property
    def total_cost(self) -> float:
        """Total cost in USD."""
        return sum(u.cost_usd for u in self.usages)
    
    @property
    def call_count(self) -> int:
        """Number of LLM calls tracked."""
        return len(self.usages)
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics."""
        if not self.usages:
            return {
                "call_count": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
            }
        
        # Group by model
        by_model = {}
        for usage in self.usages:
            if usage.model not in by_model:
                by_model[usage.model] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                }
            
            by_model[usage.model]["calls"] += 1
            by_model[usage.model]["input_tokens"] += usage.input_tokens
            by_model[usage.model]["output_tokens"] += usage.output_tokens
            by_model[usage.model]["cost_usd"] += usage.cost_usd
        
        return {
            "name": self.name,
            "call_count": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost, 4),
            "by_model": by_model,
            "duration_minutes": self._get_duration_minutes(),
        }
    
    def _get_duration_minutes(self) -> float:
        """Calculate duration from first to last call."""
        if len(self.usages) < 2:
            return 0.0
        
        first = min(u.timestamp for u in self.usages)
        last = max(u.timestamp for u in self.usages)
        
        return (last - first).total_seconds() / 60
    
    def reset(self) -> None:
        """Clear all tracked usage."""
        self.usages.clear()
    
    def is_over_budget(self) -> bool:
        """Check if cost exceeds budget limit."""
        if self.max_cost_limit is None:
            return False
        return self.total_cost >= self.max_cost_limit


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens in text using tiktoken.
    
    Args:
        text: Text to count tokens for
        model: Model name to use for encoding
    
    Returns:
        Number of tokens
    """
    try:
        # Map model names to tiktoken encodings
        if "gpt-4" in model or "gpt-3.5" in model:
            encoding = tiktoken.encoding_for_model(model)
        elif "claude" in model:
            # Claude uses similar tokenization to GPT-4
            encoding = tiktoken.get_encoding("cl100k_base")
        else:
            # Fallback to GPT-4 encoding
            encoding = tiktoken.get_encoding("cl100k_base")
        
        return len(encoding.encode(text))
    
    except Exception as e:
        logger.warning(f"Error counting tokens: {e}, using character estimate")
        # Rough estimate: ~4 chars per token
        return len(text) // 4


def estimate_cost(
    prompt: str,
    expected_output_tokens: int,
    model: str
) -> float:
    """
    Estimate cost for an LLM call before making it.
    
    Args:
        prompt: Input prompt text
        expected_output_tokens: Expected number of output tokens
        model: Model name
    
    Returns:
        Estimated cost in USD
    """
    input_tokens = count_tokens(prompt, model)
    
    usage = TokenUsage(
        input_tokens=input_tokens,
        output_tokens=expected_output_tokens,
        model=model
    )
    
    return usage.cost_usd
