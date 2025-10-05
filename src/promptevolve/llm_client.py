"""
Unified LLM Client supporting multiple providers (OpenAI, Gemini)
"""

import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
import google.generativeai as genai


class UnifiedLLMClient:
    """
    Unified client that supports both OpenAI and Google Gemini APIs
    with a consistent interface.
    """
    
    def __init__(self, provider: str = "openai", api_key: Optional[str] = None):
        """
        Initialize the unified LLM client.
        
        Args:
            provider: "openai" or "gemini"
            api_key: API key (if None, will use environment variables)
        """
        self.provider = provider.lower()
        
        if self.provider == "openai":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=self.api_key)
        elif self.provider == "gemini":
            self.api_key = api_key or os.getenv("GEMINI_API_KEY")
            genai.configure(api_key=self.api_key)
            self.client = genai
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'gemini'")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """
        Create a chat completion with unified interface.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        if self.provider == "openai":
            return self._openai_completion(messages, model, temperature, max_tokens, **kwargs)
        elif self.provider == "gemini":
            return self._gemini_completion(messages, model, temperature, max_tokens, **kwargs)
    
    def _openai_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """OpenAI-specific completion."""
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return response.choices[0].message.content
    
    def _gemini_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Gemini-specific completion."""
        # Convert messages to Gemini format
        gemini_messages = self._convert_to_gemini_format(messages)
        
        # Create model instance
        model_instance = genai.GenerativeModel(model)
        
        # Generate content
        response = model_instance.generate_content(
            gemini_messages,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )
        
        return response.text
    
    def _convert_to_gemini_format(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert OpenAI-style messages to Gemini format.
        
        Gemini uses a simpler format - we'll combine messages into a single prompt.
        """
        formatted_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                formatted_parts.append(f"Instructions: {content}")
            elif role == "user":
                formatted_parts.append(f"User: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(formatted_parts)


def create_client(config: Dict[str, Any]) -> UnifiedLLMClient:
    """
    Create a unified LLM client from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        UnifiedLLMClient instance
    """
    provider = config.get("api_provider", "openai")
    api_keys = config.get("api_keys", {})
    api_key = api_keys.get(provider)
    
    return UnifiedLLMClient(provider=provider, api_key=api_key)


def get_model_name(config: Dict[str, Any], model_type: str) -> str:
    """
    Get the appropriate model name based on provider and model type.
    
    Args:
        config: Configuration dictionary
        model_type: 'prompt_writer', 'prompt_solver', or 'prompt_verifier'
        
    Returns:
        Model name string
    """
    provider = config.get("api_provider", "openai")
    model_config = config.get("models", {}).get(model_type, {})
    
    # Return provider-specific model name
    return model_config.get(provider, model_config.get("name", "gpt-3.5-turbo"))
