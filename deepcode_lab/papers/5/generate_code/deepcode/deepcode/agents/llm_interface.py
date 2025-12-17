"""
LLM Interface Module for DeepCode.

Provides a unified interface for interacting with different LLM providers
(Anthropic Claude, OpenAI GPT) with support for async operations, retries,
and structured response handling.
"""

import asyncio
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from pydantic import BaseModel, Field


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class MessageRole(str, Enum):
    """Message roles for chat-based LLMs."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """A single message in a conversation."""
    role: MessageRole
    content: str
    name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_anthropic_format(self) -> Dict[str, str]:
        """Convert to Anthropic API format."""
        return {
            "role": self.role.value if self.role != MessageRole.SYSTEM else "user",
            "content": self.content
        }
    
    def to_openai_format(self) -> Dict[str, str]:
        """Convert to OpenAI API format."""
        msg = {
            "role": self.role.value,
            "content": self.content
        }
        if self.name:
            msg["name"] = self.name
        return msg


class LLMConfig(BaseModel):
    """Configuration for LLM interface."""
    provider: LLMProvider = LLMProvider.ANTHROPIC
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    timeout: float = 120.0
    max_retries: int = 3
    retry_delay: float = 1.0
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    
    # Provider-specific settings
    anthropic_version: str = "2023-06-01"
    
    def get_api_key(self) -> str:
        """Get API key from config or environment."""
        if self.api_key:
            return self.api_key
        
        if self.provider == LLMProvider.ANTHROPIC:
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            return key
        elif self.provider == LLMProvider.OPENAI:
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            return key
        else:
            raise ValueError(f"Unknown provider: {self.provider}")


class LLMResponse(BaseModel):
    """Response from LLM API call."""
    content: str
    model: str
    provider: LLMProvider
    finish_reason: Optional[str] = None
    usage: Dict[str, int] = Field(default_factory=dict)
    raw_response: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0
    
    @property
    def input_tokens(self) -> int:
        """Get input token count."""
        return self.usage.get("input_tokens", self.usage.get("prompt_tokens", 0))
    
    @property
    def output_tokens(self) -> int:
        """Get output token count."""
        return self.usage.get("output_tokens", self.usage.get("completion_tokens", 0))
    
    @property
    def total_tokens(self) -> int:
        """Get total token count."""
        return self.input_tokens + self.output_tokens


class LLMError(Exception):
    """Base exception for LLM errors."""
    def __init__(self, message: str, provider: Optional[LLMProvider] = None, 
                 status_code: Optional[int] = None, raw_error: Optional[Any] = None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.raw_error = raw_error


class RateLimitError(LLMError):
    """Rate limit exceeded error."""
    pass


class AuthenticationError(LLMError):
    """Authentication failed error."""
    pass


class APIError(LLMError):
    """General API error."""
    pass


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
    
    @abstractmethod
    async def complete(self, messages: List[Message], **kwargs) -> LLMResponse:
        """Generate a completion from messages."""
        pass
    
    @abstractmethod
    async def stream(self, messages: List[Message], **kwargs) -> AsyncIterator[str]:
        """Stream a completion from messages."""
        pass
    
    def complete_sync(self, messages: List[Message], **kwargs) -> LLMResponse:
        """Synchronous completion wrapper."""
        return asyncio.run(self.complete(messages, **kwargs))
    
    async def complete_with_retry(self, messages: List[Message], **kwargs) -> LLMResponse:
        """Complete with automatic retry on transient errors."""
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                return await self.complete(messages, **kwargs)
            except RateLimitError as e:
                last_error = e
                wait_time = self.config.retry_delay * (2 ** attempt)
                await asyncio.sleep(wait_time)
            except APIError as e:
                if e.status_code and e.status_code >= 500:
                    last_error = e
                    wait_time = self.config.retry_delay * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                else:
                    raise
        
        raise last_error or APIError("Max retries exceeded")


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude API."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._initialized = False
    
    def _ensure_client(self):
        """Lazily initialize the Anthropic client."""
        if not self._initialized:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(
                    api_key=self.config.get_api_key(),
                    base_url=self.config.base_url,
                    timeout=self.config.timeout
                )
                self._sync_client = anthropic.Anthropic(
                    api_key=self.config.get_api_key(),
                    base_url=self.config.base_url,
                    timeout=self.config.timeout
                )
                self._initialized = True
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
    
    async def complete(self, messages: List[Message], **kwargs) -> LLMResponse:
        """Generate completion using Claude API."""
        self._ensure_client()
        
        start_time = time.time()
        
        # Extract system message if present
        system_content = None
        chat_messages = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_content = msg.content
            else:
                chat_messages.append(msg.to_anthropic_format())
        
        # Merge kwargs with config
        model = kwargs.get("model", self.config.model)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        temperature = kwargs.get("temperature", self.config.temperature)
        
        try:
            import anthropic
            
            request_params = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": chat_messages,
            }
            
            if system_content:
                request_params["system"] = system_content
            
            response = await self._client.messages.create(**request_params)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract content from response
            content = ""
            if response.content:
                for block in response.content:
                    if hasattr(block, "text"):
                        content += block.text
            
            return LLMResponse(
                content=content,
                model=response.model,
                provider=LLMProvider.ANTHROPIC,
                finish_reason=response.stop_reason,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
                latency_ms=latency_ms
            )
            
        except anthropic.RateLimitError as e:
            raise RateLimitError(str(e), LLMProvider.ANTHROPIC, 429, e)
        except anthropic.AuthenticationError as e:
            raise AuthenticationError(str(e), LLMProvider.ANTHROPIC, 401, e)
        except anthropic.APIError as e:
            raise APIError(str(e), LLMProvider.ANTHROPIC, getattr(e, "status_code", None), e)
    
    async def stream(self, messages: List[Message], **kwargs) -> AsyncIterator[str]:
        """Stream completion using Claude API."""
        self._ensure_client()
        
        # Extract system message if present
        system_content = None
        chat_messages = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_content = msg.content
            else:
                chat_messages.append(msg.to_anthropic_format())
        
        model = kwargs.get("model", self.config.model)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        temperature = kwargs.get("temperature", self.config.temperature)
        
        try:
            import anthropic
            
            request_params = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": chat_messages,
            }
            
            if system_content:
                request_params["system"] = system_content
            
            async with self._client.messages.stream(**request_params) as stream:
                async for text in stream.text_stream:
                    yield text
                    
        except anthropic.RateLimitError as e:
            raise RateLimitError(str(e), LLMProvider.ANTHROPIC, 429, e)
        except anthropic.AuthenticationError as e:
            raise AuthenticationError(str(e), LLMProvider.ANTHROPIC, 401, e)
        except anthropic.APIError as e:
            raise APIError(str(e), LLMProvider.ANTHROPIC, getattr(e, "status_code", None), e)
    
    def complete_sync(self, messages: List[Message], **kwargs) -> LLMResponse:
        """Synchronous completion using Claude API."""
        self._ensure_client()
        
        start_time = time.time()
        
        # Extract system message if present
        system_content = None
        chat_messages = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_content = msg.content
            else:
                chat_messages.append(msg.to_anthropic_format())
        
        model = kwargs.get("model", self.config.model)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        temperature = kwargs.get("temperature", self.config.temperature)
        
        try:
            import anthropic
            
            request_params = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": chat_messages,
            }
            
            if system_content:
                request_params["system"] = system_content
            
            response = self._sync_client.messages.create(**request_params)
            
            latency_ms = (time.time() - start_time) * 1000
            
            content = ""
            if response.content:
                for block in response.content:
                    if hasattr(block, "text"):
                        content += block.text
            
            return LLMResponse(
                content=content,
                model=response.model,
                provider=LLMProvider.ANTHROPIC,
                finish_reason=response.stop_reason,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
                latency_ms=latency_ms
            )
            
        except anthropic.RateLimitError as e:
            raise RateLimitError(str(e), LLMProvider.ANTHROPIC, 429, e)
        except anthropic.AuthenticationError as e:
            raise AuthenticationError(str(e), LLMProvider.ANTHROPIC, 401, e)
        except anthropic.APIError as e:
            raise APIError(str(e), LLMProvider.ANTHROPIC, getattr(e, "status_code", None), e)


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI GPT API."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._initialized = False
    
    def _ensure_client(self):
        """Lazily initialize the OpenAI client."""
        if not self._initialized:
            try:
                import openai
                self._client = openai.AsyncOpenAI(
                    api_key=self.config.get_api_key(),
                    base_url=self.config.base_url,
                    timeout=self.config.timeout
                )
                self._sync_client = openai.OpenAI(
                    api_key=self.config.get_api_key(),
                    base_url=self.config.base_url,
                    timeout=self.config.timeout
                )
                self._initialized = True
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
    
    async def complete(self, messages: List[Message], **kwargs) -> LLMResponse:
        """Generate completion using OpenAI API."""
        self._ensure_client()
        
        start_time = time.time()
        
        # Convert messages to OpenAI format
        openai_messages = [msg.to_openai_format() for msg in messages]
        
        model = kwargs.get("model", self.config.model)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        temperature = kwargs.get("temperature", self.config.temperature)
        
        try:
            import openai
            
            response = await self._client.chat.completions.create(
                model=model,
                messages=openai_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=self.config.top_p,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            content = response.choices[0].message.content or ""
            
            return LLMResponse(
                content=content,
                model=response.model,
                provider=LLMProvider.OPENAI,
                finish_reason=response.choices[0].finish_reason,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                },
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
                latency_ms=latency_ms
            )
            
        except openai.RateLimitError as e:
            raise RateLimitError(str(e), LLMProvider.OPENAI, 429, e)
        except openai.AuthenticationError as e:
            raise AuthenticationError(str(e), LLMProvider.OPENAI, 401, e)
        except openai.APIError as e:
            raise APIError(str(e), LLMProvider.OPENAI, getattr(e, "status_code", None), e)
    
    async def stream(self, messages: List[Message], **kwargs) -> AsyncIterator[str]:
        """Stream completion using OpenAI API."""
        self._ensure_client()
        
        openai_messages = [msg.to_openai_format() for msg in messages]
        
        model = kwargs.get("model", self.config.model)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        temperature = kwargs.get("temperature", self.config.temperature)
        
        try:
            import openai
            
            stream = await self._client.chat.completions.create(
                model=model,
                messages=openai_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=self.config.top_p,
                stream=True,
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except openai.RateLimitError as e:
            raise RateLimitError(str(e), LLMProvider.OPENAI, 429, e)
        except openai.AuthenticationError as e:
            raise AuthenticationError(str(e), LLMProvider.OPENAI, 401, e)
        except openai.APIError as e:
            raise APIError(str(e), LLMProvider.OPENAI, getattr(e, "status_code", None), e)
    
    def complete_sync(self, messages: List[Message], **kwargs) -> LLMResponse:
        """Synchronous completion using OpenAI API."""
        self._ensure_client()
        
        start_time = time.time()
        
        openai_messages = [msg.to_openai_format() for msg in messages]
        
        model = kwargs.get("model", self.config.model)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        temperature = kwargs.get("temperature", self.config.temperature)
        
        try:
            import openai
            
            response = self._sync_client.chat.completions.create(
                model=model,
                messages=openai_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=self.config.top_p,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            content = response.choices[0].message.content or ""
            
            return LLMResponse(
                content=content,
                model=response.model,
                provider=LLMProvider.OPENAI,
                finish_reason=response.choices[0].finish_reason,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                },
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
                latency_ms=latency_ms
            )
            
        except openai.RateLimitError as e:
            raise RateLimitError(str(e), LLMProvider.OPENAI, 429, e)
        except openai.AuthenticationError as e:
            raise AuthenticationError(str(e), LLMProvider.OPENAI, 401, e)
        except openai.APIError as e:
            raise APIError(str(e), LLMProvider.OPENAI, getattr(e, "status_code", None), e)


class LLMInterface:
    """
    Unified interface for LLM interactions.
    
    Provides a consistent API for interacting with different LLM providers,
    with support for async operations, retries, and structured responses.
    
    Example:
        ```python
        config = LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-4-20250514")
        llm = LLMInterface(config)
        
        messages = [
            Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            Message(role=MessageRole.USER, content="Hello!")
        ]
        
        response = await llm.complete(messages)
        print(response.content)
        ```
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize LLM interface.
        
        Args:
            config: LLM configuration. If None, uses default Anthropic config.
        """
        self.config = config or LLMConfig()
        self._client = self._create_client()
    
    def _create_client(self) -> BaseLLMClient:
        """Create appropriate client based on provider."""
        if self.config.provider == LLMProvider.ANTHROPIC:
            return AnthropicClient(self.config)
        elif self.config.provider == LLMProvider.OPENAI:
            return OpenAIClient(self.config)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    async def complete(self, messages: List[Message], **kwargs) -> LLMResponse:
        """
        Generate a completion from messages.
        
        Args:
            messages: List of conversation messages
            **kwargs: Additional parameters (model, temperature, max_tokens)
            
        Returns:
            LLMResponse with generated content
        """
        return await self._client.complete_with_retry(messages, **kwargs)
    
    async def stream(self, messages: List[Message], **kwargs) -> AsyncIterator[str]:
        """
        Stream a completion from messages.
        
        Args:
            messages: List of conversation messages
            **kwargs: Additional parameters
            
        Yields:
            Text chunks as they are generated
        """
        async for chunk in self._client.stream(messages, **kwargs):
            yield chunk
    
    def complete_sync(self, messages: List[Message], **kwargs) -> LLMResponse:
        """
        Synchronous completion wrapper.
        
        Args:
            messages: List of conversation messages
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with generated content
        """
        return self._client.complete_sync(messages, **kwargs)
    
    async def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        """
        Simple generation from a prompt string.
        
        Args:
            prompt: User prompt
            system: Optional system prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated text content
        """
        messages = []
        if system:
            messages.append(Message(role=MessageRole.SYSTEM, content=system))
        messages.append(Message(role=MessageRole.USER, content=prompt))
        
        response = await self.complete(messages, **kwargs)
        return response.content
    
    def generate_sync(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        """
        Synchronous simple generation from a prompt string.
        
        Args:
            prompt: User prompt
            system: Optional system prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated text content
        """
        messages = []
        if system:
            messages.append(Message(role=MessageRole.SYSTEM, content=system))
        messages.append(Message(role=MessageRole.USER, content=prompt))
        
        response = self.complete_sync(messages, **kwargs)
        return response.content
    
    def switch_provider(self, provider: LLMProvider, model: Optional[str] = None):
        """
        Switch to a different LLM provider.
        
        Args:
            provider: New provider to use
            model: Optional model name for the new provider
        """
        self.config.provider = provider
        if model:
            self.config.model = model
        elif provider == LLMProvider.ANTHROPIC:
            self.config.model = "claude-sonnet-4-20250514"
        elif provider == LLMProvider.OPENAI:
            self.config.model = "gpt-4o"
        
        self._client = self._create_client()
    
    @classmethod
    def create_anthropic(cls, model: str = "claude-sonnet-4-20250514", **kwargs) -> "LLMInterface":
        """Create an Anthropic-configured interface."""
        config = LLMConfig(provider=LLMProvider.ANTHROPIC, model=model, **kwargs)
        return cls(config)
    
    @classmethod
    def create_openai(cls, model: str = "gpt-4o", **kwargs) -> "LLMInterface":
        """Create an OpenAI-configured interface."""
        config = LLMConfig(provider=LLMProvider.OPENAI, model=model, **kwargs)
        return cls(config)


# Convenience functions for quick usage
def create_message(role: str, content: str) -> Message:
    """Create a message with string role."""
    role_map = {
        "system": MessageRole.SYSTEM,
        "user": MessageRole.USER,
        "assistant": MessageRole.ASSISTANT,
    }
    return Message(role=role_map.get(role.lower(), MessageRole.USER), content=content)


def create_messages(messages: List[Dict[str, str]]) -> List[Message]:
    """Create messages from list of dicts."""
    return [create_message(m.get("role", "user"), m.get("content", "")) for m in messages]


# Type exports
__all__ = [
    # Enums
    "LLMProvider",
    "MessageRole",
    # Models
    "Message",
    "LLMConfig",
    "LLMResponse",
    # Errors
    "LLMError",
    "RateLimitError",
    "AuthenticationError",
    "APIError",
    # Clients
    "BaseLLMClient",
    "AnthropicClient",
    "OpenAIClient",
    # Main interface
    "LLMInterface",
    # Utilities
    "create_message",
    "create_messages",
]
