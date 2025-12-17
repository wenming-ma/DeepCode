"""
DeepCode Agents Module

This module provides the base infrastructure for all agents in the DeepCode framework,
including the abstract base agent class, LLM interface, and common utilities.

The agent architecture follows a dependency injection pattern where:
- BaseAgent provides common functionality (prompt handling, response parsing)
- LLMInterface provides unified access to multiple LLM providers
- Concrete agents inherit from BaseAgent and implement domain-specific logic
"""

from deepcode.deepcode.agents.base_agent import (
    # Core agent classes
    BaseAgent,
    SyncBaseAgent,
    
    # Configuration and state
    AgentConfig,
    AgentRole,
    AgentState,
    
    # Message and response types
    AgentMessage,
    AgentResponse,
    
    # Prompt template handling
    PromptTemplate,
)

from deepcode.deepcode.agents.llm_interface import (
    # Main interface
    LLMInterface,
    
    # Configuration
    LLMConfig,
    LLMProvider,
    
    # Message types
    Message,
    MessageRole,
    
    # Response types
    LLMResponse,
    
    # Client implementations
    BaseLLMClient,
    AnthropicClient,
    OpenAIClient,
    
    # Utility functions
    create_message,
    create_messages,
    
    # Exceptions
    LLMError,
    RateLimitError,
    AuthenticationError,
    APIError,
)

__all__ = [
    # Base agent infrastructure
    "BaseAgent",
    "SyncBaseAgent",
    "AgentConfig",
    "AgentRole",
    "AgentState",
    "AgentMessage",
    "AgentResponse",
    "PromptTemplate",
    
    # LLM interface
    "LLMInterface",
    "LLMConfig",
    "LLMProvider",
    "Message",
    "MessageRole",
    "LLMResponse",
    
    # LLM clients
    "BaseLLMClient",
    "AnthropicClient",
    "OpenAIClient",
    
    # Utility functions
    "create_message",
    "create_messages",
    
    # Exceptions
    "LLMError",
    "RateLimitError",
    "AuthenticationError",
    "APIError",
]
