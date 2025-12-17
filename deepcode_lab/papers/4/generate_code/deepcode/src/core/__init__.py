"""
DeepCode Core Module

This module provides the foundational components for the DeepCode framework:
- Data structures for documents, repositories, blueprints, and memory entries
- LLM interface abstractions for multiple providers (Claude, OpenAI, Gemini)

The core module is the foundation upon which all other DeepCode components are built.
"""

from deepcode.src.core.data_structures import (
    # Document-related structures
    Document,
    DocumentElement,
    DocumentElementType,
    
    # Repository structures
    Repository,
    SourceFile,
    DirectoryStructure,
    DependencyManifest,
    
    # Blueprint structures
    Blueprint,
    FileSpec,
    ComponentSpec,
    VerificationSpec,
    EnvironmentSpec,
    Phase,
    
    # Memory structures
    MemoryEntry,
    InterfaceItem,
    DependencyEdges,
    
    # Relationship structures
    RelationshipTuple,
)

from deepcode.src.core.llm_interface import (
    BaseLLMClient,
    ClaudeLLMClient,
    OpenAILLMClient,
    GeminiLLMClient,
    LLMResponse,
    create_llm_client,
)

__all__ = [
    # Document structures
    "Document",
    "DocumentElement",
    "DocumentElementType",
    
    # Repository structures
    "Repository",
    "SourceFile",
    "DirectoryStructure",
    "DependencyManifest",
    
    # Blueprint structures
    "Blueprint",
    "FileSpec",
    "ComponentSpec",
    "VerificationSpec",
    "EnvironmentSpec",
    "Phase",
    
    # Memory structures
    "MemoryEntry",
    "InterfaceItem",
    "DependencyEdges",
    
    # Relationship structures
    "RelationshipTuple",
    
    # LLM interfaces
    "BaseLLMClient",
    "ClaudeLLMClient",
    "OpenAILLMClient",
    "GeminiLLMClient",
    "LLMResponse",
    "create_llm_client",
]
