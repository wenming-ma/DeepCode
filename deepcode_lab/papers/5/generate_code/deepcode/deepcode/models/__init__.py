"""
DeepCode Models Package

Core data structures for the DeepCode framework implementing:
- Source Document Model: D = (d1, d2, ..., dL) - sequence of multimodal elements
- Code Repository Model: P = (T, C, M) - directory tree, code files, manifest
- Implementation Blueprint Model: B - structured implementation plan
- Memory Entry Model: mt = (Pt, It, Et) - code memory entries
- RAG Index Model: (c's, ĉt, τ, σ, γ) - reference code mappings
"""

from deepcode.deepcode.models.document import (
    Document,
    DocumentElement,
    ElementType,
    TextBlock,
    Equation,
    Table,
    Figure,
    Pseudocode,
)
from deepcode.deepcode.models.repository import (
    Repository,
    CodeFile,
    DirectoryTree,
    Manifest,
)
from deepcode.deepcode.models.blueprint import (
    Blueprint,
    FileSpecification,
    ComponentSpecification,
    VerificationProtocol,
    ExecutionEnvironment,
    StagedDevelopmentPlan,
    DependencyGraph,
)
from deepcode.deepcode.models.memory_entry import (
    MemoryEntry,
    PublicInterface,
    DependencyEdges,
    CodeMemory,
)
from deepcode.deepcode.models.rag_index import (
    RAGEntry,
    RAGIndex,
    RelationshipType,
)

__all__ = [
    # Document model
    "Document",
    "DocumentElement",
    "ElementType",
    "TextBlock",
    "Equation",
    "Table",
    "Figure",
    "Pseudocode",
    # Repository model
    "Repository",
    "CodeFile",
    "DirectoryTree",
    "Manifest",
    # Blueprint model
    "Blueprint",
    "FileSpecification",
    "ComponentSpecification",
    "VerificationProtocol",
    "ExecutionEnvironment",
    "StagedDevelopmentPlan",
    "DependencyGraph",
    # Memory entry model
    "MemoryEntry",
    "PublicInterface",
    "DependencyEdges",
    "CodeMemory",
    # RAG index model
    "RAGEntry",
    "RAGIndex",
    "RelationshipType",
]
