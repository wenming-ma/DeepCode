"""
RAG Index Model for DeepCode.

Defines the CodeRAG index structure for storing and retrieving reference code
mappings. The index maps source reference files to target blueprint files with
relationship metadata.

Index tuple format: (c's, ĉt, τ, σ, γ) where:
- c's: source file path from reference repository
- ĉt: target blueprint file to be generated
- τ: relationship type (direct_impl, pattern_reference, utility)
- σ: confidence score [0,1]
- γ: actionable context (code snippets, explanations)
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class RelationshipType(str, Enum):
    """Type of relationship between reference and target code."""
    
    DIRECT_IMPL = "direct_impl"
    """Direct implementation - reference code directly implements the target."""
    
    PATTERN_REFERENCE = "pattern_reference"
    """Pattern reference - reference shows a design pattern applicable to target."""
    
    UTILITY = "utility"
    """Utility - reference provides helper functions/utilities for target."""
    
    ALGORITHM = "algorithm"
    """Algorithm - reference implements an algorithm needed by target."""
    
    DATA_STRUCTURE = "data_structure"
    """Data structure - reference defines data structures used by target."""
    
    CONFIGURATION = "configuration"
    """Configuration - reference shows configuration patterns for target."""
    
    TEST_EXAMPLE = "test_example"
    """Test example - reference provides test patterns for target."""


class CodeSnippet(BaseModel):
    """A code snippet extracted from reference code."""
    
    content: str = Field(
        description="The actual code content"
    )
    start_line: Optional[int] = Field(
        default=None,
        description="Starting line number in source file"
    )
    end_line: Optional[int] = Field(
        default=None,
        description="Ending line number in source file"
    )
    language: str = Field(
        default="python",
        description="Programming language of the snippet"
    )
    description: Optional[str] = Field(
        default=None,
        description="Description of what this snippet does"
    )
    
    def to_context_string(self) -> str:
        """Convert snippet to context string for LLM."""
        lines = []
        if self.description:
            lines.append(f"# {self.description}")
        lines.append(f"```{self.language}")
        lines.append(self.content)
        lines.append("```")
        return "\n".join(lines)


class ActionableContext(BaseModel):
    """Actionable context extracted from reference code (γ component)."""
    
    summary: str = Field(
        description="High-level summary of the reference code"
    )
    key_concepts: List[str] = Field(
        default_factory=list,
        description="Key concepts/patterns identified in the code"
    )
    code_snippets: List[CodeSnippet] = Field(
        default_factory=list,
        description="Relevant code snippets"
    )
    usage_notes: Optional[str] = Field(
        default=None,
        description="Notes on how to use/adapt this reference"
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="External dependencies used by the reference"
    )
    adaptation_hints: List[str] = Field(
        default_factory=list,
        description="Hints for adapting this code to the target"
    )
    
    def to_context_string(self) -> str:
        """Convert to context string for LLM injection."""
        lines = [f"## Reference Context\n"]
        lines.append(f"**Summary**: {self.summary}\n")
        
        if self.key_concepts:
            lines.append("**Key Concepts**:")
            for concept in self.key_concepts:
                lines.append(f"  - {concept}")
            lines.append("")
        
        if self.code_snippets:
            lines.append("**Relevant Code**:")
            for snippet in self.code_snippets:
                lines.append(snippet.to_context_string())
                lines.append("")
        
        if self.usage_notes:
            lines.append(f"**Usage Notes**: {self.usage_notes}\n")
        
        if self.dependencies:
            lines.append(f"**Dependencies**: {', '.join(self.dependencies)}\n")
        
        if self.adaptation_hints:
            lines.append("**Adaptation Hints**:")
            for hint in self.adaptation_hints:
                lines.append(f"  - {hint}")
        
        return "\n".join(lines)


class RAGEntry(BaseModel):
    """
    Single RAG index entry mapping reference code to target file.
    
    Represents the tuple (c's, ĉt, τ, σ, γ) from the paper:
    - source_file (c's): Path to reference source file
    - target_file (ĉt): Path to target blueprint file
    - relationship_type (τ): Type of relationship
    - confidence (σ): Confidence score [0,1]
    - context (γ): Actionable context
    """
    
    source_file: str = Field(
        description="Path to reference source file (c's)"
    )
    source_repo: Optional[str] = Field(
        default=None,
        description="Name/URL of the source repository"
    )
    target_file: str = Field(
        description="Path to target blueprint file (ĉt)"
    )
    relationship_type: RelationshipType = Field(
        description="Type of relationship (τ)"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score [0,1] (σ)"
    )
    context: ActionableContext = Field(
        description="Actionable context (γ)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    
    def to_context_string(self) -> str:
        """Convert entry to context string for LLM."""
        lines = [
            f"### Reference: {self.source_file}",
            f"**Relationship**: {self.relationship_type.value} (confidence: {self.confidence:.2f})",
        ]
        if self.source_repo:
            lines.append(f"**Source Repository**: {self.source_repo}")
        lines.append("")
        lines.append(self.context.to_context_string())
        return "\n".join(lines)
    
    @classmethod
    def create(
        cls,
        source_file: str,
        target_file: str,
        relationship_type: RelationshipType,
        confidence: float,
        summary: str,
        code_snippets: Optional[List[CodeSnippet]] = None,
        source_repo: Optional[str] = None,
        **kwargs
    ) -> "RAGEntry":
        """Factory method for creating RAG entries."""
        context = ActionableContext(
            summary=summary,
            code_snippets=code_snippets or [],
            **{k: v for k, v in kwargs.items() if k in ActionableContext.model_fields}
        )
        return cls(
            source_file=source_file,
            source_repo=source_repo,
            target_file=target_file,
            relationship_type=relationship_type,
            confidence=confidence,
            context=context,
            metadata={k: v for k, v in kwargs.items() if k not in ActionableContext.model_fields}
        )


class RAGIndex(BaseModel):
    """
    CodeRAG Index for storing and retrieving reference code mappings.
    
    The index J contains entries mapping reference repository files to
    target blueprint files with relationship metadata and actionable context.
    
    Supports:
    - Adding entries from reference repository analysis
    - Retrieving relevant entries for a target file
    - Adaptive retrieval based on confidence thresholds
    """
    
    entries: List[RAGEntry] = Field(
        default_factory=list,
        description="All RAG index entries"
    )
    source_repos: List[str] = Field(
        default_factory=list,
        description="List of indexed source repositories"
    )
    target_blueprint: Optional[str] = Field(
        default=None,
        description="Name/path of the target blueprint"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Index metadata"
    )
    
    def add_entry(self, entry: RAGEntry) -> None:
        """Add an entry to the index."""
        self.entries.append(entry)
        if entry.source_repo and entry.source_repo not in self.source_repos:
            self.source_repos.append(entry.source_repo)
    
    def add_entries(self, entries: List[RAGEntry]) -> None:
        """Add multiple entries to the index."""
        for entry in entries:
            self.add_entry(entry)
    
    def retrieve(
        self,
        target_file: str,
        top_k: int = 5,
        min_confidence: float = 0.0,
        relationship_types: Optional[List[RelationshipType]] = None
    ) -> List[RAGEntry]:
        """
        Retrieve relevant entries for a target file.
        
        Args:
            target_file: Path to the target file being generated
            top_k: Maximum number of entries to return
            min_confidence: Minimum confidence threshold
            relationship_types: Filter by relationship types (None = all)
            
        Returns:
            List of relevant RAGEntry objects sorted by confidence
        """
        # Filter entries for target file
        relevant = [
            entry for entry in self.entries
            if entry.target_file == target_file
            and entry.confidence >= min_confidence
        ]
        
        # Filter by relationship type if specified
        if relationship_types:
            relevant = [
                entry for entry in relevant
                if entry.relationship_type in relationship_types
            ]
        
        # Sort by confidence (descending) and return top_k
        relevant.sort(key=lambda x: x.confidence, reverse=True)
        return relevant[:top_k]
    
    def get_best_context(
        self,
        target_file: str,
        min_confidence: float = 0.5
    ) -> Optional[Tuple[RAGEntry, str]]:
        """
        Get the best context for a target file.
        
        Args:
            target_file: Path to the target file
            min_confidence: Minimum confidence threshold
            
        Returns:
            Tuple of (best entry, context string) or None if no match
        """
        relevant = self.retrieve(target_file, top_k=1, min_confidence=min_confidence)
        if not relevant:
            return None
        
        best = relevant[0]
        return (best, best.to_context_string())
    
    def get_all_targets(self) -> List[str]:
        """Get all unique target files in the index."""
        return list(set(entry.target_file for entry in self.entries))
    
    def get_entries_by_source(self, source_file: str) -> List[RAGEntry]:
        """Get all entries from a specific source file."""
        return [entry for entry in self.entries if entry.source_file == source_file]
    
    def get_entries_by_repo(self, repo: str) -> List[RAGEntry]:
        """Get all entries from a specific repository."""
        return [entry for entry in self.entries if entry.source_repo == repo]
    
    def get_coverage_stats(self) -> Dict[str, Any]:
        """Get statistics about index coverage."""
        targets = self.get_all_targets()
        
        # Calculate average confidence per target
        target_confidences = {}
        for target in targets:
            entries = self.retrieve(target, top_k=100)
            if entries:
                target_confidences[target] = sum(e.confidence for e in entries) / len(entries)
        
        # Count by relationship type
        type_counts = {}
        for entry in self.entries:
            rel_type = entry.relationship_type.value
            type_counts[rel_type] = type_counts.get(rel_type, 0) + 1
        
        return {
            "total_entries": len(self.entries),
            "unique_targets": len(targets),
            "source_repos": len(self.source_repos),
            "avg_confidence": sum(e.confidence for e in self.entries) / len(self.entries) if self.entries else 0,
            "relationship_type_distribution": type_counts,
            "target_coverage": target_confidences
        }
    
    def to_context_string(self, target_file: str, top_k: int = 3) -> str:
        """
        Generate context string for a target file.
        
        Args:
            target_file: Path to the target file
            top_k: Number of top entries to include
            
        Returns:
            Formatted context string for LLM injection
        """
        entries = self.retrieve(target_file, top_k=top_k)
        if not entries:
            return ""
        
        lines = [
            "# Reference Code Context",
            f"The following reference implementations are relevant for `{target_file}`:\n"
        ]
        
        for i, entry in enumerate(entries, 1):
            lines.append(f"## Reference {i}")
            lines.append(entry.to_context_string())
            lines.append("")
        
        return "\n".join(lines)
    
    def merge(self, other: "RAGIndex") -> "RAGIndex":
        """Merge another index into this one."""
        merged = RAGIndex(
            entries=self.entries + other.entries,
            source_repos=list(set(self.source_repos + other.source_repos)),
            target_blueprint=self.target_blueprint or other.target_blueprint,
            metadata={**self.metadata, **other.metadata}
        )
        return merged
    
    @classmethod
    def create(
        cls,
        entries: Optional[List[RAGEntry]] = None,
        source_repos: Optional[List[str]] = None,
        target_blueprint: Optional[str] = None
    ) -> "RAGIndex":
        """Factory method for creating RAG index."""
        return cls(
            entries=entries or [],
            source_repos=source_repos or [],
            target_blueprint=target_blueprint
        )


# Type aliases for convenience
RAGContext = Tuple[RAGEntry, str]
RetrievalResult = List[RAGEntry]


def decide_retrieval(
    context_complexity: float,
    blueprint_detail_level: float,
    threshold: float = 0.5
) -> bool:
    """
    Adaptive retrieval decision function δ(Xt, ĉt) ∈ {0,1}.
    
    Decides whether to retrieve reference code based on:
    - Task complexity (higher = more likely to retrieve)
    - Blueprint detail level (lower = more likely to retrieve)
    
    Args:
        context_complexity: Estimated complexity of the target [0,1]
        blueprint_detail_level: How detailed the blueprint is [0,1]
        threshold: Decision threshold
        
    Returns:
        True if retrieval should be performed, False otherwise
    """
    # Simple heuristic: retrieve if complexity is high or detail is low
    retrieval_score = context_complexity * (1 - blueprint_detail_level * 0.5)
    return retrieval_score >= threshold


def calculate_confidence(
    semantic_similarity: float,
    structural_match: float,
    api_overlap: float,
    weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)
) -> float:
    """
    Calculate confidence score for a RAG entry.
    
    Args:
        semantic_similarity: Semantic similarity between source and target [0,1]
        structural_match: Structural similarity (file organization, etc.) [0,1]
        api_overlap: Overlap in API/interface patterns [0,1]
        weights: Weights for each component
        
    Returns:
        Weighted confidence score [0,1]
    """
    w1, w2, w3 = weights
    return w1 * semantic_similarity + w2 * structural_match + w3 * api_overlap
