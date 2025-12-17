"""
DeepCode Phase 1: Blueprint Generation

This module implements the Blueprint Generation phase of the DeepCode pipeline,
which transforms source documents into structured implementation blueprints through:

1. Document Parsing - Convert PDFs/Markdown to structured Document model
2. Content Segmentation - Hierarchical indexing with keyword-chunk associations
3. Concept Analysis - High-level structural and conceptual mapping
4. Algorithm Extraction - Low-level technical detail extraction
5. Planning - Blueprint synthesis with dependency resolution

The output is an Implementation Blueprint (B) containing:
- Project file hierarchy
- Component specifications
- Verification protocol
- Execution environment
- Staged development plan
"""

from deepcode.deepcode.phase1_blueprint.document_parser import (
    DocumentParser,
    DocumentParserConfig,
    parse_document,
    parse_pdf,
    parse_markdown,
    parse_content,
)

from deepcode.deepcode.phase1_blueprint.content_segmenter import (
    ContentSegmenter,
    ContentSegmenterConfig,
    ContentChunk,
    IndexedStructure,
    ChunkType,
    segment_document,
    get_chunks_for_concept_analysis,
    get_chunks_for_algorithm_analysis,
    CONCEPT_KEYWORDS,
    ALGORITHM_KEYWORDS,
    TECHNICAL_KEYWORDS,
)

# These will be imported once implemented
# Placeholder imports with lazy loading pattern
def __getattr__(name: str):
    """Lazy loading for components not yet implemented."""
    
    # Concept Agent components
    if name in ('ConceptAgent', 'ConceptAgentConfig', 'ConceptualAnalysisSchema'):
        from deepcode.deepcode.phase1_blueprint.concept_agent import (
            ConceptAgent,
            ConceptAgentConfig,
            ConceptualAnalysisSchema,
        )
        return locals()[name]
    
    # Algorithm Agent components
    if name in ('AlgorithmAgent', 'AlgorithmAgentConfig', 'AlgorithmicImplementationSchema'):
        from deepcode.deepcode.phase1_blueprint.algorithm_agent import (
            AlgorithmAgent,
            AlgorithmAgentConfig,
            AlgorithmicImplementationSchema,
        )
        return locals()[name]
    
    # Planning Agent components
    if name in ('PlanningAgent', 'PlanningAgentConfig'):
        from deepcode.deepcode.phase1_blueprint.planning_agent import (
            PlanningAgent,
            PlanningAgentConfig,
        )
        return locals()[name]
    
    # Schema components
    if name in (
        'PaperStructureMap',
        'MethodDecompositionMap', 
        'ImplementationMap',
        'ReproductionRoadmap',
        'PseudocodeSpec',
        'EquationSpec',
        'NetworkArchitectureSpec',
        'HyperparameterSpec',
    ):
        from deepcode.deepcode.phase1_blueprint.schemas import (
            PaperStructureMap,
            MethodDecompositionMap,
            ImplementationMap,
            ReproductionRoadmap,
            PseudocodeSpec,
            EquationSpec,
            NetworkArchitectureSpec,
            HyperparameterSpec,
        )
        return locals()[name]
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Public API
__all__ = [
    # Document Parser
    "DocumentParser",
    "DocumentParserConfig",
    "parse_document",
    "parse_pdf",
    "parse_markdown",
    "parse_content",
    
    # Content Segmenter
    "ContentSegmenter",
    "ContentSegmenterConfig",
    "ContentChunk",
    "IndexedStructure",
    "ChunkType",
    "segment_document",
    "get_chunks_for_concept_analysis",
    "get_chunks_for_algorithm_analysis",
    "CONCEPT_KEYWORDS",
    "ALGORITHM_KEYWORDS",
    "TECHNICAL_KEYWORDS",
    
    # Concept Agent (lazy loaded)
    "ConceptAgent",
    "ConceptAgentConfig",
    "ConceptualAnalysisSchema",
    
    # Algorithm Agent (lazy loaded)
    "AlgorithmAgent",
    "AlgorithmAgentConfig",
    "AlgorithmicImplementationSchema",
    
    # Planning Agent (lazy loaded)
    "PlanningAgent",
    "PlanningAgentConfig",
    
    # Schemas (lazy loaded)
    "PaperStructureMap",
    "MethodDecompositionMap",
    "ImplementationMap",
    "ReproductionRoadmap",
    "PseudocodeSpec",
    "EquationSpec",
    "NetworkArchitectureSpec",
    "HyperparameterSpec",
]
