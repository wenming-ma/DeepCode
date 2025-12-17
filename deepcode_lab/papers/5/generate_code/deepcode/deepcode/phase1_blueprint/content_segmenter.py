"""
Hierarchical Content Segmentation for DeepCode.

This module implements the content segmentation algorithm that processes parsed
documents into indexed structures with keyword-chunk associations for efficient
retrieval by analysis agents.

Algorithm:
    def segment_document(D):
        # Step 1: Structural Parsing
        for element in D:
            if is_section_header(element):
                create_new_chunk(heading=element)
            else:
                append_to_current_chunk(element)
        
        # Step 2: Keyword-Chunk Association
        S = {}
        for chunk in chunks:
            keywords = extract_keywords(chunk.heading)
            S[chunk.heading] = (keywords, chunk.content)
        return S  # Indexed structure
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field

from deepcode.deepcode.models.document import (
    Document,
    DocumentElement,
    DocumentSection,
    ElementType,
    TextBlock,
    Equation,
    Table,
    Figure,
    Pseudocode,
    SectionHeader,
    CodeBlock,
)
from deepcode.deepcode.utils.logger import get_logger

logger = get_logger(__name__)


class ChunkType(str, Enum):
    """Types of content chunks."""
    INTRODUCTION = "introduction"
    BACKGROUND = "background"
    RELATED_WORK = "related_work"
    METHOD = "method"
    ALGORITHM = "algorithm"
    EXPERIMENT = "experiment"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    APPENDIX = "appendix"
    ABSTRACT = "abstract"
    OTHER = "other"


class ContentChunk(BaseModel):
    """A chunk of content with associated metadata."""
    
    chunk_id: str = Field(description="Unique identifier for the chunk")
    heading: str = Field(description="Section heading or title")
    level: int = Field(default=1, description="Heading level (1=top level)")
    chunk_type: ChunkType = Field(default=ChunkType.OTHER, description="Semantic type of chunk")
    keywords: List[str] = Field(default_factory=list, description="Associated keywords")
    elements: List[DocumentElement] = Field(default_factory=list, description="Content elements")
    parent_chunk_id: Optional[str] = Field(default=None, description="Parent chunk ID for hierarchy")
    child_chunk_ids: List[str] = Field(default_factory=list, description="Child chunk IDs")
    section_number: Optional[str] = Field(default=None, description="Section number if available")
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def text_content(self) -> str:
        """Get concatenated text content of all elements."""
        texts = []
        for element in self.elements:
            if hasattr(element, 'to_context_string'):
                texts.append(element.to_context_string())
            elif hasattr(element, 'content'):
                texts.append(str(element.content))
        return "\n\n".join(texts)
    
    @property
    def element_count(self) -> int:
        """Get number of elements in chunk."""
        return len(self.elements)
    
    def get_elements_by_type(self, element_type: ElementType) -> List[DocumentElement]:
        """Get all elements of a specific type."""
        return [e for e in self.elements if e.element_type == element_type]
    
    def has_element_type(self, element_type: ElementType) -> bool:
        """Check if chunk contains elements of a specific type."""
        return any(e.element_type == element_type for e in self.elements)
    
    def to_context_string(self) -> str:
        """Convert chunk to context string for LLM."""
        parts = [f"## {self.heading}"]
        if self.section_number:
            parts[0] = f"## {self.section_number} {self.heading}"
        if self.keywords:
            parts.append(f"Keywords: {', '.join(self.keywords)}")
        parts.append("")
        parts.append(self.text_content)
        return "\n".join(parts)


class IndexedStructure(BaseModel):
    """Indexed structure S mapping headings to (keywords, content) pairs."""
    
    chunks: Dict[str, ContentChunk] = Field(
        default_factory=dict,
        description="Mapping of chunk_id to ContentChunk"
    )
    keyword_index: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Mapping of keywords to chunk_ids"
    )
    type_index: Dict[ChunkType, List[str]] = Field(
        default_factory=dict,
        description="Mapping of chunk types to chunk_ids"
    )
    hierarchy: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Parent to children mapping"
    )
    source_document: Optional[str] = Field(
        default=None,
        description="Source document title"
    )
    
    class Config:
        arbitrary_types_allowed = True
    
    def add_chunk(self, chunk: ContentChunk) -> None:
        """Add a chunk to the indexed structure."""
        self.chunks[chunk.chunk_id] = chunk
        
        # Update keyword index
        for keyword in chunk.keywords:
            keyword_lower = keyword.lower()
            if keyword_lower not in self.keyword_index:
                self.keyword_index[keyword_lower] = []
            if chunk.chunk_id not in self.keyword_index[keyword_lower]:
                self.keyword_index[keyword_lower].append(chunk.chunk_id)
        
        # Update type index
        if chunk.chunk_type not in self.type_index:
            self.type_index[chunk.chunk_type] = []
        if chunk.chunk_id not in self.type_index[chunk.chunk_type]:
            self.type_index[chunk.chunk_type].append(chunk.chunk_id)
        
        # Update hierarchy
        if chunk.parent_chunk_id:
            if chunk.parent_chunk_id not in self.hierarchy:
                self.hierarchy[chunk.parent_chunk_id] = []
            if chunk.chunk_id not in self.hierarchy[chunk.parent_chunk_id]:
                self.hierarchy[chunk.parent_chunk_id].append(chunk.chunk_id)
    
    def get_chunk(self, chunk_id: str) -> Optional[ContentChunk]:
        """Get a chunk by ID."""
        return self.chunks.get(chunk_id)
    
    def get_chunks_by_keyword(self, keyword: str) -> List[ContentChunk]:
        """Get all chunks associated with a keyword."""
        keyword_lower = keyword.lower()
        chunk_ids = self.keyword_index.get(keyword_lower, [])
        return [self.chunks[cid] for cid in chunk_ids if cid in self.chunks]
    
    def get_chunks_by_keywords(self, keywords: List[str]) -> List[ContentChunk]:
        """Get all chunks matching any of the keywords."""
        matching_ids: Set[str] = set()
        for keyword in keywords:
            keyword_lower = keyword.lower()
            matching_ids.update(self.keyword_index.get(keyword_lower, []))
        return [self.chunks[cid] for cid in matching_ids if cid in self.chunks]
    
    def get_chunks_by_type(self, chunk_type: ChunkType) -> List[ContentChunk]:
        """Get all chunks of a specific type."""
        chunk_ids = self.type_index.get(chunk_type, [])
        return [self.chunks[cid] for cid in chunk_ids if cid in self.chunks]
    
    def get_children(self, chunk_id: str) -> List[ContentChunk]:
        """Get child chunks of a parent chunk."""
        child_ids = self.hierarchy.get(chunk_id, [])
        return [self.chunks[cid] for cid in child_ids if cid in self.chunks]
    
    def get_all_chunks(self) -> List[ContentChunk]:
        """Get all chunks in order."""
        return list(self.chunks.values())
    
    def search(
        self,
        keywords: Optional[List[str]] = None,
        chunk_types: Optional[List[ChunkType]] = None,
        element_types: Optional[List[ElementType]] = None,
    ) -> List[ContentChunk]:
        """Search for chunks matching criteria."""
        results: Set[str] = set(self.chunks.keys())
        
        # Filter by keywords
        if keywords:
            keyword_matches: Set[str] = set()
            for keyword in keywords:
                keyword_lower = keyword.lower()
                keyword_matches.update(self.keyword_index.get(keyword_lower, []))
            results &= keyword_matches
        
        # Filter by chunk types
        if chunk_types:
            type_matches: Set[str] = set()
            for ct in chunk_types:
                type_matches.update(self.type_index.get(ct, []))
            results &= type_matches
        
        # Filter by element types
        if element_types:
            element_matches: Set[str] = set()
            for chunk_id in results:
                chunk = self.chunks[chunk_id]
                for et in element_types:
                    if chunk.has_element_type(et):
                        element_matches.add(chunk_id)
                        break
            results &= element_matches
        
        return [self.chunks[cid] for cid in results if cid in self.chunks]
    
    def to_context_string(self, max_chunks: Optional[int] = None) -> str:
        """Convert indexed structure to context string."""
        chunks = list(self.chunks.values())
        if max_chunks:
            chunks = chunks[:max_chunks]
        
        parts = [f"# Document: {self.source_document or 'Unknown'}\n"]
        for chunk in chunks:
            parts.append(chunk.to_context_string())
            parts.append("")
        
        return "\n".join(parts)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the indexed structure."""
        element_counts: Dict[str, int] = {}
        for chunk in self.chunks.values():
            for element in chunk.elements:
                et = element.element_type.value if hasattr(element.element_type, 'value') else str(element.element_type)
                element_counts[et] = element_counts.get(et, 0) + 1
        
        return {
            "total_chunks": len(self.chunks),
            "total_keywords": len(self.keyword_index),
            "chunk_types": {ct.value: len(ids) for ct, ids in self.type_index.items()},
            "element_counts": element_counts,
            "hierarchy_depth": self._calculate_max_depth(),
        }
    
    def _calculate_max_depth(self) -> int:
        """Calculate maximum hierarchy depth."""
        if not self.chunks:
            return 0
        
        max_depth = 1
        for chunk in self.chunks.values():
            depth = 1
            current = chunk
            while current.parent_chunk_id and current.parent_chunk_id in self.chunks:
                depth += 1
                current = self.chunks[current.parent_chunk_id]
            max_depth = max(max_depth, depth)
        
        return max_depth


class ContentSegmenterConfig(BaseModel):
    """Configuration for content segmentation."""
    
    min_chunk_elements: int = Field(
        default=1,
        description="Minimum elements per chunk"
    )
    max_chunk_elements: int = Field(
        default=100,
        description="Maximum elements per chunk before splitting"
    )
    extract_keywords: bool = Field(
        default=True,
        description="Whether to extract keywords from headings"
    )
    keyword_extraction_method: str = Field(
        default="pattern",
        description="Method for keyword extraction: 'pattern' or 'simple'"
    )
    merge_small_chunks: bool = Field(
        default=True,
        description="Whether to merge very small chunks"
    )
    preserve_hierarchy: bool = Field(
        default=True,
        description="Whether to preserve section hierarchy"
    )


# Keyword patterns for chunk type classification
CHUNK_TYPE_PATTERNS: Dict[ChunkType, List[str]] = {
    ChunkType.INTRODUCTION: [
        r"introduction", r"intro\b", r"overview", r"motivation"
    ],
    ChunkType.BACKGROUND: [
        r"background", r"preliminar", r"notation", r"definition"
    ],
    ChunkType.RELATED_WORK: [
        r"related\s*work", r"prior\s*work", r"literature", r"previous"
    ],
    ChunkType.METHOD: [
        r"method", r"approach", r"framework", r"architecture", r"model",
        r"proposed", r"our\s*approach", r"technique", r"system"
    ],
    ChunkType.ALGORITHM: [
        r"algorithm", r"procedure", r"pseudocode", r"implementation"
    ],
    ChunkType.EXPERIMENT: [
        r"experiment", r"evaluation", r"setup", r"dataset", r"benchmark"
    ],
    ChunkType.RESULTS: [
        r"result", r"finding", r"performance", r"comparison", r"analysis"
    ],
    ChunkType.DISCUSSION: [
        r"discussion", r"limitation", r"future\s*work", r"ablation"
    ],
    ChunkType.CONCLUSION: [
        r"conclusion", r"summary", r"concluding"
    ],
    ChunkType.APPENDIX: [
        r"appendix", r"supplement", r"additional"
    ],
    ChunkType.ABSTRACT: [
        r"abstract"
    ],
}

# Keywords to extract from content
CONCEPT_KEYWORDS = [
    "introduction", "method", "overview", "architecture", "framework",
    "model", "approach", "system", "design", "structure"
]

ALGORITHM_KEYWORDS = [
    "algorithm", "hyperparameter", "equation", "training", "loss",
    "optimization", "learning", "inference", "forward", "backward",
    "gradient", "update", "iteration", "convergence"
]

TECHNICAL_KEYWORDS = [
    "neural", "network", "layer", "attention", "transformer",
    "convolution", "embedding", "encoder", "decoder", "classifier",
    "regression", "classification", "segmentation", "detection"
]


class ContentSegmenter:
    """
    Hierarchical Content Segmenter for document analysis.
    
    Implements the segmentation algorithm:
    1. Structural Parsing - identify section boundaries
    2. Keyword-Chunk Association - extract and index keywords
    """
    
    def __init__(self, config: Optional[ContentSegmenterConfig] = None):
        """Initialize the content segmenter."""
        self.config = config or ContentSegmenterConfig()
        self._chunk_counter = 0
    
    def segment(self, document: Document) -> IndexedStructure:
        """
        Segment a document into indexed chunks.
        
        Args:
            document: Parsed document to segment
            
        Returns:
            IndexedStructure with keyword-chunk associations
        """
        logger.info(f"Segmenting document: {document.title}")
        
        self._chunk_counter = 0
        indexed = IndexedStructure(source_document=document.title)
        
        # Process document sections if available
        if document.sections:
            self._process_sections(document.sections, indexed, parent_id=None)
        
        # Process flat elements if no sections or as supplement
        if document.elements and (not document.sections or not indexed.chunks):
            self._process_flat_elements(document.elements, indexed)
        
        # Post-process: merge small chunks if configured
        if self.config.merge_small_chunks:
            self._merge_small_chunks(indexed)
        
        stats = indexed.get_statistics()
        logger.info(
            f"Segmentation complete: {stats['total_chunks']} chunks, "
            f"{stats['total_keywords']} keywords"
        )
        
        return indexed
    
    def _process_sections(
        self,
        sections: List[DocumentSection],
        indexed: IndexedStructure,
        parent_id: Optional[str],
    ) -> None:
        """Process document sections recursively."""
        for section in sections:
            chunk = self._create_chunk_from_section(section, parent_id)
            indexed.add_chunk(chunk)
            
            # Process subsections
            if section.subsections and self.config.preserve_hierarchy:
                self._process_sections(
                    section.subsections,
                    indexed,
                    parent_id=chunk.chunk_id
                )
    
    def _process_flat_elements(
        self,
        elements: List[DocumentElement],
        indexed: IndexedStructure,
    ) -> None:
        """Process flat list of elements into chunks."""
        current_chunk_elements: List[DocumentElement] = []
        current_heading = "Document Content"
        current_level = 1
        
        for element in elements:
            # Check if this is a section header
            if element.element_type == ElementType.SECTION_HEADER:
                # Save current chunk if it has content
                if current_chunk_elements:
                    chunk = self._create_chunk(
                        heading=current_heading,
                        level=current_level,
                        elements=current_chunk_elements,
                    )
                    indexed.add_chunk(chunk)
                    current_chunk_elements = []
                
                # Start new chunk
                current_heading = element.content
                current_level = getattr(element, 'level', 1)
            else:
                current_chunk_elements.append(element)
        
        # Don't forget the last chunk
        if current_chunk_elements:
            chunk = self._create_chunk(
                heading=current_heading,
                level=current_level,
                elements=current_chunk_elements,
            )
            indexed.add_chunk(chunk)
    
    def _create_chunk_from_section(
        self,
        section: DocumentSection,
        parent_id: Optional[str],
    ) -> ContentChunk:
        """Create a content chunk from a document section."""
        chunk_id = self._generate_chunk_id()
        
        # Determine chunk type from heading
        chunk_type = self._classify_chunk_type(section.title)
        
        # Extract keywords
        keywords = self._extract_keywords(section.title, section.elements)
        
        return ContentChunk(
            chunk_id=chunk_id,
            heading=section.title,
            level=section.level,
            chunk_type=chunk_type,
            keywords=keywords,
            elements=section.elements,
            parent_chunk_id=parent_id,
            section_number=section.section_number,
        )
    
    def _create_chunk(
        self,
        heading: str,
        level: int,
        elements: List[DocumentElement],
        parent_id: Optional[str] = None,
        section_number: Optional[str] = None,
    ) -> ContentChunk:
        """Create a content chunk from elements."""
        chunk_id = self._generate_chunk_id()
        chunk_type = self._classify_chunk_type(heading)
        keywords = self._extract_keywords(heading, elements)
        
        return ContentChunk(
            chunk_id=chunk_id,
            heading=heading,
            level=level,
            chunk_type=chunk_type,
            keywords=keywords,
            elements=elements,
            parent_chunk_id=parent_id,
            section_number=section_number,
        )
    
    def _generate_chunk_id(self) -> str:
        """Generate a unique chunk ID."""
        self._chunk_counter += 1
        return f"chunk_{self._chunk_counter:04d}"
    
    def _classify_chunk_type(self, heading: str) -> ChunkType:
        """Classify chunk type based on heading."""
        heading_lower = heading.lower()
        
        for chunk_type, patterns in CHUNK_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, heading_lower):
                    return chunk_type
        
        return ChunkType.OTHER
    
    def _extract_keywords(
        self,
        heading: str,
        elements: List[DocumentElement],
    ) -> List[str]:
        """Extract keywords from heading and content."""
        if not self.config.extract_keywords:
            return []
        
        keywords: Set[str] = set()
        
        # Extract from heading
        heading_keywords = self._extract_keywords_from_text(heading)
        keywords.update(heading_keywords)
        
        # Extract from content elements
        for element in elements:
            if element.element_type == ElementType.TEXT:
                content = element.content if hasattr(element, 'content') else ""
                content_keywords = self._extract_keywords_from_text(content)
                keywords.update(content_keywords)
            elif element.element_type == ElementType.PSEUDOCODE:
                keywords.add("algorithm")
                keywords.add("pseudocode")
                if hasattr(element, 'algorithm_name') and element.algorithm_name:
                    keywords.add(element.algorithm_name.lower())
            elif element.element_type == ElementType.EQUATION:
                keywords.add("equation")
                keywords.add("mathematical")
            elif element.element_type == ElementType.TABLE:
                keywords.add("table")
                keywords.add("data")
            elif element.element_type == ElementType.FIGURE:
                keywords.add("figure")
                keywords.add("visualization")
        
        return list(keywords)
    
    def _extract_keywords_from_text(self, text: str) -> Set[str]:
        """Extract keywords from text using pattern matching."""
        keywords: Set[str] = set()
        text_lower = text.lower()
        
        # Check for concept keywords
        for keyword in CONCEPT_KEYWORDS:
            if keyword in text_lower:
                keywords.add(keyword)
        
        # Check for algorithm keywords
        for keyword in ALGORITHM_KEYWORDS:
            if keyword in text_lower:
                keywords.add(keyword)
        
        # Check for technical keywords
        for keyword in TECHNICAL_KEYWORDS:
            if keyword in text_lower:
                keywords.add(keyword)
        
        # Extract capitalized terms (potential proper nouns/methods)
        capitalized = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', text)
        for term in capitalized:
            if len(term) > 3 and term.lower() not in {'the', 'this', 'that', 'these', 'those'}:
                keywords.add(term.lower())
        
        return keywords
    
    def _merge_small_chunks(self, indexed: IndexedStructure) -> None:
        """Merge chunks that are too small."""
        # This is a simplified implementation
        # In practice, you might want more sophisticated merging logic
        pass


def segment_document(document: Document, config: Optional[ContentSegmenterConfig] = None) -> IndexedStructure:
    """
    Segment a document into indexed chunks.
    
    This is the main entry point for document segmentation.
    
    Args:
        document: Parsed document to segment
        config: Optional configuration
        
    Returns:
        IndexedStructure with keyword-chunk associations
    """
    segmenter = ContentSegmenter(config)
    return segmenter.segment(document)


def get_chunks_for_concept_analysis(indexed: IndexedStructure) -> List[ContentChunk]:
    """
    Get chunks relevant for concept analysis.
    
    Returns chunks matching concept-related keywords:
    ["introduction", "method", "overview", "architecture", "framework"]
    """
    return indexed.search(
        keywords=CONCEPT_KEYWORDS,
        chunk_types=[
            ChunkType.INTRODUCTION,
            ChunkType.METHOD,
            ChunkType.BACKGROUND,
        ]
    )


def get_chunks_for_algorithm_analysis(indexed: IndexedStructure) -> List[ContentChunk]:
    """
    Get chunks relevant for algorithm analysis.
    
    Returns chunks matching algorithm-related keywords:
    ["algorithm", "hyperparameter", "equation", "training", "loss"]
    """
    return indexed.search(
        keywords=ALGORITHM_KEYWORDS,
        chunk_types=[ChunkType.ALGORITHM, ChunkType.METHOD],
        element_types=[ElementType.PSEUDOCODE, ElementType.EQUATION],
    )


# Convenience exports
__all__ = [
    "ContentSegmenter",
    "ContentSegmenterConfig",
    "ContentChunk",
    "ChunkType",
    "IndexedStructure",
    "segment_document",
    "get_chunks_for_concept_analysis",
    "get_chunks_for_algorithm_analysis",
    "CONCEPT_KEYWORDS",
    "ALGORITHM_KEYWORDS",
    "TECHNICAL_KEYWORDS",
]
