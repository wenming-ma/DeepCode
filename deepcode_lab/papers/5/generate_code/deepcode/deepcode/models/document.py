"""
Source Document Model for DeepCode.

Defines the document structure D = (d1, d2, ..., dL) as a sequence of multimodal elements.
Element types include: text blocks, equations, tables, figures, and pseudocode.
"""

from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


class ElementType(str, Enum):
    """Types of elements that can appear in a source document."""
    TEXT = "text"
    EQUATION = "equation"
    TABLE = "table"
    FIGURE = "figure"
    PSEUDOCODE = "pseudocode"
    SECTION_HEADER = "section_header"
    CITATION = "citation"
    LIST = "list"
    CODE_BLOCK = "code_block"


class DocumentElement(BaseModel):
    """
    Base class for all document elements.
    
    Each element di in D = (d1, ..., dL) represents a discrete unit
    of content from the source document.
    """
    element_type: ElementType
    content: str
    position: int = Field(default=0, description="Position index in document")
    page_number: Optional[int] = Field(default=None, description="Source page number")
    section: Optional[str] = Field(default=None, description="Parent section name")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"{self.element_type.value}: {self.content[:50]}..."
    
    def to_context_string(self) -> str:
        """Convert element to string suitable for LLM context."""
        prefix = f"[{self.element_type.value.upper()}]"
        if self.section:
            prefix = f"[{self.section}] {prefix}"
        return f"{prefix}\n{self.content}"


class TextBlock(DocumentElement):
    """
    A block of natural language text from the document.
    
    Represents paragraphs, descriptions, and prose content.
    """
    element_type: ElementType = ElementType.TEXT
    is_abstract: bool = Field(default=False, description="Whether this is abstract text")
    is_conclusion: bool = Field(default=False, description="Whether this is conclusion text")
    
    @classmethod
    def create(cls, content: str, **kwargs) -> "TextBlock":
        return cls(content=content, **kwargs)


class Equation(DocumentElement):
    """
    A mathematical equation from the document.
    
    Captures both inline and display equations with their LaTeX representation.
    """
    element_type: ElementType = ElementType.EQUATION
    latex: Optional[str] = Field(default=None, description="LaTeX representation")
    equation_number: Optional[str] = Field(default=None, description="Equation reference number")
    variables: List[str] = Field(default_factory=list, description="Variables used in equation")
    is_inline: bool = Field(default=False, description="Whether equation is inline")
    
    @classmethod
    def create(cls, content: str, latex: Optional[str] = None, **kwargs) -> "Equation":
        return cls(content=content, latex=latex or content, **kwargs)
    
    def to_context_string(self) -> str:
        """Convert equation to string with LaTeX if available."""
        if self.latex:
            return f"[EQUATION {self.equation_number or ''}]\n$$\n{self.latex}\n$$"
        return f"[EQUATION {self.equation_number or ''}]\n{self.content}"


class Table(DocumentElement):
    """
    A table from the document.
    
    Stores tabular data with headers and rows for structured information
    like hyperparameters, experimental results, etc.
    """
    element_type: ElementType = ElementType.TABLE
    caption: Optional[str] = Field(default=None, description="Table caption")
    table_number: Optional[str] = Field(default=None, description="Table reference number")
    headers: List[str] = Field(default_factory=list, description="Column headers")
    rows: List[List[str]] = Field(default_factory=list, description="Table data rows")
    
    @classmethod
    def create(
        cls, 
        content: str, 
        headers: Optional[List[str]] = None,
        rows: Optional[List[List[str]]] = None,
        **kwargs
    ) -> "Table":
        return cls(
            content=content, 
            headers=headers or [], 
            rows=rows or [],
            **kwargs
        )
    
    def to_dict(self) -> List[Dict[str, str]]:
        """Convert table to list of dictionaries."""
        if not self.headers or not self.rows:
            return []
        return [dict(zip(self.headers, row)) for row in self.rows]
    
    def to_context_string(self) -> str:
        """Convert table to markdown format for LLM context."""
        lines = [f"[TABLE {self.table_number or ''}]"]
        if self.caption:
            lines.append(f"Caption: {self.caption}")
        if self.headers:
            lines.append("| " + " | ".join(self.headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(self.headers)) + " |")
            for row in self.rows:
                lines.append("| " + " | ".join(row) + " |")
        else:
            lines.append(self.content)
        return "\n".join(lines)


class Figure(DocumentElement):
    """
    A figure from the document.
    
    Represents diagrams, charts, architecture visualizations, etc.
    """
    element_type: ElementType = ElementType.FIGURE
    caption: Optional[str] = Field(default=None, description="Figure caption")
    figure_number: Optional[str] = Field(default=None, description="Figure reference number")
    image_path: Optional[str] = Field(default=None, description="Path to extracted image")
    alt_text: Optional[str] = Field(default=None, description="Alternative text description")
    
    @classmethod
    def create(cls, content: str, caption: Optional[str] = None, **kwargs) -> "Figure":
        return cls(content=content, caption=caption, **kwargs)
    
    def to_context_string(self) -> str:
        """Convert figure to string for LLM context."""
        lines = [f"[FIGURE {self.figure_number or ''}]"]
        if self.caption:
            lines.append(f"Caption: {self.caption}")
        if self.alt_text:
            lines.append(f"Description: {self.alt_text}")
        elif self.content:
            lines.append(f"Description: {self.content}")
        return "\n".join(lines)


class Pseudocode(DocumentElement):
    """
    Pseudocode or algorithm box from the document.
    
    Captures algorithmic descriptions that need to be translated to real code.
    """
    element_type: ElementType = ElementType.PSEUDOCODE
    algorithm_name: Optional[str] = Field(default=None, description="Algorithm name/title")
    algorithm_number: Optional[str] = Field(default=None, description="Algorithm reference number")
    inputs: List[str] = Field(default_factory=list, description="Input parameters")
    outputs: List[str] = Field(default_factory=list, description="Output values")
    steps: List[str] = Field(default_factory=list, description="Algorithm steps")
    
    @classmethod
    def create(
        cls, 
        content: str, 
        algorithm_name: Optional[str] = None,
        steps: Optional[List[str]] = None,
        **kwargs
    ) -> "Pseudocode":
        return cls(
            content=content, 
            algorithm_name=algorithm_name,
            steps=steps or [],
            **kwargs
        )
    
    def to_context_string(self) -> str:
        """Convert pseudocode to string for LLM context."""
        lines = [f"[ALGORITHM {self.algorithm_number or ''}: {self.algorithm_name or 'Unnamed'}]"]
        if self.inputs:
            lines.append(f"Input: {', '.join(self.inputs)}")
        if self.outputs:
            lines.append(f"Output: {', '.join(self.outputs)}")
        lines.append("```")
        if self.steps:
            for i, step in enumerate(self.steps, 1):
                lines.append(f"{i}. {step}")
        else:
            lines.append(self.content)
        lines.append("```")
        return "\n".join(lines)


class SectionHeader(DocumentElement):
    """
    A section or subsection header from the document.
    
    Used for hierarchical document structure.
    """
    element_type: ElementType = ElementType.SECTION_HEADER
    level: int = Field(default=1, description="Header level (1=section, 2=subsection, etc.)")
    section_number: Optional[str] = Field(default=None, description="Section number (e.g., '3.1')")
    
    @classmethod
    def create(cls, content: str, level: int = 1, **kwargs) -> "SectionHeader":
        return cls(content=content, level=level, **kwargs)
    
    def to_context_string(self) -> str:
        """Convert header to markdown format."""
        prefix = "#" * self.level
        num = f"{self.section_number} " if self.section_number else ""
        return f"{prefix} {num}{self.content}"


class Citation(DocumentElement):
    """
    A citation reference from the document.
    
    Tracks references to other papers that may contain implementation details.
    """
    element_type: ElementType = ElementType.CITATION
    citation_key: Optional[str] = Field(default=None, description="Citation key (e.g., '[1]')")
    authors: List[str] = Field(default_factory=list, description="Author names")
    title: Optional[str] = Field(default=None, description="Cited paper title")
    year: Optional[int] = Field(default=None, description="Publication year")
    url: Optional[str] = Field(default=None, description="URL if available")
    
    @classmethod
    def create(cls, content: str, citation_key: Optional[str] = None, **kwargs) -> "Citation":
        return cls(content=content, citation_key=citation_key, **kwargs)


class CodeBlock(DocumentElement):
    """
    A code block from the document.
    
    Represents actual code snippets included in the paper.
    """
    element_type: ElementType = ElementType.CODE_BLOCK
    language: Optional[str] = Field(default=None, description="Programming language")
    
    @classmethod
    def create(cls, content: str, language: Optional[str] = None, **kwargs) -> "CodeBlock":
        return cls(content=content, language=language, **kwargs)
    
    def to_context_string(self) -> str:
        """Convert code block to markdown format."""
        lang = self.language or ""
        return f"```{lang}\n{self.content}\n```"


# Type alias for any document element
AnyElement = Union[
    TextBlock, Equation, Table, Figure, Pseudocode, 
    SectionHeader, Citation, CodeBlock, DocumentElement
]


class DocumentSection(BaseModel):
    """
    A section of the document containing related elements.
    
    Used for hierarchical organization of document content.
    """
    title: str
    level: int = Field(default=1, description="Section nesting level")
    section_number: Optional[str] = Field(default=None)
    elements: List[DocumentElement] = Field(default_factory=list)
    subsections: List["DocumentSection"] = Field(default_factory=list)
    
    def get_all_elements(self) -> List[DocumentElement]:
        """Get all elements including from subsections."""
        all_elements = list(self.elements)
        for subsection in self.subsections:
            all_elements.extend(subsection.get_all_elements())
        return all_elements
    
    def get_elements_by_type(self, element_type: ElementType) -> List[DocumentElement]:
        """Get all elements of a specific type."""
        return [e for e in self.get_all_elements() if e.element_type == element_type]
    
    def to_context_string(self) -> str:
        """Convert section to string for LLM context."""
        lines = [f"{'#' * self.level} {self.section_number or ''} {self.title}".strip()]
        for element in self.elements:
            lines.append(element.to_context_string())
        for subsection in self.subsections:
            lines.append(subsection.to_context_string())
        return "\n\n".join(lines)


class Document(BaseModel):
    """
    Source document model D = (d1, d2, ..., dL).
    
    Represents a complete research paper or technical document as a sequence
    of multimodal elements. This is the primary input to the DeepCode pipeline.
    
    Attributes:
        title: Document title
        authors: List of author names
        abstract: Document abstract text
        elements: Ordered sequence of document elements
        sections: Hierarchical section structure
        metadata: Additional document metadata
    """
    title: str = Field(default="", description="Document title")
    authors: List[str] = Field(default_factory=list, description="Author names")
    abstract: Optional[str] = Field(default=None, description="Document abstract")
    elements: List[DocumentElement] = Field(
        default_factory=list, 
        description="Ordered sequence of all elements D=(d1,...,dL)"
    )
    sections: List[DocumentSection] = Field(
        default_factory=list,
        description="Hierarchical section structure"
    )
    source_path: Optional[str] = Field(default=None, description="Path to source file")
    source_type: Optional[str] = Field(default=None, description="Source type (pdf, markdown, etc.)")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def L(self) -> int:
        """Return the number of elements in the document."""
        return len(self.elements)
    
    def __len__(self) -> int:
        return self.L
    
    def __iter__(self):
        return iter(self.elements)
    
    def __getitem__(self, index: int) -> DocumentElement:
        return self.elements[index]
    
    def add_element(self, element: DocumentElement) -> None:
        """Add an element to the document."""
        element.position = len(self.elements)
        self.elements.append(element)
    
    def get_elements_by_type(self, element_type: ElementType) -> List[DocumentElement]:
        """Get all elements of a specific type."""
        return [e for e in self.elements if e.element_type == element_type]
    
    def get_text_blocks(self) -> List[TextBlock]:
        """Get all text blocks."""
        return [e for e in self.elements if isinstance(e, TextBlock)]
    
    def get_equations(self) -> List[Equation]:
        """Get all equations."""
        return [e for e in self.elements if isinstance(e, Equation)]
    
    def get_tables(self) -> List[Table]:
        """Get all tables."""
        return [e for e in self.elements if isinstance(e, Table)]
    
    def get_figures(self) -> List[Figure]:
        """Get all figures."""
        return [e for e in self.elements if isinstance(e, Figure)]
    
    def get_pseudocode(self) -> List[Pseudocode]:
        """Get all pseudocode/algorithm blocks."""
        return [e for e in self.elements if isinstance(e, Pseudocode)]
    
    def get_section_headers(self) -> List[SectionHeader]:
        """Get all section headers."""
        return [e for e in self.elements if isinstance(e, SectionHeader)]
    
    def get_citations(self) -> List[Citation]:
        """Get all citations."""
        return [e for e in self.elements if isinstance(e, Citation)]
    
    def get_elements_in_section(self, section_name: str) -> List[DocumentElement]:
        """Get all elements belonging to a specific section."""
        return [e for e in self.elements if e.section == section_name]
    
    def get_elements_by_keywords(self, keywords: List[str]) -> List[DocumentElement]:
        """
        Get elements whose content contains any of the specified keywords.
        
        This implements the keyword-chunk association from the paper's
        hierarchical content segmentation algorithm.
        """
        keywords_lower = [k.lower() for k in keywords]
        matching = []
        for element in self.elements:
            content_lower = element.content.lower()
            if any(kw in content_lower for kw in keywords_lower):
                matching.append(element)
        return matching
    
    def to_context_string(self, max_length: Optional[int] = None) -> str:
        """
        Convert document to a string suitable for LLM context.
        
        Args:
            max_length: Optional maximum character length
            
        Returns:
            Formatted string representation of the document
        """
        lines = []
        
        # Header
        if self.title:
            lines.append(f"# {self.title}")
        if self.authors:
            lines.append(f"Authors: {', '.join(self.authors)}")
        if self.abstract:
            lines.append(f"\n## Abstract\n{self.abstract}")
        
        lines.append("\n## Content\n")
        
        # Elements
        for element in self.elements:
            lines.append(element.to_context_string())
            lines.append("")  # Empty line between elements
        
        result = "\n".join(lines)
        
        if max_length and len(result) > max_length:
            result = result[:max_length] + "\n... [truncated]"
        
        return result
    
    def to_indexed_structure(self) -> Dict[str, tuple]:
        """
        Convert document to indexed structure S for hierarchical content segmentation.
        
        Returns:
            Dictionary mapping section headings to (keywords, content) tuples
            as described in the paper's segmentation algorithm.
        """
        S = {}
        current_section = "Introduction"
        current_content = []
        
        for element in self.elements:
            if isinstance(element, SectionHeader):
                # Save previous section
                if current_content:
                    keywords = self._extract_keywords(current_section)
                    S[current_section] = (keywords, current_content.copy())
                # Start new section
                current_section = element.content
                current_content = []
            else:
                current_content.append(element)
        
        # Save last section
        if current_content:
            keywords = self._extract_keywords(current_section)
            S[current_section] = (keywords, current_content)
        
        return S
    
    def _extract_keywords(self, heading: str) -> List[str]:
        """Extract keywords from a section heading."""
        # Simple keyword extraction - split on spaces and filter
        words = heading.lower().replace(":", "").replace("-", " ").split()
        # Filter out common words
        stopwords = {"the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "with"}
        return [w for w in words if w not in stopwords and len(w) > 2]
    
    @classmethod
    def from_elements(
        cls, 
        elements: List[DocumentElement],
        title: str = "",
        **kwargs
    ) -> "Document":
        """Create a document from a list of elements."""
        doc = cls(title=title, **kwargs)
        for i, element in enumerate(elements):
            element.position = i
            doc.elements.append(element)
        return doc
    
    def model_dump_json(self, **kwargs) -> str:
        """Serialize document to JSON string."""
        return super().model_dump_json(**kwargs)
    
    @classmethod
    def model_validate_json(cls, json_data: str) -> "Document":
        """Deserialize document from JSON string."""
        return super().model_validate_json(json_data)


# Update forward references for recursive models
DocumentSection.model_rebuild()
