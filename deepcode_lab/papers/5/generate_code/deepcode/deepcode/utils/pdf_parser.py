"""
PDF Parser Utility Module for DeepCode.

This module provides comprehensive PDF parsing capabilities for extracting
text, structure, and multimodal elements from research papers and technical
documents. It supports extraction of sections, equations, tables, figures,
and pseudocode blocks.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field


class PDFElementType(Enum):
    """Types of elements that can be extracted from PDFs."""
    TEXT = "text"
    HEADING = "heading"
    EQUATION = "equation"
    TABLE = "table"
    FIGURE = "figure"
    PSEUDOCODE = "pseudocode"
    CODE = "code"
    LIST = "list"
    CITATION = "citation"
    CAPTION = "caption"
    ABSTRACT = "abstract"
    UNKNOWN = "unknown"


class PDFParserConfig(BaseModel):
    """Configuration for PDF parsing."""
    
    # Extraction options
    extract_images: bool = Field(default=False, description="Extract images from PDF")
    extract_tables: bool = Field(default=True, description="Extract tables from PDF")
    extract_equations: bool = Field(default=True, description="Extract equations from PDF")
    extract_code: bool = Field(default=True, description="Extract code blocks from PDF")
    
    # Processing options
    preserve_layout: bool = Field(default=False, description="Preserve original layout")
    merge_lines: bool = Field(default=True, description="Merge broken lines")
    remove_headers_footers: bool = Field(default=True, description="Remove headers/footers")
    remove_page_numbers: bool = Field(default=True, description="Remove page numbers")
    
    # Section detection
    detect_sections: bool = Field(default=True, description="Detect section structure")
    section_patterns: List[str] = Field(
        default=[
            r"^\d+\.?\s+[A-Z]",  # "1. Introduction" or "1 Introduction"
            r"^[IVX]+\.?\s+[A-Z]",  # Roman numerals
            r"^(?:Abstract|Introduction|Related Work|Method|Methodology|Approach|"
            r"Experiments?|Results?|Discussion|Conclusion|References|Appendix)",
        ],
        description="Regex patterns for section headers"
    )
    
    # Output options
    max_pages: Optional[int] = Field(default=None, description="Maximum pages to process")
    output_format: str = Field(default="structured", description="Output format: raw, structured")
    
    class Config:
        extra = "allow"


class PDFBoundingBox(BaseModel):
    """Bounding box for PDF elements."""
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    page: int = 0
    
    @property
    def width(self) -> float:
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        return self.y1 - self.y0
    
    @property
    def area(self) -> float:
        return self.width * self.height


class PDFElement(BaseModel):
    """A single element extracted from a PDF."""
    element_type: PDFElementType = PDFElementType.TEXT
    content: str = ""
    page_number: int = 0
    position: Optional[PDFBoundingBox] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_text(self) -> str:
        """Convert element to plain text representation."""
        if self.element_type == PDFElementType.HEADING:
            return f"\n## {self.content}\n"
        elif self.element_type == PDFElementType.EQUATION:
            return f"\n$${self.content}$$\n"
        elif self.element_type == PDFElementType.CODE:
            lang = self.metadata.get("language", "")
            return f"\n```{lang}\n{self.content}\n```\n"
        elif self.element_type == PDFElementType.TABLE:
            return f"\n[TABLE: {self.content}]\n"
        elif self.element_type == PDFElementType.FIGURE:
            caption = self.metadata.get("caption", "")
            return f"\n[FIGURE: {caption or self.content}]\n"
        elif self.element_type == PDFElementType.PSEUDOCODE:
            name = self.metadata.get("algorithm_name", "Algorithm")
            return f"\n[ALGORITHM: {name}]\n{self.content}\n"
        else:
            return self.content


class PDFPage(BaseModel):
    """Represents a single page from a PDF."""
    page_number: int
    width: float = 0.0
    height: float = 0.0
    text: str = ""
    elements: List[PDFElement] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def get_elements_by_type(self, element_type: PDFElementType) -> List[PDFElement]:
        """Get all elements of a specific type."""
        return [e for e in self.elements if e.element_type == element_type]
    
    def to_text(self) -> str:
        """Convert page to plain text."""
        if self.elements:
            return "\n".join(e.to_text() for e in self.elements)
        return self.text


class PDFSection(BaseModel):
    """A section extracted from a PDF document."""
    title: str
    level: int = 1
    section_number: Optional[str] = None
    content: str = ""
    elements: List[PDFElement] = Field(default_factory=list)
    subsections: List["PDFSection"] = Field(default_factory=list)
    start_page: int = 0
    end_page: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def get_full_text(self) -> str:
        """Get full text including subsections."""
        parts = [self.content]
        for subsection in self.subsections:
            parts.append(f"\n### {subsection.title}\n")
            parts.append(subsection.get_full_text())
        return "\n".join(parts)
    
    def get_all_elements(self) -> List[PDFElement]:
        """Get all elements including from subsections."""
        all_elements = list(self.elements)
        for subsection in self.subsections:
            all_elements.extend(subsection.get_all_elements())
        return all_elements


class PDFParseResult(BaseModel):
    """Result of parsing a PDF document."""
    success: bool = True
    source_path: str = ""
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    abstract: Optional[str] = None
    pages: List[PDFPage] = Field(default_factory=list)
    sections: List[PDFSection] = Field(default_factory=list)
    elements: List[PDFElement] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    @property
    def page_count(self) -> int:
        return len(self.pages)
    
    @property
    def text(self) -> str:
        """Get full document text."""
        return "\n\n".join(page.to_text() for page in self.pages)
    
    def get_section(self, title: str) -> Optional[PDFSection]:
        """Find a section by title (case-insensitive partial match)."""
        title_lower = title.lower()
        for section in self.sections:
            if title_lower in section.title.lower():
                return section
            for subsection in section.subsections:
                if title_lower in subsection.title.lower():
                    return subsection
        return None
    
    def get_elements_by_type(self, element_type: PDFElementType) -> List[PDFElement]:
        """Get all elements of a specific type across all pages."""
        return [e for e in self.elements if e.element_type == element_type]
    
    def get_equations(self) -> List[PDFElement]:
        """Get all equations from the document."""
        return self.get_elements_by_type(PDFElementType.EQUATION)
    
    def get_tables(self) -> List[PDFElement]:
        """Get all tables from the document."""
        return self.get_elements_by_type(PDFElementType.TABLE)
    
    def get_figures(self) -> List[PDFElement]:
        """Get all figures from the document."""
        return self.get_elements_by_type(PDFElementType.FIGURE)
    
    def get_pseudocode(self) -> List[PDFElement]:
        """Get all pseudocode/algorithm blocks from the document."""
        return self.get_elements_by_type(PDFElementType.PSEUDOCODE)


class PDFParser:
    """
    PDF Parser for extracting structured content from research papers.
    
    Supports multiple PDF libraries (PyMuPDF/fitz, pypdf) with fallback.
    Extracts text, sections, equations, tables, figures, and pseudocode.
    """
    
    def __init__(self, config: Optional[PDFParserConfig] = None):
        """Initialize the PDF parser with configuration."""
        self.config = config or PDFParserConfig()
        self._backend: Optional[str] = None
        self._detect_backend()
    
    def _detect_backend(self) -> None:
        """Detect available PDF parsing backend."""
        # Try PyMuPDF first (better quality)
        try:
            import fitz  # PyMuPDF
            self._backend = "pymupdf"
            return
        except ImportError:
            pass
        
        # Fall back to pypdf
        try:
            import pypdf
            self._backend = "pypdf"
            return
        except ImportError:
            pass
        
        # No backend available
        self._backend = None
    
    @property
    def backend(self) -> Optional[str]:
        """Get the current PDF parsing backend."""
        return self._backend
    
    def parse(self, source: Union[str, Path, bytes]) -> PDFParseResult:
        """
        Parse a PDF document and extract structured content.
        
        Args:
            source: Path to PDF file or PDF bytes
            
        Returns:
            PDFParseResult with extracted content
        """
        result = PDFParseResult()
        
        # Handle source path
        if isinstance(source, (str, Path)):
            source_path = Path(source)
            if not source_path.exists():
                result.success = False
                result.errors.append(f"File not found: {source_path}")
                return result
            result.source_path = str(source_path)
        
        # Check backend availability
        if self._backend is None:
            result.success = False
            result.errors.append(
                "No PDF parsing backend available. "
                "Install pymupdf or pypdf: pip install pymupdf pypdf"
            )
            return result
        
        try:
            if self._backend == "pymupdf":
                return self._parse_with_pymupdf(source, result)
            elif self._backend == "pypdf":
                return self._parse_with_pypdf(source, result)
            else:
                result.success = False
                result.errors.append(f"Unknown backend: {self._backend}")
                return result
        except Exception as e:
            result.success = False
            result.errors.append(f"Parse error: {str(e)}")
            return result
    
    def _parse_with_pymupdf(
        self, 
        source: Union[str, Path, bytes], 
        result: PDFParseResult
    ) -> PDFParseResult:
        """Parse PDF using PyMuPDF (fitz)."""
        import fitz
        
        # Open document
        if isinstance(source, bytes):
            doc = fitz.open(stream=source, filetype="pdf")
        else:
            doc = fitz.open(str(source))
        
        try:
            # Extract metadata
            result.metadata = dict(doc.metadata) if doc.metadata else {}
            result.title = result.metadata.get("title", "")
            
            # Determine page range
            max_pages = self.config.max_pages or doc.page_count
            page_range = range(min(max_pages, doc.page_count))
            
            all_text_parts = []
            
            for page_num in page_range:
                page = doc[page_num]
                
                # Extract page text
                text = page.get_text("text")
                
                # Clean text if configured
                if self.config.remove_headers_footers:
                    text = self._remove_headers_footers(text, page_num)
                
                if self.config.merge_lines:
                    text = self._merge_broken_lines(text)
                
                # Create page object
                pdf_page = PDFPage(
                    page_number=page_num + 1,
                    width=page.rect.width,
                    height=page.rect.height,
                    text=text,
                    metadata={"rotation": page.rotation}
                )
                
                # Extract elements from page
                elements = self._extract_elements_from_text(text, page_num + 1)
                pdf_page.elements = elements
                result.elements.extend(elements)
                
                result.pages.append(pdf_page)
                all_text_parts.append(text)
            
            # Extract document structure
            full_text = "\n\n".join(all_text_parts)
            
            # Extract abstract
            result.abstract = self._extract_abstract(full_text)
            
            # Extract sections if configured
            if self.config.detect_sections:
                result.sections = self._extract_sections(full_text, result.pages)
            
            # Extract authors from first page
            if result.pages:
                result.authors = self._extract_authors(result.pages[0].text)
            
            result.success = True
            
        finally:
            doc.close()
        
        return result
    
    def _parse_with_pypdf(
        self, 
        source: Union[str, Path, bytes], 
        result: PDFParseResult
    ) -> PDFParseResult:
        """Parse PDF using pypdf."""
        from pypdf import PdfReader
        import io
        
        # Open document
        if isinstance(source, bytes):
            reader = PdfReader(io.BytesIO(source))
        else:
            reader = PdfReader(str(source))
        
        # Extract metadata
        if reader.metadata:
            result.metadata = {
                k.replace("/", ""): v 
                for k, v in reader.metadata.items() 
                if v is not None
            }
            result.title = result.metadata.get("Title", "")
        
        # Determine page range
        max_pages = self.config.max_pages or len(reader.pages)
        page_range = range(min(max_pages, len(reader.pages)))
        
        all_text_parts = []
        
        for page_num in page_range:
            page = reader.pages[page_num]
            
            # Extract text
            text = page.extract_text() or ""
            
            # Clean text if configured
            if self.config.remove_headers_footers:
                text = self._remove_headers_footers(text, page_num)
            
            if self.config.merge_lines:
                text = self._merge_broken_lines(text)
            
            # Get page dimensions
            mediabox = page.mediabox
            width = float(mediabox.width) if mediabox else 0.0
            height = float(mediabox.height) if mediabox else 0.0
            
            # Create page object
            pdf_page = PDFPage(
                page_number=page_num + 1,
                width=width,
                height=height,
                text=text
            )
            
            # Extract elements from page
            elements = self._extract_elements_from_text(text, page_num + 1)
            pdf_page.elements = elements
            result.elements.extend(elements)
            
            result.pages.append(pdf_page)
            all_text_parts.append(text)
        
        # Extract document structure
        full_text = "\n\n".join(all_text_parts)
        
        # Extract abstract
        result.abstract = self._extract_abstract(full_text)
        
        # Extract sections if configured
        if self.config.detect_sections:
            result.sections = self._extract_sections(full_text, result.pages)
        
        # Extract authors from first page
        if result.pages:
            result.authors = self._extract_authors(result.pages[0].text)
        
        result.success = True
        return result
    
    def _remove_headers_footers(self, text: str, page_num: int) -> str:
        """Remove common headers and footers from page text."""
        lines = text.split("\n")
        
        if len(lines) < 3:
            return text
        
        # Remove likely header (first line if short)
        if len(lines[0].strip()) < 50 and not lines[0].strip().startswith(("Abstract", "1", "Introduction")):
            lines = lines[1:]
        
        # Remove likely footer (last line if short or contains page number)
        if lines:
            last_line = lines[-1].strip()
            if len(last_line) < 30 or re.match(r"^\d+$", last_line):
                lines = lines[:-1]
        
        return "\n".join(lines)
    
    def _merge_broken_lines(self, text: str) -> str:
        """Merge lines that were broken due to PDF formatting."""
        # Replace single newlines within paragraphs with spaces
        # Keep double newlines as paragraph separators
        
        # First, normalize multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        # Merge lines that don't end with sentence-ending punctuation
        lines = text.split("\n")
        merged = []
        current = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                if current:
                    merged.append(current)
                    current = ""
                merged.append("")
            elif current and not current.endswith((".", "!", "?", ":", ";", ")")):
                # Check if line starts with lowercase (continuation)
                if line and line[0].islower():
                    current = current + " " + line
                else:
                    merged.append(current)
                    current = line
            else:
                if current:
                    merged.append(current)
                current = line
        
        if current:
            merged.append(current)
        
        return "\n".join(merged)
    
    def _extract_abstract(self, text: str) -> Optional[str]:
        """Extract abstract from document text."""
        # Look for explicit "Abstract" section
        abstract_patterns = [
            r"(?i)^abstract\s*\n+(.*?)(?=\n\s*(?:1\.?\s+)?(?:introduction|keywords|index terms))",
            r"(?i)abstract[:\s]+(.*?)(?=\n\s*(?:1\.?\s+)?(?:introduction|keywords))",
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, text, re.DOTALL | re.MULTILINE)
            if match:
                abstract = match.group(1).strip()
                # Clean up the abstract
                abstract = re.sub(r"\s+", " ", abstract)
                if len(abstract) > 50:  # Reasonable abstract length
                    return abstract
        
        return None
    
    def _extract_authors(self, first_page_text: str) -> List[str]:
        """Extract author names from first page."""
        authors = []
        
        # Look for common author patterns
        lines = first_page_text.split("\n")[:20]  # Check first 20 lines
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip title (usually first non-empty line)
            if i < 3 and len(line) > 50:
                continue
            
            # Look for lines with names (capitalized words, possibly with commas)
            if re.match(r"^[A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+", line):
                # Check if it looks like an author line
                if not any(kw in line.lower() for kw in ["abstract", "introduction", "university", "department"]):
                    # Split by common separators
                    potential_authors = re.split(r"[,;]|\s+and\s+", line)
                    for author in potential_authors:
                        author = author.strip()
                        if author and len(author) > 3 and len(author) < 50:
                            # Basic name validation
                            if re.match(r"^[A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+", author):
                                authors.append(author)
        
        return authors[:10]  # Limit to 10 authors
    
    def _extract_elements_from_text(
        self, 
        text: str, 
        page_number: int
    ) -> List[PDFElement]:
        """Extract structured elements from page text."""
        elements = []
        
        # Extract equations
        if self.config.extract_equations:
            elements.extend(self._extract_equations(text, page_number))
        
        # Extract algorithm/pseudocode blocks
        if self.config.extract_code:
            elements.extend(self._extract_pseudocode(text, page_number))
        
        # Extract tables
        if self.config.extract_tables:
            elements.extend(self._extract_table_references(text, page_number))
        
        # Extract figure references
        elements.extend(self._extract_figure_references(text, page_number))
        
        return elements
    
    def _extract_equations(self, text: str, page_number: int) -> List[PDFElement]:
        """Extract mathematical equations from text."""
        elements = []
        
        # Pattern for numbered equations
        equation_patterns = [
            r"\((\d+)\)\s*([^\n]+)",  # (1) equation
            r"([^\n]+)\s+\((\d+)\)",  # equation (1)
        ]
        
        for pattern in equation_patterns:
            for match in re.finditer(pattern, text):
                if pattern.startswith(r"\("):
                    eq_num, eq_content = match.groups()
                else:
                    eq_content, eq_num = match.groups()
                
                # Filter out non-equation matches
                if any(c in eq_content for c in ["=", "∑", "∫", "∏", "≤", "≥", "→"]):
                    elements.append(PDFElement(
                        element_type=PDFElementType.EQUATION,
                        content=eq_content.strip(),
                        page_number=page_number,
                        metadata={"equation_number": eq_num}
                    ))
        
        return elements
    
    def _extract_pseudocode(self, text: str, page_number: int) -> List[PDFElement]:
        """Extract algorithm/pseudocode blocks from text."""
        elements = []
        
        # Pattern for algorithm blocks
        algo_pattern = r"(?i)algorithm\s+(\d+)[:\s]+([^\n]+)\n((?:.*?\n)*?)(?=\n\s*(?:algorithm|$))"
        
        for match in re.finditer(algo_pattern, text, re.MULTILINE):
            algo_num = match.group(1)
            algo_name = match.group(2).strip()
            algo_content = match.group(3).strip()
            
            elements.append(PDFElement(
                element_type=PDFElementType.PSEUDOCODE,
                content=algo_content,
                page_number=page_number,
                metadata={
                    "algorithm_number": algo_num,
                    "algorithm_name": algo_name
                }
            ))
        
        return elements
    
    def _extract_table_references(self, text: str, page_number: int) -> List[PDFElement]:
        """Extract table references and captions from text."""
        elements = []
        
        # Pattern for table captions
        table_pattern = r"(?i)table\s+(\d+)[:\.]?\s*([^\n]+)"
        
        for match in re.finditer(table_pattern, text):
            table_num = match.group(1)
            caption = match.group(2).strip()
            
            elements.append(PDFElement(
                element_type=PDFElementType.TABLE,
                content=caption,
                page_number=page_number,
                metadata={"table_number": table_num, "caption": caption}
            ))
        
        return elements
    
    def _extract_figure_references(self, text: str, page_number: int) -> List[PDFElement]:
        """Extract figure references and captions from text."""
        elements = []
        
        # Pattern for figure captions
        figure_pattern = r"(?i)(?:figure|fig\.?)\s+(\d+)[:\.]?\s*([^\n]+)"
        
        for match in re.finditer(figure_pattern, text):
            fig_num = match.group(1)
            caption = match.group(2).strip()
            
            elements.append(PDFElement(
                element_type=PDFElementType.FIGURE,
                content=caption,
                page_number=page_number,
                metadata={"figure_number": fig_num, "caption": caption}
            ))
        
        return elements
    
    def _extract_sections(
        self, 
        text: str, 
        pages: List[PDFPage]
    ) -> List[PDFSection]:
        """Extract hierarchical section structure from document."""
        sections = []
        
        # Combined pattern for section headers
        section_pattern = r"(?m)^(\d+(?:\.\d+)*\.?)\s+([A-Z][^\n]+)|^((?:Abstract|Introduction|Related Work|Background|Method(?:ology)?|Approach|Experiments?|Results?|Discussion|Conclusion|References|Appendix)[^\n]*)"
        
        matches = list(re.finditer(section_pattern, text))
        
        for i, match in enumerate(matches):
            if match.group(1):  # Numbered section
                section_num = match.group(1).rstrip(".")
                title = match.group(2).strip()
                level = section_num.count(".") + 1
            else:  # Named section
                section_num = None
                title = match.group(3).strip()
                level = 1
            
            # Get content until next section
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start_pos:end_pos].strip()
            
            # Determine page range
            start_page = 1
            end_page = len(pages)
            for page in pages:
                if match.group(0) in page.text:
                    start_page = page.page_number
                    break
            
            section = PDFSection(
                title=title,
                level=level,
                section_number=section_num,
                content=content,
                start_page=start_page,
                end_page=end_page
            )
            
            # Add to appropriate parent or root
            if level == 1 or not sections:
                sections.append(section)
            else:
                # Find parent section
                parent = sections[-1] if sections else None
                if parent and parent.level < level:
                    parent.subsections.append(section)
                else:
                    sections.append(section)
        
        return sections


# Convenience functions

def extract_text_from_pdf(
    path: Union[str, Path],
    max_pages: Optional[int] = None
) -> str:
    """
    Extract raw text from a PDF file.
    
    Args:
        path: Path to PDF file
        max_pages: Maximum number of pages to extract
        
    Returns:
        Extracted text as a single string
    """
    config = PDFParserConfig(
        max_pages=max_pages,
        detect_sections=False,
        output_format="raw"
    )
    parser = PDFParser(config)
    result = parser.parse(path)
    
    if not result.success:
        raise ValueError(f"Failed to parse PDF: {'; '.join(result.errors)}")
    
    return result.text


def extract_sections_from_pdf(
    path: Union[str, Path],
    max_pages: Optional[int] = None
) -> List[PDFSection]:
    """
    Extract structured sections from a PDF file.
    
    Args:
        path: Path to PDF file
        max_pages: Maximum number of pages to process
        
    Returns:
        List of PDFSection objects representing document structure
    """
    config = PDFParserConfig(
        max_pages=max_pages,
        detect_sections=True
    )
    parser = PDFParser(config)
    result = parser.parse(path)
    
    if not result.success:
        raise ValueError(f"Failed to parse PDF: {'; '.join(result.errors)}")
    
    return result.sections


def parse_pdf(
    source: Union[str, Path, bytes],
    config: Optional[PDFParserConfig] = None
) -> PDFParseResult:
    """
    Parse a PDF document with full extraction.
    
    Args:
        source: Path to PDF file or PDF bytes
        config: Optional parser configuration
        
    Returns:
        PDFParseResult with all extracted content
    """
    parser = PDFParser(config)
    return parser.parse(source)


__all__ = [
    # Main classes
    "PDFParser",
    "PDFParserConfig",
    "PDFParseResult",
    "PDFPage",
    "PDFSection",
    "PDFElement",
    "PDFBoundingBox",
    "PDFElementType",
    # Convenience functions
    "extract_text_from_pdf",
    "extract_sections_from_pdf",
    "parse_pdf",
]
