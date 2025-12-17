"""
Document Parsing Agent for DeepCode.

This module provides the Document Parsing Agent that converts source documents
(PDFs, Markdown files) into the structured Document model for further processing
in the blueprint generation phase.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
    Citation,
    CodeBlock,
)
from deepcode.deepcode.utils.pdf_parser import (
    PDFParser,
    PDFParserConfig,
    PDFParseResult,
    PDFSection,
    PDFElement,
    PDFElementType,
)
from deepcode.deepcode.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentParserConfig(BaseModel):
    """Configuration for document parsing."""
    
    # PDF parsing options
    extract_images: bool = Field(default=True, description="Extract images/figures from PDF")
    extract_tables: bool = Field(default=True, description="Extract tables from PDF")
    extract_equations: bool = Field(default=True, description="Extract equations from PDF")
    extract_code: bool = Field(default=True, description="Extract code blocks from PDF")
    detect_sections: bool = Field(default=True, description="Detect document sections")
    max_pages: Optional[int] = Field(default=None, description="Maximum pages to parse")
    
    # Markdown parsing options
    parse_latex: bool = Field(default=True, description="Parse LaTeX equations in Markdown")
    parse_code_blocks: bool = Field(default=True, description="Parse code blocks in Markdown")
    
    # General options
    merge_short_paragraphs: bool = Field(default=True, description="Merge short consecutive paragraphs")
    min_paragraph_length: int = Field(default=50, description="Minimum paragraph length to keep separate")
    
    def to_pdf_config(self) -> PDFParserConfig:
        """Convert to PDFParserConfig."""
        return PDFParserConfig(
            extract_images=self.extract_images,
            extract_tables=self.extract_tables,
            extract_equations=self.extract_equations,
            extract_code=self.extract_code,
            detect_sections=self.detect_sections,
            max_pages=self.max_pages,
        )


class DocumentParser:
    """
    Document Parsing Agent for converting source documents to Document model.
    
    Supports:
    - PDF documents (research papers, technical documents)
    - Markdown files
    - Plain text files
    
    The parser extracts multimodal elements including:
    - Text blocks and paragraphs
    - Section headers and structure
    - Mathematical equations (LaTeX)
    - Tables with headers and data
    - Figures with captions
    - Pseudocode/algorithm boxes
    - Code blocks
    - Citations and references
    """
    
    def __init__(self, config: Optional[DocumentParserConfig] = None):
        """
        Initialize the document parser.
        
        Args:
            config: Parser configuration options
        """
        self.config = config or DocumentParserConfig()
        self._pdf_parser: Optional[PDFParser] = None
        
    @property
    def pdf_parser(self) -> PDFParser:
        """Lazy initialization of PDF parser."""
        if self._pdf_parser is None:
            self._pdf_parser = PDFParser(self.config.to_pdf_config())
        return self._pdf_parser
    
    def parse(self, source: Union[str, Path]) -> Document:
        """
        Parse a document from file path.
        
        Args:
            source: Path to the document file
            
        Returns:
            Parsed Document model
            
        Raises:
            ValueError: If file type is not supported
            FileNotFoundError: If file does not exist
        """
        path = Path(source)
        
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")
        
        suffix = path.suffix.lower()
        
        logger.info(f"Parsing document: {path.name} (type: {suffix})")
        
        if suffix == ".pdf":
            return self._parse_pdf(path)
        elif suffix in (".md", ".markdown"):
            return self._parse_markdown(path)
        elif suffix == ".txt":
            return self._parse_text(path)
        else:
            raise ValueError(f"Unsupported document type: {suffix}")
    
    def parse_content(
        self,
        content: str,
        source_type: str = "text",
        title: Optional[str] = None,
    ) -> Document:
        """
        Parse document from string content.
        
        Args:
            content: Document content as string
            source_type: Type of content ("markdown", "text")
            title: Optional document title
            
        Returns:
            Parsed Document model
        """
        if source_type == "markdown":
            return self._parse_markdown_content(content, title)
        else:
            return self._parse_text_content(content, title)
    
    def _parse_pdf(self, path: Path) -> Document:
        """Parse a PDF document."""
        logger.debug(f"Parsing PDF: {path}")
        
        # Use PDF parser to extract content
        result: PDFParseResult = self.pdf_parser.parse(path)
        
        # Extract metadata
        title = result.metadata.get("title", path.stem) if result.metadata else path.stem
        authors = self._extract_authors(result)
        abstract = self._extract_abstract(result)
        
        # Convert PDF elements to Document elements
        elements: List[DocumentElement] = []
        sections: List[DocumentSection] = []
        
        # Process sections if available
        if result.sections:
            sections = self._convert_pdf_sections(result.sections)
        
        # Process pages and elements
        for page in result.pages:
            for pdf_elem in page.elements:
                doc_elem = self._convert_pdf_element(pdf_elem)
                if doc_elem:
                    elements.append(doc_elem)
        
        # If no elements from pages, try to extract from raw text
        if not elements and result.text:
            elements = self._extract_elements_from_text(result.text)
        
        document = Document(
            title=title,
            authors=authors,
            abstract=abstract,
            elements=elements,
            sections=sections,
            source_path=str(path),
            source_type="pdf",
            metadata=result.metadata or {},
        )
        
        logger.info(f"Parsed PDF with {len(elements)} elements and {len(sections)} sections")
        return document
    
    def _parse_markdown(self, path: Path) -> Document:
        """Parse a Markdown document."""
        logger.debug(f"Parsing Markdown: {path}")
        
        content = path.read_text(encoding="utf-8")
        return self._parse_markdown_content(content, path.stem, str(path))
    
    def _parse_markdown_content(
        self,
        content: str,
        title: Optional[str] = None,
        source_path: Optional[str] = None,
    ) -> Document:
        """Parse Markdown content string."""
        elements: List[DocumentElement] = []
        sections: List[DocumentSection] = []
        
        # Extract title from first H1 if not provided
        if not title:
            title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
            title = title_match.group(1) if title_match else "Untitled"
        
        # Extract authors (common patterns)
        authors = self._extract_authors_from_markdown(content)
        
        # Extract abstract
        abstract = self._extract_abstract_from_markdown(content)
        
        # Parse content into elements
        lines = content.split("\n")
        current_section: Optional[DocumentSection] = None
        current_text: List[str] = []
        position = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check for section headers
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if header_match:
                # Flush current text
                if current_text:
                    text_content = "\n".join(current_text).strip()
                    if text_content:
                        elements.append(TextBlock.create(
                            content=text_content,
                            position=position,
                        ))
                        position += 1
                    current_text = []
                
                level = len(header_match.group(1))
                header_text = header_match.group(2)
                
                # Create section header element
                elements.append(SectionHeader.create(
                    content=header_text,
                    level=level,
                    position=position,
                ))
                position += 1
                
                # Create section
                section = DocumentSection(
                    title=header_text,
                    level=level,
                    elements=[],
                )
                sections.append(section)
                current_section = section
                i += 1
                continue
            
            # Check for code blocks
            if line.startswith("```"):
                # Flush current text
                if current_text:
                    text_content = "\n".join(current_text).strip()
                    if text_content:
                        elements.append(TextBlock.create(
                            content=text_content,
                            position=position,
                        ))
                        position += 1
                    current_text = []
                
                # Extract language
                language = line[3:].strip() or "text"
                
                # Find end of code block
                code_lines = []
                i += 1
                while i < len(lines) and not lines[i].startswith("```"):
                    code_lines.append(lines[i])
                    i += 1
                
                code_content = "\n".join(code_lines)
                
                # Check if it's pseudocode or algorithm
                if self._is_pseudocode(code_content, language):
                    elements.append(Pseudocode.create(
                        content=code_content,
                        algorithm_name=self._extract_algorithm_name(code_content),
                        position=position,
                    ))
                else:
                    elements.append(CodeBlock(
                        element_type=ElementType.CODE_BLOCK,
                        content=code_content,
                        language=language,
                        position=position,
                    ))
                position += 1
                i += 1
                continue
            
            # Check for LaTeX equations
            if self.config.parse_latex:
                # Display equations ($$...$$)
                if line.strip().startswith("$$"):
                    # Flush current text
                    if current_text:
                        text_content = "\n".join(current_text).strip()
                        if text_content:
                            elements.append(TextBlock.create(
                                content=text_content,
                                position=position,
                            ))
                            position += 1
                        current_text = []
                    
                    # Find end of equation
                    eq_lines = [line]
                    if not line.strip().endswith("$$") or line.strip() == "$$":
                        i += 1
                        while i < len(lines) and "$$" not in lines[i]:
                            eq_lines.append(lines[i])
                            i += 1
                        if i < len(lines):
                            eq_lines.append(lines[i])
                    
                    eq_content = "\n".join(eq_lines)
                    latex = eq_content.replace("$$", "").strip()
                    
                    elements.append(Equation.create(
                        content=eq_content,
                        latex=latex,
                        is_inline=False,
                        position=position,
                    ))
                    position += 1
                    i += 1
                    continue
            
            # Check for tables
            if "|" in line and i + 1 < len(lines) and re.match(r"^\|[-:\s|]+\|$", lines[i + 1]):
                # Flush current text
                if current_text:
                    text_content = "\n".join(current_text).strip()
                    if text_content:
                        elements.append(TextBlock.create(
                            content=text_content,
                            position=position,
                        ))
                        position += 1
                    current_text = []
                
                # Parse table
                table_lines = [line]
                i += 1
                while i < len(lines) and "|" in lines[i]:
                    table_lines.append(lines[i])
                    i += 1
                
                table = self._parse_markdown_table(table_lines, position)
                if table:
                    elements.append(table)
                    position += 1
                continue
            
            # Regular text line
            current_text.append(line)
            i += 1
        
        # Flush remaining text
        if current_text:
            text_content = "\n".join(current_text).strip()
            if text_content:
                elements.append(TextBlock.create(
                    content=text_content,
                    position=position,
                ))
        
        document = Document(
            title=title,
            authors=authors,
            abstract=abstract,
            elements=elements,
            sections=sections,
            source_path=source_path,
            source_type="markdown",
        )
        
        logger.info(f"Parsed Markdown with {len(elements)} elements and {len(sections)} sections")
        return document
    
    def _parse_text(self, path: Path) -> Document:
        """Parse a plain text document."""
        logger.debug(f"Parsing text: {path}")
        
        content = path.read_text(encoding="utf-8")
        return self._parse_text_content(content, path.stem, str(path))
    
    def _parse_text_content(
        self,
        content: str,
        title: Optional[str] = None,
        source_path: Optional[str] = None,
    ) -> Document:
        """Parse plain text content."""
        elements = self._extract_elements_from_text(content)
        
        document = Document(
            title=title or "Untitled",
            elements=elements,
            source_path=source_path,
            source_type="text",
        )
        
        logger.info(f"Parsed text with {len(elements)} elements")
        return document
    
    def _extract_elements_from_text(self, text: str) -> List[DocumentElement]:
        """Extract elements from plain text."""
        elements: List[DocumentElement] = []
        
        # Split into paragraphs
        paragraphs = re.split(r"\n\s*\n", text)
        
        position = 0
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if it looks like a section header
            if self._is_section_header(para):
                level = self._detect_header_level(para)
                elements.append(SectionHeader.create(
                    content=para,
                    level=level,
                    position=position,
                ))
            # Check for equation patterns
            elif self._contains_equation(para):
                elements.append(Equation.create(
                    content=para,
                    latex=self._extract_latex(para),
                    position=position,
                ))
            # Check for pseudocode patterns
            elif self._is_pseudocode(para):
                elements.append(Pseudocode.create(
                    content=para,
                    algorithm_name=self._extract_algorithm_name(para),
                    position=position,
                ))
            else:
                elements.append(TextBlock.create(
                    content=para,
                    position=position,
                ))
            
            position += 1
        
        return elements
    
    def _convert_pdf_sections(self, pdf_sections: List[PDFSection]) -> List[DocumentSection]:
        """Convert PDF sections to Document sections."""
        doc_sections: List[DocumentSection] = []
        
        for pdf_sec in pdf_sections:
            elements = []
            for elem in pdf_sec.elements:
                doc_elem = self._convert_pdf_element(elem)
                if doc_elem:
                    elements.append(doc_elem)
            
            # Recursively convert subsections
            subsections = self._convert_pdf_sections(pdf_sec.subsections) if pdf_sec.subsections else []
            
            doc_section = DocumentSection(
                title=pdf_sec.title,
                level=pdf_sec.level,
                section_number=pdf_sec.section_number,
                elements=elements,
                subsections=subsections,
            )
            doc_sections.append(doc_section)
        
        return doc_sections
    
    def _convert_pdf_element(self, pdf_elem: PDFElement) -> Optional[DocumentElement]:
        """Convert a PDF element to a Document element."""
        elem_type = pdf_elem.element_type
        content = pdf_elem.content
        position = pdf_elem.position.y0 if pdf_elem.position else 0
        page = pdf_elem.page_number
        
        if elem_type == PDFElementType.TEXT:
            return TextBlock.create(
                content=content,
                position=int(position),
                page_number=page,
            )
        
        elif elem_type == PDFElementType.HEADING:
            level = pdf_elem.metadata.get("level", 1) if pdf_elem.metadata else 1
            return SectionHeader.create(
                content=content,
                level=level,
                position=int(position),
                page_number=page,
            )
        
        elif elem_type == PDFElementType.EQUATION:
            latex = pdf_elem.metadata.get("latex", content) if pdf_elem.metadata else content
            return Equation.create(
                content=content,
                latex=latex,
                position=int(position),
                page_number=page,
            )
        
        elif elem_type == PDFElementType.TABLE:
            headers = pdf_elem.metadata.get("headers", []) if pdf_elem.metadata else []
            rows = pdf_elem.metadata.get("rows", []) if pdf_elem.metadata else []
            caption = pdf_elem.metadata.get("caption", "") if pdf_elem.metadata else ""
            return Table.create(
                content=content,
                headers=headers,
                rows=rows,
                caption=caption,
                position=int(position),
                page_number=page,
            )
        
        elif elem_type == PDFElementType.FIGURE:
            caption = pdf_elem.metadata.get("caption", "") if pdf_elem.metadata else ""
            image_path = pdf_elem.metadata.get("image_path") if pdf_elem.metadata else None
            return Figure.create(
                content=content,
                caption=caption,
                image_path=image_path,
                position=int(position),
                page_number=page,
            )
        
        elif elem_type == PDFElementType.PSEUDOCODE:
            algo_name = pdf_elem.metadata.get("algorithm_name", "") if pdf_elem.metadata else ""
            return Pseudocode.create(
                content=content,
                algorithm_name=algo_name,
                position=int(position),
                page_number=page,
            )
        
        elif elem_type == PDFElementType.CODE:
            language = pdf_elem.metadata.get("language", "text") if pdf_elem.metadata else "text"
            return CodeBlock(
                element_type=ElementType.CODE_BLOCK,
                content=content,
                language=language,
                position=int(position),
                page_number=page,
            )
        
        elif elem_type == PDFElementType.CITATION:
            return Citation(
                element_type=ElementType.CITATION,
                content=content,
                position=int(position),
                page_number=page,
            )
        
        elif elem_type == PDFElementType.ABSTRACT:
            return TextBlock.create(
                content=content,
                is_abstract=True,
                position=int(position),
                page_number=page,
            )
        
        # Default: treat as text
        return TextBlock.create(
            content=content,
            position=int(position),
            page_number=page,
        )
    
    def _extract_authors(self, result: PDFParseResult) -> List[str]:
        """Extract authors from PDF result."""
        if result.metadata and "authors" in result.metadata:
            authors = result.metadata["authors"]
            if isinstance(authors, list):
                return authors
            elif isinstance(authors, str):
                return [a.strip() for a in authors.split(",")]
        return []
    
    def _extract_abstract(self, result: PDFParseResult) -> Optional[str]:
        """Extract abstract from PDF result."""
        # Check metadata
        if result.metadata and "abstract" in result.metadata:
            return result.metadata["abstract"]
        
        # Look for abstract section
        for section in result.sections:
            if section.title.lower() == "abstract":
                return section.get_full_text()
        
        # Look for abstract element
        for page in result.pages:
            for elem in page.elements:
                if elem.element_type == PDFElementType.ABSTRACT:
                    return elem.content
        
        return None
    
    def _extract_authors_from_markdown(self, content: str) -> List[str]:
        """Extract authors from Markdown content."""
        # Look for author patterns
        patterns = [
            r"(?:Author|Authors|By):\s*(.+)",
            r"\*\*(?:Author|Authors)\*\*:\s*(.+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                authors_str = match.group(1)
                return [a.strip() for a in re.split(r"[,;]|(?:\s+and\s+)", authors_str)]
        
        return []
    
    def _extract_abstract_from_markdown(self, content: str) -> Optional[str]:
        """Extract abstract from Markdown content."""
        # Look for abstract section
        patterns = [
            r"##?\s*Abstract\s*\n+(.+?)(?=\n##|\n\*\*|\Z)",
            r"\*\*Abstract\*\*[:\s]*(.+?)(?=\n##|\n\*\*|\Z)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _parse_markdown_table(self, lines: List[str], position: int) -> Optional[Table]:
        """Parse a Markdown table."""
        if len(lines) < 2:
            return None
        
        # Parse header row
        header_line = lines[0]
        headers = [cell.strip() for cell in header_line.split("|") if cell.strip()]
        
        # Skip separator line
        # Parse data rows
        rows = []
        for line in lines[2:]:
            if "|" in line:
                row = [cell.strip() for cell in line.split("|") if cell.strip()]
                if row:
                    rows.append(row)
        
        content = "\n".join(lines)
        
        return Table.create(
            content=content,
            headers=headers,
            rows=rows,
            position=position,
        )
    
    def _is_section_header(self, text: str) -> bool:
        """Check if text looks like a section header."""
        text = text.strip()
        
        # Short text that ends without punctuation
        if len(text) < 100 and not text.endswith((".", ",", ";", ":")):
            # Check for common header patterns
            patterns = [
                r"^\d+\.?\s+\w+",  # "1. Introduction" or "1 Introduction"
                r"^[A-Z][A-Z\s]+$",  # ALL CAPS
                r"^(?:Introduction|Abstract|Method|Results|Discussion|Conclusion|References|Appendix)",
            ]
            for pattern in patterns:
                if re.match(pattern, text, re.IGNORECASE):
                    return True
        
        return False
    
    def _detect_header_level(self, text: str) -> int:
        """Detect the level of a section header."""
        text = text.strip()
        
        # Check for numbered sections
        match = re.match(r"^(\d+(?:\.\d+)*)", text)
        if match:
            return match.group(1).count(".") + 1
        
        # Check for common top-level sections
        top_level = ["introduction", "abstract", "conclusion", "references"]
        if any(text.lower().startswith(s) for s in top_level):
            return 1
        
        return 2  # Default to level 2
    
    def _contains_equation(self, text: str) -> bool:
        """Check if text contains mathematical equations."""
        patterns = [
            r"\$\$.+\$\$",  # Display math
            r"\$.+\$",  # Inline math
            r"\\begin\{equation\}",  # LaTeX equation environment
            r"\\frac\{",  # Fraction
            r"\\sum",  # Summation
            r"\\int",  # Integral
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.DOTALL):
                return True
        
        return False
    
    def _extract_latex(self, text: str) -> str:
        """Extract LaTeX from text."""
        # Try to extract from delimiters
        patterns = [
            r"\$\$(.+?)\$\$",
            r"\$(.+?)\$",
            r"\\begin\{equation\}(.+?)\\end\{equation\}",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return text
    
    def _is_pseudocode(self, text: str, language: str = "") -> bool:
        """Check if text is pseudocode or algorithm."""
        # Check language hint
        if language.lower() in ("pseudocode", "algorithm", "pseudo"):
            return True
        
        # Check for algorithm keywords
        keywords = [
            r"\bAlgorithm\b",
            r"\bInput\s*:",
            r"\bOutput\s*:",
            r"\bProcedure\b",
            r"\bFunction\b",
            r"\bwhile\b.*\bdo\b",
            r"\bfor\s+each\b",
            r"\bif\b.*\bthen\b",
            r"\breturn\b",
        ]
        
        keyword_count = sum(1 for kw in keywords if re.search(kw, text, re.IGNORECASE))
        return keyword_count >= 2
    
    def _extract_algorithm_name(self, text: str) -> str:
        """Extract algorithm name from pseudocode."""
        patterns = [
            r"Algorithm\s*\d*[:\s]+(\w+)",
            r"Procedure\s+(\w+)",
            r"Function\s+(\w+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return ""


# Convenience functions

def parse_document(
    source: Union[str, Path],
    config: Optional[DocumentParserConfig] = None,
) -> Document:
    """
    Parse a document from file path.
    
    Args:
        source: Path to the document file
        config: Optional parser configuration
        
    Returns:
        Parsed Document model
    """
    parser = DocumentParser(config)
    return parser.parse(source)


def parse_pdf(
    path: Union[str, Path],
    config: Optional[DocumentParserConfig] = None,
) -> Document:
    """
    Parse a PDF document.
    
    Args:
        path: Path to the PDF file
        config: Optional parser configuration
        
    Returns:
        Parsed Document model
    """
    parser = DocumentParser(config)
    return parser._parse_pdf(Path(path))


def parse_markdown(
    path: Union[str, Path],
    config: Optional[DocumentParserConfig] = None,
) -> Document:
    """
    Parse a Markdown document.
    
    Args:
        path: Path to the Markdown file
        config: Optional parser configuration
        
    Returns:
        Parsed Document model
    """
    parser = DocumentParser(config)
    return parser._parse_markdown(Path(path))


def parse_content(
    content: str,
    source_type: str = "markdown",
    title: Optional[str] = None,
    config: Optional[DocumentParserConfig] = None,
) -> Document:
    """
    Parse document from string content.
    
    Args:
        content: Document content as string
        source_type: Type of content ("markdown", "text")
        title: Optional document title
        config: Optional parser configuration
        
    Returns:
        Parsed Document model
    """
    parser = DocumentParser(config)
    return parser.parse_content(content, source_type, title)


__all__ = [
    "DocumentParser",
    "DocumentParserConfig",
    "parse_document",
    "parse_pdf",
    "parse_markdown",
    "parse_content",
]
