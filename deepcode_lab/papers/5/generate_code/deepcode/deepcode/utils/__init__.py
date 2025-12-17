"""
DeepCode Utils Package

This module provides utility functions and classes for the DeepCode framework,
including logging configuration and PDF parsing capabilities.
"""

from deepcode.deepcode.utils.logger import (
    Logger,
    LogLevel,
    LogConfig,
    get_logger,
    setup_logging,
    log_execution_time,
    log_agent_call,
)

from deepcode.deepcode.utils.pdf_parser import (
    PDFParser,
    PDFParserConfig,
    PDFPage,
    PDFParseResult,
    extract_text_from_pdf,
    extract_sections_from_pdf,
)

__all__ = [
    # Logger exports
    "Logger",
    "LogLevel",
    "LogConfig",
    "get_logger",
    "setup_logging",
    "log_execution_time",
    "log_agent_call",
    # PDF Parser exports
    "PDFParser",
    "PDFParserConfig",
    "PDFPage",
    "PDFParseResult",
    "extract_text_from_pdf",
    "extract_sections_from_pdf",
]
