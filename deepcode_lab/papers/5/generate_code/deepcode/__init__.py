"""
DeepCode: Open Agentic Coding Framework

A fully autonomous framework for high-fidelity document-to-codebase synthesis
that treats repository synthesis as a channel optimization problem through
principled information-flow management.

Core Components:
- Blueprint Generation (Phase 1): Document parsing, concept/algorithm extraction
- Code Generation (Phase 2): CodeMem stateful generation, CodeRAG knowledge injection
- Verification (Phase 3): Static analysis, sandbox execution, iterative refinement
"""

__version__ = "0.1.0"
__author__ = "DeepCode Team"

from deepcode.orchestrator import DeepCodeOrchestrator

__all__ = ["DeepCodeOrchestrator", "__version__"]
