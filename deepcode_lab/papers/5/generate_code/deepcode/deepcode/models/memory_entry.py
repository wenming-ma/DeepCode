"""
Memory Entry Model for CodeMem - Stateful Code Generation

Defines the memory entry structure mt = (Pt, It, Et) where:
- Pt: Core purpose (natural language summary)
- It: Public interface (classes, functions, signatures)
- Et: Dependency edges (afferent/efferent couplings)

This module implements the CodeMemory class for tracking implemented files
and their relationships during code generation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field


class FunctionSignature(BaseModel):
    """Represents a function or method signature."""
    
    name: str = Field(..., description="Function name")
    parameters: List[str] = Field(default_factory=list, description="Parameter names with types")
    return_type: Optional[str] = Field(None, description="Return type annotation")
    is_async: bool = Field(False, description="Whether function is async")
    is_method: bool = Field(False, description="Whether this is a class method")
    is_classmethod: bool = Field(False, description="Whether this is a classmethod")
    is_staticmethod: bool = Field(False, description="Whether this is a staticmethod")
    docstring: Optional[str] = Field(None, description="Function docstring")
    decorators: List[str] = Field(default_factory=list, description="Applied decorators")
    
    def to_signature_string(self) -> str:
        """Convert to a readable signature string."""
        prefix = "async " if self.is_async else ""
        params = ", ".join(self.parameters)
        ret = f" -> {self.return_type}" if self.return_type else ""
        return f"{prefix}def {self.name}({params}){ret}"


class ClassInterface(BaseModel):
    """Represents a class interface with its methods and attributes."""
    
    name: str = Field(..., description="Class name")
    bases: List[str] = Field(default_factory=list, description="Base classes")
    methods: List[FunctionSignature] = Field(default_factory=list, description="Class methods")
    class_attributes: List[str] = Field(default_factory=list, description="Class-level attributes")
    instance_attributes: List[str] = Field(default_factory=list, description="Instance attributes from __init__")
    docstring: Optional[str] = Field(None, description="Class docstring")
    decorators: List[str] = Field(default_factory=list, description="Class decorators")
    
    def get_public_methods(self) -> List[FunctionSignature]:
        """Get only public methods (not starting with _)."""
        return [m for m in self.methods if not m.name.startswith('_') or m.name.startswith('__')]
    
    def to_interface_string(self) -> str:
        """Convert to a readable interface string."""
        lines = []
        
        # Class declaration
        bases_str = f"({', '.join(self.bases)})" if self.bases else ""
        lines.append(f"class {self.name}{bases_str}:")
        
        # Docstring
        if self.docstring:
            lines.append(f'    """{self.docstring}"""')
        
        # Class attributes
        for attr in self.class_attributes:
            lines.append(f"    {attr}")
        
        # Methods
        for method in self.methods:
            sig = method.to_signature_string()
            lines.append(f"    {sig}: ...")
        
        return "\n".join(lines)


class PublicInterface(BaseModel):
    """
    Public interface of a code file (It component).
    
    Contains all exported classes, functions, and constants
    that other files may depend on.
    """
    
    classes: List[ClassInterface] = Field(
        default_factory=list,
        description="Class definitions with their interfaces"
    )
    functions: List[FunctionSignature] = Field(
        default_factory=list,
        description="Module-level function signatures"
    )
    constants: List[str] = Field(
        default_factory=list,
        description="Module-level constants (UPPER_CASE names)"
    )
    type_aliases: List[str] = Field(
        default_factory=list,
        description="Type alias definitions"
    )
    exports: List[str] = Field(
        default_factory=list,
        description="Explicitly exported names (__all__)"
    )
    
    def get_all_names(self) -> Set[str]:
        """Get all public names defined in this interface."""
        names = set()
        names.update(c.name for c in self.classes)
        names.update(f.name for f in self.functions)
        names.update(self.constants)
        names.update(self.type_aliases)
        return names
    
    def to_interface_string(self) -> str:
        """Convert to a readable interface summary."""
        lines = []
        
        # Constants
        if self.constants:
            lines.append("# Constants")
            for const in self.constants:
                lines.append(f"{const}")
            lines.append("")
        
        # Type aliases
        if self.type_aliases:
            lines.append("# Type Aliases")
            for alias in self.type_aliases:
                lines.append(f"{alias}")
            lines.append("")
        
        # Functions
        if self.functions:
            lines.append("# Functions")
            for func in self.functions:
                lines.append(func.to_signature_string())
            lines.append("")
        
        # Classes
        if self.classes:
            lines.append("# Classes")
            for cls in self.classes:
                lines.append(cls.to_interface_string())
                lines.append("")
        
        return "\n".join(lines)


class DependencyType(str, Enum):
    """Type of dependency relationship."""
    
    IMPORT = "import"  # Direct import
    FROM_IMPORT = "from_import"  # from X import Y
    TYPE_HINT = "type_hint"  # Used only in type hints
    INHERITANCE = "inheritance"  # Class inheritance
    COMPOSITION = "composition"  # Object composition
    FUNCTION_CALL = "function_call"  # Function/method call


class DependencyEdge(BaseModel):
    """A single dependency edge to another file."""
    
    target_file: str = Field(..., description="Path to the dependent file")
    dependency_type: DependencyType = Field(..., description="Type of dependency")
    imported_names: List[str] = Field(
        default_factory=list,
        description="Specific names imported from target"
    )
    is_optional: bool = Field(
        False,
        description="Whether this is an optional/conditional import"
    )


class DependencyEdges(BaseModel):
    """
    Dependency edges for a code file (Et component).
    
    Tracks both afferent (incoming) and efferent (outgoing) couplings.
    """
    
    afferent: List[DependencyEdge] = Field(
        default_factory=list,
        description="Files that depend on this file (incoming)"
    )
    efferent: List[DependencyEdge] = Field(
        default_factory=list,
        description="Files this file depends on (outgoing)"
    )
    external_dependencies: List[str] = Field(
        default_factory=list,
        description="External package dependencies (e.g., 'numpy', 'torch')"
    )
    
    def get_direct_dependencies(self) -> List[str]:
        """Get list of files this file directly depends on."""
        return [edge.target_file for edge in self.efferent]
    
    def get_dependents(self) -> List[str]:
        """Get list of files that depend on this file."""
        return [edge.target_file for edge in self.afferent]
    
    def add_efferent(
        self,
        target_file: str,
        dependency_type: DependencyType,
        imported_names: Optional[List[str]] = None
    ) -> None:
        """Add an outgoing dependency."""
        edge = DependencyEdge(
            target_file=target_file,
            dependency_type=dependency_type,
            imported_names=imported_names or []
        )
        self.efferent.append(edge)
    
    def add_afferent(
        self,
        source_file: str,
        dependency_type: DependencyType,
        imported_names: Optional[List[str]] = None
    ) -> None:
        """Add an incoming dependency."""
        edge = DependencyEdge(
            target_file=source_file,
            dependency_type=dependency_type,
            imported_names=imported_names or []
        )
        self.afferent.append(edge)


class MemoryEntry(BaseModel):
    """
    Code memory entry mt = (Pt, It, Et).
    
    Represents the summarized state of an implemented code file,
    used for context injection during subsequent file generation.
    
    Components:
    - Pt (purpose): Natural language summary of the file's purpose
    - It (interface): Public interface (classes, functions, signatures)
    - Et (edges): Dependency edges (afferent/efferent couplings)
    """
    
    file_path: str = Field(..., description="Path to the implemented file")
    
    # Pt: Core purpose
    purpose: str = Field(
        ...,
        description="Natural language summary of the file's core purpose"
    )
    
    # It: Public interface
    interface: PublicInterface = Field(
        default_factory=PublicInterface,
        description="Public interface of the file"
    )
    
    # Et: Dependency edges
    edges: DependencyEdges = Field(
        default_factory=DependencyEdges,
        description="Dependency relationships"
    )
    
    # Additional metadata
    implementation_order: int = Field(
        0,
        description="Order in which this file was implemented"
    )
    code_hash: Optional[str] = Field(
        None,
        description="Hash of the code content for change detection"
    )
    line_count: int = Field(0, description="Number of lines in the file")
    
    def to_context_string(self, include_full_interface: bool = True) -> str:
        """
        Convert memory entry to a context string for LLM injection.
        
        Args:
            include_full_interface: Whether to include full interface details
            
        Returns:
            Formatted context string
        """
        lines = [
            f"## File: {self.file_path}",
            f"**Purpose**: {self.purpose}",
            ""
        ]
        
        # Dependencies
        deps = self.edges.get_direct_dependencies()
        if deps:
            lines.append(f"**Dependencies**: {', '.join(deps)}")
        
        ext_deps = self.edges.external_dependencies
        if ext_deps:
            lines.append(f"**External Packages**: {', '.join(ext_deps)}")
        
        lines.append("")
        
        # Interface
        if include_full_interface:
            lines.append("**Public Interface**:")
            lines.append("```python")
            lines.append(self.interface.to_interface_string())
            lines.append("```")
        else:
            # Compact version - just names
            names = self.interface.get_all_names()
            if names:
                lines.append(f"**Exports**: {', '.join(sorted(names))}")
        
        return "\n".join(lines)
    
    def get_importable_names(self) -> Set[str]:
        """Get all names that can be imported from this file."""
        return self.interface.get_all_names()
    
    @classmethod
    def create(
        cls,
        file_path: str,
        purpose: str,
        interface: Optional[PublicInterface] = None,
        edges: Optional[DependencyEdges] = None,
        implementation_order: int = 0
    ) -> "MemoryEntry":
        """Factory method to create a memory entry."""
        return cls(
            file_path=file_path,
            purpose=purpose,
            interface=interface or PublicInterface(),
            edges=edges or DependencyEdges(),
            implementation_order=implementation_order
        )


class CodeMemory(BaseModel):
    """
    Code Memory (M) - Tracks all implemented files and their relationships.
    
    Implements the SelectRelevantMemory function to identify dependencies
    from the blueprint and return memory entries for files the target depends on.
    """
    
    entries: Dict[str, MemoryEntry] = Field(
        default_factory=dict,
        description="Map of file paths to their memory entries"
    )
    implementation_counter: int = Field(
        0,
        description="Counter for tracking implementation order"
    )
    
    def add_entry(self, entry: MemoryEntry) -> None:
        """Add a memory entry for an implemented file."""
        entry.implementation_order = self.implementation_counter
        self.entries[entry.file_path] = entry
        self.implementation_counter += 1
        
        # Update afferent edges for dependencies
        for dep_file in entry.edges.get_direct_dependencies():
            if dep_file in self.entries:
                dep_entry = self.entries[dep_file]
                dep_entry.edges.add_afferent(
                    source_file=entry.file_path,
                    dependency_type=DependencyType.IMPORT
                )
    
    def get_entry(self, file_path: str) -> Optional[MemoryEntry]:
        """Get memory entry for a specific file."""
        return self.entries.get(file_path)
    
    def has_file(self, file_path: str) -> bool:
        """Check if a file has been implemented."""
        return file_path in self.entries
    
    def get_implemented_files(self) -> List[str]:
        """Get list of all implemented file paths."""
        return list(self.entries.keys())
    
    def select_relevant_memory(
        self,
        target_file: str,
        dependencies: List[str],
        max_entries: int = 10
    ) -> List[MemoryEntry]:
        """
        Select relevant memory entries for generating a target file.
        
        This implements the SelectRelevantMemory function from the paper:
        Identify dependencies from blueprint, return memory entries
        for files the target depends on.
        
        Args:
            target_file: The file being generated
            dependencies: List of files the target depends on (from blueprint)
            max_entries: Maximum number of entries to return
            
        Returns:
            List of relevant memory entries, ordered by relevance
        """
        relevant = []
        
        # First priority: Direct dependencies from blueprint
        for dep_file in dependencies:
            if dep_file in self.entries:
                relevant.append(self.entries[dep_file])
        
        # Second priority: Transitive dependencies (files that dependencies depend on)
        transitive = set()
        for entry in relevant:
            for trans_dep in entry.edges.get_direct_dependencies():
                if trans_dep in self.entries and trans_dep not in dependencies:
                    transitive.add(trans_dep)
        
        for trans_file in transitive:
            if len(relevant) < max_entries:
                relevant.append(self.entries[trans_file])
        
        # Sort by implementation order (most recent first for context relevance)
        relevant.sort(key=lambda e: e.implementation_order, reverse=True)
        
        return relevant[:max_entries]
    
    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """
        Get the full dependency graph of implemented files.
        
        Returns:
            Dict mapping file paths to their dependencies
        """
        graph = {}
        for file_path, entry in self.entries.items():
            graph[file_path] = entry.edges.get_direct_dependencies()
        return graph
    
    def get_implementation_order(self) -> List[str]:
        """Get files in the order they were implemented."""
        sorted_entries = sorted(
            self.entries.values(),
            key=lambda e: e.implementation_order
        )
        return [e.file_path for e in sorted_entries]
    
    def to_context_string(
        self,
        files: Optional[List[str]] = None,
        compact: bool = False
    ) -> str:
        """
        Convert memory to a context string for LLM injection.
        
        Args:
            files: Specific files to include (None = all)
            compact: Use compact format
            
        Returns:
            Formatted context string
        """
        entries_to_include = []
        
        if files:
            for f in files:
                if f in self.entries:
                    entries_to_include.append(self.entries[f])
        else:
            entries_to_include = list(self.entries.values())
        
        # Sort by implementation order
        entries_to_include.sort(key=lambda e: e.implementation_order)
        
        lines = ["# Code Memory Context", ""]
        for entry in entries_to_include:
            lines.append(entry.to_context_string(include_full_interface=not compact))
            lines.append("")
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)
    
    def clear(self) -> None:
        """Clear all memory entries."""
        self.entries.clear()
        self.implementation_counter = 0


# Convenience type aliases
MemoryContext = Tuple[List[MemoryEntry], str]  # (entries, formatted_context)
