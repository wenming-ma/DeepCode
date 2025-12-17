"""
Code Repository Model for DeepCode.

Defines the repository model P = (T, C, M) where:
- T: Directory tree structure
- C: Code files (content)
- M: Manifest (metadata and configuration)

This model represents the generated codebase output from the DeepCode pipeline.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple
from datetime import datetime
import hashlib
import json

from pydantic import BaseModel, Field


class FileStatus(Enum):
    """Status of a code file in the repository."""
    PENDING = "pending"          # Not yet generated
    GENERATED = "generated"      # Generated but not verified
    VERIFIED = "verified"        # Passed verification
    ERROR = "error"              # Has errors
    MODIFIED = "modified"        # Modified after generation


class CodeFile(BaseModel):
    """
    Represents a single code file in the repository.
    
    Attributes:
        path: Relative path from repository root
        content: File content as string
        language: Programming language (e.g., 'python', 'yaml')
        status: Current status of the file
        checksum: SHA-256 hash of content for change detection
        line_count: Number of lines in the file
        created_at: Timestamp when file was created
        modified_at: Timestamp of last modification
        metadata: Additional file-specific metadata
    """
    path: str = Field(..., description="Relative path from repository root")
    content: str = Field(default="", description="File content")
    language: str = Field(default="python", description="Programming language")
    status: FileStatus = Field(default=FileStatus.PENDING, description="File status")
    checksum: Optional[str] = Field(default=None, description="SHA-256 hash of content")
    line_count: int = Field(default=0, description="Number of lines")
    created_at: Optional[datetime] = Field(default=None, description="Creation timestamp")
    modified_at: Optional[datetime] = Field(default=None, description="Last modification timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    def model_post_init(self, __context: Any) -> None:
        """Post-initialization to compute derived fields."""
        if self.content:
            self.line_count = len(self.content.splitlines())
            self.checksum = hashlib.sha256(self.content.encode()).hexdigest()
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.modified_at is None:
            self.modified_at = self.created_at
    
    def update_content(self, new_content: str) -> None:
        """Update file content and recompute derived fields."""
        self.content = new_content
        self.line_count = len(new_content.splitlines())
        self.checksum = hashlib.sha256(new_content.encode()).hexdigest()
        self.modified_at = datetime.now()
        if self.status == FileStatus.VERIFIED:
            self.status = FileStatus.MODIFIED
    
    def get_extension(self) -> str:
        """Get file extension."""
        return Path(self.path).suffix
    
    def get_filename(self) -> str:
        """Get filename without directory."""
        return Path(self.path).name
    
    def get_directory(self) -> str:
        """Get parent directory path."""
        return str(Path(self.path).parent)
    
    @classmethod
    def create(
        cls,
        path: str,
        content: str = "",
        language: Optional[str] = None,
        **kwargs: Any
    ) -> "CodeFile":
        """Factory method to create a CodeFile with auto-detected language."""
        if language is None:
            language = cls._detect_language(path)
        return cls(path=path, content=content, language=language, **kwargs)
    
    @staticmethod
    def _detect_language(path: str) -> str:
        """Detect programming language from file extension."""
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".rs": "rust",
            ".go": "go",
            ".rb": "ruby",
            ".php": "php",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".toml": "toml",
            ".md": "markdown",
            ".txt": "text",
            ".sh": "bash",
            ".bat": "batch",
            ".sql": "sql",
            ".html": "html",
            ".css": "css",
            ".xml": "xml",
        }
        ext = Path(path).suffix.lower()
        return ext_map.get(ext, "text")


class DirectoryNode(BaseModel):
    """
    Represents a node in the directory tree.
    
    Attributes:
        name: Directory or file name
        path: Full relative path from root
        is_directory: Whether this is a directory
        children: Child nodes (for directories)
        file_ref: Reference to CodeFile (for files)
    """
    name: str = Field(..., description="Node name")
    path: str = Field(..., description="Full relative path")
    is_directory: bool = Field(default=True, description="Is this a directory?")
    children: Dict[str, "DirectoryNode"] = Field(
        default_factory=dict, 
        description="Child nodes"
    )
    file_ref: Optional[str] = Field(
        default=None, 
        description="Reference to file path in CodeFiles"
    )
    
    def add_child(self, node: "DirectoryNode") -> None:
        """Add a child node."""
        self.children[node.name] = node
    
    def get_child(self, name: str) -> Optional["DirectoryNode"]:
        """Get a child node by name."""
        return self.children.get(name)
    
    def list_files(self) -> List[str]:
        """Recursively list all file paths under this node."""
        files = []
        if not self.is_directory and self.file_ref:
            files.append(self.file_ref)
        for child in self.children.values():
            files.extend(child.list_files())
        return files
    
    def list_directories(self) -> List[str]:
        """Recursively list all directory paths under this node."""
        dirs = []
        if self.is_directory:
            dirs.append(self.path)
            for child in self.children.values():
                dirs.extend(child.list_directories())
        return dirs
    
    def to_tree_string(self, prefix: str = "", is_last: bool = True) -> str:
        """Generate a tree visualization string."""
        connector = "└── " if is_last else "├── "
        result = f"{prefix}{connector}{self.name}\n" if prefix else f"{self.name}/\n"
        
        if self.is_directory:
            children = list(self.children.values())
            for i, child in enumerate(children):
                is_last_child = i == len(children) - 1
                new_prefix = prefix + ("    " if is_last else "│   ")
                result += child.to_tree_string(new_prefix, is_last_child)
        
        return result


class DirectoryTree(BaseModel):
    """
    Represents the complete directory tree structure T.
    
    Attributes:
        root: Root directory node
        root_path: Name of the root directory
    """
    root: DirectoryNode = Field(..., description="Root directory node")
    root_path: str = Field(default=".", description="Root directory path")
    
    @classmethod
    def create(cls, root_name: str = "project") -> "DirectoryTree":
        """Create an empty directory tree."""
        root = DirectoryNode(name=root_name, path=root_name, is_directory=True)
        return cls(root=root, root_path=root_name)
    
    def add_file(self, file_path: str) -> None:
        """Add a file path to the tree, creating directories as needed."""
        parts = Path(file_path).parts
        current = self.root
        
        # Create directory nodes
        for i, part in enumerate(parts[:-1]):
            if part not in current.children:
                dir_path = str(Path(*parts[:i+1]))
                new_node = DirectoryNode(
                    name=part, 
                    path=dir_path, 
                    is_directory=True
                )
                current.add_child(new_node)
            current = current.children[part]
        
        # Create file node
        if parts:
            file_name = parts[-1]
            file_node = DirectoryNode(
                name=file_name,
                path=file_path,
                is_directory=False,
                file_ref=file_path
            )
            current.add_child(file_node)
    
    def remove_file(self, file_path: str) -> bool:
        """Remove a file from the tree."""
        parts = Path(file_path).parts
        if not parts:
            return False
        
        # Navigate to parent
        current = self.root
        for part in parts[:-1]:
            if part not in current.children:
                return False
            current = current.children[part]
        
        # Remove file
        file_name = parts[-1]
        if file_name in current.children:
            del current.children[file_name]
            return True
        return False
    
    def get_structure(self) -> str:
        """Get a string representation of the directory structure."""
        return self.root.to_tree_string()
    
    def list_all_files(self) -> List[str]:
        """List all file paths in the tree."""
        return self.root.list_files()
    
    def list_all_directories(self) -> List[str]:
        """List all directory paths in the tree."""
        return self.root.list_directories()
    
    def path_exists(self, path: str) -> bool:
        """Check if a path exists in the tree."""
        parts = Path(path).parts
        current = self.root
        
        for part in parts:
            if part not in current.children:
                return False
            current = current.children[part]
        return True


class Manifest(BaseModel):
    """
    Repository manifest containing metadata and configuration M.
    
    Attributes:
        name: Project name
        version: Project version
        description: Project description
        authors: List of authors
        source_document: Path to source document (paper)
        created_at: Creation timestamp
        modified_at: Last modification timestamp
        python_version: Required Python version
        dependencies: Project dependencies
        entry_points: Main entry point files
        test_command: Command to run tests
        build_command: Command to build project
        metadata: Additional metadata
    """
    name: str = Field(..., description="Project name")
    version: str = Field(default="0.1.0", description="Project version")
    description: str = Field(default="", description="Project description")
    authors: List[str] = Field(default_factory=list, description="Authors")
    source_document: Optional[str] = Field(
        default=None, 
        description="Source document path"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, 
        description="Creation timestamp"
    )
    modified_at: datetime = Field(
        default_factory=datetime.now, 
        description="Last modification"
    )
    python_version: str = Field(default=">=3.10", description="Python version")
    dependencies: Dict[str, str] = Field(
        default_factory=dict, 
        description="Dependencies with versions"
    )
    entry_points: List[str] = Field(
        default_factory=list, 
        description="Entry point files"
    )
    test_command: str = Field(default="pytest", description="Test command")
    build_command: Optional[str] = Field(default=None, description="Build command")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional metadata"
    )
    
    def add_dependency(self, name: str, version: str = "*") -> None:
        """Add a dependency."""
        self.dependencies[name] = version
        self.modified_at = datetime.now()
    
    def remove_dependency(self, name: str) -> bool:
        """Remove a dependency."""
        if name in self.dependencies:
            del self.dependencies[name]
            self.modified_at = datetime.now()
            return True
        return False
    
    def to_pyproject_toml(self) -> str:
        """Generate pyproject.toml content."""
        deps = [f'"{k}>={v}"' if v != "*" else f'"{k}"' 
                for k, v in self.dependencies.items()]
        deps_str = ",\n    ".join(deps)
        
        return f'''[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "{self.name}"
version = "{self.version}"
description = "{self.description}"
authors = [{", ".join(f'{{"name": "{a}"}}' for a in self.authors)}]
requires-python = "{self.python_version}"
dependencies = [
    {deps_str}
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "mypy>=1.0.0",
    "ruff>=0.1.0",
]
'''
    
    def to_requirements_txt(self) -> str:
        """Generate requirements.txt content."""
        lines = []
        for name, version in sorted(self.dependencies.items()):
            if version == "*":
                lines.append(name)
            else:
                lines.append(f"{name}>={version}")
        return "\n".join(lines)


class Repository(BaseModel):
    """
    Code Repository Model P = (T, C, M).
    
    Represents the complete generated codebase with:
    - T (tree): Directory structure
    - C (code): Code file contents
    - M (manifest): Project metadata
    
    Attributes:
        tree: Directory tree structure
        files: Dictionary of code files by path
        manifest: Project manifest
        generation_log: Log of generation events
    """
    tree: DirectoryTree = Field(..., description="Directory tree T")
    files: Dict[str, CodeFile] = Field(
        default_factory=dict, 
        description="Code files C"
    )
    manifest: Manifest = Field(..., description="Manifest M")
    generation_log: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Generation event log"
    )
    
    @classmethod
    def create(
        cls,
        name: str,
        description: str = "",
        source_document: Optional[str] = None,
        **kwargs: Any
    ) -> "Repository":
        """Factory method to create a new repository."""
        tree = DirectoryTree.create(root_name=name)
        manifest = Manifest(
            name=name,
            description=description,
            source_document=source_document,
            **kwargs
        )
        return cls(tree=tree, files={}, manifest=manifest)
    
    def add_file(
        self,
        path: str,
        content: str = "",
        language: Optional[str] = None,
        status: FileStatus = FileStatus.GENERATED,
        **kwargs: Any
    ) -> CodeFile:
        """
        Add a new file to the repository.
        
        Args:
            path: Relative file path
            content: File content
            language: Programming language (auto-detected if None)
            status: Initial file status
            **kwargs: Additional CodeFile attributes
            
        Returns:
            The created CodeFile
        """
        code_file = CodeFile.create(
            path=path,
            content=content,
            language=language,
            status=status,
            **kwargs
        )
        self.files[path] = code_file
        self.tree.add_file(path)
        
        # Log the addition
        self._log_event("file_added", {"path": path, "status": status.value})
        
        return code_file
    
    def update_file(self, path: str, content: str) -> Optional[CodeFile]:
        """Update an existing file's content."""
        if path not in self.files:
            return None
        
        self.files[path].update_content(content)
        self._log_event("file_updated", {"path": path})
        return self.files[path]
    
    def get_file(self, path: str) -> Optional[CodeFile]:
        """Get a file by path."""
        return self.files.get(path)
    
    def remove_file(self, path: str) -> bool:
        """Remove a file from the repository."""
        if path not in self.files:
            return False
        
        del self.files[path]
        self.tree.remove_file(path)
        self._log_event("file_removed", {"path": path})
        return True
    
    def get_structure(self) -> str:
        """Get the directory tree structure as a string."""
        return self.tree.get_structure()
    
    def list_files(self, status: Optional[FileStatus] = None) -> List[str]:
        """
        List all file paths, optionally filtered by status.
        
        Args:
            status: Filter by this status (None for all files)
            
        Returns:
            List of file paths
        """
        if status is None:
            return list(self.files.keys())
        return [p for p, f in self.files.items() if f.status == status]
    
    def get_files_by_language(self, language: str) -> List[CodeFile]:
        """Get all files of a specific language."""
        return [f for f in self.files.values() if f.language == language]
    
    def get_files_by_directory(self, directory: str) -> List[CodeFile]:
        """Get all files in a specific directory."""
        return [f for f in self.files.values() 
                if f.get_directory() == directory or 
                f.get_directory().startswith(directory + "/")]
    
    def validate_consistency(self) -> Tuple[bool, List[str]]:
        """
        Validate repository consistency.
        
        Checks:
        - All files in tree exist in files dict
        - All files in dict exist in tree
        - No empty files marked as verified
        
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        # Check tree files exist in dict
        tree_files = set(self.tree.list_all_files())
        dict_files = set(self.files.keys())
        
        missing_in_dict = tree_files - dict_files
        if missing_in_dict:
            issues.append(f"Files in tree but not in dict: {missing_in_dict}")
        
        missing_in_tree = dict_files - tree_files
        if missing_in_tree:
            issues.append(f"Files in dict but not in tree: {missing_in_tree}")
        
        # Check for empty verified files
        for path, file in self.files.items():
            if file.status == FileStatus.VERIFIED and not file.content.strip():
                issues.append(f"Empty file marked as verified: {path}")
        
        return len(issues) == 0, issues
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get repository statistics."""
        total_lines = sum(f.line_count for f in self.files.values())
        by_status = {}
        by_language = {}
        
        for file in self.files.values():
            status_name = file.status.value
            by_status[status_name] = by_status.get(status_name, 0) + 1
            by_language[file.language] = by_language.get(file.language, 0) + 1
        
        return {
            "total_files": len(self.files),
            "total_lines": total_lines,
            "by_status": by_status,
            "by_language": by_language,
            "directories": len(self.tree.list_all_directories()),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert repository to dictionary for serialization."""
        return {
            "manifest": self.manifest.model_dump(),
            "files": {p: f.model_dump() for p, f in self.files.items()},
            "structure": self.get_structure(),
            "statistics": self.get_statistics(),
        }
    
    def export_to_disk(self, output_dir: str) -> List[str]:
        """
        Export repository to disk.
        
        Args:
            output_dir: Directory to export to
            
        Returns:
            List of created file paths
        """
        created = []
        output_path = Path(output_dir)
        
        for file_path, code_file in self.files.items():
            full_path = output_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(code_file.content)
            created.append(str(full_path))
        
        # Write manifest files
        requirements_path = output_path / "requirements.txt"
        requirements_path.write_text(self.manifest.to_requirements_txt())
        created.append(str(requirements_path))
        
        return created
    
    def _log_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log a generation event."""
        self.generation_log.append({
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "details": details,
        })
    
    def __iter__(self) -> Iterator[Tuple[str, CodeFile]]:
        """Iterate over files."""
        return iter(self.files.items())
    
    def __len__(self) -> int:
        """Get number of files."""
        return len(self.files)
    
    def __contains__(self, path: str) -> bool:
        """Check if a file exists."""
        return path in self.files
