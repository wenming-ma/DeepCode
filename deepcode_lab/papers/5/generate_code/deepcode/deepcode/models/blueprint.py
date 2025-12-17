"""
Implementation Blueprint Model for DeepCode.

Defines the Blueprint model B with its 5 sections:
1. project_file_hierarchy - Directory structure and file organization
2. component_specification - Detailed component descriptions
3. verification_protocol - Testing and validation requirements
4. execution_environment - Runtime and dependency specifications
5. staged_development_plan - Ordered implementation sequence

The blueprint serves as the central planning artifact that guides
code generation and verification phases.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field


class FileType(str, Enum):
    """Types of files in the project structure."""
    PYTHON = "python"
    CONFIG = "config"
    DATA = "data"
    DOCUMENTATION = "documentation"
    TEST = "test"
    RESOURCE = "resource"
    OTHER = "other"


class ComponentType(str, Enum):
    """Types of components in the codebase."""
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    CONSTANT = "constant"
    TYPE_ALIAS = "type_alias"
    PROTOCOL = "protocol"
    DATACLASS = "dataclass"
    ENUM = "enum"


class TestType(str, Enum):
    """Types of tests in verification protocol."""
    UNIT = "unit"
    INTEGRATION = "integration"
    END_TO_END = "end_to_end"
    PERFORMANCE = "performance"
    REGRESSION = "regression"


class DependencyNode(BaseModel):
    """A node in the dependency graph."""
    file_path: str = Field(..., description="Path to the file")
    dependencies: List[str] = Field(default_factory=list, description="Files this file depends on")
    dependents: List[str] = Field(default_factory=list, description="Files that depend on this file")
    
    def add_dependency(self, dep_path: str) -> None:
        """Add a dependency to this node."""
        if dep_path not in self.dependencies:
            self.dependencies.append(dep_path)
    
    def add_dependent(self, dep_path: str) -> None:
        """Add a dependent to this node."""
        if dep_path not in self.dependents:
            self.dependents.append(dep_path)


class DependencyGraph(BaseModel):
    """
    Dependency graph for the project.
    
    Tracks inter-file dependencies to determine implementation order
    and ensure correct context is available during code generation.
    """
    nodes: Dict[str, DependencyNode] = Field(
        default_factory=dict,
        description="Map of file paths to dependency nodes"
    )
    
    def add_file(self, file_path: str) -> DependencyNode:
        """Add a file to the dependency graph."""
        if file_path not in self.nodes:
            self.nodes[file_path] = DependencyNode(file_path=file_path)
        return self.nodes[file_path]
    
    def add_dependency(self, from_file: str, to_file: str) -> None:
        """Add a dependency edge from one file to another."""
        from_node = self.add_file(from_file)
        to_node = self.add_file(to_file)
        from_node.add_dependency(to_file)
        to_node.add_dependent(from_file)
    
    def get_dependencies(self, file_path: str) -> List[str]:
        """Get all files that a given file depends on."""
        if file_path in self.nodes:
            return self.nodes[file_path].dependencies.copy()
        return []
    
    def get_dependents(self, file_path: str) -> List[str]:
        """Get all files that depend on a given file."""
        if file_path in self.nodes:
            return self.nodes[file_path].dependents.copy()
        return []
    
    def get_transitive_dependencies(self, file_path: str, visited: Optional[Set[str]] = None) -> Set[str]:
        """Get all transitive dependencies of a file."""
        if visited is None:
            visited = set()
        
        if file_path in visited:
            return set()
        
        visited.add(file_path)
        result = set()
        
        for dep in self.get_dependencies(file_path):
            result.add(dep)
            result.update(self.get_transitive_dependencies(dep, visited))
        
        return result
    
    def topological_sort(self) -> List[str]:
        """
        Return files in topological order (dependencies before dependents).
        
        Uses Kahn's algorithm for topological sorting.
        """
        # Calculate in-degrees
        in_degree: Dict[str, int] = {path: 0 for path in self.nodes}
        for node in self.nodes.values():
            for dep in node.dependencies:
                if dep in in_degree:
                    pass  # External dependencies don't count
        
        # Count actual in-degrees from internal dependencies
        for path, node in self.nodes.items():
            in_degree[path] = len([d for d in node.dependencies if d in self.nodes])
        
        # Start with nodes that have no dependencies
        queue = [path for path, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            for dependent in self.get_dependents(current):
                if dependent in in_degree:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)
        
        # Handle cycles by adding remaining nodes
        remaining = [path for path in self.nodes if path not in result]
        result.extend(remaining)
        
        return result


class FileSpecification(BaseModel):
    """
    Specification for a single file in the project.
    
    Part of the project_file_hierarchy section of the blueprint.
    """
    path: str = Field(..., description="Relative path to the file")
    file_type: FileType = Field(default=FileType.PYTHON, description="Type of file")
    description: str = Field(default="", description="Purpose and contents of the file")
    dependencies: List[str] = Field(
        default_factory=list,
        description="Other files this file depends on"
    )
    exports: List[str] = Field(
        default_factory=list,
        description="Public names exported by this file"
    )
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Implementation priority (1=highest, 10=lowest)"
    )
    estimated_lines: Optional[int] = Field(
        default=None,
        description="Estimated lines of code"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional file-specific metadata"
    )
    
    @classmethod
    def create(
        cls,
        path: str,
        description: str = "",
        file_type: FileType = FileType.PYTHON,
        dependencies: Optional[List[str]] = None,
        exports: Optional[List[str]] = None,
        priority: int = 5,
        **kwargs: Any
    ) -> "FileSpecification":
        """Factory method for creating file specifications."""
        return cls(
            path=path,
            file_type=file_type,
            description=description,
            dependencies=dependencies or [],
            exports=exports or [],
            priority=priority,
            **kwargs
        )


class ParameterSpec(BaseModel):
    """Specification for a function/method parameter."""
    name: str = Field(..., description="Parameter name")
    type_hint: str = Field(default="Any", description="Type annotation")
    default: Optional[str] = Field(default=None, description="Default value if any")
    description: str = Field(default="", description="Parameter description")


class MethodSpec(BaseModel):
    """Specification for a method within a class."""
    name: str = Field(..., description="Method name")
    parameters: List[ParameterSpec] = Field(default_factory=list, description="Method parameters")
    return_type: str = Field(default="None", description="Return type annotation")
    description: str = Field(default="", description="Method description")
    is_async: bool = Field(default=False, description="Whether method is async")
    is_classmethod: bool = Field(default=False, description="Whether method is a classmethod")
    is_staticmethod: bool = Field(default=False, description="Whether method is a staticmethod")
    is_property: bool = Field(default=False, description="Whether method is a property")


class ClassSpec(BaseModel):
    """Specification for a class component."""
    name: str = Field(..., description="Class name")
    base_classes: List[str] = Field(default_factory=list, description="Parent classes")
    description: str = Field(default="", description="Class description")
    attributes: List[ParameterSpec] = Field(default_factory=list, description="Class attributes")
    methods: List[MethodSpec] = Field(default_factory=list, description="Class methods")
    is_dataclass: bool = Field(default=False, description="Whether class is a dataclass")
    is_abstract: bool = Field(default=False, description="Whether class is abstract")


class FunctionSpec(BaseModel):
    """Specification for a standalone function."""
    name: str = Field(..., description="Function name")
    parameters: List[ParameterSpec] = Field(default_factory=list, description="Function parameters")
    return_type: str = Field(default="None", description="Return type annotation")
    description: str = Field(default="", description="Function description")
    is_async: bool = Field(default=False, description="Whether function is async")


class ComponentSpecification(BaseModel):
    """
    Detailed specification for a component in the codebase.
    
    Part of the component_specification section of the blueprint.
    """
    file_path: str = Field(..., description="File containing this component")
    component_type: ComponentType = Field(..., description="Type of component")
    name: str = Field(..., description="Component name")
    description: str = Field(default="", description="Detailed description")
    
    # Type-specific specifications
    class_spec: Optional[ClassSpec] = Field(default=None, description="Class specification if component is a class")
    function_spec: Optional[FunctionSpec] = Field(default=None, description="Function specification if component is a function")
    
    # Algorithm details (from paper)
    algorithm_reference: Optional[str] = Field(
        default=None,
        description="Reference to algorithm in paper (e.g., 'Algorithm 1')"
    )
    pseudocode: Optional[str] = Field(
        default=None,
        description="Pseudocode from paper"
    )
    equations: List[str] = Field(
        default_factory=list,
        description="Related equations from paper"
    )
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Hyperparameters mentioned in paper"
    )
    
    # Implementation notes
    implementation_notes: List[str] = Field(
        default_factory=list,
        description="Notes for implementation"
    )
    paper_references: List[str] = Field(
        default_factory=list,
        description="References to paper sections"
    )
    
    def to_context_string(self) -> str:
        """Convert to string for LLM context."""
        lines = [
            f"Component: {self.name} ({self.component_type.value})",
            f"File: {self.file_path}",
            f"Description: {self.description}"
        ]
        
        if self.algorithm_reference:
            lines.append(f"Algorithm Reference: {self.algorithm_reference}")
        
        if self.pseudocode:
            lines.append(f"Pseudocode:\n{self.pseudocode}")
        
        if self.equations:
            lines.append("Equations:")
            for eq in self.equations:
                lines.append(f"  - {eq}")
        
        if self.hyperparameters:
            lines.append("Hyperparameters:")
            for key, value in self.hyperparameters.items():
                lines.append(f"  - {key}: {value}")
        
        if self.implementation_notes:
            lines.append("Implementation Notes:")
            for note in self.implementation_notes:
                lines.append(f"  - {note}")
        
        return "\n".join(lines)


class TestCase(BaseModel):
    """Specification for a test case."""
    name: str = Field(..., description="Test case name")
    description: str = Field(default="", description="What the test verifies")
    test_type: TestType = Field(default=TestType.UNIT, description="Type of test")
    target_component: str = Field(..., description="Component being tested")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Test inputs")
    expected_outputs: Dict[str, Any] = Field(default_factory=dict, description="Expected outputs")
    assertions: List[str] = Field(default_factory=list, description="Assertions to make")


class VerificationProtocol(BaseModel):
    """
    Verification and testing requirements.
    
    Part of the verification_protocol section of the blueprint.
    """
    test_cases: List[TestCase] = Field(
        default_factory=list,
        description="Specific test cases to implement"
    )
    success_criteria: List[str] = Field(
        default_factory=list,
        description="Criteria for successful implementation"
    )
    validation_steps: List[str] = Field(
        default_factory=list,
        description="Steps to validate the implementation"
    )
    expected_outputs: Dict[str, str] = Field(
        default_factory=dict,
        description="Expected outputs for key operations"
    )
    performance_requirements: Dict[str, Any] = Field(
        default_factory=dict,
        description="Performance requirements (speed, memory, etc.)"
    )
    
    def add_test_case(self, test_case: TestCase) -> None:
        """Add a test case to the protocol."""
        self.test_cases.append(test_case)
    
    def add_success_criterion(self, criterion: str) -> None:
        """Add a success criterion."""
        self.success_criteria.append(criterion)
    
    def to_context_string(self) -> str:
        """Convert to string for LLM context."""
        lines = ["Verification Protocol:"]
        
        if self.success_criteria:
            lines.append("\nSuccess Criteria:")
            for i, criterion in enumerate(self.success_criteria, 1):
                lines.append(f"  {i}. {criterion}")
        
        if self.validation_steps:
            lines.append("\nValidation Steps:")
            for i, step in enumerate(self.validation_steps, 1):
                lines.append(f"  {i}. {step}")
        
        if self.test_cases:
            lines.append(f"\nTest Cases: {len(self.test_cases)} defined")
        
        return "\n".join(lines)


class PackageDependency(BaseModel):
    """Specification for an external package dependency."""
    name: str = Field(..., description="Package name")
    version: str = Field(default="", description="Version constraint (e.g., '>=1.0.0')")
    optional: bool = Field(default=False, description="Whether dependency is optional")
    purpose: str = Field(default="", description="Why this dependency is needed")


class ExecutionEnvironment(BaseModel):
    """
    Runtime and dependency specifications.
    
    Part of the execution_environment section of the blueprint.
    """
    python_version: str = Field(default=">=3.10", description="Required Python version")
    dependencies: List[PackageDependency] = Field(
        default_factory=list,
        description="External package dependencies"
    )
    dev_dependencies: List[PackageDependency] = Field(
        default_factory=list,
        description="Development dependencies"
    )
    environment_variables: Dict[str, str] = Field(
        default_factory=dict,
        description="Required environment variables"
    )
    system_requirements: List[str] = Field(
        default_factory=list,
        description="System-level requirements"
    )
    hardware_requirements: Dict[str, str] = Field(
        default_factory=dict,
        description="Hardware requirements (RAM, GPU, etc.)"
    )
    
    def add_dependency(
        self,
        name: str,
        version: str = "",
        optional: bool = False,
        purpose: str = ""
    ) -> None:
        """Add a package dependency."""
        self.dependencies.append(PackageDependency(
            name=name,
            version=version,
            optional=optional,
            purpose=purpose
        ))
    
    def to_requirements_txt(self) -> str:
        """Generate requirements.txt content."""
        lines = []
        for dep in self.dependencies:
            if dep.version:
                lines.append(f"{dep.name}{dep.version}")
            else:
                lines.append(dep.name)
        return "\n".join(lines)
    
    def to_context_string(self) -> str:
        """Convert to string for LLM context."""
        lines = [
            "Execution Environment:",
            f"  Python: {self.python_version}",
            f"  Dependencies: {len(self.dependencies)} packages"
        ]
        
        if self.environment_variables:
            lines.append("  Environment Variables:")
            for key in self.environment_variables:
                lines.append(f"    - {key}")
        
        return "\n".join(lines)


class ImplementationStage(BaseModel):
    """A stage in the staged development plan."""
    stage_number: int = Field(..., description="Stage number (1-indexed)")
    name: str = Field(..., description="Stage name")
    description: str = Field(default="", description="Stage description")
    files: List[str] = Field(default_factory=list, description="Files to implement in this stage")
    prerequisites: List[int] = Field(
        default_factory=list,
        description="Stage numbers that must be completed first"
    )
    validation_criteria: List[str] = Field(
        default_factory=list,
        description="Criteria to validate stage completion"
    )


class StagedDevelopmentPlan(BaseModel):
    """
    Ordered implementation sequence.
    
    Part of the staged_development_plan section of the blueprint.
    Defines the order in which files should be implemented to
    ensure dependencies are satisfied.
    """
    stages: List[ImplementationStage] = Field(
        default_factory=list,
        description="Implementation stages in order"
    )
    file_order: List[str] = Field(
        default_factory=list,
        description="Flat list of files in implementation order"
    )
    
    def add_stage(self, stage: ImplementationStage) -> None:
        """Add a stage to the plan."""
        self.stages.append(stage)
        self.file_order.extend(stage.files)
    
    def get_stage_for_file(self, file_path: str) -> Optional[ImplementationStage]:
        """Get the stage that contains a given file."""
        for stage in self.stages:
            if file_path in stage.files:
                return stage
        return None
    
    def get_files_before(self, file_path: str) -> List[str]:
        """Get all files that should be implemented before a given file."""
        if file_path not in self.file_order:
            return []
        index = self.file_order.index(file_path)
        return self.file_order[:index]
    
    def __iter__(self):
        """Iterate over files in implementation order."""
        return iter(self.file_order)
    
    def __len__(self) -> int:
        """Return number of files in the plan."""
        return len(self.file_order)


class Blueprint(BaseModel):
    """
    Implementation Blueprint Model B.
    
    The central planning artifact that guides code generation and verification.
    Contains 5 sections:
    1. project_file_hierarchy - Directory structure and file organization
    2. component_specification - Detailed component descriptions
    3. verification_protocol - Testing and validation requirements
    4. execution_environment - Runtime and dependency specifications
    5. staged_development_plan - Ordered implementation sequence
    """
    
    # Metadata
    title: str = Field(..., description="Project title")
    description: str = Field(default="", description="Project description")
    source_document: Optional[str] = Field(
        default=None,
        description="Path to source document (paper)"
    )
    version: str = Field(default="1.0.0", description="Blueprint version")
    
    # Section 1: Project File Hierarchy
    project_file_hierarchy: List[FileSpecification] = Field(
        default_factory=list,
        description="All files in the project with their specifications"
    )
    
    # Section 2: Component Specification
    component_specifications: List[ComponentSpecification] = Field(
        default_factory=list,
        description="Detailed specifications for each component"
    )
    
    # Section 3: Verification Protocol
    verification_protocol: VerificationProtocol = Field(
        default_factory=VerificationProtocol,
        description="Testing and validation requirements"
    )
    
    # Section 4: Execution Environment
    execution_environment: ExecutionEnvironment = Field(
        default_factory=ExecutionEnvironment,
        description="Runtime and dependency specifications"
    )
    
    # Section 5: Staged Development Plan
    staged_development_plan: StagedDevelopmentPlan = Field(
        default_factory=StagedDevelopmentPlan,
        description="Ordered implementation sequence"
    )
    
    # Internal dependency graph
    _dependency_graph: Optional[DependencyGraph] = None
    
    def model_post_init(self, __context: Any) -> None:
        """Build dependency graph after initialization."""
        self._build_dependency_graph()
    
    def _build_dependency_graph(self) -> None:
        """Build the internal dependency graph from file specifications."""
        self._dependency_graph = DependencyGraph()
        
        for file_spec in self.project_file_hierarchy:
            self._dependency_graph.add_file(file_spec.path)
            for dep in file_spec.dependencies:
                self._dependency_graph.add_dependency(file_spec.path, dep)
    
    def get_dependency_graph(self) -> DependencyGraph:
        """Get the dependency graph."""
        if self._dependency_graph is None:
            self._build_dependency_graph()
        return self._dependency_graph
    
    def get_dependencies(self, file_path: str) -> List[str]:
        """Get all files that a given file depends on."""
        return self.get_dependency_graph().get_dependencies(file_path)
    
    def get_transitive_dependencies(self, file_path: str) -> Set[str]:
        """Get all transitive dependencies of a file."""
        return self.get_dependency_graph().get_transitive_dependencies(file_path)
    
    def get_implementation_order(self) -> List[str]:
        """
        Get files in implementation order.
        
        Uses the staged development plan if available,
        otherwise falls back to topological sort.
        """
        if self.staged_development_plan.file_order:
            return self.staged_development_plan.file_order
        return self.get_dependency_graph().topological_sort()
    
    def get_file_specification(self, file_path: str) -> Optional[FileSpecification]:
        """Get the specification for a specific file."""
        for spec in self.project_file_hierarchy:
            if spec.path == file_path:
                return spec
        return None
    
    def get_component_specifications(self, file_path: str) -> List[ComponentSpecification]:
        """Get all component specifications for a file."""
        return [
            spec for spec in self.component_specifications
            if spec.file_path == file_path
        ]
    
    def add_file(self, file_spec: FileSpecification) -> None:
        """Add a file specification to the blueprint."""
        self.project_file_hierarchy.append(file_spec)
        self._build_dependency_graph()
    
    def add_component(self, component_spec: ComponentSpecification) -> None:
        """Add a component specification to the blueprint."""
        self.component_specifications.append(component_spec)
    
    def to_context_string(self, include_components: bool = True) -> str:
        """
        Convert blueprint to string for LLM context.
        
        Args:
            include_components: Whether to include detailed component specs
        """
        lines = [
            f"# Blueprint: {self.title}",
            f"Description: {self.description}",
            "",
            "## Project Structure",
            f"Total files: {len(self.project_file_hierarchy)}",
            ""
        ]
        
        # File hierarchy
        for file_spec in self.project_file_hierarchy:
            lines.append(f"- {file_spec.path}: {file_spec.description}")
        
        # Component specifications
        if include_components and self.component_specifications:
            lines.append("")
            lines.append("## Components")
            for comp in self.component_specifications:
                lines.append(comp.to_context_string())
                lines.append("")
        
        # Verification protocol
        lines.append(self.verification_protocol.to_context_string())
        
        # Execution environment
        lines.append("")
        lines.append(self.execution_environment.to_context_string())
        
        # Implementation order
        lines.append("")
        lines.append("## Implementation Order")
        for i, file_path in enumerate(self.get_implementation_order(), 1):
            lines.append(f"  {i}. {file_path}")
        
        return "\n".join(lines)
    
    def get_context_for_file(self, target_file: str) -> str:
        """
        Get relevant blueprint context for implementing a specific file.
        
        Includes:
        - File specification
        - Component specifications for the file
        - Dependency information
        - Related verification criteria
        """
        lines = [f"# Context for implementing: {target_file}", ""]
        
        # File specification
        file_spec = self.get_file_specification(target_file)
        if file_spec:
            lines.append("## File Specification")
            lines.append(f"Type: {file_spec.file_type.value}")
            lines.append(f"Description: {file_spec.description}")
            lines.append(f"Exports: {', '.join(file_spec.exports) if file_spec.exports else 'None specified'}")
            lines.append("")
        
        # Dependencies
        deps = self.get_dependencies(target_file)
        if deps:
            lines.append("## Dependencies")
            for dep in deps:
                dep_spec = self.get_file_specification(dep)
                if dep_spec:
                    lines.append(f"- {dep}: {dep_spec.description}")
                else:
                    lines.append(f"- {dep}")
            lines.append("")
        
        # Component specifications
        components = self.get_component_specifications(target_file)
        if components:
            lines.append("## Components to Implement")
            for comp in components:
                lines.append(comp.to_context_string())
                lines.append("")
        
        return "\n".join(lines)
    
    @classmethod
    def create(
        cls,
        title: str,
        description: str = "",
        source_document: Optional[str] = None,
        **kwargs: Any
    ) -> "Blueprint":
        """Factory method for creating blueprints."""
        return cls(
            title=title,
            description=description,
            source_document=source_document,
            **kwargs
        )


# Type aliases for convenience
FileHierarchy = List[FileSpecification]
ComponentSpecs = List[ComponentSpecification]
