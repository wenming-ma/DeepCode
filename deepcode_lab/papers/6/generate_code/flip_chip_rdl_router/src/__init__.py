"""
Flip-Chip RDL Router Source Package

This package provides a complete implementation of a network-flow-based
RDL (Redistribution Layer) routing algorithm for flip-chip package design.

The router uses a two-stage approach:
1. Global Routing: MCMF (Minimum-Cost Maximum-Flow) algorithm for optimal
   IO pad to bump pad assignment
2. Detailed Routing: River routing methodology for single-layer RDL
   interconnection with planarity constraints

Subpackages:
    - data_structures: Core data structures (Chip, BumpPad, IOPad, Net)
    - core: Core algorithms (MCMF, geometry, river routing)
    - routing: Routing algorithms (global, detailed, net ordering, layer assignment)
    - utils: Utilities (parser, visualizer)

Example Usage:
    >>> from flip_chip_rdl_router.src import Chip, full_routing_pipeline
    >>> chip = Chip("test", die_width=1000, die_height=1000)
    >>> chip.create_bump_grid(10, 10, pitch=100)
    >>> chip.create_peripheral_io_pads(num_pads_per_side=25)
    >>> result = full_routing_pipeline(chip)
    >>> print(f"Routed {result['num_routed']} nets")
"""

__version__ = '1.0.0'
__author__ = 'Flip-Chip RDL Router Team'

# =============================================================================
# Data Structures
# =============================================================================
from .data_structures import (
    Chip,
    BumpPad,
    IOPad,
    Net,
)

# =============================================================================
# Core Algorithms
# =============================================================================
from .core import (
    # Geometry primitives
    Point,
    Segment,
    BoundingBox,
    manhattan_distance,
    euclidean_distance,
    generate_l_shape_path,
    generate_z_shape_path,
    simplify_path,
    path_length,
    path_manhattan_length,
    count_bends,
    paths_intersect,
    would_cross,
    is_planar_routing,
    
    # MCMF algorithm
    Edge,
    MCMF,
    MCMFMatrix,
    create_bipartite_mcmf,
    INF,
    
    # Network builder
    NodeType,
    NetworkNode,
    NetworkEdge,
    FlowNetwork,
    NetworkBuilder,
    build_routing_network,
    solve_assignment,
    
    # River routing
    RouteType,
    RoutingDirection,
    RouteSegment,
    Route,
    RiverRoutingResult,
    RiverRouter,
    river_route,
    check_planarity,
)

# =============================================================================
# Routing Algorithms
# =============================================================================
from .routing import (
    # Global routing
    GlobalRouter,
    GlobalRoutingResult,
    Assignment,
    RoutingConstraints,
    RoutingStatus,
    global_route,
    compute_lower_bound_wirelength,
    
    # Detailed routing
    DetailedRouter,
    DetailedRoutingResult,
    DetailedRoute,
    DetailedRoutingStatus,
    DRCChecker,
    DRCViolation,
    DRCViolationType,
    detailed_route,
    route_with_global,
    
    # Net ordering
    NetOrderer,
    OrderingResult,
    OrderingStrategy,
    OrderingStatus,
    NetAssignment,
    PrecedenceGraph,
    order_nets,
    must_route_before,
    find_optimal_ordering,
    
    # Layer assignment
    LayerAssigner,
    LayerAssignmentResult,
    LayerRoute,
    LayerInfo,
    Via,
    LayerType,
    AssignmentStrategy,
    AssignmentStatus,
    assign_layers,
    assign_chip_layers,
    validate_single_layer_routing,
    
    # Pipeline functions
    full_routing_pipeline,
    quick_route,
)

# =============================================================================
# Utilities
# =============================================================================
from .utils import (
    ChipParser,
    ParseResult,
    BenchmarkGenerator,
    load_chip,
    save_chip_to_yaml,
    save_chip_to_json,
)

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Version
    '__version__',
    
    # Data Structures
    'Chip',
    'BumpPad',
    'IOPad',
    'Net',
    
    # Geometry
    'Point',
    'Segment',
    'BoundingBox',
    'manhattan_distance',
    'euclidean_distance',
    'generate_l_shape_path',
    'generate_z_shape_path',
    'simplify_path',
    'path_length',
    'path_manhattan_length',
    'count_bends',
    'paths_intersect',
    'would_cross',
    'is_planar_routing',
    
    # MCMF
    'Edge',
    'MCMF',
    'MCMFMatrix',
    'create_bipartite_mcmf',
    'INF',
    
    # Network Builder
    'NodeType',
    'NetworkNode',
    'NetworkEdge',
    'FlowNetwork',
    'NetworkBuilder',
    'build_routing_network',
    'solve_assignment',
    
    # River Routing
    'RouteType',
    'RoutingDirection',
    'RouteSegment',
    'Route',
    'RiverRoutingResult',
    'RiverRouter',
    'river_route',
    'check_planarity',
    
    # Global Routing
    'GlobalRouter',
    'GlobalRoutingResult',
    'Assignment',
    'RoutingConstraints',
    'RoutingStatus',
    'global_route',
    'compute_lower_bound_wirelength',
    
    # Detailed Routing
    'DetailedRouter',
    'DetailedRoutingResult',
    'DetailedRoute',
    'DetailedRoutingStatus',
    'DRCChecker',
    'DRCViolation',
    'DRCViolationType',
    'detailed_route',
    'route_with_global',
    
    # Net Ordering
    'NetOrderer',
    'OrderingResult',
    'OrderingStrategy',
    'OrderingStatus',
    'NetAssignment',
    'PrecedenceGraph',
    'order_nets',
    'must_route_before',
    'find_optimal_ordering',
    
    # Layer Assignment
    'LayerAssigner',
    'LayerAssignmentResult',
    'LayerRoute',
    'LayerInfo',
    'Via',
    'LayerType',
    'AssignmentStrategy',
    'AssignmentStatus',
    'assign_layers',
    'assign_chip_layers',
    'validate_single_layer_routing',
    
    # Pipeline Functions
    'full_routing_pipeline',
    'quick_route',
    
    # Utilities
    'ChipParser',
    'ParseResult',
    'BenchmarkGenerator',
    'load_chip',
    'save_chip_to_yaml',
    'save_chip_to_json',
]


def get_version():
    """Return the package version string."""
    return __version__


def get_info():
    """Return package information dictionary."""
    return {
        'name': 'flip_chip_rdl_router',
        'version': __version__,
        'description': 'Network-flow-based RDL routing for flip-chip design',
        'algorithms': ['MCMF', 'River Routing', 'Topological Net Ordering'],
        'stages': ['Global Routing', 'Detailed Routing', 'Layer Assignment'],
    }


def create_simple_chip(name: str = "simple_chip",
                       die_size: float = 1000.0,
                       bump_grid = 10,
                       io_per_side: int = 25,
                       bump_pitch: float = 100.0) -> Chip:
    """
    Create a simple chip configuration for quick testing.

    Args:
        name: Chip name
        die_size: Die width and height (square die)
        bump_grid: Number of bumps per row/column (int) or tuple (rows, cols)
        io_per_side: Number of IO pads per die side
        bump_pitch: Spacing between bump pads

    Returns:
        Configured Chip object ready for routing
    """
    chip = Chip(
        name=name,
        die_width=die_size,
        die_height=die_size,
        bump_pitch=bump_pitch
    )

    # Parse bump_grid parameter
    if isinstance(bump_grid, tuple):
        rows, cols = bump_grid
    else:
        rows = cols = bump_grid

    # Create bump pad grid
    chip.create_bump_grid(
        rows=rows,
        cols=cols
    )
    
    # Create peripheral IO pads
    chip.create_peripheral_io_pads(
        num_pads_per_side=io_per_side
    )
    
    return chip


def route_chip(chip: Chip,
               wire_width: float = 10.0,
               wire_spacing: float = 10.0,
               enable_drc: bool = True) -> dict:
    """
    Route a chip using the full routing pipeline.
    
    This is a convenience function that runs global routing,
    detailed routing, and layer assignment in sequence.
    
    Args:
        chip: Chip object to route
        wire_width: Wire width in microns
        wire_spacing: Wire spacing in microns
        enable_drc: Enable DRC checking
        
    Returns:
        Dictionary containing routing results and statistics
    """
    return full_routing_pipeline(
        chip=chip,
        wire_width=wire_width,
        wire_spacing=wire_spacing,
        enable_drc=enable_drc
    )
