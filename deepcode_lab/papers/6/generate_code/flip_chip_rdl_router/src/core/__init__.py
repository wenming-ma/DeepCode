"""
Core algorithms package for flip-chip RDL routing.

This package contains the fundamental algorithms and utilities:
- MCMF: Minimum-Cost Maximum-Flow algorithm for global routing optimization
- Network Builder: Flow network construction from chip layouts
- River Router: Detailed single-layer routing with planarity constraints
- Geometry: Geometric primitives and utilities for routing calculations
"""

from .geometry import (
    Point,
    Segment,
    BoundingBox,
    cross_product,
    segments_intersect,
    segment_intersection_point,
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
    count_crossings,
    is_planar_routing,
)

from .mcmf import (
    Edge,
    MCMF,
    MCMFMatrix,
    create_bipartite_mcmf,
    INF,
)

from .network_builder import (
    NodeType,
    NetworkNode,
    NetworkEdge,
    FlowNetwork,
    NetworkBuilder,
    build_routing_network,
    solve_assignment,
)

from .river_router import (
    RouteType,
    RoutingDirection,
    RouteSegment,
    Route,
    RiverRoutingResult,
    RiverRouter,
    river_route,
    check_planarity,
)

__all__ = [
    # Geometry primitives
    'Point',
    'Segment',
    'BoundingBox',
    # Geometry functions
    'cross_product',
    'segments_intersect',
    'segment_intersection_point',
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
    'count_crossings',
    'is_planar_routing',
    # MCMF algorithm
    'Edge',
    'MCMF',
    'MCMFMatrix',
    'create_bipartite_mcmf',
    'INF',
    # Network builder
    'NodeType',
    'NetworkNode',
    'NetworkEdge',
    'FlowNetwork',
    'NetworkBuilder',
    'build_routing_network',
    'solve_assignment',
    # River router
    'RouteType',
    'RoutingDirection',
    'RouteSegment',
    'Route',
    'RiverRoutingResult',
    'RiverRouter',
    'river_route',
    'check_planarity',
]

__version__ = '1.0.0'
