"""
Routing module for flip-chip RDL routing.

This module provides the routing algorithms and utilities for flip-chip
redistribution layer (RDL) routing, including:
- Global routing using minimum-cost maximum-flow (MCMF)
- Detailed routing using river routing methodology
- Net ordering for planar routing
- Layer assignment for multi-layer RDL

The routing flow is:
1. Global routing: Assign IO pads to bump pads (MCMF optimization)
2. Net ordering: Determine routing sequence for planarity
3. Detailed routing: Generate actual wire paths (river routing)
4. Layer assignment: Assign routes to RDL layers (optional for multi-layer)
"""

from .global_router import (
    GlobalRouter,
    GlobalRoutingResult,
    Assignment,
    RoutingConstraints,
    RoutingStatus,
    global_route,
    compute_lower_bound_wirelength,
)

from .detailed_router import (
    DetailedRouter,
    DetailedRoutingResult,
    DetailedRoute,
    DetailedRoutingStatus,
    DRCChecker,
    DRCViolation,
    DRCViolationType,
    detailed_route,
    route_with_global,
)

from .net_ordering import (
    NetOrderer,
    OrderingResult,
    OrderingStrategy,
    OrderingStatus,
    NetAssignment,
    PrecedenceGraph,
    order_nets,
    must_route_before,
    find_optimal_ordering,
)

from .layer_assignment import (
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
)

from ..core.geometry import BoundingBox

__all__ = [
    # Global routing
    'GlobalRouter',
    'GlobalRoutingResult',
    'Assignment',
    'RoutingConstraints',
    'RoutingStatus',
    'global_route',
    'compute_lower_bound_wirelength',
    
    # Detailed routing
    'DetailedRouter',
    'DetailedRoutingResult',
    'DetailedRoute',
    'DetailedRoutingStatus',
    'DRCChecker',
    'DRCViolation',
    'DRCViolationType',
    'detailed_route',
    'route_with_global',
    
    # Net ordering
    'NetOrderer',
    'OrderingResult',
    'OrderingStrategy',
    'OrderingStatus',
    'NetAssignment',
    'PrecedenceGraph',
    'order_nets',
    'must_route_before',
    'find_optimal_ordering',
    
    # Layer assignment
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
]

__version__ = '1.0.0'


def full_routing_pipeline(chip, constraints=None, wire_width=10.0, wire_spacing=10.0,
                          enable_drc=True, num_layers=1, layer_strategy=None):
    """
    Execute the complete routing pipeline on a chip.
    
    This is a convenience function that runs all routing stages:
    1. Global routing (MCMF-based IO-to-bump assignment)
    2. Detailed routing (river routing for wire paths)
    3. Layer assignment (for multi-layer RDL)
    
    Args:
        chip: Chip object with IO pads and bump pads
        constraints: Optional RoutingConstraints for global routing
        wire_width: Wire width in microns (default: 10.0)
        wire_spacing: Wire spacing in microns (default: 10.0)
        enable_drc: Enable DRC checking (default: True)
        num_layers: Number of RDL layers (default: 1)
        layer_strategy: Layer assignment strategy (default: None for auto)
    
    Returns:
        dict: Dictionary containing:
            - 'global_result': GlobalRoutingResult
            - 'detailed_result': DetailedRoutingResult
            - 'layer_result': LayerAssignmentResult (if num_layers > 1)
            - 'success': bool indicating overall success
            - 'statistics': dict with routing statistics
    """
    from .global_router import GlobalRouter, RoutingStatus
    from .detailed_router import DetailedRouter, DetailedRoutingStatus
    from .layer_assignment import LayerAssigner, AssignmentStrategy, AssignmentStatus
    
    result = {
        'global_result': None,
        'detailed_result': None,
        'layer_result': None,
        'success': False,
        'statistics': {}
    }
    
    # Stage 1: Global routing
    global_router = GlobalRouter(constraints=constraints)
    global_result = global_router.route(chip)
    result['global_routing'] = global_result
    
    if global_result.status == RoutingStatus.FAILED:
        result['statistics']['failure_stage'] = 'global_routing'
        return result
    
    # Stage 2: Detailed routing
    detailed_router = DetailedRouter(
        wire_width=wire_width,
        wire_spacing=wire_spacing,
        enable_drc=enable_drc
    )
    
    # Set boundary from chip dimensions
    detailed_router.set_boundary(BoundingBox(0, 0, chip.die_width, chip.die_height))
    
    detailed_result = detailed_router.route(chip)
    result['detailed_routing'] = detailed_result
    
    if detailed_result.status == DetailedRoutingStatus.FAILED:
        result['statistics']['failure_stage'] = 'detailed_routing'
        return result
    
    # Stage 3: Layer assignment (if multi-layer)
    if num_layers > 1:
        strategy = layer_strategy or AssignmentStrategy.LOAD_BALANCED
        layer_assigner = LayerAssigner(
            num_layers=num_layers,
            strategy=strategy,
            wire_width=wire_width,
            wire_spacing=wire_spacing
        )
        
        # Convert detailed routes to format expected by layer assigner
        routes_for_assignment = []
        for route in detailed_result.routes:
            routes_for_assignment.append({
                'net_id': route.net_id,
                'path': route.path,
                'wire_width': route.wire_width
            })
        
        layer_result = layer_assigner.assign(routes_for_assignment)
        result['layer_assignment'] = layer_result
        
        if layer_result.status == AssignmentStatus.FAILED:
            result['statistics']['failure_stage'] = 'layer_assignment'
            return result
    
    # Compute overall statistics
    result['success'] = True
    result['statistics'] = {
        'success': True,
        'num_io_pads': chip.num_io_pads,
        'num_bump_pads': chip.num_bump_pads,
        'num_nets_routed': detailed_result.num_routed,
        'num_nets_failed': detailed_result.num_failed,
        'total_wirelength': detailed_result.total_wirelength,
        'routability_rate': detailed_result.routability_rate,
        'num_drc_violations': len(detailed_result.drc_violations),
        'global_routing_cost': global_result.total_cost,
        'num_layers_used': num_layers
    }
    
    return result


def quick_route(chip, wire_width=10.0, wire_spacing=10.0):
    """
    Quick routing function for simple use cases.
    
    Performs global and detailed routing with default settings.
    
    Args:
        chip: Chip object with IO pads and bump pads
        wire_width: Wire width in microns
        wire_spacing: Wire spacing in microns
    
    Returns:
        tuple: (success: bool, total_wirelength: float, num_routed: int)
    """
    result = full_routing_pipeline(
        chip,
        wire_width=wire_width,
        wire_spacing=wire_spacing,
        enable_drc=False,
        num_layers=1
    )
    
    if result['success'] and result['detailed_result']:
        return (
            True,
            result['detailed_result'].total_wirelength,
            result['detailed_result'].num_routed
        )
    else:
        return (False, 0.0, 0)
