"""
Layer Assignment Module for Flip-Chip RDL Routing

This module handles RDL layer assignment for multi-layer routing scenarios,
assigning routes to appropriate redistribution layers while managing
via placement and layer transitions.

For single-layer RDL (the primary focus of this router), this module
provides validation and simple assignment. For multi-layer scenarios,
it implements layer assignment optimization.
"""

from typing import List, Optional, Tuple, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import math

from ..core.geometry import Point, Segment, BoundingBox, manhattan_distance, paths_intersect
from ..data_structures import Chip, IOPad, BumpPad, Net


class LayerType(Enum):
    """Types of RDL layers."""
    SIGNAL = "signal"
    POWER = "power"
    GROUND = "ground"
    MIXED = "mixed"


class AssignmentStrategy(Enum):
    """Layer assignment strategies."""
    SINGLE_LAYER = "single_layer"  # All routes on one layer
    GREEDY = "greedy"  # Assign to first available layer
    LOAD_BALANCED = "load_balanced"  # Balance routes across layers
    CROSSING_BASED = "crossing_based"  # Minimize crossings per layer
    CONGESTION_AWARE = "congestion_aware"  # Consider routing congestion


class AssignmentStatus(Enum):
    """Status of layer assignment."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    INSUFFICIENT_LAYERS = "insufficient_layers"


@dataclass
class Via:
    """Represents a via connecting two layers."""
    x: float
    y: float
    from_layer: int
    to_layer: int
    diameter: float = 50.0
    name: str = ""
    net_id: int = -1
    
    @property
    def position(self) -> Tuple[float, float]:
        """Get via position."""
        return (self.x, self.y)
    
    @property
    def layers(self) -> Tuple[int, int]:
        """Get connected layers (sorted)."""
        return (min(self.from_layer, self.to_layer), 
                max(self.from_layer, self.to_layer))
    
    def overlaps(self, other: 'Via', min_spacing: float = 10.0) -> bool:
        """Check if this via overlaps with another."""
        dist = math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
        min_dist = (self.diameter + other.diameter) / 2 + min_spacing
        return dist < min_dist
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'x': self.x,
            'y': self.y,
            'from_layer': self.from_layer,
            'to_layer': self.to_layer,
            'diameter': self.diameter,
            'name': self.name,
            'net_id': self.net_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Via':
        """Create from dictionary."""
        return cls(
            x=data['x'],
            y=data['y'],
            from_layer=data['from_layer'],
            to_layer=data['to_layer'],
            diameter=data.get('diameter', 50.0),
            name=data.get('name', ''),
            net_id=data.get('net_id', -1)
        )


@dataclass
class LayerRoute:
    """Route segment on a specific layer."""
    net_id: int
    layer: int
    path: List[Point] = field(default_factory=list)
    wire_width: float = 10.0
    
    @property
    def wirelength(self) -> float:
        """Calculate total wirelength."""
        if len(self.path) < 2:
            return 0.0
        total = 0.0
        for i in range(len(self.path) - 1):
            total += manhattan_distance(
                (self.path[i].x, self.path[i].y),
                (self.path[i+1].x, self.path[i+1].y)
            )
        return total
    
    @property
    def bounding_box(self) -> Optional[BoundingBox]:
        """Get bounding box of route."""
        if not self.path:
            return None
        xs = [p.x for p in self.path]
        ys = [p.y for p in self.path]
        return BoundingBox(min(xs), min(ys), max(xs), max(ys))
    
    def intersects(self, other: 'LayerRoute') -> bool:
        """Check if this route intersects another on the same layer."""
        if self.layer != other.layer:
            return False
        if len(self.path) < 2 or len(other.path) < 2:
            return False
        return paths_intersect(self.path, other.path, exclude_endpoints=True)


@dataclass
class LayerInfo:
    """Information about an RDL layer."""
    layer_id: int
    layer_type: LayerType = LayerType.SIGNAL
    wire_width: float = 10.0
    wire_spacing: float = 10.0
    capacity: int = -1  # -1 means unlimited
    assigned_routes: List[int] = field(default_factory=list)  # Net IDs
    
    @property
    def num_routes(self) -> int:
        """Number of routes on this layer."""
        return len(self.assigned_routes)
    
    @property
    def is_full(self) -> bool:
        """Check if layer is at capacity."""
        if self.capacity < 0:
            return False
        return self.num_routes >= self.capacity
    
    @property
    def available_capacity(self) -> int:
        """Get remaining capacity."""
        if self.capacity < 0:
            return float('inf')
        return max(0, self.capacity - self.num_routes)


@dataclass
class LayerAssignmentResult:
    """Result of layer assignment."""
    status: AssignmentStatus
    layer_assignments: Dict[int, int] = field(default_factory=dict)  # net_id -> layer
    vias: List[Via] = field(default_factory=list)
    layer_routes: List[LayerRoute] = field(default_factory=list)
    unassigned_nets: List[int] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_assigned(self) -> int:
        """Number of assigned nets."""
        return len(self.layer_assignments)
    
    @property
    def num_unassigned(self) -> int:
        """Number of unassigned nets."""
        return len(self.unassigned_nets)
    
    @property
    def num_vias(self) -> int:
        """Total number of vias."""
        return len(self.vias)
    
    @property
    def is_complete(self) -> bool:
        """Check if all nets are assigned."""
        return self.num_unassigned == 0
    
    def get_layer_for_net(self, net_id: int) -> Optional[int]:
        """Get assigned layer for a net."""
        return self.layer_assignments.get(net_id)
    
    def get_nets_on_layer(self, layer: int) -> List[int]:
        """Get all nets assigned to a layer."""
        return [net_id for net_id, l in self.layer_assignments.items() if l == layer]
    
    def get_vias_for_net(self, net_id: int) -> List[Via]:
        """Get all vias for a net."""
        return [v for v in self.vias if v.net_id == net_id]


class LayerAssigner:
    """
    Layer assignment algorithm for RDL routing.
    
    For single-layer RDL (primary use case), validates that all routes
    can be placed on a single layer without crossings.
    
    For multi-layer scenarios, assigns routes to layers to minimize
    crossings and balance load.
    """
    
    def __init__(self, 
                 num_layers: int = 1,
                 strategy: AssignmentStrategy = AssignmentStrategy.SINGLE_LAYER,
                 wire_width: float = 10.0,
                 wire_spacing: float = 10.0,
                 via_diameter: float = 50.0):
        """
        Initialize layer assigner.
        
        Args:
            num_layers: Number of available RDL layers
            strategy: Assignment strategy to use
            wire_width: Default wire width
            wire_spacing: Minimum wire spacing
            via_diameter: Via diameter for layer transitions
        """
        self.num_layers = num_layers
        self.strategy = strategy
        self.wire_width = wire_width
        self.wire_spacing = wire_spacing
        self.via_diameter = via_diameter
        
        # Initialize layer info
        self.layers: List[LayerInfo] = []
        for i in range(num_layers):
            self.layers.append(LayerInfo(
                layer_id=i + 1,  # 1-indexed layers
                wire_width=wire_width,
                wire_spacing=wire_spacing
            ))
        
        # Track assignments
        self.assignments: Dict[int, int] = {}  # net_id -> layer
        self.routes: Dict[int, List[Point]] = {}  # net_id -> path
        self.vias: List[Via] = []
    
    def set_layer_type(self, layer: int, layer_type: LayerType) -> None:
        """Set the type of a layer."""
        if 1 <= layer <= self.num_layers:
            self.layers[layer - 1].layer_type = layer_type
    
    def set_layer_capacity(self, layer: int, capacity: int) -> None:
        """Set the capacity of a layer."""
        if 1 <= layer <= self.num_layers:
            self.layers[layer - 1].capacity = capacity
    
    def add_route(self, net_id: int, path: List[Point]) -> None:
        """Add a route to be assigned."""
        self.routes[net_id] = path
    
    def clear(self) -> None:
        """Clear all assignments and routes."""
        self.assignments.clear()
        self.routes.clear()
        self.vias.clear()
        for layer in self.layers:
            layer.assigned_routes.clear()
    
    def assign(self, routes: Optional[Dict[int, List[Point]]] = None) -> LayerAssignmentResult:
        """
        Perform layer assignment.
        
        Args:
            routes: Optional dictionary of net_id -> path to assign
            
        Returns:
            LayerAssignmentResult with assignments and statistics
        """
        if routes:
            self.routes = routes
        
        if not self.routes:
            return LayerAssignmentResult(
                status=AssignmentStatus.SUCCESS,
                statistics={'message': 'No routes to assign'}
            )
        
        # Choose assignment method based on strategy
        if self.strategy == AssignmentStrategy.SINGLE_LAYER:
            return self._assign_single_layer()
        elif self.strategy == AssignmentStrategy.GREEDY:
            return self._assign_greedy()
        elif self.strategy == AssignmentStrategy.LOAD_BALANCED:
            return self._assign_load_balanced()
        elif self.strategy == AssignmentStrategy.CROSSING_BASED:
            return self._assign_crossing_based()
        elif self.strategy == AssignmentStrategy.CONGESTION_AWARE:
            return self._assign_congestion_aware()
        else:
            return self._assign_single_layer()
    
    def _assign_single_layer(self) -> LayerAssignmentResult:
        """
        Assign all routes to a single layer.
        
        This is the primary mode for flip-chip RDL routing where
        river routing ensures no crossings on a single layer.
        """
        layer = 1
        layer_routes = []
        
        for net_id, path in self.routes.items():
            self.assignments[net_id] = layer
            self.layers[0].assigned_routes.append(net_id)
            
            layer_routes.append(LayerRoute(
                net_id=net_id,
                layer=layer,
                path=path,
                wire_width=self.wire_width
            ))
        
        # Check for crossings (should be none if river routing was used)
        crossings = self._count_crossings_on_layer(layer_routes)
        
        status = AssignmentStatus.SUCCESS if crossings == 0 else AssignmentStatus.PARTIAL
        
        return LayerAssignmentResult(
            status=status,
            layer_assignments=self.assignments.copy(),
            vias=[],  # No vias needed for single layer
            layer_routes=layer_routes,
            unassigned_nets=[],
            statistics={
                'num_layers_used': 1,
                'crossings': crossings,
                'total_wirelength': sum(r.wirelength for r in layer_routes),
                'routes_per_layer': {1: len(layer_routes)}
            }
        )
    
    def _assign_greedy(self) -> LayerAssignmentResult:
        """
        Greedy layer assignment - assign to first non-conflicting layer.
        """
        layer_routes: Dict[int, List[LayerRoute]] = {i+1: [] for i in range(self.num_layers)}
        unassigned = []
        
        for net_id, path in self.routes.items():
            assigned = False
            
            for layer_idx in range(self.num_layers):
                layer = layer_idx + 1
                layer_info = self.layers[layer_idx]
                
                if layer_info.is_full:
                    continue
                
                # Check if route conflicts with existing routes on this layer
                new_route = LayerRoute(net_id=net_id, layer=layer, path=path, 
                                       wire_width=self.wire_width)
                
                conflicts = False
                for existing in layer_routes[layer]:
                    if new_route.intersects(existing):
                        conflicts = True
                        break
                
                if not conflicts:
                    self.assignments[net_id] = layer
                    layer_info.assigned_routes.append(net_id)
                    layer_routes[layer].append(new_route)
                    assigned = True
                    break
            
            if not assigned:
                unassigned.append(net_id)
        
        # Flatten layer routes
        all_routes = []
        for routes in layer_routes.values():
            all_routes.extend(routes)
        
        status = AssignmentStatus.SUCCESS if not unassigned else AssignmentStatus.PARTIAL
        if unassigned and len(unassigned) == len(self.routes):
            status = AssignmentStatus.FAILED
        
        return LayerAssignmentResult(
            status=status,
            layer_assignments=self.assignments.copy(),
            vias=self.vias.copy(),
            layer_routes=all_routes,
            unassigned_nets=unassigned,
            statistics={
                'num_layers_used': len([l for l in self.layers if l.num_routes > 0]),
                'routes_per_layer': {l.layer_id: l.num_routes for l in self.layers}
            }
        )
    
    def _assign_load_balanced(self) -> LayerAssignmentResult:
        """
        Load-balanced assignment - distribute routes evenly across layers.
        """
        layer_routes: Dict[int, List[LayerRoute]] = {i+1: [] for i in range(self.num_layers)}
        unassigned = []
        
        # Sort routes by wirelength (longer routes first for better balancing)
        sorted_nets = sorted(
            self.routes.items(),
            key=lambda x: sum(manhattan_distance(
                (x[1][i].x, x[1][i].y),
                (x[1][i+1].x, x[1][i+1].y)
            ) for i in range(len(x[1])-1)) if len(x[1]) > 1 else 0,
            reverse=True
        )
        
        for net_id, path in sorted_nets:
            # Find layer with minimum load that doesn't conflict
            best_layer = None
            min_load = float('inf')
            
            for layer_idx in range(self.num_layers):
                layer = layer_idx + 1
                layer_info = self.layers[layer_idx]
                
                if layer_info.is_full:
                    continue
                
                new_route = LayerRoute(net_id=net_id, layer=layer, path=path,
                                       wire_width=self.wire_width)
                
                conflicts = False
                for existing in layer_routes[layer]:
                    if new_route.intersects(existing):
                        conflicts = True
                        break
                
                if not conflicts and layer_info.num_routes < min_load:
                    min_load = layer_info.num_routes
                    best_layer = layer
            
            if best_layer is not None:
                self.assignments[net_id] = best_layer
                self.layers[best_layer - 1].assigned_routes.append(net_id)
                layer_routes[best_layer].append(
                    LayerRoute(net_id=net_id, layer=best_layer, path=path,
                               wire_width=self.wire_width)
                )
            else:
                unassigned.append(net_id)
        
        all_routes = []
        for routes in layer_routes.values():
            all_routes.extend(routes)
        
        status = AssignmentStatus.SUCCESS if not unassigned else AssignmentStatus.PARTIAL
        
        return LayerAssignmentResult(
            status=status,
            layer_assignments=self.assignments.copy(),
            vias=self.vias.copy(),
            layer_routes=all_routes,
            unassigned_nets=unassigned,
            statistics={
                'num_layers_used': len([l for l in self.layers if l.num_routes > 0]),
                'routes_per_layer': {l.layer_id: l.num_routes for l in self.layers},
                'load_balance': self._calculate_load_balance()
            }
        )
    
    def _assign_crossing_based(self) -> LayerAssignmentResult:
        """
        Crossing-based assignment - minimize crossings on each layer.
        """
        # For single layer, this is same as single_layer
        if self.num_layers == 1:
            return self._assign_single_layer()
        
        # Build crossing graph
        crossing_pairs = self._find_crossing_pairs()
        
        # Use graph coloring approach
        layer_routes: Dict[int, List[LayerRoute]] = {i+1: [] for i in range(self.num_layers)}
        unassigned = []
        
        # Sort by number of crossings (most crossings first)
        crossing_count = {}
        for net_id in self.routes:
            crossing_count[net_id] = sum(1 for p in crossing_pairs if net_id in p)
        
        sorted_nets = sorted(self.routes.keys(), key=lambda x: crossing_count[x], reverse=True)
        
        for net_id in sorted_nets:
            path = self.routes[net_id]
            best_layer = None
            min_crossings = float('inf')
            
            for layer_idx in range(self.num_layers):
                layer = layer_idx + 1
                layer_info = self.layers[layer_idx]
                
                if layer_info.is_full:
                    continue
                
                # Count crossings with routes already on this layer
                crossings = 0
                for existing_net in layer_info.assigned_routes:
                    if (net_id, existing_net) in crossing_pairs or \
                       (existing_net, net_id) in crossing_pairs:
                        crossings += 1
                
                if crossings < min_crossings:
                    min_crossings = crossings
                    best_layer = layer
            
            if best_layer is not None:
                self.assignments[net_id] = best_layer
                self.layers[best_layer - 1].assigned_routes.append(net_id)
                layer_routes[best_layer].append(
                    LayerRoute(net_id=net_id, layer=best_layer, path=path,
                               wire_width=self.wire_width)
                )
            else:
                unassigned.append(net_id)
        
        all_routes = []
        for routes in layer_routes.values():
            all_routes.extend(routes)
        
        # Count total crossings
        total_crossings = sum(
            self._count_crossings_on_layer(layer_routes[l])
            for l in layer_routes
        )
        
        status = AssignmentStatus.SUCCESS if not unassigned else AssignmentStatus.PARTIAL
        
        return LayerAssignmentResult(
            status=status,
            layer_assignments=self.assignments.copy(),
            vias=self.vias.copy(),
            layer_routes=all_routes,
            unassigned_nets=unassigned,
            statistics={
                'num_layers_used': len([l for l in self.layers if l.num_routes > 0]),
                'total_crossings': total_crossings,
                'routes_per_layer': {l.layer_id: l.num_routes for l in self.layers}
            }
        )
    
    def _assign_congestion_aware(self) -> LayerAssignmentResult:
        """
        Congestion-aware assignment - consider routing density.
        """
        # For now, use load-balanced as base
        return self._assign_load_balanced()
    
    def _find_crossing_pairs(self) -> Set[Tuple[int, int]]:
        """Find all pairs of routes that would cross."""
        crossing_pairs = set()
        net_ids = list(self.routes.keys())
        
        for i in range(len(net_ids)):
            for j in range(i + 1, len(net_ids)):
                net_i, net_j = net_ids[i], net_ids[j]
                path_i, path_j = self.routes[net_i], self.routes[net_j]
                
                if paths_intersect(path_i, path_j, exclude_endpoints=True):
                    crossing_pairs.add((net_i, net_j))
        
        return crossing_pairs
    
    def _count_crossings_on_layer(self, routes: List[LayerRoute]) -> int:
        """Count number of crossings among routes on a layer."""
        crossings = 0
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                if routes[i].intersects(routes[j]):
                    crossings += 1
        return crossings
    
    def _calculate_load_balance(self) -> float:
        """Calculate load balance metric (0 = perfect, higher = worse)."""
        if self.num_layers <= 1:
            return 0.0
        
        loads = [l.num_routes for l in self.layers]
        if not any(loads):
            return 0.0
        
        avg_load = sum(loads) / len(loads)
        if avg_load == 0:
            return 0.0
        
        variance = sum((l - avg_load) ** 2 for l in loads) / len(loads)
        return math.sqrt(variance) / avg_load
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get assignment statistics."""
        return {
            'num_layers': self.num_layers,
            'num_routes': len(self.routes),
            'num_assigned': len(self.assignments),
            'strategy': self.strategy.value,
            'layers_used': len([l for l in self.layers if l.num_routes > 0]),
            'routes_per_layer': {l.layer_id: l.num_routes for l in self.layers},
            'load_balance': self._calculate_load_balance()
        }


def assign_layers(routes: Dict[int, List[Point]], 
                  num_layers: int = 1,
                  strategy: AssignmentStrategy = AssignmentStrategy.SINGLE_LAYER) -> LayerAssignmentResult:
    """
    Convenience function for layer assignment.
    
    Args:
        routes: Dictionary mapping net_id to path (list of Points)
        num_layers: Number of available layers
        strategy: Assignment strategy
        
    Returns:
        LayerAssignmentResult with assignments
    """
    assigner = LayerAssigner(num_layers=num_layers, strategy=strategy)
    return assigner.assign(routes)


def assign_chip_layers(chip: Chip, 
                       strategy: AssignmentStrategy = AssignmentStrategy.SINGLE_LAYER) -> LayerAssignmentResult:
    """
    Assign layers for a chip's routed nets.
    
    Args:
        chip: Chip with routed nets
        strategy: Assignment strategy
        
    Returns:
        LayerAssignmentResult with assignments
    """
    # Extract routes from chip's routed nets
    routes = {}
    for net in chip.routed_nets:
        if net.route_path:
            # Convert to Point objects if needed
            path = []
            for p in net.route_path:
                if isinstance(p, Point):
                    path.append(p)
                elif isinstance(p, (tuple, list)) and len(p) >= 2:
                    path.append(Point(p[0], p[1]))
            routes[net.net_id] = path
    
    assigner = LayerAssigner(
        num_layers=chip.rdl_layers,
        strategy=strategy,
        wire_width=chip.wire_width,
        wire_spacing=chip.wire_spacing
    )
    
    return assigner.assign(routes)


def validate_single_layer_routing(routes: List[List[Point]]) -> Tuple[bool, int]:
    """
    Validate that routes can be placed on a single layer without crossings.
    
    Args:
        routes: List of paths (each path is a list of Points)
        
    Returns:
        Tuple of (is_valid, num_crossings)
    """
    crossings = 0
    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            if paths_intersect(routes[i], routes[j], exclude_endpoints=True):
                crossings += 1
    
    return crossings == 0, crossings


# Test function
def test_layer_assignment():
    """Test layer assignment functionality."""
    print("Testing Layer Assignment...")
    
    # Create test routes
    routes = {
        1: [Point(0, 0), Point(100, 0), Point(100, 100)],
        2: [Point(0, 50), Point(50, 50), Point(50, 150)],
        3: [Point(0, 100), Point(150, 100), Point(150, 200)],
    }
    
    # Test single layer assignment
    print("\n1. Single Layer Assignment:")
    result = assign_layers(routes, num_layers=1, strategy=AssignmentStrategy.SINGLE_LAYER)
    print(f"   Status: {result.status.value}")
    print(f"   Assigned: {result.num_assigned}")
    print(f"   Crossings: {result.statistics.get('crossings', 0)}")
    
    # Test greedy assignment with multiple layers
    print("\n2. Greedy Multi-Layer Assignment:")
    result = assign_layers(routes, num_layers=2, strategy=AssignmentStrategy.GREEDY)
    print(f"   Status: {result.status.value}")
    print(f"   Routes per layer: {result.statistics.get('routes_per_layer', {})}")
    
    # Test load-balanced assignment
    print("\n3. Load-Balanced Assignment:")
    result = assign_layers(routes, num_layers=2, strategy=AssignmentStrategy.LOAD_BALANCED)
    print(f"   Status: {result.status.value}")
    print(f"   Routes per layer: {result.statistics.get('routes_per_layer', {})}")
    print(f"   Load balance: {result.statistics.get('load_balance', 0):.3f}")
    
    # Test crossing-based assignment
    print("\n4. Crossing-Based Assignment:")
    result = assign_layers(routes, num_layers=2, strategy=AssignmentStrategy.CROSSING_BASED)
    print(f"   Status: {result.status.value}")
    print(f"   Total crossings: {result.statistics.get('total_crossings', 0)}")
    
    # Test validation
    print("\n5. Single Layer Validation:")
    route_list = list(routes.values())
    is_valid, num_crossings = validate_single_layer_routing(route_list)
    print(f"   Valid: {is_valid}")
    print(f"   Crossings: {num_crossings}")
    
    print("\nLayer Assignment tests completed!")


if __name__ == "__main__":
    test_layer_assignment()
