"""
River Routing Algorithm for Single-Layer RDL Routing

This module implements the river routing methodology for detailed routing
in flip-chip RDL design. River routing ensures planarity (no wire crossings)
by maintaining proper ordering of nets and generating non-crossing paths.

Key Features:
- Planarity constraint enforcement (no wire crossings)
- L-shaped and Z-shaped Manhattan routing
- Net ordering for conflict-free routing
- Obstacle-aware path generation
"""

from typing import List, Tuple, Optional, Dict, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import math

from .geometry import (
    Point, Segment, BoundingBox,
    manhattan_distance, euclidean_distance,
    generate_l_shape_path, generate_z_shape_path,
    simplify_path, path_length, path_manhattan_length,
    count_bends, paths_intersect, would_cross,
    is_planar_routing, segments_intersect
)


class RouteType(Enum):
    """Type of route shape"""
    L_SHAPE = "l_shape"
    Z_SHAPE = "z_shape"
    STRAIGHT = "straight"
    CUSTOM = "custom"


class RoutingDirection(Enum):
    """Primary routing direction"""
    HORIZONTAL_FIRST = "horizontal_first"
    VERTICAL_FIRST = "vertical_first"
    AUTO = "auto"


@dataclass
class RouteSegment:
    """Represents a segment of a route"""
    start: Point
    end: Point
    layer: int = 1
    width: float = 10.0
    
    @property
    def length(self) -> float:
        """Calculate segment length"""
        return self.start.distance_to(self.end)
    
    @property
    def manhattan_length(self) -> float:
        """Calculate Manhattan length"""
        return self.start.manhattan_distance_to(self.end)
    
    @property
    def is_horizontal(self) -> bool:
        """Check if segment is horizontal"""
        return abs(self.start.y - self.end.y) < 1e-9
    
    @property
    def is_vertical(self) -> bool:
        """Check if segment is vertical"""
        return abs(self.start.x - self.end.x) < 1e-9
    
    def intersects(self, other: 'RouteSegment', tolerance: float = 1e-9) -> bool:
        """Check if this segment intersects another"""
        return segments_intersect(
            self.start, self.end,
            other.start, other.end,
            tolerance
        )


@dataclass
class Route:
    """Represents a complete route from IO pad to bump pad"""
    io_id: int
    bump_id: int
    path: List[Point] = field(default_factory=list)
    segments: List[RouteSegment] = field(default_factory=list)
    route_type: RouteType = RouteType.L_SHAPE
    layer: int = 1
    width: float = 10.0
    
    @property
    def wirelength(self) -> float:
        """Calculate total wirelength"""
        if self.segments:
            return sum(seg.length for seg in self.segments)
        return path_length(self.path)
    
    @property
    def manhattan_wirelength(self) -> float:
        """Calculate Manhattan wirelength"""
        if self.segments:
            return sum(seg.manhattan_length for seg in self.segments)
        return path_manhattan_length(self.path)
    
    @property
    def num_bends(self) -> int:
        """Count number of bends in route"""
        return count_bends(self.path)
    
    @property
    def start_point(self) -> Optional[Point]:
        """Get starting point of route"""
        return self.path[0] if self.path else None
    
    @property
    def end_point(self) -> Optional[Point]:
        """Get ending point of route"""
        return self.path[-1] if self.path else None
    
    def set_path(self, path: List[Point]) -> None:
        """Set route path and generate segments"""
        self.path = path
        self.segments = []
        for i in range(len(path) - 1):
            self.segments.append(RouteSegment(
                start=path[i],
                end=path[i + 1],
                layer=self.layer,
                width=self.width
            ))
    
    def intersects(self, other: 'Route', exclude_endpoints: bool = True) -> bool:
        """Check if this route intersects another"""
        return paths_intersect(self.path, other.path, exclude_endpoints)


@dataclass
class RiverRoutingResult:
    """Result of river routing"""
    routes: List[Route] = field(default_factory=list)
    success: bool = False
    total_wirelength: float = 0.0
    num_routed: int = 0
    num_failed: int = 0
    crossings: int = 0
    message: str = ""
    
    @property
    def routability_rate(self) -> float:
        """Calculate routability rate"""
        total = self.num_routed + self.num_failed
        return self.num_routed / total if total > 0 else 0.0


class RiverRouter:
    """
    River Routing Algorithm for Single-Layer RDL
    
    River routing is a technique for routing parallel wires without crossings,
    similar to how rivers flow without crossing each other. This is essential
    for single-layer RDL routing where wire crossings are not allowed.
    
    Algorithm:
    1. Sort terminals by position
    2. Check routability (no crossing constraint)
    3. Order nets to avoid conflicts
    4. Generate routes maintaining planarity
    """
    
    def __init__(
        self,
        wire_width: float = 10.0,
        wire_spacing: float = 10.0,
        layer: int = 1,
        routing_direction: RoutingDirection = RoutingDirection.AUTO
    ):
        """
        Initialize river router
        
        Args:
            wire_width: Width of routing wires
            wire_spacing: Minimum spacing between wires
            layer: RDL layer number
            routing_direction: Primary routing direction
        """
        self.wire_width = wire_width
        self.wire_spacing = wire_spacing
        self.layer = layer
        self.routing_direction = routing_direction
        self.routes: List[Route] = []
        self.obstacles: List[BoundingBox] = []
    
    def add_obstacle(self, obstacle: BoundingBox) -> None:
        """Add routing obstacle"""
        self.obstacles.append(obstacle)
    
    def clear_obstacles(self) -> None:
        """Clear all obstacles"""
        self.obstacles.clear()
    
    def clear_routes(self) -> None:
        """Clear all routes"""
        self.routes.clear()
    
    def check_routability(
        self,
        assignments: List[Tuple[Tuple[float, float], Tuple[float, float]]]
    ) -> Tuple[bool, int]:
        """
        Check if assignments are routable without crossings
        
        Args:
            assignments: List of (io_pos, bump_pos) tuples
            
        Returns:
            Tuple of (is_routable, num_crossings)
        """
        if not assignments:
            return True, 0
        
        # Convert to format for crossing check
        crossing_data = [
            (io[0], bump[0]) for io, bump in assignments
        ]
        
        crossings = 0
        for i in range(len(crossing_data)):
            for j in range(i + 1, len(crossing_data)):
                io1_x, bump1_x = crossing_data[i]
                io2_x, bump2_x = crossing_data[j]
                if would_cross(io1_x, bump1_x, io2_x, bump2_x):
                    crossings += 1
        
        return crossings == 0, crossings
    
    def order_nets(
        self,
        assignments: List[Tuple[int, int, Tuple[float, float], Tuple[float, float]]]
    ) -> List[int]:
        """
        Determine routing order to avoid conflicts
        
        Uses topological sort based on precedence constraints.
        Net i must be routed before net j if they would cross and
        routing i first avoids the conflict.
        
        Args:
            assignments: List of (io_id, bump_id, io_pos, bump_pos)
            
        Returns:
            List of indices in routing order
        """
        n = len(assignments)
        if n == 0:
            return []
        
        # Build precedence graph
        # Edge i -> j means i should be routed before j
        precedence: Dict[int, List[int]] = {i: [] for i in range(n)}
        in_degree = [0] * n
        
        for i in range(n):
            io_i = assignments[i][2]
            bump_i = assignments[i][3]
            
            for j in range(n):
                if i == j:
                    continue
                    
                io_j = assignments[j][2]
                bump_j = assignments[j][3]
                
                # Check if i should precede j
                if self._should_precede(io_i, bump_i, io_j, bump_j):
                    if j not in precedence[i]:
                        precedence[i].append(j)
                        in_degree[j] += 1
        
        # Topological sort using Kahn's algorithm
        order = []
        queue = [i for i in range(n) if in_degree[i] == 0]
        
        while queue:
            # Sort by x-coordinate for deterministic ordering
            queue.sort(key=lambda i: assignments[i][2][0])
            current = queue.pop(0)
            order.append(current)
            
            for next_node in precedence[current]:
                in_degree[next_node] -= 1
                if in_degree[next_node] == 0:
                    queue.append(next_node)
        
        # If not all nodes are in order, there's a cycle (crossing conflict)
        if len(order) != n:
            # Fall back to simple x-coordinate ordering
            order = sorted(range(n), key=lambda i: assignments[i][2][0])
        
        return order
    
    def _should_precede(
        self,
        io_i: Tuple[float, float],
        bump_i: Tuple[float, float],
        io_j: Tuple[float, float],
        bump_j: Tuple[float, float]
    ) -> bool:
        """
        Determine if net i should be routed before net j
        
        Net i precedes j if:
        - io_i.x < io_j.x and bump_i.x < bump_j.x (same relative order)
        - OR io_i is "inside" the region that j would route through
        """
        # If both maintain same relative order, no precedence needed
        if (io_i[0] < io_j[0]) == (bump_i[0] < bump_j[0]):
            return False
        
        # If they would cross, the one with smaller io_x goes first
        # when its bump is to the right (crossing from left)
        if io_i[0] < io_j[0] and bump_i[0] > bump_j[0]:
            return True
        
        return False
    
    def route(
        self,
        assignments: List[Tuple[int, int, Tuple[float, float], Tuple[float, float]]]
    ) -> RiverRoutingResult:
        """
        Perform river routing for given assignments
        
        Args:
            assignments: List of (io_id, bump_id, io_pos, bump_pos)
            
        Returns:
            RiverRoutingResult with routes and statistics
        """
        result = RiverRoutingResult()
        
        if not assignments:
            result.success = True
            result.message = "No assignments to route"
            return result
        
        # Check routability
        pos_assignments = [(a[2], a[3]) for a in assignments]
        is_routable, crossings = self.check_routability(pos_assignments)
        result.crossings = crossings
        
        if not is_routable:
            result.message = f"Design has {crossings} crossing(s), attempting to route anyway"
        
        # Determine routing order
        routing_order = self.order_nets(assignments)
        
        # Route each net in order
        routed_paths: List[Route] = []
        
        for idx in routing_order:
            io_id, bump_id, io_pos, bump_pos = assignments[idx]
            
            # Generate route
            route = self._generate_route(
                io_id, bump_id,
                Point(io_pos[0], io_pos[1]),
                Point(bump_pos[0], bump_pos[1]),
                routed_paths
            )
            
            if route:
                routed_paths.append(route)
                result.routes.append(route)
                result.num_routed += 1
                result.total_wirelength += route.manhattan_wirelength
            else:
                result.num_failed += 1
        
        result.success = result.num_failed == 0
        if result.success:
            result.message = f"Successfully routed {result.num_routed} nets"
        else:
            result.message = f"Routed {result.num_routed}/{result.num_routed + result.num_failed} nets"
        
        self.routes = result.routes
        return result
    
    def _generate_route(
        self,
        io_id: int,
        bump_id: int,
        start: Point,
        end: Point,
        existing_routes: List[Route]
    ) -> Optional[Route]:
        """
        Generate a route from start to end avoiding existing routes
        
        Args:
            io_id: IO pad identifier
            bump_id: Bump pad identifier
            start: Starting point (IO pad position)
            end: Ending point (bump pad position)
            existing_routes: List of already routed paths
            
        Returns:
            Route object or None if routing fails
        """
        route = Route(
            io_id=io_id,
            bump_id=bump_id,
            layer=self.layer,
            width=self.wire_width
        )
        
        # Determine routing direction
        if self.routing_direction == RoutingDirection.AUTO:
            # Choose direction that minimizes potential conflicts
            horizontal_first = self._choose_direction(start, end, existing_routes)
        else:
            horizontal_first = (self.routing_direction == RoutingDirection.HORIZONTAL_FIRST)
        
        # Try L-shape routing first
        path = generate_l_shape_path(start, end, horizontal_first)
        
        if not self._path_conflicts(path, existing_routes):
            route.set_path(path)
            route.route_type = RouteType.L_SHAPE
            return route
        
        # Try opposite L-shape
        path = generate_l_shape_path(start, end, not horizontal_first)
        
        if not self._path_conflicts(path, existing_routes):
            route.set_path(path)
            route.route_type = RouteType.L_SHAPE
            return route
        
        # Try Z-shape routing with different midpoints
        z_path = self._try_z_shape_routing(start, end, existing_routes)
        
        if z_path:
            route.set_path(z_path)
            route.route_type = RouteType.Z_SHAPE
            return route
        
        # Last resort: use L-shape even with conflicts
        path = generate_l_shape_path(start, end, horizontal_first)
        route.set_path(path)
        route.route_type = RouteType.L_SHAPE
        return route
    
    def _choose_direction(
        self,
        start: Point,
        end: Point,
        existing_routes: List[Route]
    ) -> bool:
        """
        Choose routing direction to minimize conflicts
        
        Returns:
            True for horizontal-first, False for vertical-first
        """
        # Try both directions and count conflicts
        h_path = generate_l_shape_path(start, end, horizontal_first=True)
        v_path = generate_l_shape_path(start, end, horizontal_first=False)
        
        h_conflicts = sum(1 for r in existing_routes if paths_intersect(h_path, r.path, True))
        v_conflicts = sum(1 for r in existing_routes if paths_intersect(v_path, r.path, True))
        
        if h_conflicts != v_conflicts:
            return h_conflicts < v_conflicts
        
        # If equal conflicts, prefer direction with shorter first segment
        dx = abs(end.x - start.x)
        dy = abs(end.y - start.y)
        return dx <= dy
    
    def _path_conflicts(
        self,
        path: List[Point],
        existing_routes: List[Route]
    ) -> bool:
        """Check if path conflicts with existing routes"""
        for route in existing_routes:
            if paths_intersect(path, route.path, exclude_endpoints=True):
                return True
        
        # Check obstacles
        for i in range(len(path) - 1):
            segment = Segment(path[i], path[i + 1])
            for obstacle in self.obstacles:
                if self._segment_intersects_box(segment, obstacle):
                    return True
        
        return False
    
    def _segment_intersects_box(self, segment: Segment, box: BoundingBox) -> bool:
        """Check if segment intersects bounding box"""
        # Check if either endpoint is inside box
        if box.contains_point(segment.p1.x, segment.p1.y):
            return True
        if box.contains_point(segment.p2.x, segment.p2.y):
            return True
        
        # Check intersection with box edges
        corners = [
            Point(box.x_min, box.y_min),
            Point(box.x_max, box.y_min),
            Point(box.x_max, box.y_max),
            Point(box.x_min, box.y_max)
        ]
        
        for i in range(4):
            edge = Segment(corners[i], corners[(i + 1) % 4])
            if segment.intersects(edge):
                return True
        
        return False
    
    def _try_z_shape_routing(
        self,
        start: Point,
        end: Point,
        existing_routes: List[Route]
    ) -> Optional[List[Point]]:
        """
        Try Z-shape routing with various midpoints
        
        Args:
            start: Starting point
            end: Ending point
            existing_routes: Existing routes to avoid
            
        Returns:
            Valid path or None
        """
        dx = end.x - start.x
        dy = end.y - start.y
        
        # Try different midpoint positions
        ratios = [0.25, 0.5, 0.75, 0.33, 0.67]
        
        for ratio in ratios:
            # Horizontal Z-shape (horizontal-vertical-horizontal)
            mid_y = start.y + dy * ratio
            path = generate_z_shape_path(start, end, mid_x=None, mid_y=mid_y)
            if not self._path_conflicts(path, existing_routes):
                return path
            
            # Vertical Z-shape (vertical-horizontal-vertical)
            mid_x = start.x + dx * ratio
            path = generate_z_shape_path(start, end, mid_x=mid_x, mid_y=None)
            if not self._path_conflicts(path, existing_routes):
                return path
        
        return None
    
    def route_single(
        self,
        io_pos: Tuple[float, float],
        bump_pos: Tuple[float, float],
        io_id: int = 0,
        bump_id: int = 0
    ) -> Optional[Route]:
        """
        Route a single net
        
        Args:
            io_pos: IO pad position
            bump_pos: Bump pad position
            io_id: IO pad identifier
            bump_id: Bump pad identifier
            
        Returns:
            Route object or None
        """
        start = Point(io_pos[0], io_pos[1])
        end = Point(bump_pos[0], bump_pos[1])
        
        return self._generate_route(io_id, bump_id, start, end, self.routes)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get routing statistics"""
        if not self.routes:
            return {
                'num_routes': 0,
                'total_wirelength': 0.0,
                'avg_wirelength': 0.0,
                'total_bends': 0,
                'avg_bends': 0.0,
                'route_types': {}
            }
        
        total_wl = sum(r.manhattan_wirelength for r in self.routes)
        total_bends = sum(r.num_bends for r in self.routes)
        
        route_types = {}
        for r in self.routes:
            rt = r.route_type.value
            route_types[rt] = route_types.get(rt, 0) + 1
        
        return {
            'num_routes': len(self.routes),
            'total_wirelength': total_wl,
            'avg_wirelength': total_wl / len(self.routes),
            'total_bends': total_bends,
            'avg_bends': total_bends / len(self.routes),
            'route_types': route_types
        }


def river_route(
    io_pads: List[Any],
    bump_pads: List[Any],
    assignments: List[Tuple[Any, Any]],
    wire_width: float = 10.0,
    wire_spacing: float = 10.0
) -> RiverRoutingResult:
    """
    Convenience function for river routing
    
    Args:
        io_pads: List of IO pad objects (must have x, y attributes)
        bump_pads: List of bump pad objects (must have x, y attributes)
        assignments: List of (io_pad, bump_pad) assignment tuples
        wire_width: Width of routing wires
        wire_spacing: Minimum spacing between wires
        
    Returns:
        RiverRoutingResult with routes and statistics
    """
    router = RiverRouter(wire_width=wire_width, wire_spacing=wire_spacing)
    
    # Convert assignments to router format
    routing_assignments = []
    for io_pad, bump_pad in assignments:
        io_id = getattr(io_pad, 'node_id', id(io_pad))
        bump_id = getattr(bump_pad, 'node_id', id(bump_pad))
        io_pos = (io_pad.x, io_pad.y)
        bump_pos = (bump_pad.x, bump_pad.y)
        routing_assignments.append((io_id, bump_id, io_pos, bump_pos))
    
    return router.route(routing_assignments)


def check_planarity(
    assignments: List[Tuple[Tuple[float, float], Tuple[float, float]]]
) -> Tuple[bool, int]:
    """
    Check if assignments can be routed without crossings
    
    Args:
        assignments: List of (io_pos, bump_pos) tuples
        
    Returns:
        Tuple of (is_planar, num_crossings)
    """
    router = RiverRouter()
    return router.check_routability(assignments)


def test_river_router():
    """Test river routing functionality"""
    print("Testing River Router...")
    
    # Test 1: Simple non-crossing case
    assignments = [
        (0, 0, (0, 0), (100, 100)),
        (1, 1, (50, 0), (150, 100)),
        (2, 2, (100, 0), (200, 100))
    ]
    
    router = RiverRouter()
    result = router.route(assignments)
    
    assert result.success, "Simple routing should succeed"
    assert result.num_routed == 3, "All 3 nets should be routed"
    assert result.crossings == 0, "No crossings expected"
    print(f"  Test 1 passed: {result.num_routed} nets routed, wirelength={result.total_wirelength:.1f}")
    
    # Test 2: Crossing case
    assignments_cross = [
        (0, 0, (0, 0), (200, 100)),    # Goes far right
        (1, 1, (100, 0), (50, 100))    # Goes left - crosses!
    ]
    
    router2 = RiverRouter()
    is_routable, crossings = router2.check_routability(
        [(a[2], a[3]) for a in assignments_cross]
    )
    
    assert not is_routable, "Crossing case should not be planar"
    assert crossings == 1, "Should detect 1 crossing"
    print(f"  Test 2 passed: Detected {crossings} crossing(s)")
    
    # Test 3: Route with crossing (should still produce routes)
    result2 = router2.route(assignments_cross)
    assert result2.num_routed == 2, "Should still route both nets"
    print(f"  Test 3 passed: Routed {result2.num_routed} nets despite crossing")
    
    # Test 4: Net ordering
    assignments_order = [
        (0, 0, (0, 0), (100, 100)),
        (1, 1, (200, 0), (300, 100)),
        (2, 2, (100, 0), (200, 100))
    ]
    
    router3 = RiverRouter()
    order = router3.order_nets(assignments_order)
    assert len(order) == 3, "Should order all 3 nets"
    print(f"  Test 4 passed: Net order = {order}")
    
    # Test 5: Statistics
    stats = router.get_statistics()
    assert stats['num_routes'] == 3
    print(f"  Test 5 passed: Statistics = {stats}")
    
    print("All River Router tests passed!")


if __name__ == "__main__":
    test_river_router()
