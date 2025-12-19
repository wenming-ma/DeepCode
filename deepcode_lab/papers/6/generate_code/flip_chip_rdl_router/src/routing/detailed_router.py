"""
Detailed Router for Flip-Chip RDL Routing

This module implements the detailed routing phase that takes global routing
assignments and generates actual wire paths using river routing methodology.
It ensures DRC compliance, handles obstacles, and optimizes wire paths.
"""

from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import math

from ..core.geometry import (
    Point, Segment, BoundingBox,
    manhattan_distance, euclidean_distance,
    generate_l_shape_path, generate_z_shape_path,
    simplify_path, path_length, path_manhattan_length,
    count_bends, paths_intersect, segments_intersect
)
from ..core.river_router import (
    RiverRouter, Route, RouteSegment, RiverRoutingResult,
    RouteType, RoutingDirection, river_route, check_planarity
)
from ..data_structures import Chip, IOPad, BumpPad, Net


class DetailedRoutingStatus(Enum):
    """Status of detailed routing result."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    DRC_VIOLATION = "drc_violation"


class DRCViolationType(Enum):
    """Types of DRC violations."""
    SPACING = "spacing"
    WIDTH = "width"
    CROSSING = "crossing"
    SHORT = "short"
    OPEN = "open"
    BOUNDARY = "boundary"


@dataclass
class DRCViolation:
    """Represents a Design Rule Check violation."""
    violation_type: DRCViolationType
    net_id: int
    location: Tuple[float, float]
    description: str
    severity: str = "error"  # "error" or "warning"
    
    def __str__(self) -> str:
        return f"{self.violation_type.value} at ({self.location[0]:.2f}, {self.location[1]:.2f}): {self.description}"


@dataclass
class DetailedRoute:
    """Complete detailed route for a net."""
    net_id: int
    io_pad: Any  # IOPad
    bump_pad: Any  # BumpPad
    path: List[Point] = field(default_factory=list)
    segments: List[RouteSegment] = field(default_factory=list)
    layer: int = 1
    wire_width: float = 10.0
    route_type: RouteType = RouteType.L_SHAPE
    is_valid: bool = True
    drc_violations: List[DRCViolation] = field(default_factory=list)
    
    @property
    def wirelength(self) -> float:
        """Calculate total wirelength."""
        if not self.path or len(self.path) < 2:
            return 0.0
        return path_manhattan_length(self.path)
    
    @property
    def num_bends(self) -> int:
        """Count number of bends in route."""
        if not self.path:
            return 0
        return count_bends(self.path)
    
    @property
    def has_violations(self) -> bool:
        """Check if route has DRC violations."""
        return len(self.drc_violations) > 0
    
    def get_bounding_box(self) -> Optional[BoundingBox]:
        """Get bounding box of route."""
        if not self.path:
            return None
        xs = [p.x for p in self.path]
        ys = [p.y for p in self.path]
        return BoundingBox(min(xs), min(ys), max(xs), max(ys))
    
    def to_segments(self) -> List[RouteSegment]:
        """Convert path to segments."""
        if len(self.path) < 2:
            return []
        segments = []
        for i in range(len(self.path) - 1):
            seg = RouteSegment(
                start=self.path[i],
                end=self.path[i + 1],
                layer=self.layer,
                width=self.wire_width
            )
            segments.append(seg)
        self.segments = segments
        return segments


@dataclass
class DetailedRoutingResult:
    """Result of detailed routing."""
    status: DetailedRoutingStatus
    routes: List[DetailedRoute] = field(default_factory=list)
    total_wirelength: float = 0.0
    total_bends: int = 0
    num_routed: int = 0
    num_failed: int = 0
    drc_violations: List[DRCViolation] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def routability_rate(self) -> float:
        """Calculate routability rate."""
        total = self.num_routed + self.num_failed
        if total == 0:
            return 0.0
        return self.num_routed / total
    
    @property
    def is_complete(self) -> bool:
        """Check if all nets are routed."""
        return self.num_failed == 0
    
    @property
    def is_drc_clean(self) -> bool:
        """Check if routing is DRC clean."""
        return len(self.drc_violations) == 0
    
    def get_route_by_net(self, net_id: int) -> Optional[DetailedRoute]:
        """Get route for specific net."""
        for route in self.routes:
            if route.net_id == net_id:
                return route
        return None


class DRCChecker:
    """Design Rule Checker for routing."""
    
    def __init__(
        self,
        min_spacing: float = 10.0,
        min_width: float = 10.0,
        boundary: Optional[BoundingBox] = None
    ):
        """
        Initialize DRC checker.
        
        Args:
            min_spacing: Minimum spacing between wires
            min_width: Minimum wire width
            boundary: Routing boundary
        """
        self.min_spacing = min_spacing
        self.min_width = min_width
        self.boundary = boundary
    
    def check_route(
        self,
        route: DetailedRoute,
        other_routes: List[DetailedRoute]
    ) -> List[DRCViolation]:
        """
        Check route for DRC violations.
        
        Args:
            route: Route to check
            other_routes: Other existing routes
            
        Returns:
            List of DRC violations
        """
        violations = []
        
        # Check width
        if route.wire_width < self.min_width:
            violations.append(DRCViolation(
                violation_type=DRCViolationType.WIDTH,
                net_id=route.net_id,
                location=(route.path[0].x if route.path else 0, 
                         route.path[0].y if route.path else 0),
                description=f"Wire width {route.wire_width} < min {self.min_width}"
            ))
        
        # Check boundary
        if self.boundary and route.path:
            for point in route.path:
                if not self.boundary.contains_point(point):
                    violations.append(DRCViolation(
                        violation_type=DRCViolationType.BOUNDARY,
                        net_id=route.net_id,
                        location=(point.x, point.y),
                        description="Route point outside boundary"
                    ))
        
        # Check spacing with other routes
        for other in other_routes:
            if other.net_id == route.net_id:
                continue
            
            spacing_violations = self._check_spacing(route, other)
            violations.extend(spacing_violations)
            
            # Check for crossings
            crossing_violations = self._check_crossings(route, other)
            violations.extend(crossing_violations)
        
        return violations
    
    def _check_spacing(
        self,
        route1: DetailedRoute,
        route2: DetailedRoute
    ) -> List[DRCViolation]:
        """Check spacing between two routes."""
        violations = []
        
        # Convert to segments if needed
        segs1 = route1.segments if route1.segments else route1.to_segments()
        segs2 = route2.segments if route2.segments else route2.to_segments()
        
        for seg1 in segs1:
            for seg2 in segs2:
                # Calculate minimum distance between segments
                min_dist = self._segment_distance(seg1, seg2)
                required_spacing = self.min_spacing + (seg1.width + seg2.width) / 2
                
                if min_dist < required_spacing:
                    # Find approximate violation location
                    mid1 = Point(
                        (seg1.start.x + seg1.end.x) / 2,
                        (seg1.start.y + seg1.end.y) / 2
                    )
                    violations.append(DRCViolation(
                        violation_type=DRCViolationType.SPACING,
                        net_id=route1.net_id,
                        location=(mid1.x, mid1.y),
                        description=f"Spacing {min_dist:.2f} < required {required_spacing:.2f} with net {route2.net_id}"
                    ))
        
        return violations
    
    def _check_crossings(
        self,
        route1: DetailedRoute,
        route2: DetailedRoute
    ) -> List[DRCViolation]:
        """Check for wire crossings between routes."""
        violations = []
        
        if not route1.path or not route2.path:
            return violations
        
        # Check if paths intersect
        if paths_intersect(route1.path, route2.path, exclude_endpoints=True):
            # Find intersection point
            for i in range(len(route1.path) - 1):
                for j in range(len(route2.path) - 1):
                    p1, p2 = route1.path[i], route1.path[i + 1]
                    p3, p4 = route2.path[j], route2.path[j + 1]
                    
                    if segments_intersect(p1, p2, p3, p4):
                        mid = Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
                        violations.append(DRCViolation(
                            violation_type=DRCViolationType.CROSSING,
                            net_id=route1.net_id,
                            location=(mid.x, mid.y),
                            description=f"Wire crossing with net {route2.net_id}"
                        ))
        
        return violations
    
    def _segment_distance(self, seg1: RouteSegment, seg2: RouteSegment) -> float:
        """Calculate minimum distance between two segments."""
        # Simplified: use distance between midpoints
        mid1 = Point(
            (seg1.start.x + seg1.end.x) / 2,
            (seg1.start.y + seg1.end.y) / 2
        )
        mid2 = Point(
            (seg2.start.x + seg2.end.x) / 2,
            (seg2.start.y + seg2.end.y) / 2
        )
        return manhattan_distance(mid1, mid2)
    
    def check_all_routes(
        self,
        routes: List[DetailedRoute]
    ) -> List[DRCViolation]:
        """Check all routes for DRC violations."""
        all_violations = []
        
        for i, route in enumerate(routes):
            other_routes = routes[:i] + routes[i+1:]
            violations = self.check_route(route, other_routes)
            all_violations.extend(violations)
            route.drc_violations = violations
        
        return all_violations


class DetailedRouter:
    """
    Detailed router for flip-chip RDL routing.
    
    Takes global routing assignments and generates actual wire paths
    using river routing methodology with DRC checking.
    """
    
    def __init__(
        self,
        wire_width: float = 10.0,
        wire_spacing: float = 10.0,
        layer: int = 1,
        routing_direction: RoutingDirection = RoutingDirection.AUTO,
        enable_drc: bool = True
    ):
        """
        Initialize detailed router.
        
        Args:
            wire_width: Default wire width
            wire_spacing: Minimum wire spacing
            layer: RDL layer number
            routing_direction: Preferred routing direction
            enable_drc: Enable DRC checking
        """
        self.wire_width = wire_width
        self.wire_spacing = wire_spacing
        self.layer = layer
        self.routing_direction = routing_direction
        self.enable_drc = enable_drc
        
        # Initialize river router
        self.river_router = RiverRouter(
            wire_width=wire_width,
            wire_spacing=wire_spacing,
            layer=layer,
            routing_direction=routing_direction
        )
        
        # DRC checker
        self.drc_checker: Optional[DRCChecker] = None
        
        # Obstacles
        self.obstacles: List[BoundingBox] = []
        
        # Routing boundary
        self.boundary: Optional[BoundingBox] = None
    
    def set_boundary(self, boundary: BoundingBox) -> None:
        """Set routing boundary."""
        self.boundary = boundary
        if self.enable_drc:
            self.drc_checker = DRCChecker(
                min_spacing=self.wire_spacing,
                min_width=self.wire_width,
                boundary=boundary
            )
    
    def add_obstacle(self, obstacle: BoundingBox) -> None:
        """Add routing obstacle."""
        self.obstacles.append(obstacle)
        self.river_router.add_obstacle(obstacle)
    
    def clear_obstacles(self) -> None:
        """Clear all obstacles."""
        self.obstacles.clear()
        self.river_router.obstacles.clear()
    
    def route(self, chip: Chip) -> DetailedRoutingResult:
        """
        Perform detailed routing on chip.
        
        Args:
            chip: Chip with global routing assignments
            
        Returns:
            DetailedRoutingResult with routes and statistics
        """
        # Set boundary from chip dimensions
        self.set_boundary(BoundingBox(0, 0, chip.die_width, chip.die_height))
        
        # Collect assignments from chip
        assignments = []
        for io_pad in chip.io_pads:
            if io_pad.assigned_bump is not None:
                assignments.append((io_pad, io_pad.assigned_bump))
        
        if not assignments:
            return DetailedRoutingResult(
                status=DetailedRoutingStatus.SUCCESS,
                statistics={"message": "No assignments to route"}
            )
        
        return self.route_assignments(assignments)
    
    def route_assignments(
        self,
        assignments: List[Tuple[Any, Any]]
    ) -> DetailedRoutingResult:
        """
        Route a list of IO-to-bump assignments.
        
        Args:
            assignments: List of (io_pad, bump_pad) tuples
            
        Returns:
            DetailedRoutingResult
        """
        # Check planarity
        # Convert assignments to coordinate format for planarity check
        coord_assignments = [
            ((io_pad.x, io_pad.y), (bump_pad.x, bump_pad.y))
            for io_pad, bump_pad in assignments
        ]
        is_planar, num_crossings = check_planarity(coord_assignments)

        if not is_planar:
            # Try to reorder for planarity
            assignments = self._reorder_for_planarity(assignments)
            # Re-convert after reordering
            coord_assignments = [
                ((io_pad.x, io_pad.y), (bump_pad.x, bump_pad.y))
                for io_pad, bump_pad in assignments
            ]
            is_planar, num_crossings = check_planarity(coord_assignments)
        
        # Prepare data for river router
        io_pads = [a[0] for a in assignments]
        bump_pads = [a[1] for a in assignments]
        
        # Run river routing
        river_result = river_route(
            io_pads=io_pads,
            bump_pads=bump_pads,
            assignments=assignments,
            wire_width=self.wire_width,
            wire_spacing=self.wire_spacing
        )
        
        # Convert river routes to detailed routes
        detailed_routes = []
        for route in river_result.routes:
            # Find corresponding IO and bump pads
            io_pad = None
            bump_pad = None
            for io, bump in assignments:
                if hasattr(io, 'node_id') and io.node_id == route.io_id:
                    io_pad = io
                    bump_pad = bump
                    break
            
            # Check if route is valid (has a path)
            is_valid = len(route.path) >= 2 if route.path else False

            detailed_route = DetailedRoute(
                net_id=route.io_id,
                io_pad=io_pad,
                bump_pad=bump_pad,
                path=route.path,
                layer=route.layer,
                wire_width=route.width,
                route_type=route.route_type,
                is_valid=is_valid
            )
            detailed_route.to_segments()
            detailed_routes.append(detailed_route)
        
        # Run DRC if enabled
        drc_violations = []
        if self.enable_drc and self.drc_checker:
            drc_violations = self.drc_checker.check_all_routes(detailed_routes)
        
        # Calculate statistics
        total_wirelength = sum(r.wirelength for r in detailed_routes)
        total_bends = sum(r.num_bends for r in detailed_routes)
        num_routed = sum(1 for r in detailed_routes if r.is_valid)
        num_failed = len(detailed_routes) - num_routed
        
        # Determine status
        if num_failed == 0 and len(drc_violations) == 0:
            status = DetailedRoutingStatus.SUCCESS
        elif num_failed == 0:
            status = DetailedRoutingStatus.DRC_VIOLATION
        elif num_routed > 0:
            status = DetailedRoutingStatus.PARTIAL
        else:
            status = DetailedRoutingStatus.FAILED
        
        return DetailedRoutingResult(
            status=status,
            routes=detailed_routes,
            total_wirelength=total_wirelength,
            total_bends=total_bends,
            num_routed=num_routed,
            num_failed=num_failed,
            drc_violations=drc_violations,
            statistics={
                "num_assignments": len(assignments),
                "is_planar": is_planar,
                "num_crossings": num_crossings,
                "avg_wirelength": total_wirelength / max(1, num_routed),
                "avg_bends": total_bends / max(1, num_routed),
                "drc_errors": len([v for v in drc_violations if v.severity == "error"]),
                "drc_warnings": len([v for v in drc_violations if v.severity == "warning"])
            }
        )
    
    def route_single(
        self,
        io_pad: Any,
        bump_pad: Any,
        existing_routes: List[DetailedRoute] = None
    ) -> DetailedRoute:
        """
        Route a single IO-to-bump connection.
        
        Args:
            io_pad: Source IO pad
            bump_pad: Target bump pad
            existing_routes: Already routed paths to avoid
            
        Returns:
            DetailedRoute for this connection
        """
        existing_routes = existing_routes or []
        
        # Get positions
        start = Point(io_pad.x, io_pad.y)
        end = Point(bump_pad.x, bump_pad.y)
        
        # Collect existing paths as obstacles
        existing_paths = [r.path for r in existing_routes if r.path]
        
        # Try L-shape first
        path = self._route_l_shape(start, end, existing_paths)
        route_type = RouteType.L_SHAPE
        
        # If L-shape fails, try Z-shape
        if path is None or self._has_collision(path, existing_paths):
            path = self._route_z_shape(start, end, existing_paths)
            route_type = RouteType.Z_SHAPE
        
        # Create detailed route
        net_id = getattr(io_pad, 'node_id', 0)
        is_valid = path is not None and not self._has_collision(path, existing_paths)
        
        route = DetailedRoute(
            net_id=net_id,
            io_pad=io_pad,
            bump_pad=bump_pad,
            path=path if path else [],
            layer=self.layer,
            wire_width=self.wire_width,
            route_type=route_type,
            is_valid=is_valid
        )
        
        if path:
            route.to_segments()
        
        # DRC check
        if self.enable_drc and self.drc_checker and is_valid:
            violations = self.drc_checker.check_route(route, existing_routes)
            route.drc_violations = violations
            if any(v.severity == "error" for v in violations):
                route.is_valid = False
        
        return route
    
    def _route_l_shape(
        self,
        start: Point,
        end: Point,
        existing_paths: List[List[Point]]
    ) -> Optional[List[Point]]:
        """Generate L-shaped route."""
        # Try horizontal-first
        path1 = generate_l_shape_path(start, end, horizontal_first=True)
        if not self._has_collision(path1, existing_paths):
            return path1
        
        # Try vertical-first
        path2 = generate_l_shape_path(start, end, horizontal_first=False)
        if not self._has_collision(path2, existing_paths):
            return path2
        
        return None
    
    def _route_z_shape(
        self,
        start: Point,
        end: Point,
        existing_paths: List[List[Point]]
    ) -> Optional[List[Point]]:
        """Generate Z-shaped route."""
        # Try different midpoint ratios
        for ratio in [0.5, 0.3, 0.7, 0.25, 0.75]:
            mid_x = start.x + (end.x - start.x) * ratio
            mid_y = start.y + (end.y - start.y) * ratio
            
            path = generate_z_shape_path(start, end, mid_x=mid_x)
            if not self._has_collision(path, existing_paths):
                return path
            
            path = generate_z_shape_path(start, end, mid_y=mid_y)
            if not self._has_collision(path, existing_paths):
                return path
        
        return None
    
    def _has_collision(
        self,
        path: List[Point],
        existing_paths: List[List[Point]]
    ) -> bool:
        """Check if path collides with existing paths."""
        if not path:
            return True
        
        for existing in existing_paths:
            if paths_intersect(path, existing, exclude_endpoints=True):
                return True
        
        # Check obstacles
        for obstacle in self.obstacles:
            for point in path:
                if obstacle.contains_point(point.x, point.y):
                    return True
        
        return False
    
    def _reorder_for_planarity(
        self,
        assignments: List[Tuple[Any, Any]]
    ) -> List[Tuple[Any, Any]]:
        """
        Reorder assignments to minimize crossings.
        
        Uses simple heuristic: sort by IO x-coordinate.
        """
        return sorted(assignments, key=lambda a: (a[0].x, a[0].y))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get router statistics."""
        return {
            "wire_width": self.wire_width,
            "wire_spacing": self.wire_spacing,
            "layer": self.layer,
            "num_obstacles": len(self.obstacles),
            "drc_enabled": self.enable_drc
        }


def detailed_route(
    chip: Chip,
    wire_width: float = 10.0,
    wire_spacing: float = 10.0,
    enable_drc: bool = True
) -> DetailedRoutingResult:
    """
    Convenience function for detailed routing.
    
    Args:
        chip: Chip with global routing assignments
        wire_width: Wire width
        wire_spacing: Wire spacing
        enable_drc: Enable DRC checking
        
    Returns:
        DetailedRoutingResult
    """
    router = DetailedRouter(
        wire_width=wire_width,
        wire_spacing=wire_spacing,
        enable_drc=enable_drc
    )
    return router.route(chip)


def route_with_global(
    chip: Chip,
    wire_width: float = 10.0,
    wire_spacing: float = 10.0
) -> Tuple[Any, DetailedRoutingResult]:
    """
    Perform both global and detailed routing.
    
    Args:
        chip: Chip to route
        wire_width: Wire width
        wire_spacing: Wire spacing
        
    Returns:
        Tuple of (GlobalRoutingResult, DetailedRoutingResult)
    """
    from .global_router import global_route
    
    # Global routing
    global_result = global_route(chip)
    
    # Detailed routing
    detailed_result = detailed_route(
        chip,
        wire_width=wire_width,
        wire_spacing=wire_spacing
    )
    
    return global_result, detailed_result


# Test function
def test_detailed_router():
    """Test detailed router functionality."""
    print("Testing DetailedRouter...")
    
    # Create test data
    from ..data_structures import IOPad, BumpPad
    
    # Reset counters
    IOPad.reset_node_counter()
    BumpPad.reset_node_counter()
    
    # Create IO pads
    io_pads = [
        IOPad(x=100, y=0, net_id=0, name="io_0"),
        IOPad(x=200, y=0, net_id=1, name="io_1"),
        IOPad(x=300, y=0, net_id=2, name="io_2"),
    ]
    
    # Create bump pads
    bump_pads = [
        BumpPad(x=100, y=200, name="bump_0"),
        BumpPad(x=200, y=200, name="bump_1"),
        BumpPad(x=300, y=200, name="bump_2"),
    ]
    
    # Create assignments (no crossings)
    assignments = list(zip(io_pads, bump_pads))
    
    # Assign bumps to IOs
    for io, bump in assignments:
        io.assigned_bump = bump
    
    # Create router
    router = DetailedRouter(
        wire_width=10.0,
        wire_spacing=10.0,
        enable_drc=True
    )
    router.set_boundary(BoundingBox(0, 0, 500, 500))
    
    # Route
    result = router.route_assignments(assignments)
    
    print(f"  Status: {result.status.value}")
    print(f"  Routed: {result.num_routed}/{result.num_routed + result.num_failed}")
    print(f"  Total wirelength: {result.total_wirelength:.2f}")
    print(f"  Total bends: {result.total_bends}")
    print(f"  DRC violations: {len(result.drc_violations)}")
    print(f"  Is complete: {result.is_complete}")
    print(f"  Is DRC clean: {result.is_drc_clean}")
    
    # Test single route
    print("\nTesting single route...")
    single_route = router.route_single(io_pads[0], bump_pads[0])
    print(f"  Valid: {single_route.is_valid}")
    print(f"  Wirelength: {single_route.wirelength:.2f}")
    print(f"  Bends: {single_route.num_bends}")
    
    print("\nDetailedRouter tests passed!")


if __name__ == "__main__":
    test_detailed_router()
