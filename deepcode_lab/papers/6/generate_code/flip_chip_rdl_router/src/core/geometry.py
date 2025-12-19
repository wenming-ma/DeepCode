"""
Geometric primitives and utilities for flip-chip RDL routing.

This module provides fundamental geometric operations including:
- Point and vector operations
- Line segment intersection detection
- Bounding box calculations
- Manhattan and Euclidean distance computations
- Path generation utilities
"""

from typing import List, Tuple, Optional, Union
import math


class Point:
    """Represents a 2D point with x, y coordinates."""
    
    __slots__ = ['x', 'y']
    
    def __init__(self, x: float, y: float):
        """
        Initialize a point.
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        self.x = float(x)
        self.y = float(y)
    
    def __repr__(self) -> str:
        return f"Point({self.x}, {self.y})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Point):
            return False
        return abs(self.x - other.x) < 1e-9 and abs(self.y - other.y) < 1e-9
    
    def __hash__(self) -> int:
        return hash((round(self.x, 6), round(self.y, 6)))
    
    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Point') -> 'Point':
        return Point(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> 'Point':
        return Point(self.x * scalar, self.y * scalar)
    
    def __truediv__(self, scalar: float) -> 'Point':
        return Point(self.x / scalar, self.y / scalar)
    
    @property
    def tuple(self) -> Tuple[float, float]:
        """Return point as tuple."""
        return (self.x, self.y)
    
    def distance_to(self, other: 'Point') -> float:
        """Calculate Euclidean distance to another point."""
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)
    
    def manhattan_distance_to(self, other: 'Point') -> float:
        """Calculate Manhattan distance to another point."""
        return abs(self.x - other.x) + abs(self.y - other.y)
    
    def midpoint(self, other: 'Point') -> 'Point':
        """Calculate midpoint between this point and another."""
        return Point((self.x + other.x) / 2, (self.y + other.y) / 2)
    
    def copy(self) -> 'Point':
        """Create a copy of this point."""
        return Point(self.x, self.y)
    
    @staticmethod
    def from_tuple(t: Tuple[float, float]) -> 'Point':
        """Create a Point from a tuple."""
        return Point(t[0], t[1])


class Segment:
    """Represents a line segment between two points."""
    
    __slots__ = ['p1', 'p2']
    
    def __init__(self, p1: Point, p2: Point):
        """
        Initialize a segment.
        
        Args:
            p1: Start point
            p2: End point
        """
        self.p1 = p1
        self.p2 = p2
    
    def __repr__(self) -> str:
        return f"Segment({self.p1}, {self.p2})"
    
    @property
    def length(self) -> float:
        """Calculate segment length."""
        return self.p1.distance_to(self.p2)
    
    @property
    def manhattan_length(self) -> float:
        """Calculate Manhattan length of segment."""
        return self.p1.manhattan_distance_to(self.p2)
    
    @property
    def is_horizontal(self) -> bool:
        """Check if segment is horizontal."""
        return abs(self.p1.y - self.p2.y) < 1e-9
    
    @property
    def is_vertical(self) -> bool:
        """Check if segment is vertical."""
        return abs(self.p1.x - self.p2.x) < 1e-9
    
    @property
    def is_manhattan(self) -> bool:
        """Check if segment is Manhattan (horizontal or vertical)."""
        return self.is_horizontal or self.is_vertical
    
    @property
    def midpoint(self) -> Point:
        """Get midpoint of segment."""
        return self.p1.midpoint(self.p2)
    
    @property
    def bounding_box(self) -> 'BoundingBox':
        """Get bounding box of segment."""
        return BoundingBox(
            min(self.p1.x, self.p2.x),
            min(self.p1.y, self.p2.y),
            max(self.p1.x, self.p2.x),
            max(self.p1.y, self.p2.y)
        )
    
    def contains_point(self, p: Point, tolerance: float = 1e-9) -> bool:
        """Check if point lies on segment."""
        # Check if point is collinear and within bounds
        cross = (p.y - self.p1.y) * (self.p2.x - self.p1.x) - \
                (p.x - self.p1.x) * (self.p2.y - self.p1.y)
        if abs(cross) > tolerance:
            return False
        
        # Check if point is within segment bounds
        if p.x < min(self.p1.x, self.p2.x) - tolerance:
            return False
        if p.x > max(self.p1.x, self.p2.x) + tolerance:
            return False
        if p.y < min(self.p1.y, self.p2.y) - tolerance:
            return False
        if p.y > max(self.p1.y, self.p2.y) + tolerance:
            return False
        
        return True
    
    def intersects(self, other: 'Segment', tolerance: float = 1e-9) -> bool:
        """
        Check if this segment intersects another segment.
        
        Uses cross product method for robust intersection detection.
        """
        return segments_intersect(self.p1, self.p2, other.p1, other.p2, tolerance)
    
    def intersection_point(self, other: 'Segment') -> Optional[Point]:
        """
        Find intersection point with another segment.
        
        Returns None if segments don't intersect or are parallel.
        """
        return segment_intersection_point(self.p1, self.p2, other.p1, other.p2)
    
    def copy(self) -> 'Segment':
        """Create a copy of this segment."""
        return Segment(self.p1.copy(), self.p2.copy())


class BoundingBox:
    """Represents an axis-aligned bounding box."""
    
    __slots__ = ['x_min', 'y_min', 'x_max', 'y_max']
    
    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float):
        """
        Initialize a bounding box.
        
        Args:
            x_min: Minimum x coordinate
            y_min: Minimum y coordinate
            x_max: Maximum x coordinate
            y_max: Maximum y coordinate
        """
        self.x_min = float(x_min)
        self.y_min = float(y_min)
        self.x_max = float(x_max)
        self.y_max = float(y_max)
    
    def __repr__(self) -> str:
        return f"BoundingBox({self.x_min}, {self.y_min}, {self.x_max}, {self.y_max})"
    
    @property
    def width(self) -> float:
        """Get width of bounding box."""
        return self.x_max - self.x_min
    
    @property
    def height(self) -> float:
        """Get height of bounding box."""
        return self.y_max - self.y_min
    
    @property
    def area(self) -> float:
        """Get area of bounding box."""
        return self.width * self.height
    
    @property
    def perimeter(self) -> float:
        """Get perimeter of bounding box."""
        return 2 * (self.width + self.height)
    
    @property
    def half_perimeter(self) -> float:
        """Get half-perimeter (HPWL estimate)."""
        return self.width + self.height
    
    @property
    def center(self) -> Point:
        """Get center point of bounding box."""
        return Point(
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2
        )
    
    @property
    def corners(self) -> List[Point]:
        """Get corner points of bounding box."""
        return [
            Point(self.x_min, self.y_min),
            Point(self.x_max, self.y_min),
            Point(self.x_max, self.y_max),
            Point(self.x_min, self.y_max)
        ]
    
    def contains_point(self, p: Point, tolerance: float = 0) -> bool:
        """Check if point is inside bounding box."""
        return (self.x_min - tolerance <= p.x <= self.x_max + tolerance and
                self.y_min - tolerance <= p.y <= self.y_max + tolerance)
    
    def contains_box(self, other: 'BoundingBox') -> bool:
        """Check if another bounding box is fully contained."""
        return (self.x_min <= other.x_min and self.x_max >= other.x_max and
                self.y_min <= other.y_min and self.y_max >= other.y_max)
    
    def overlaps(self, other: 'BoundingBox', tolerance: float = 0) -> bool:
        """Check if bounding boxes overlap."""
        return not (self.x_max + tolerance < other.x_min or
                    other.x_max + tolerance < self.x_min or
                    self.y_max + tolerance < other.y_min or
                    other.y_max + tolerance < self.y_min)
    
    def union(self, other: 'BoundingBox') -> 'BoundingBox':
        """Create union of two bounding boxes."""
        return BoundingBox(
            min(self.x_min, other.x_min),
            min(self.y_min, other.y_min),
            max(self.x_max, other.x_max),
            max(self.y_max, other.y_max)
        )
    
    def intersection(self, other: 'BoundingBox') -> Optional['BoundingBox']:
        """Create intersection of two bounding boxes."""
        if not self.overlaps(other):
            return None
        return BoundingBox(
            max(self.x_min, other.x_min),
            max(self.y_min, other.y_min),
            min(self.x_max, other.x_max),
            min(self.y_max, other.y_max)
        )
    
    def expand(self, margin: float) -> 'BoundingBox':
        """Create expanded bounding box with margin."""
        return BoundingBox(
            self.x_min - margin,
            self.y_min - margin,
            self.x_max + margin,
            self.y_max + margin
        )
    
    def copy(self) -> 'BoundingBox':
        """Create a copy of this bounding box."""
        return BoundingBox(self.x_min, self.y_min, self.x_max, self.y_max)
    
    @staticmethod
    def from_points(points: List[Point]) -> 'BoundingBox':
        """Create bounding box from list of points."""
        if not points:
            return BoundingBox(0, 0, 0, 0)
        
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]
        return BoundingBox(
            min(x_coords), min(y_coords),
            max(x_coords), max(y_coords)
        )


# ============================================================================
# Utility Functions
# ============================================================================

def cross_product(o: Point, a: Point, b: Point) -> float:
    """
    Calculate cross product of vectors OA and OB.
    
    Returns positive if counter-clockwise, negative if clockwise, 0 if collinear.
    """
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)


def ccw(a: Point, b: Point, c: Point) -> int:
    """
    Determine orientation of three points.
    
    Returns:
        1 if counter-clockwise
        -1 if clockwise
        0 if collinear
    """
    val = cross_product(a, b, c)
    if val > 1e-9:
        return 1
    elif val < -1e-9:
        return -1
    return 0


def segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point, 
                       tolerance: float = 1e-9) -> bool:
    """
    Check if segment (p1, p2) intersects segment (p3, p4).
    
    Uses the cross product method for robust intersection detection.
    Handles collinear cases and endpoint touches.
    """
    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)
    
    # Standard intersection case
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    
    # Collinear cases
    if d1 == 0 and on_segment(p3, p4, p1):
        return True
    if d2 == 0 and on_segment(p3, p4, p2):
        return True
    if d3 == 0 and on_segment(p1, p2, p3):
        return True
    if d4 == 0 and on_segment(p1, p2, p4):
        return True
    
    return False


def on_segment(p: Point, q: Point, r: Point) -> bool:
    """Check if point r lies on segment pq (assuming collinear)."""
    return (min(p.x, q.x) <= r.x <= max(p.x, q.x) and
            min(p.y, q.y) <= r.y <= max(p.y, q.y))


def segment_intersection_point(p1: Point, p2: Point, p3: Point, p4: Point) -> Optional[Point]:
    """
    Find intersection point of two line segments.
    
    Returns None if segments don't intersect or are parallel.
    """
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    x3, y3 = p3.x, p3.y
    x4, y4 = p4.x, p4.y
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if abs(denom) < 1e-9:
        return None  # Parallel or coincident
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    
    if 0 <= t <= 1 and 0 <= u <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return Point(x, y)
    
    return None


def manhattan_distance(p1: Union[Point, Tuple[float, float]], 
                       p2: Union[Point, Tuple[float, float]]) -> float:
    """Calculate Manhattan distance between two points."""
    if isinstance(p1, tuple):
        p1 = Point(p1[0], p1[1])
    if isinstance(p2, tuple):
        p2 = Point(p2[0], p2[1])
    return abs(p1.x - p2.x) + abs(p1.y - p2.y)


def euclidean_distance(p1: Union[Point, Tuple[float, float]], 
                       p2: Union[Point, Tuple[float, float]]) -> float:
    """Calculate Euclidean distance between two points."""
    if isinstance(p1, tuple):
        p1 = Point(p1[0], p1[1])
    if isinstance(p2, tuple):
        p2 = Point(p2[0], p2[1])
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    return math.sqrt(dx * dx + dy * dy)


# ============================================================================
# Path Generation Utilities
# ============================================================================

def generate_l_shape_path(start: Point, end: Point, 
                          horizontal_first: bool = True) -> List[Point]:
    """
    Generate an L-shaped Manhattan path between two points.
    
    Args:
        start: Starting point
        end: Ending point
        horizontal_first: If True, go horizontal then vertical; else vertical first
    
    Returns:
        List of points forming the L-shaped path
    """
    if horizontal_first:
        corner = Point(end.x, start.y)
    else:
        corner = Point(start.x, end.y)
    
    # Avoid degenerate cases
    if start == corner:
        return [start, end]
    if corner == end:
        return [start, end]
    
    return [start, corner, end]


def generate_z_shape_path(start: Point, end: Point, 
                          mid_x: Optional[float] = None,
                          mid_y: Optional[float] = None) -> List[Point]:
    """
    Generate a Z-shaped Manhattan path between two points.
    
    Args:
        start: Starting point
        end: Ending point
        mid_x: X coordinate for middle segment (if horizontal middle)
        mid_y: Y coordinate for middle segment (if vertical middle)
    
    Returns:
        List of points forming the Z-shaped path
    """
    if mid_x is not None:
        # Vertical-horizontal-vertical path
        p1 = Point(start.x, start.y)
        p2 = Point(start.x, mid_y if mid_y else (start.y + end.y) / 2)
        p3 = Point(end.x, p2.y)
        p4 = Point(end.x, end.y)
        return [p1, p2, p3, p4]
    elif mid_y is not None:
        # Horizontal-vertical-horizontal path
        p1 = Point(start.x, start.y)
        p2 = Point(mid_x if mid_x else (start.x + end.x) / 2, start.y)
        p3 = Point(p2.x, end.y)
        p4 = Point(end.x, end.y)
        return [p1, p2, p3, p4]
    else:
        # Default: use midpoint
        mid_y = (start.y + end.y) / 2
        p1 = Point(start.x, start.y)
        p2 = Point(start.x, mid_y)
        p3 = Point(end.x, mid_y)
        p4 = Point(end.x, end.y)
        return [p1, p2, p3, p4]


def simplify_path(path: List[Point], tolerance: float = 1e-9) -> List[Point]:
    """
    Simplify a path by removing redundant collinear points.
    
    Args:
        path: List of points forming a path
        tolerance: Tolerance for collinearity check
    
    Returns:
        Simplified path with redundant points removed
    """
    if len(path) <= 2:
        return path
    
    simplified = [path[0]]
    
    for i in range(1, len(path) - 1):
        # Check if point is collinear with previous and next
        cross = cross_product(simplified[-1], path[i], path[i + 1])
        if abs(cross) > tolerance:
            simplified.append(path[i])
    
    simplified.append(path[-1])
    return simplified


def path_length(path: List[Point]) -> float:
    """Calculate total length of a path."""
    if len(path) < 2:
        return 0.0
    
    total = 0.0
    for i in range(len(path) - 1):
        total += path[i].distance_to(path[i + 1])
    return total


def path_manhattan_length(path: List[Point]) -> float:
    """Calculate total Manhattan length of a path."""
    if len(path) < 2:
        return 0.0
    
    total = 0.0
    for i in range(len(path) - 1):
        total += path[i].manhattan_distance_to(path[i + 1])
    return total


def count_bends(path: List[Point]) -> int:
    """Count number of bends (direction changes) in a path."""
    if len(path) < 3:
        return 0
    
    bends = 0
    for i in range(1, len(path) - 1):
        # Check if direction changes
        dx1 = path[i].x - path[i - 1].x
        dy1 = path[i].y - path[i - 1].y
        dx2 = path[i + 1].x - path[i].x
        dy2 = path[i + 1].y - path[i].y
        
        # Normalize directions
        if abs(dx1) > 1e-9:
            dx1 = dx1 / abs(dx1)
        if abs(dy1) > 1e-9:
            dy1 = dy1 / abs(dy1)
        if abs(dx2) > 1e-9:
            dx2 = dx2 / abs(dx2)
        if abs(dy2) > 1e-9:
            dy2 = dy2 / abs(dy2)
        
        # Check if direction changed
        if abs(dx1 - dx2) > 1e-9 or abs(dy1 - dy2) > 1e-9:
            bends += 1
    
    return bends


def path_to_segments(path: List[Point]) -> List[Segment]:
    """Convert a path to a list of segments."""
    segments = []
    for i in range(len(path) - 1):
        segments.append(Segment(path[i], path[i + 1]))
    return segments


def paths_intersect(path1: List[Point], path2: List[Point], 
                    exclude_endpoints: bool = True) -> bool:
    """
    Check if two paths intersect.
    
    Args:
        path1: First path
        path2: Second path
        exclude_endpoints: If True, don't count endpoint touches as intersections
    
    Returns:
        True if paths intersect
    """
    segments1 = path_to_segments(path1)
    segments2 = path_to_segments(path2)
    
    for s1 in segments1:
        for s2 in segments2:
            if s1.intersects(s2):
                if exclude_endpoints:
                    # Check if intersection is at endpoints only
                    intersection = s1.intersection_point(s2)
                    if intersection:
                        is_endpoint = (
                            intersection == path1[0] or 
                            intersection == path1[-1] or
                            intersection == path2[0] or 
                            intersection == path2[-1]
                        )
                        if not is_endpoint:
                            return True
                else:
                    return True
    
    return False


# ============================================================================
# Crossing Detection for River Routing
# ============================================================================

def would_cross(io1_x: float, bump1_x: float, 
                io2_x: float, bump2_x: float) -> bool:
    """
    Check if two IO-to-bump assignments would cross in river routing.
    
    In river routing, two wires cross if their relative order changes
    from source to destination.
    
    Args:
        io1_x: X coordinate of first IO pad
        bump1_x: X coordinate of first bump pad
        io2_x: X coordinate of second IO pad
        bump2_x: X coordinate of second bump pad
    
    Returns:
        True if the assignments would result in crossing wires
    """
    # Wires cross if relative order reverses
    io_order = io1_x < io2_x
    bump_order = bump1_x < bump2_x
    
    # Handle equal positions
    if abs(io1_x - io2_x) < 1e-9 or abs(bump1_x - bump2_x) < 1e-9:
        return False
    
    return io_order != bump_order


def count_crossings(assignments: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> int:
    """
    Count total number of crossings in a set of IO-to-bump assignments.
    
    Args:
        assignments: List of ((io_x, io_y), (bump_x, bump_y)) tuples
    
    Returns:
        Number of crossing pairs
    """
    crossings = 0
    n = len(assignments)
    
    for i in range(n):
        for j in range(i + 1, n):
            io1, bump1 = assignments[i]
            io2, bump2 = assignments[j]
            
            if would_cross(io1[0], bump1[0], io2[0], bump2[0]):
                crossings += 1
    
    return crossings


def is_planar_routing(assignments: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> bool:
    """
    Check if a set of assignments can be routed without crossings.
    
    Args:
        assignments: List of ((io_x, io_y), (bump_x, bump_y)) tuples
    
    Returns:
        True if routing is planar (no crossings)
    """
    return count_crossings(assignments) == 0
