"""
Net data structure for flip-chip RDL routing.

Represents a net/wire connection between IO pads and bump pads,
storing routing information and path data for the RDL layer.
"""

from typing import List, Optional, Tuple, Any
import math


class Net:
    """
    Represents a net (electrical connection) in the flip-chip design.
    
    A net connects one or more IO pads to bump pads through the RDL layer.
    It stores the routing path, wire properties, and connection information.
    
    Attributes:
        net_id: Unique identifier for this net
        name: Human-readable net name
        io_pads: List of IO pads belonging to this net
        bump_pads: List of bump pads assigned to this net
        route_path: List of (x, y) coordinates defining the routed path
        wire_width: Width of the wire in micrometers
        wire_spacing: Minimum spacing to other wires in micrometers
        layer: RDL layer number for this net
        is_routed: Whether the net has been successfully routed
        is_power: Whether this is a power/ground net
        priority: Routing priority (higher = route first)
    """
    
    _net_counter = 0  # Class-level counter for auto-generating net IDs
    
    def __init__(
        self,
        net_id: Optional[int] = None,
        name: str = "",
        wire_width: float = 10.0,
        wire_spacing: float = 10.0,
        layer: int = 1,
        is_power: bool = False,
        priority: int = 0
    ):
        """
        Initialize a Net instance.
        
        Args:
            net_id: Unique net identifier (auto-generated if None)
            name: Human-readable net name
            wire_width: Width of the wire in micrometers
            wire_spacing: Minimum spacing to other wires in micrometers
            layer: RDL layer number
            is_power: Whether this is a power/ground net
            priority: Routing priority (higher = route first)
        """
        if net_id is None:
            self.net_id = Net._net_counter
            Net._net_counter += 1
        else:
            self.net_id = net_id
            Net._net_counter = max(Net._net_counter, net_id + 1)
        
        self.name = name if name else f"net_{self.net_id}"
        self.wire_width = wire_width
        self.wire_spacing = wire_spacing
        self.layer = layer
        self.is_power = is_power
        self.priority = priority
        
        # Connection lists
        self.io_pads: List[Any] = []  # List of IOPad objects
        self.bump_pads: List[Any] = []  # List of BumpPad objects
        
        # Routing information
        self.route_path: List[Tuple[float, float]] = []
        self.route_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        self.is_routed: bool = False
        self.routing_order: int = -1  # Order in which this net should be routed
        
        # Metrics
        self.wirelength: float = 0.0
        self.num_bends: int = 0
    
    @classmethod
    def reset_net_counter(cls) -> None:
        """Reset the net ID counter to 0."""
        cls._net_counter = 0
    
    @property
    def num_pins(self) -> int:
        """Return total number of pins (IO pads + bump pads) in this net."""
        return len(self.io_pads) + len(self.bump_pads)
    
    @property
    def num_io_pads(self) -> int:
        """Return number of IO pads in this net."""
        return len(self.io_pads)
    
    @property
    def num_bump_pads(self) -> int:
        """Return number of bump pads in this net."""
        return len(self.bump_pads)
    
    @property
    def is_single_pin(self) -> bool:
        """Check if this is a single-pin net (no routing needed)."""
        return self.num_pins <= 1
    
    @property
    def is_two_pin(self) -> bool:
        """Check if this is a two-pin net (simple point-to-point routing)."""
        return len(self.io_pads) == 1 and len(self.bump_pads) == 1
    
    @property
    def is_multi_pin(self) -> bool:
        """Check if this is a multi-pin net."""
        return self.num_pins > 2
    
    def add_io_pad(self, io_pad: Any) -> None:
        """
        Add an IO pad to this net.
        
        Args:
            io_pad: IOPad object to add
        """
        if io_pad not in self.io_pads:
            self.io_pads.append(io_pad)
            io_pad.net_id = self.net_id
    
    def remove_io_pad(self, io_pad: Any) -> bool:
        """
        Remove an IO pad from this net.
        
        Args:
            io_pad: IOPad object to remove
            
        Returns:
            True if removed, False if not found
        """
        if io_pad in self.io_pads:
            self.io_pads.remove(io_pad)
            io_pad.net_id = -1
            return True
        return False
    
    def add_bump_pad(self, bump_pad: Any) -> None:
        """
        Add a bump pad to this net.
        
        Args:
            bump_pad: BumpPad object to add
        """
        if bump_pad not in self.bump_pads:
            self.bump_pads.append(bump_pad)
            bump_pad.net_id = self.net_id
    
    def remove_bump_pad(self, bump_pad: Any) -> bool:
        """
        Remove a bump pad from this net.
        
        Args:
            bump_pad: BumpPad object to remove
            
        Returns:
            True if removed, False if not found
        """
        if bump_pad in self.bump_pads:
            self.bump_pads.remove(bump_pad)
            bump_pad.net_id = -1
            return True
        return False
    
    def set_route_path(self, path: List[Tuple[float, float]]) -> None:
        """
        Set the routing path for this net.
        
        Args:
            path: List of (x, y) coordinates defining the route
        """
        self.route_path = path
        self.is_routed = len(path) >= 2
        
        if self.is_routed:
            self._calculate_metrics()
    
    def add_route_segment(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float]
    ) -> None:
        """
        Add a route segment to this net.
        
        Args:
            start: Starting point (x, y)
            end: Ending point (x, y)
        """
        self.route_segments.append((start, end))
        
        # Update path if needed
        if not self.route_path:
            self.route_path.append(start)
        self.route_path.append(end)
        
        self.is_routed = True
        self._calculate_metrics()
    
    def clear_route(self) -> None:
        """Clear all routing information for this net."""
        self.route_path = []
        self.route_segments = []
        self.is_routed = False
        self.wirelength = 0.0
        self.num_bends = 0
    
    def _calculate_metrics(self) -> None:
        """Calculate routing metrics (wirelength, bends) from path."""
        if len(self.route_path) < 2:
            self.wirelength = 0.0
            self.num_bends = 0
            return
        
        # Calculate total wirelength
        total_length = 0.0
        for i in range(len(self.route_path) - 1):
            x1, y1 = self.route_path[i]
            x2, y2 = self.route_path[i + 1]
            # Manhattan distance for rectilinear routing
            total_length += abs(x2 - x1) + abs(y2 - y1)
        
        self.wirelength = total_length
        
        # Count bends (direction changes)
        bends = 0
        for i in range(1, len(self.route_path) - 1):
            x0, y0 = self.route_path[i - 1]
            x1, y1 = self.route_path[i]
            x2, y2 = self.route_path[i + 1]
            
            # Direction vectors
            dx1, dy1 = x1 - x0, y1 - y0
            dx2, dy2 = x2 - x1, y2 - y1
            
            # Check for direction change
            if (dx1 != 0 and dy2 != 0) or (dy1 != 0 and dx2 != 0):
                bends += 1
        
        self.num_bends = bends
    
    def get_bounding_box(self) -> Optional[Tuple[float, float, float, float]]:
        """
        Get the bounding box of all pins in this net.
        
        Returns:
            Tuple (x_min, y_min, x_max, y_max) or None if no pins
        """
        all_points = []
        
        for io_pad in self.io_pads:
            all_points.append((io_pad.x, io_pad.y))
        
        for bump_pad in self.bump_pads:
            all_points.append((bump_pad.x, bump_pad.y))
        
        if not all_points:
            return None
        
        x_coords = [p[0] for p in all_points]
        y_coords = [p[1] for p in all_points]
        
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    
    def get_half_perimeter_wirelength(self) -> float:
        """
        Calculate the half-perimeter wirelength (HPWL) estimate.
        
        HPWL is a lower bound on the actual wirelength for a net.
        
        Returns:
            Half-perimeter wirelength estimate
        """
        bbox = self.get_bounding_box()
        if bbox is None:
            return 0.0
        
        x_min, y_min, x_max, y_max = bbox
        return (x_max - x_min) + (y_max - y_min)
    
    def get_centroid(self) -> Optional[Tuple[float, float]]:
        """
        Get the centroid of all pins in this net.
        
        Returns:
            Tuple (x, y) of centroid or None if no pins
        """
        all_points = []
        
        for io_pad in self.io_pads:
            all_points.append((io_pad.x, io_pad.y))
        
        for bump_pad in self.bump_pads:
            all_points.append((bump_pad.x, bump_pad.y))
        
        if not all_points:
            return None
        
        x_avg = sum(p[0] for p in all_points) / len(all_points)
        y_avg = sum(p[1] for p in all_points) / len(all_points)
        
        return (x_avg, y_avg)
    
    def crosses(self, other: 'Net') -> bool:
        """
        Check if this net's route crosses another net's route.
        
        This is used for planarity checking in river routing.
        
        Args:
            other: Another Net to check crossing with
            
        Returns:
            True if routes cross, False otherwise
        """
        if not self.is_routed or not other.is_routed:
            return False
        
        # Check segment intersections
        for seg1 in self.route_segments:
            for seg2 in other.route_segments:
                if self._segments_intersect(seg1, seg2):
                    return True
        
        return False
    
    @staticmethod
    def _segments_intersect(
        seg1: Tuple[Tuple[float, float], Tuple[float, float]],
        seg2: Tuple[Tuple[float, float], Tuple[float, float]]
    ) -> bool:
        """
        Check if two line segments intersect.
        
        Args:
            seg1: First segment ((x1, y1), (x2, y2))
            seg2: Second segment ((x3, y3), (x4, y4))
            
        Returns:
            True if segments intersect (not at endpoints)
        """
        (x1, y1), (x2, y2) = seg1
        (x3, y3), (x4, y4) = seg2
        
        def ccw(ax, ay, bx, by, cx, cy):
            return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)
        
        # Check if segments intersect (excluding endpoints)
        if ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4) and \
           ccw(x1, y1, x2, y2, x3, y3) != ccw(x1, y1, x2, y2, x4, y4):
            # Check it's not just touching at endpoints
            if (x1, y1) != (x3, y3) and (x1, y1) != (x4, y4) and \
               (x2, y2) != (x3, y3) and (x2, y2) != (x4, y4):
                return True
        
        return False
    
    def would_cross(self, other: 'Net') -> bool:
        """
        Check if routing this net would cross another net.
        
        Used for net ordering in river routing - checks if the
        IO-to-bump assignments would create a crossing.
        
        Args:
            other: Another Net to check potential crossing with
            
        Returns:
            True if nets would cross based on terminal positions
        """
        if not self.io_pads or not self.bump_pads:
            return False
        if not other.io_pads or not other.bump_pads:
            return False
        
        # Get representative terminals
        io1 = self.io_pads[0]
        bump1 = self.bump_pads[0] if self.bump_pads else None
        io2 = other.io_pads[0]
        bump2 = other.bump_pads[0] if other.bump_pads else None
        
        if bump1 is None or bump2 is None:
            return False
        
        # Check crossing condition:
        # If io1.x < io2.x but bump1.x > bump2.x (or vice versa), they cross
        io_order = io1.x < io2.x
        bump_order = bump1.x < bump2.x
        
        return io_order != bump_order
    
    def copy(self) -> 'Net':
        """
        Create a copy of this net (without copying connected pads).
        
        Returns:
            New Net instance with same properties
        """
        new_net = Net(
            net_id=None,  # Get new ID
            name=f"{self.name}_copy",
            wire_width=self.wire_width,
            wire_spacing=self.wire_spacing,
            layer=self.layer,
            is_power=self.is_power,
            priority=self.priority
        )
        new_net.route_path = self.route_path.copy()
        new_net.route_segments = self.route_segments.copy()
        new_net.is_routed = self.is_routed
        new_net.routing_order = self.routing_order
        new_net.wirelength = self.wirelength
        new_net.num_bends = self.num_bends
        return new_net
    
    def to_dict(self) -> dict:
        """
        Convert net to dictionary for serialization.
        
        Returns:
            Dictionary representation of the net
        """
        return {
            'net_id': self.net_id,
            'name': self.name,
            'wire_width': self.wire_width,
            'wire_spacing': self.wire_spacing,
            'layer': self.layer,
            'is_power': self.is_power,
            'priority': self.priority,
            'io_pad_ids': [io.node_id for io in self.io_pads],
            'bump_pad_ids': [bp.node_id for bp in self.bump_pads],
            'route_path': self.route_path,
            'is_routed': self.is_routed,
            'routing_order': self.routing_order,
            'wirelength': self.wirelength,
            'num_bends': self.num_bends
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Net':
        """
        Create a Net from dictionary representation.
        
        Note: IO pads and bump pads must be reconnected separately.
        
        Args:
            data: Dictionary with net data
            
        Returns:
            New Net instance
        """
        net = cls(
            net_id=data.get('net_id'),
            name=data.get('name', ''),
            wire_width=data.get('wire_width', 10.0),
            wire_spacing=data.get('wire_spacing', 10.0),
            layer=data.get('layer', 1),
            is_power=data.get('is_power', False),
            priority=data.get('priority', 0)
        )
        net.route_path = data.get('route_path', [])
        net.is_routed = data.get('is_routed', False)
        net.routing_order = data.get('routing_order', -1)
        net.wirelength = data.get('wirelength', 0.0)
        net.num_bends = data.get('num_bends', 0)
        return net
    
    def __repr__(self) -> str:
        """Return string representation of the net."""
        status = "routed" if self.is_routed else "unrouted"
        return (f"Net(id={self.net_id}, name='{self.name}', "
                f"ios={len(self.io_pads)}, bumps={len(self.bump_pads)}, "
                f"status={status}, wl={self.wirelength:.1f})")
    
    def __eq__(self, other: object) -> bool:
        """Check equality based on net_id."""
        if not isinstance(other, Net):
            return False
        return self.net_id == other.net_id
    
    def __hash__(self) -> int:
        """Hash based on net_id."""
        return hash(self.net_id)
    
    def __lt__(self, other: 'Net') -> bool:
        """Compare nets by priority (for sorting)."""
        if self.priority != other.priority:
            return self.priority > other.priority  # Higher priority first
        return self.net_id < other.net_id
