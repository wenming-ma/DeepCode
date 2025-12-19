"""
IO Pad data structure for flip-chip RDL routing.

Represents IO buffer pads on the die periphery that need to be routed
to bump pads through the redistribution layer (RDL).
"""

from typing import List, Optional, Tuple, Any
import math


class IOPad:
    """
    Represents an IO pad (wire-bonding pad) on the die periphery.
    
    IO pads are the source terminals that need to be connected to bump pads
    through the RDL layer. In the flow network, each IO pad is connected
    to the source node with capacity 1.
    
    Attributes:
        x (float): X coordinate of the IO pad center
        y (float): Y coordinate of the IO pad center
        net_id (int): Network/signal identifier this pad belongs to
        name (str): Optional name/identifier for the pad
        node_id (int): Unique node ID for flow network construction
        assigned_bump (BumpPad): The bump pad this IO is assigned to (after global routing)
        width (float): Width of the IO pad in micrometers
        height (float): Height of the IO pad in micrometers
        side (str): Which side of the die ('top', 'bottom', 'left', 'right')
        layer (int): Metal layer of the IO pad
        is_power (bool): Whether this is a power/ground pad
        is_routed (bool): Whether this pad has been successfully routed
    """
    
    # Class-level counter for automatic node ID assignment
    _node_counter: int = 0
    
    def __init__(
        self,
        x: float,
        y: float,
        net_id: int = -1,
        name: str = "",
        width: float = 50.0,
        height: float = 50.0,
        node_id: Optional[int] = None,
        side: str = "bottom",
        layer: int = 1,
        is_power: bool = False
    ):
        """
        Initialize an IO pad.
        
        Args:
            x: X coordinate of pad center (micrometers)
            y: Y coordinate of pad center (micrometers)
            net_id: Network/signal identifier (-1 for unassigned)
            name: Optional name for the pad
            width: Pad width in micrometers (default 50um)
            height: Pad height in micrometers (default 50um)
            node_id: Explicit node ID for flow network (auto-assigned if None)
            side: Die side location ('top', 'bottom', 'left', 'right')
            layer: Metal layer number
            is_power: True if this is a power/ground pad
        """
        self.x = float(x)
        self.y = float(y)
        self.net_id = net_id
        self.name = name if name else f"io_{IOPad._node_counter}"
        self.width = float(width)
        self.height = float(height)
        self.side = side.lower()
        self.layer = layer
        self.is_power = is_power
        self.is_routed = False
        
        # Flow network node ID
        if node_id is not None:
            self.node_id = node_id
        else:
            self.node_id = IOPad._node_counter
            IOPad._node_counter += 1
        
        # Routing assignment (set during global routing)
        self.assigned_bump: Optional[Any] = None  # Will be BumpPad instance
        
        # Route path (set during detailed routing)
        self.route_path: List[Tuple[float, float]] = []
    
    @classmethod
    def reset_node_counter(cls, start_value: int = 0) -> None:
        """
        Reset the node counter for flow network ID assignment.
        
        Args:
            start_value: Starting value for the counter
        """
        cls._node_counter = start_value
    
    @property
    def pos(self) -> Tuple[float, float]:
        """Return position as (x, y) tuple."""
        return (self.x, self.y)
    
    @property
    def position(self) -> Tuple[float, float]:
        """Alias for pos property."""
        return self.pos
    
    @property
    def center(self) -> Tuple[float, float]:
        """Return center position (same as pos for IO pads)."""
        return self.pos
    
    @property
    def is_assigned(self) -> bool:
        """Check if this IO pad has been assigned to a bump pad."""
        return self.assigned_bump is not None
    
    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """
        Return bounding box as (x_min, y_min, x_max, y_max).
        """
        half_w = self.width / 2
        half_h = self.height / 2
        return (
            self.x - half_w,
            self.y - half_h,
            self.x + half_w,
            self.y + half_h
        )
    
    def assign_to_bump(self, bump_pad: Any) -> bool:
        """
        Assign this IO pad to a bump pad.
        
        Args:
            bump_pad: The BumpPad to assign to
            
        Returns:
            True if assignment successful, False otherwise
        """
        if self.assigned_bump is not None:
            return False  # Already assigned
        
        self.assigned_bump = bump_pad
        return True
    
    def unassign(self) -> Optional[Any]:
        """
        Remove bump pad assignment.
        
        Returns:
            The previously assigned bump pad, or None
        """
        prev_bump = self.assigned_bump
        self.assigned_bump = None
        self.is_routed = False
        self.route_path = []
        return prev_bump
    
    def set_route_path(self, path: List[Tuple[float, float]]) -> None:
        """
        Set the detailed route path for this IO pad.
        
        Args:
            path: List of (x, y) coordinates forming the route
        """
        self.route_path = path
        self.is_routed = len(path) > 0
    
    def manhattan_distance(self, other: 'IOPad') -> float:
        """
        Calculate Manhattan distance to another IO pad.
        
        Args:
            other: Another IOPad instance
            
        Returns:
            Manhattan distance in micrometers
        """
        return abs(self.x - other.x) + abs(self.y - other.y)
    
    def manhattan_distance_to_point(self, x: float, y: float) -> float:
        """
        Calculate Manhattan distance to a point.
        
        Args:
            x: X coordinate of the point
            y: Y coordinate of the point
            
        Returns:
            Manhattan distance in micrometers
        """
        return abs(self.x - x) + abs(self.y - y)
    
    def euclidean_distance(self, other: 'IOPad') -> float:
        """
        Calculate Euclidean distance to another IO pad.
        
        Args:
            other: Another IOPad instance
            
        Returns:
            Euclidean distance in micrometers
        """
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)
    
    def distance_to_point(self, x: float, y: float) -> float:
        """
        Calculate Euclidean distance to a point.
        
        Args:
            x: X coordinate of the point
            y: Y coordinate of the point
            
        Returns:
            Euclidean distance in micrometers
        """
        dx = self.x - x
        dy = self.y - y
        return math.sqrt(dx * dx + dy * dy)
    
    def distance_to_bump(self, bump_pad: Any) -> float:
        """
        Calculate Manhattan distance to a bump pad.
        
        This is the cost metric used in the MCMF flow network.
        
        Args:
            bump_pad: A BumpPad instance
            
        Returns:
            Manhattan distance in micrometers
        """
        return abs(self.x - bump_pad.x) + abs(self.y - bump_pad.y)
    
    def contains_point(self, x: float, y: float) -> bool:
        """
        Check if a point is within this IO pad's boundary.
        
        Args:
            x: X coordinate to check
            y: Y coordinate to check
            
        Returns:
            True if point is inside the pad
        """
        half_w = self.width / 2
        half_h = self.height / 2
        return (
            self.x - half_w <= x <= self.x + half_w and
            self.y - half_h <= y <= self.y + half_h
        )
    
    def overlaps(self, other: 'IOPad', spacing: float = 0.0) -> bool:
        """
        Check if this IO pad overlaps with another (including spacing).
        
        Args:
            other: Another IOPad instance
            spacing: Minimum required spacing between pads
            
        Returns:
            True if pads overlap or violate spacing
        """
        x_min1, y_min1, x_max1, y_max1 = self.bounding_box
        x_min2, y_min2, x_max2, y_max2 = other.bounding_box
        
        # Add spacing to the check
        return not (
            x_max1 + spacing < x_min2 or
            x_max2 + spacing < x_min1 or
            y_max1 + spacing < y_min2 or
            y_max2 + spacing < y_min1
        )
    
    def copy(self) -> 'IOPad':
        """
        Create a deep copy of this IO pad.
        
        Returns:
            New IOPad instance with same attributes
        """
        new_pad = IOPad(
            x=self.x,
            y=self.y,
            net_id=self.net_id,
            name=self.name + "_copy",
            width=self.width,
            height=self.height,
            node_id=None,  # Get new node ID
            side=self.side,
            layer=self.layer,
            is_power=self.is_power
        )
        new_pad.is_routed = self.is_routed
        new_pad.route_path = self.route_path.copy()
        return new_pad
    
    @staticmethod
    def create_peripheral_pads(
        die_width: float,
        die_height: float,
        num_pads_per_side: int,
        pad_width: float = 50.0,
        pad_height: float = 50.0,
        margin: float = 100.0
    ) -> List['IOPad']:
        """
        Create IO pads arranged around the die periphery.
        
        Args:
            die_width: Width of the die in micrometers
            die_height: Height of the die in micrometers
            num_pads_per_side: Number of pads on each side
            pad_width: Width of each pad
            pad_height: Height of each pad
            margin: Distance from die edge to pad centers
            
        Returns:
            List of IOPad instances arranged on all four sides
        """
        pads = []
        net_id = 0
        
        # Bottom side (left to right)
        if num_pads_per_side > 0:
            spacing = (die_width - 2 * margin) / max(1, num_pads_per_side - 1)
            for i in range(num_pads_per_side):
                x = margin + i * spacing if num_pads_per_side > 1 else die_width / 2
                pads.append(IOPad(
                    x=x,
                    y=margin,
                    net_id=net_id,
                    name=f"io_bottom_{i}",
                    width=pad_width,
                    height=pad_height,
                    side="bottom"
                ))
                net_id += 1
        
        # Right side (bottom to top)
        if num_pads_per_side > 0:
            spacing = (die_height - 2 * margin) / max(1, num_pads_per_side - 1)
            for i in range(num_pads_per_side):
                y = margin + i * spacing if num_pads_per_side > 1 else die_height / 2
                pads.append(IOPad(
                    x=die_width - margin,
                    y=y,
                    net_id=net_id,
                    name=f"io_right_{i}",
                    width=pad_width,
                    height=pad_height,
                    side="right"
                ))
                net_id += 1
        
        # Top side (right to left)
        if num_pads_per_side > 0:
            spacing = (die_width - 2 * margin) / max(1, num_pads_per_side - 1)
            for i in range(num_pads_per_side):
                x = die_width - margin - i * spacing if num_pads_per_side > 1 else die_width / 2
                pads.append(IOPad(
                    x=x,
                    y=die_height - margin,
                    net_id=net_id,
                    name=f"io_top_{i}",
                    width=pad_width,
                    height=pad_height,
                    side="top"
                ))
                net_id += 1
        
        # Left side (top to bottom)
        if num_pads_per_side > 0:
            spacing = (die_height - 2 * margin) / max(1, num_pads_per_side - 1)
            for i in range(num_pads_per_side):
                y = die_height - margin - i * spacing if num_pads_per_side > 1 else die_height / 2
                pads.append(IOPad(
                    x=margin,
                    y=y,
                    net_id=net_id,
                    name=f"io_left_{i}",
                    width=pad_width,
                    height=pad_height,
                    side="left"
                ))
                net_id += 1
        
        return pads
    
    def to_dict(self) -> dict:
        """
        Convert IO pad to dictionary for serialization.
        
        Returns:
            Dictionary representation of the IO pad
        """
        return {
            'x': self.x,
            'y': self.y,
            'net_id': self.net_id,
            'name': self.name,
            'width': self.width,
            'height': self.height,
            'node_id': self.node_id,
            'side': self.side,
            'layer': self.layer,
            'is_power': self.is_power,
            'is_routed': self.is_routed,
            'assigned_bump_id': self.assigned_bump.node_id if self.assigned_bump else None,
            'route_path': self.route_path
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'IOPad':
        """
        Create IO pad from dictionary.
        
        Args:
            data: Dictionary with IO pad attributes
            
        Returns:
            New IOPad instance
        """
        pad = cls(
            x=data['x'],
            y=data['y'],
            net_id=data.get('net_id', -1),
            name=data.get('name', ''),
            width=data.get('width', 50.0),
            height=data.get('height', 50.0),
            node_id=data.get('node_id'),
            side=data.get('side', 'bottom'),
            layer=data.get('layer', 1),
            is_power=data.get('is_power', False)
        )
        pad.is_routed = data.get('is_routed', False)
        pad.route_path = data.get('route_path', [])
        return pad
    
    def __repr__(self) -> str:
        """String representation of the IO pad."""
        assigned = f"->bump_{self.assigned_bump.node_id}" if self.assigned_bump else ""
        return (
            f"IOPad(name='{self.name}', pos=({self.x:.1f}, {self.y:.1f}), "
            f"net={self.net_id}, side='{self.side}'{assigned})"
        )
    
    def __eq__(self, other: object) -> bool:
        """Check equality based on node_id."""
        if not isinstance(other, IOPad):
            return False
        return self.node_id == other.node_id
    
    def __hash__(self) -> int:
        """Hash based on node_id for use in sets/dicts."""
        return hash(self.node_id)
