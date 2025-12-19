"""
Bump Pad Data Structure for Flip-Chip RDL Routing.

This module defines the BumpPad class representing bump pads (solder balls)
in flip-chip package design. Bump pads serve as connection points between
the die and the package substrate.
"""

from typing import List, Optional, Tuple, Any


class BumpPad:
    """
    Represents a bump pad (solder ball) in flip-chip package design.
    
    Bump pads are arranged in a grid pattern on the die surface and serve
    as connection points for IO pads through the RDL (Redistribution Layer).
    
    Attributes:
        x (float): X-coordinate of bump pad center (in micrometers)
        y (float): Y-coordinate of bump pad center (in micrometers)
        capacity (int): Maximum number of IO pads that can connect to this bump
        assigned_ios (List): List of IO pads assigned to this bump pad
        node_id (int): Unique identifier for flow network representation
        name (str): Optional name/identifier for the bump pad
        diameter (float): Bump pad diameter (in micrometers)
        pitch (float): Pitch between adjacent bump pads (in micrometers)
        layer (int): RDL layer assignment (0 = unassigned)
        net_id (int): Net identifier if bump is part of a specific net
        is_power (bool): True if this is a power/ground bump
        is_blocked (bool): True if bump is blocked/unavailable for routing
    """
    
    # Class-level counter for automatic node ID assignment
    _node_counter = 0
    
    def __init__(
        self,
        x: float,
        y: float,
        capacity: int = 1,
        name: str = "",
        diameter: float = 80.0,
        pitch: float = 100.0,
        node_id: Optional[int] = None,
        net_id: int = -1,
        is_power: bool = False,
        is_blocked: bool = False
    ):
        """
        Initialize a BumpPad instance.
        
        Args:
            x: X-coordinate of bump pad center (micrometers)
            y: Y-coordinate of bump pad center (micrometers)
            capacity: Maximum number of IO connections (default: 1)
            name: Optional identifier string
            diameter: Bump pad diameter in micrometers (default: 80um)
            pitch: Pitch between bumps in micrometers (default: 100um)
            node_id: Unique node ID for flow network (auto-assigned if None)
            net_id: Net identifier (-1 if unassigned)
            is_power: Whether this is a power/ground bump
            is_blocked: Whether bump is blocked for routing
        """
        self.x = float(x)
        self.y = float(y)
        self.capacity = capacity
        self.name = name if name else f"bump_{BumpPad._node_counter}"
        self.diameter = diameter
        self.pitch = pitch
        self.net_id = net_id
        self.is_power = is_power
        self.is_blocked = is_blocked
        
        # Assigned IO pads list
        self.assigned_ios: List[Any] = []
        
        # Layer assignment (0 = unassigned, 1+ = RDL layer number)
        self.layer = 0
        
        # Node ID for flow network representation
        if node_id is not None:
            self.node_id = node_id
        else:
            self.node_id = BumpPad._node_counter
            BumpPad._node_counter += 1
    
    @property
    def pos(self) -> Tuple[float, float]:
        """Return position as (x, y) tuple."""
        return (self.x, self.y)
    
    @property
    def position(self) -> Tuple[float, float]:
        """Alias for pos property."""
        return self.pos
    
    @property
    def remaining_capacity(self) -> int:
        """Return remaining capacity for IO assignments."""
        return max(0, self.capacity - len(self.assigned_ios))
    
    @property
    def is_full(self) -> bool:
        """Check if bump pad has reached its capacity."""
        return len(self.assigned_ios) >= self.capacity
    
    @property
    def is_available(self) -> bool:
        """Check if bump pad is available for new assignments."""
        return not self.is_blocked and not self.is_full
    
    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """
        Return bounding box of bump pad.
        
        Returns:
            Tuple of (x_min, y_min, x_max, y_max)
        """
        half_d = self.diameter / 2.0
        return (
            self.x - half_d,
            self.y - half_d,
            self.x + half_d,
            self.y + half_d
        )
    
    def assign_io(self, io_pad: Any) -> bool:
        """
        Assign an IO pad to this bump pad.
        
        Args:
            io_pad: The IO pad to assign
            
        Returns:
            True if assignment successful, False if at capacity or blocked
        """
        if self.is_blocked:
            return False
        if len(self.assigned_ios) >= self.capacity:
            return False
        if io_pad in self.assigned_ios:
            return False  # Already assigned
        
        self.assigned_ios.append(io_pad)
        return True
    
    def unassign_io(self, io_pad: Any) -> bool:
        """
        Remove an IO pad assignment from this bump pad.
        
        Args:
            io_pad: The IO pad to unassign
            
        Returns:
            True if unassignment successful, False if IO was not assigned
        """
        if io_pad in self.assigned_ios:
            self.assigned_ios.remove(io_pad)
            return True
        return False
    
    def clear_assignments(self) -> None:
        """Clear all IO pad assignments."""
        self.assigned_ios.clear()
    
    def manhattan_distance(self, other: 'BumpPad') -> float:
        """
        Calculate Manhattan distance to another bump pad.
        
        Args:
            other: Another BumpPad instance
            
        Returns:
            Manhattan distance in micrometers
        """
        return abs(self.x - other.x) + abs(self.y - other.y)
    
    def euclidean_distance(self, other: 'BumpPad') -> float:
        """
        Calculate Euclidean distance to another bump pad.
        
        Args:
            other: Another BumpPad instance
            
        Returns:
            Euclidean distance in micrometers
        """
        import math
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def distance_to_point(self, x: float, y: float) -> float:
        """
        Calculate Manhattan distance to a point.
        
        Args:
            x: X-coordinate of point
            y: Y-coordinate of point
            
        Returns:
            Manhattan distance in micrometers
        """
        return abs(self.x - x) + abs(self.y - y)
    
    def contains_point(self, x: float, y: float) -> bool:
        """
        Check if a point is within the bump pad area.
        
        Args:
            x: X-coordinate of point
            y: Y-coordinate of point
            
        Returns:
            True if point is within bump pad diameter
        """
        import math
        dist = math.sqrt((self.x - x)**2 + (self.y - y)**2)
        return dist <= self.diameter / 2.0
    
    def overlaps(self, other: 'BumpPad') -> bool:
        """
        Check if this bump pad overlaps with another.
        
        Args:
            other: Another BumpPad instance
            
        Returns:
            True if bump pads overlap
        """
        min_dist = (self.diameter + other.diameter) / 2.0
        return self.euclidean_distance(other) < min_dist
    
    def get_grid_position(self, origin_x: float = 0, origin_y: float = 0) -> Tuple[int, int]:
        """
        Get grid position based on pitch.
        
        Args:
            origin_x: X-coordinate of grid origin
            origin_y: Y-coordinate of grid origin
            
        Returns:
            Tuple of (row, column) indices
        """
        col = int(round((self.x - origin_x) / self.pitch))
        row = int(round((self.y - origin_y) / self.pitch))
        return (row, col)
    
    def copy(self) -> 'BumpPad':
        """
        Create a copy of this bump pad.
        
        Returns:
            New BumpPad instance with same attributes
        """
        new_bump = BumpPad(
            x=self.x,
            y=self.y,
            capacity=self.capacity,
            name=self.name + "_copy",
            diameter=self.diameter,
            pitch=self.pitch,
            node_id=None,  # Get new node ID
            net_id=self.net_id,
            is_power=self.is_power,
            is_blocked=self.is_blocked
        )
        new_bump.layer = self.layer
        return new_bump
    
    @classmethod
    def reset_node_counter(cls) -> None:
        """Reset the class-level node counter (useful for testing)."""
        cls._node_counter = 0
    
    @classmethod
    def create_grid(
        cls,
        rows: int,
        cols: int,
        pitch: float = 100.0,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
        diameter: float = 80.0,
        capacity: int = 1
    ) -> List['BumpPad']:
        """
        Create a grid of bump pads.
        
        Args:
            rows: Number of rows in grid
            cols: Number of columns in grid
            pitch: Spacing between bump centers (micrometers)
            origin_x: X-coordinate of first bump
            origin_y: Y-coordinate of first bump
            diameter: Bump pad diameter (micrometers)
            capacity: Capacity per bump pad
            
        Returns:
            List of BumpPad instances arranged in grid
        """
        bumps = []
        for row in range(rows):
            for col in range(cols):
                x = origin_x + col * pitch
                y = origin_y + row * pitch
                bump = cls(
                    x=x,
                    y=y,
                    capacity=capacity,
                    name=f"bump_r{row}_c{col}",
                    diameter=diameter,
                    pitch=pitch
                )
                bumps.append(bump)
        return bumps
    
    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"BumpPad(name='{self.name}', pos=({self.x:.1f}, {self.y:.1f}), "
            f"capacity={self.capacity}, assigned={len(self.assigned_ios)}, "
            f"node_id={self.node_id})"
        )
    
    def __str__(self) -> str:
        """Return human-readable string."""
        status = "blocked" if self.is_blocked else (
            "full" if self.is_full else f"{self.remaining_capacity} available"
        )
        return f"BumpPad '{self.name}' at ({self.x:.1f}, {self.y:.1f}) [{status}]"
    
    def __eq__(self, other: object) -> bool:
        """Check equality based on node_id."""
        if not isinstance(other, BumpPad):
            return False
        return self.node_id == other.node_id
    
    def __hash__(self) -> int:
        """Hash based on node_id for use in sets/dicts."""
        return hash(self.node_id)
    
    def __lt__(self, other: 'BumpPad') -> bool:
        """Less than comparison for sorting (by position)."""
        if self.x != other.x:
            return self.x < other.x
        return self.y < other.y
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary with bump pad attributes
        """
        return {
            'name': self.name,
            'x': self.x,
            'y': self.y,
            'capacity': self.capacity,
            'diameter': self.diameter,
            'pitch': self.pitch,
            'node_id': self.node_id,
            'net_id': self.net_id,
            'is_power': self.is_power,
            'is_blocked': self.is_blocked,
            'layer': self.layer,
            'assigned_count': len(self.assigned_ios)
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'BumpPad':
        """
        Create BumpPad from dictionary.
        
        Args:
            data: Dictionary with bump pad attributes
            
        Returns:
            New BumpPad instance
        """
        bump = cls(
            x=data['x'],
            y=data['y'],
            capacity=data.get('capacity', 1),
            name=data.get('name', ''),
            diameter=data.get('diameter', 80.0),
            pitch=data.get('pitch', 100.0),
            node_id=data.get('node_id'),
            net_id=data.get('net_id', -1),
            is_power=data.get('is_power', False),
            is_blocked=data.get('is_blocked', False)
        )
        bump.layer = data.get('layer', 0)
        return bump
