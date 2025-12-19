"""
Chip class representing the flip-chip package layout.

This module defines the Chip class which serves as the main container for
all routing elements including IO pads, bump pads, nets, and RDL layer
configuration. It provides methods for chip initialization, validation,
and routing statistics.
"""

from typing import List, Optional, Tuple, Dict, Any
from .bump_pad import BumpPad
from .io_pad import IOPad
from .net import Net


class Chip:
    """
    Represents a flip-chip package layout for RDL routing.
    
    The Chip class is the main container that holds:
    - IO pads on the die periphery (signal sources)
    - Bump pads in a grid array (routing targets)
    - Nets connecting IO pads to bump pads
    - RDL layer configuration and design rules
    
    Attributes:
        name (str): Chip identifier name
        die_width (float): Die width in micrometers
        die_height (float): Die height in micrometers
        io_pads (List[IOPad]): List of IO pads on die periphery
        bump_pads (List[BumpPad]): List of bump pads (solder balls)
        nets (List[Net]): List of nets to be routed
        rdl_layers (int): Number of RDL layers available
        bump_pitch (float): Pitch between bump pads in micrometers
        wire_width (float): Default wire width in micrometers
        wire_spacing (float): Minimum wire spacing in micrometers
    """
    
    def __init__(
        self,
        name: str = "chip",
        die_width: float = 10000.0,
        die_height: float = 10000.0,
        rdl_layers: int = 1,
        bump_pitch: float = 100.0,
        bump_diameter: float = 80.0,
        wire_width: float = 10.0,
        wire_spacing: float = 10.0,
        io_pad_width: float = 50.0,
        io_pad_height: float = 50.0
    ):
        """
        Initialize a Chip instance.
        
        Args:
            name: Chip identifier name
            die_width: Die width in micrometers
            die_height: Die height in micrometers
            rdl_layers: Number of RDL layers (default 1 for single-layer routing)
            bump_pitch: Pitch between bump pads in micrometers
            bump_diameter: Diameter of bump pads in micrometers
            wire_width: Default wire width in micrometers
            wire_spacing: Minimum wire spacing in micrometers
            io_pad_width: Default IO pad width in micrometers
            io_pad_height: Default IO pad height in micrometers
        """
        self.name = name
        self.die_width = die_width
        self.die_height = die_height
        self.rdl_layers = rdl_layers
        self.bump_pitch = bump_pitch
        self.bump_diameter = bump_diameter
        self.wire_width = wire_width
        self.wire_spacing = wire_spacing
        self.io_pad_width = io_pad_width
        self.io_pad_height = io_pad_height
        
        # Collections
        self.io_pads: List[IOPad] = []
        self.bump_pads: List[BumpPad] = []
        self.nets: List[Net] = []
        
        # Routing grid parameters
        self.grid_origin_x: float = 0.0
        self.grid_origin_y: float = 0.0
        self.grid_step_x: float = bump_pitch
        self.grid_step_y: float = bump_pitch
        
        # Routing results
        self.routed_nets: List[Net] = []
        self.unrouted_nets: List[Net] = []
        self.total_wirelength: float = 0.0
        
        # Statistics
        self._routing_complete: bool = False
    
    @property
    def num_io_pads(self) -> int:
        """Return the number of IO pads."""
        return len(self.io_pads)
    
    @property
    def num_bump_pads(self) -> int:
        """Return the number of bump pads."""
        return len(self.bump_pads)
    
    @property
    def num_nets(self) -> int:
        """Return the number of nets."""
        return len(self.nets)
    
    @property
    def die_area(self) -> float:
        """Return the die area in square micrometers."""
        return self.die_width * self.die_height
    
    @property
    def total_bump_capacity(self) -> int:
        """Return the total capacity of all bump pads."""
        return sum(bump.capacity for bump in self.bump_pads)
    
    @property
    def available_bump_capacity(self) -> int:
        """Return the remaining available capacity of all bump pads."""
        return sum(bump.remaining_capacity for bump in self.bump_pads)
    
    @property
    def is_routable(self) -> bool:
        """
        Check if the chip configuration is potentially routable.
        
        Returns:
            True if total bump capacity >= number of IO pads
        """
        return self.total_bump_capacity >= self.num_io_pads
    
    @property
    def routing_complete(self) -> bool:
        """Return whether routing has been completed."""
        return self._routing_complete
    
    @property
    def routability_rate(self) -> float:
        """
        Calculate the routability rate (percentage of routed nets).
        
        Returns:
            Percentage of successfully routed nets (0-100)
        """
        if self.num_nets == 0:
            return 0.0
        return (len(self.routed_nets) / self.num_nets) * 100.0
    
    def add_io_pad(self, io_pad: IOPad) -> None:
        """
        Add an IO pad to the chip.
        
        Args:
            io_pad: IOPad instance to add
        """
        self.io_pads.append(io_pad)
    
    def add_bump_pad(self, bump_pad: BumpPad) -> None:
        """
        Add a bump pad to the chip.
        
        Args:
            bump_pad: BumpPad instance to add
        """
        self.bump_pads.append(bump_pad)
    
    def add_net(self, net: Net) -> None:
        """
        Add a net to the chip.
        
        Args:
            net: Net instance to add
        """
        self.nets.append(net)
    
    def remove_io_pad(self, io_pad: IOPad) -> bool:
        """
        Remove an IO pad from the chip.
        
        Args:
            io_pad: IOPad instance to remove
            
        Returns:
            True if removed successfully, False if not found
        """
        if io_pad in self.io_pads:
            self.io_pads.remove(io_pad)
            return True
        return False
    
    def remove_bump_pad(self, bump_pad: BumpPad) -> bool:
        """
        Remove a bump pad from the chip.
        
        Args:
            bump_pad: BumpPad instance to remove
            
        Returns:
            True if removed successfully, False if not found
        """
        if bump_pad in self.bump_pads:
            self.bump_pads.remove(bump_pad)
            return True
        return False
    
    def remove_net(self, net: Net) -> bool:
        """
        Remove a net from the chip.
        
        Args:
            net: Net instance to remove
            
        Returns:
            True if removed successfully, False if not found
        """
        if net in self.nets:
            self.nets.remove(net)
            return True
        return False
    
    def get_io_pad_by_name(self, name: str) -> Optional[IOPad]:
        """
        Find an IO pad by name.
        
        Args:
            name: Name of the IO pad to find
            
        Returns:
            IOPad instance if found, None otherwise
        """
        for io_pad in self.io_pads:
            if io_pad.name == name:
                return io_pad
        return None
    
    def get_io_pad_by_node_id(self, node_id: int) -> Optional[IOPad]:
        """
        Find an IO pad by node ID.
        
        Args:
            node_id: Node ID of the IO pad to find
            
        Returns:
            IOPad instance if found, None otherwise
        """
        for io_pad in self.io_pads:
            if io_pad.node_id == node_id:
                return io_pad
        return None
    
    def get_bump_pad_by_name(self, name: str) -> Optional[BumpPad]:
        """
        Find a bump pad by name.
        
        Args:
            name: Name of the bump pad to find
            
        Returns:
            BumpPad instance if found, None otherwise
        """
        for bump_pad in self.bump_pads:
            if bump_pad.name == name:
                return bump_pad
        return None
    
    def get_bump_pad_by_node_id(self, node_id: int) -> Optional[BumpPad]:
        """
        Find a bump pad by node ID.
        
        Args:
            node_id: Node ID of the bump pad to find
            
        Returns:
            BumpPad instance if found, None otherwise
        """
        for bump_pad in self.bump_pads:
            if bump_pad.node_id == node_id:
                return bump_pad
        return None
    
    def get_bump_pad_at_grid(self, row: int, col: int) -> Optional[BumpPad]:
        """
        Find a bump pad at a specific grid position.
        
        Args:
            row: Grid row index
            col: Grid column index
            
        Returns:
            BumpPad instance if found, None otherwise
        """
        for bump_pad in self.bump_pads:
            grid_pos = bump_pad.get_grid_position(self.bump_pitch, 
                                                   self.grid_origin_x, 
                                                   self.grid_origin_y)
            if grid_pos == (row, col):
                return bump_pad
        return None
    
    def get_net_by_id(self, net_id: int) -> Optional[Net]:
        """
        Find a net by ID.
        
        Args:
            net_id: ID of the net to find
            
        Returns:
            Net instance if found, None otherwise
        """
        for net in self.nets:
            if net.net_id == net_id:
                return net
        return None
    
    def get_net_by_name(self, name: str) -> Optional[Net]:
        """
        Find a net by name.
        
        Args:
            name: Name of the net to find
            
        Returns:
            Net instance if found, None otherwise
        """
        for net in self.nets:
            if net.name == name:
                return net
        return None
    
    def get_io_pads_by_side(self, side: str) -> List[IOPad]:
        """
        Get all IO pads on a specific side of the die.
        
        Args:
            side: Side name ('top', 'bottom', 'left', 'right')
            
        Returns:
            List of IO pads on the specified side
        """
        return [io_pad for io_pad in self.io_pads if io_pad.side == side]
    
    def get_available_bump_pads(self) -> List[BumpPad]:
        """
        Get all bump pads with remaining capacity.
        
        Returns:
            List of bump pads that can accept more IO assignments
        """
        return [bump for bump in self.bump_pads if bump.is_available]
    
    def get_assigned_io_pads(self) -> List[IOPad]:
        """
        Get all IO pads that have been assigned to bump pads.
        
        Returns:
            List of assigned IO pads
        """
        return [io_pad for io_pad in self.io_pads if io_pad.is_assigned]
    
    def get_unassigned_io_pads(self) -> List[IOPad]:
        """
        Get all IO pads that have not been assigned to bump pads.
        
        Returns:
            List of unassigned IO pads
        """
        return [io_pad for io_pad in self.io_pads if not io_pad.is_assigned]
    
    def create_bump_grid(
        self,
        rows: int,
        cols: int,
        origin_x: Optional[float] = None,
        origin_y: Optional[float] = None,
        capacity: int = 1
    ) -> List[BumpPad]:
        """
        Create a grid of bump pads and add them to the chip.
        
        Args:
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            origin_x: X coordinate of grid origin (default: centered)
            origin_y: Y coordinate of grid origin (default: centered)
            capacity: Capacity of each bump pad
            
        Returns:
            List of created BumpPad instances
        """
        # Calculate centered origin if not specified
        if origin_x is None:
            grid_width = (cols - 1) * self.bump_pitch
            origin_x = (self.die_width - grid_width) / 2
        if origin_y is None:
            grid_height = (rows - 1) * self.bump_pitch
            origin_y = (self.die_height - grid_height) / 2
        
        self.grid_origin_x = origin_x
        self.grid_origin_y = origin_y
        
        # Create bump pads using BumpPad's factory method
        new_bumps = BumpPad.create_grid(
            rows=rows,
            cols=cols,
            pitch=self.bump_pitch,
            origin_x=origin_x,
            origin_y=origin_y,
            diameter=self.bump_diameter,
            capacity=capacity
        )
        
        # Add to chip
        for bump in new_bumps:
            self.bump_pads.append(bump)
        
        return new_bumps
    
    def create_peripheral_io_pads(
        self,
        num_pads_per_side: int,
        margin: float = 100.0,
        sides: Optional[List[str]] = None
    ) -> List[IOPad]:
        """
        Create IO pads around the die periphery.
        
        Args:
            num_pads_per_side: Number of IO pads per side
            margin: Margin from die edge in micrometers
            sides: List of sides to place pads ('top', 'bottom', 'left', 'right')
                   Default is all four sides
            
        Returns:
            List of created IOPad instances
        """
        if sides is None:
            sides = ['top', 'bottom', 'left', 'right']
        
        new_io_pads = IOPad.create_peripheral_pads(
            die_width=self.die_width,
            die_height=self.die_height,
            num_pads_per_side=num_pads_per_side,
            margin=margin,
            pad_width=self.io_pad_width,
            pad_height=self.io_pad_height
        )
        
        # Add to chip
        for io_pad in new_io_pads:
            self.io_pads.append(io_pad)
        
        return new_io_pads
    
    def create_nets_from_assignments(self) -> List[Net]:
        """
        Create nets based on IO pad to bump pad assignments.
        
        This method creates a net for each assigned IO pad.
        
        Returns:
            List of created Net instances
        """
        new_nets = []
        for io_pad in self.io_pads:
            if io_pad.is_assigned:
                net = Net(
                    name=f"net_{io_pad.name}",
                    wire_width=self.wire_width,
                    wire_spacing=self.wire_spacing
                )
                net.add_io_pad(io_pad)
                net.add_bump_pad(io_pad.assigned_bump)
                self.nets.append(net)
                new_nets.append(net)
        return new_nets
    
    def clear_routing(self) -> None:
        """Clear all routing results and assignments."""
        # Clear IO pad assignments
        for io_pad in self.io_pads:
            io_pad.unassign()
        
        # Clear bump pad assignments
        for bump_pad in self.bump_pads:
            bump_pad.assigned_ios.clear()
        
        # Clear net routes
        for net in self.nets:
            net.clear_route()
        
        # Reset routing state
        self.routed_nets.clear()
        self.unrouted_nets.clear()
        self.total_wirelength = 0.0
        self._routing_complete = False
    
    def calculate_total_wirelength(self) -> float:
        """
        Calculate the total wirelength of all routed nets.
        
        Returns:
            Total wirelength in micrometers
        """
        total = 0.0
        for net in self.routed_nets:
            total += net.wirelength
        self.total_wirelength = total
        return total
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive routing statistics.
        
        Returns:
            Dictionary containing routing statistics
        """
        stats = {
            'chip_name': self.name,
            'die_width': self.die_width,
            'die_height': self.die_height,
            'die_area': self.die_area,
            'num_io_pads': self.num_io_pads,
            'num_bump_pads': self.num_bump_pads,
            'num_nets': self.num_nets,
            'total_bump_capacity': self.total_bump_capacity,
            'available_bump_capacity': self.available_bump_capacity,
            'is_routable': self.is_routable,
            'routing_complete': self._routing_complete,
            'num_routed_nets': len(self.routed_nets),
            'num_unrouted_nets': len(self.unrouted_nets),
            'routability_rate': self.routability_rate,
            'total_wirelength': self.total_wirelength,
            'rdl_layers': self.rdl_layers,
            'bump_pitch': self.bump_pitch,
            'wire_width': self.wire_width,
            'wire_spacing': self.wire_spacing
        }
        
        # Calculate average wirelength per net
        if len(self.routed_nets) > 0:
            stats['avg_wirelength_per_net'] = self.total_wirelength / len(self.routed_nets)
        else:
            stats['avg_wirelength_per_net'] = 0.0
        
        # Count total bends
        total_bends = sum(net.num_bends for net in self.routed_nets)
        stats['total_bends'] = total_bends
        
        return stats
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the chip configuration.
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        # Check die dimensions
        if self.die_width <= 0:
            errors.append("Die width must be positive")
        if self.die_height <= 0:
            errors.append("Die height must be positive")
        
        # Check bump pitch
        if self.bump_pitch <= 0:
            errors.append("Bump pitch must be positive")
        if self.bump_pitch < self.bump_diameter:
            errors.append("Bump pitch must be >= bump diameter")
        
        # Check wire rules
        if self.wire_width <= 0:
            errors.append("Wire width must be positive")
        if self.wire_spacing < 0:
            errors.append("Wire spacing must be non-negative")
        
        # Check IO pads are within die bounds
        for io_pad in self.io_pads:
            if io_pad.x < 0 or io_pad.x > self.die_width:
                errors.append(f"IO pad {io_pad.name} x-coordinate out of bounds")
            if io_pad.y < 0 or io_pad.y > self.die_height:
                errors.append(f"IO pad {io_pad.name} y-coordinate out of bounds")
        
        # Check bump pads are within die bounds
        for bump_pad in self.bump_pads:
            if bump_pad.x < 0 or bump_pad.x > self.die_width:
                errors.append(f"Bump pad {bump_pad.name} x-coordinate out of bounds")
            if bump_pad.y < 0 or bump_pad.y > self.die_height:
                errors.append(f"Bump pad {bump_pad.name} y-coordinate out of bounds")
        
        # Check routability
        if not self.is_routable:
            errors.append(f"Insufficient bump capacity: {self.total_bump_capacity} < {self.num_io_pads}")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert chip to dictionary representation.
        
        Returns:
            Dictionary containing chip data
        """
        return {
            'name': self.name,
            'die_width': self.die_width,
            'die_height': self.die_height,
            'rdl_layers': self.rdl_layers,
            'bump_pitch': self.bump_pitch,
            'bump_diameter': self.bump_diameter,
            'wire_width': self.wire_width,
            'wire_spacing': self.wire_spacing,
            'io_pad_width': self.io_pad_width,
            'io_pad_height': self.io_pad_height,
            'grid_origin_x': self.grid_origin_x,
            'grid_origin_y': self.grid_origin_y,
            'io_pads': [io_pad.to_dict() for io_pad in self.io_pads],
            'bump_pads': [bump_pad.to_dict() for bump_pad in self.bump_pads],
            'nets': [net.to_dict() for net in self.nets]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chip':
        """
        Create a Chip instance from dictionary representation.
        
        Args:
            data: Dictionary containing chip data
            
        Returns:
            New Chip instance
        """
        chip = cls(
            name=data.get('name', 'chip'),
            die_width=data.get('die_width', 10000.0),
            die_height=data.get('die_height', 10000.0),
            rdl_layers=data.get('rdl_layers', 1),
            bump_pitch=data.get('bump_pitch', 100.0),
            bump_diameter=data.get('bump_diameter', 80.0),
            wire_width=data.get('wire_width', 10.0),
            wire_spacing=data.get('wire_spacing', 10.0),
            io_pad_width=data.get('io_pad_width', 50.0),
            io_pad_height=data.get('io_pad_height', 50.0)
        )
        
        chip.grid_origin_x = data.get('grid_origin_x', 0.0)
        chip.grid_origin_y = data.get('grid_origin_y', 0.0)
        
        # Load IO pads
        for io_data in data.get('io_pads', []):
            io_pad = IOPad.from_dict(io_data)
            chip.io_pads.append(io_pad)
        
        # Load bump pads
        for bump_data in data.get('bump_pads', []):
            bump_pad = BumpPad.from_dict(bump_data)
            chip.bump_pads.append(bump_pad)
        
        # Load nets
        for net_data in data.get('nets', []):
            net = Net.from_dict(net_data)
            chip.nets.append(net)
        
        return chip
    
    def __repr__(self) -> str:
        """Return string representation of the chip."""
        return (f"Chip(name='{self.name}', "
                f"die={self.die_width}x{self.die_height}, "
                f"io_pads={self.num_io_pads}, "
                f"bump_pads={self.num_bump_pads}, "
                f"nets={self.num_nets})")
    
    def __str__(self) -> str:
        """Return human-readable string representation."""
        return (f"Chip '{self.name}':\n"
                f"  Die size: {self.die_width} x {self.die_height} um\n"
                f"  IO pads: {self.num_io_pads}\n"
                f"  Bump pads: {self.num_bump_pads}\n"
                f"  Nets: {self.num_nets}\n"
                f"  RDL layers: {self.rdl_layers}\n"
                f"  Bump pitch: {self.bump_pitch} um")
