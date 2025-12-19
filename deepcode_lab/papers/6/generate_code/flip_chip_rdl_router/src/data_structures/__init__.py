"""
Data structures package for flip-chip RDL routing.

This package provides the core data structures used throughout the routing pipeline:
- Chip: Main container for flip-chip package layout
- BumpPad: Represents bump pad (solder ball) connection points
- IOPad: Represents IO buffer pads on die periphery
- Net: Represents electrical connections between pads

Example usage:
    from src.data_structures import Chip, BumpPad, IOPad, Net
    
    # Create a chip
    chip = Chip(name="test_chip", die_width=5000, die_height=5000)
    
    # Add bump pads
    chip.create_bump_grid(rows=10, cols=10, pitch=100)
    
    # Add IO pads
    chip.create_peripheral_io_pads(num_pads_per_side=25)
"""

from .bump_pad import BumpPad
from .io_pad import IOPad
from .net import Net
from .chip import Chip

__all__ = ['Chip', 'BumpPad', 'IOPad', 'Net']

# Package version
__version__ = '1.0.0'
