"""
Utility modules for flip-chip RDL routing.

This package provides utility functions and classes for:
- Input file parsing (YAML, JSON, text, DEF formats)
- Routing visualization and output generation
- Benchmark generation for testing

Modules:
    parser: Input file parsing and benchmark generation
    visualizer: Routing visualization and output generation
"""

from .parser import (
    ChipParser,
    ParseResult,
    BenchmarkGenerator,
    load_chip,
    save_chip_to_yaml,
    save_chip_to_json,
)

# Visualizer will be imported when implemented
# from .visualizer import (
#     RoutingVisualizer,
#     visualize_routing,
#     save_routing_image,
# )

__all__ = [
    # Parser exports
    'ChipParser',
    'ParseResult',
    'BenchmarkGenerator',
    'load_chip',
    'save_chip_to_yaml',
    'save_chip_to_json',
    # Visualizer exports (to be added)
    # 'RoutingVisualizer',
    # 'visualize_routing',
    # 'save_routing_image',
]

__version__ = '1.0.0'
