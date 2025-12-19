# Flip-Chip RDL Router

A Network-Flow-Based RDL Routing Algorithm for Flip-Chip Design

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project implements a two-stage routing algorithm for flip-chip package design:

1. **Global Routing**: Uses Minimum-Cost Maximum-Flow (MCMF) algorithm to optimally assign IO pads to bump pads while minimizing total wirelength
2. **Detailed Routing**: Employs river routing methodology for single-layer RDL (Redistribution Layer) interconnection with planarity constraints

The algorithm ensures 100% routability on standard configurations while minimizing total wirelength.

## Features

- **MCMF-based Global Routing**: Optimal IO-to-bump assignment using successive shortest path algorithm
- **River Routing**: Planar detailed routing with no wire crossings
- **Net Ordering**: Topological sort-based ordering to ensure crossing-free routing
- **DRC Checking**: Design rule checking for spacing, width, and boundary violations
- **Multi-layer Support**: Layer assignment for multi-layer RDL configurations
- **Visualization**: Routing result visualization with matplotlib
- **Benchmarking**: Comprehensive experiment framework with scalability analysis

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone or navigate to the project directory
cd flip_chip_rdl_router

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

**Core Dependencies:**
- `numpy>=1.21.0` - Matrix operations for flow network
- `scipy>=1.7.0` - Sparse matrix support
- `networkx>=2.6` - Graph algorithms (optional, for validation)
- `matplotlib>=3.4.0` - Visualization
- `pyyaml>=5.4` - Configuration parsing
- `pytest>=6.2.0` - Testing framework

## Quick Start

### Command Line Interface

```bash
# Run demonstration
python main.py --demo

# Run all benchmarks
python main.py --benchmark all

# Run specific benchmark
python main.py --benchmark medium

# Route from configuration file
python main.py --input config.yaml --output results/

# Interactive mode
python main.py --interactive
```

### Python API

```python
from src import Chip, full_routing_pipeline, quick_route

# Create a simple chip
chip = Chip(
    name="test_chip",
    die_width=5000,  # micrometers
    die_height=5000,
    bump_pitch=200
)

# Create bump pad grid (10x10)
chip.create_bump_grid(rows=10, cols=10, pitch=200)

# Create peripheral IO pads
chip.create_peripheral_io_pads(num_pads_per_side=10)

# Run full routing pipeline
results = full_routing_pipeline(
    chip,
    wire_width=10.0,
    wire_spacing=10.0,
    enable_drc=True
)

# Check results
print(f"Routing success: {results['success']}")
print(f"Total wirelength: {results['total_wirelength']:.2f} um")
print(f"Routability: {results['routability_rate']*100:.1f}%")
```

### Quick Route (Simplified API)

```python
from src import create_simple_chip, route_chip

# Create chip with helper function
chip = create_simple_chip(
    name="quick_test",
    die_size=3000,
    bump_grid=(8, 8),
    io_per_side=8,
    bump_pitch=150
)

# Route with defaults
results = route_chip(chip)
print(f"Success: {results['success']}")
```

## Project Structure

```
flip_chip_rdl_router/
├── src/
│   ├── core/                    # Core algorithms
│   │   ├── mcmf.py             # Min-cost max-flow solver
│   │   ├── network_builder.py  # Flow network construction
│   │   ├── river_router.py     # River routing algorithm
│   │   └── geometry.py         # Geometric primitives
│   ├── routing/                 # Routing modules
│   │   ├── global_router.py    # MCMF-based global routing
│   │   ├── detailed_router.py  # Detailed routing with DRC
│   │   ├── net_ordering.py     # Net ordering for planarity
│   │   └── layer_assignment.py # Multi-layer assignment
│   ├── data_structures/         # Data models
│   │   ├── chip.py             # Chip/package representation
│   │   ├── bump_pad.py         # Bump pad model
│   │   ├── io_pad.py           # IO pad model
│   │   └── net.py              # Net/wire model
│   └── utils/                   # Utilities
│       ├── parser.py           # Configuration parser
│       └── visualizer.py       # Routing visualization
├── experiments/
│   ├── run_experiments.py      # Experiment runner
│   └── evaluate.py             # Result evaluation
├── main.py                      # CLI entry point
├── config.yaml                  # Default configuration
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Algorithm Details

### Global Routing (MCMF)

The global routing phase constructs a flow network:

```
Source (S) → IO Pads → Bump Pads → Sink (T)
```

- **Source to IO pads**: Capacity = 1, Cost = 0
- **IO pads to Bump pads**: Capacity = 1, Cost = Manhattan distance
- **Bump pads to Sink**: Capacity = bump capacity, Cost = 0

The MCMF algorithm finds the minimum-cost assignment that routes all IOs.

### River Routing

The detailed routing phase uses river routing methodology:

1. **Net Ordering**: Build precedence graph based on crossing constraints
2. **Topological Sort**: Order nets to avoid conflicts
3. **Path Generation**: Generate L-shape or Z-shape Manhattan paths
4. **Planarity Check**: Ensure no wire crossings

### Routing Strategies

| Strategy | Description |
|----------|-------------|
| L-Shape | Simple two-segment path (horizontal then vertical or vice versa) |
| Z-Shape | Three-segment path with intermediate horizontal/vertical segment |
| Auto | Automatically selects best strategy based on obstacles |

## Configuration

### YAML Configuration

```yaml
chip:
  die_width: 5000
  die_height: 5000
  bump_pitch: 200
  bump_diameter: 80

routing:
  wire_width: 10.0
  wire_spacing: 10.0
  enable_drc: true

algorithm:
  mcmf:
    use_spfa: true
  river_routing:
    direction: auto
```

### Programmatic Configuration

```python
from src.routing import GlobalRouter, RoutingConstraints

constraints = RoutingConstraints(
    max_distance=2000,      # Maximum routing distance
    min_distance=50,        # Minimum routing distance
    require_planarity=True  # Enforce no crossings
)

router = GlobalRouter(constraints=constraints)
result = router.route(chip)
```

## Benchmarks

### Standard Benchmarks

| Benchmark | IOs | Bumps | Expected Runtime |
|-----------|-----|-------|------------------|
| small     | 20  | 5×5   | < 1 sec |
| medium    | 60  | 15×15 | < 5 sec |
| large     | 120 | 30×30 | < 30 sec |
| xlarge    | 200 | 50×50 | < 2 min |

### Running Benchmarks

```bash
# Run all benchmarks
python main.py --benchmark all

# Run experiments with analysis
python experiments/run_experiments.py --all

# Run scalability analysis
python experiments/run_experiments.py --scalability

# Evaluate results
python experiments/evaluate.py results/
```

### Expected Results

- **Routability**: 100% on standard configurations
- **Wirelength**: Within 5% of theoretical minimum (lower bound)
- **DRC Violations**: 0 (when DRC enabled)

## API Reference

### Core Classes

#### Chip
```python
class Chip:
    def __init__(self, name, die_width, die_height, rdl_layers=1, bump_pitch=100)
    def create_bump_grid(self, rows, cols, pitch=None, origin_x=None, origin_y=None)
    def create_peripheral_io_pads(self, num_pads_per_side, margin=100)
    def get_routing_statistics(self) -> dict
```

#### GlobalRouter
```python
class GlobalRouter:
    def __init__(self, constraints=None)
    def route(self, chip) -> GlobalRoutingResult
    def route_from_pads(self, io_pads, bump_pads) -> GlobalRoutingResult
```

#### DetailedRouter
```python
class DetailedRouter:
    def __init__(self, wire_width=10.0, wire_spacing=10.0, enable_drc=True)
    def route(self, chip) -> DetailedRoutingResult
    def route_assignments(self, assignments) -> DetailedRoutingResult
```

### Convenience Functions

```python
# Full routing pipeline
full_routing_pipeline(chip, constraints=None, wire_width=10.0, 
                      wire_spacing=10.0, enable_drc=True) -> dict

# Quick route with defaults
quick_route(chip, wire_width=10.0, wire_spacing=10.0) -> tuple

# Create simple chip
create_simple_chip(name, die_size, bump_grid, io_per_side, bump_pitch) -> Chip

# Load chip from file
load_chip(file_path) -> Chip
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test module
pytest tests/test_mcmf.py

# Run quick verification
python main.py --demo
```

## Visualization

```python
from src.utils.visualizer import RoutingVisualizer, visualize_routing

# Create visualizer
viz = RoutingVisualizer(chip)

# Plot routing results
viz.plot_routing(results['detailed_result'])

# Save to file
viz.save("routing_result.png", dpi=300)

# Quick visualization
visualize_routing(chip, results, output_path="result.png")
```

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure you're in the project directory
cd flip_chip_rdl_router
# Add to Python path if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Memory Issues with Large Designs**
```python
# Use sparse MCMF for large designs
from src.core import MCMF
mcmf = MCMF(n_nodes, use_sparse=True)
```

**Routing Failures**
- Check if bump capacity is sufficient for IO count
- Verify IO pads are within die boundaries
- Ensure bump grid covers routing area

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or via CLI
python main.py --demo --verbose
```

## Performance Tips

1. **Large Designs**: Use SPFA optimization (enabled by default)
2. **Memory**: For >1000 IOs, consider sparse matrix representation
3. **Parallelization**: Net ordering can be parallelized for independent groups
4. **Caching**: Reuse flow networks for incremental routing

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- Network flow algorithms: Ahuja, R.K., Magnanti, T.L., Orlin, J.B. "Network Flows: Theory, Algorithms, and Applications"
- River routing: Leiserson, C.E., Pinter, R.Y. "Optimal Placement for River Routing"
- Flip-chip technology: Lau, J.H. "Flip Chip Technologies"

## Acknowledgments

This implementation is based on the network-flow-based RDL routing algorithm for flip-chip design, which combines MCMF optimization with river routing methodology for efficient single-layer redistribution layer routing.

---

**Version**: 1.0.0  
**Author**: Flip-Chip RDL Router Team  
**Contact**: For issues and questions, please open a GitHub issue.
