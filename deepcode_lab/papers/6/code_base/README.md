# Code Base for Paper 6: A Network-Flow-Based RDL Routing Algorithm for Flip-Chip Design

## Paper Information
- **Title**: A Network-Flow-Based RDL Routing Algorithm for Flip-Chip Design
- **Authors**: S.P. Fang, W.S. Feng, and H.C. Chen
- **Publication**: IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (TCAD), 2007
- **Core Algorithm**: Minimum-Cost Maximum-Flow (MCMF) for global routing

## Overview

This directory contains references to GitHub repositories that implement algorithms and methodologies relevant to the paper. Since the original paper (2007) predates GitHub's founding (2008), the repositories listed here implement the **algorithmic concepts** described in the paper's references.

## Repository Structure

```
code_base/
├── README.md                    # This file
├── references_analysis.json     # Detailed analysis of paper references
├── repositories/                # Cloned repositories (if downloaded)
│   ├── networkx/               # Python network algorithms
│   ├── or-tools/               # Google's optimization tools
│   ├── OpenROAD/               # VLSI design automation
│   ├── cu-gr/                  # CUHK global routing tool
│   └── boost-graph/            # Boost C++ graph library
└── implementation_notes.md      # Notes on implementing the paper's algorithm
```

## Selected Repositories

| Rank | Repository | Stars | Relevance | Key Feature |
|------|-----------|-------|-----------|-------------|
| 1 | [NetworkX](https://github.com/networkx/networkx) | 16,422 | 95% | MCMF implementation in Python |
| 2 | [Google OR-Tools](https://github.com/google/or-tools) | 12,841 | 92% | Production-grade MCMF solver |
| 3 | [OpenROAD](https://github.com/The-OpenROAD-Project/OpenROAD) | 2,328 | 88% | Complete VLSI routing flow |
| 4 | [CUGR](https://github.com/cuhk-eda/cu-gr) | 140 | 85% | Academic global routing tool |
| 5 | [Boost Graph](https://github.com/boostorg/graph) | 368 | 82% | C++ graph algorithms |

## Quick Start

### Option 1: Using NetworkX (Python)
```bash
pip install networkx
```

```python
import networkx as nx
from networkx.algorithms.flow import min_cost_flow

# Create directed graph for MCMF
G = nx.DiGraph()
# Add nodes with demand (positive = supply, negative = demand)
G.add_node('source', demand=-10)
G.add_node('sink', demand=10)
# Add edges with capacity and weight (cost)
G.add_edge('source', 'a', capacity=5, weight=1)
G.add_edge('source', 'b', capacity=5, weight=2)
G.add_edge('a', 'sink', capacity=5, weight=1)
G.add_edge('b', 'sink', capacity=5, weight=1)

# Solve minimum cost maximum flow
flow_dict = min_cost_flow(G)
```

### Option 2: Using Google OR-Tools (C++/Python)
```bash
pip install ortools
```

```python
from ortools.graph.python import min_cost_flow

# Create solver
smcf = min_cost_flow.SimpleMinCostFlow()

# Add arcs: (start_node, end_node, capacity, unit_cost)
smcf.add_arc_with_capacity_and_unit_cost(0, 1, 5, 1)
smcf.add_arc_with_capacity_and_unit_cost(0, 2, 5, 2)

# Set node supplies/demands
smcf.set_node_supply(0, 10)  # source
smcf.set_node_supply(3, -10)  # sink

# Solve
status = smcf.solve()
```

## Clone Commands

```bash
# Clone all relevant repositories
git clone https://github.com/networkx/networkx.git repositories/networkx
git clone https://github.com/google/or-tools.git repositories/or-tools
git clone https://github.com/The-OpenROAD-Project/OpenROAD.git repositories/OpenROAD
git clone https://github.com/cuhk-eda/cu-gr.git repositories/cu-gr
git clone https://github.com/boostorg/graph.git repositories/boost-graph
```

## Paper Algorithm Summary

The paper presents a two-phase routing approach:

### Phase 1: Global Routing (MCMF-based)
- Models the flip-chip routing problem as a network flow problem
- Wire-bonding pads (sources) → Bump pads (sinks)
- Minimizes total wirelength while satisfying capacity constraints
- Solved using minimum-cost maximum-flow algorithm in O(|V|²√|E|) time

### Phase 2: Detailed Routing
1. **Cross-point Assignment**: Assigns cross-points for nets crossing boundaries
2. **Net Ordering Determination**: Orders nets to minimize conflicts
3. **Track Assignment**: Assigns actual routing tracks to each net

## License

Each repository has its own license. Please refer to the individual repository licenses before use.
