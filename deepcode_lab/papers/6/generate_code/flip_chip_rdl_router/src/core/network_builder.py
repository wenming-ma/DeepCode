"""
Network Builder Module for Flip-Chip RDL Routing

This module constructs flow networks from chip layouts for the MCMF algorithm.
The network structure connects:
- Source node S to all IO pads (capacity=1, cost=0)
- IO pad nodes to bump pad nodes (capacity=1, cost=manhattan_distance)
- Bump pad nodes to sink node T (capacity=bump_capacity, cost=0)

The MCMF solution on this network provides optimal IO-to-bump assignments
that minimize total wirelength.
"""

from typing import List, Tuple, Dict, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import math

from .mcmf import MCMF, INF
from .geometry import manhattan_distance, Point


class NodeType(Enum):
    """Types of nodes in the flow network."""
    SOURCE = 0
    SINK = 1
    IO_PAD = 2
    BUMP_PAD = 3


@dataclass
class NetworkNode:
    """Represents a node in the flow network."""
    node_id: int
    node_type: NodeType
    original_id: Optional[int] = None  # ID in original data structure
    name: str = ""
    x: float = 0.0
    y: float = 0.0
    
    def __hash__(self):
        return hash(self.node_id)
    
    def __eq__(self, other):
        if isinstance(other, NetworkNode):
            return self.node_id == other.node_id
        return False


@dataclass
class NetworkEdge:
    """Represents an edge in the flow network."""
    from_node: int
    to_node: int
    capacity: int
    cost: float
    edge_type: str = ""  # "source_to_io", "io_to_bump", "bump_to_sink"


class FlowNetwork:
    """
    Represents the complete flow network for RDL routing.
    
    Network Structure:
    - Node 0: Source (S)
    - Nodes 1 to n_io: IO pads
    - Nodes n_io+1 to n_io+n_bump: Bump pads
    - Node n_io+n_bump+1: Sink (T)
    """
    
    def __init__(self):
        self.nodes: List[NetworkNode] = []
        self.edges: List[NetworkEdge] = []
        self.n_nodes: int = 0
        self.source_id: int = 0
        self.sink_id: int = 0
        
        # Mappings between original IDs and network node IDs
        self.io_to_node: Dict[int, int] = {}  # original IO id -> network node id
        self.bump_to_node: Dict[int, int] = {}  # original bump id -> network node id
        self.node_to_io: Dict[int, int] = {}  # network node id -> original IO id
        self.node_to_bump: Dict[int, int] = {}  # network node id -> original bump id
        
        # Store original objects for reference
        self.io_pads: List[Any] = []
        self.bump_pads: List[Any] = []
        
    def get_io_node_id(self, io_original_id: int) -> int:
        """Get network node ID for an IO pad."""
        return self.io_to_node.get(io_original_id, -1)
    
    def get_bump_node_id(self, bump_original_id: int) -> int:
        """Get network node ID for a bump pad."""
        return self.bump_to_node.get(bump_original_id, -1)
    
    def get_original_io_id(self, node_id: int) -> int:
        """Get original IO pad ID from network node ID."""
        return self.node_to_io.get(node_id, -1)
    
    def get_original_bump_id(self, node_id: int) -> int:
        """Get original bump pad ID from network node ID."""
        return self.node_to_bump.get(node_id, -1)
    
    def get_node(self, node_id: int) -> Optional[NetworkNode]:
        """Get node by ID."""
        if 0 <= node_id < len(self.nodes):
            return self.nodes[node_id]
        return None
    
    def get_io_pad(self, node_id: int) -> Optional[Any]:
        """Get original IO pad object from network node ID."""
        original_id = self.node_to_io.get(node_id, -1)
        if original_id >= 0 and original_id < len(self.io_pads):
            return self.io_pads[original_id]
        return None
    
    def get_bump_pad(self, node_id: int) -> Optional[Any]:
        """Get original bump pad object from network node ID."""
        original_id = self.node_to_bump.get(node_id, -1)
        if original_id >= 0 and original_id < len(self.bump_pads):
            return self.bump_pads[original_id]
        return None


class NetworkBuilder:
    """
    Builds flow networks from chip layouts for MCMF-based global routing.
    
    The network construction follows the paper's approach:
    1. Create source node connected to all IO pads
    2. Create edges from IO pads to valid bump pads with Manhattan distance cost
    3. Create sink node connected from all bump pads
    """
    
    def __init__(self):
        self.network: Optional[FlowNetwork] = None
        self.mcmf: Optional[MCMF] = None
        
        # Configuration parameters
        self.max_distance: float = INF  # Maximum allowed routing distance
        self.distance_threshold: float = INF  # Only connect if distance < threshold
        self.use_power_constraints: bool = True
        self.allow_blocked_bumps: bool = False
        
    def build_network(self, chip: Any, constraints: Optional[Dict] = None) -> Tuple[MCMF, FlowNetwork]:
        """
        Build flow network from chip layout.

        Args:
            chip: Chip object containing io_pads and bump_pads
            constraints: Optional routing constraints

        Returns:
            Tuple of (MCMF solver, FlowNetwork metadata)
        """
        io_pads = chip.io_pads
        bump_pads = chip.bump_pads

        return self.build_network_from_pads(io_pads, bump_pads, constraints)
    
    def build_network_from_pads(
        self,
        io_pads: List[Any],
        bump_pads: List[Any],
        constraints: Optional[Dict] = None
    ) -> Tuple[MCMF, FlowNetwork]:
        """
        Build flow network from IO pads and bump pads.
        
        Args:
            io_pads: List of IOPad objects
            bump_pads: List of BumpPad objects
            constraints: Optional routing constraints
            
        Returns:
            Tuple of (MCMF solver, FlowNetwork metadata)
        """
        # Apply constraints if provided
        if constraints:
            self.max_distance = constraints.get('max_distance', INF)
            self.distance_threshold = constraints.get('distance_threshold', INF)
            self.use_power_constraints = constraints.get('use_power_constraints', True)
            self.allow_blocked_bumps = constraints.get('allow_blocked_bumps', False)
        
        # Filter out blocked bump pads if not allowed
        available_bumps = bump_pads
        if not self.allow_blocked_bumps:
            available_bumps = [b for b in bump_pads if not getattr(b, 'is_blocked', False)]
        
        n_io = len(io_pads)
        n_bump = len(available_bumps)
        
        # Total nodes: source + IO pads + bump pads + sink
        n_nodes = 1 + n_io + n_bump + 1
        
        # Create flow network metadata
        network = FlowNetwork()
        network.n_nodes = n_nodes
        network.source_id = 0
        network.sink_id = n_nodes - 1
        network.io_pads = list(io_pads)
        network.bump_pads = list(available_bumps)
        
        # Create MCMF solver
        mcmf = MCMF(n_nodes)
        
        # Add source node
        source_node = NetworkNode(
            node_id=0,
            node_type=NodeType.SOURCE,
            name="SOURCE"
        )
        network.nodes.append(source_node)
        
        # Add IO pad nodes (IDs 1 to n_io)
        for i, io_pad in enumerate(io_pads):
            node_id = 1 + i
            io_node = NetworkNode(
                node_id=node_id,
                node_type=NodeType.IO_PAD,
                original_id=i,
                name=getattr(io_pad, 'name', f"IO_{i}"),
                x=io_pad.x,
                y=io_pad.y
            )
            network.nodes.append(io_node)
            network.io_to_node[i] = node_id
            network.node_to_io[node_id] = i
        
        # Add bump pad nodes (IDs n_io+1 to n_io+n_bump)
        for j, bump_pad in enumerate(available_bumps):
            node_id = 1 + n_io + j
            bump_node = NetworkNode(
                node_id=node_id,
                node_type=NodeType.BUMP_PAD,
                original_id=j,
                name=getattr(bump_pad, 'name', f"BUMP_{j}"),
                x=bump_pad.x,
                y=bump_pad.y
            )
            network.nodes.append(bump_node)
            network.bump_to_node[j] = node_id
            network.node_to_bump[node_id] = j
        
        # Add sink node
        sink_node = NetworkNode(
            node_id=n_nodes - 1,
            node_type=NodeType.SINK,
            name="SINK"
        )
        network.nodes.append(sink_node)
        
        # Add edges: Source -> IO pads (capacity=1, cost=0)
        for i in range(n_io):
            io_node_id = network.io_to_node[i]
            mcmf.add_edge(network.source_id, io_node_id, capacity=1, cost=0)
            network.edges.append(NetworkEdge(
                from_node=network.source_id,
                to_node=io_node_id,
                capacity=1,
                cost=0,
                edge_type="source_to_io"
            ))
        
        # Add edges: IO pads -> Bump pads (capacity=1, cost=manhattan_distance)
        for i, io_pad in enumerate(io_pads):
            io_node_id = network.io_to_node[i]
            io_pos = (io_pad.x, io_pad.y)
            
            for j, bump_pad in enumerate(available_bumps):
                # Check if assignment is valid
                if not self._is_valid_assignment(io_pad, bump_pad):
                    continue
                
                bump_node_id = network.bump_to_node[j]
                bump_pos = (bump_pad.x, bump_pad.y)
                
                # Calculate Manhattan distance as cost
                dist = manhattan_distance(io_pos, bump_pos)
                
                # Skip if distance exceeds threshold
                if dist > self.distance_threshold:
                    continue
                
                mcmf.add_edge(io_node_id, bump_node_id, capacity=1, cost=dist)
                network.edges.append(NetworkEdge(
                    from_node=io_node_id,
                    to_node=bump_node_id,
                    capacity=1,
                    cost=dist,
                    edge_type="io_to_bump"
                ))
        
        # Add edges: Bump pads -> Sink (capacity=bump_capacity, cost=0)
        for j, bump_pad in enumerate(available_bumps):
            bump_node_id = network.bump_to_node[j]
            capacity = getattr(bump_pad, 'capacity', 1)
            
            mcmf.add_edge(bump_node_id, network.sink_id, capacity=capacity, cost=0)
            network.edges.append(NetworkEdge(
                from_node=bump_node_id,
                to_node=network.sink_id,
                capacity=capacity,
                cost=0,
                edge_type="bump_to_sink"
            ))
        
        self.network = network
        self.mcmf = mcmf
        
        return mcmf, network
    
    def _is_valid_assignment(self, io_pad: Any, bump_pad: Any) -> bool:
        """
        Check if an IO pad can be assigned to a bump pad.
        
        Args:
            io_pad: IOPad object
            bump_pad: BumpPad object
            
        Returns:
            True if assignment is valid
        """
        # Check power/ground constraints
        if self.use_power_constraints:
            io_is_power = getattr(io_pad, 'is_power', False)
            bump_is_power = getattr(bump_pad, 'is_power', False)
            
            # Power IOs should go to power bumps, signal IOs to signal bumps
            if io_is_power != bump_is_power:
                return False
        
        # Check if bump is blocked
        if not self.allow_blocked_bumps:
            if getattr(bump_pad, 'is_blocked', False):
                return False
        
        # Check if bump has remaining capacity
        remaining = getattr(bump_pad, 'remaining_capacity', 1)
        if remaining <= 0:
            return False
        
        return True
    
    def build_bipartite_network(
        self,
        io_pads: List[Any],
        bump_pads: List[Any],
        cost_matrix: Optional[List[List[float]]] = None
    ) -> Tuple[MCMF, FlowNetwork]:
        """
        Build a bipartite matching network with custom cost matrix.
        
        Args:
            io_pads: List of IOPad objects
            bump_pads: List of BumpPad objects
            cost_matrix: Optional n_io x n_bump cost matrix
            
        Returns:
            Tuple of (MCMF solver, FlowNetwork metadata)
        """
        n_io = len(io_pads)
        n_bump = len(bump_pads)
        n_nodes = 1 + n_io + n_bump + 1
        
        network = FlowNetwork()
        network.n_nodes = n_nodes
        network.source_id = 0
        network.sink_id = n_nodes - 1
        network.io_pads = list(io_pads)
        network.bump_pads = list(bump_pads)
        
        mcmf = MCMF(n_nodes)
        
        # Add source node
        network.nodes.append(NetworkNode(0, NodeType.SOURCE, name="SOURCE"))
        
        # Add IO nodes
        for i, io_pad in enumerate(io_pads):
            node_id = 1 + i
            network.nodes.append(NetworkNode(
                node_id, NodeType.IO_PAD, i,
                getattr(io_pad, 'name', f"IO_{i}"),
                io_pad.x, io_pad.y
            ))
            network.io_to_node[i] = node_id
            network.node_to_io[node_id] = i
        
        # Add bump nodes
        for j, bump_pad in enumerate(bump_pads):
            node_id = 1 + n_io + j
            network.nodes.append(NetworkNode(
                node_id, NodeType.BUMP_PAD, j,
                getattr(bump_pad, 'name', f"BUMP_{j}"),
                bump_pad.x, bump_pad.y
            ))
            network.bump_to_node[j] = node_id
            network.node_to_bump[node_id] = j
        
        # Add sink node
        network.nodes.append(NetworkNode(n_nodes - 1, NodeType.SINK, name="SINK"))
        
        # Source -> IO edges
        for i in range(n_io):
            mcmf.add_edge(0, 1 + i, 1, 0)
        
        # IO -> Bump edges with costs
        for i in range(n_io):
            for j in range(n_bump):
                if cost_matrix is not None:
                    cost = cost_matrix[i][j]
                else:
                    io_pos = (io_pads[i].x, io_pads[i].y)
                    bump_pos = (bump_pads[j].x, bump_pads[j].y)
                    cost = manhattan_distance(io_pos, bump_pos)
                
                if cost < INF:
                    mcmf.add_edge(1 + i, 1 + n_io + j, 1, cost)
        
        # Bump -> Sink edges
        for j in range(n_bump):
            capacity = getattr(bump_pads[j], 'capacity', 1)
            mcmf.add_edge(1 + n_io + j, n_nodes - 1, capacity, 0)
        
        self.network = network
        self.mcmf = mcmf
        
        return mcmf, network
    
    def extract_assignments(
        self,
        mcmf: MCMF,
        network: FlowNetwork
    ) -> List[Tuple[Any, Any]]:
        """
        Extract IO-to-bump assignments from MCMF solution.
        
        Args:
            mcmf: Solved MCMF instance
            network: Flow network metadata
            
        Returns:
            List of (IOPad, BumpPad) assignment tuples
        """
        assignments = []
        
        # Get all edges with positive flow
        flow_edges = mcmf.get_flow_edges()
        
        for from_node, to_node, flow in flow_edges:
            # Check if this is an IO->Bump edge
            if from_node in network.node_to_io and to_node in network.node_to_bump:
                io_idx = network.node_to_io[from_node]
                bump_idx = network.node_to_bump[to_node]
                
                io_pad = network.io_pads[io_idx]
                bump_pad = network.bump_pads[bump_idx]
                
                assignments.append((io_pad, bump_pad))
        
        return assignments
    
    def get_assignment_cost(
        self,
        assignments: List[Tuple[Any, Any]]
    ) -> float:
        """
        Calculate total cost (wirelength) of assignments.
        
        Args:
            assignments: List of (IOPad, BumpPad) tuples
            
        Returns:
            Total Manhattan distance
        """
        total_cost = 0.0
        for io_pad, bump_pad in assignments:
            io_pos = (io_pad.x, io_pad.y)
            bump_pos = (bump_pad.x, bump_pad.y)
            total_cost += manhattan_distance(io_pos, bump_pos)
        return total_cost
    
    def validate_network(self, network: FlowNetwork) -> Tuple[bool, List[str]]:
        """
        Validate the constructed flow network.
        
        Args:
            network: FlowNetwork to validate
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        # Check node count
        expected_nodes = 2 + len(network.io_pads) + len(network.bump_pads)
        if network.n_nodes != expected_nodes:
            errors.append(f"Node count mismatch: {network.n_nodes} vs expected {expected_nodes}")
        
        # Check source and sink
        if network.source_id != 0:
            errors.append(f"Source ID should be 0, got {network.source_id}")
        if network.sink_id != network.n_nodes - 1:
            errors.append(f"Sink ID should be {network.n_nodes - 1}, got {network.sink_id}")
        
        # Check mappings
        for i in range(len(network.io_pads)):
            if i not in network.io_to_node:
                errors.append(f"IO pad {i} not in io_to_node mapping")
        
        for j in range(len(network.bump_pads)):
            if j not in network.bump_to_node:
                errors.append(f"Bump pad {j} not in bump_to_node mapping")
        
        # Check edge connectivity
        source_edges = sum(1 for e in network.edges if e.from_node == network.source_id)
        if source_edges != len(network.io_pads):
            errors.append(f"Source should have {len(network.io_pads)} outgoing edges, got {source_edges}")
        
        sink_edges = sum(1 for e in network.edges if e.to_node == network.sink_id)
        if sink_edges != len(network.bump_pads):
            errors.append(f"Sink should have {len(network.bump_pads)} incoming edges, got {sink_edges}")
        
        return len(errors) == 0, errors
    
    def get_network_statistics(self, network: FlowNetwork) -> Dict[str, Any]:
        """
        Get statistics about the flow network.
        
        Args:
            network: FlowNetwork to analyze
            
        Returns:
            Dictionary of statistics
        """
        io_to_bump_edges = [e for e in network.edges if e.edge_type == "io_to_bump"]
        
        costs = [e.cost for e in io_to_bump_edges]
        
        stats = {
            'n_nodes': network.n_nodes,
            'n_edges': len(network.edges),
            'n_io_pads': len(network.io_pads),
            'n_bump_pads': len(network.bump_pads),
            'n_io_to_bump_edges': len(io_to_bump_edges),
            'total_bump_capacity': sum(
                getattr(b, 'capacity', 1) for b in network.bump_pads
            ),
            'avg_edge_cost': sum(costs) / len(costs) if costs else 0,
            'min_edge_cost': min(costs) if costs else 0,
            'max_edge_cost': max(costs) if costs else 0,
            'edge_density': len(io_to_bump_edges) / (
                len(network.io_pads) * len(network.bump_pads)
            ) if network.io_pads and network.bump_pads else 0
        }
        
        return stats


def build_routing_network(
    io_pads: List[Any],
    bump_pads: List[Any],
    constraints: Optional[Dict] = None
) -> Tuple[MCMF, FlowNetwork]:
    """
    Convenience function to build a routing network.
    
    Args:
        io_pads: List of IOPad objects
        bump_pads: List of BumpPad objects
        constraints: Optional routing constraints
        
    Returns:
        Tuple of (MCMF solver, FlowNetwork metadata)
    """
    builder = NetworkBuilder()
    return builder.build_network_from_pads(io_pads, bump_pads, constraints)


def solve_assignment(
    io_pads: List[Any],
    bump_pads: List[Any],
    constraints: Optional[Dict] = None
) -> Tuple[List[Tuple[Any, Any]], float, int]:
    """
    Solve the IO-to-bump assignment problem.
    
    Args:
        io_pads: List of IOPad objects
        bump_pads: List of BumpPad objects
        constraints: Optional routing constraints
        
    Returns:
        Tuple of (assignments, total_cost, total_flow)
    """
    builder = NetworkBuilder()
    mcmf, network = builder.build_network_from_pads(io_pads, bump_pads, constraints)
    
    # Solve MCMF
    total_flow, total_cost = mcmf.min_cost_max_flow(network.source_id, network.sink_id)
    
    # Extract assignments
    assignments = builder.extract_assignments(mcmf, network)
    
    return assignments, total_cost, total_flow


# Test function
def test_network_builder():
    """Test the network builder with simple examples."""
    from ..data_structures import IOPad, BumpPad
    
    # Create simple test case
    io_pads = [
        IOPad(0, 0, net_id=0, name="IO_0"),
        IOPad(100, 0, net_id=1, name="IO_1"),
        IOPad(200, 0, net_id=2, name="IO_2"),
    ]
    
    bump_pads = [
        BumpPad(50, 100, capacity=1, name="BUMP_0"),
        BumpPad(150, 100, capacity=1, name="BUMP_1"),
        BumpPad(250, 100, capacity=1, name="BUMP_2"),
    ]
    
    # Build network
    builder = NetworkBuilder()
    mcmf, network = builder.build_network_from_pads(io_pads, bump_pads)
    
    # Validate network
    is_valid, errors = builder.validate_network(network)
    print(f"Network valid: {is_valid}")
    if errors:
        print(f"Errors: {errors}")
    
    # Get statistics
    stats = builder.get_network_statistics(network)
    print(f"Network statistics: {stats}")
    
    # Solve MCMF
    total_flow, total_cost = mcmf.min_cost_max_flow(network.source_id, network.sink_id)
    print(f"Total flow: {total_flow}, Total cost: {total_cost}")
    
    # Extract assignments
    assignments = builder.extract_assignments(mcmf, network)
    print(f"Assignments: {[(io.name, bump.name) for io, bump in assignments]}")
    
    # Verify
    assert total_flow == 3, f"Expected flow 3, got {total_flow}"
    assert len(assignments) == 3, f"Expected 3 assignments, got {len(assignments)}"
    
    print("Network builder tests passed!")


if __name__ == "__main__":
    test_network_builder()
