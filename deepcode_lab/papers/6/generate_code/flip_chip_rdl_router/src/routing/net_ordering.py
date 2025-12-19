"""
Net Ordering Module for River Routing

This module implements net ordering algorithms for river routing in flip-chip
RDL design. Proper net ordering is essential to ensure planar (crossing-free)
routing by determining the sequence in which nets should be routed.

The key insight is that if two nets would cross when routed, they must be
ordered such that one is routed before the other to avoid conflicts.
"""

from typing import List, Tuple, Dict, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import math

# Import geometry utilities
from ..core.geometry import (
    Point, manhattan_distance, would_cross, is_planar_routing
)


class OrderingStrategy(Enum):
    """Net ordering strategy enumeration."""
    TOPOLOGICAL = "topological"      # Based on crossing constraints
    LEFT_TO_RIGHT = "left_to_right"  # Simple left-to-right by IO position
    RIGHT_TO_LEFT = "right_to_left"  # Simple right-to-left by IO position
    BOTTOM_TO_TOP = "bottom_to_top"  # Bottom-to-top by IO position
    TOP_TO_BOTTOM = "top_to_bottom"  # Top-to-bottom by IO position
    MINIMUM_CROSSING = "min_crossing"  # Minimize total crossings
    HYBRID = "hybrid"                # Combination of strategies


class OrderingStatus(Enum):
    """Status of net ordering result."""
    SUCCESS = "success"
    CYCLE_DETECTED = "cycle_detected"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass
class NetAssignment:
    """Represents a net assignment for ordering purposes."""
    net_id: int
    io_x: float
    io_y: float
    bump_x: float
    bump_y: float
    io_pad: Any = None
    bump_pad: Any = None
    priority: int = 0
    
    @property
    def io_pos(self) -> Tuple[float, float]:
        """Get IO position as tuple."""
        return (self.io_x, self.io_y)
    
    @property
    def bump_pos(self) -> Tuple[float, float]:
        """Get bump position as tuple."""
        return (self.bump_x, self.bump_y)
    
    def crosses(self, other: 'NetAssignment') -> bool:
        """Check if this assignment crosses another."""
        return would_cross(self.io_x, self.bump_x, other.io_x, other.bump_x)


@dataclass
class PrecedenceEdge:
    """Represents a precedence constraint between two nets."""
    from_net: int
    to_net: int
    reason: str = ""


@dataclass
class OrderingResult:
    """Result of net ordering algorithm."""
    status: OrderingStatus
    ordered_nets: List[int]
    ordered_assignments: List[NetAssignment]
    precedence_graph: Dict[int, List[int]]
    num_crossings: int = 0
    cycle_nets: List[int] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_valid(self) -> bool:
        """Check if ordering is valid (no cycles)."""
        return self.status == OrderingStatus.SUCCESS
    
    @property
    def num_nets(self) -> int:
        """Get number of ordered nets."""
        return len(self.ordered_nets)


class PrecedenceGraph:
    """
    Precedence graph for net ordering.
    
    Nodes represent nets, edges represent "must route before" constraints.
    An edge from A to B means net A must be routed before net B.
    """
    
    def __init__(self):
        self.nodes: Set[int] = set()
        self.edges: Dict[int, Set[int]] = {}  # node -> set of successors
        self.reverse_edges: Dict[int, Set[int]] = {}  # node -> set of predecessors
        self.edge_reasons: Dict[Tuple[int, int], str] = {}
    
    def add_node(self, node_id: int) -> None:
        """Add a node to the graph."""
        self.nodes.add(node_id)
        if node_id not in self.edges:
            self.edges[node_id] = set()
        if node_id not in self.reverse_edges:
            self.reverse_edges[node_id] = set()
    
    def add_edge(self, from_node: int, to_node: int, reason: str = "") -> None:
        """Add a precedence edge: from_node must be routed before to_node."""
        self.add_node(from_node)
        self.add_node(to_node)
        self.edges[from_node].add(to_node)
        self.reverse_edges[to_node].add(from_node)
        self.edge_reasons[(from_node, to_node)] = reason
    
    def has_edge(self, from_node: int, to_node: int) -> bool:
        """Check if edge exists."""
        return to_node in self.edges.get(from_node, set())
    
    def get_successors(self, node_id: int) -> Set[int]:
        """Get all nodes that must be routed after this node."""
        return self.edges.get(node_id, set())
    
    def get_predecessors(self, node_id: int) -> Set[int]:
        """Get all nodes that must be routed before this node."""
        return self.reverse_edges.get(node_id, set())
    
    def in_degree(self, node_id: int) -> int:
        """Get number of predecessors."""
        return len(self.reverse_edges.get(node_id, set()))
    
    def out_degree(self, node_id: int) -> int:
        """Get number of successors."""
        return len(self.edges.get(node_id, set()))
    
    def detect_cycle(self) -> Tuple[bool, List[int]]:
        """
        Detect if graph has a cycle using DFS.
        
        Returns:
            Tuple of (has_cycle, cycle_nodes)
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {node: WHITE for node in self.nodes}
        parent = {node: None for node in self.nodes}
        cycle_nodes = []
        
        def dfs(node: int) -> bool:
            color[node] = GRAY
            for neighbor in self.edges.get(node, set()):
                if color[neighbor] == GRAY:
                    # Found cycle - reconstruct it
                    cycle = [neighbor]
                    current = node
                    while current != neighbor:
                        cycle.append(current)
                        current = parent[current]
                    cycle.append(neighbor)
                    cycle_nodes.extend(reversed(cycle))
                    return True
                if color[neighbor] == WHITE:
                    parent[neighbor] = node
                    if dfs(neighbor):
                        return True
            color[node] = BLACK
            return False
        
        for node in self.nodes:
            if color[node] == WHITE:
                if dfs(node):
                    return True, cycle_nodes
        
        return False, []
    
    def topological_sort(self) -> Tuple[bool, List[int]]:
        """
        Perform topological sort using Kahn's algorithm.
        
        Returns:
            Tuple of (success, sorted_nodes)
        """
        # Calculate in-degrees
        in_degree = {node: len(self.reverse_edges.get(node, set())) 
                     for node in self.nodes}
        
        # Initialize queue with nodes having no predecessors
        queue = deque([node for node in self.nodes if in_degree[node] == 0])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            for successor in self.edges.get(node, set()):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)
        
        # Check if all nodes were processed
        if len(result) != len(self.nodes):
            return False, result
        
        return True, result
    
    def to_dict(self) -> Dict[int, List[int]]:
        """Convert to dictionary representation."""
        return {node: list(successors) for node, successors in self.edges.items()}


class NetOrderer:
    """
    Net ordering algorithm for river routing.
    
    Determines the order in which nets should be routed to avoid
    crossing conflicts in single-layer planar routing.
    """
    
    def __init__(self, strategy: OrderingStrategy = OrderingStrategy.TOPOLOGICAL):
        """
        Initialize net orderer.
        
        Args:
            strategy: Ordering strategy to use
        """
        self.strategy = strategy
        self.assignments: List[NetAssignment] = []
        self.precedence_graph = PrecedenceGraph()
    
    def set_assignments(self, assignments: List[NetAssignment]) -> None:
        """Set the net assignments to order."""
        self.assignments = assignments
    
    def add_assignment(self, net_id: int, io_x: float, io_y: float,
                       bump_x: float, bump_y: float,
                       io_pad: Any = None, bump_pad: Any = None,
                       priority: int = 0) -> None:
        """Add a net assignment."""
        assignment = NetAssignment(
            net_id=net_id,
            io_x=io_x,
            io_y=io_y,
            bump_x=bump_x,
            bump_y=bump_y,
            io_pad=io_pad,
            bump_pad=bump_pad,
            priority=priority
        )
        self.assignments.append(assignment)
    
    def build_precedence_graph(self) -> PrecedenceGraph:
        """
        Build precedence graph from assignments.
        
        For river routing, net i must be routed before net j if:
        - io_i.x < io_j.x AND bump_i.x > bump_j.x (crossing from left)
        - OR io_i.x > io_j.x AND bump_i.x < bump_j.x (crossing from right)
        
        The net with smaller IO x-coordinate should be routed first
        when there's a crossing.
        """
        self.precedence_graph = PrecedenceGraph()
        
        # Add all nets as nodes
        for assignment in self.assignments:
            self.precedence_graph.add_node(assignment.net_id)
        
        # Check all pairs for crossing constraints
        n = len(self.assignments)
        for i in range(n):
            for j in range(i + 1, n):
                a_i = self.assignments[i]
                a_j = self.assignments[j]
                
                if a_i.crosses(a_j):
                    # Determine which should be routed first
                    # The one with smaller IO x goes first (left-to-right convention)
                    if a_i.io_x < a_j.io_x:
                        self.precedence_graph.add_edge(
                            a_i.net_id, a_j.net_id,
                            f"Net {a_i.net_id} crosses {a_j.net_id}, left-first"
                        )
                    else:
                        self.precedence_graph.add_edge(
                            a_j.net_id, a_i.net_id,
                            f"Net {a_j.net_id} crosses {a_i.net_id}, left-first"
                        )
        
        return self.precedence_graph
    
    def order_nets(self) -> OrderingResult:
        """
        Order nets according to the selected strategy.
        
        Returns:
            OrderingResult with ordered nets
        """
        if not self.assignments:
            return OrderingResult(
                status=OrderingStatus.SUCCESS,
                ordered_nets=[],
                ordered_assignments=[],
                precedence_graph={},
                num_crossings=0
            )
        
        if self.strategy == OrderingStrategy.TOPOLOGICAL:
            return self._order_topological()
        elif self.strategy == OrderingStrategy.LEFT_TO_RIGHT:
            return self._order_by_position(key=lambda a: a.io_x)
        elif self.strategy == OrderingStrategy.RIGHT_TO_LEFT:
            return self._order_by_position(key=lambda a: -a.io_x)
        elif self.strategy == OrderingStrategy.BOTTOM_TO_TOP:
            return self._order_by_position(key=lambda a: a.io_y)
        elif self.strategy == OrderingStrategy.TOP_TO_BOTTOM:
            return self._order_by_position(key=lambda a: -a.io_y)
        elif self.strategy == OrderingStrategy.MINIMUM_CROSSING:
            return self._order_minimum_crossing()
        elif self.strategy == OrderingStrategy.HYBRID:
            return self._order_hybrid()
        else:
            return self._order_topological()
    
    def _order_topological(self) -> OrderingResult:
        """Order nets using topological sort on precedence graph."""
        # Build precedence graph
        self.build_precedence_graph()
        
        # Check for cycles
        has_cycle, cycle_nodes = self.precedence_graph.detect_cycle()
        
        if has_cycle:
            # Try to break cycles and get partial ordering
            return self._handle_cycle(cycle_nodes)
        
        # Perform topological sort
        success, ordered_ids = self.precedence_graph.topological_sort()
        
        if not success:
            return OrderingResult(
                status=OrderingStatus.FAILED,
                ordered_nets=ordered_ids,
                ordered_assignments=[],
                precedence_graph=self.precedence_graph.to_dict(),
                num_crossings=self._count_crossings()
            )
        
        # Map ordered IDs to assignments
        id_to_assignment = {a.net_id: a for a in self.assignments}
        ordered_assignments = [id_to_assignment[net_id] for net_id in ordered_ids
                               if net_id in id_to_assignment]
        
        return OrderingResult(
            status=OrderingStatus.SUCCESS,
            ordered_nets=ordered_ids,
            ordered_assignments=ordered_assignments,
            precedence_graph=self.precedence_graph.to_dict(),
            num_crossings=0,  # Topological order has no crossings
            statistics={
                'num_precedence_edges': sum(len(s) for s in self.precedence_graph.edges.values()),
                'strategy': self.strategy.value
            }
        )
    
    def _order_by_position(self, key) -> OrderingResult:
        """Order nets by position using given key function."""
        sorted_assignments = sorted(self.assignments, key=key)
        ordered_ids = [a.net_id for a in sorted_assignments]
        
        # Count crossings in this ordering
        num_crossings = self._count_crossings_in_order(sorted_assignments)
        
        status = OrderingStatus.SUCCESS if num_crossings == 0 else OrderingStatus.PARTIAL
        
        return OrderingResult(
            status=status,
            ordered_nets=ordered_ids,
            ordered_assignments=sorted_assignments,
            precedence_graph={},
            num_crossings=num_crossings,
            statistics={
                'strategy': self.strategy.value
            }
        )
    
    def _order_minimum_crossing(self) -> OrderingResult:
        """
        Order nets to minimize total crossings.
        
        Uses a greedy approach: at each step, select the net that
        introduces the fewest new crossings.
        """
        remaining = set(range(len(self.assignments)))
        ordered_indices = []
        
        while remaining:
            best_idx = None
            best_crossings = float('inf')
            
            for idx in remaining:
                # Count crossings if we add this net next
                crossings = 0
                for ordered_idx in ordered_indices:
                    if self.assignments[idx].crosses(self.assignments[ordered_idx]):
                        crossings += 1
                
                if crossings < best_crossings:
                    best_crossings = crossings
                    best_idx = idx
            
            ordered_indices.append(best_idx)
            remaining.remove(best_idx)
        
        ordered_assignments = [self.assignments[i] for i in ordered_indices]
        ordered_ids = [a.net_id for a in ordered_assignments]
        num_crossings = self._count_crossings_in_order(ordered_assignments)
        
        status = OrderingStatus.SUCCESS if num_crossings == 0 else OrderingStatus.PARTIAL
        
        return OrderingResult(
            status=status,
            ordered_nets=ordered_ids,
            ordered_assignments=ordered_assignments,
            precedence_graph={},
            num_crossings=num_crossings,
            statistics={
                'strategy': self.strategy.value
            }
        )
    
    def _order_hybrid(self) -> OrderingResult:
        """
        Hybrid ordering: try topological first, fall back to minimum crossing.
        """
        # Try topological ordering first
        topo_result = self._order_topological()
        
        if topo_result.status == OrderingStatus.SUCCESS:
            return topo_result
        
        # Fall back to minimum crossing
        min_cross_result = self._order_minimum_crossing()
        min_cross_result.statistics['fallback_from'] = 'topological'
        
        return min_cross_result
    
    def _handle_cycle(self, cycle_nodes: List[int]) -> OrderingResult:
        """
        Handle cycles in precedence graph.
        
        When cycles exist, the routing is not perfectly planar.
        We try to find a partial ordering that minimizes crossings.
        """
        # Use minimum crossing strategy as fallback
        result = self._order_minimum_crossing()
        result.status = OrderingStatus.CYCLE_DETECTED
        result.cycle_nets = cycle_nodes
        result.statistics['cycle_detected'] = True
        result.statistics['cycle_size'] = len(cycle_nodes)
        
        return result
    
    def _count_crossings(self) -> int:
        """Count total number of crossing pairs."""
        count = 0
        n = len(self.assignments)
        for i in range(n):
            for j in range(i + 1, n):
                if self.assignments[i].crosses(self.assignments[j]):
                    count += 1
        return count
    
    def _count_crossings_in_order(self, ordered: List[NetAssignment]) -> int:
        """
        Count crossings that would occur with given ordering.
        
        In proper river routing order, earlier nets should not cross later nets.
        """
        count = 0
        n = len(ordered)
        for i in range(n):
            for j in range(i + 1, n):
                if ordered[i].crosses(ordered[j]):
                    count += 1
        return count
    
    def check_routability(self) -> Tuple[bool, int]:
        """
        Check if assignments can be routed without crossings.
        
        Returns:
            Tuple of (is_routable, num_crossings)
        """
        num_crossings = self._count_crossings()
        
        # Build precedence graph and check for cycles
        self.build_precedence_graph()
        has_cycle, _ = self.precedence_graph.detect_cycle()
        
        # Routable if no cycles (can find valid topological order)
        return not has_cycle, num_crossings
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get ordering statistics."""
        return {
            'num_assignments': len(self.assignments),
            'num_crossings': self._count_crossings(),
            'num_precedence_nodes': len(self.precedence_graph.nodes),
            'num_precedence_edges': sum(len(s) for s in self.precedence_graph.edges.values()),
            'strategy': self.strategy.value
        }


def order_nets(assignments: List[Tuple[Any, Any]], 
               strategy: OrderingStrategy = OrderingStrategy.TOPOLOGICAL) -> OrderingResult:
    """
    Convenience function to order nets from IO-bump assignments.
    
    Args:
        assignments: List of (io_pad, bump_pad) tuples
        strategy: Ordering strategy to use
    
    Returns:
        OrderingResult with ordered nets
    """
    orderer = NetOrderer(strategy=strategy)
    
    for i, (io_pad, bump_pad) in enumerate(assignments):
        # Handle different input formats
        if hasattr(io_pad, 'x') and hasattr(io_pad, 'y'):
            io_x, io_y = io_pad.x, io_pad.y
        elif isinstance(io_pad, (tuple, list)):
            io_x, io_y = io_pad[0], io_pad[1]
        else:
            io_x, io_y = float(io_pad), 0.0
        
        if hasattr(bump_pad, 'x') and hasattr(bump_pad, 'y'):
            bump_x, bump_y = bump_pad.x, bump_pad.y
        elif isinstance(bump_pad, (tuple, list)):
            bump_x, bump_y = bump_pad[0], bump_pad[1]
        else:
            bump_x, bump_y = float(bump_pad), 0.0
        
        net_id = getattr(io_pad, 'net_id', i) if hasattr(io_pad, 'net_id') else i
        
        orderer.add_assignment(
            net_id=net_id,
            io_x=io_x,
            io_y=io_y,
            bump_x=bump_x,
            bump_y=bump_y,
            io_pad=io_pad,
            bump_pad=bump_pad
        )
    
    return orderer.order_nets()


def must_route_before(net_i: NetAssignment, net_j: NetAssignment) -> bool:
    """
    Check if net_i must be routed before net_j.
    
    Net i must be routed before net j if:
    - They would cross AND
    - io_i.x < io_j.x (left-to-right convention)
    
    Args:
        net_i: First net assignment
        net_j: Second net assignment
    
    Returns:
        True if net_i must be routed before net_j
    """
    if not net_i.crosses(net_j):
        return False
    
    # Left-to-right convention: smaller IO x goes first
    return net_i.io_x < net_j.io_x


def find_optimal_ordering(assignments: List[NetAssignment]) -> List[NetAssignment]:
    """
    Find optimal ordering that minimizes crossings.
    
    For small numbers of nets, tries all strategies and returns best.
    
    Args:
        assignments: List of net assignments
    
    Returns:
        Optimally ordered list of assignments
    """
    if len(assignments) <= 1:
        return assignments
    
    orderer = NetOrderer()
    orderer.set_assignments(assignments)
    
    best_result = None
    best_crossings = float('inf')
    
    for strategy in [OrderingStrategy.TOPOLOGICAL, 
                     OrderingStrategy.LEFT_TO_RIGHT,
                     OrderingStrategy.MINIMUM_CROSSING]:
        orderer.strategy = strategy
        result = orderer.order_nets()
        
        if result.num_crossings < best_crossings:
            best_crossings = result.num_crossings
            best_result = result
        
        if best_crossings == 0:
            break
    
    return best_result.ordered_assignments if best_result else assignments


def test_net_ordering():
    """Test net ordering functionality."""
    print("Testing Net Ordering Module...")
    
    # Test 1: Simple non-crossing case
    print("\nTest 1: Non-crossing assignments")
    orderer = NetOrderer()
    orderer.add_assignment(0, 0, 0, 100, 100)
    orderer.add_assignment(1, 50, 0, 150, 100)
    orderer.add_assignment(2, 100, 0, 200, 100)
    
    result = orderer.order_nets()
    print(f"  Status: {result.status}")
    print(f"  Ordered nets: {result.ordered_nets}")
    print(f"  Crossings: {result.num_crossings}")
    assert result.status == OrderingStatus.SUCCESS
    assert result.num_crossings == 0
    
    # Test 2: Crossing case
    print("\nTest 2: Crossing assignments")
    orderer2 = NetOrderer()
    orderer2.add_assignment(0, 0, 0, 200, 100)    # IO at x=0, bump at x=200
    orderer2.add_assignment(1, 100, 0, 100, 100)  # IO at x=100, bump at x=100
    
    result2 = orderer2.order_nets()
    print(f"  Status: {result2.status}")
    print(f"  Ordered nets: {result2.ordered_nets}")
    print(f"  Crossings: {result2.num_crossings}")
    # Net 0 should come before net 1 (smaller IO x)
    assert result2.ordered_nets[0] == 0
    
    # Test 3: Multiple crossings
    print("\nTest 3: Multiple crossing assignments")
    orderer3 = NetOrderer()
    orderer3.add_assignment(0, 0, 0, 300, 100)
    orderer3.add_assignment(1, 100, 0, 200, 100)
    orderer3.add_assignment(2, 200, 0, 100, 100)
    
    result3 = orderer3.order_nets()
    print(f"  Status: {result3.status}")
    print(f"  Ordered nets: {result3.ordered_nets}")
    print(f"  Crossings: {result3.num_crossings}")
    
    # Test 4: Convenience function
    print("\nTest 4: Convenience function")
    assignments = [
        ((0, 0), (100, 100)),
        ((50, 0), (150, 100)),
    ]
    result4 = order_nets(assignments)
    print(f"  Status: {result4.status}")
    print(f"  Num nets: {result4.num_nets}")
    
    # Test 5: Routability check
    print("\nTest 5: Routability check")
    orderer5 = NetOrderer()
    orderer5.add_assignment(0, 0, 0, 100, 100)
    orderer5.add_assignment(1, 50, 0, 50, 100)
    
    is_routable, crossings = orderer5.check_routability()
    print(f"  Is routable: {is_routable}")
    print(f"  Crossings: {crossings}")
    
    print("\n✓ All net ordering tests passed!")


if __name__ == "__main__":
    test_net_ordering()
