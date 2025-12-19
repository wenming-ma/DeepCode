"""
Minimum-Cost Maximum-Flow (MCMF) Algorithm Implementation

This module implements the Successive Shortest Path algorithm with Bellman-Ford
for finding minimum-cost maximum-flow in a flow network. This is the core
algorithm used for global routing optimization in flip-chip RDL routing.

The algorithm finds the maximum flow from source to sink while minimizing
the total cost (sum of flow * edge_cost for all edges).

Algorithm Complexity:
- Time: O(V * E * F) where V=vertices, E=edges, F=max flow
- With SPFA optimization: O(V * E * F) average case, better in practice
- Space: O(V^2) for adjacency matrix representation
"""

from typing import List, Tuple, Optional, Dict, Any
from collections import deque
import math

# Constants
INF = float('inf')


class Edge:
    """
    Represents a directed edge in the flow network.
    
    Attributes:
        to: Destination node index
        capacity: Maximum flow capacity
        cost: Cost per unit flow
        flow: Current flow through edge
        reverse_idx: Index of reverse edge in adjacency list
    """
    __slots__ = ['to', 'capacity', 'cost', 'flow', 'reverse_idx']
    
    def __init__(self, to: int, capacity: int, cost: float, reverse_idx: int):
        self.to = to
        self.capacity = capacity
        self.cost = cost
        self.flow = 0
        self.reverse_idx = reverse_idx
    
    @property
    def residual_capacity(self) -> int:
        """Returns remaining capacity on this edge."""
        return self.capacity - self.flow
    
    def __repr__(self) -> str:
        return f"Edge(to={self.to}, cap={self.capacity}, cost={self.cost}, flow={self.flow})"


class MCMF:
    """
    Minimum-Cost Maximum-Flow solver using Successive Shortest Path algorithm.
    
    This implementation uses:
    - Adjacency list representation for sparse graphs
    - SPFA (Shortest Path Faster Algorithm) for finding augmenting paths
    - Supports negative edge costs (required for residual graph)
    
    Usage:
        mcmf = MCMF(n_nodes)
        mcmf.add_edge(u, v, capacity, cost)
        max_flow, min_cost = mcmf.min_cost_max_flow(source, sink)
    """
    
    def __init__(self, n_nodes: int):
        """
        Initialize MCMF solver.
        
        Args:
            n_nodes: Number of nodes in the network
        """
        self.n = n_nodes
        self.graph: List[List[Edge]] = [[] for _ in range(n_nodes)]
        self.dist: List[float] = [INF] * n_nodes
        self.parent: List[int] = [-1] * n_nodes
        self.parent_edge: List[int] = [-1] * n_nodes
        self.in_queue: List[bool] = [False] * n_nodes
        
    def add_edge(self, u: int, v: int, capacity: int, cost: float) -> None:
        """
        Add a directed edge from u to v with given capacity and cost.
        
        This also adds a reverse edge with 0 capacity and negative cost
        for the residual graph.
        
        Args:
            u: Source node index
            v: Destination node index
            capacity: Maximum flow capacity (non-negative integer)
            cost: Cost per unit flow (can be any real number)
        """
        # Forward edge: u -> v
        forward_edge = Edge(v, capacity, cost, len(self.graph[v]))
        # Reverse edge: v -> u (for residual graph)
        reverse_edge = Edge(u, 0, -cost, len(self.graph[u]))
        
        self.graph[u].append(forward_edge)
        self.graph[v].append(reverse_edge)
    
    def add_undirected_edge(self, u: int, v: int, capacity: int, cost: float) -> None:
        """
        Add an undirected edge between u and v.
        
        This adds two directed edges, one in each direction.
        
        Args:
            u: First node index
            v: Second node index
            capacity: Maximum flow capacity in each direction
            cost: Cost per unit flow
        """
        self.add_edge(u, v, capacity, cost)
        self.add_edge(v, u, capacity, cost)
    
    def _bellman_ford(self, source: int, sink: int) -> bool:
        """
        Find shortest path from source to sink using Bellman-Ford algorithm.
        
        This handles negative edge weights which appear in the residual graph.
        
        Args:
            source: Source node index
            sink: Sink node index
            
        Returns:
            True if a path exists, False otherwise
        """
        self.dist = [INF] * self.n
        self.parent = [-1] * self.n
        self.parent_edge = [-1] * self.n
        
        self.dist[source] = 0
        
        # Relax edges V-1 times
        for _ in range(self.n - 1):
            updated = False
            for u in range(self.n):
                if self.dist[u] == INF:
                    continue
                for edge_idx, edge in enumerate(self.graph[u]):
                    if edge.residual_capacity > 0:
                        new_dist = self.dist[u] + edge.cost
                        if new_dist < self.dist[edge.to]:
                            self.dist[edge.to] = new_dist
                            self.parent[edge.to] = u
                            self.parent_edge[edge.to] = edge_idx
                            updated = True
            if not updated:
                break
        
        return self.dist[sink] < INF
    
    def _spfa(self, source: int, sink: int) -> bool:
        """
        Find shortest path using SPFA (Shortest Path Faster Algorithm).
        
        SPFA is an optimization of Bellman-Ford that typically runs faster
        in practice by using a queue to process only updated nodes.
        
        Args:
            source: Source node index
            sink: Sink node index
            
        Returns:
            True if a path exists, False otherwise
        """
        self.dist = [INF] * self.n
        self.parent = [-1] * self.n
        self.parent_edge = [-1] * self.n
        self.in_queue = [False] * self.n
        
        self.dist[source] = 0
        queue = deque([source])
        self.in_queue[source] = True
        
        while queue:
            u = queue.popleft()
            self.in_queue[u] = False
            
            for edge_idx, edge in enumerate(self.graph[u]):
                if edge.residual_capacity > 0:
                    new_dist = self.dist[u] + edge.cost
                    if new_dist < self.dist[edge.to]:
                        self.dist[edge.to] = new_dist
                        self.parent[edge.to] = u
                        self.parent_edge[edge.to] = edge_idx
                        
                        if not self.in_queue[edge.to]:
                            queue.append(edge.to)
                            self.in_queue[edge.to] = True
        
        return self.dist[sink] < INF
    
    def min_cost_max_flow(self, source: int, sink: int, 
                          use_spfa: bool = True) -> Tuple[int, float]:
        """
        Compute minimum-cost maximum-flow from source to sink.
        
        Uses the Successive Shortest Path algorithm:
        1. Find shortest path from source to sink in residual graph
        2. Augment flow along this path
        3. Repeat until no path exists
        
        Args:
            source: Source node index
            sink: Sink node index
            use_spfa: If True, use SPFA; otherwise use Bellman-Ford
            
        Returns:
            Tuple of (max_flow, min_cost)
        """
        total_flow = 0
        total_cost = 0.0
        
        find_path = self._spfa if use_spfa else self._bellman_ford
        
        while find_path(source, sink):
            # Find minimum residual capacity along the path
            path_flow = INF
            v = sink
            while v != source:
                u = self.parent[v]
                edge_idx = self.parent_edge[v]
                edge = self.graph[u][edge_idx]
                path_flow = min(path_flow, edge.residual_capacity)
                v = u
            
            # Augment flow along the path
            v = sink
            while v != source:
                u = self.parent[v]
                edge_idx = self.parent_edge[v]
                edge = self.graph[u][edge_idx]
                
                # Update forward edge
                edge.flow += path_flow
                total_cost += path_flow * edge.cost
                
                # Update reverse edge
                reverse_edge = self.graph[v][edge.reverse_idx]
                reverse_edge.flow -= path_flow
                
                v = u
            
            total_flow += path_flow
        
        return total_flow, total_cost
    
    def min_cost_flow(self, source: int, sink: int, 
                      required_flow: int, use_spfa: bool = True) -> Tuple[int, float]:
        """
        Compute minimum-cost flow with a specific flow requirement.
        
        Finds the minimum cost to send exactly required_flow units from
        source to sink. Returns actual flow achieved (may be less than
        required if network capacity is insufficient).
        
        Args:
            source: Source node index
            sink: Sink node index
            required_flow: Desired flow amount
            use_spfa: If True, use SPFA; otherwise use Bellman-Ford
            
        Returns:
            Tuple of (actual_flow, total_cost)
        """
        total_flow = 0
        total_cost = 0.0
        
        find_path = self._spfa if use_spfa else self._bellman_ford
        
        while total_flow < required_flow and find_path(source, sink):
            # Find minimum residual capacity along the path
            path_flow = required_flow - total_flow
            v = sink
            while v != source:
                u = self.parent[v]
                edge_idx = self.parent_edge[v]
                edge = self.graph[u][edge_idx]
                path_flow = min(path_flow, edge.residual_capacity)
                v = u
            
            # Augment flow along the path
            v = sink
            while v != source:
                u = self.parent[v]
                edge_idx = self.parent_edge[v]
                edge = self.graph[u][edge_idx]
                
                edge.flow += path_flow
                total_cost += path_flow * edge.cost
                
                reverse_edge = self.graph[v][edge.reverse_idx]
                reverse_edge.flow -= path_flow
                
                v = u
            
            total_flow += path_flow
        
        return total_flow, total_cost
    
    def get_flow_on_edge(self, u: int, v: int) -> int:
        """
        Get the current flow on edge (u, v).
        
        Args:
            u: Source node
            v: Destination node
            
        Returns:
            Flow amount on the edge, or 0 if edge doesn't exist
        """
        for edge in self.graph[u]:
            if edge.to == v and edge.capacity > 0:  # Forward edge
                return edge.flow
        return 0
    
    def get_flow_edges(self) -> List[Tuple[int, int, int, float]]:
        """
        Get all edges with positive flow.
        
        Returns:
            List of tuples (from_node, to_node, flow, cost)
        """
        flow_edges = []
        for u in range(self.n):
            for edge in self.graph[u]:
                if edge.flow > 0 and edge.capacity > 0:  # Forward edge with flow
                    flow_edges.append((u, edge.to, edge.flow, edge.cost))
        return flow_edges
    
    def get_cut_edges(self, source: int) -> List[Tuple[int, int]]:
        """
        Get edges in the minimum cut.
        
        After running max flow, finds edges that form the min cut
        (edges from reachable to non-reachable nodes in residual graph).
        
        Args:
            source: Source node index
            
        Returns:
            List of edge tuples (u, v) in the minimum cut
        """
        # Find reachable nodes from source in residual graph
        reachable = [False] * self.n
        queue = deque([source])
        reachable[source] = True
        
        while queue:
            u = queue.popleft()
            for edge in self.graph[u]:
                if edge.residual_capacity > 0 and not reachable[edge.to]:
                    reachable[edge.to] = True
                    queue.append(edge.to)
        
        # Find cut edges
        cut_edges = []
        for u in range(self.n):
            if reachable[u]:
                for edge in self.graph[u]:
                    if not reachable[edge.to] and edge.capacity > 0:
                        cut_edges.append((u, edge.to))
        
        return cut_edges
    
    def reset_flow(self) -> None:
        """Reset all flow values to zero."""
        for u in range(self.n):
            for edge in self.graph[u]:
                edge.flow = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the flow network.
        
        Returns:
            Dictionary with network statistics
        """
        total_capacity = 0
        total_flow = 0
        num_edges = 0
        
        for u in range(self.n):
            for edge in self.graph[u]:
                if edge.capacity > 0:  # Forward edges only
                    num_edges += 1
                    total_capacity += edge.capacity
                    total_flow += edge.flow
        
        return {
            'num_nodes': self.n,
            'num_edges': num_edges,
            'total_capacity': total_capacity,
            'total_flow': total_flow,
            'utilization': total_flow / total_capacity if total_capacity > 0 else 0
        }
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"MCMF(nodes={stats['num_nodes']}, edges={stats['num_edges']}, "
                f"flow={stats['total_flow']}/{stats['total_capacity']})")


class MCMFMatrix:
    """
    Alternative MCMF implementation using adjacency matrix.
    
    This is simpler but uses O(V^2) space. Better for dense graphs
    or when V is small.
    """
    
    def __init__(self, n_nodes: int):
        """
        Initialize MCMF solver with matrix representation.
        
        Args:
            n_nodes: Number of nodes in the network
        """
        self.n = n_nodes
        self.capacity = [[0] * n_nodes for _ in range(n_nodes)]
        self.cost = [[0.0] * n_nodes for _ in range(n_nodes)]
        self.flow = [[0] * n_nodes for _ in range(n_nodes)]
    
    def add_edge(self, u: int, v: int, cap: int, cost: float) -> None:
        """
        Add a directed edge from u to v.
        
        Args:
            u: Source node
            v: Destination node
            cap: Edge capacity
            cost: Cost per unit flow
        """
        self.capacity[u][v] = cap
        self.cost[u][v] = cost
        self.cost[v][u] = -cost  # Reverse edge cost for residual graph
    
    def _bellman_ford(self, source: int, sink: int) -> Tuple[bool, List[int]]:
        """
        Find shortest path using Bellman-Ford.
        
        Args:
            source: Source node
            sink: Sink node
            
        Returns:
            Tuple of (path_exists, parent_array)
        """
        dist = [INF] * self.n
        parent = [-1] * self.n
        dist[source] = 0
        
        for _ in range(self.n - 1):
            updated = False
            for u in range(self.n):
                if dist[u] == INF:
                    continue
                for v in range(self.n):
                    residual = self.capacity[u][v] - self.flow[u][v]
                    if residual > 0:
                        new_dist = dist[u] + self.cost[u][v]
                        if new_dist < dist[v]:
                            dist[v] = new_dist
                            parent[v] = u
                            updated = True
            if not updated:
                break
        
        return dist[sink] < INF, parent
    
    def min_cost_max_flow(self, source: int, sink: int) -> Tuple[int, float]:
        """
        Compute minimum-cost maximum-flow.
        
        Args:
            source: Source node
            sink: Sink node
            
        Returns:
            Tuple of (max_flow, min_cost)
        """
        total_flow = 0
        total_cost = 0.0
        
        while True:
            found, parent = self._bellman_ford(source, sink)
            if not found:
                break
            
            # Find minimum capacity along path
            path_flow = INF
            v = sink
            while v != source:
                u = parent[v]
                residual = self.capacity[u][v] - self.flow[u][v]
                path_flow = min(path_flow, residual)
                v = u
            
            # Augment flow along path
            v = sink
            while v != source:
                u = parent[v]
                self.flow[u][v] += path_flow
                self.flow[v][u] -= path_flow
                total_cost += path_flow * self.cost[u][v]
                v = u
            
            total_flow += path_flow
        
        return total_flow, total_cost
    
    def get_flow_on_edge(self, u: int, v: int) -> int:
        """Get flow on edge (u, v)."""
        return max(0, self.flow[u][v])
    
    def reset_flow(self) -> None:
        """Reset all flows to zero."""
        for i in range(self.n):
            for j in range(self.n):
                self.flow[i][j] = 0


def create_bipartite_mcmf(left_nodes: int, right_nodes: int, 
                          edges: List[Tuple[int, int, float]]) -> Tuple[MCMF, int, int]:
    """
    Create MCMF network for bipartite matching with costs.
    
    This is a convenience function for the common case of matching
    left nodes to right nodes (e.g., IO pads to bump pads).
    
    Args:
        left_nodes: Number of nodes on left side
        right_nodes: Number of nodes on right side
        edges: List of (left_idx, right_idx, cost) tuples
        
    Returns:
        Tuple of (mcmf_solver, source_node, sink_node)
    """
    # Node layout: [source, left_0..left_n-1, right_0..right_m-1, sink]
    total_nodes = 2 + left_nodes + right_nodes
    source = 0
    sink = total_nodes - 1
    
    mcmf = MCMF(total_nodes)
    
    # Source to left nodes (capacity 1, cost 0)
    for i in range(left_nodes):
        mcmf.add_edge(source, 1 + i, 1, 0)
    
    # Left to right edges
    for left_idx, right_idx, cost in edges:
        left_node = 1 + left_idx
        right_node = 1 + left_nodes + right_idx
        mcmf.add_edge(left_node, right_node, 1, cost)
    
    # Right nodes to sink (capacity 1, cost 0)
    for j in range(right_nodes):
        mcmf.add_edge(1 + left_nodes + j, sink, 1, 0)
    
    return mcmf, source, sink


def test_mcmf():
    """Test MCMF implementation with simple examples."""
    print("Testing MCMF implementation...")
    
    # Test 1: Simple 4-node network
    # Source(0) -> A(1) -> B(2) -> Sink(3)
    #           \-> C(2) ->/
    mcmf = MCMF(4)
    mcmf.add_edge(0, 1, 2, 1)  # S->A: cap=2, cost=1
    mcmf.add_edge(0, 2, 2, 2)  # S->C: cap=2, cost=2
    mcmf.add_edge(1, 3, 2, 1)  # A->T: cap=2, cost=1
    mcmf.add_edge(2, 3, 2, 1)  # C->T: cap=2, cost=1
    
    flow, cost = mcmf.min_cost_max_flow(0, 3)
    print(f"Test 1: flow={flow}, cost={cost}")
    assert flow == 4, f"Expected flow=4, got {flow}"
    assert cost == 10, f"Expected cost=10, got {cost}"
    
    # Test 2: Bipartite matching
    # 3 left nodes, 3 right nodes
    edges = [
        (0, 0, 1), (0, 1, 2),  # Left 0 can go to Right 0 (cost 1) or Right 1 (cost 2)
        (1, 1, 1), (1, 2, 3),  # Left 1 can go to Right 1 (cost 1) or Right 2 (cost 3)
        (2, 0, 2), (2, 2, 1),  # Left 2 can go to Right 0 (cost 2) or Right 2 (cost 1)
    ]
    mcmf, source, sink = create_bipartite_mcmf(3, 3, edges)
    flow, cost = mcmf.min_cost_max_flow(source, sink)
    print(f"Test 2: flow={flow}, cost={cost}")
    assert flow == 3, f"Expected flow=3, got {flow}"
    # Optimal: L0->R0(1), L1->R1(1), L2->R2(1) = cost 3
    assert cost == 3, f"Expected cost=3, got {cost}"
    
    # Test 3: Matrix implementation
    mcmf_mat = MCMFMatrix(4)
    mcmf_mat.add_edge(0, 1, 2, 1)
    mcmf_mat.add_edge(0, 2, 2, 2)
    mcmf_mat.add_edge(1, 3, 2, 1)
    mcmf_mat.add_edge(2, 3, 2, 1)
    
    flow, cost = mcmf_mat.min_cost_max_flow(0, 3)
    print(f"Test 3 (Matrix): flow={flow}, cost={cost}")
    assert flow == 4, f"Expected flow=4, got {flow}"
    
    print("All MCMF tests passed!")


if __name__ == "__main__":
    test_mcmf()
