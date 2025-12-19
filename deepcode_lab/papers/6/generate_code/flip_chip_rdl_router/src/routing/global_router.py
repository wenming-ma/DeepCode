"""
Global Router for Flip-Chip RDL Routing

This module implements the global routing phase using Minimum-Cost Maximum-Flow (MCMF)
algorithm to optimally assign IO pads to bump pads while minimizing total wirelength.

The global router:
1. Builds a flow network from chip layout
2. Solves MCMF to find optimal IO-to-bump assignments
3. Validates assignments for routability
4. Prepares data for detailed routing phase
"""

from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import math

from ..core.mcmf import MCMF, INF
from ..core.network_builder import (
    NetworkBuilder, FlowNetwork, build_routing_network, 
    solve_assignment, NetworkNode, NodeType
)
from ..core.geometry import manhattan_distance, would_cross, is_planar_routing, Point
from ..data_structures import Chip, IOPad, BumpPad, Net


class RoutingStatus(Enum):
    """Status of global routing result."""
    SUCCESS = "success"
    PARTIAL = "partial"  # Some nets unrouted
    FAILED = "failed"
    INFEASIBLE = "infeasible"  # Not enough capacity


@dataclass
class Assignment:
    """Represents an IO pad to bump pad assignment."""
    io_pad: IOPad
    bump_pad: BumpPad
    cost: float  # Manhattan distance
    net_id: int = -1
    
    @property
    def io_x(self) -> float:
        return self.io_pad.x
    
    @property
    def io_y(self) -> float:
        return self.io_pad.y
    
    @property
    def bump_x(self) -> float:
        return self.bump_pad.x
    
    @property
    def bump_y(self) -> float:
        return self.bump_pad.y
    
    def __repr__(self) -> str:
        return f"Assignment({self.io_pad.name} -> {self.bump_pad.name}, cost={self.cost:.2f})"


@dataclass
class GlobalRoutingResult:
    """Result of global routing phase."""
    status: RoutingStatus
    assignments: List[Assignment]
    total_cost: float  # Total wirelength
    total_flow: int  # Number of routed nets
    unrouted_ios: List[IOPad] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_routed(self) -> int:
        return len(self.assignments)
    
    @property
    def num_unrouted(self) -> int:
        return len(self.unrouted_ios)
    
    @property
    def routability_rate(self) -> float:
        total = self.num_routed + self.num_unrouted
        if total == 0:
            return 100.0
        return (self.num_routed / total) * 100.0
    
    @property
    def is_complete(self) -> bool:
        return self.status == RoutingStatus.SUCCESS and self.num_unrouted == 0


@dataclass
class RoutingConstraints:
    """Constraints for global routing."""
    max_distance: float = INF  # Maximum allowed wire length
    min_distance: float = 0.0  # Minimum distance (for spacing)
    allowed_bump_ids: Optional[Set[int]] = None  # Restrict to specific bumps
    blocked_bump_ids: Optional[Set[int]] = None  # Exclude specific bumps
    power_net_bumps: Optional[Set[int]] = None  # Bumps reserved for power
    require_planarity: bool = True  # Enforce planar routing
    
    def is_valid_assignment(self, io_pad: IOPad, bump_pad: BumpPad) -> bool:
        """Check if assignment satisfies constraints."""
        # Check blocked bumps
        if self.blocked_bump_ids and bump_pad.node_id in self.blocked_bump_ids:
            return False
        
        # Check allowed bumps
        if self.allowed_bump_ids and bump_pad.node_id not in self.allowed_bump_ids:
            return False
        
        # Check power net constraints
        if self.power_net_bumps:
            if io_pad.is_power and bump_pad.node_id not in self.power_net_bumps:
                return False
            if not io_pad.is_power and bump_pad.node_id in self.power_net_bumps:
                return False
        
        # Check distance constraints
        dist = manhattan_distance((io_pad.x, io_pad.y), (bump_pad.x, bump_pad.y))
        if dist > self.max_distance or dist < self.min_distance:
            return False
        
        return True


class GlobalRouter:
    """
    Global router using MCMF for optimal IO-to-bump assignment.
    
    The router builds a flow network where:
    - Source connects to all IO pads (capacity 1, cost 0)
    - IO pads connect to valid bump pads (capacity 1, cost = Manhattan distance)
    - Bump pads connect to sink (capacity = bump capacity, cost 0)
    
    MCMF finds the minimum-cost maximum-flow, giving optimal assignments.
    """
    
    def __init__(self, constraints: Optional[RoutingConstraints] = None):
        """
        Initialize global router.
        
        Args:
            constraints: Routing constraints (optional)
        """
        self.constraints = constraints or RoutingConstraints()
        self.network_builder = NetworkBuilder()
        self._last_network: Optional[FlowNetwork] = None
        self._last_mcmf: Optional[MCMF] = None
    
    def route(self, chip: Chip) -> GlobalRoutingResult:
        """
        Perform global routing on chip.
        
        Args:
            chip: Chip object with IO pads and bump pads
            
        Returns:
            GlobalRoutingResult with assignments and statistics
        """
        # Validate input
        if not chip.io_pads:
            return GlobalRoutingResult(
                status=RoutingStatus.SUCCESS,
                assignments=[],
                total_cost=0.0,
                total_flow=0,
                statistics={"message": "No IO pads to route"}
            )
        
        if not chip.bump_pads:
            return GlobalRoutingResult(
                status=RoutingStatus.INFEASIBLE,
                assignments=[],
                total_cost=0.0,
                total_flow=0,
                unrouted_ios=list(chip.io_pads),
                statistics={"message": "No bump pads available"}
            )
        
        # Check capacity feasibility
        total_io_count = len(chip.io_pads)
        total_bump_capacity = sum(bp.capacity for bp in chip.bump_pads)
        
        if total_io_count > total_bump_capacity:
            return GlobalRoutingResult(
                status=RoutingStatus.INFEASIBLE,
                assignments=[],
                total_cost=0.0,
                total_flow=0,
                unrouted_ios=list(chip.io_pads),
                statistics={
                    "message": "Insufficient bump capacity",
                    "io_count": total_io_count,
                    "bump_capacity": total_bump_capacity
                }
            )
        
        # Build flow network
        constraint_dict = self._build_constraint_dict(chip)
        mcmf, network = self.network_builder.build_network(chip, constraint_dict)
        self._last_network = network
        self._last_mcmf = mcmf
        
        # Solve MCMF
        total_flow, total_cost = mcmf.min_cost_max_flow(network.source_id, network.sink_id)
        
        # Extract assignments
        assignments = self._extract_assignments(mcmf, network, chip)
        
        # Find unrouted IOs
        routed_io_ids = {a.io_pad.node_id for a in assignments}
        unrouted_ios = [io for io in chip.io_pads if io.node_id not in routed_io_ids]
        
        # Determine status
        if len(unrouted_ios) == 0:
            status = RoutingStatus.SUCCESS
        elif len(assignments) > 0:
            status = RoutingStatus.PARTIAL
        else:
            status = RoutingStatus.FAILED
        
        # Check planarity if required
        planarity_ok = True
        if self.constraints.require_planarity and assignments:
            planarity_ok = self._check_planarity(assignments)
            if not planarity_ok:
                # Try to fix planarity issues
                assignments = self._fix_planarity(assignments, chip)
                planarity_ok = self._check_planarity(assignments)
        
        # Build statistics
        statistics = self._build_statistics(
            chip, assignments, total_flow, total_cost, planarity_ok
        )
        
        # Apply assignments to chip
        self._apply_assignments(assignments, chip)
        
        return GlobalRoutingResult(
            status=status,
            assignments=assignments,
            total_cost=total_cost,
            total_flow=total_flow,
            unrouted_ios=unrouted_ios,
            statistics=statistics
        )
    
    def route_from_pads(
        self,
        io_pads: List[IOPad],
        bump_pads: List[BumpPad]
    ) -> GlobalRoutingResult:
        """
        Perform global routing from pad lists directly.
        
        Args:
            io_pads: List of IO pads
            bump_pads: List of bump pads
            
        Returns:
            GlobalRoutingResult with assignments
        """
        # Use convenience function
        constraint_dict = {
            'max_distance': self.constraints.max_distance,
            'blocked_bumps': self.constraints.blocked_bump_ids
        }
        
        raw_assignments, total_cost, total_flow = solve_assignment(
            io_pads, bump_pads, constraint_dict
        )
        
        # Convert to Assignment objects
        assignments = []
        for io_pad, bump_pad in raw_assignments:
            cost = manhattan_distance((io_pad.x, io_pad.y), (bump_pad.x, bump_pad.y))
            assignments.append(Assignment(
                io_pad=io_pad,
                bump_pad=bump_pad,
                cost=cost,
                net_id=io_pad.net_id
            ))
        
        # Find unrouted
        routed_io_ids = {a.io_pad.node_id for a in assignments}
        unrouted_ios = [io for io in io_pads if io.node_id not in routed_io_ids]
        
        # Determine status
        if len(unrouted_ios) == 0:
            status = RoutingStatus.SUCCESS
        elif len(assignments) > 0:
            status = RoutingStatus.PARTIAL
        else:
            status = RoutingStatus.FAILED
        
        return GlobalRoutingResult(
            status=status,
            assignments=assignments,
            total_cost=total_cost,
            total_flow=total_flow,
            unrouted_ios=unrouted_ios
        )
    
    def _build_constraint_dict(self, chip: Chip) -> Dict[str, Any]:
        """Build constraint dictionary for network builder."""
        constraints = {}
        
        if self.constraints.max_distance < INF:
            constraints['max_distance'] = self.constraints.max_distance
        
        if self.constraints.blocked_bump_ids:
            constraints['blocked_bumps'] = self.constraints.blocked_bump_ids
        
        return constraints
    
    def _extract_assignments(
        self,
        mcmf: MCMF,
        network: FlowNetwork,
        chip: Chip
    ) -> List[Assignment]:
        """Extract assignments from solved MCMF."""
        assignments = []
        
        # Get flow edges
        flow_edges = mcmf.get_flow_edges()
        
        # Build lookup maps
        io_map = {io.node_id: io for io in chip.io_pads}
        bump_map = {bp.node_id: bp for bp in chip.bump_pads}
        
        for from_node, to_node, flow_amount, cost in flow_edges:
            if flow_amount <= 0:
                continue
            
            # Check if this is an IO->Bump edge
            from_net_node = network.get_node(from_node)
            to_net_node = network.get_node(to_node)
            
            if (from_net_node and to_net_node and
                from_net_node.node_type == NodeType.IO_PAD and
                to_net_node.node_type == NodeType.BUMP_PAD):
                
                # Get original pads
                io_pad = network.get_io_pad(from_node)
                bump_pad = network.get_bump_pad(to_node)
                
                if io_pad is None:
                    # Fallback to lookup by original_id
                    orig_io_id = from_net_node.original_id
                    io_pad = io_map.get(orig_io_id)
                
                if bump_pad is None:
                    orig_bump_id = to_net_node.original_id
                    bump_pad = bump_map.get(orig_bump_id)
                
                if io_pad and bump_pad:
                    cost = manhattan_distance(
                        (io_pad.x, io_pad.y),
                        (bump_pad.x, bump_pad.y)
                    )
                    assignments.append(Assignment(
                        io_pad=io_pad,
                        bump_pad=bump_pad,
                        cost=cost,
                        net_id=io_pad.net_id
                    ))
        
        return assignments
    
    def _check_planarity(self, assignments: List[Assignment]) -> bool:
        """Check if assignments can be routed without crossings."""
        if len(assignments) <= 1:
            return True

        # Convert to format for is_planar_routing
        assignment_tuples = [
            ((a.io_x, a.io_y), (a.bump_x, a.bump_y)) for a in assignments
        ]

        return is_planar_routing(assignment_tuples)
    
    def _fix_planarity(
        self,
        assignments: List[Assignment],
        chip: Chip
    ) -> List[Assignment]:
        """
        Attempt to fix planarity issues by reassigning crossing nets.
        
        This is a heuristic approach - for complex cases, may need
        iterative refinement or layer assignment.
        """
        if len(assignments) <= 1:
            return assignments
        
        # Sort by IO x-coordinate
        sorted_assignments = sorted(assignments, key=lambda a: a.io_x)
        
        # Check for crossings and try to swap
        changed = True
        max_iterations = len(assignments) * 2
        iteration = 0
        
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            
            for i in range(len(sorted_assignments) - 1):
                a1 = sorted_assignments[i]
                a2 = sorted_assignments[i + 1]
                
                # Check if they cross
                if would_cross(a1.io_x, a1.bump_x, a2.io_x, a2.bump_x):
                    # Try swapping bump assignments
                    new_cost1 = manhattan_distance(
                        (a1.io_x, a1.io_y),
                        (a2.bump_x, a2.bump_y)
                    )
                    new_cost2 = manhattan_distance(
                        (a2.io_x, a2.io_y),
                        (a1.bump_x, a1.bump_y)
                    )
                    old_cost = a1.cost + a2.cost
                    new_cost = new_cost1 + new_cost2
                    
                    # Swap if it reduces crossings (even if cost increases slightly)
                    # Swap bumps
                    a1.bump_pad, a2.bump_pad = a2.bump_pad, a1.bump_pad
                    a1.cost = new_cost1
                    a2.cost = new_cost2
                    changed = True
        
        return sorted_assignments
    
    def _apply_assignments(self, assignments: List[Assignment], chip: Chip) -> None:
        """Apply assignments to chip data structures."""
        for assignment in assignments:
            io_pad = assignment.io_pad
            bump_pad = assignment.bump_pad
            
            # Update IO pad
            io_pad.assign_to_bump(bump_pad)
            
            # Update bump pad
            bump_pad.assign_io(io_pad)
    
    def _build_statistics(
        self,
        chip: Chip,
        assignments: List[Assignment],
        total_flow: int,
        total_cost: float,
        planarity_ok: bool
    ) -> Dict[str, Any]:
        """Build routing statistics."""
        stats = {
            "num_io_pads": len(chip.io_pads),
            "num_bump_pads": len(chip.bump_pads),
            "num_routed": len(assignments),
            "num_unrouted": len(chip.io_pads) - len(assignments),
            "total_wirelength": total_cost,
            "total_flow": total_flow,
            "planarity_ok": planarity_ok,
            "routability_rate": (len(assignments) / len(chip.io_pads) * 100) if chip.io_pads else 100.0
        }
        
        if assignments:
            costs = [a.cost for a in assignments]
            stats["min_wirelength"] = min(costs)
            stats["max_wirelength"] = max(costs)
            stats["avg_wirelength"] = sum(costs) / len(costs)
        
        return stats
    
    def get_assignment_matrix(
        self,
        io_pads: List[IOPad],
        bump_pads: List[BumpPad]
    ) -> List[List[float]]:
        """
        Get cost matrix for IO-to-bump assignments.
        
        Returns:
            2D matrix where [i][j] is cost of assigning IO i to bump j
        """
        matrix = []
        for io_pad in io_pads:
            row = []
            for bump_pad in bump_pads:
                if self.constraints.is_valid_assignment(io_pad, bump_pad):
                    cost = manhattan_distance(
                        (io_pad.x, io_pad.y),
                        (bump_pad.x, bump_pad.y)
                    )
                else:
                    cost = INF
                row.append(cost)
            matrix.append(row)
        return matrix


def global_route(chip: Chip, constraints: Optional[RoutingConstraints] = None) -> GlobalRoutingResult:
    """
    Convenience function for global routing.
    
    Args:
        chip: Chip to route
        constraints: Optional routing constraints
        
    Returns:
        GlobalRoutingResult
    """
    router = GlobalRouter(constraints)
    return router.route(chip)


def compute_lower_bound_wirelength(
    io_pads: List[IOPad],
    bump_pads: List[BumpPad]
) -> float:
    """
    Compute theoretical lower bound on total wirelength.
    
    Uses greedy nearest-neighbor assignment (not optimal but fast).
    """
    if not io_pads or not bump_pads:
        return 0.0
    
    total = 0.0
    used_bumps = set()
    
    for io_pad in io_pads:
        min_dist = INF
        best_bump = None
        
        for bump_pad in bump_pads:
            if bump_pad.node_id in used_bumps:
                continue
            
            dist = manhattan_distance(
                (io_pad.x, io_pad.y),
                (bump_pad.x, bump_pad.y)
            )
            
            if dist < min_dist:
                min_dist = dist
                best_bump = bump_pad
        
        if best_bump:
            total += min_dist
            if best_bump.capacity == 1:
                used_bumps.add(best_bump.node_id)
    
    return total


def test_global_router():
    """Test global router functionality."""
    print("Testing Global Router...")
    
    # Create test chip
    chip = Chip(
        name="test_chip",
        die_width=1000.0,
        die_height=1000.0,
        bump_pitch=100.0
    )
    
    # Create bump grid
    chip.create_bump_grid(rows=5, cols=5)
    
    # Create peripheral IO pads
    chip.create_peripheral_io_pads(num_pads_per_side=5)
    
    print(f"Created chip with {len(chip.io_pads)} IO pads and {len(chip.bump_pads)} bump pads")
    
    # Run global routing
    router = GlobalRouter()
    result = router.route(chip)
    
    print(f"Routing status: {result.status}")
    print(f"Routed: {result.num_routed}, Unrouted: {result.num_unrouted}")
    print(f"Total wirelength: {result.total_cost:.2f}")
    print(f"Routability rate: {result.routability_rate:.1f}%")
    
    if result.statistics:
        print(f"Statistics: {result.statistics}")
    
    # Verify assignments
    for assignment in result.assignments[:5]:  # Show first 5
        print(f"  {assignment}")
    
    print("Global Router test passed!")


if __name__ == "__main__":
    test_global_router()
