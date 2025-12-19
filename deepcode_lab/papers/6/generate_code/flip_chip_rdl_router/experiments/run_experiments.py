#!/usr/bin/env python3
"""
Experiment Runner for Flip-Chip RDL Router

This module provides comprehensive experiment execution capabilities for
benchmarking and evaluating the network-flow-based RDL routing algorithm.
It supports various benchmark configurations, scalability analysis, and
comparison studies.

Author: Flip-Chip RDL Router Team
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import math

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_structures import Chip, BumpPad, IOPad, Net
from src.core import MCMF, NetworkBuilder, RiverRouter, Point, manhattan_distance
from src.routing import (
    GlobalRouter, DetailedRouter, LayerAssigner,
    GlobalRoutingResult, DetailedRoutingResult, LayerAssignmentResult,
    RoutingConstraints, RoutingStatus, DetailedRoutingStatus,
    full_routing_pipeline, quick_route
)
from src.utils import ChipParser, BenchmarkGenerator, load_chip, save_chip_to_json


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    num_ios: int
    bump_rows: int
    bump_cols: int
    die_width: float = 5000.0
    die_height: float = 5000.0
    bump_pitch: float = 100.0
    wire_width: float = 10.0
    wire_spacing: float = 10.0
    enable_drc: bool = True
    num_layers: int = 1
    num_runs: int = 1  # Number of runs for averaging
    description: str = ""


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    config_name: str
    run_id: int
    success: bool
    
    # Timing metrics (seconds)
    total_time: float = 0.0
    global_routing_time: float = 0.0
    detailed_routing_time: float = 0.0
    layer_assignment_time: float = 0.0
    
    # Routing metrics
    num_ios: int = 0
    num_bumps: int = 0
    num_routed: int = 0
    num_failed: int = 0
    routability_rate: float = 0.0
    
    # Wirelength metrics
    total_wirelength: float = 0.0
    average_wirelength: float = 0.0
    min_wirelength: float = 0.0
    max_wirelength: float = 0.0
    lower_bound_wirelength: float = 0.0
    wirelength_ratio: float = 0.0  # actual / lower_bound
    
    # Quality metrics
    num_bends: int = 0
    average_bends: float = 0.0
    num_drc_violations: int = 0
    num_crossings: int = 0
    
    # Memory usage (if available)
    peak_memory_mb: float = 0.0
    
    # Error information
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ExperimentSummary:
    """Summary of multiple experiment runs."""
    config_name: str
    num_runs: int
    
    # Averaged metrics
    avg_total_time: float = 0.0
    std_total_time: float = 0.0
    avg_routability: float = 0.0
    avg_wirelength: float = 0.0
    avg_wirelength_ratio: float = 0.0
    
    # Best/worst cases
    best_routability: float = 0.0
    worst_routability: float = 0.0
    best_wirelength: float = float('inf')
    worst_wirelength: float = 0.0
    
    # Success rate
    success_rate: float = 0.0
    
    results: List[ExperimentResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d['results'] = [r.to_dict() for r in self.results]
        return d


class ExperimentRunner:
    """
    Main experiment runner for flip-chip RDL routing benchmarks.
    
    Supports:
    - Standard benchmark configurations
    - Scalability analysis
    - Algorithm comparison
    - Statistical analysis with multiple runs
    """
    
    # Standard benchmark configurations
    STANDARD_BENCHMARKS = {
        'tiny': ExperimentConfig(
            name='tiny',
            num_ios=16,
            bump_rows=4,
            bump_cols=4,
            die_width=1000.0,
            die_height=1000.0,
            description='Tiny test case for debugging'
        ),
        'small': ExperimentConfig(
            name='small',
            num_ios=100,
            bump_rows=10,
            bump_cols=10,
            die_width=2000.0,
            die_height=2000.0,
            description='Small benchmark (100 IOs)'
        ),
        'medium': ExperimentConfig(
            name='medium',
            num_ios=400,
            bump_rows=20,
            bump_cols=20,
            die_width=4000.0,
            die_height=4000.0,
            description='Medium benchmark (400 IOs)'
        ),
        'large': ExperimentConfig(
            name='large',
            num_ios=900,
            bump_rows=30,
            bump_cols=30,
            die_width=6000.0,
            die_height=6000.0,
            description='Large benchmark (900 IOs)'
        ),
        'xlarge': ExperimentConfig(
            name='xlarge',
            num_ios=1600,
            bump_rows=40,
            bump_cols=40,
            die_width=8000.0,
            die_height=8000.0,
            description='Extra large benchmark (1600 IOs)'
        ),
        'industrial': ExperimentConfig(
            name='industrial',
            num_ios=2500,
            bump_rows=50,
            bump_cols=50,
            die_width=10000.0,
            die_height=10000.0,
            description='Industrial-scale benchmark (2500 IOs)'
        ),
    }
    
    # Scalability test configurations
    SCALABILITY_CONFIGS = [
        (25, 5, 5),      # 25 IOs
        (100, 10, 10),   # 100 IOs
        (225, 15, 15),   # 225 IOs
        (400, 20, 20),   # 400 IOs
        (625, 25, 25),   # 625 IOs
        (900, 30, 30),   # 900 IOs
        (1225, 35, 35),  # 1225 IOs
        (1600, 40, 40),  # 1600 IOs
        (2025, 45, 45),  # 2025 IOs
        (2500, 50, 50),  # 2500 IOs
    ]
    
    def __init__(self, output_dir: str = "results", verbose: bool = True):
        """
        Initialize experiment runner.
        
        Args:
            output_dir: Directory for saving results
            verbose: Whether to print progress information
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.results: List[ExperimentResult] = []
        self.summaries: Dict[str, ExperimentSummary] = {}
    
    def log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def create_chip_from_config(self, config: ExperimentConfig) -> Chip:
        """
        Create a chip instance from experiment configuration.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Configured Chip instance
        """
        chip = Chip(
            name=f"benchmark_{config.name}",
            die_width=config.die_width,
            die_height=config.die_height,
            bump_pitch=config.bump_pitch,
            wire_width=config.wire_width,
            wire_spacing=config.wire_spacing
        )
        
        # Create bump pad grid
        chip.create_bump_grid(
            rows=config.bump_rows,
            cols=config.bump_cols,
            pitch=config.bump_pitch
        )
        
        # Calculate IOs per side (distribute evenly)
        ios_per_side = config.num_ios // 4
        remainder = config.num_ios % 4
        
        # Create peripheral IO pads
        chip.create_peripheral_io_pads(
            pads_per_side=ios_per_side + (1 if remainder > 0 else 0)
        )
        
        # Trim to exact number of IOs if needed
        while len(chip.io_pads) > config.num_ios:
            chip.io_pads.pop()
        
        return chip
    
    def run_single_experiment(
        self,
        config: ExperimentConfig,
        run_id: int = 0
    ) -> ExperimentResult:
        """
        Run a single experiment with the given configuration.
        
        Args:
            config: Experiment configuration
            run_id: Run identifier for multiple runs
            
        Returns:
            ExperimentResult with all metrics
        """
        result = ExperimentResult(
            config_name=config.name,
            run_id=run_id,
            success=False
        )
        
        try:
            # Create chip
            chip = self.create_chip_from_config(config)
            result.num_ios = len(chip.io_pads)
            result.num_bumps = len(chip.bump_pads)
            
            # Run routing pipeline with timing
            start_time = time.time()
            
            # Global routing
            global_start = time.time()
            global_router = GlobalRouter()
            global_result = global_router.route(chip)
            result.global_routing_time = time.time() - global_start
            
            # Detailed routing
            detailed_start = time.time()
            detailed_router = DetailedRouter(
                wire_width=config.wire_width,
                wire_spacing=config.wire_spacing,
                enable_drc=config.enable_drc
            )
            detailed_result = detailed_router.route(chip)
            result.detailed_routing_time = time.time() - detailed_start
            
            # Layer assignment
            layer_start = time.time()
            layer_assigner = LayerAssigner(num_layers=config.num_layers)
            layer_result = layer_assigner.assign(chip)
            result.layer_assignment_time = time.time() - layer_start
            
            result.total_time = time.time() - start_time
            
            # Extract metrics from global routing
            if global_result:
                result.num_routed = global_result.num_routed
                result.num_failed = global_result.num_unrouted
                result.routability_rate = global_result.routability_rate
                
                # Calculate wirelength metrics
                if global_result.assignments:
                    wirelengths = [a.cost for a in global_result.assignments]
                    result.total_wirelength = sum(wirelengths)
                    result.average_wirelength = result.total_wirelength / len(wirelengths) if wirelengths else 0
                    result.min_wirelength = min(wirelengths) if wirelengths else 0
                    result.max_wirelength = max(wirelengths) if wirelengths else 0
            
            # Extract metrics from detailed routing
            if detailed_result:
                result.num_drc_violations = len(detailed_result.drc_violations)
                
                # Count bends
                total_bends = 0
                for route in detailed_result.routes:
                    total_bends += route.num_bends
                result.num_bends = total_bends
                result.average_bends = total_bends / len(detailed_result.routes) if detailed_result.routes else 0
            
            # Calculate lower bound wirelength (sum of Manhattan distances)
            lower_bound = 0.0
            for io_pad in chip.io_pads:
                if io_pad.assigned_bump:
                    lower_bound += manhattan_distance(
                        (io_pad.x, io_pad.y),
                        (io_pad.assigned_bump.x, io_pad.assigned_bump.y)
                    )
            result.lower_bound_wirelength = lower_bound
            
            if lower_bound > 0:
                result.wirelength_ratio = result.total_wirelength / lower_bound
            
            result.success = True
            
        except Exception as e:
            result.error_message = str(e)
            result.success = False
        
        return result
    
    def run_benchmark(
        self,
        benchmark_name: str,
        num_runs: int = 1
    ) -> ExperimentSummary:
        """
        Run a standard benchmark configuration.
        
        Args:
            benchmark_name: Name of benchmark (tiny, small, medium, large, xlarge, industrial)
            num_runs: Number of runs for statistical averaging
            
        Returns:
            ExperimentSummary with aggregated results
        """
        if benchmark_name not in self.STANDARD_BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {benchmark_name}. "
                           f"Available: {list(self.STANDARD_BENCHMARKS.keys())}")
        
        config = self.STANDARD_BENCHMARKS[benchmark_name]
        config.num_runs = num_runs
        
        self.log(f"\n{'='*60}")
        self.log(f"Running benchmark: {benchmark_name}")
        self.log(f"Configuration: {config.num_ios} IOs, {config.bump_rows}x{config.bump_cols} bumps")
        self.log(f"Number of runs: {num_runs}")
        self.log(f"{'='*60}")
        
        results = []
        for run_id in range(num_runs):
            self.log(f"\nRun {run_id + 1}/{num_runs}...")
            result = self.run_single_experiment(config, run_id)
            results.append(result)
            self.results.append(result)
            
            if result.success:
                self.log(f"  Time: {result.total_time:.3f}s, "
                        f"Routability: {result.routability_rate:.1%}, "
                        f"Wirelength: {result.total_wirelength:.1f}")
            else:
                self.log(f"  FAILED: {result.error_message}")
        
        # Create summary
        summary = self._create_summary(config.name, results)
        self.summaries[benchmark_name] = summary
        
        return summary
    
    def run_all_benchmarks(self, num_runs: int = 1) -> Dict[str, ExperimentSummary]:
        """
        Run all standard benchmarks.
        
        Args:
            num_runs: Number of runs per benchmark
            
        Returns:
            Dictionary of benchmark name to summary
        """
        self.log("\n" + "="*70)
        self.log("RUNNING ALL STANDARD BENCHMARKS")
        self.log("="*70)
        
        summaries = {}
        for name in self.STANDARD_BENCHMARKS:
            try:
                summary = self.run_benchmark(name, num_runs)
                summaries[name] = summary
            except Exception as e:
                self.log(f"Error running {name}: {e}")
        
        self._print_summary_table(summaries)
        return summaries
    
    def run_scalability_analysis(
        self,
        max_size: int = 2500,
        num_runs: int = 1
    ) -> List[ExperimentSummary]:
        """
        Run scalability analysis with increasing problem sizes.
        
        Args:
            max_size: Maximum number of IOs to test
            num_runs: Number of runs per configuration
            
        Returns:
            List of experiment summaries
        """
        self.log("\n" + "="*70)
        self.log("SCALABILITY ANALYSIS")
        self.log("="*70)
        
        summaries = []
        
        for num_ios, rows, cols in self.SCALABILITY_CONFIGS:
            if num_ios > max_size:
                break
            
            config = ExperimentConfig(
                name=f"scale_{num_ios}",
                num_ios=num_ios,
                bump_rows=rows,
                bump_cols=cols,
                die_width=rows * 200.0,
                die_height=cols * 200.0,
                num_runs=num_runs,
                description=f"Scalability test with {num_ios} IOs"
            )
            
            self.log(f"\nTesting {num_ios} IOs ({rows}x{cols} grid)...")
            
            results = []
            for run_id in range(num_runs):
                result = self.run_single_experiment(config, run_id)
                results.append(result)
                self.results.append(result)
            
            summary = self._create_summary(config.name, results)
            summaries.append(summary)
            
            self.log(f"  Avg time: {summary.avg_total_time:.3f}s, "
                    f"Routability: {summary.avg_routability:.1%}")
        
        self._print_scalability_table(summaries)
        return summaries
    
    def run_custom_experiment(
        self,
        config: ExperimentConfig,
        num_runs: int = 1
    ) -> ExperimentSummary:
        """
        Run a custom experiment configuration.
        
        Args:
            config: Custom experiment configuration
            num_runs: Number of runs
            
        Returns:
            ExperimentSummary
        """
        self.log(f"\nRunning custom experiment: {config.name}")
        
        results = []
        for run_id in range(num_runs):
            result = self.run_single_experiment(config, run_id)
            results.append(result)
            self.results.append(result)
        
        summary = self._create_summary(config.name, results)
        self.summaries[config.name] = summary
        
        return summary
    
    def _create_summary(
        self,
        config_name: str,
        results: List[ExperimentResult]
    ) -> ExperimentSummary:
        """Create summary from multiple experiment results."""
        summary = ExperimentSummary(
            config_name=config_name,
            num_runs=len(results),
            results=results
        )
        
        successful_results = [r for r in results if r.success]
        
        if successful_results:
            # Calculate averages
            times = [r.total_time for r in successful_results]
            summary.avg_total_time = sum(times) / len(times)
            
            if len(times) > 1:
                mean = summary.avg_total_time
                variance = sum((t - mean) ** 2 for t in times) / len(times)
                summary.std_total_time = math.sqrt(variance)
            
            routabilities = [r.routability_rate for r in successful_results]
            summary.avg_routability = sum(routabilities) / len(routabilities)
            summary.best_routability = max(routabilities)
            summary.worst_routability = min(routabilities)
            
            wirelengths = [r.total_wirelength for r in successful_results]
            summary.avg_wirelength = sum(wirelengths) / len(wirelengths)
            summary.best_wirelength = min(wirelengths)
            summary.worst_wirelength = max(wirelengths)
            
            ratios = [r.wirelength_ratio for r in successful_results if r.wirelength_ratio > 0]
            if ratios:
                summary.avg_wirelength_ratio = sum(ratios) / len(ratios)
        
        summary.success_rate = len(successful_results) / len(results) if results else 0
        
        return summary
    
    def _print_summary_table(self, summaries: Dict[str, ExperimentSummary]):
        """Print formatted summary table."""
        self.log("\n" + "="*90)
        self.log("BENCHMARK SUMMARY")
        self.log("="*90)
        self.log(f"{'Benchmark':<12} {'IOs':>6} {'Time(s)':>10} {'Routability':>12} "
                f"{'Wirelength':>12} {'WL Ratio':>10} {'Success':>8}")
        self.log("-"*90)
        
        for name, summary in summaries.items():
            if summary.results:
                num_ios = summary.results[0].num_ios
                self.log(f"{name:<12} {num_ios:>6} {summary.avg_total_time:>10.3f} "
                        f"{summary.avg_routability:>11.1%} {summary.avg_wirelength:>12.1f} "
                        f"{summary.avg_wirelength_ratio:>10.2f} {summary.success_rate:>7.0%}")
        
        self.log("="*90)
    
    def _print_scalability_table(self, summaries: List[ExperimentSummary]):
        """Print scalability analysis table."""
        self.log("\n" + "="*80)
        self.log("SCALABILITY ANALYSIS RESULTS")
        self.log("="*80)
        self.log(f"{'IOs':>8} {'Time(s)':>12} {'Time/IO(ms)':>14} {'Routability':>12} {'Success':>10}")
        self.log("-"*80)
        
        for summary in summaries:
            if summary.results:
                num_ios = summary.results[0].num_ios
                time_per_io = (summary.avg_total_time / num_ios * 1000) if num_ios > 0 else 0
                self.log(f"{num_ios:>8} {summary.avg_total_time:>12.3f} {time_per_io:>14.3f} "
                        f"{summary.avg_routability:>11.1%} {summary.success_rate:>9.0%}")
        
        self.log("="*80)
    
    def save_results(self, filename: str = None):
        """
        Save all results to JSON file.
        
        Args:
            filename: Output filename (default: results_<timestamp>.json)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'num_experiments': len(self.results),
            'results': [r.to_dict() for r in self.results],
            'summaries': {k: v.to_dict() for k, v in self.summaries.items()}
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.log(f"\nResults saved to: {output_path}")
    
    def generate_report(self, filename: str = None) -> str:
        """
        Generate a text report of all experiments.
        
        Args:
            filename: Output filename (optional)
            
        Returns:
            Report text
        """
        lines = []
        lines.append("="*70)
        lines.append("FLIP-CHIP RDL ROUTING EXPERIMENT REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("="*70)
        lines.append("")
        
        # Summary statistics
        lines.append("OVERALL STATISTICS")
        lines.append("-"*40)
        lines.append(f"Total experiments: {len(self.results)}")
        successful = sum(1 for r in self.results if r.success)
        lines.append(f"Successful: {successful}")
        lines.append(f"Failed: {len(self.results) - successful}")
        lines.append("")
        
        # Per-benchmark summaries
        if self.summaries:
            lines.append("BENCHMARK SUMMARIES")
            lines.append("-"*40)
            for name, summary in self.summaries.items():
                lines.append(f"\n{name}:")
                lines.append(f"  Runs: {summary.num_runs}")
                lines.append(f"  Success rate: {summary.success_rate:.1%}")
                lines.append(f"  Avg time: {summary.avg_total_time:.3f}s (±{summary.std_total_time:.3f})")
                lines.append(f"  Avg routability: {summary.avg_routability:.1%}")
                lines.append(f"  Avg wirelength: {summary.avg_wirelength:.1f}")
                lines.append(f"  Wirelength ratio: {summary.avg_wirelength_ratio:.2f}")
        
        report = "\n".join(lines)
        
        if filename:
            output_path = self.output_dir / filename
            with open(output_path, 'w') as f:
                f.write(report)
            self.log(f"Report saved to: {output_path}")
        
        return report


def run_quick_test():
    """Run a quick test to verify the experiment runner works."""
    print("Running quick test...")
    
    runner = ExperimentRunner(output_dir="test_results", verbose=True)
    
    # Run tiny benchmark
    summary = runner.run_benchmark('tiny', num_runs=1)
    
    print(f"\nQuick test completed!")
    print(f"Success: {summary.success_rate:.0%}")
    print(f"Time: {summary.avg_total_time:.3f}s")
    print(f"Routability: {summary.avg_routability:.1%}")
    
    return summary.success_rate > 0


def main():
    """Main entry point for experiment runner."""
    parser = argparse.ArgumentParser(
        description="Flip-Chip RDL Router Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiments.py --benchmark small
  python run_experiments.py --all --runs 3
  python run_experiments.py --scalability --max-size 1000
  python run_experiments.py --custom 500 25 25
        """
    )
    
    parser.add_argument(
        '--benchmark', '-b',
        choices=['tiny', 'small', 'medium', 'large', 'xlarge', 'industrial'],
        help='Run specific benchmark'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Run all standard benchmarks'
    )
    parser.add_argument(
        '--scalability', '-s',
        action='store_true',
        help='Run scalability analysis'
    )
    parser.add_argument(
        '--max-size',
        type=int,
        default=2500,
        help='Maximum IOs for scalability analysis (default: 2500)'
    )
    parser.add_argument(
        '--custom', '-c',
        nargs=3,
        type=int,
        metavar=('IOS', 'ROWS', 'COLS'),
        help='Run custom configuration: num_ios bump_rows bump_cols'
    )
    parser.add_argument(
        '--runs', '-r',
        type=int,
        default=1,
        help='Number of runs per configuration (default: 1)'
    )
    parser.add_argument(
        '--output', '-o',
        default='results',
        help='Output directory (default: results)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run quick test'
    )
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.test:
        success = run_quick_test()
        return 0 if success else 1
    
    # Create runner
    runner = ExperimentRunner(
        output_dir=args.output,
        verbose=not args.quiet
    )
    
    # Run requested experiments
    if args.benchmark:
        runner.run_benchmark(args.benchmark, num_runs=args.runs)
    
    elif args.all:
        runner.run_all_benchmarks(num_runs=args.runs)
    
    elif args.scalability:
        runner.run_scalability_analysis(max_size=args.max_size, num_runs=args.runs)
    
    elif args.custom:
        num_ios, rows, cols = args.custom
        config = ExperimentConfig(
            name=f"custom_{num_ios}",
            num_ios=num_ios,
            bump_rows=rows,
            bump_cols=cols,
            die_width=rows * 200.0,
            die_height=cols * 200.0,
            description=f"Custom configuration: {num_ios} IOs"
        )
        runner.run_custom_experiment(config, num_runs=args.runs)
    
    else:
        # Default: run small benchmark
        print("No experiment specified. Running small benchmark...")
        runner.run_benchmark('small', num_runs=args.runs)
    
    # Save results
    runner.save_results()
    
    # Generate report
    report = runner.generate_report("report.txt")
    print("\n" + report)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
