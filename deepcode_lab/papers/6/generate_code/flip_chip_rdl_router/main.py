#!/usr/bin/env python3
"""
Flip-Chip RDL Router - Main Entry Point

A network-flow-based RDL routing algorithm for flip-chip design using:
1. Global routing via minimum-cost maximum-flow (MCMF) algorithm
2. Detailed routing using river routing methodology for single-layer RDL

Usage:
    python main.py [options] [input_file]
    
Examples:
    python main.py                          # Run with default test case
    python main.py config.yaml              # Route from config file
    python main.py --benchmark 10x10        # Run 10x10 benchmark
    python main.py --demo                   # Run demonstration
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Import from package
from src import (
    # Data structures
    Chip, BumpPad, IOPad, Net,
    # Core algorithms
    MCMF, NetworkBuilder, RiverRouter,
    Point, manhattan_distance,
    # Routing
    GlobalRouter, DetailedRouter, LayerAssigner,
    GlobalRoutingResult, DetailedRoutingResult, LayerAssignmentResult,
    RoutingConstraints, RoutingStatus, DetailedRoutingStatus,
    full_routing_pipeline, quick_route,
    # Utilities
    ChipParser, BenchmarkGenerator, load_chip,
    save_chip_to_json, save_chip_to_yaml,
    # Convenience
    create_simple_chip, route_chip, get_version, get_info
)


def print_banner():
    """Print application banner."""
    print("=" * 60)
    print("  Flip-Chip RDL Router")
    print("  Network-Flow-Based Routing Algorithm")
    print(f"  Version {get_version()}")
    print("=" * 60)
    print()


def print_chip_info(chip: Chip):
    """Print chip configuration information."""
    print("Chip Configuration:")
    print(f"  Name: {chip.name}")
    print(f"  Die Size: {chip.die_width} x {chip.die_height} um")
    print(f"  Die Area: {chip.die_area / 1e6:.2f} mm²")
    print(f"  IO Pads: {chip.num_io_pads}")
    print(f"  Bump Pads: {chip.num_bump_pads}")
    print(f"  Total Bump Capacity: {chip.total_bump_capacity}")
    print(f"  Bump Pitch: {chip.bump_pitch} um")
    print(f"  RDL Layers: {chip.rdl_layers}")
    print(f"  Wire Width/Spacing: {chip.wire_width}/{chip.wire_spacing} um")
    print()


def print_routing_results(results: Dict[str, Any]):
    """Print routing results summary."""
    print("Routing Results:")
    print("-" * 40)
    
    # Global routing results
    if 'global_routing' in results:
        gr = results['global_routing']
        print(f"  Global Routing:")
        print(f"    Status: {gr.status.name}")
        print(f"    Routed: {gr.num_routed}/{gr.num_routed + gr.num_unrouted}")
        print(f"    Total Cost: {gr.total_cost:.2f}")
        if gr.statistics:
            print(f"    Avg Assignment Distance: {gr.statistics.get('average_cost', 0):.2f}")
    
    # Detailed routing results
    if 'detailed_routing' in results:
        dr = results['detailed_routing']
        print(f"  Detailed Routing:")
        print(f"    Status: {dr.status.name}")
        print(f"    Routed: {dr.num_routed}/{dr.num_routed + dr.num_failed}")
        print(f"    Total Wirelength: {dr.total_wirelength:.2f} um")
        print(f"    DRC Violations: {len(dr.drc_violations)}")
        if dr.statistics:
            print(f"    Avg Wirelength: {dr.statistics.get('average_wirelength', 0):.2f}")
            print(f"    Avg Bends: {dr.statistics.get('average_bends', 0):.2f}")
    
    # Layer assignment results
    if 'layer_assignment' in results:
        la = results['layer_assignment']
        print(f"  Layer Assignment:")
        print(f"    Status: {la.status.name}")
        print(f"    Assigned: {la.num_assigned}")
        print(f"    Vias: {len(la.vias)}")
    
    # Overall statistics
    if 'statistics' in results:
        stats = results['statistics']
        print(f"  Overall:")
        print(f"    Success: {stats.get('success', False)}")
        print(f"    Routability: {stats.get('routability_rate', 0)*100:.1f}%")
        print(f"    Total Runtime: {stats.get('total_time', 0)*1000:.2f} ms")
    
    print()


def run_demo():
    """Run a demonstration of the routing system."""
    print("Running Demonstration...")
    print()
    
    # Create a simple test chip
    print("Creating test chip (5x5 bump grid, 16 IOs)...")
    chip = create_simple_chip(
        name="demo_chip",
        die_size=1000.0,
        bump_grid=(5, 5),
        io_per_side=4,
        bump_pitch=150.0
    )
    
    print_chip_info(chip)
    
    # Run routing
    print("Running routing pipeline...")
    start_time = time.time()
    
    results = route_chip(
        chip,
        wire_width=10.0,
        wire_spacing=10.0,
        enable_drc=True
    )
    
    elapsed = time.time() - start_time
    results['statistics']['total_time'] = elapsed
    
    print_routing_results(results)
    
    # Print some route details
    if results.get('detailed_routing') and results['detailed_routing'].routes:
        print("Sample Routes (first 3):")
        for i, route in enumerate(results['detailed_routing'].routes[:3]):
            print(f"  Route {i+1}: Net {route.net_id}")
            print(f"    IO: ({route.io_pad.x:.1f}, {route.io_pad.y:.1f})")
            print(f"    Bump: ({route.bump_pad.x:.1f}, {route.bump_pad.y:.1f})")
            print(f"    Wirelength: {route.wirelength:.2f} um")
            print(f"    Bends: {route.num_bends}")
        print()
    
    return results


def run_benchmark(benchmark_name: str, verbose: bool = True):
    """Run a specific benchmark configuration."""
    print(f"Running Benchmark: {benchmark_name}")
    print()
    
    # Parse benchmark name (e.g., "10x10", "20x20", "small", "medium", "large")
    if 'x' in benchmark_name.lower():
        parts = benchmark_name.lower().split('x')
        try:
            rows = int(parts[0])
            cols = int(parts[1]) if len(parts) > 1 else rows
        except ValueError:
            print(f"Error: Invalid benchmark format '{benchmark_name}'")
            return None
    elif benchmark_name.lower() == 'small':
        rows, cols = 5, 5
    elif benchmark_name.lower() == 'medium':
        rows, cols = 15, 15
    elif benchmark_name.lower() == 'large':
        rows, cols = 30, 30
    elif benchmark_name.lower() == 'xlarge':
        rows, cols = 50, 50
    else:
        print(f"Error: Unknown benchmark '{benchmark_name}'")
        print("Available: small, medium, large, xlarge, or NxM (e.g., 10x10)")
        return None
    
    # Generate benchmark chip
    chip = BenchmarkGenerator.generate_grid_benchmark(
        name=f"benchmark_{rows}x{cols}",
        num_io_pads=4 * max(rows, cols),
        bump_rows=rows,
        bump_cols=cols,
        bump_pitch=100.0,
        die_margin=200.0
    )
    
    if verbose:
        print_chip_info(chip)
    
    # Run routing
    print("Running routing...")
    start_time = time.time()
    
    results = full_routing_pipeline(
        chip,
        wire_width=10.0,
        wire_spacing=10.0,
        enable_drc=True,
        num_layers=1
    )
    
    elapsed = time.time() - start_time
    results['statistics']['total_time'] = elapsed
    
    if verbose:
        print_routing_results(results)
    
    # Summary line
    stats = results.get('statistics', {})
    print(f"Summary: {rows}x{cols} grid, "
          f"{chip.num_io_pads} IOs, "
          f"routability={stats.get('routability_rate', 0)*100:.1f}%, "
          f"wirelength={results.get('detailed_routing', {}).total_wirelength if results.get('detailed_routing') else 0:.0f}um, "
          f"time={elapsed*1000:.1f}ms")
    print()
    
    return results


def run_all_benchmarks():
    """Run all standard benchmarks and report results."""
    print("Running All Standard Benchmarks")
    print("=" * 60)
    print()
    
    benchmarks = ['small', 'medium', 'large']
    results_summary = []
    
    for bench in benchmarks:
        result = run_benchmark(bench, verbose=False)
        if result:
            stats = result.get('statistics', {})
            dr = result.get('detailed_routing')
            results_summary.append({
                'name': bench,
                'success': stats.get('success', False),
                'routability': stats.get('routability_rate', 0),
                'wirelength': dr.total_wirelength if dr else 0,
                'time': stats.get('total_time', 0)
            })
    
    # Print summary table
    print("\nBenchmark Summary:")
    print("-" * 70)
    print(f"{'Benchmark':<12} {'Success':<10} {'Routability':<12} {'Wirelength':<15} {'Time (ms)':<10}")
    print("-" * 70)
    
    for r in results_summary:
        print(f"{r['name']:<12} "
              f"{'Yes' if r['success'] else 'No':<10} "
              f"{r['routability']*100:.1f}%{'':<7} "
              f"{r['wirelength']:<15.0f} "
              f"{r['time']*1000:<10.1f}")
    
    print("-" * 70)
    print()


def route_from_file(file_path: str, output_path: Optional[str] = None, verbose: bool = True):
    """Route a chip from configuration file."""
    print(f"Loading configuration from: {file_path}")
    
    # Load chip
    chip = load_chip(file_path)
    if chip is None:
        print(f"Error: Failed to load chip from '{file_path}'")
        return None
    
    if verbose:
        print_chip_info(chip)
    
    # Run routing
    print("Running routing pipeline...")
    start_time = time.time()
    
    results = full_routing_pipeline(
        chip,
        wire_width=chip.wire_width,
        wire_spacing=chip.wire_spacing,
        enable_drc=True,
        num_layers=chip.rdl_layers
    )
    
    elapsed = time.time() - start_time
    results['statistics']['total_time'] = elapsed
    
    if verbose:
        print_routing_results(results)
    
    # Save output if requested
    if output_path:
        output_file = Path(output_path)
        if output_file.suffix in ['.yaml', '.yml']:
            save_chip_to_yaml(chip, output_path)
        else:
            save_chip_to_json(chip, output_path)
        print(f"Results saved to: {output_path}")
    
    return results


def interactive_mode():
    """Run in interactive mode for experimentation."""
    print("Interactive Mode")
    print("Type 'help' for available commands, 'quit' to exit")
    print()
    
    chip = None
    results = None
    
    while True:
        try:
            cmd = input("rdl> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        
        if not cmd:
            continue
        
        parts = cmd.split()
        command = parts[0]
        args = parts[1:]
        
        if command in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        elif command == 'help':
            print("Available commands:")
            print("  create <size> [grid]  - Create chip (e.g., 'create 1000 5x5')")
            print("  load <file>           - Load chip from file")
            print("  info                  - Show chip information")
            print("  route                 - Run routing on current chip")
            print("  benchmark <name>      - Run benchmark (small/medium/large/NxM)")
            print("  demo                  - Run demonstration")
            print("  save <file>           - Save chip to file")
            print("  quit                  - Exit")
        
        elif command == 'create':
            try:
                size = float(args[0]) if args else 1000.0
                grid = args[1] if len(args) > 1 else '5x5'
                grid_parts = grid.split('x')
                rows = int(grid_parts[0])
                cols = int(grid_parts[1]) if len(grid_parts) > 1 else rows
                
                chip = create_simple_chip(
                    name="interactive_chip",
                    die_size=size,
                    bump_grid=(rows, cols),
                    io_per_side=max(rows, cols),
                    bump_pitch=size / (max(rows, cols) + 2)
                )
                print(f"Created chip: {size}um die, {rows}x{cols} bumps")
            except (ValueError, IndexError) as e:
                print(f"Error: {e}")
        
        elif command == 'load':
            if args:
                chip = load_chip(args[0])
                if chip:
                    print(f"Loaded chip: {chip.name}")
                else:
                    print("Failed to load chip")
            else:
                print("Usage: load <filename>")
        
        elif command == 'info':
            if chip:
                print_chip_info(chip)
            else:
                print("No chip loaded. Use 'create' or 'load' first.")
        
        elif command == 'route':
            if chip:
                results = route_chip(chip)
                print_routing_results(results)
            else:
                print("No chip loaded. Use 'create' or 'load' first.")
        
        elif command == 'benchmark':
            name = args[0] if args else 'small'
            run_benchmark(name)
        
        elif command == 'demo':
            run_demo()
        
        elif command == 'save':
            if chip and args:
                if args[0].endswith('.yaml') or args[0].endswith('.yml'):
                    save_chip_to_yaml(chip, args[0])
                else:
                    save_chip_to_json(chip, args[0])
                print(f"Saved to {args[0]}")
            else:
                print("Usage: save <filename>")
        
        else:
            print(f"Unknown command: {command}. Type 'help' for available commands.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Flip-Chip RDL Router - Network-Flow-Based Routing Algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                      Run with default demo
  python main.py config.yaml          Route from config file
  python main.py --benchmark 10x10    Run 10x10 benchmark
  python main.py --benchmark all      Run all benchmarks
  python main.py --demo               Run demonstration
  python main.py --interactive        Interactive mode
        """
    )
    
    parser.add_argument(
        'input_file',
        nargs='?',
        help='Input configuration file (YAML/JSON)'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output file for results'
    )
    
    parser.add_argument(
        '-b', '--benchmark',
        help='Run benchmark (small/medium/large/xlarge/NxM/all)'
    )
    
    parser.add_argument(
        '-d', '--demo',
        action='store_true',
        help='Run demonstration'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    parser.add_argument(
        '-v', '--version',
        action='store_true',
        help='Show version information'
    )
    
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show package information'
    )
    
    args = parser.parse_args()
    
    # Handle version/info requests
    if args.version:
        print(f"Flip-Chip RDL Router v{get_version()}")
        return 0
    
    if args.info:
        info = get_info()
        print("Package Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        return 0
    
    # Print banner unless quiet
    if not args.quiet:
        print_banner()
    
    # Handle different modes
    if args.interactive:
        interactive_mode()
        return 0
    
    if args.benchmark:
        if args.benchmark.lower() == 'all':
            run_all_benchmarks()
        else:
            run_benchmark(args.benchmark, verbose=not args.quiet)
        return 0
    
    if args.demo:
        run_demo()
        return 0
    
    if args.input_file:
        result = route_from_file(
            args.input_file,
            output_path=args.output,
            verbose=not args.quiet
        )
        return 0 if result and result.get('statistics', {}).get('success') else 1
    
    # Default: run demo
    run_demo()
    return 0


if __name__ == '__main__':
    sys.exit(main())
