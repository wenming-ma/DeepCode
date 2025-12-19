"""
Input file parser for flip-chip RDL routing.

This module provides parsers for various input file formats including:
- Custom YAML/JSON configuration files
- DEF (Design Exchange Format) simplified subset
- Simple text-based benchmark formats

Author: Flip-Chip RDL Router
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import math

# Try to import yaml, fall back to json-only if not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from ..data_structures import Chip, BumpPad, IOPad, Net


@dataclass
class ParseResult:
    """Result of parsing operation."""
    success: bool
    chip: Optional[Chip] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        self.success = False
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)


class ChipParser:
    """
    Parser for chip configuration files.
    
    Supports multiple input formats:
    - YAML configuration files
    - JSON configuration files
    - Simple text format for benchmarks
    """
    
    def __init__(self):
        """Initialize the parser."""
        self.supported_formats = ['.yaml', '.yml', '.json', '.txt', '.def']
    
    def parse(self, file_path: Union[str, Path]) -> ParseResult:
        """
        Parse a chip configuration file.
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            ParseResult containing the parsed chip or errors
        """
        path = Path(file_path)
        result = ParseResult(success=True)
        
        if not path.exists():
            result.add_error(f"File not found: {file_path}")
            return result
        
        suffix = path.suffix.lower()
        
        if suffix in ['.yaml', '.yml']:
            return self._parse_yaml(path)
        elif suffix == '.json':
            return self._parse_json(path)
        elif suffix == '.txt':
            return self._parse_text(path)
        elif suffix == '.def':
            return self._parse_def(path)
        else:
            result.add_error(f"Unsupported file format: {suffix}")
            return result
    
    def _parse_yaml(self, path: Path) -> ParseResult:
        """Parse YAML configuration file."""
        result = ParseResult(success=True)
        
        if not YAML_AVAILABLE:
            result.add_error("YAML parsing requires PyYAML package. Install with: pip install pyyaml")
            return result
        
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            return self._parse_dict(data, result)
        except yaml.YAMLError as e:
            result.add_error(f"YAML parsing error: {e}")
            return result
        except Exception as e:
            result.add_error(f"Error reading file: {e}")
            return result
    
    def _parse_json(self, path: Path) -> ParseResult:
        """Parse JSON configuration file."""
        result = ParseResult(success=True)
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return self._parse_dict(data, result)
        except json.JSONDecodeError as e:
            result.add_error(f"JSON parsing error: {e}")
            return result
        except Exception as e:
            result.add_error(f"Error reading file: {e}")
            return result
    
    def _parse_dict(self, data: Dict[str, Any], result: ParseResult) -> ParseResult:
        """
        Parse chip configuration from dictionary.
        
        Expected format:
        {
            "chip": {
                "name": "chip_name",
                "die_width": 1000.0,
                "die_height": 1000.0,
                "rdl_layers": 1,
                "bump_pitch": 100.0,
                "bump_diameter": 80.0,
                "wire_width": 10.0,
                "wire_spacing": 10.0
            },
            "bump_pads": [
                {"x": 100, "y": 100, "name": "bump_0", "capacity": 1},
                ...
            ],
            "io_pads": [
                {"x": 50, "y": 0, "name": "io_0", "net_id": 0, "side": "bottom"},
                ...
            ],
            "nets": [
                {"net_id": 0, "name": "net_0", "io_pads": [0], "bump_pads": [0]},
                ...
            ]
        }
        """
        try:
            # Parse chip configuration
            chip_config = data.get('chip', {})
            
            chip = Chip(
                name=chip_config.get('name', 'unnamed_chip'),
                die_width=float(chip_config.get('die_width', 1000.0)),
                die_height=float(chip_config.get('die_height', 1000.0)),
                rdl_layers=int(chip_config.get('rdl_layers', 1)),
                bump_pitch=float(chip_config.get('bump_pitch', 100.0)),
                bump_diameter=float(chip_config.get('bump_diameter', 80.0)),
                wire_width=float(chip_config.get('wire_width', 10.0)),
                wire_spacing=float(chip_config.get('wire_spacing', 10.0))
            )
            
            # Parse bump pads
            bump_pads_data = data.get('bump_pads', [])
            bump_pad_map = {}  # Map name/index to BumpPad
            
            for i, bp_data in enumerate(bump_pads_data):
                bump_pad = BumpPad(
                    x=float(bp_data.get('x', 0)),
                    y=float(bp_data.get('y', 0)),
                    capacity=int(bp_data.get('capacity', 1)),
                    name=bp_data.get('name', f'bump_{i}'),
                    diameter=float(bp_data.get('diameter', chip.bump_diameter)),
                    pitch=float(bp_data.get('pitch', chip.bump_pitch)),
                    net_id=int(bp_data.get('net_id', -1)),
                    is_power=bp_data.get('is_power', False),
                    is_blocked=bp_data.get('is_blocked', False)
                )
                chip.add_bump_pad(bump_pad)
                bump_pad_map[bump_pad.name] = bump_pad
                bump_pad_map[i] = bump_pad
            
            # Parse IO pads
            io_pads_data = data.get('io_pads', [])
            io_pad_map = {}  # Map name/index to IOPad
            
            for i, io_data in enumerate(io_pads_data):
                io_pad = IOPad(
                    x=float(io_data.get('x', 0)),
                    y=float(io_data.get('y', 0)),
                    net_id=int(io_data.get('net_id', -1)),
                    name=io_data.get('name', f'io_{i}'),
                    width=float(io_data.get('width', chip.io_pad_width)),
                    height=float(io_data.get('height', chip.io_pad_height)),
                    side=io_data.get('side', 'bottom'),
                    layer=int(io_data.get('layer', 1)),
                    is_power=io_data.get('is_power', False)
                )
                chip.add_io_pad(io_pad)
                io_pad_map[io_pad.name] = io_pad
                io_pad_map[i] = io_pad
            
            # Parse nets
            nets_data = data.get('nets', [])
            
            for net_data in nets_data:
                net = Net(
                    net_id=int(net_data.get('net_id', -1)),
                    name=net_data.get('name', ''),
                    wire_width=float(net_data.get('wire_width', chip.wire_width)),
                    wire_spacing=float(net_data.get('wire_spacing', chip.wire_spacing)),
                    layer=int(net_data.get('layer', 1)),
                    is_power=net_data.get('is_power', False),
                    priority=int(net_data.get('priority', 0))
                )
                
                # Add IO pads to net
                for io_ref in net_data.get('io_pads', []):
                    if io_ref in io_pad_map:
                        net.add_io_pad(io_pad_map[io_ref])
                
                # Add bump pads to net
                for bump_ref in net_data.get('bump_pads', []):
                    if bump_ref in bump_pad_map:
                        net.add_bump_pad(bump_pad_map[bump_ref])
                
                chip.add_net(net)
            
            result.chip = chip
            result.statistics = {
                'num_io_pads': chip.num_io_pads,
                'num_bump_pads': chip.num_bump_pads,
                'num_nets': chip.num_nets,
                'die_area': chip.die_area
            }
            
        except Exception as e:
            result.add_error(f"Error parsing configuration: {e}")
        
        return result
    
    def _parse_text(self, path: Path) -> ParseResult:
        """
        Parse simple text format for benchmarks.
        
        Format:
        # Comment lines start with #
        CHIP name die_width die_height
        BUMP x y [capacity] [name]
        IO x y [net_id] [name] [side]
        NET net_id io_indices bump_indices
        """
        result = ParseResult(success=True)
        
        try:
            chip = None
            bump_pads = []
            io_pads = []
            nets_data = []
            
            with open(path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    keyword = parts[0].upper()
                    
                    if keyword == 'CHIP':
                        if len(parts) >= 4:
                            chip = Chip(
                                name=parts[1],
                                die_width=float(parts[2]),
                                die_height=float(parts[3])
                            )
                        else:
                            result.add_warning(f"Line {line_num}: Invalid CHIP format")
                    
                    elif keyword == 'BUMP':
                        if len(parts) >= 3:
                            x = float(parts[1])
                            y = float(parts[2])
                            capacity = int(parts[3]) if len(parts) > 3 else 1
                            name = parts[4] if len(parts) > 4 else f'bump_{len(bump_pads)}'
                            bump_pads.append(BumpPad(x=x, y=y, capacity=capacity, name=name))
                        else:
                            result.add_warning(f"Line {line_num}: Invalid BUMP format")
                    
                    elif keyword == 'IO':
                        if len(parts) >= 3:
                            x = float(parts[1])
                            y = float(parts[2])
                            net_id = int(parts[3]) if len(parts) > 3 else -1
                            name = parts[4] if len(parts) > 4 else f'io_{len(io_pads)}'
                            side = parts[5] if len(parts) > 5 else 'bottom'
                            io_pads.append(IOPad(x=x, y=y, net_id=net_id, name=name, side=side))
                        else:
                            result.add_warning(f"Line {line_num}: Invalid IO format")
                    
                    elif keyword == 'NET':
                        if len(parts) >= 4:
                            net_id = int(parts[1])
                            io_indices = [int(x) for x in parts[2].split(',') if x]
                            bump_indices = [int(x) for x in parts[3].split(',') if x]
                            nets_data.append((net_id, io_indices, bump_indices))
                        else:
                            result.add_warning(f"Line {line_num}: Invalid NET format")
                    
                    elif keyword == 'GRID':
                        # GRID rows cols pitch origin_x origin_y
                        if len(parts) >= 4 and chip is not None:
                            rows = int(parts[1])
                            cols = int(parts[2])
                            pitch = float(parts[3])
                            origin_x = float(parts[4]) if len(parts) > 4 else pitch
                            origin_y = float(parts[5]) if len(parts) > 5 else pitch
                            
                            grid_bumps = BumpPad.create_grid(
                                rows=rows, cols=cols, pitch=pitch,
                                origin_x=origin_x, origin_y=origin_y
                            )
                            bump_pads.extend(grid_bumps)
                    
                    elif keyword == 'PERIPHERAL':
                        # PERIPHERAL num_per_side margin
                        if len(parts) >= 2 and chip is not None:
                            num_per_side = int(parts[1])
                            margin = float(parts[2]) if len(parts) > 2 else 50.0
                            
                            peripheral_ios = IOPad.create_peripheral_pads(
                                die_width=chip.die_width,
                                die_height=chip.die_height,
                                num_pads_per_side=num_per_side,
                                margin=margin
                            )
                            io_pads.extend(peripheral_ios)
            
            # Create chip if not defined
            if chip is None:
                chip = Chip(name='benchmark', die_width=1000.0, die_height=1000.0)
            
            # Add bump pads
            for bp in bump_pads:
                chip.add_bump_pad(bp)
            
            # Add IO pads
            for io in io_pads:
                chip.add_io_pad(io)
            
            # Create nets
            for net_id, io_indices, bump_indices in nets_data:
                net = Net(net_id=net_id, name=f'net_{net_id}')
                for idx in io_indices:
                    if 0 <= idx < len(chip.io_pads):
                        net.add_io_pad(chip.io_pads[idx])
                for idx in bump_indices:
                    if 0 <= idx < len(chip.bump_pads):
                        net.add_bump_pad(chip.bump_pads[idx])
                chip.add_net(net)
            
            result.chip = chip
            result.statistics = {
                'num_io_pads': chip.num_io_pads,
                'num_bump_pads': chip.num_bump_pads,
                'num_nets': chip.num_nets
            }
            
        except Exception as e:
            result.add_error(f"Error parsing text file: {e}")
        
        return result
    
    def _parse_def(self, path: Path) -> ParseResult:
        """
        Parse simplified DEF (Design Exchange Format) file.
        
        This is a simplified parser that handles basic DEF constructs.
        """
        result = ParseResult(success=True)
        
        try:
            chip = Chip(name='def_design')
            
            with open(path, 'r') as f:
                content = f.read()
            
            lines = content.split('\n')
            i = 0
            
            while i < len(lines):
                line = lines[i].strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    i += 1
                    continue
                
                # Parse DESIGN name
                if line.startswith('DESIGN'):
                    parts = line.split()
                    if len(parts) >= 2:
                        chip.name = parts[1].rstrip(';')
                
                # Parse DIEAREA
                elif line.startswith('DIEAREA'):
                    # DIEAREA ( x1 y1 ) ( x2 y2 ) ;
                    coords = []
                    for part in line.split():
                        try:
                            coords.append(float(part))
                        except ValueError:
                            pass
                    if len(coords) >= 4:
                        chip.die_width = coords[2] - coords[0]
                        chip.die_height = coords[3] - coords[1]
                
                # Parse COMPONENTS (for bump pads)
                elif line.startswith('COMPONENTS'):
                    i += 1
                    while i < len(lines) and not lines[i].strip().startswith('END COMPONENTS'):
                        comp_line = lines[i].strip()
                        if comp_line.startswith('-'):
                            # Parse component: - name type + PLACED ( x y ) orientation ;
                            parts = comp_line.split()
                            if len(parts) >= 2:
                                name = parts[1]
                                x, y = 0.0, 0.0
                                # Find PLACED coordinates
                                for j, p in enumerate(parts):
                                    if p == 'PLACED' or p == '+':
                                        if j + 3 < len(parts):
                                            try:
                                                x = float(parts[j+2].strip('()'))
                                                y = float(parts[j+3].strip('()'))
                                            except (ValueError, IndexError):
                                                pass
                                
                                if 'BUMP' in name.upper() or 'BALL' in name.upper():
                                    chip.add_bump_pad(BumpPad(x=x, y=y, name=name))
                        i += 1
                
                # Parse PINS (for IO pads)
                elif line.startswith('PINS'):
                    i += 1
                    while i < len(lines) and not lines[i].strip().startswith('END PINS'):
                        pin_line = lines[i].strip()
                        if pin_line.startswith('-'):
                            parts = pin_line.split()
                            if len(parts) >= 2:
                                name = parts[1]
                                x, y = 0.0, 0.0
                                # Find PLACED coordinates
                                for j, p in enumerate(parts):
                                    if p == 'PLACED' or p == 'FIXED':
                                        if j + 3 < len(parts):
                                            try:
                                                x = float(parts[j+2].strip('()'))
                                                y = float(parts[j+3].strip('()'))
                                            except (ValueError, IndexError):
                                                pass
                                
                                chip.add_io_pad(IOPad(x=x, y=y, name=name))
                        i += 1
                
                # Parse NETS
                elif line.startswith('NETS'):
                    i += 1
                    current_net = None
                    while i < len(lines) and not lines[i].strip().startswith('END NETS'):
                        net_line = lines[i].strip()
                        if net_line.startswith('-'):
                            # New net definition
                            parts = net_line.split()
                            if len(parts) >= 2:
                                net_name = parts[1]
                                current_net = Net(name=net_name)
                                chip.add_net(current_net)
                        elif current_net and net_line.startswith('('):
                            # Pin connection: ( instance pin )
                            pass  # Would need to link to components
                        i += 1
                
                i += 1
            
            result.chip = chip
            result.statistics = {
                'num_io_pads': chip.num_io_pads,
                'num_bump_pads': chip.num_bump_pads,
                'num_nets': chip.num_nets
            }
            
        except Exception as e:
            result.add_error(f"Error parsing DEF file: {e}")
        
        return result


class BenchmarkGenerator:
    """
    Generator for benchmark test cases.
    
    Creates synthetic chip configurations for testing and evaluation.
    """
    
    @staticmethod
    def generate_grid_benchmark(
        name: str,
        num_io_pads: int,
        bump_rows: int,
        bump_cols: int,
        bump_pitch: float = 100.0,
        die_margin: float = 100.0
    ) -> Chip:
        """
        Generate a benchmark with grid bump array and peripheral IO pads.
        
        Args:
            name: Benchmark name
            num_io_pads: Total number of IO pads (distributed on all sides)
            bump_rows: Number of bump rows
            bump_cols: Number of bump columns
            bump_pitch: Spacing between bumps
            die_margin: Margin from bumps to die edge
            
        Returns:
            Configured Chip object
        """
        # Calculate die dimensions
        die_width = (bump_cols + 1) * bump_pitch + 2 * die_margin
        die_height = (bump_rows + 1) * bump_pitch + 2 * die_margin
        
        chip = Chip(
            name=name,
            die_width=die_width,
            die_height=die_height,
            bump_pitch=bump_pitch
        )
        
        # Create bump grid
        chip.create_bump_grid(
            rows=bump_rows,
            cols=bump_cols,
            origin_x=die_margin + bump_pitch,
            origin_y=die_margin + bump_pitch
        )
        
        # Create peripheral IO pads
        pads_per_side = num_io_pads // 4
        chip.create_peripheral_io_pads(
            num_pads_per_side=pads_per_side,
            margin=die_margin / 2
        )
        
        return chip
    
    @staticmethod
    def generate_random_benchmark(
        name: str,
        num_io_pads: int,
        num_bump_pads: int,
        die_width: float = 1000.0,
        die_height: float = 1000.0,
        seed: Optional[int] = None
    ) -> Chip:
        """
        Generate a benchmark with random pad positions.
        
        Args:
            name: Benchmark name
            num_io_pads: Number of IO pads
            num_bump_pads: Number of bump pads
            die_width: Die width
            die_height: Die height
            seed: Random seed for reproducibility
            
        Returns:
            Configured Chip object
        """
        import random
        if seed is not None:
            random.seed(seed)
        
        chip = Chip(
            name=name,
            die_width=die_width,
            die_height=die_height
        )
        
        # Generate random bump pads in interior
        margin = 100.0
        for i in range(num_bump_pads):
            x = random.uniform(margin, die_width - margin)
            y = random.uniform(margin, die_height - margin)
            chip.add_bump_pad(BumpPad(x=x, y=y, name=f'bump_{i}'))
        
        # Generate IO pads on periphery
        for i in range(num_io_pads):
            side = i % 4
            if side == 0:  # Bottom
                x = random.uniform(margin, die_width - margin)
                y = 0
                side_name = 'bottom'
            elif side == 1:  # Right
                x = die_width
                y = random.uniform(margin, die_height - margin)
                side_name = 'right'
            elif side == 2:  # Top
                x = random.uniform(margin, die_width - margin)
                y = die_height
                side_name = 'top'
            else:  # Left
                x = 0
                y = random.uniform(margin, die_height - margin)
                side_name = 'left'
            
            chip.add_io_pad(IOPad(x=x, y=y, name=f'io_{i}', side=side_name))
        
        return chip
    
    @staticmethod
    def generate_standard_benchmarks() -> List[Chip]:
        """
        Generate a set of standard benchmarks for evaluation.
        
        Returns:
            List of benchmark Chip objects
        """
        benchmarks = []
        
        # Small benchmark: 100 IOs, 10x10 bumps
        benchmarks.append(BenchmarkGenerator.generate_grid_benchmark(
            name='small_100',
            num_io_pads=100,
            bump_rows=10,
            bump_cols=10
        ))
        
        # Medium benchmark: 500 IOs, 20x20 bumps
        benchmarks.append(BenchmarkGenerator.generate_grid_benchmark(
            name='medium_500',
            num_io_pads=500,
            bump_rows=20,
            bump_cols=20
        ))
        
        # Large benchmark: 1000 IOs, 30x30 bumps
        benchmarks.append(BenchmarkGenerator.generate_grid_benchmark(
            name='large_1000',
            num_io_pads=1000,
            bump_rows=30,
            bump_cols=30
        ))
        
        # Extra large benchmark: 2000 IOs, 50x50 bumps
        benchmarks.append(BenchmarkGenerator.generate_grid_benchmark(
            name='xlarge_2000',
            num_io_pads=2000,
            bump_rows=50,
            bump_cols=50
        ))
        
        return benchmarks


def save_chip_to_yaml(chip: Chip, file_path: Union[str, Path]) -> bool:
    """
    Save chip configuration to YAML file.
    
    Args:
        chip: Chip object to save
        file_path: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    if not YAML_AVAILABLE:
        print("YAML saving requires PyYAML package")
        return False
    
    try:
        data = chip.to_dict()
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        return True
    except Exception as e:
        print(f"Error saving to YAML: {e}")
        return False


def save_chip_to_json(chip: Chip, file_path: Union[str, Path], indent: int = 2) -> bool:
    """
    Save chip configuration to JSON file.
    
    Args:
        chip: Chip object to save
        file_path: Output file path
        indent: JSON indentation level
        
    Returns:
        True if successful, False otherwise
    """
    try:
        data = chip.to_dict()
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)
        return True
    except Exception as e:
        print(f"Error saving to JSON: {e}")
        return False


def load_chip(file_path: Union[str, Path]) -> Optional[Chip]:
    """
    Load chip from configuration file.
    
    Convenience function that handles parsing and error checking.
    
    Args:
        file_path: Path to configuration file
        
    Returns:
        Chip object if successful, None otherwise
    """
    parser = ChipParser()
    result = parser.parse(file_path)
    
    if result.success:
        return result.chip
    else:
        for error in result.errors:
            print(f"Error: {error}")
        return None


# Test function
def test_parser():
    """Test parser functionality."""
    print("Testing parser module...")
    
    # Test benchmark generation
    print("\n1. Testing benchmark generation...")
    chip = BenchmarkGenerator.generate_grid_benchmark(
        name='test_benchmark',
        num_io_pads=100,
        bump_rows=10,
        bump_cols=10
    )
    print(f"   Generated chip: {chip.name}")
    print(f"   IO pads: {chip.num_io_pads}")
    print(f"   Bump pads: {chip.num_bump_pads}")
    print(f"   Die size: {chip.die_width} x {chip.die_height}")
    
    # Test JSON serialization
    print("\n2. Testing JSON serialization...")
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        if save_chip_to_json(chip, temp_path):
            print(f"   Saved to: {temp_path}")
            
            # Test loading
            loaded_chip = load_chip(temp_path)
            if loaded_chip:
                print(f"   Loaded chip: {loaded_chip.name}")
                print(f"   IO pads: {loaded_chip.num_io_pads}")
                print(f"   Bump pads: {loaded_chip.num_bump_pads}")
                assert loaded_chip.num_io_pads == chip.num_io_pads
                assert loaded_chip.num_bump_pads == chip.num_bump_pads
                print("   JSON round-trip: PASSED")
            else:
                print("   Failed to load chip")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    # Test standard benchmarks
    print("\n3. Testing standard benchmarks...")
    benchmarks = BenchmarkGenerator.generate_standard_benchmarks()
    for bm in benchmarks:
        print(f"   {bm.name}: {bm.num_io_pads} IOs, {bm.num_bump_pads} bumps")
    
    print("\nParser tests completed!")


if __name__ == '__main__':
    test_parser()
