# File: gpu_benchmark.py
# Comprehensive GPU Benchmark with Multiple Devices, Data Types, and Memory Bandwidth

import torch
import time
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional

# Try to import optional backends
try:
    import intel_extension_for_pytorch as ipex
    from torch import xpu
    XPU_AVAILABLE = True
except ImportError:
    XPU_AVAILABLE = False

# Check MPS availability (Apple Silicon)
MPS_AVAILABLE = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
CUDA_AVAILABLE = torch.cuda.is_available()

def synchronize(device: torch.device):
    """Synchronize the given device"""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "xpu" and XPU_AVAILABLE:
        xpu.synchronize()
    # CPU doesn't need synchronization

def get_dtype_from_string(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype"""
    dtype_map = {
        'float32': torch.float32,
        'float64': torch.float64,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'int8': torch.int8,
        'int16': torch.int16,
        'int32': torch.int32,
        'int64': torch.int64,
        'uint8': torch.uint8,
        'bool': torch.bool,
        'float': torch.float32,  # alias
        'double': torch.float64,  # alias
        'half': torch.float16,   # alias
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Supported: {list(dtype_map.keys())}")
    return dtype_map[dtype_str]

def get_device_info(device: torch.device) -> Dict:
    """Get information about the device"""
    info = {'type': device.type}
    
    if device.type == 'cuda':
        info['name'] = torch.cuda.get_device_name(device)
        info['total_memory'] = torch.cuda.get_device_properties(device).total_memory
        info['compute_capability'] = torch.cuda.get_device_capability(device)
        info['multi_processor_count'] = torch.cuda.get_device_properties(device).multi_processor_count
    elif device.type == 'cpu':
        info['name'] = 'CPU'
        info['cores'] = torch.get_num_threads()
    elif device.type == 'mps' and MPS_AVAILABLE:
        info['name'] = 'Apple Silicon GPU'
    elif device.type == 'xpu' and XPU_AVAILABLE:
        info['name'] = 'Intel GPU'
    
    return info

def memory_bandwidth_benchmark(device: torch.device, dtype: torch.dtype, 
                              num_trails: int = 2, max_size_gb: float = 16.0) -> List[Dict]:
    """Measure memory bandwidth using copy operations"""
    print(f"\n{'='*70}")
    print(f"MEMORY BANDWIDTH BENCHMARK - {dtype}")
    print(f"{'='*70}")
    
    results = []
    
    # Generate sizes from 1MB to max_size_gb GB (doubling each time)
    bytes_per_element = torch.tensor(1, dtype=dtype).element_size()
    min_elements = 2**20 // bytes_per_element  # 1 MB
    max_elements = int(max_size_gb * 1e9) // bytes_per_element
    
    sizes = []
    current = min_elements
    while current <= max_elements:
        sizes.append(current)
        current *= 2
    
    print(f"{'Size (GB)':<12} {'Time (ms)':<12} {'Bandwidth (GB/s)':<15}")
    print("-" * 70)
    
    for size_elements in sizes:
        try:
            # Create tensors
            a = torch.randn(size_elements, device=device)
            b = torch.randn(size_elements, device=device)
            a = a.to(dtype)
            b = b.to(dtype)
            
            # Warm-up
            for _ in range(3):
                a.copy_(b)
                synchronize(device)
            
            # Benchmark
            total_time = 0
            for _ in range(num_trails):
                synchronize(device)
                start_time = time.perf_counter()
                a.copy_(b)
                synchronize(device)
                end_time = time.perf_counter()
                total_time += (end_time - start_time) * 1000  # Convert to ms
            
            avg_time = total_time / num_trails
            
            # Calculate bandwidth
            bytes_copied = a.nelement() * a.element_size()
            bandwidth = (2 * bytes_copied) / (avg_time / 1000) / 1e9  # GB/s
            
            size_gb = bytes_copied / 1e9
            
            print(f"{size_gb:<12.3f} {avg_time:<12.3f} {bandwidth:<15.2f}")
            
            results.append({
                'size_gb': size_gb,
                'time_ms': avg_time,
                'bandwidth_gbs': bandwidth
            })
            
            # Clean up
            del a, b
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
        except torch.cuda.OutOfMemoryError:
            print(f"{size_gb:<12.3f} {'OOM':<12} {'-':<15}")
            break
        except Exception as e:
            print(f"{size_gb:<12.3f} {'Error':<12} {str(e):<15}")
            break
    
    return results

def flops_benchmark(device: torch.device, dtype: torch.dtype, 
                    num_trails: int = 2, memory_percent: float = 80.0) -> Dict:
    """Measure FLOPs/TOPS using matrix multiplication"""
    print(f"\n{'='*70}")
    print(f"FLOPS/TOPS BENCHMARK - {dtype}")
    print(f"{'='*70}")
    
    # Get available memory for the device
    if device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(device).total_memory
        free_memory = torch.cuda.mem_get_info(device)[0]
        available_memory = min(free_memory, total_memory * memory_percent / 100)
    else:
        # For non-CUDA devices, estimate available memory
        available_memory = 2 * 1024**3  # 2 GB default for non-CUDA
    
    # Calculate matrix size based on available memory
    bytes_per_element = torch.tensor(1, dtype=dtype).element_size()
    
    # We need 3 matrices: A, B, and result
    safe_memory = available_memory * 0.8  # Use 80% of available memory
    elements_per_matrix = safe_memory / (3 * bytes_per_element)
    size = int(elements_per_matrix ** 0.5)
    
    # Round down to multiple of 256 for better performance
    size = (size // 256) * 256
    size = max(size, 1024)  # Minimum size
    
    print(f"Matrix size: {size}x{size}")
    print(f"Memory per matrix: {size*size*bytes_per_element/1e9:.2f} GB")
    print(f"Total test memory: {size*size*bytes_per_element*3/1e9:.2f} GB")
    print(f"Iterations: {num_trails}")
    print("-" * 70)
    
    try:
        # Create matrices
        if dtype in [torch.int8, torch.uint8]:
            a = torch.randint(-128, 127, (size, size), device=device, dtype=dtype)
            b = torch.randint(-128, 127, (size, size), device=device, dtype=dtype)
        elif dtype == torch.bool:
            a = torch.randint(0, 2, (size, size), device=device, dtype=dtype)
            b = torch.randint(0, 2, (size, size), device=device, dtype=dtype)
        else:
            a = torch.randn(size, size, device=device, dtype=dtype)
            b = torch.randn(size, size, device=device, dtype=dtype)
        
        # Warm-up
        for _ in range(3):
            _ = torch.matmul(a, b)
        synchronize(device)
        
        # Benchmark
        start_time = time.perf_counter()
        for _ in range(num_trails):
            result = torch.matmul(a, b)
        synchronize(device)
        elapsed_time = time.perf_counter() - start_time
        
        # Calculate performance
        ops = 2.0 * size ** 3 * num_trails
        
        if dtype in [torch.int8, torch.uint8]:
            # INT8 operations: 4x per clock with tensor cores
            performance = (4.0 * ops) / elapsed_time / 1e12  # TOPS
            unit = "TOPS"
        elif dtype in [torch.float16, torch.bfloat16]:
            performance = ops / elapsed_time / 1e12  # TFLOPS
            unit = "TFLOPS"
        elif dtype in [torch.float32, torch.float64]:
            performance = ops / elapsed_time / 1e12  # TFLOPS
            unit = "TFLOPS"
        elif dtype in [torch.int16, torch.int32, torch.int64]:
            # Integer operations count as 1 op
            performance = ops / elapsed_time / 1e12  # TOPS
            unit = "TOPS"
        elif dtype == torch.bool:
            # Boolean operations are simpler
            performance = ops / elapsed_time / 1e12  # TOPS
            unit = "TOPS"
        else:
            performance = ops / elapsed_time / 1e12  # TFLOPS
            unit = "TFLOPS"
        
        # Get peak memory if on CUDA
        if device.type == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated(device) / 1e9
        else:
            peak_memory = size * size * bytes_per_element * 3 / 1e9
        
        print(f"\nResults:")
        print(f"  Performance: {performance:.2f} {unit}")
        print(f"  Time: {elapsed_time:.3f} seconds")
        print(f"  Peak Memory: {peak_memory:.2f} GB")
        
        # Clean up
        del a, b, result
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return {
            'dtype': str(dtype),
            'size': size,
            'performance': performance,
            'unit': unit,
            'time': elapsed_time,
            'peak_memory_gb': peak_memory,
            'success': True
        }
        
    except Exception as e:
        print(f"\nError: {e}")
        return {
            'dtype': str(dtype),
            'size': size,
            'performance': 0,
            'unit': 'N/A',
            'time': 0,
            'peak_memory_gb': 0,
            'success': False,
            'error': str(e)
        }

def benchmark_all_precisions(device: torch.device, num_trails: int = 2, 
                            memory_percent: float = 80.0) -> List[Dict]:
    """Benchmark all common precisions"""
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE PRECISION BENCHMARK")
    print(f"{'='*70}")
    
    results = []
    precisions = [
        ('float32', torch.float32),
        ('float16', torch.float16),
        ('bfloat16', torch.bfloat16),
        ('int8', torch.int8),
    ]
    
    # Check for BF16 support
    if device.type == 'cuda':
        major, _ = torch.cuda.get_device_capability(device)
        if major < 8:
            # Remove BF16 for pre-Ampere GPUs
            precisions = [p for p in precisions if p[0] != 'bfloat16']
    
    for name, dtype in precisions:
        print(f"\nBenchmarking {name}...")
        result = flops_benchmark(device, dtype, num_trails, memory_percent)
        result['name'] = name
        results.append(result)
    
    return results

def print_summary(results: List[Dict], bandwidth_results: Optional[Dict] = None):
    """Print a summary of all benchmark results"""
    print(f"\n{'='*70}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'='*70}")
    
    if results:
        print(f"\n{'Precision':<12} {'Size':<10} {'Performance':<15} {'Time':<10} {'Memory':<10}")
        print("-" * 70)
        
        for r in results:
            if r['success']:
                perf_str = f"{r['performance']:.2f} {r['unit']}"
                print(f"{r['name']:<12} {r['size']:<10} {perf_str:<15} {r['time']:<10.3f} {r['peak_memory_gb']:<10.2f}")
            else:
                print(f"{r['name']:<12} {'-':<10} {'Failed':<15} {'-':<10} {'-':<10}")
                if 'error' in r:
                    print(f"  Error: {r['error']}")
    
    if bandwidth_results:
        print(f"\n{'='*70}")
        print(f"MEMORY BANDWIDTH SUMMARY")
        print(f"{'='*70}")
        
        # Find maximum bandwidth achieved
        max_bw = max([r['bandwidth_gbs'] for r in bandwidth_results]) if bandwidth_results else 0
        print(f"Peak Bandwidth: {max_bw:.2f} GB/s")
        
        # Print bandwidth for different sizes
        print(f"\n{'Size (GB)':<12} {'Bandwidth (GB/s)':<15}")
        print("-" * 70)
        for r in bandwidth_results[:5]:  # Show first 5 sizes
            print(f"{r['size_gb']:<12.3f} {r['bandwidth_gbs']:<15.2f}")
        if len(bandwidth_results) > 5:
            print("... (showing first 5 sizes)")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive GPU/CPU Benchmark Tool')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu', 'mps', 'xpu'],
                       help='Device to benchmark (default: cuda)')
    parser.add_argument('--dtype', type=str, default=None,
                       help='Specific data type to benchmark (e.g., float32, int8). If not specified, benchmark all.')
    parser.add_argument('--num-trials', type=int, default=2,
                       help='Number of trials for each benchmark (default: 2)')
    parser.add_argument('--memory-percent', type=float, default=80.0,
                       help='Percentage of available memory to use (default: 80)')
    parser.add_argument('--test-bandwidth', action='store_true',
                       help='Run memory bandwidth tests')
    parser.add_argument('--test-flops', action='store_true',
                       help='Run FLOPs/TOPS tests')
    parser.add_argument('--max-size-gb', type=float, default=16.0,
                       help='Maximum size for bandwidth test in GB (default: 16)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and not CUDA_AVAILABLE:
        print("CUDA is not available. Falling back to CPU.")
        args.device = 'cpu'
    elif args.device == 'mps' and not MPS_AVAILABLE:
        print("MPS is not available. Falling back to CPU.")
        args.device = 'cpu'
    elif args.device == 'xpu' and not XPU_AVAILABLE:
        print("XPU is not available. Falling back to CPU.")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    device_info = get_device_info(device)
    
    print(f"{'='*70}")
    print(f"COMPREHENSIVE BENCHMARK TOOL")
    print(f"{'='*70}")
    print(f"Device: {device_info.get('name', args.device)}")
    print(f"Device Type: {device_info['type']}")
    
    if device.type == 'cuda':
        print(f"Total Memory: {device_info['total_memory'] / 1e9:.2f} GB")
        print(f"Compute Capability: {device_info['compute_capability'][0]}.{device_info['compute_capability'][1]}")
        print(f"CUDA Cores: {device_info['multi_processor_count']}")
    print(f"{'='*70}")
    
    results = []
    bandwidth_results = []
    
    # Run FLOPs tests
    if args.test_flops or (not args.test_bandwidth and not args.test_flops):
        if args.dtype:
            # Benchmark specific dtype
            dtype = get_dtype_from_string(args.dtype)
            result = flops_benchmark(device, dtype, args.num_trials, args.memory_percent)
            result['name'] = args.dtype
            results.append(result)
        else:
            # Benchmark all precisions
            results = benchmark_all_precisions(device, args.num_trials, args.memory_percent)
    
    # Run bandwidth tests
    if args.test_bandwidth or (not args.test_bandwidth and not args.test_flops):
        if args.dtype:
            dtype = get_dtype_from_string(args.dtype)
            bw_results = memory_bandwidth_benchmark(device, dtype, args.num_trials, args.max_size_gb)
            bandwidth_results.extend(bw_results)
        else:
            # Test bandwidth with float32 (most common)
            bw_results = memory_bandwidth_benchmark(device, torch.float32, args.num_trials, args.max_size_gb)
            bandwidth_results.extend(bw_results)
    
    # Print summary
    print_summary(results, bandwidth_results if bandwidth_results else None)
    
    # Final cleanup
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    print(f"\nBenchmark completed successfully!")

if __name__ == "__main__":
    main()
