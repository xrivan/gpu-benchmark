import torch
import time
import argparse
import sys
from typing import Dict, List, Optional, Tuple

# ==========================================
# 1. Backend Abstraction Layer
# ==========================================
class DeviceManager:
    def __init__(self, device_str: str):
        self.device_str = device_str
        
        # logical fallback
        if device_str == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            self.device = torch.device('cpu')
        elif device_str == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            print("Warning: MPS not available, falling back to CPU")
            self.device = torch.device('cpu')
        elif device_str == 'xpu' and not hasattr(torch, 'xpu'):
             print("Warning: XPU not available, falling back to CPU")
             self.device = torch.device('cpu')
        # Add NPU device fallback check
        elif device_str == 'npu':
            try:
                # Check if torch_npu module is available
                import torch_npu
                if not torch_npu.npu.is_available():
                    print("Warning: NPU not available, falling back to CPU")
                    self.device = torch.device('cpu')
                else:
                    self.device = torch.device(device_str)
            except ImportError:
                print("Warning: torch_npu module not found, falling back to CPU")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device_str)
            
        self.type = self.device.type

    def synchronize(self):
        if self.type == 'cpu': 
            return
        if self.type == 'cuda':
            torch.cuda.synchronize()
        elif self.type == 'mps':
            torch.mps.synchronize()
        elif self.type == 'xpu':
            torch.xpu.synchronize()
        # Add NPU synchronization
        elif self.type == 'npu':
            import torch_npu
            torch_npu.npu.synchronize()
        # Fallback for future/custom backends
        elif hasattr(torch, self.type):
            mod = getattr(torch, self.type)
            if hasattr(mod, 'synchronize'):
                mod.synchronize()

    def empty_cache(self):
        if self.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.type == 'mps':
            torch.mps.empty_cache()
        elif self.type == 'xpu':
            torch.xpu.empty_cache()
        # Add NPU cache emptying
        elif self.type == 'npu':
            import torch_npu
            torch_npu.npu.empty_cache()

    def get_info(self):
        info = {'type': self.type, 'name': self.type.upper()}
        if self.type == 'cuda':
            info['name'] = torch.cuda.get_device_name(self.device)
            props = torch.cuda.get_device_properties(self.device)
            info['total_memory_gb'] = props.total_memory / 1e9
            info['capability'] = (props.major, props.minor)
        # Add NPU device info
        elif self.type == 'npu':
            try:
                import torch_npu
                info['name'] = torch_npu.npu.get_device_name(self.device)
                #info['name'] = f"HUAWEI NPU Device {torch_npu.npu.current_device()}"
                # Get NPU memory info if available
                if hasattr(torch_npu.npu, 'get_device_properties'):
                    props = torch_npu.npu.get_device_properties(self.device)
                    if hasattr(props, 'total_memory'):
                        info['total_memory_gb'] = props.total_memory / 1e9
                else:
                    # Default value if we can't get actual memory
                    info['total_memory_gb'] = 0.0
                # NPU doesn't have compute capability like CUDA
                info['capability'] = (0, 0)
            except ImportError:
                info['name'] = "NPU (torch_npu not available)"
                info['total_memory_gb'] = 8.0
                info['capability'] = (0, 0)
        return info

# ==========================================
# 2. Hardware Capability Checks
# ==========================================
def is_dtype_supported(device_mgr: DeviceManager, dtype: torch.dtype) -> bool:
    """
    Checks if the specific hardware actually supports the math for this dtype
    to avoid 'not implemented' errors.
    """
    # CPU usually supports standard types, simpler to skip checks
    if device_mgr.type == 'cpu':
        return True

    # CUDA Specific Checks
    if device_mgr.type == 'cuda':
        cap = torch.cuda.get_device_capability(device_mgr.device)
        major, minor = cap
        
        # FP8 requires Compute Capability 8.9 (Ada) or 9.0+ (Hopper)
        # Note: PyTorch support for Ada FP8 is limited, mostly Hopper (9.0)
        is_fp8 = 'float8' in str(dtype)
        if is_fp8:
            if major < 9: 
                return False
            return True
            
        # BF16 requires Ampere (8.0+)
        if dtype == torch.bfloat16 and major < 8:
            return False

    # Add NPU dtype support checks
    # NPU typically supports common dtypes like float32, float16, int8
    # May not support bfloat16 or specialized FP8 formats
    if device_mgr.type == 'npu':
        # NPU might not support bfloat16 or FP8
        if dtype == torch.bfloat16:
            return False
        if 'float8' in str(dtype):
            return False
        # NPU should support float32, float16, int8
        return dtype in [torch.float32, torch.float16, torch.int8]

    return True

# ==========================================
# 3. Benchmark Functions
# ==========================================
def benchmark_bandwidth(manager: DeviceManager, max_size_gb: float = 2.0, num_trails: int = 5):
    """
    Measures Device to Device copy bandwidth.
    """
    print(f"\n{'='*60}")
    print(f"MEMORY BANDWIDTH ({manager.device})")
    print(f"{'='*60}")
    
    # Use float32 for bandwidth test
    dtype = torch.float32
    element_size = torch.tensor(1, dtype=dtype).element_size()
    
    # Generate sizes: 16MB -> Max GB
    start_elements = (16 * 1024 * 1024) // element_size
    max_elements = int(max_size_gb * 1e9) // element_size
    
    sizes = []
    curr = start_elements
    while curr <= max_elements:
        sizes.append(curr)
        curr *= 2

    print(f"{'Size (GB)':<12} {'Time (ms)':<12} {'Bandwidth (GB/s)':<15}")
    print("-" * 60)

    max_bw = 0
    
    for elements in sizes:
        try:
            size_gb = (elements * element_size) / 1e9
            
            # Allocation
            # Handle NPU tensor creation specifically
            if manager.type == 'npu':
                a = torch.randn(elements, dtype=dtype).npu()
                b = torch.randn(elements, dtype=dtype).npu()
            else:
                a = torch.randn(elements, device=manager.device, dtype=dtype)
                b = torch.randn(elements, device=manager.device, dtype=dtype)
            
            # Warmup
            for _ in range(2):
                a.copy_(b)
                manager.synchronize()
            
            # Measure
            start = time.perf_counter()
            for _ in range(num_trails):
                a.copy_(b)
            manager.synchronize()
            end = time.perf_counter()
            
            # Stats
            avg_time_ms = ((end - start) / num_trails) * 1000
            # 2x because Read + Write
            bytes_transferred = elements * element_size * 2 
            bw_gbs = (bytes_transferred / 1e9) / (avg_time_ms / 1000)
            
            if bw_gbs > max_bw: max_bw = bw_gbs

            print(f"{size_gb:<12.3f} {avg_time_ms:<12.3f} {bw_gbs:<15.2f}")
            
            del a, b
            manager.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{size_gb:<12.3f} {'OOM':<12} {'-':<15}")
                break
            else:
                print(f"Error: {e}")
                break

    print(f"\nPeak Bandwidth: {max_bw:.2f} GB/s")
    return max_bw

def benchmark_flops(manager: DeviceManager, dtype: torch.dtype, size: int, num_trails: int = 5):
    """
    Measures Compute Performance (TFLOPS/TOPS)
    """
    device = manager.device
    dtype_str = str(dtype).replace("torch.", "")
    
    # 1. Check Capability first
    if not is_dtype_supported(manager, dtype):
        print(f"Skipping {dtype_str}: Hardware capability too low or unsupported.")
        return

    print(f"Benchmarking {dtype_str} on {device} (Size: {size}x{size})...")

    try:
        # 2. Data Preparation
        # Handle NPU tensor creation
        if manager.type == 'npu':
            # NPU tensors need to be created on CPU first then moved
            if dtype == torch.int8:
                a = torch.randint(-128, 127, (size, size), dtype=dtype).npu()
                b = torch.randint(-128, 127, (size, size), dtype=dtype).npu()
            elif 'float8' in dtype_str:
                a = torch.randn(size, size, dtype=torch.float32).to(dtype).npu()
                b = torch.randn(size, size, dtype=torch.float32).to(dtype).npu()
            else:
                a = torch.randn(size, size, dtype=dtype).npu()
                b = torch.randn(size, size, dtype=dtype).npu()
        else:
            # Handle Integer initialization separately
            if dtype == torch.int8:
                # Random integers between -128 and 127
                a = torch.randint(-128, 127, (size, size), device=device, dtype=dtype)
                b = torch.randint(-128, 127, (size, size), device=device, dtype=dtype)
            # Handle FP8 initialization (needs Float -> Cast)
            elif 'float8' in dtype_str:
                a = torch.randn(size, size, device=device, dtype=torch.float32).to(dtype)
                b = torch.randn(size, size, device=device, dtype=torch.float32).to(dtype)
            else:
                a = torch.randn(size, size, device=device, dtype=dtype)
                b = torch.randn(size, size, device=device, dtype=dtype)

        # 3. Execution Selection
        # We define a localized function to call the right math op
        def run_op():
            if dtype == torch.int8 and manager.type == 'cuda':
                # Use _int_mm for CUDA INT8 Tensor Cores
                return torch._int_mm(a, b)
            elif 'float8' in dtype_str and hasattr(torch, '_scaled_mm') and manager.type == 'cuda':
                 # Hopper FP8 usually requires scaled_mm for Tensor Core usage
                 # Using dummy scales
                 scale_a = torch.tensor(1.0, device=device)
                 scale_b = torch.tensor(1.0, device=device)
                 return torch._scaled_mm(a, b, scale_a, scale_b)
            else:
                # NPU and other backends use standard matmul
                return torch.matmul(a, b)

        # Warmup
        for _ in range(3):
            _ = run_op()
        manager.synchronize()

        # Measure
        start = time.perf_counter()
        for _ in range(num_trails):
            _ = run_op()
        manager.synchronize()
        end = time.perf_counter()

        # Calc Stats
        elapsed = end - start
        avg_time = elapsed / num_trails
        
        # Standard Matrix Mult Ops = 2 * N^3
        ops = 2.0 * (size ** 3) 
        perf_tflops = (ops / avg_time) / 1e12

        print(f"  > Result: {perf_tflops:.2f} TFLOPS/TOPS (Time: {avg_time*1000:.2f}ms)")
        
        # Cleanup
        del a, b
        manager.empty_cache()

    except Exception as e:
        print(f"  > Failed: {e}")
        # Helpful hint for the specific user error seen before
        if "addmm" in str(e) and "Char" in str(e):
            print("    (Hint: Standard matmul doesn't support INT8, tried fallback but failed)")

# ==========================================
# 4. Main
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='Future-Ready GPU Benchmark')
    # Add npu to device options
    parser.add_argument('--device', default='cuda', help='cuda, mps, xpu, npu, cpu')
    # Default size set to None so we can calculate it dynamically
    parser.add_argument('--size', type=int, default=None, help='Matrix size for FLOPs')
    parser.add_argument('--mem-size', type=float, default=None, help='Max GB for Bandwidth test')
    parser.add_argument('--skip-bw', action='store_true', help='Skip bandwidth test')
    parser.add_argument('--use-tf32', action='store_true')
    args = parser.parse_args()

    mgr = DeviceManager(args.device)
    info = mgr.get_info()
    
    print(f"\nBenchmark started on: {info.get('name')}")
    
    # --- SMART DEFAULTS ---
    # If user didn't specify size, calculate based on VRAM
    total_mem_gb = info.get('total_memory_gb', 8.0) # Default to 8 if unknown
    
    if args.size is None:
        # For NPU, use a more conservative default size
        if mgr.type == 'npu':
            args.size = 8192  # Conservative default for NPU
            print(f"Using NPU Matrix Size: {args.size}")
        else:
            # Aim to use ~3-4 GB of VRAM for matrices (safe for most)
            # But if VRAM > 40GB (Hopper), go big.
            if total_mem_gb > 40:
                 args.size = 40960 # Heavy load for H100
                 print(f"Detected High-VRAM GPU (>40GB). Auto-scaling Matrix Size to {args.size}")
            else:
                 args.size = 16384 # Safe default for Consumer
                 print(f"Using Standard Matrix Size: {args.size}")

    if args.mem_size is None:
        # Use 25% of VRAM for bandwidth test, capped at 24GB
        target = total_mem_gb * 0.25
        args.mem_size = min(target, 24.0)
        # But ensure at least 2GB
        args.mem_size = max(args.mem_size, 2.0)
        print(f"Auto-scaling Bandwidth Test Size to {args.mem_size:.1f} GB")
    # ----------------------

    if 'capability' in info and info['capability'] != (0, 0):
        print(f"Compute Capability: {info['capability'][0]}.{info['capability'][1]}")

    # 1. Run Bandwidth
    if not args.skip_bw:
        benchmark_bandwidth(mgr, max_size_gb=args.mem_size)

    # 2. Run FLOPs
    print(f"\n{'='*60}")
    print(f"COMPUTE PERFORMANCE ({args.size}x{args.size})")
    print(f"{'='*60}")
    
    # TF32 is CUDA-specific, not applicable for NPU
    if args.use_tf32 and mgr.type == 'cuda':
        print("Enable TF32: ON")
        torch.set_float32_matmul_precision('high')
    elif args.use_tf32 and mgr.type != 'cuda':
        print("Note: TF32 is CUDA-specific, ignoring --use-tf32 flag")

    # Define list of types to test
    dtypes = [torch.float32, torch.float16, torch.bfloat16, torch.int8]
    
    # Dynamically add FP8 if PyTorch supports it
    if hasattr(torch, 'float8_e4m3fn'):
        dtypes.append(torch.float8_e4m3fn)

    for dt in dtypes:
        benchmark_flops(mgr, dt, size=args.size)

    print("\nBenchmark Complete.")

if __name__ == "__main__":
    main()
