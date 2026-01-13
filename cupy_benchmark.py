# cupy_benchmark.py
# Pure CuPy benchmark for FP32/FP16/INT8 matmul (no PyTorch)
# Measures real TFLOPS/TOPS on CUDA (Tensor Cores if available)

import cupy as cp  # CuPy (NumPy on GPU)
import time
import math

def is_dtype_supported(gpu_type, dtype):
    """Simple check: CuPy supports INT8/FP16 on CUDA 7.5+ (Volta+ hardware)."""
    if gpu_type != 'cuda':
        return False
    if dtype == cp.int8:
        return cp.cuda.runtime.getDeviceProperties(0)['major'] >= 7  # Volta+
    return True

def benchmark_flops(gpu_type: str, dtype, size: int, num_trials: int = 5):
    """
    Measures Compute Performance (TFLOPS/TOPS) – CuPy version
    """
    dtype_str = str(dtype).replace("cupy.", "")
    
    # 1. Check Capability first
    if not is_dtype_supported(gpu_type, dtype):
        print(f"Skipping {dtype_str}: Hardware capability too low or unsupported.")
        return
    
    print(f"Benchmarking {dtype_str} on {gpu_type} (Size: {size}x{size})...")
    
    try:
        # 2. Data Preparation
        if dtype == cp.int8:
            # Random integers between -128 and 127
            a = cp.random.randint(-128, 127, (size, size), dtype=dtype)
            b = cp.random.randint(-128, 127, (size, size), dtype=dtype)
        else:
            # FP16/FP32: random normal
            a = cp.random.randn(size, size, dtype=dtype)
            b = cp.random.randn(size, size, dtype=dtype)
        
        # 3. Execution: CuPy matmul (cuBLAS-backed, supports INT8 Tensor Cores)
        def run_op():
            return cp.matmul(a, b)
        
        # Warmup
        for _ in range(3):
            _ = run_op()
        cp.cuda.Device().synchronize()  # Equivalent to torch.cuda.synchronize()
        
        # Measure
        times = []
        for _ in range(num_trials):
            start = time.perf_counter()
            _ = run_op()
            cp.cuda.Device().synchronize()
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = sum(times) / num_trials
        
        # Calc Stats: Standard Matrix Mult Ops = 2 * N^3
        ops = 2.0 * (size ** 3)
        if dtype == cp.int8:
            perf_tops = (ops / avg_time) / 1e12  # TOPS for INT8
            unit = "TOPS"
        else:
            perf_tflops = (ops / avg_time) / 1e12  # TFLOPS for FP
            unit = "TFLOPS"
            perf_tops = perf_tflops  # Alias for consistency
        
        print(f"  > Result: {perf_tops:.2f} {unit} (Avg Time: {avg_time*1000:.2f}ms)")
        
        # Cleanup
        del a, b
        cp.get_default_memory_pool().free_all_blocks()  # Equivalent to empty_cache()
        
    except Exception as e:
        print(f"  > Failed: {e}")

# Main: Auto-size for your 12GB RTX 3060 (~5GB safe usage)
if cp.cuda.is_available():
    gpu_type = 'cuda'
    total_mem_gb = cp.cuda.Device(0).mem_info[1] / 1e9
    safe_size = int(math.sqrt((5e9 / 3 / 2)))  # ~5GB for FP16 (2 bytes/elem)
    safe_size = (safe_size // 128) * 128
    safe_size = max(safe_size, 8192)
else:
    print("CUDA not available – run on CPU with NumPy (no Tensor Cores).")
    exit()

print(f"RTX 3060 detected ({total_mem_gb:.1f} GB VRAM) – Using size {safe_size}x{safe_size}")
print("=" * 60)

# Run benchmarks for FP32, FP16, INT8
benchmark_flops(gpu_type, cp.float32, safe_size)
benchmark_flops(gpu_type, cp.float16, safe_size)
benchmark_flops(gpu_type, cp.int8, safe_size)
