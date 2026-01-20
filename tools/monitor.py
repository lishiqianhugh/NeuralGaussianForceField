import torch
import time
import functools
import numpy as np
import pandas as pd
from contextlib import contextmanager

@contextmanager
def timer(description="Execution time"):
    """
    Context manager to time the execution of a block of code.
    Includes CUDA synchronization for accurate GPU timing.
    
    Args:
        description (str): A description of the operation being timed.
        
    Usage:
        with timer("My operation"):
            # code to time
            
        # Or as a function decorator
        @timer("Function execution")
        def my_function():
            # function code
    """
    # Synchronize CUDA before starting timing
    torch.cuda.synchronize()
    
    start = time.time()
    try:
        yield
    finally:
        # Synchronize CUDA before ending timing
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"{description}: {elapsed:.4f} seconds")

def timeit(func=None, *, description=None):
    """
    Decorator to time function execution.
    
    Args:
        func: The function to decorate
        description: Optional description, defaults to function name
        
    Usage:
        @timeit
        def my_function():
            pass
            
        @timeit(description="Custom description")
        def another_function():
            pass
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            desc = description or f.__name__
            with timer(desc):
                return f(*args, **kwargs)
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)

@contextmanager
def memory_monitor(description="Memory usage"):
    """
    Context manager to monitor GPU memory usage during code execution.
    Includes cache clearing for accurate memory measurement.
    
    Args:
        description (str): A description of the operation being monitored.
        
    Usage:
        with memory_monitor("My operation"):
            # code to monitor
            
        # Or as a function decorator
        @memory_monitor("Function memory")
        def my_function():
            # function code
    """

    # Clear cache and synchronize before measurement
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Reset peak memory stats and record initial memory usage
    torch.cuda.reset_peak_memory_stats()
    initial_allocated = torch.cuda.memory_allocated()
    initial_reserved = torch.cuda.memory_reserved()
    
    try:
        yield
    finally:
        # Synchronize and clear cache before final measurement
        torch.cuda.synchronize()
        
        # Record final and peak memory usage
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        
        torch.cuda.empty_cache()
        final_allocated = torch.cuda.memory_allocated()
        final_reserved = torch.cuda.memory_reserved()
        
        # Calculate differences
        allocated_diff = final_allocated - initial_allocated
        reserved_diff = final_reserved - initial_reserved
        
        print(f"{description} - Memory Usage:")
        print(f"  Allocated: {initial_allocated / 1024**2:.2f} MB -> {final_allocated / 1024**2:.2f} MB (Δ: {allocated_diff / 1024**2:+.2f} MB)")
        print(f"  Peak Allocated: {peak_allocated / 1024**2:.2f} MB")
        print(f"  Reserved:  {initial_reserved / 1024**2:.2f} MB -> {final_reserved / 1024**2:.2f} MB (Δ: {reserved_diff / 1024**2:+.2f} MB)")
        print(f"  Peak Reserved:  {peak_reserved / 1024**2:.2f} MB")

def memoryit(func=None, *, description=None):
    """
    Decorator to monitor GPU memory usage during function execution.
    
    Args:
        func: The function to decorate
        description: Optional description, defaults to function name
        
    Usage:
        @memoryit
        def my_function():
            pass
            
        @memoryit(description="Custom description")
        def another_function():
            pass
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            desc = description or f.__name__
            with memory_monitor(desc):
                return f(*args, **kwargs)
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)

def save_data_as_csv(data, file_path):
    """
    Saves a PyTorch Tensor or NumPy array to a CSV file.

    Args:
        data: The input data (PyTorch Tensor or NumPy array).
        file_path: The path where the CSV file will be saved.
    """
    # Convert PyTorch Tensor to NumPy array
    if isinstance(data, torch.Tensor):
        # Move tensor to CPU if it's on GPU
        if data.is_cuda:
            data = data.cpu()
        # Detach tensor from computation graph
        data = data.detach()
        # Convert to NumPy array
        numpy_array = data.numpy()
    elif isinstance(data, np.ndarray):
        numpy_array = data
    else:
        raise TypeError("Input must be a PyTorch Tensor or a NumPy array.")

    # Reshape if dimension is greater than 2
    if len(numpy_array.shape) > 2:
        numpy_array = numpy_array.reshape(-1, numpy_array.shape[-1])
    elif len(numpy_array.shape) == 1:
        # Reshape 1D array to 2D array with one column
        numpy_array = numpy_array.reshape(-1, 1)

    # Create a Pandas DataFrame
    df = pd.DataFrame(numpy_array)

    # Save DataFrame to CSV file
    df.to_csv(file_path, index=False, header=False)
