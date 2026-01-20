import os
import json
import debugpy
import socket
import random
import torch
import functools
from contextlib import contextmanager

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

def update_vscode_launch_file(host: str, port: int):
    """Update the .vscode/launch.json file with the new host and port."""
    launch_file_path = ".vscode/launch.json"
    # Desired configuration
    new_config = {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "bash_debug",
                "type": "debugpy",
                "request": "attach",
                "connect": {
                    "host": host,
                    "port": port
                },
                "justMyCode": False
            },
        ]
    }

    # Ensure the .vscode directory exists
    if not os.path.exists(".vscode"):
        os.makedirs(".vscode")

    # Write the updated configuration to launch.json
    with open(launch_file_path, "w") as f:
        json.dump(new_config, f, indent=4)
    print(f"Updated {launch_file_path} with host: {host} and port: {port}")

def is_port_in_use(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def setup_debug(is_main_process=True, max_retries=10, port_range=(10000, 20000)):
    if is_main_process:
        host = os.environ['SLURM_NODELIST'].split(',')[0]

        for _ in range(max_retries):
            port = random.randint(*port_range)
            try:
                if is_port_in_use(host, port):
                    print(f"Port {port} is already in use, trying another...")
                    continue

                # 更新 launch.json
                update_vscode_launch_file(host, port)

                print("master_addr = ", host)
                debugpy.listen((host, port))
                print(f"Waiting for debugger attach at port {port}...", flush=True)
                debugpy.wait_for_client()
                print("Debugger attached", flush=True)
                return
            except Exception as e:
                print(f"Failed to bind to port {port}: {e}")

        raise RuntimeError("Could not find a free port for debugpy after several attempts.")