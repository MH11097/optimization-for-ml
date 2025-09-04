#!/usr/bin/env python3
"""
Script to run all algorithm setup experiments automatically without popup windows.
Automatically discovers and runs all setup scripts in the src/algorithms/ directory.
"""

import os
import sys
import importlib.util
import traceback
from pathlib import Path
from typing import List, Tuple, Dict
import time


def find_all_setup_scripts(base_dir: str = "src/algorithms") -> List[Tuple[str, str]]:
    """
    Find all setup scripts in the algorithms directory.
    
    Returns:
        List of tuples (algorithm_name, script_path)
    """
    setup_scripts = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Directory {base_dir} not found!")
        return setup_scripts
    
    # Walk through all subdirectories
    for algorithm_dir in base_path.iterdir():
        if algorithm_dir.is_dir() and algorithm_dir.name != "__pycache__":
            algorithm_name = algorithm_dir.name
            
            # Look for setup scripts (files matching pattern)
            for script_file in algorithm_dir.glob("*setup*.py"):
                if script_file.name != "__pycache__" and not script_file.name.startswith("_"):
                    try:
                        relative_path = str(script_file.relative_to(Path.cwd()))
                    except ValueError:
                        # Fallback to absolute path if relative path fails
                        relative_path = str(script_file.absolute())
                    setup_scripts.append((algorithm_name, relative_path))
    
    # Sort by algorithm name and script name for consistent order
    setup_scripts.sort(key=lambda x: (x[0], x[1]))
    return setup_scripts


def run_setup_script(script_path: str) -> Dict[str, any]:
    """
    Run a single setup script and return results.
    
    Args:
        script_path: Path to the setup script
        
    Returns:
        Dictionary with execution results
    """
    result = {
        'script': script_path,
        'success': False,
        'error': None,
        'execution_time': 0
    }
    
    try:
        start_time = time.time()
        
        # Convert path to module name
        module_name = script_path.replace('/', '.').replace('\\', '.').replace('.py', '')
        
        # Load and execute the module
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None:
            raise ImportError(f"Could not load spec for {script_path}")
            
        module = importlib.util.module_from_spec(spec)
        
        # Add to sys.modules to handle relative imports
        sys.modules[module_name] = module
        
        # Execute the module
        spec.loader.exec_module(module)
        
        # Call main function if it exists
        if hasattr(module, 'main'):
            module.main()
        
        result['success'] = True
        result['execution_time'] = time.time() - start_time
        
    except Exception as e:
        result['error'] = str(e).encode('ascii', errors='replace').decode('ascii')
        result['execution_time'] = time.time() - start_time
        safe_error = str(e).encode('ascii', errors='replace').decode('ascii')
        try:
            print(f"Error running {script_path}: {safe_error}")
        except UnicodeEncodeError:
            print(f"Error running {script_path}: [Unicode encoding error]")
        # print(f"   Full traceback: {traceback.format_exc()}")
    
    return result


def print_progress_bar(current: int, total: int, script_name: str, width: int = 50):
    """Print a progress bar for the current execution."""
    percent = current / total
    filled = int(width * percent)
    bar = '#' * filled + '-' * (width - filled)
    print(f'\r[{bar}] {percent:.1%} - Running: {script_name[:40]}...', end='', flush=True)


def main():
    """Main function to run all setup scripts."""
    print("AUTOMATED SETUP RUNNER")
    print("=" * 60)
    
    # Find all setup scripts
    print("Discovering setup scripts...")
    setup_scripts = find_all_setup_scripts()
    
    if not setup_scripts:
        print("No setup scripts found!")
        return
    
    print(f"Found {len(setup_scripts)} setup scripts:")
    for i, (algorithm, script) in enumerate(setup_scripts[:5], 1):  # Show first 5
        print(f"   {i}. {algorithm}: {Path(script).name}")
    if len(setup_scripts) > 5:
        print(f"   ... and {len(setup_scripts) - 5} more")
    
    print(f"\nStarting execution of {len(setup_scripts)} experiments...")
    print("   (No popup windows will appear - all plots saved to files)")
    print("-" * 60)
    
    # Track results
    results = []
    successful = 0
    failed = 0
    
    # Execute each script
    for i, (algorithm_name, script_path) in enumerate(setup_scripts, 1):
        script_display_name = f"{algorithm_name}/{Path(script_path).name}"
        print_progress_bar(i-1, len(setup_scripts), script_display_name)
        
        # Run the script
        result = run_setup_script(script_path)
        results.append(result)
        
        if result['success']:
            successful += 1
            status = "[OK]"
        else:
            failed += 1
            status = "[FAIL]"
        
        # Clear progress bar and show result
        print(f"\r{status} {script_display_name:<50} ({result['execution_time']:.1f}s)")
    
    # Final summary
    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)
    print(f"   Total Experiments: {len(setup_scripts)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Success Rate: {successful/len(setup_scripts)*100:.1f}%")
    
    if failed > 0:
        print(f"\nFailed experiments:")
        for result in results:
            if not result['success']:
                script_name = Path(result['script']).name
                print(f"   - {script_name}: {result['error']}")
    
    total_time = sum(r['execution_time'] for r in results)
    print(f"\nTotal execution time: {total_time:.1f} seconds")
    print(f"All results saved to: data/03_algorithms/")
    print(f"All visualization plots saved automatically!")
    
    print("\nAll experiments completed! Check the data directory for results.")


if __name__ == "__main__":
    main()