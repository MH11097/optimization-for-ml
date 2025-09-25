#!/usr/bin/env python3
"""
Script to run experimental setups from number x to number y
"""
import subprocess
import sys
import os
import glob
import argparse
def run_setup(algorithm, setup_number):
    """Run all setup files matching the given setup number"""
    # Search for files matching the pattern across all subdirectories
    pattern = f"src/experimental_setups/{algorithm}/{setup_number:03d}_*.py"
    matching_files = glob.glob(pattern, recursive=True)
    
    if not matching_files:
        print(f"[SKIP] Setup {setup_number}: No files found matching pattern {pattern}")
        return False
    
    all_success = True
    for script_path in matching_files:
        print(f"[RUN] Running setup {setup_number}: {script_path}")
        
        try:
            result = subprocess.run([sys.executable, script_path], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"[OK] Setup {setup_number}: {script_path} completed successfully")
            else:
                print(f"[FAIL] Setup {setup_number}: {script_path} failed with return code {result.returncode}")
                if result.stderr:
                    print(f"   Error: {result.stderr[:10000]}")
                all_success = False
                
      
        except Exception as e:
            print(f"[ERROR] Setup {setup_number}: {script_path} exception - {e}")
            all_success = False
    
    return all_success
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run experimental setups from number x to number y",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "algorithm", 
        type=str, 
        nargs="?", 
        default='gradient_descent',
        help="Algorithm"
    )
    parser.add_argument(
        "start", 
        type=int, 
        nargs="?", 
        default=111,
        help="Starting setup number"
    )
    parser.add_argument(
        "end", 
        type=int, 
        nargs="?", 
        default=130,
        help="Ending setup number (inclusive)"
    )
    return parser.parse_args()
def main():
    """Run experimental setups from start to end number"""
    args = parse_arguments()
    
    print(f"Starting experimental setups for {args.algorithm} {args.start}-{args.end}...")
    print("=" * 50)
    
    successful = 0
    failed = 0
    
    for setup_num in range(args.start, args.end + 1):
        success = run_setup(args.algorithm, setup_num)
        if success:
            successful += 1
        else:
            failed += 1
        print()  # Empty line for readability
    
    print("=" * 50)
    print(f"Summary:")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {successful + failed}")
    
    if failed == 0:
        print("All setups completed successfully!")
    else:
        print(f"WARNING: {failed} setups failed")
if __name__ == "__main__":
    main()