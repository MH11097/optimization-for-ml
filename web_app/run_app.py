#!/usr/bin/env python3
"""
Startup script for the Optimization Algorithm Visualizer web application.
This script provides an easy way to run the web application with proper
configuration and error handling.
"""
import os
import sys
from pathlib import Path
# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import flask
        import pandas
        import numpy
        print("[OK] All required dependencies are available")
        return True
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("Please install requirements: pip install -r ../requirements.txt")
        return False
def check_data_directory():
    """Check if data directory exists."""
    data_dir = project_root / "data" / "03_algorithms"
    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}")
        print("Please ensure the optimization algorithm data is available")
        return False
    
    # Check for at least one algorithm
    algorithm_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if not algorithm_dirs:
        print(f"[ERROR] No algorithm data found in: {data_dir}")
        return False
    
    print(f"[OK] Found {len(algorithm_dirs)} algorithm(s): {', '.join([d.name for d in algorithm_dirs])}")
    return True
def main():
    """Main function to start the web application."""
    print("=" * 60)
    print("Optimization Algorithm Visualizer")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check data directory
    if not check_data_directory():
        print("\nWarning: Data directory issues detected.")
        print("The application will start but may not display data properly.")
    
    # Import and run app
    try:
        from app import app
        print("\n" + "=" * 60)
        print("Starting web application...")
        print("Access the application at: http://localhost:5000")
        print("Press Ctrl+C to stop the server")
        print("=" * 60)
        
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        print("\n\nApplication stopped by user.")
    except Exception as e:
        print(f"\nError starting application: {e}")
        sys.exit(1)
if __name__ == "__main__":
    main()