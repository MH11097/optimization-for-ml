"""
Path Utilities for Import Management
This module provides clean utilities to add project paths to sys.path
without cluttering the codebase with repetitive path manipulations.
"""
import sys
from pathlib import Path
from typing import Optional

def add_project_root(file_path: Optional[str] = None) -> Path:
    """
    Add the project root directory to sys.path if not already present.
    
    Args:
        file_path: __file__ from the calling module (optional)
        
    Returns:
        Path: The project root directory
    """
    if file_path:
        # Calculate project root relative to the calling file
        current_file = Path(file_path)
        # Navigate up to find the project root (contains src/ directory)
        project_root = current_file.parent
        while project_root.name != 'optimization-for-ml' and project_root.parent != project_root:
            project_root = project_root.parent
    else:
        # Default to assuming we're in src/ somewhere
        project_root = Path(__file__).parent.parent
    
    # Add to sys.path if not present
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    return project_root

def ensure_src_imports(file_path: str) -> None:
    """
    Ensure that src imports work properly from any file.
    
    Args:
        file_path: __file__ from the calling module
    """
    add_project_root(file_path)

# For backward compatibility
def setup_imports(file_path: Optional[str] = None) -> None:
    """Legacy function name for backward compatibility."""
    add_project_root(file_path)