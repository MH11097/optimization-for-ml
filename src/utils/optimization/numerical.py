"""
Numerical stability utilities for optimization algorithms.
"""
import numpy as np
from typing import Dict, Any, Optional

def check_numerical_stability(values: np.ndarray, name: str = "values") -> Dict[str, Any]:
    """
    Check numerical stability of values.
    Args:
        values: Array of values to check
        name: Name for reporting
    Returns:
        Dictionary with stability information
    """
    stability_info = {
        'is_stable': True,
        'has_nan': False,
        'has_inf': False,
        'issues': []
    }
    # Check for NaN
    if np.any(np.isnan(values)):
        stability_info['has_nan'] = True
        stability_info['is_stable'] = False
        stability_info['issues'].append(f"{name} contains NaN values")
    # Check for infinity
    if np.any(np.isinf(values)):
        stability_info['has_inf'] = True
        stability_info['is_stable'] = False
        stability_info['issues'].append(f"{name} contains infinite values")
    # Check for very large values
    max_val = np.max(np.abs(values))
    if max_val > 1e12:
        stability_info['issues'].append(f"{name} contains very large values: {max_val}")
    return stability_info

def handle_numerical_issues(values: np.ndarray, strategy: str = 'clip') -> np.ndarray:
    """
    Handle numerical issues in optimization values.
    Args:
        values: Values to fix
        strategy: Strategy for handling issues ('clip', 'replace')
    Returns:
        Fixed values
    """
    if strategy == 'clip':
        # Clip to reasonable range
        return np.clip(values, -1e12, 1e12)
    elif strategy == 'replace':
        # Replace NaN/Inf with zeros
        fixed = values.copy()
        fixed[np.isnan(fixed)] = 0
        fixed[np.isinf(fixed)] = 0
        return fixed
    else:
        return values