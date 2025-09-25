"""
Advanced Mixins
This module provides enhanced mixins for complexity tracking,
results management, and other advanced functionality.
Maintains backward compatibility with original utils.model_mixins
"""
from .complexity import ComplexityTrackingMixin, ComplexityTracker
from .results import ResultsManagementMixin, OptimizationResultsMixin
from .validation import ValidationMixin
__all__ = [
    # Modern interfaces
    'ComplexityTrackingMixin',
    'ComplexityTracker',
    'ResultsManagementMixin',
    'ValidationMixin',
    # Backward compatibility
    'OptimizationResultsMixin',
]