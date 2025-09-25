"""
Model Mixins - Backward Compatibility Module
This module provides backward compatibility with the original model_mixins.py
by re-exporting classes with their original names and signatures.
"""
# Import from refactored mixins
from .mixins.complexity import ComplexityTrackingMixin, ComplexityTracker
from .mixins.results import OptimizationResultsMixin, ResultsManagementMixin
from .mixins.validation import ValidationMixin
# Export for backward compatibility
__all__ = [
    'ComplexityTrackingMixin',
    'OptimizationResultsMixin',
    'ComplexityTracker',
    'ResultsManagementMixin',
    'ValidationMixin',
]