"""
Complexity Tracking Mixins
This module provides mixins for tracking computational complexity
in optimization algorithms, providing backward compatibility with
the original model_mixins module.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

class ComplexityTracker:
    """
    Simple complexity tracker for basic operations counting
    """
    def __init__(self, problem_size: Optional[Tuple[int, int]] = None):
        self.problem_size = problem_size
        self.function_evaluations = 0
        self.gradient_evaluations = 0
        self.matrix_operations = 0
        self.vector_operations = 0
        self.total_operations = 0
        self.peak_memory = 0
        self.convergence_iteration = None
        self.tracking_active = False
    def start_tracking(self):
        """Start complexity tracking"""
        self.tracking_active = True
    def stop_tracking(self):
        """Stop complexity tracking"""
        self.tracking_active = False
    def record_function_evaluation(self, matrix_shape=None):
        """Record a function evaluation"""
        if self.tracking_active:
            self.function_evaluations += 1
            self.total_operations += 1
    def record_gradient_evaluation(self, matrix_shape=None):
        """Record a gradient evaluation"""
        if self.tracking_active:
            self.gradient_evaluations += 1
            self.total_operations += 1
    def record_matrix_operation(self, operation_type="basic"):
        """Record a matrix operation"""
        if self.tracking_active:
            self.matrix_operations += 1
            self.total_operations += 1
    def record_vector_operation(self, vector_size=None, operation_type="basic"):
        """Record a vector operation"""
        if self.tracking_active:
            self.vector_operations += 1
            self.total_operations += 1
    def record_memory_allocation(self, size):
        """Record memory allocation"""
        if self.tracking_active:
            self.peak_memory = max(self.peak_memory, size)
    def mark_convergence(self, iteration):
        """Mark the convergence iteration"""
        if self.tracking_active:
            self.convergence_iteration = iteration
    def get_summary(self) -> Dict[str, Any]:
        """Get complexity summary"""
        return {
            "total_operations": self.total_operations,
            "function_evaluations": self.function_evaluations,
            "gradient_evaluations": self.gradient_evaluations,
            "matrix_operations": self.matrix_operations,
            "vector_operations": self.vector_operations,
            "peak_memory": self.peak_memory,
            "convergence_iteration": self.convergence_iteration,
            "problem_size": self.problem_size
        }

class ComplexityTrackingMixin:
    """
    Mixin class to add computational complexity tracking to optimization models
    This mixin provides:
    - Initialization of complexity tracker
    - Common tracking operations
    - Integration with save_results methods
    Maintains backward compatibility with original utils.model_mixins
    """
    def init_complexity_tracker(self, X, y):
        """
        Initialize complexity tracker with problem size
        Args:
            X: Feature matrix (without bias)
            y: Target vector
        """
        self.complexity_tracker = ComplexityTracker(
            problem_size=(X.shape[0], X.shape[1])
        )
        self.complexity_tracker.start_tracking()
    def track_function_evaluation(self, matrix_shape=None):
        """Record function evaluation with optional matrix size"""
        if hasattr(self, 'complexity_tracker') and self.complexity_tracker:
            self.complexity_tracker.record_function_evaluation(matrix_shape)
    def track_gradient_evaluation(self, matrix_shape=None):
        """Record gradient evaluation with optional matrix size"""
        if hasattr(self, 'complexity_tracker') and self.complexity_tracker:
            self.complexity_tracker.record_gradient_evaluation(matrix_shape)
    def track_vector_operation(self, vector_size, operation_type="basic"):
        """Record vector operations"""
        if hasattr(self, 'complexity_tracker') and self.complexity_tracker:
            self.complexity_tracker.record_vector_operation(vector_size, operation_type)
    def track_matrix_operation(self, operation_type="basic"):
        """Record matrix operations"""
        if hasattr(self, 'complexity_tracker') and self.complexity_tracker:
            self.complexity_tracker.record_matrix_operation(operation_type)
    def track_memory_allocation(self, size):
        """Record memory allocation"""
        if hasattr(self, 'complexity_tracker') and self.complexity_tracker:
            self.complexity_tracker.record_memory_allocation(size)
    def mark_convergence_tracking(self, iteration):
        """Mark convergence iteration"""
        if hasattr(self, 'complexity_tracker') and self.complexity_tracker:
            self.complexity_tracker.mark_convergence(iteration)
    def end_iteration_tracking(self):
        """End iteration tracking"""
        if hasattr(self, 'complexity_tracker') and self.complexity_tracker:
            self.complexity_tracker.stop_tracking()
    def get_complexity_summary(self):
        """Get complexity summary if available"""
        if hasattr(self, 'complexity_tracker') and self.complexity_tracker:
            return self.complexity_tracker.get_summary()
        return None
    def integrate_complexity_to_results(self, results_data, results_dir=None):
        """
        Integrate complexity analysis into results data
        Args:
            results_data: Existing results dictionary
            results_dir: Optional directory to save detailed complexity file
        Returns:
            Updated results dictionary with complexity data
        """
        complexity_analysis = self.get_complexity_summary()
        if complexity_analysis:
            # Add to main results
            results_data["computational_complexity"] = complexity_analysis
            # Save separate detailed complexity file if results_dir provided
            if results_dir:
                complexity_path = Path(results_dir) / "complexity_analysis.json"
                with open(complexity_path, 'w') as f:
                    json.dump(complexity_analysis, f, indent=2)
                print(f"[SAVED] Detailed complexity analysis saved to: {complexity_path.name}")
        return results_data
    def print_complexity_summary(self):
        """Print a summary of complexity metrics"""
        summary = self.get_complexity_summary()
        if summary:
            print(f"[COMPLEXITY] Summary:")
            print(f"   Total operations: {summary.get('total_operations', 0):,}")
            print(f"   Function evaluations: {summary.get('function_evaluations', 0)}")
            print(f"   Gradient evaluations: {summary.get('gradient_evaluations', 0)}")
            print(f"   Peak memory: {summary.get('peak_memory', 0):,} elements")