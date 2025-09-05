"""
Model Mixins for Common Functionality

Provides mixin classes for common functionality across optimization models,
including computational complexity tracking.

Author: Claude Code Assistant
Date: 2025-01-09
"""

from .computational_complexity import ComputationalComplexityTracker
from pathlib import Path
import json


class ComplexityTrackingMixin:
    """
    Mixin class to add computational complexity tracking to optimization models
    
    This mixin provides:
    - Initialization of complexity tracker
    - Common tracking operations
    - Integration with save_results methods
    """
    
    def init_complexity_tracker(self, X, y):
        """
        Initialize complexity tracker with problem size
        
        Args:
            X: Feature matrix (without bias)
            y: Target vector
        """
        self.complexity_tracker = ComputationalComplexityTracker(
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
    
    def track_memory_allocation(self, size):
        """Record memory allocation"""
        if hasattr(self, 'complexity_tracker') and self.complexity_tracker:
            self.complexity_tracker.record_memory_allocation(size)
    
    def track_matrix_operation(self, matrix_shape):
        """Record matrix operations"""
        if hasattr(self, 'complexity_tracker') and self.complexity_tracker:
            self.complexity_tracker.record_matrix_vector_multiplication(matrix_shape)
    
    def end_iteration_tracking(self):
        """End current iteration tracking"""
        if hasattr(self, 'complexity_tracker') and self.complexity_tracker:
            self.complexity_tracker.end_iteration()
    
    def mark_convergence_tracking(self, iteration):
        """Mark convergence iteration"""
        if hasattr(self, 'complexity_tracker') and self.complexity_tracker:
            self.complexity_tracker.mark_convergence(iteration)
    
    def get_complexity_analysis(self, final_iteration, converged):
        """Get comprehensive complexity analysis"""
        if hasattr(self, 'complexity_tracker') and self.complexity_tracker:
            return self.complexity_tracker.get_complexity_analysis(final_iteration, converged)
        return None
    
    def get_complexity_summary(self):
        """Get complexity summary stats"""
        if hasattr(self, 'complexity_tracker') and self.complexity_tracker:
            return self.complexity_tracker.get_summary_stats()
        return {}
    
    def save_complexity_to_results(self, results_data, final_iteration, converged, results_dir=None):
        """
        Add complexity metrics to results dictionary and optionally save separate file
        
        Args:
            results_data: Dictionary to add complexity metrics to
            final_iteration: Final iteration count
            converged: Whether algorithm converged
            results_dir: Optional path to save separate complexity analysis file
            
        Returns:
            Updated results_data dictionary
        """
        complexity_analysis = self.get_complexity_analysis(final_iteration, converged)
        
        if complexity_analysis:
            # Add to main results
            results_data["computational_complexity"] = complexity_analysis
            
            # Save separate detailed complexity file if results_dir provided
            if results_dir:
                complexity_path = Path(results_dir) / "complexity_analysis.json"
                with open(complexity_path, 'w') as f:
                    json.dump(complexity_analysis, f, indent=2)
                print(f"📊 Detailed complexity analysis saved to: {complexity_path.name}")
        
        return results_data
    
    def print_complexity_summary(self):
        """Print a summary of complexity metrics"""
        summary = self.get_complexity_summary()
        if summary:
            print(f"📊 Complexity Summary:")
            print(f"   Total operations: {summary.get('total_operations', 0):,}")
            print(f"   Function evaluations: {summary.get('function_evaluations', 0)}")
            print(f"   Gradient evaluations: {summary.get('gradient_evaluations', 0)}")
            print(f"   Peak memory: {summary.get('peak_memory', 0):,} elements")


class OptimizationResultsMixin:
    """
    Mixin class for standardized optimization results handling
    """
    
    def create_standard_results_dict(self, algorithm_name, loss_function):
        """
        Create a standardized results dictionary structure
        
        Args:
            algorithm_name: Name of the algorithm
            loss_function: Loss function used
            
        Returns:
            Standard results dictionary structure
        """
        if not hasattr(self, 'weights') or self.weights is None:
            raise ValueError("Model has not been trained. Call fit() first.")
            
        return {
            "algorithm": algorithm_name,
            "loss_function": loss_function.upper(),
            "training_results": {
                "training_time": getattr(self, 'training_time', 0.0),
                "converged": getattr(self, 'converged', False),
                "final_iteration": getattr(self, 'final_iteration', 0),
                "final_loss": float(self.loss_history[-1]) if self.loss_history else 0.0,
                "final_gradient_norm": float(self.gradient_norms[-1]) if self.gradient_norms else 0.0
            },
            "weights_analysis": self._get_weights_analysis(),
            "convergence_analysis": self._get_convergence_analysis()
        }
    
    def _get_weights_analysis(self):
        """Get standardized weights analysis"""
        import numpy as np
        
        if self.weights is None:
            return {}
            
        return {
            "n_features": len(self.weights) - 1,  # Excluding bias
            "n_weights_total": len(self.weights),  # Including bias  
            "bias_value": float(self.weights[-1]),
            "weights_without_bias": self.weights[:-1].tolist(),
            "complete_weight_vector": self.weights.tolist(),
            "weights_stats": {
                "min": float(np.min(self.weights[:-1])),
                "max": float(np.max(self.weights[:-1])),
                "mean": float(np.mean(self.weights[:-1])),
                "std": float(np.std(self.weights[:-1]))
            }
        }
    
    def _get_convergence_analysis(self):
        """Get standardized convergence analysis"""
        final_iteration = getattr(self, 'final_iteration', 0)
        
        analysis = {
            "iterations_to_converge": final_iteration
        }
        
        if hasattr(self, 'loss_history') and len(self.loss_history) > 1:
            analysis.update({
                "final_cost_change": float(self.loss_history[-1] - self.loss_history[-2]),
                "loss_reduction_ratio": float(self.loss_history[0] / self.loss_history[-1])
            })
            
        return analysis