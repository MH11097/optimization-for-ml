"""
Computational Complexity Tracking Module

Provides tools to track and analyze computational complexity of optimization algorithms
independent of hardware performance. Tracks operations count, memory usage, and
algorithmic efficiency metrics.

Author: Claude Code Assistant  
Date: 2025-01-09
"""

import numpy as np
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class ComplexityMetrics:
    """Container for computational complexity metrics"""
    function_evaluations: int = 0
    gradient_evaluations: int = 0
    matrix_vector_multiplications: int = 0
    vector_operations: int = 0
    memory_allocations: int = 0
    peak_memory_size: int = 0
    total_operations: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for JSON serialization"""
        return {
            "function_evaluations": self.function_evaluations,
            "gradient_evaluations": self.gradient_evaluations,
            "matrix_vector_multiplications": self.matrix_vector_multiplications,
            "vector_operations": self.vector_operations,
            "memory_allocations": self.memory_allocations,
            "peak_memory_size": self.peak_memory_size,
            "total_operations": self.total_operations
        }


class ComputationalComplexityTracker:
    """
    Tracks computational complexity metrics for optimization algorithms
    
    Provides hardware-independent metrics to evaluate algorithm efficiency:
    - Operations counting (function/gradient evaluations)
    - Memory usage tracking
    - Algorithmic complexity analysis
    """
    
    def __init__(self, problem_size: Optional[tuple] = None):
        """
        Initialize complexity tracker
        
        Args:
            problem_size: (n_samples, n_features) for complexity analysis
        """
        self.metrics = ComplexityMetrics()
        self.problem_size = problem_size
        self.n_samples, self.n_features = problem_size if problem_size else (0, 0)
        
        # Track per-iteration metrics
        self.iteration_metrics: List[ComplexityMetrics] = []
        self.current_iteration_metrics = ComplexityMetrics()
        
        # Algorithm-specific parameters
        self.convergence_iteration: Optional[int] = None
        self.start_time: Optional[float] = None
        
    def reset(self):
        """Reset all metrics to zero"""
        self.metrics = ComplexityMetrics()
        self.iteration_metrics.clear()
        self.current_iteration_metrics = ComplexityMetrics()
        self.convergence_iteration = None
        self.start_time = None
    
    def start_tracking(self):
        """Start tracking computational operations"""
        self.start_time = time.time()
        self.reset()
    
    def record_function_evaluation(self, matrix_size: Optional[tuple] = None):
        """
        Record a function evaluation (loss computation)
        
        Args:
            matrix_size: (rows, cols) for complexity calculation
        """
        self.metrics.function_evaluations += 1
        self.current_iteration_metrics.function_evaluations += 1
        
        # Estimate operations for loss computation
        if matrix_size:
            n, d = matrix_size
            ops = n * d + n  # Matrix-vector multiply + residual computation
            self.metrics.total_operations += ops
            self.current_iteration_metrics.total_operations += ops
    
    def record_gradient_evaluation(self, matrix_size: Optional[tuple] = None):
        """
        Record a gradient evaluation
        
        Args:
            matrix_size: (rows, cols) for complexity calculation  
        """
        self.metrics.gradient_evaluations += 1
        self.current_iteration_metrics.gradient_evaluations += 1
        
        # Estimate operations for gradient computation
        if matrix_size:
            n, d = matrix_size
            ops = n * d * 2  # X^T @ residuals typically
            self.metrics.total_operations += ops
            self.current_iteration_metrics.total_operations += ops
    
    def record_matrix_vector_multiplication(self, matrix_shape: tuple):
        """
        Record matrix-vector multiplication
        
        Args:
            matrix_shape: (rows, cols) of the matrix
        """
        self.metrics.matrix_vector_multiplications += 1
        self.current_iteration_metrics.matrix_vector_multiplications += 1
        
        rows, cols = matrix_shape
        ops = rows * cols
        self.metrics.total_operations += ops
        self.current_iteration_metrics.total_operations += ops
    
    def record_vector_operation(self, vector_size: int, operation_type: str = "basic"):
        """
        Record vector operations (addition, scaling, etc.)
        
        Args:
            vector_size: Size of the vector
            operation_type: Type of operation ('basic', 'norm', 'dot')
        """
        self.metrics.vector_operations += 1
        self.current_iteration_metrics.vector_operations += 1
        
        if operation_type == "basic":
            ops = vector_size
        elif operation_type == "norm":
            ops = vector_size * 2  # square + sum + sqrt
        elif operation_type == "dot":
            ops = vector_size * 2  # multiply + sum
        else:
            ops = vector_size
            
        self.metrics.total_operations += ops
        self.current_iteration_metrics.total_operations += ops
    
    def record_memory_allocation(self, size: int):
        """
        Record memory allocation
        
        Args:
            size: Number of elements allocated
        """
        self.metrics.memory_allocations += 1
        self.current_iteration_metrics.memory_allocations += 1
        
        if size > self.metrics.peak_memory_size:
            self.metrics.peak_memory_size = size
            self.current_iteration_metrics.peak_memory_size = size
    
    def end_iteration(self):
        """End current iteration and save metrics"""
        self.iteration_metrics.append(self.current_iteration_metrics)
        self.current_iteration_metrics = ComplexityMetrics()
    
    def mark_convergence(self, iteration: int):
        """Mark the iteration when algorithm converged"""
        self.convergence_iteration = iteration
    
    def get_complexity_analysis(self, final_iteration: int, converged: bool) -> Dict[str, Any]:
        """
        Analyze computational complexity and return comprehensive metrics
        
        Args:
            final_iteration: Final iteration number
            converged: Whether algorithm converged
            
        Returns:
            Dictionary with comprehensive complexity analysis
        """
        total_iterations = len(self.iteration_metrics)
        
        # Basic metrics
        analysis = {
            "basic_metrics": self.metrics.to_dict(),
            "iterations": {
                "total_iterations": total_iterations,
                "final_iteration": final_iteration,
                "converged": converged,
                "convergence_iteration": self.convergence_iteration
            }
        }
        
        # Per-iteration averages
        if total_iterations > 0:
            analysis["per_iteration_averages"] = {
                "function_evaluations_per_iter": self.metrics.function_evaluations / total_iterations,
                "gradient_evaluations_per_iter": self.metrics.gradient_evaluations / total_iterations,
                "operations_per_iter": self.metrics.total_operations / total_iterations,
                "matrix_operations_per_iter": self.metrics.matrix_vector_multiplications / total_iterations
            }
        
        # Efficiency metrics
        if converged and self.convergence_iteration:
            analysis["efficiency_metrics"] = {
                "operations_to_convergence": sum(m.total_operations for m in self.iteration_metrics[:self.convergence_iteration]),
                "convergence_efficiency": self.convergence_iteration / max(final_iteration, 1),
                "operations_per_convergence_iter": self.metrics.total_operations / self.convergence_iteration if self.convergence_iteration > 0 else 0
            }
        
        # Problem complexity scaling
        if self.problem_size:
            n, d = self.problem_size
            problem_complexity = n * d
            analysis["scalability_metrics"] = {
                "problem_size": {"n_samples": n, "n_features": d},
                "problem_complexity_factor": problem_complexity,
                "operations_per_problem_unit": self.metrics.total_operations / problem_complexity if problem_complexity > 0 else 0,
                "memory_efficiency": self.metrics.peak_memory_size / (n + d) if (n + d) > 0 else 0,
                "theoretical_complexity": self._estimate_theoretical_complexity(n, d, total_iterations)
            }
        
        # Operation distribution
        total_ops = self.metrics.total_operations
        if total_ops > 0:
            analysis["operation_distribution"] = {
                "function_eval_percentage": (self.metrics.function_evaluations * 100) / total_ops,
                "gradient_eval_percentage": (self.metrics.gradient_evaluations * 100) / total_ops, 
                "matrix_ops_percentage": (self.metrics.matrix_vector_multiplications * 100) / total_ops,
                "vector_ops_percentage": (self.metrics.vector_operations * 100) / total_ops
            }
        
        return analysis
    
    def _estimate_theoretical_complexity(self, n: int, d: int, iterations: int) -> str:
        """
        Estimate theoretical computational complexity
        
        Args:
            n: Number of samples
            d: Number of features
            iterations: Number of iterations
            
        Returns:
            String representation of complexity (e.g., "O(n*d*k)")
        """
        # For most first-order optimization methods
        return f"O({n}*{d}*{iterations})"
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for quick analysis"""
        return {
            "total_operations": self.metrics.total_operations,
            "function_evaluations": self.metrics.function_evaluations,
            "gradient_evaluations": self.metrics.gradient_evaluations,
            "iterations_tracked": len(self.iteration_metrics),
            "peak_memory": self.metrics.peak_memory_size,
            "converged_at": self.convergence_iteration
        }