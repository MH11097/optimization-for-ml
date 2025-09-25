"""
Quasi-Newton Optimizer
This module implements the Quasi-Newton family of optimization algorithms
including BFGS, L-BFGS, and SR1 methods with unified interface.
"""
import numpy as np
from typing import Dict, Any, Optional, Literal, List
import sys
from pathlib import Path
from collections import deque
# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from optimization.base import IterativeOptimizer
from utils.optimization_utils import (
    tinh_gia_tri_ham_loss,
    tinh_gradient_ham_loss
)

class QuasiNewtonOptimizer(IterativeOptimizer):
    """
    Quasi-Newton Optimizer with unified interface.
    
    Supports:
    - BFGS (Broyden-Fletcher-Goldfarb-Shanno)
    - L-BFGS (Limited-memory BFGS)
    - SR1 (Symmetric Rank-1)
    - Line search methods for step size control
    
    Architecture: Follows OOP and DRY principles with strategy pattern for different methods.
    """
    
    def __init__(self,
                 # Core parameters
                 loss_type: str = 'ols',
                 regularization: float = 0.01,
                 convergence_tolerance: float = 1e-3,
                 max_iterations: int = 10000,
                 convergence_check_freq: int = 1,
                 random_state: Optional[int] = None,
                 
                 # Quasi-Newton specific parameters
                 method: Literal['bfgs', 'lbfgs', 'sr1'] = 'bfgs',
                 memory_size: int = 10,  # For L-BFGS
                 line_search_method: str = 'backtracking',  # 'backtracking', 'wolfe'
                 
                 # Line search parameters
                 backtrack_c1: float = 1e-4,  # Armijo condition
                 backtrack_c2: float = 0.9,   # Curvature condition (Wolfe)
                 backtrack_rho: float = 0.8,   # Reduction factor
                 max_line_search_steps: int = 20,
                 
                 # Numerical stability
                 hessian_init_scale: float = 1.0,
                 sr1_skip_threshold: float = 1e-8,  # Skip SR1 update if denominator too small
                 
                 # Legacy compatibility
                 ham_loss: Optional[str] = None,
                 diem_dung: Optional[float] = None,
                 learning_rate: Optional[float] = None):
        """
        Initialize Quasi-Newton optimizer.
        
        Args:
            loss_type: Type of loss function ('ols', 'ridge', 'lasso')
            regularization: Regularization parameter
            convergence_tolerance: Convergence threshold
            max_iterations: Maximum number of iterations
            convergence_check_freq: Frequency of convergence checking
            random_state: Random seed
            method: Quasi-Newton method ('bfgs', 'lbfgs', 'sr1')
            memory_size: Memory size for L-BFGS
            line_search_method: Line search method
            backtrack_c1: Armijo condition parameter
            backtrack_c2: Curvature condition parameter
            backtrack_rho: Backtracking reduction factor
            max_line_search_steps: Maximum line search steps
            hessian_init_scale: Initial Hessian approximation scale
            sr1_skip_threshold: Threshold for skipping SR1 updates
        """
        super().__init__(
            loss_type=loss_type,
            regularization=regularization,
            convergence_tolerance=convergence_tolerance,
            max_iterations=max_iterations,
            convergence_check_freq=convergence_check_freq,
            random_state=random_state,
            ham_loss=ham_loss,
            diem_dung=diem_dung
        )
        
        # Quasi-Newton specific parameters
        self.method = method.lower()
        self.memory_size = memory_size
        self.line_search_method = line_search_method
        self.backtrack_c1 = backtrack_c1
        self.backtrack_c2 = backtrack_c2
        self.backtrack_rho = backtrack_rho
        self.max_line_search_steps = max_line_search_steps
        self.hessian_init_scale = hessian_init_scale
        self.sr1_skip_threshold = sr1_skip_threshold
        
        # Internal state
        self.hessian_approx: Optional[np.ndarray] = None  # For BFGS, SR1
        self.s_vectors: Optional[deque] = None  # For L-BFGS: s_k = x_{k+1} - x_k
        self.y_vectors: Optional[deque] = None  # For L-BFGS: y_k = ∇f_{k+1} - ∇f_k
        self.rho_values: Optional[deque] = None  # For L-BFGS: ρ_k = 1/(y_k^T s_k)
        
        self.previous_gradient: Optional[np.ndarray] = None
        self.previous_weights: Optional[np.ndarray] = None
        
        # History tracking
        self.hessian_updates: list = []
        self.line_search_steps_history: list = []
        self.curvature_conditions: list = []
        
    def _initialize_algorithm_specific_params(self, X: np.ndarray, y: np.ndarray) -> None:
        """Initialize Quasi-Newton specific parameters."""
        n_features = len(self.weights)
        
        print(f"   Method: {self.method.upper()}")
        print(f"   Line search: {self.line_search_method}")
        if self.method == 'lbfgs':
            print(f"   Memory size: {self.memory_size}")
        
        # Initialize method-specific data structures
        if self.method == 'bfgs' or self.method == 'sr1':
            # Full Hessian approximation
            self.hessian_approx = self.hessian_init_scale * np.eye(n_features)
        elif self.method == 'lbfgs':
            # Limited memory storage
            self.s_vectors = deque(maxlen=self.memory_size)
            self.y_vectors = deque(maxlen=self.memory_size)
            self.rho_values = deque(maxlen=self.memory_size)
        
        # Initialize previous values (will be set in first iteration)
        self.previous_gradient = None
        self.previous_weights = None
        
    def _compute_update_direction(self, X: np.ndarray, y: np.ndarray, iteration: int) -> np.ndarray:
        """
        Compute quasi-Newton update direction.
        
        Uses the quasi-Newton direction: d = -B^(-1) * g
        where B is the Hessian approximation and g is the gradient.
        """
        # Compute current gradient
        current_gradient, _ = self.grad_func(X, y, self.weights)
        self.track_gradient_evaluation(X.shape)
        
        # Update Hessian approximation (except for first iteration)
        if iteration > 0 and self.previous_gradient is not None:
            self._update_hessian_approximation(current_gradient)
        
        # Compute search direction
        if self.method == 'bfgs' or self.method == 'sr1':
            # Full matrix case: solve B * d = -g
            try:
                direction = -np.linalg.solve(self.hessian_approx, current_gradient)
                self.track_linear_solve(len(current_gradient))
            except np.linalg.LinAlgError:
                print("   Warning: Singular Hessian approximation, using gradient direction")
                direction = -current_gradient
        elif self.method == 'lbfgs':
            # L-BFGS two-loop recursion
            direction = -self._lbfgs_multiply(current_gradient)
        
        # Store current values for next iteration
        self.previous_gradient = current_gradient.copy()
        self.previous_weights = self.weights.copy()
        
        return direction
    
    def _update_hessian_approximation(self, current_gradient: np.ndarray) -> None:
        """
        Update Hessian approximation using quasi-Newton updates.
        
        Implements BFGS, L-BFGS, and SR1 update rules.
        """
        # Compute differences
        s_k = self.weights - self.previous_weights  # Step
        y_k = current_gradient - self.previous_gradient  # Gradient difference
        
        # Check curvature condition: s^T * y > 0
        sy_dot = np.dot(s_k, y_k)
        curvature_positive = sy_dot > 1e-10
        self.curvature_conditions.append(curvature_positive)
        
        if self.method == 'bfgs':
            if curvature_positive:
                # BFGS update: B_{k+1} = B_k - (B_k*s_k*s_k^T*B_k)/(s_k^T*B_k*s_k) + (y_k*y_k^T)/(y_k^T*s_k)
                Bs = self.hessian_approx @ s_k
                sBs = np.dot(s_k, Bs)
                
                if sBs > 1e-10:
                    # Rank-2 update
                    self.hessian_approx -= np.outer(Bs, Bs) / sBs
                    self.hessian_approx += np.outer(y_k, y_k) / sy_dot
                    self.hessian_updates.append('bfgs_full')
                else:
                    self.hessian_updates.append('bfgs_skipped')
            else:
                # Skip update due to negative curvature
                self.hessian_updates.append('bfgs_skipped_curvature')
                
        elif self.method == 'sr1':
            # SR1 update: B_{k+1} = B_k + (y_k - B_k*s_k)*(y_k - B_k*s_k)^T / ((y_k - B_k*s_k)^T * s_k)
            Bs = self.hessian_approx @ s_k
            u = y_k - Bs
            us_dot = np.dot(u, s_k)
            
            if abs(us_dot) > self.sr1_skip_threshold * np.linalg.norm(s_k) * np.linalg.norm(u):
                # Safe to update
                self.hessian_approx += np.outer(u, u) / us_dot
                self.hessian_updates.append('sr1_full')
            else:
                # Skip update due to potential numerical issues
                self.hessian_updates.append('sr1_skipped')
                
        elif self.method == 'lbfgs':
            if curvature_positive:
                # Store vectors for L-BFGS
                self.s_vectors.append(s_k.copy())
                self.y_vectors.append(y_k.copy())
                self.rho_values.append(1.0 / sy_dot)
                self.hessian_updates.append('lbfgs_stored')
            else:
                self.hessian_updates.append('lbfgs_skipped_curvature')
    
    def _lbfgs_multiply(self, gradient: np.ndarray) -> np.ndarray:
        """
        L-BFGS two-loop recursion for computing B^(-1) * gradient.
        
        Implements the efficient L-BFGS multiplication algorithm.
        """
        if not self.s_vectors:
            # No stored vectors yet, use identity
            return gradient
        
        q = gradient.copy()
        alphas = []
        
        # First loop (backwards)
        for s, y, rho in zip(reversed(self.s_vectors), 
                           reversed(self.y_vectors), 
                           reversed(self.rho_values)):
            alpha = rho * np.dot(s, q)
            alphas.append(alpha)
            q -= alpha * y
        
        # Apply initial Hessian approximation (scaled identity)
        if len(self.s_vectors) > 0:
            # Use Nocedal & Wright scaling: γ_k = (s_{k-1}^T y_{k-1}) / (y_{k-1}^T y_{k-1})
            s_last = self.s_vectors[-1]
            y_last = self.y_vectors[-1]
            gamma = np.dot(s_last, y_last) / np.dot(y_last, y_last)
            r = gamma * q
        else:
            r = q
        
        # Second loop (forwards)
        alphas.reverse()
        for s, y, rho, alpha in zip(self.s_vectors, self.y_vectors, 
                                  self.rho_values, alphas):
            beta = rho * np.dot(y, r)
            r += s * (alpha - beta)
        
        return r
    
    def _compute_step_size(self, X: np.ndarray, y: np.ndarray, 
                          direction: np.ndarray, iteration: int) -> float:
        """Compute step size using line search."""
        if self.line_search_method == 'backtracking':
            step_size, steps = self._backtracking_line_search(X, y, direction)
        elif self.line_search_method == 'wolfe':
            step_size, steps = self._wolfe_line_search(X, y, direction)
        else:
            step_size, steps = 1.0, 0
        
        self.line_search_steps_history.append(steps)
        return step_size
    
    def _backtracking_line_search(self, X: np.ndarray, y: np.ndarray, direction: np.ndarray) -> tuple:
        """Backtracking line search with Armijo condition."""
        current_loss = self.loss_func(X, y, self.weights)
        gradient, _ = self.grad_func(X, y, self.weights)
        directional_derivative = np.dot(gradient, direction)
        
        step_size = 1.0
        steps = 0
        
        for _ in range(self.max_line_search_steps):
            test_weights = self.weights + step_size * direction
            test_loss = self.loss_func(X, y, test_weights)
            
            # Armijo condition
            if test_loss <= current_loss + self.backtrack_c1 * step_size * directional_derivative:
                break
                
            step_size *= self.backtrack_rho
            steps += 1
        
        return step_size, steps
    
    def _wolfe_line_search(self, X: np.ndarray, y: np.ndarray, direction: np.ndarray) -> tuple:
        """Wolfe line search with strong Wolfe conditions."""
        # Simplified Wolfe line search (could be enhanced with zoom phase)
        current_loss = self.loss_func(X, y, self.weights)
        gradient, _ = self.grad_func(X, y, self.weights)
        directional_derivative = np.dot(gradient, direction)
        
        step_size = 1.0
        steps = 0
        
        for _ in range(self.max_line_search_steps):
            test_weights = self.weights + step_size * direction
            test_loss = self.loss_func(X, y, test_weights)
            test_gradient, _ = self.grad_func(X, y, test_weights)
            test_directional_derivative = np.dot(test_gradient, direction)
            
            # Armijo condition
            armijo = test_loss <= current_loss + self.backtrack_c1 * step_size * directional_derivative
            
            # Curvature condition
            curvature = abs(test_directional_derivative) <= self.backtrack_c2 * abs(directional_derivative)
            
            if armijo and curvature:
                break
            
            if not armijo:
                step_size *= self.backtrack_rho
            else:
                # Armijo satisfied but curvature not, could zoom here
                break
                
            steps += 1
        
        return step_size, steps
    
    def track_linear_solve(self, n: int) -> None:
        """Track linear system solve complexity."""
        self.track_matrix_operation("linear_solve")
    
    def _get_algorithm_specific_results(self) -> Dict[str, Any]:
        """Get Quasi-Newton specific results."""
        results = super()._get_algorithm_specific_results()
        
        results.update({
            'algorithm_type': f'QuasiNewton_{self.method.upper()}',
            'method': self.method,
            'memory_size': self.memory_size if self.method == 'lbfgs' else None,
            'line_search_method': self.line_search_method,
            'hessian_updates': self.hessian_updates.copy(),
            'line_search_steps_history': self.line_search_steps_history.copy(),
            'curvature_conditions': self.curvature_conditions.copy(),
            'avg_line_search_steps': np.mean(self.line_search_steps_history) if self.line_search_steps_history else 0,
            'positive_curvature_rate': np.mean(self.curvature_conditions) if self.curvature_conditions else 0,
            'update_success_rate': len([u for u in self.hessian_updates if 'skipped' not in u]) / len(self.hessian_updates) if self.hessian_updates else 0
        })
        
        return results
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about available strategies."""
        return {
            'methods': ['bfgs', 'lbfgs', 'sr1'],
            'line_search_methods': ['backtracking', 'wolfe'],
            'current_config': {
                'method': self.method,
                'memory_size': self.memory_size,
                'line_search_method': self.line_search_method
            }
        }