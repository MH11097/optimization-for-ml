"""
Newton Method Optimizer
This module implements the Newton Method family of second-order optimization algorithms
with various damping strategies and line search methods.
"""
import numpy as np
from typing import Dict, Any, Optional, Literal
import sys
from pathlib import Path
# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from optimization.base import IterativeOptimizer
from utils.optimization_utils import (
    tinh_gia_tri_ham_loss,
    tinh_gradient_ham_loss,
    tinh_hessian_ham_loss
)

class NewtonOptimizer(IterativeOptimizer):
    """
    Newton Method Optimizer with unified interface.
    
    Supports:
    - Pure Newton method
    - Damped Newton with various damping strategies
    - Line search methods (backtracking, Armijo)
    - Regularization for ill-conditioned Hessians
    
    Architecture: Follows OOP and DRY principles with template method pattern.
    """
    
    def __init__(self,
                 # Core parameters
                 loss_type: str = 'ols',
                 regularization: float = 0.01,
                 convergence_tolerance: float = 1e-3,
                 max_iterations: int = 100000,  # Newton typically needs fewer iterations
                 convergence_check_freq: int = 1,
                 random_state: Optional[int] = None,
                 
                 # Newton-specific parameters
                 damping_strategy: str = 'none',  # 'none', 'constant', 'adaptive'
                 damping_factor: float = 0.5,
                 line_search_method: str = 'none',  # 'none', 'backtracking', 'armijo'
                 hessian_regularization: float = 1e-6,  # For numerical stability
                 
                 # Line search parameters
                 backtrack_c1: float = 1e-4,  # Armijo condition parameter
                 backtrack_rho: float = 0.8,   # Backtracking reduction factor
                 max_line_search_steps: int = 20,
                 
                 # Legacy compatibility
                 ham_loss: Optional[str] = None,
                 diem_dung: Optional[float] = None,
                 learning_rate: Optional[float] = None):  # Legacy compatibility
        """
        Initialize Newton Method optimizer.
        
        Args:
            loss_type: Type of loss function ('ols', 'ridge', 'lasso')
            regularization: Regularization parameter
            convergence_tolerance: Convergence threshold
            max_iterations: Maximum number of iterations
            convergence_check_freq: Frequency of convergence checking
            random_state: Random seed
            damping_strategy: Damping strategy for step size control
            damping_factor: Damping factor (0 < factor < 1)
            line_search_method: Line search method
            hessian_regularization: Regularization for Hessian inversion
            backtrack_c1: Armijo condition parameter
            backtrack_rho: Backtracking reduction factor
            max_line_search_steps: Maximum line search steps
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
        
        # Newton-specific parameters
        self.damping_strategy = damping_strategy
        self.damping_factor = damping_factor
        self.line_search_method = line_search_method
        self.hessian_regularization = hessian_regularization
        self.backtrack_c1 = backtrack_c1
        self.backtrack_rho = backtrack_rho
        self.max_line_search_steps = max_line_search_steps
        
        # Internal state
        self.current_damping: float = 1.0
        self.hessian_condition_numbers: list = []
        self.damping_history: list = []
        self.line_search_steps_history: list = []
        
    def _initialize_algorithm_specific_params(self, X: np.ndarray, y: np.ndarray) -> None:
        """Initialize Newton-specific parameters."""
        print(f"   Damping strategy: {self.damping_strategy}")
        print(f"   Line search: {self.line_search_method}")
        print(f"   Hessian regularization: {self.hessian_regularization}")
        
        # Initialize damping
        self.current_damping = 1.0 if self.damping_strategy == 'none' else self.damping_factor
        
    def _compute_update_direction(self, X: np.ndarray, y: np.ndarray, iteration: int) -> np.ndarray:
        """
        Compute Newton update direction.
        
        Uses the Newton direction: d = -H^(-1) * g
        where H is the Hessian and g is the gradient.
        """
        # Compute gradient
        gradient, _ = self.grad_func(X, y, self.weights)
        self.track_gradient_evaluation(X.shape)
        
        # Compute Hessian
        hessian = self._compute_hessian(X, y)
        self.track_hessian_evaluation(X.shape)
        
        # Regularize Hessian for numerical stability
        regularized_hessian = hessian + self.hessian_regularization * np.eye(len(hessian))
        
        # Check condition number
        try:
            condition_number = np.linalg.cond(regularized_hessian)
            self.hessian_condition_numbers.append(condition_number)
            
            if condition_number > 1e12:
                print(f"   Warning: Ill-conditioned Hessian (cond={condition_number:.2e})")
        except:
            condition_number = np.inf
            self.hessian_condition_numbers.append(condition_number)
        
        # Solve for Newton direction: H * d = -g
        try:
            newton_direction = np.linalg.solve(regularized_hessian, -gradient)
            self.track_linear_solve(len(hessian))
        except np.linalg.LinAlgError:
            print("   Warning: Hessian is singular, using gradient direction")
            newton_direction = -gradient
        
        return newton_direction
    
    def _compute_step_size(self, X: np.ndarray, y: np.ndarray, 
                          direction: np.ndarray, iteration: int) -> float:
        """
        Compute step size using damping and line search.
        
        Combines damping strategies with line search methods.
        """
        # Step 1: Apply damping strategy
        if self.damping_strategy == 'none':
            base_step_size = 1.0
        elif self.damping_strategy == 'constant':
            base_step_size = self.damping_factor
        elif self.damping_strategy == 'adaptive':
            # Adaptive damping based on function decrease
            base_step_size = self._compute_adaptive_damping(X, y, direction)
        else:
            raise ValueError(f"Unknown damping strategy: {self.damping_strategy}")
        
        # Step 2: Apply line search if specified
        if self.line_search_method == 'none':
            final_step_size = base_step_size
            line_search_steps = 0
        elif self.line_search_method in ['backtracking', 'armijo']:
            final_step_size, line_search_steps = self._line_search(X, y, direction, base_step_size)
        else:
            raise ValueError(f"Unknown line search method: {self.line_search_method}")
        
        # Track history
        self.damping_history.append(base_step_size)
        self.line_search_steps_history.append(line_search_steps)
        self.current_damping = final_step_size
        
        return final_step_size
    
    def _compute_hessian(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute Hessian matrix."""
        return tinh_hessian_ham_loss(X, y, self.weights, self.loss_type, self.regularization)
    
    def _compute_adaptive_damping(self, X: np.ndarray, y: np.ndarray, direction: np.ndarray) -> float:
        """
        Compute adaptive damping factor.
        
        Uses the ratio of predicted vs actual function decrease.
        """
        current_loss = self.loss_func(X, y, self.weights)
        gradient, _ = self.grad_func(X, y, self.weights)
        
        # Predicted quadratic decrease: Δq = g^T * d + 0.5 * d^T * H * d  
        hessian = self._compute_hessian(X, y)
        predicted_decrease = np.dot(gradient, direction) + 0.5 * np.dot(direction, hessian @ direction)
        
        # Try a full step to see actual decrease
        test_weights = self.weights + direction
        test_loss = self.loss_func(X, y, test_weights)
        actual_decrease = current_loss - test_loss
        
        # Compute ratio and adjust damping
        if abs(predicted_decrease) > 1e-12:
            ratio = actual_decrease / predicted_decrease
            if ratio < 0.25:
                # Poor agreement, reduce step
                damping = max(0.1, self.current_damping * 0.5)
            elif ratio > 0.75:
                # Good agreement, possibly increase step
                damping = min(1.0, self.current_damping * 1.2)
            else:
                # Reasonable agreement, keep current
                damping = self.current_damping
        else:
            damping = self.damping_factor
        
        return damping
    
    def _line_search(self, X: np.ndarray, y: np.ndarray, direction: np.ndarray, 
                    initial_step: float) -> tuple:
        """
        Perform line search to find acceptable step size.
        
        Implements backtracking line search with Armijo condition.
        """
        current_loss = self.loss_func(X, y, self.weights)
        gradient, _ = self.grad_func(X, y, self.weights)
        directional_derivative = np.dot(gradient, direction)
        
        step_size = initial_step
        steps = 0
        
        for _ in range(self.max_line_search_steps):
            # Test new point
            test_weights = self.weights + step_size * direction
            test_loss = self.loss_func(X, y, test_weights)
            
            # Check Armijo condition: f(x + α*d) ≤ f(x) + c1*α*∇f^T*d
            armijo_condition = current_loss + self.backtrack_c1 * step_size * directional_derivative
            
            if test_loss <= armijo_condition:
                break
                
            # Reduce step size
            step_size *= self.backtrack_rho
            steps += 1
        
        return step_size, steps
    
    def track_hessian_evaluation(self, shape: tuple) -> None:
        """Track Hessian computation complexity."""
        self.track_matrix_operation("hessian_computation")
        
    def track_linear_solve(self, n: int) -> None:
        """Track linear system solve complexity."""
        self.track_matrix_operation("linear_solve")
    
    def _get_algorithm_specific_results(self) -> Dict[str, Any]:
        """Get Newton-specific results."""
        results = super()._get_algorithm_specific_results()
        
        results.update({
            'algorithm_type': 'Newton',
            'damping_strategy': self.damping_strategy,
            'line_search_method': self.line_search_method,
            'final_damping': self.current_damping,
            'hessian_condition_numbers': self.hessian_condition_numbers.copy(),
            'damping_history': self.damping_history.copy(),
            'line_search_steps_history': self.line_search_steps_history.copy(),
            'avg_condition_number': np.mean(self.hessian_condition_numbers) if self.hessian_condition_numbers else 0,
            'avg_line_search_steps': np.mean(self.line_search_steps_history) if self.line_search_steps_history else 0,
        })
        
        return results
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about available strategies."""
        return {
            'damping_strategies': ['none', 'constant', 'adaptive'],
            'line_search_methods': ['none', 'backtracking', 'armijo'],
            'current_config': {
                'damping_strategy': self.damping_strategy,
                'line_search_method': self.line_search_method,
                'hessian_regularization': self.hessian_regularization
            }
        }