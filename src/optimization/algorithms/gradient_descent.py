"""
Gradient Descent optimizer implementation.
This module provides a refactored Gradient Descent algorithm that leverages
the component-based architecture for maximum flexibility and reusability.
"""
import numpy as np
from typing import Dict, Any, Optional, Union
from ..base import IterativeOptimizer
from ..components import (
    StepSizeStrategy, create_step_size_strategy,
    LineSearchStrategy, create_line_search_strategy,
    MomentumStrategy, create_momentum_strategy
)

class GradientDescentOptimizer(IterativeOptimizer):
    """
    Gradient Descent optimizer sử dụng component-based architecture.
    
    Hỗ trợ:
    - Nhiều step size strategies (constant, decay, adaptive, etc.)
    - Line search methods (Armijo, Wolfe)
    - Momentum (standard, Nesterov)
    - Unified interface với existing algorithms
    
    Parameters:
        ham_loss: Loại hàm loss ('ols', 'ridge', 'lasso')
        learning_rate: Learning rate ban đầu  
        step_size_method: Phương pháp step size ('constant', 'linear_decay', etc.)
        line_search_method: Phương pháp line search ('none', 'armijo', 'wolfe')
        momentum_method: Phương pháp momentum ('none', 'standard', 'nesterov')
        momentum_coefficient: Hệ số momentum (nếu dùng momentum)
    """
    
    def __init__(self,
                 ham_loss: str = 'ols',
                 learning_rate: float = 0.01,
                 regularization: float = 0.01,
                 diem_dung: float = 1e-3,
                 max_iterations: int = 100000,
                 convergence_check_freq: int = 1,
                 random_state: Optional[int] = None,
                 # Step size parameters
                 step_size_method: str = 'constant',
                 decay_rate: float = 0.95,
                 decay_steps: int = 100,
                 min_learning_rate: float = 1e-6,
                 # Adaptive parameters  
                 adaptive_beta1: float = 0.9,
                 adaptive_beta2: float = 0.999,
                 adaptive_eps: float = 1e-8,
                 # Line search parameters
                 line_search_method: str = 'none',
                 backtrack_c1: float = 1e-4,
                 backtrack_rho: float = 0.8,
                 wolfe_c2: float = 0.9,
                 max_line_search_iter: int = 50,
                 # Momentum parameters
                 momentum_method: str = 'none',
                 momentum_coefficient: float = 0.9):
        """
        Khởi tạo GradientDescentOptimizer với component strategies.
        """
        super().__init__(
            ham_loss=ham_loss,
            regularization=regularization,
            diem_dung=diem_dung,
            max_iterations=max_iterations,
            convergence_check_freq=convergence_check_freq,
            random_state=random_state
        )
        
        # Store parameters for strategy creation
        self.learning_rate = learning_rate
        self.step_size_method = step_size_method
        self.line_search_method = line_search_method
        self.momentum_method = momentum_method
        self.momentum_coefficient = momentum_coefficient
        
        # Create step size strategy with appropriate parameters for each method
        if step_size_method == 'constant':
            step_size_params = {'learning_rate': learning_rate}
        elif step_size_method in ['linear_decay', 'exponential_decay', 'sqrt_decay']:
            step_size_params = {
                'initial_learning_rate': learning_rate,
                'max_iterations': max_iterations,
                'min_learning_rate': min_learning_rate,
                'decay_rate': decay_rate,
                'decay_steps': decay_steps
            }
        elif step_size_method == 'adaptive':
            step_size_params = {
                'initial_learning_rate': learning_rate,
                'beta1': adaptive_beta1,
                'beta2': adaptive_beta2,
                'epsilon': adaptive_eps
            }
        else:
            step_size_params = {'learning_rate': learning_rate}
            
        self.step_size_strategy = create_step_size_strategy(step_size_method, **step_size_params)
        
        # Create line search strategy with appropriate parameters
        if line_search_method in ['armijo', 'backtracking']:
            line_search_params = {
                'c1': backtrack_c1,
                'rho': backtrack_rho,
                'max_backtracks': max_line_search_iter
            }
        elif line_search_method == 'wolfe':
            line_search_params = {
                'c1': backtrack_c1,
                'c2': wolfe_c2,
                'rho': backtrack_rho,
                'max_backtracks': max_line_search_iter
            }
        elif line_search_method in ['none', 'fixed']:
            line_search_params = {'fixed_step_size': 1.0}
        else:
            line_search_params = {}
            
        self.line_search_strategy = create_line_search_strategy(line_search_method, **line_search_params)
        
        # Create momentum strategy with appropriate parameters
        if momentum_method in ['standard', 'classical', 'nesterov', 'nag']:
            momentum_params = {'momentum_coefficient': momentum_coefficient}
        else:  # 'none'
            momentum_params = {}
            
        self.momentum_strategy = create_momentum_strategy(momentum_method, **momentum_params)
        
        # For Wolfe line search - store context
        if hasattr(self.line_search_strategy, '_loss_type'):
            self.line_search_strategy._loss_type = self.ham_loss
            self.line_search_strategy._regularization = self.regularization
    
    def _initialize_algorithm_specific_params(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Khởi tạo các strategies về trạng thái ban đầu.
        """
        self.step_size_strategy.reset()
        self.momentum_strategy.reset()
        
        # No additional initialization needed for basic GD
        pass
    
    def _compute_update_direction(self, 
                                X: np.ndarray, 
                                y: np.ndarray, 
                                iteration: int) -> np.ndarray:
        """
        Tính hướng update sử dụng momentum strategy.
        
        For gradient descent, direction is based on gradient and momentum.
        """
        # Compute gradient
        gradient_w, _ = self.grad_func(X, y, self.weights)
        
        # Track gradient computation
        self.track_gradient_evaluation(X.shape)
        
        # Handle Nesterov momentum special case
        if hasattr(self.momentum_strategy, 'get_lookahead_weights'):
            # For Nesterov, we need to compute gradient at lookahead position
            # Get current step size for lookahead
            current_step_size = self.step_size_strategy.compute_step_size(
                iteration, gradient_w, X, y, self.weights
            )
            
            # Compute lookahead weights
            lookahead_weights = self.momentum_strategy.get_lookahead_weights(
                self.weights, current_step_size
            )
            
            # Compute gradient at lookahead position
            lookahead_gradient, _ = self.grad_func(X, y, lookahead_weights)
            self.track_gradient_evaluation(X.shape)
            
            # Use lookahead gradient for momentum update
            direction = self.momentum_strategy.update(lookahead_gradient, iteration)
        else:
            # Standard momentum or no momentum
            direction = self.momentum_strategy.update(gradient_w, iteration)
        
        return direction
    
    def _compute_step_size(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          direction: np.ndarray,
                          iteration: int) -> Union[float, np.ndarray]:
        """
        Tính step size kết hợp step size strategy và line search.
        """
        # Get base step size from strategy
        gradient_w, _ = self.grad_func(X, y, self.weights)
        base_step_size = self.step_size_strategy.compute_step_size(
            iteration, gradient_w, X, y, self.weights
        )
        
        # Apply line search if enabled
        if self.line_search_method != 'none':
            if isinstance(base_step_size, np.ndarray):
                # For adaptive methods, use mean as initial step size for line search
                initial_step_size = float(np.mean(base_step_size))
            else:
                initial_step_size = base_step_size
            
            # Perform line search
            line_search_step_size = self.line_search_strategy.search(
                loss_func=self.loss_func,
                X=X, y=y,
                weights=self.weights,
                gradient=gradient_w,
                direction=direction,
                initial_step_size=initial_step_size
            )
            
            # For adaptive methods, scale the adaptive rates by line search result
            if isinstance(base_step_size, np.ndarray):
                scaling_factor = line_search_step_size / initial_step_size
                return base_step_size * scaling_factor
            else:
                return line_search_step_size
        
        return base_step_size
    
    def _get_algorithm_specific_results(self) -> Dict[str, Any]:
        """
        Lấy kết quả đặc thù cho Gradient Descent.
        """
        # Get base results from iterative optimizer
        results = super()._get_algorithm_specific_results()
        
        # Add GD-specific information
        results.update({
            'algorithm_type': 'gradient_descent',
            'step_size_strategy': self.step_size_strategy.get_parameters(),
            'line_search_strategy': self.line_search_strategy.get_parameters(),
            'momentum_strategy': self.momentum_strategy.get_parameters(),
            'base_learning_rate': self.learning_rate,
            'uses_line_search': self.line_search_method != 'none',
            'uses_momentum': self.momentum_method != 'none',
        })
        
        # Add momentum history if available
        if hasattr(self.momentum_strategy, 'velocity') and self.momentum_strategy.velocity is not None:
            results['final_velocity_norm'] = float(np.linalg.norm(self.momentum_strategy.velocity))
        
        return results
    
    def get_strategy_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Lấy thông tin chi tiết về các strategies đang sử dụng.
        
        Returns:
            Dictionary chứa thông tin về step size, line search, và momentum strategies
        """
        return {
            'step_size': self.step_size_strategy.get_parameters(),
            'line_search': self.line_search_strategy.get_parameters(), 
            'momentum': self.momentum_strategy.get_parameters()
        }
    
    def update_strategies(self, **strategy_params) -> None:
        """
        Cập nhật parameters của strategies (để fine-tuning trong quá trình training).
        
        Args:
            **strategy_params: Parameters mới cho các strategies
        """
        # Note: This would require implementing update methods in strategies
        # For now, strategies are immutable after creation
        raise NotImplementedError(
            "Strategy updates during training chưa được implement. "
            "Tạo optimizer mới với parameters mới."
        )

def create_gradient_descent_optimizer(config: Dict[str, Any]) -> GradientDescentOptimizer:
    """
    Factory function tạo GradientDescentOptimizer từ config dictionary.
    
    Args:
        config: Dictionary chứa tất cả parameters
        
    Returns:
        Configured GradientDescentOptimizer instance
    """
    return GradientDescentOptimizer(**config)