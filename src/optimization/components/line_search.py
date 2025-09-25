"""
Line search strategies for optimization algorithms.
This module provides line search implementations following the Strategy pattern,
using the existing backtracking_line_search from utils.
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional
# Import existing line search implementation
# Add project root to path for imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
from utils.optimization_utils import backtracking_line_search

class LineSearchStrategy(ABC):
    """
    Abstract base class cho tất cả các chiến lược line search.
    
    Strategy pattern cho phép chuyển đổi dễ dàng giữa các phương pháp
    line search khác nhau.
    """
    
    @abstractmethod
    def search(self,
               loss_func: Callable,
               X: np.ndarray,
               y: np.ndarray,
               weights: np.ndarray,
               gradient: np.ndarray,
               direction: np.ndarray,
               initial_step_size: float = 1.0) -> float:
        """
        Thực hiện line search để tìm step size phù hợp.
        
        Args:
            loss_func: Hàm tính loss
            X: Ma trận đặc trưng
            y: Vector target
            weights: Vector trọng số hiện tại
            gradient: Gradient hiện tại
            direction: Hướng tìm kiếm
            initial_step_size: Step size ban đầu để thử
            
        Returns:
            Step size tối ưu
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Lấy tham số của line search strategy."""
        pass

class ArmijoLineSearch(LineSearchStrategy):
    """
    Armijo line search (backtracking line search với Armijo condition).
    
    Sử dụng implementation có sẵn từ utils.optimization_utils.
    """
    
    def __init__(self,
                 c1: float = 1e-4,
                 rho: float = 0.8,
                 max_backtracks: int = 50):
        """
        Khởi tạo Armijo line search.
        
        Args:
            c1: Armijo constant (0 < c1 < 1)
            rho: Backtracking reduction factor (0 < rho < 1)  
            max_backtracks: Số lần backtrack tối đa
        """
        if not (0 < c1 < 1):
            raise ValueError("c1 phải trong khoảng (0, 1)")
        if not (0 < rho < 1):
            raise ValueError("rho phải trong khoảng (0, 1)")
        if max_backtracks <= 0:
            raise ValueError("max_backtracks phải > 0")
        
        self.c1 = c1
        self.rho = rho  
        self.max_backtracks = max_backtracks
    
    def search(self,
               loss_func: Callable,
               X: np.ndarray,
               y: np.ndarray,
               weights: np.ndarray,
               gradient: np.ndarray,
               direction: np.ndarray,
               initial_step_size: float = 1.0) -> float:
        """
        Thực hiện Armijo line search.
        
        Adapt to existing backtracking_line_search signature.
        """
        # Create cost function that matches expected signature
        cost_func = lambda point: loss_func(X, y, point)
        current_cost = cost_func(weights)
        
        # Use existing backtracking line search implementation
        step_size = backtracking_line_search(
            cost_func=cost_func,
            gradient=gradient,
            weights=weights,
            direction=direction,
            alpha=initial_step_size,
            rho=self.rho,
            c1=self.c1
        )
        
        return step_size
    
    def get_parameters(self) -> Dict[str, Any]:
        """Lấy tham số Armijo line search."""
        return {
            'line_search': 'armijo',
            'c1': self.c1,
            'rho': self.rho,
            'max_backtracks': self.max_backtracks
        }

class WolfeLineSearch(LineSearchStrategy):
    """
    Wolfe line search với cả Armijo và curvature conditions.
    
    Implementation tùy chỉnh dựa trên Wolfe conditions.
    """
    
    def __init__(self,
                 c1: float = 1e-4,
                 c2: float = 0.9,
                 rho: float = 0.8,
                 max_backtracks: int = 50):
        """
        Khởi tạo Wolfe line search.
        
        Args:
            c1: Armijo constant (0 < c1 < c2 < 1)
            c2: Curvature constant (c1 < c2 < 1) 
            rho: Backtracking reduction factor (0 < rho < 1)
            max_backtracks: Số lần backtrack tối đa
        """
        if not (0 < c1 < c2 < 1):
            raise ValueError("Cần 0 < c1 < c2 < 1")
        if not (0 < rho < 1):
            raise ValueError("rho phải trong khoảng (0, 1)")
        if max_backtracks <= 0:
            raise ValueError("max_backtracks phải > 0")
        
        self.c1 = c1
        self.c2 = c2
        self.rho = rho
        self.max_backtracks = max_backtracks
    
    def search(self,
               loss_func: Callable,
               X: np.ndarray,
               y: np.ndarray,
               weights: np.ndarray,
               gradient: np.ndarray,
               direction: np.ndarray,
               initial_step_size: float = 1.0) -> float:
        """
        Thực hiện Wolfe line search với cả Armijo và curvature conditions.
        """
        from utils.optimization_utils import tinh_gradient_ham_loss
        
        alpha = initial_step_size
        current_loss = loss_func(X, y, weights)
        
        # Pre-compute directional derivative
        directional_derivative = np.dot(gradient, direction)
        
        # Armijo và Curvature conditions
        def armijo_condition(a):
            new_weights = weights + a * direction
            new_loss = loss_func(X, y, new_weights)
            return new_loss <= current_loss + self.c1 * a * directional_derivative
        
        def curvature_condition(a):
            new_weights = weights + a * direction
            # Get gradient at new point - assuming same signature as base loss function
            new_gradient, _ = tinh_gradient_ham_loss(
                loss_type=getattr(self, '_loss_type', 'ols'),  # Default fallback
                X=X, y=y, w=new_weights, b=None,
                regularization=getattr(self, '_regularization', 0.0)
            )
            new_directional_derivative = np.dot(new_gradient, direction)
            return abs(new_directional_derivative) <= self.c2 * abs(directional_derivative)
        
        # Backtracking loop with both conditions
        for _ in range(self.max_backtracks):
            if armijo_condition(alpha) and curvature_condition(alpha):
                break
            alpha *= self.rho
            
            # Prevent alpha from becoming too small
            if alpha < 1e-10:
                alpha = 1e-10
                break
        
        return alpha
    
    def get_parameters(self) -> Dict[str, Any]:
        """Lấy tham số Wolfe line search."""
        return {
            'line_search': 'wolfe',
            'c1': self.c1,
            'c2': self.c2,
            'rho': self.rho,
            'max_backtracks': self.max_backtracks
        }

class NoLineSearch(LineSearchStrategy):
    """
    Không sử dụng line search - trả về step size cố định.
    
    Hữu ích khi muốn disable line search và sử dụng step size strategies.
    """
    
    def __init__(self, fixed_step_size: float = 1.0):
        """
        Khởi tạo no line search.
        
        Args:
            fixed_step_size: Step size cố định sẽ trả về
        """
        if fixed_step_size <= 0:
            raise ValueError("fixed_step_size phải > 0")
        
        self.fixed_step_size = fixed_step_size
    
    def search(self,
               loss_func: Callable,
               X: np.ndarray,
               y: np.ndarray,
               weights: np.ndarray,
               gradient: np.ndarray,
               direction: np.ndarray,
               initial_step_size: float = 1.0) -> float:
        """Trả về step size cố định (không search)."""
        return self.fixed_step_size
    
    def get_parameters(self) -> Dict[str, Any]:
        """Lấy tham số no line search."""
        return {
            'line_search': 'none',
            'fixed_step_size': self.fixed_step_size
        }

def create_line_search_strategy(method: str, **params) -> LineSearchStrategy:
    """
    Factory function tạo line search strategy.
    
    Args:
        method: Tên phương pháp ('armijo', 'wolfe', 'none')
        **params: Các tham số cho strategy
        
    Returns:
        LineSearchStrategy instance
    """
    strategies = {
        'armijo': ArmijoLineSearch,
        'backtracking': ArmijoLineSearch,  # Alias
        'wolfe': WolfeLineSearch,
        'none': NoLineSearch,
        'fixed': NoLineSearch  # Alias
    }
    
    if method not in strategies:
        raise ValueError(f"Unknown line search method: {method}. "
                        f"Available: {list(strategies.keys())}")
    
    return strategies[method](**params)