"""
Momentum strategies for optimization algorithms.
This module provides momentum implementations following the Strategy pattern.
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class MomentumStrategy(ABC):
    """
    Abstract base class cho tất cả các chiến lược momentum.
    """
    
    @abstractmethod
    def update(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        """
        Cập nhật momentum và trả về direction để update weights.
        
        Args:
            gradient: Gradient hiện tại
            iteration: Số iteration hiện tại
            
        Returns:
            Direction để update weights
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset momentum về trạng thái ban đầu."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Lấy tham số của momentum strategy."""
        pass

class NoMomentum(MomentumStrategy):
    """
    Không sử dụng momentum - trả về gradient gốc.
    """
    
    def update(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        """Trả về negative gradient (steepest descent direction)."""
        return -gradient
    
    def reset(self) -> None:
        """No momentum không cần reset."""
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """Lấy tham số no momentum."""
        return {
            'momentum': 'none',
            'momentum_coefficient': 0.0
        }

class StandardMomentum(MomentumStrategy):
    """
    Standard momentum (classical momentum).
    
    v_t = β * v_{t-1} + ∇L_t
    θ_{t+1} = θ_t - α * v_t
    """
    
    def __init__(self, momentum_coefficient: float = 0.9):
        """
        Khởi tạo standard momentum.
        
        Args:
            momentum_coefficient: Hệ số momentum β (0 <= β < 1)
        """
        if not (0 <= momentum_coefficient < 1):
            raise ValueError("momentum_coefficient phải trong khoảng [0, 1)")
        
        self.momentum_coefficient = momentum_coefficient
        self.velocity: Optional[np.ndarray] = None
    
    def update(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        """
        Cập nhật velocity với standard momentum.
        
        Args:
            gradient: Gradient hiện tại
            iteration: Số iteration (không sử dụng cho standard momentum)
            
        Returns:
            Direction để update weights (negative velocity)
        """
        # Initialize velocity on first iteration
        if self.velocity is None:
            self.velocity = np.zeros_like(gradient)
        
        # Update velocity: v = β * v + ∇L
        self.velocity = self.momentum_coefficient * self.velocity + gradient
        
        # Return negative velocity (descent direction)
        return -self.velocity
    
    def reset(self) -> None:
        """Reset velocity về zero."""
        self.velocity = None
    
    def get_parameters(self) -> Dict[str, Any]:
        """Lấy tham số standard momentum."""
        return {
            'momentum': 'standard',
            'momentum_coefficient': self.momentum_coefficient
        }

class NesterovMomentum(MomentumStrategy):
    """
    Nesterov accelerated gradient (NAG).
    
    v_t = β * v_{t-1} + ∇L(θ_t - α * β * v_{t-1})
    θ_{t+1} = θ_t - α * v_t
    
    Note: Requires access to loss function for lookahead gradient computation.
    """
    
    def __init__(self, momentum_coefficient: float = 0.9):
        """
        Khởi tạo Nesterov momentum.
        
        Args:
            momentum_coefficient: Hệ số momentum β (0 <= β < 1)
        """
        if not (0 <= momentum_coefficient < 1):
            raise ValueError("momentum_coefficient phải trong khoảng [0, 1)")
        
        self.momentum_coefficient = momentum_coefficient
        self.velocity: Optional[np.ndarray] = None
    
    def update(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        """
        Cập nhật velocity với Nesterov momentum.
        
        Note: Gradient được giả định là đã được tính tại lookahead position
        bởi calling algorithm.
        
        Args:
            gradient: Gradient tại lookahead position
            iteration: Số iteration
            
        Returns:
            Direction để update weights (negative velocity)
        """
        # Initialize velocity on first iteration
        if self.velocity is None:
            self.velocity = np.zeros_like(gradient)
        
        # Update velocity: v = β * v + ∇L_lookahead
        self.velocity = self.momentum_coefficient * self.velocity + gradient
        
        # Return negative velocity (descent direction)
        return -self.velocity
    
    def reset(self) -> None:
        """Reset velocity về zero."""
        self.velocity = None
    
    def get_parameters(self) -> Dict[str, Any]:
        """Lấy tham số Nesterov momentum."""
        return {
            'momentum': 'nesterov',
            'momentum_coefficient': self.momentum_coefficient
        }
    
    def get_lookahead_weights(self, current_weights: np.ndarray, step_size: float) -> np.ndarray:
        """
        Tính lookahead weights cho Nesterov momentum.
        
        Args:
            current_weights: Weights hiện tại
            step_size: Step size hiện tại
            
        Returns:
            Lookahead weights
        """
        if self.velocity is None:
            return current_weights
        
        # Lookahead: θ_lookahead = θ - α * β * v
        return current_weights - step_size * self.momentum_coefficient * self.velocity

def create_momentum_strategy(method: str, **params) -> MomentumStrategy:
    """
    Factory function tạo momentum strategy.
    
    Args:
        method: Tên phương pháp ('none', 'standard', 'nesterov')
        **params: Các tham số cho strategy
        
    Returns:
        MomentumStrategy instance
    """
    strategies = {
        'none': NoMomentum,
        'standard': StandardMomentum,
        'classical': StandardMomentum,  # Alias
        'nesterov': NesterovMomentum,
        'nag': NesterovMomentum  # Alias
    }
    
    if method not in strategies:
        raise ValueError(f"Unknown momentum method: {method}. "
                        f"Available: {list(strategies.keys())}")
    
    return strategies[method](**params)