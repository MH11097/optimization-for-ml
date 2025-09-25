"""
Step size strategies for optimization algorithms.
This module provides various strategies for computing step sizes,
following the Strategy design pattern for extensibility.
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union

class StepSizeStrategy(ABC):
    """
    Abstract base class cho tất cả các chiến lược step size.
    
    Strategy pattern cho phép chuyển đổi dễ dàng giữa các phương pháp
    tính step size khác nhau.
    """
    
    @abstractmethod
    def compute_step_size(self,
                         iteration: int,
                         gradient: np.ndarray,
                         X: Optional[np.ndarray] = None,
                         y: Optional[np.ndarray] = None,
                         weights: Optional[np.ndarray] = None,
                         **kwargs) -> Union[float, np.ndarray]:
        """
        Tính step size cho iteration hiện tại.
        
        Args:
            iteration: Số iteration hiện tại (bắt đầu từ 0)
            gradient: Gradient hiện tại
            X: Ma trận đặc trưng (nếu cần cho line search)
            y: Vector target (nếu cần cho line search) 
            weights: Vector trọng số hiện tại (nếu cần cho line search)
            **kwargs: Các tham số bổ sung
            
        Returns:
            Step size (scalar hoặc vector)
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset strategy về trạng thái ban đầu."""
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """Lấy tham số của strategy."""
        return {}

class ConstantStepSize(StepSizeStrategy):
    """
    Chiến lược step size cố định.
    
    Step size không thay đổi theo iteration.
    """
    
    def __init__(self, learning_rate: float = 0.01):
        """
        Khởi tạo constant step size.
        
        Args:
            learning_rate: Giá trị step size cố định
        """
        if learning_rate <= 0:
            raise ValueError("learning_rate phải > 0")
        
        self.learning_rate = learning_rate
    
    def compute_step_size(self,
                         iteration: int,
                         gradient: np.ndarray,
                         X: Optional[np.ndarray] = None,
                         y: Optional[np.ndarray] = None,
                         weights: Optional[np.ndarray] = None,
                         **kwargs) -> float:
        """Trả về step size cố định."""
        return self.learning_rate
    
    def reset(self) -> None:
        """Constant step size không cần reset."""
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """Lấy tham số constant step size."""
        return {
            'strategy': 'constant',
            'learning_rate': self.learning_rate
        }

class LinearDecayStepSize(StepSizeStrategy):
    """
    Chiến lược giảm step size tuyến tính.
    
    Step size giảm tuyến tính theo iteration:
    lr(t) = lr_0 * (1 - t/max_iterations)
    """
    
    def __init__(self, 
                 initial_learning_rate: float = 0.01,
                 max_iterations: int = 10000,
                 min_learning_rate: float = 0.0001):
        """
        Khởi tạo linear decay step size.
        
        Args:
            initial_learning_rate: Learning rate ban đầu
            max_iterations: Số iteration tối đa
            min_learning_rate: Learning rate tối thiểu
        """
        if initial_learning_rate <= 0:
            raise ValueError("initial_learning_rate phải > 0")
        if max_iterations <= 0:
            raise ValueError("max_iterations phải > 0")
        if min_learning_rate < 0:
            raise ValueError("min_learning_rate phải >= 0")
        
        self.initial_learning_rate = initial_learning_rate
        self.max_iterations = max_iterations
        self.min_learning_rate = min_learning_rate
    
    def compute_step_size(self,
                         iteration: int,
                         gradient: np.ndarray,
                         X: Optional[np.ndarray] = None,
                         y: Optional[np.ndarray] = None,
                         weights: Optional[np.ndarray] = None,
                         **kwargs) -> float:
        """Tính step size với linear decay."""
        decay_factor = 1.0 - (iteration / max(self.max_iterations, 1))
        learning_rate = self.initial_learning_rate * max(decay_factor, 0.0)
        return max(learning_rate, self.min_learning_rate)
    
    def reset(self) -> None:
        """Linear decay không cần reset state."""
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """Lấy tham số linear decay."""
        return {
            'strategy': 'linear_decay',
            'initial_learning_rate': self.initial_learning_rate,
            'max_iterations': self.max_iterations,
            'min_learning_rate': self.min_learning_rate
        }

class ExponentialDecayStepSize(StepSizeStrategy):
    """
    Chiến lược giảm step size exponential.
    
    Step size giảm exponential theo iteration:
    lr(t) = lr_0 * decay_rate^(t/decay_steps)
    """
    
    def __init__(self,
                 initial_learning_rate: float = 0.01,
                 decay_rate: float = 0.95,
                 decay_steps: int = 100):
        """
        Khởi tạo exponential decay step size.
        
        Args:
            initial_learning_rate: Learning rate ban đầu
            decay_rate: Tỷ lệ giảm (0 < decay_rate < 1)
            decay_steps: Số steps để decay
        """
        if initial_learning_rate <= 0:
            raise ValueError("initial_learning_rate phải > 0")
        if not (0 < decay_rate < 1):
            raise ValueError("decay_rate phải trong khoảng (0, 1)")
        if decay_steps <= 0:
            raise ValueError("decay_steps phải > 0")
        
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
    
    def compute_step_size(self,
                         iteration: int,
                         gradient: np.ndarray,
                         X: Optional[np.ndarray] = None,
                         y: Optional[np.ndarray] = None,
                         weights: Optional[np.ndarray] = None,
                         **kwargs) -> float:
        """Tính step size với exponential decay."""
        return self.initial_learning_rate * (self.decay_rate ** (iteration / self.decay_steps))
    
    def reset(self) -> None:
        """Exponential decay không cần reset state."""
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """Lấy tham số exponential decay."""
        return {
            'strategy': 'exponential_decay',
            'initial_learning_rate': self.initial_learning_rate,
            'decay_rate': self.decay_rate,
            'decay_steps': self.decay_steps
        }

class SqrtDecayStepSize(StepSizeStrategy):
    """
    Chiến lược giảm step size theo căn bậc hai.
    
    Step size giảm theo căn bậc hai:
    lr(t) = lr_0 / sqrt(t + 1)
    """
    
    def __init__(self, initial_learning_rate: float = 0.01):
        """
        Khởi tạo sqrt decay step size.
        
        Args:
            initial_learning_rate: Learning rate ban đầu
        """
        if initial_learning_rate <= 0:
            raise ValueError("initial_learning_rate phải > 0")
        
        self.initial_learning_rate = initial_learning_rate
    
    def compute_step_size(self,
                         iteration: int,
                         gradient: np.ndarray,
                         X: Optional[np.ndarray] = None,
                         y: Optional[np.ndarray] = None,
                         weights: Optional[np.ndarray] = None,
                         **kwargs) -> float:
        """Tính step size với sqrt decay."""
        return self.initial_learning_rate / np.sqrt(iteration + 1)
    
    def reset(self) -> None:
        """Sqrt decay không cần reset state."""
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """Lấy tham số sqrt decay.""" 
        return {
            'strategy': 'sqrt_decay',
            'initial_learning_rate': self.initial_learning_rate
        }

class AdaptiveStepSize(StepSizeStrategy):
    """
    Chiến lược step size adaptive (Adam-like).
    
    Sử dụng first và second moment estimates để tính adaptive step sizes.
    """
    
    def __init__(self,
                 initial_learning_rate: float = 0.01,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8):
        """
        Khởi tạo adaptive step size.
        
        Args:
            initial_learning_rate: Learning rate ban đầu
            beta1: Exponential decay cho first moment
            beta2: Exponential decay cho second moment  
            epsilon: Smoothing term để tránh chia cho 0
        """
        if initial_learning_rate <= 0:
            raise ValueError("initial_learning_rate phải > 0")
        if not (0 < beta1 < 1):
            raise ValueError("beta1 phải trong khoảng (0, 1)")
        if not (0 < beta2 < 1):
            raise ValueError("beta2 phải trong khoảng (0, 1)")
        if epsilon <= 0:
            raise ValueError("epsilon phải > 0")
        
        self.initial_learning_rate = initial_learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # State variables
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Time step
    
    def compute_step_size(self,
                         iteration: int,
                         gradient: np.ndarray,
                         X: Optional[np.ndarray] = None,
                         y: Optional[np.ndarray] = None,
                         weights: Optional[np.ndarray] = None,
                         **kwargs) -> np.ndarray:
        """Tính adaptive step size (trả về vector)."""
        self.t += 1
        
        # Initialize moments on first iteration
        if self.m is None:
            self.m = np.zeros_like(gradient)
            self.v = np.zeros_like(gradient)
        
        # Update biased first and second moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Return adaptive step size vector
        return self.initial_learning_rate / (np.sqrt(v_hat) + self.epsilon)
    
    def reset(self) -> None:
        """Reset adaptive state."""
        self.m = None
        self.v = None
        self.t = 0
    
    def get_parameters(self) -> Dict[str, Any]:
        """Lấy tham số adaptive step size."""
        return {
            'strategy': 'adaptive',
            'initial_learning_rate': self.initial_learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon
        }

def create_step_size_strategy(method: str, **params) -> StepSizeStrategy:
    """
    Factory function tạo step size strategy.
    
    Args:
        method: Tên phương pháp ('constant', 'linear_decay', 'exponential_decay', 
                'sqrt_decay', 'adaptive')
        **params: Các tham số cho strategy
        
    Returns:
        StepSizeStrategy instance
    """
    strategies = {
        'constant': ConstantStepSize,
        'linear_decay': LinearDecayStepSize,  
        'exponential_decay': ExponentialDecayStepSize,
        'sqrt_decay': SqrtDecayStepSize,
        'adaptive': AdaptiveStepSize
    }
    
    if method not in strategies:
        raise ValueError(f"Unknown step size method: {method}. "
                        f"Available: {list(strategies.keys())}")
    
    return strategies[method](**params)