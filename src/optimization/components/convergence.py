"""
Enhanced convergence checking utilities.
This module provides specialized convergence checkers that extend
the base convergence logic from utils.
"""
import numpy as np
from typing import Tuple, List, Optional
# Add project root to path for imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
from utils.optimization_utils import kiem_tra_dieu_kien_dung

class ConvergenceChecker:
    """
    Enhanced convergence checker với additional logic cho các thuật toán khác nhau.
    """
    
    def __init__(self,
                 tolerance: float = 1e-5,
                 max_iterations: int = 10000,
                 patience: int = 10,
                 min_delta: float = 1e-7,
                 divergence_threshold: float = 1e10):
        """
        Khởi tạo convergence checker.
        
        Args:
            tolerance: Ngưỡng hội tụ cho gradient norm
            max_iterations: Số iteration tối đa
            patience: Số iterations chờ đợi improvement cho early stopping
            min_delta: Mức cải thiện tối thiểu cho early stopping
            divergence_threshold: Ngưỡng phát hiện divergence
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.patience = patience
        self.min_delta = min_delta
        self.divergence_threshold = divergence_threshold
        
    def check_convergence(self,
                         gradient_norm: float,
                         cost_change: float,
                         iteration: int,
                         loss_value: float,
                         weights: Optional[np.ndarray] = None) -> Tuple[bool, bool, str]:
        """
        Kiểm tra convergence sử dụng logic cơ bản từ utils.
        
        Args:
            gradient_norm: Norm của gradient hiện tại
            cost_change: Sự thay đổi cost
            iteration: Iteration hiện tại
            loss_value: Giá trị loss hiện tại
            weights: Vector weights hiện tại
            
        Returns:
            (should_stop, converged, reason)
        """
        return kiem_tra_dieu_kien_dung(
            gradient_norm=gradient_norm,
            cost_change=cost_change,
            iteration=iteration,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            loss_value=loss_value,
            weights=weights
        )
    
    def check_early_stopping(self, loss_history: List[float]) -> Tuple[bool, str]:
        """
        Kiểm tra early stopping dựa trên loss plateau.
        
        Args:
            loss_history: Lịch sử loss values
            
        Returns:
            (should_stop, reason)
        """
        if len(loss_history) < self.patience + 1:
            return False, ""
        
        recent_losses = loss_history[-(self.patience + 1):]
        best_recent_loss = min(recent_losses[:-1])
        current_loss = recent_losses[-1]
        
        improvement = best_recent_loss - current_loss
        
        if improvement < self.min_delta:
            return True, f"Early stopping: No improvement for {self.patience} iterations"
        
        return False, ""
    
    def check_divergence(self, loss_history: List[float]) -> Tuple[bool, str]:
        """
        Kiểm tra divergence dựa trên loss growth.
        
        Args:
            loss_history: Lịch sử loss values
            
        Returns:
            (is_diverging, reason)
        """
        if len(loss_history) < 2:
            return False, ""
        
        current_loss = loss_history[-1]
        
        # Check for NaN or Inf
        if np.isnan(current_loss) or np.isinf(current_loss):
            return True, "Loss became NaN or Inf (numerical instability)"
        
        # Check for excessive growth
        if current_loss > self.divergence_threshold:
            return True, f"Loss exceeded divergence threshold ({self.divergence_threshold})"
        
        # Check for sustained increase
        if len(loss_history) >= 5:
            recent_losses = loss_history[-5:]
            if all(recent_losses[i] < recent_losses[i+1] for i in range(len(recent_losses)-1)):
                ratio = recent_losses[-1] / recent_losses[0]
                if ratio > 2.0:  # Loss doubled over 5 iterations
                    return True, "Loss increased consistently (possible divergence)"
        
        return False, ""
    
    def check_stagnation(self, 
                        loss_history: List[float],
                        gradient_norms: List[float]) -> Tuple[bool, str]:
        """
        Kiểm tra stagnation (không cải thiện trong thời gian dài).
        
        Args:
            loss_history: Lịch sử loss values
            gradient_norms: Lịch sử gradient norms
            
        Returns:
            (is_stagnating, reason)
        """
        min_history_length = max(20, self.patience * 2)
        
        if len(loss_history) < min_history_length:
            return False, ""
        
        # Check loss stagnation
        recent_losses = loss_history[-min_history_length:]
        loss_std = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)
        
        # Very small relative variation in loss
        if loss_mean > 0 and (loss_std / loss_mean) < 1e-6:
            return True, f"Loss stagnant for {min_history_length} iterations"
        
        # Check gradient stagnation
        if len(gradient_norms) >= min_history_length:
            recent_gradients = gradient_norms[-min_history_length:]
            grad_std = np.std(recent_gradients)
            grad_mean = np.mean(recent_gradients)
            
            # Very small relative variation in gradient norm
            if grad_mean > 0 and (grad_std / grad_mean) < 1e-6:
                return True, f"Gradient stagnant for {min_history_length} iterations"
        
        return False, ""
    
    def comprehensive_check(self,
                           gradient_norm: float,
                           cost_change: float,
                           iteration: int,
                           loss_value: float,
                           loss_history: List[float],
                           gradient_norms: List[float],
                           weights: Optional[np.ndarray] = None) -> Tuple[bool, bool, str]:
        """
        Comprehensive convergence check kết hợp tất cả các kiểm tra.
        
        Args:
            gradient_norm: Norm của gradient hiện tại
            cost_change: Sự thay đổi cost
            iteration: Iteration hiện tại
            loss_value: Giá trị loss hiện tại
            loss_history: Lịch sử loss values
            gradient_norms: Lịch sử gradient norms  
            weights: Vector weights hiện tại
            
        Returns:
            (should_stop, converged, reason)
        """
        # 1. Check basic convergence
        should_stop, converged, reason = self.check_convergence(
            gradient_norm, cost_change, iteration, loss_value, weights
        )
        
        if should_stop:
            return should_stop, converged, reason
        
        # 2. Check divergence
        is_diverging, div_reason = self.check_divergence(loss_history)
        if is_diverging:
            return True, False, div_reason
        
        # 3. Check early stopping
        early_stop, early_reason = self.check_early_stopping(loss_history)
        if early_stop:
            return True, False, early_reason
        
        # 4. Check stagnation
        is_stagnant, stag_reason = self.check_stagnation(loss_history, gradient_norms)
        if is_stagnant:
            return True, False, stag_reason
        
        # No stopping condition met
        return False, False, ""

class SGDConvergenceChecker(ConvergenceChecker):
    """
    Specialized convergence checker cho Stochastic Gradient Descent.
    
    SGD có noise trong gradients nên cần tolerance cao hơn.
    """
    
    def __init__(self,
                 tolerance: float = 1e-4,  # Higher tolerance for SGD noise
                 max_iterations: int = 10000,
                 patience: int = 20,       # More patience due to noise
                 min_delta: float = 1e-6,  # Smaller minimum delta
                 divergence_threshold: float = 1e10,
                 noise_tolerance_factor: float = 10.0):
        """
        Khởi tạo SGD convergence checker với tolerance cao hơn cho noise.
        
        Args:
            tolerance: Ngưỡng hội tụ (cao hơn cho SGD)
            max_iterations: Số iteration tối đa
            patience: Patience cao hơn do noise
            min_delta: Mức cải thiện tối thiểu
            divergence_threshold: Ngưỡng divergence
            noise_tolerance_factor: Factor tăng tolerance cho cost change
        """
        super().__init__(tolerance, max_iterations, patience, min_delta, divergence_threshold)
        self.noise_tolerance_factor = noise_tolerance_factor
    
    def check_convergence(self,
                         gradient_norm: float,
                         cost_change: float,
                         iteration: int,
                         loss_value: float,
                         weights: Optional[np.ndarray] = None) -> Tuple[bool, bool, str]:
        """
        SGD convergence check với tolerance cao hơn cho cost changes.
        """
        # Use base convergence check but with adjusted tolerance for cost change
        adjusted_tolerance = self.tolerance * self.noise_tolerance_factor
        
        return kiem_tra_dieu_kien_dung(
            gradient_norm=gradient_norm,
            cost_change=cost_change,
            iteration=iteration,
            tolerance=self.tolerance,      # Keep original for gradient
            max_iterations=self.max_iterations,
            loss_value=loss_value,
            weights=weights
        )