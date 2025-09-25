# -*- coding: utf-8 -*-
"""
Abstract base class for all optimization algorithms.
This module defines the common interface that all optimizers must implement,
ensuring consistent behavior and enabling polymorphic usage.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import sys
# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from utils.model_mixins import ComplexityTrackingMixin, OptimizationResultsMixin
from utils.optimization_utils import (
    add_bias_column,
    du_doan,
    danh_gia_mo_hinh,
    in_ket_qua_danh_gia,
    tinh_gia_tri_ham_loss,
    tinh_gradient_ham_loss
)

class BaseOptimizer(ComplexityTrackingMixin, OptimizationResultsMixin, ABC):
    """
    Abstract base class for all optimization algorithms.
    
    Inherits from:
        - ComplexityTrackingMixin: Computational complexity tracking
        - OptimizationResultsMixin: Results management
    
    Common attributes:
        loss_type (str): Type of loss function ('ols', 'ridge', 'lasso')
        regularization (float): Regularization parameter
        convergence_tolerance (float): Convergence threshold
        max_iterations (int): Maximum number of iterations
        convergence_check_freq (int): Frequency of convergence checking
        
    State attributes:
        weights: Current weight vector
        loss_history: History of loss values
        gradient_norms: History of gradient norms
        weights_history: History of weight vectors
        training_time: Training time
        converged: Convergence status
        final_iteration: Final iteration number
    """
    
    def __init__(self, 
                 loss_type: str = 'ols',
                 regularization: float = 0.01,
                 convergence_tolerance: float = 1e-3,
                 max_iterations: int = 10000,
                 convergence_check_freq: int = 1,
                 random_state: Optional[int] = None,
                 # Legacy parameter support
                 ham_loss: Optional[str] = None,
                 diem_dung: Optional[float] = None):
        """
        Initialize base optimizer.
        
        Args:
            loss_type: Type of loss function
            regularization: Regularization parameter  
            convergence_tolerance: Convergence threshold
            max_iterations: Maximum number of iterations
            convergence_check_freq: Frequency of convergence checking
            random_state: Seed for random number generator
            ham_loss: Legacy parameter name for loss_type
            diem_dung: Legacy parameter name for convergence_tolerance
        """
        # Initialize mixins
        super().__init__()
        
        # Handle legacy parameter names
        if ham_loss is not None:
            loss_type = ham_loss
        if diem_dung is not None:
            convergence_tolerance = diem_dung
        
        # Core parameters
        self.loss_type = loss_type.lower()
        self.ham_loss = self.loss_type  # Legacy compatibility
        self.regularization = regularization
        self.convergence_tolerance = convergence_tolerance
        self.diem_dung = convergence_tolerance  # Legacy compatibility
        self.max_iterations = max_iterations
        self.convergence_check_freq = convergence_check_freq
        self.random_state = random_state
        
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
        
        # Initialize loss and gradient functions using existing utilities
        self.loss_func = lambda X, y, w: tinh_gia_tri_ham_loss(
            X, y, w, self.loss_type, self.regularization
        )
        self.grad_func = lambda X, y, w: tinh_gradient_ham_loss(
            X, y, w, self.loss_type, self.regularization
        )
        
        # State variables - will be initialized in fit()
        self.weights: Optional[np.ndarray] = None
        self.loss_history: list = []
        self.gradient_norms: list = []
        self.weights_history: list = []
        self.training_time: float = 0.0
        self.converged: bool = False
        self.final_iteration: int = 0
        
    @abstractmethod
    def _initialize_algorithm_specific_params(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Initialize algorithm-specific parameters.
        
        Args:
            X: Feature matrix (with bias)
            y: Target vector
        """
        pass
    
    @abstractmethod
    def _perform_single_iteration(self, X: np.ndarray, y: np.ndarray, iteration: int) -> None:
        """
        Perform one iteration of the algorithm.
        
        Args:
            X: Feature matrix (with bias)
            y: Target vector  
            iteration: Current iteration number
        """
        pass
    
    @abstractmethod
    def _get_algorithm_specific_results(self) -> Dict[str, Any]:
        """
        Get algorithm-specific results.
        
        Returns:
            Dictionary containing algorithm-specific results
        """
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Template method for training process.
        
        Args:
            X: Feature matrix (without bias)
            y: Target vector
            
        Returns:
            Dictionary containing training results
        """
        import time
        from utils.optimization_utils import kiem_tra_dieu_kien_dung
        
        print(f"Training {self.__class__.__name__} - {self.loss_type.upper()}")
        if self.loss_type in ['ridge', 'lasso']:
            print(f"   Regularization: {self.regularization}")
        
        # Add bias column to X
        X_with_bias = add_bias_column(X)
        print(f"   Feature: {X.shape[1]} (+1 bias)")
        
        # Initialize complexity tracker
        self.init_complexity_tracker(X, y)
        
        # Initialize weights with fixed point for consistent comparison
        n_features_with_bias = X_with_bias.shape[1]
        # Use fixed seed for reproducible initialization
        np.random.seed(42)
        self.weights = np.random.normal(0, 0.01, n_features_with_bias)
        # Reset to original random state if it was set
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Algorithm-specific initialization
        self._initialize_algorithm_specific_params(X_with_bias, y)
        
        # Reset state
        self.loss_history = []
        self.gradient_norms = []
        self.weights_history = []
        self.converged = False
        self.final_iteration = 0
        
        start_time = time.time()
        
        # Main optimization loop
        for iteration in range(self.max_iterations):
            # Perform one iteration of the specific algorithm
            self._perform_single_iteration(X_with_bias, y, iteration)
            
            # Check convergence at specified frequency
            should_check_convergence = (
                (iteration + 1) % self.convergence_check_freq == 0 or
                iteration == self.max_iterations - 1
            )
            
            if should_check_convergence:
                # Compute metrics for convergence check
                loss_value = self.loss_func(X_with_bias, y, self.weights)
                gradient_w, _ = self.grad_func(X_with_bias, y, self.weights)
                gradient_norm = np.linalg.norm(gradient_w)
                
                # Track complexity
                self.track_function_evaluation(X_with_bias.shape)
                self.track_gradient_evaluation(X_with_bias.shape)
                self.track_vector_operation(len(gradient_w), "norm")
                
                # Store history
                self.loss_history.append(loss_value)
                self.gradient_norms.append(gradient_norm)
                self.weights_history.append(self.weights.copy())
                
                # Check convergence
                cost_change = 0.0 if len(self.loss_history) <= 1 else (
                    self.loss_history[-2] - self.loss_history[-1]
                )
                
                should_stop, converged, reason = kiem_tra_dieu_kien_dung(
                    gradient_norm=gradient_norm,
                    cost_change=cost_change,
                    iteration=iteration,
                    tolerance=self.convergence_tolerance,
                    max_iterations=self.max_iterations,
                    loss_value=loss_value,
                    weights=self.weights
                )
                
                if should_stop:
                    if converged:
                        print(f"[SUCCESS] {self.__class__.__name__} converged: {reason}")
                        self.mark_convergence_tracking(iteration + 1)
                    else:
                        print(f"[WARNING] {self.__class__.__name__} stopped (not converged): {reason}")
                    self.converged = converged
                    self.final_iteration = iteration + 1
                    break
                
                # Progress logging
                if (iteration + 1) % 100 == 0:
                    # Get current learning rate if available
                    learning_rate_str = ""
                    if hasattr(self, 'learning_rate_history') and self.learning_rate_history:
                        current_lr = self.learning_rate_history[-1]
                        learning_rate_str = f", LR = {current_lr:.6f}"

                    print(f"   Iteration {iteration + 1}: Loss = {loss_value:.6f}, "
                          f"Gradient = {gradient_norm:.6f}{learning_rate_str}")
            
            # End iteration tracking
            self.end_iteration_tracking()
        
        self.training_time = time.time() - start_time
        
        if not self.converged:
            print(f"[LIMIT REACHED] {self.__class__.__name__} stopped: Max {self.max_iterations} iterations reached")
            self.final_iteration = self.max_iterations
        
        print(f"Training time: {self.training_time:.2f}s")
        print(f"Final loss: {self.loss_history[-1]:.6f}")
        print(f"Final gradient norm: {self.gradient_norms[-1]:.6f}")
        
        # Print complexity summary
        self.print_complexity_summary()
        
        # Get best results
        best_results = self._get_best_results()
        print(f"[BEST] Best results (lowest gradient norm):")
        print(f"   Best iteration: {best_results['best_iteration']}")
        print(f"   Best loss: {best_results['best_loss']:.6f}")
        print(f"   Best gradient norm: {best_results['best_gradient_norm']:.6f}")
        
        # Create results dictionary
        results = self.create_standard_results_dict(
            algorithm_name=self.__class__.__name__,
            loss_function=self.loss_type
        )
        
        # Add best results information
        results['best_results'] = {
            'best_weights': best_results['best_weights'],
            'best_loss': best_results['best_loss'],
            'best_iteration': best_results['best_iteration'],
            'best_gradient_norm': best_results['best_gradient_norm']
        }
        
        # Add standard fields for compatibility
        results.update({
            'weights': best_results['best_weights'],
            'bias': best_results['best_weights'][-1],
            'loss_history': self.loss_history,
            'gradient_norms': self.gradient_norms,
            'weights_history': self.weights_history,
            'training_time': self.training_time,
            'converged': self.converged,
            'final_iteration': self.final_iteration,
            'best_iteration': best_results['best_iteration'],
            'best_loss': best_results['best_loss'],
            'best_gradient_norm': best_results['best_gradient_norm'],
            'final_loss': self.loss_history[-1],
            'final_gradient_norm': self.gradient_norms[-1]
        })
        
        # Add algorithm-specific results
        algorithm_specific = self._get_algorithm_specific_results()
        results.update(algorithm_specific)
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with data X.
        
        Args:
            X: Feature matrix (without bias)
            
        Returns:
            Prediction vector
        """
        if self.weights is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        X_with_bias = add_bias_column(X)
        return du_doan(X_with_bias, self.weights, None)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test feature matrix
            y_test: Test target vector
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.weights is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        # Extract bias and weights for compatibility with existing function
        bias_value = self.weights[-1]
        weights_without_bias = self.weights[:-1]
        
        metrics = danh_gia_mo_hinh(weights_without_bias, X_test, y_test, bias_value)
        # Store the latest evaluation metrics for use in save_results
        self._latest_ml_metrics = metrics
        in_ket_qua_danh_gia(metrics, self.training_time,
                           f"{self.__class__.__name__} - {self.loss_type.upper()}")
        return metrics
    
    def _get_best_results(self) -> Dict[str, Any]:
        """
        Get best results based on lowest gradient norm.
        
        Returns:
            Dictionary containing best_weights, best_loss, best_gradient_norm, best_iteration
        """
        if not self.gradient_norms:
            raise ValueError("No gradient norm history available to find best results")
        
        best_idx = np.argmin(self.gradient_norms)
        
        return {
            'best_weights': self.weights_history[best_idx],
            'best_loss': float(self.loss_history[best_idx]),
            'best_gradient_norm': float(self.gradient_norms[best_idx]),
            'best_iteration': int(best_idx * self.convergence_check_freq)
        }