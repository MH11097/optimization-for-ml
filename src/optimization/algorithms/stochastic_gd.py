"""
Stochastic Gradient Descent Optimizer
This module implements the Stochastic Gradient Descent family of algorithms
with various batch strategies, step size schedules, and momentum variants.
"""
import numpy as np
from typing import Dict, Any, Optional, Union, Literal
import sys
from pathlib import Path
# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from optimization.base import IterativeOptimizer
from utils.optimization_utils import (
    add_bias_column,
    tinh_gia_tri_ham_loss,
    tinh_gradient_ham_loss
)

class StochasticGDOptimizer(IterativeOptimizer):
    """
    Stochastic Gradient Descent Optimizer with unified interface.
    
    Supports:
    - Multiple batch strategies (fixed, adaptive, random)
    - Various step size schedules (constant, linear decay, exponential decay, sqrt decay)
    - Momentum variants (standard, Nesterov)
    - Shuffling strategies (per epoch, random)
    
    Architecture: Follows OOP and DRY principles with template method pattern.
    """
    
    def __init__(self,
                 # Core parameters
                 loss_type: str = 'ols',
                 learning_rate: float = 0.01,
                 regularization: float = 0.01,
                 convergence_tolerance: float = 1e-3,
                 max_iterations: int = 100000,
                 convergence_check_freq: int = 100,
                 random_state: Optional[int] = None,
                 
                 # SGD-specific parameters
                 batch_size: Union[int, Literal['full']] = 256,
                 batch_strategy: str = 'fixed',  # 'fixed', 'adaptive', 'random'
                 step_size_method: str = 'constant',  # 'constant', 'linear_decay', 'exponential_decay', 'sqrt_decay', 'fixed_step_length', 'lipschitz_step_length'
                 shuffle_strategy: str = 'per_epoch',  # 'per_epoch', 'random', 'none'
                 momentum_method: str = 'none',  # 'none', 'standard', 'nesterov'
                 momentum_coefficient: float = 0.9,
                 
                 # Step size schedule parameters
                 decay_rate: float = 0.1,
                 decay_steps: Optional[int] = None,

                 # Step length parameters (alternative to learning rate)
                 fixed_step_length: float = 0.04,  # Fixed step length for normalization
                 use_lipschitz_step_length: bool = False,  # Use Lipschitz-based step length
                 
                 # Legacy compatibility
                 ham_loss: Optional[str] = None,
                 diem_dung: Optional[float] = None):
        """
        Initialize Stochastic Gradient Descent optimizer.
        
        Args:
            loss_type: Type of loss function ('ols', 'ridge', 'lasso')
            learning_rate: Initial learning rate
            regularization: Regularization parameter
            convergence_tolerance: Convergence threshold
            max_iterations: Maximum number of iterations (epochs)
            convergence_check_freq: Frequency of convergence checking
            random_state: Random seed for reproducibility
            batch_size: Batch size (int or 'full' for full batch)
            batch_strategy: Batch selection strategy
            step_size_method: Step size schedule method or step length method
            shuffle_strategy: Data shuffling strategy
            momentum_method: Momentum variant
            momentum_coefficient: Momentum coefficient (0.9 typical)
            decay_rate: Decay rate for step size schedules
            decay_steps: Steps for decay schedule
            fixed_step_length: Fixed step length value for normalization approaches
            use_lipschitz_step_length: Whether to use Lipschitz constant for step length
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
        
        # SGD-specific parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.batch_strategy = batch_strategy
        self.step_size_method = step_size_method
        self.shuffle_strategy = shuffle_strategy
        self.momentum_method = momentum_method
        self.momentum_coefficient = momentum_coefficient
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps or max_iterations // 4

        # Step length parameters
        self.fixed_step_length = fixed_step_length
        self.use_lipschitz_step_length = use_lipschitz_step_length

        # Lipschitz constant (computed when needed)
        self.lipschitz_constant: Optional[float] = None
        
        # Internal state
        self.momentum_vector: Optional[np.ndarray] = None
        self.current_lr: float = learning_rate
        self.data_indices: Optional[np.ndarray] = None
        self.current_batch_indices: Optional[np.ndarray] = None
        
        # History tracking
        self.batch_losses: list = []
        self.learning_rates: list = []
        
    def _initialize_algorithm_specific_params(self, X: np.ndarray, y: np.ndarray) -> None:
        """Initialize SGD-specific parameters."""
        n_samples = X.shape[0]
        
        # Initialize data indices
        self.data_indices = np.arange(n_samples)
        
        # Initialize momentum vector if needed
        if self.momentum_method != 'none':
            self.momentum_vector = np.zeros_like(self.weights)
        
        # Set effective batch size
        if self.batch_size == 'full' or self.batch_size >= n_samples:
            self.effective_batch_size = n_samples
        else:
            self.effective_batch_size = min(self.batch_size, n_samples)
        
        print(f"   Batch size: {self.effective_batch_size} ({self.batch_strategy})")
        print(f"   Step schedule: {self.step_size_method}")
        print(f"   Momentum: {self.momentum_method}")
        print(f"   Shuffle: {self.shuffle_strategy}")
        
    def _compute_update_direction(self, X: np.ndarray, y: np.ndarray, iteration: int) -> np.ndarray:
        """
        Compute update direction using stochastic gradient.
        
        Template method that handles:
        1. Batch selection
        2. Gradient computation on batch
        3. Momentum application (if enabled)
        """
        # Step 1: Select batch
        batch_X, batch_y = self._select_batch(X, y, iteration)
        
        # Step 2: Compute gradient on batch
        gradient, _ = self.grad_func(batch_X, batch_y, self.weights)
        
        # Track batch operations
        self.track_gradient_evaluation(batch_X.shape)
        
        # Step 3: Apply momentum if enabled
        if self.momentum_method == 'standard':
            # Standard momentum: v_t = γv_{t-1} + η∇f
            self.momentum_vector = (self.momentum_coefficient * self.momentum_vector + 
                                   gradient)
            update_direction = -self.momentum_vector
        elif self.momentum_method == 'nesterov':
            # Nesterov momentum: v_t = γv_{t-1} + η∇f(w - γv_{t-1})
            # Simplified implementation
            self.momentum_vector = (self.momentum_coefficient * self.momentum_vector + 
                                   gradient)
            update_direction = -(self.momentum_coefficient * self.momentum_vector + gradient)
        else:
            # No momentum: standard gradient descent direction
            update_direction = -gradient
        
        # Track momentum history
        if self.momentum_method != 'none':
            momentum_norm = np.linalg.norm(self.momentum_vector)
            self.momentum_history.append(momentum_norm)
        
        # Track batch loss for monitoring
        batch_loss = self.loss_func(batch_X, batch_y, self.weights)
        self.batch_losses.append(batch_loss)
        
        return update_direction
    
    def _compute_step_size(self, X: np.ndarray, y: np.ndarray,
                          direction: np.ndarray, iteration: int) -> float:
        """
        Compute step size using specified schedule or step length method.

        Supports both traditional step size schedules and step length approaches.
        """
        if self.step_size_method == 'constant':
            step_size = self.learning_rate

        elif self.step_size_method == 'linear_decay':
            # Linear decay: lr_t = lr_0 * (1 - t/T)
            decay_factor = 1.0 - iteration / self.max_iterations
            step_size = self.learning_rate * decay_factor

        elif self.step_size_method == 'exponential_decay':
            # Exponential decay: lr_t = lr_0 * γ^(t/decay_steps)
            decay_factor = self.decay_rate ** (iteration / self.decay_steps)
            step_size = self.learning_rate * decay_factor

        elif self.step_size_method == 'sqrt_decay':
            # Square root decay: lr_t = lr_0 / sqrt(1 + t)
            step_size = self.learning_rate / np.sqrt(1 + iteration)

        elif self.step_size_method == 'fixed_step_length':
            # Fixed step length: step_size = step_length / ||gradient||
            gradient_norm = np.linalg.norm(direction)
            if gradient_norm > 1e-12:  # Avoid division by zero
                step_size = self.fixed_step_length / gradient_norm
            else:
                step_size = self.fixed_step_length  # Fallback for zero gradient

        elif self.step_size_method == 'lipschitz_step_length':
            # Lipschitz-based step length: uses computed Lipschitz constant
            if self.lipschitz_constant is None:
                self._compute_lipschitz_constant(X, y)

            gradient_norm = np.linalg.norm(direction)
            if gradient_norm > 1e-12:
                # Conservative step length based on Lipschitz constant
                theoretical_step_length = 1.0 / self.lipschitz_constant
                step_size = theoretical_step_length / gradient_norm
            else:
                step_size = 1.0 / self.lipschitz_constant  # Fallback

        else:
            raise ValueError(f"Unknown step size method: {self.step_size_method}")

        # Update current learning rate and track history
        self.current_lr = step_size
        self.learning_rates.append(step_size)

        return step_size
    
    def _select_batch(self, X: np.ndarray, y: np.ndarray, iteration: int) -> tuple:
        """
        Select batch using specified strategy.
        
        Implements various batch selection strategies with DRY principle.
        """
        n_samples = X.shape[0]
        
        # Handle shuffling strategy
        if self.shuffle_strategy == 'per_epoch' and iteration % (n_samples // self.effective_batch_size) == 0:
            np.random.shuffle(self.data_indices)
        elif self.shuffle_strategy == 'random':
            np.random.shuffle(self.data_indices)
        
        # Select batch based on strategy
        if self.batch_strategy == 'fixed':
            # Fixed sequential batches
            start_idx = (iteration * self.effective_batch_size) % n_samples
            end_idx = min(start_idx + self.effective_batch_size, n_samples)
            
            if end_idx - start_idx < self.effective_batch_size and n_samples > self.effective_batch_size:
                # Wrap around if needed
                remaining = self.effective_batch_size - (end_idx - start_idx)
                batch_indices = np.concatenate([
                    self.data_indices[start_idx:end_idx],
                    self.data_indices[:remaining]
                ])
            else:
                batch_indices = self.data_indices[start_idx:end_idx]
                
        elif self.batch_strategy == 'random':
            # Random sampling with replacement
            batch_indices = np.random.choice(self.data_indices, 
                                           size=self.effective_batch_size, 
                                           replace=False)
        else:
            raise ValueError(f"Unknown batch strategy: {self.batch_strategy}")
        
        self.current_batch_indices = batch_indices
        return X[batch_indices], y[batch_indices]
    
    def _get_algorithm_specific_results(self) -> Dict[str, Any]:
        """Get SGD-specific results."""
        results = super()._get_algorithm_specific_results()
        
        results.update({
            'algorithm_type': 'StochasticGD',
            'batch_size': self.effective_batch_size,
            'batch_strategy': self.batch_strategy,
            'step_size_method': self.step_size_method,
            'shuffle_strategy': self.shuffle_strategy,
            'momentum_method': self.momentum_method,
            'momentum_coefficient': self.momentum_coefficient,
            'final_learning_rate': self.current_lr,
            'batch_losses_history': self.batch_losses.copy(),
            'learning_rates_history': self.learning_rates.copy(),
            'fixed_step_length': self.fixed_step_length,
            'lipschitz_constant': self.lipschitz_constant,
        })
        
        return results

    def _compute_lipschitz_constant(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Compute Lipschitz constant for the loss function.

        Uses the maximum eigenvalue of the Hessian matrix as Lipschitz constant.
        """
        try:
            # Import Lipschitz utility
            from utils.lipschitz_simple import compute_lipschitz_constant_and_optimal_lr

            # Compute Lipschitz constant for current loss type
            results = compute_lipschitz_constant_and_optimal_lr(
                loss_type=self.loss_type,
                regularization=self.regularization,
                verbose=False
            )

            self.lipschitz_constant = results['lipschitz_constant']
            print(f"   Computed Lipschitz constant: {self.lipschitz_constant:.6f}")

        except ImportError:
            # Fallback: approximate Lipschitz constant
            print("   Warning: Lipschitz utility not available, using approximation")
            if self.loss_type == 'ols':
                # For OLS: L ≈ lambda_max(X^T X) / n
                XTX = X.T @ X / X.shape[0]
                eigenvalues = np.linalg.eigvals(XTX)
                self.lipschitz_constant = np.max(eigenvalues)
            elif self.loss_type == 'ridge':
                # For Ridge: L ≈ lambda_max(X^T X) / n + 2α
                XTX = X.T @ X / X.shape[0]
                eigenvalues = np.linalg.eigvals(XTX)
                self.lipschitz_constant = np.max(eigenvalues) + 2 * self.regularization
            else:
                # Default fallback
                self.lipschitz_constant = 10.0

            print(f"   Approximated Lipschitz constant: {self.lipschitz_constant:.6f}")
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about available strategies."""
        return {
            'batch_strategies': ['fixed', 'random'],
            'step_size_methods': ['constant', 'linear_decay', 'exponential_decay', 'sqrt_decay', 'fixed_step_length', 'lipschitz_step_length'],
            'shuffle_strategies': ['per_epoch', 'random', 'none'],
            'momentum_methods': ['none', 'standard', 'nesterov'],
            'current_config': {
                'batch_size': self.effective_batch_size,
                'batch_strategy': self.batch_strategy,
                'step_size_method': self.step_size_method,
                'momentum_method': self.momentum_method
            }
        }