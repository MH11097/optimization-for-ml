"""
PyTorch L-BFGS wrapper for Newton-like optimization comparison.
"""

import numpy as np
from typing import Dict, Any, Optional
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
sys.path.append(project_root)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None

from src.optimization.external.base_library_wrapper import BaseLibraryWrapper


class PyTorchLBFGSWrapper(BaseLibraryWrapper):
    """
    Wrapper for PyTorch L-BFGS optimizer.
    
    Provides Newton-like optimization functionality using PyTorch's L-BFGS
    implementation, which approximates the Hessian using limited memory.
    """
    
    def __init__(self,
                 learning_rate: float = 1.0,
                 loss_type: str = 'ols',
                 regularization: float = 0.01,
                 max_iterations: int = 100000,
                 convergence_tolerance: float = 1e-3,
                 random_state: Optional[int] = None,
                 # L-BFGS specific parameters
                 max_iter: int = 20,
                 max_eval: Optional[int] = None,
                 tolerance_grad: Optional[float] = None,
                 tolerance_change: Optional[float] = None,
                 history_size: int = 100,
                 line_search_fn: Optional[str] = None,
                 **kwargs):
        """
        Initialize PyTorch L-BFGS wrapper.
        
        Args:
            learning_rate: Learning rate (step size) for L-BFGS
            loss_type: Type of loss function ('ols', 'ridge', 'lasso')
            regularization: Regularization parameter
            max_iterations: Maximum number of epochs
            convergence_tolerance: Convergence tolerance
            random_state: Random seed
            max_iter: Maximum iterations per L-BFGS step
            max_eval: Maximum function evaluations per optimization step
            tolerance_grad: Gradient tolerance
            tolerance_change: Parameter change tolerance
            history_size: History size for L-BFGS
            line_search_fn: Line search function ('strong_wolfe' or None)
            **kwargs: Additional parameters
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PyTorchLBFGSWrapper")
        
        # Store L-BFGS specific parameters
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_eval = max_eval if max_eval is not None else max_iter * 5 // 4
        self.tolerance_grad = tolerance_grad if tolerance_grad is not None else convergence_tolerance
        self.tolerance_change = tolerance_change if tolerance_change is not None else convergence_tolerance
        self.history_size = history_size
        self.line_search_fn = line_search_fn
        self.torch_kwargs = kwargs
        
        super().__init__(
            library_name='pytorch',
            algorithm_name='LBFGS',
            loss_type=loss_type,
            regularization=regularization,
            max_iterations=max_iterations,
            convergence_tolerance=convergence_tolerance,
            random_state=random_state,
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_eval=self.max_eval,
            tolerance_grad=self.tolerance_grad,
            tolerance_change=self.tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn,
            **kwargs
        )
        
        # PyTorch model components
        self.model = None
        self.criterion = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _create_pytorch_model(self, n_features: int) -> nn.Module:
        """
        Create simple linear model for PyTorch.
        
        Args:
            n_features: Number of features (including bias)
            
        Returns:
            PyTorch linear model
        """
        class LinearRegression(nn.Module):
            def __init__(self, n_features):
                super(LinearRegression, self).__init__()
                self.linear = nn.Linear(n_features, 1, bias=False)
                
                # Initialize weights to match our initialization
                with torch.no_grad():
                    self.linear.weight.data = torch.from_numpy(self.weights.reshape(1, -1)).float()
            
            def forward(self, x):
                return self.linear(x)
        
        return LinearRegression(n_features).to(self.device)
    
    def _create_loss_function(self) -> nn.Module:
        """
        Create PyTorch loss function.
        
        Returns:
            PyTorch loss function
        """
        if self.loss_type == 'ols':
            return nn.MSELoss()
        elif self.loss_type in ['ridge', 'lasso']:
            # Use MSE loss and add regularization manually
            return nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
    
    def _create_external_optimizer(self, n_features: int) -> optim.LBFGS:
        """
        Create PyTorch L-BFGS optimizer.
        
        Args:
            n_features: Number of features (including bias)
            
        Returns:
            PyTorch L-BFGS optimizer
        """
        # Create model and loss function
        self.model = self._create_pytorch_model(n_features)
        self.criterion = self._create_loss_function()
        
        # Set random seed for PyTorch
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        
        # Create L-BFGS optimizer
        lbfgs_params = {
            'lr': self.learning_rate,
            'max_iter': self.max_iter,
            'max_eval': self.max_eval,
            'tolerance_grad': self.tolerance_grad,
            'tolerance_change': self.tolerance_change,
            'history_size': self.history_size,
            'line_search_fn': self.line_search_fn
        }
        
        # Add any additional parameters
        lbfgs_params.update(self.torch_kwargs)
        
        return optim.LBFGS(self.model.parameters(), **lbfgs_params)
    
    def _create_closure(self, X_tensor: torch.Tensor, y_tensor: torch.Tensor):
        """
        Create closure function for L-BFGS optimizer.
        
        Args:
            X_tensor: Input features tensor
            y_tensor: Target values tensor
            
        Returns:
            Closure function for L-BFGS
        """
        def closure():
            self.external_optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            
            # Add regularization
            if self.loss_type == 'ridge' and self.regularization > 0:
                l2_penalty = self.regularization * torch.norm(self.model.linear.weight) ** 2
                loss = loss + l2_penalty
            elif self.loss_type == 'lasso' and self.regularization > 0:
                l1_penalty = self.regularization * torch.norm(self.model.linear.weight, p=1)
                loss = loss + l1_penalty
            
            loss.backward()
            return loss
        
        return closure
    
    def _optimize_external(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Run PyTorch L-BFGS optimization.
        
        Args:
            X: Feature matrix (with bias)
            y: Target vector
            
        Returns:
            Dictionary containing optimization results
        """
        try:
            # Convert to PyTorch tensors
            X_tensor = torch.from_numpy(X).float().to(self.device)
            y_tensor = torch.from_numpy(y.reshape(-1, 1)).float().to(self.device)
            
            # Create closure function
            closure = self._create_closure(X_tensor, y_tensor)
            
            converged = False
            epoch = 0
            
            # L-BFGS optimization loop
            for epoch in range(self.max_iterations):
                # Perform L-BFGS step
                loss = self.external_optimizer.step(closure)
                
                # Check convergence
                if epoch % self.convergence_check_freq == 0 or epoch == self.max_iterations - 1:
                    # Get current weights and gradient
                    current_weights = self.model.linear.weight.data.cpu().numpy().flatten()
                    
                    # Compute gradient for convergence check
                    self.external_optimizer.zero_grad()
                    outputs = self.model(X_tensor)
                    loss_val = self.criterion(outputs, y_tensor)
                    
                    # Add regularization to loss
                    if self.loss_type == 'ridge' and self.regularization > 0:
                        l2_penalty = self.regularization * torch.norm(self.model.linear.weight) ** 2
                        loss_val = loss_val + l2_penalty
                    elif self.loss_type == 'lasso' and self.regularization > 0:
                        l1_penalty = self.regularization * torch.norm(self.model.linear.weight, p=1)
                        loss_val = loss_val + l1_penalty
                    
                    loss_val.backward()
                    
                    # Get gradient norm
                    grad_norm = torch.norm(self.model.linear.weight.grad).item()
                    
                    # Store in callback history
                    self.callback_history['losses'].append(float(loss_val.item()))
                    self.callback_history['gradient_norms'].append(float(grad_norm))
                    self.callback_history['weights'].append(current_weights.copy())
                    self.callback_history['iterations'].append(epoch)
                    
                    # Check convergence
                    if grad_norm < self.convergence_tolerance:
                        converged = True
                        print(f"PyTorch L-BFGS converged at epoch {epoch} (gradient norm: {grad_norm:.6f})")
                        break
                
                # Check if L-BFGS internal convergence was reached
                # (This is approximate as PyTorch doesn't expose internal convergence status)
                if hasattr(self.external_optimizer, 'state') and len(self.external_optimizer.state) > 0:
                    state = list(self.external_optimizer.state.values())[0]
                    if 'n_iter' in state and state['n_iter'] >= self.max_iter:
                        # L-BFGS completed its internal iterations
                        break
            
            # Extract final weights
            final_weights = self.model.linear.weight.data.cpu().numpy().flatten()
            
            return {
                'final_weights': final_weights,
                'converged': converged,
                'iterations': epoch + 1,
                'final_loss': float(loss.item()) if torch.is_tensor(loss) else float(loss)
            }
            
        except Exception as e:
            print(f"PyTorch L-BFGS optimization failed: {str(e)}")
            return {
                'final_weights': self.weights,
                'converged': False,
                'iterations': 0,
                'error': str(e)
            }
    
    def _get_algorithm_specific_results(self) -> Dict[str, Any]:
        """
        Get PyTorch L-BFGS specific results.
        
        Returns:
            Dictionary containing PyTorch L-BFGS specific results
        """
        base_results = super()._get_algorithm_specific_results()
        
        pytorch_specific = {
            'pytorch_lbfgs_params': {
                'learning_rate': self.learning_rate,
                'max_iter': self.max_iter,
                'max_eval': self.max_eval,
                'tolerance_grad': self.tolerance_grad,
                'tolerance_change': self.tolerance_change,
                'history_size': self.history_size,
                'line_search_fn': self.line_search_fn,
                'device': str(self.device)
            }
        }
        
        # Add model and optimizer state information if available
        if self.model is not None:
            pytorch_specific['pytorch_model'] = {
                'model_type': type(self.model).__name__,
                'n_parameters': sum(p.numel() for p in self.model.parameters()),
                'device': str(next(self.model.parameters()).device)
            }
        
        if self.external_optimizer is not None and hasattr(self.external_optimizer, 'state'):
            # Extract L-BFGS optimizer state information
            state_info = {}
            for param_group in self.external_optimizer.param_groups:
                for param in param_group['params']:
                    if param in self.external_optimizer.state:
                        param_state = self.external_optimizer.state[param]
                        state_info.update({
                            'n_iter': param_state.get('n_iter', 0),
                            'prev_loss': param_state.get('prev_loss', None),
                            'H_diag': len(param_state.get('H_diag', [])) if 'H_diag' in param_state else 0
                        })
                        break
            
            pytorch_specific['lbfgs_state'] = state_info
        
        base_results['algorithm_specific'].update(pytorch_specific)
        return base_results


def create_pytorch_lbfgs_optimizer(learning_rate: float = 1.0,
                                 loss_type: str = 'ols',
                                 regularization: float = 0.01,
                                 max_iterations: int = 100000,
                                 convergence_tolerance: float = 1e-3,
                                 random_state: Optional[int] = None,
                                 **kwargs) -> PyTorchLBFGSWrapper:
    """
    Factory function to create PyTorch L-BFGS optimizer.
    
    Args:
        learning_rate: Learning rate for L-BFGS
        loss_type: Type of loss function ('ols', 'ridge', 'lasso')
        regularization: Regularization parameter
        max_iterations: Maximum number of iterations
        convergence_tolerance: Convergence tolerance
        random_state: Random seed
        **kwargs: Additional parameters
        
    Returns:
        PyTorchLBFGSWrapper instance
    """
    return PyTorchLBFGSWrapper(
        learning_rate=learning_rate,
        loss_type=loss_type,
        regularization=regularization,
        max_iterations=max_iterations,
        convergence_tolerance=convergence_tolerance,
        random_state=random_state,
        **kwargs
    )