"""
PyTorch SGD wrapper for gradient descent comparison.
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


class PyTorchSGDWrapper(BaseLibraryWrapper):
    """
    Wrapper for PyTorch SGD optimizer.
    
    Provides gradient descent functionality using PyTorch's implementation
    with support for momentum, weight decay, and various configurations.
    """
    
    def __init__(self,
                 learning_rate: float = 0.001,
                 loss_type: str = 'ols',
                 regularization: float = 0.01,
                 max_iterations: int = 100000,
                 convergence_tolerance: float = 1e-3,
                 random_state: Optional[int] = None,
                 # PyTorch SGD-specific parameters
                 momentum: float = 0.0,
                 dampening: float = 0.0,
                 weight_decay: float = 0.0,
                 nesterov: bool = False,
                 batch_size: Optional[int] = None,
                 **kwargs):
        """
        Initialize PyTorch SGD wrapper.
        
        Args:
            learning_rate: Learning rate for SGD
            loss_type: Type of loss function ('ols', 'ridge', 'lasso')
            regularization: Regularization parameter
            max_iterations: Maximum number of iterations
            convergence_tolerance: Convergence tolerance
            random_state: Random seed
            momentum: Momentum factor
            dampening: Dampening for momentum
            weight_decay: Weight decay (L2 penalty)
            nesterov: Enable Nesterov momentum
            batch_size: Batch size for mini-batch SGD (None for full batch)
            **kwargs: Additional parameters
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PyTorchSGDWrapper")
        
        # Store PyTorch-specific parameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.batch_size = batch_size
        self.torch_kwargs = kwargs
        
        super().__init__(
            library_name='pytorch',
            algorithm_name='SGD',
            loss_type=loss_type,
            regularization=regularization,
            max_iterations=max_iterations,
            convergence_tolerance=convergence_tolerance,
            random_state=random_state,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            batch_size=batch_size,
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
                # Create linear layer without bias (we handle bias manually in weights)
                self.linear = nn.Linear(n_features, 1, bias=False)
                
                # Initialize weights to match our initialization
                with torch.no_grad():
                    # Use the same initialization as our base optimizer
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
            # Use MSE loss and add regularization in optimization loop
            return nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
    
    def _create_external_optimizer(self, n_features: int) -> optim.SGD:
        """
        Create PyTorch SGD optimizer.
        
        Args:
            n_features: Number of features (including bias)
            
        Returns:
            PyTorch SGD optimizer
        """
        # Create model and loss function
        self.model = self._create_pytorch_model(n_features)
        self.criterion = self._create_loss_function()
        
        # Set random seed for PyTorch
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        
        # Create SGD optimizer
        sgd_params = {
            'lr': self.learning_rate,
            'momentum': self.momentum,
            'dampening': self.dampening,
            'weight_decay': self.weight_decay if self.loss_type == 'ridge' else 0.0,
            'nesterov': self.nesterov
        }
        
        # Add any additional parameters
        sgd_params.update(self.torch_kwargs)
        
        return optim.SGD(self.model.parameters(), **sgd_params)
    
    def _optimize_external(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Run PyTorch SGD optimization.
        
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
            
            n_samples = X.shape[0]
            
            # Determine batch size
            if self.batch_size is None or self.batch_size >= n_samples:
                batch_size = n_samples  # Full batch
                n_batches = 1
            else:
                batch_size = self.batch_size
                n_batches = (n_samples + batch_size - 1) // batch_size
            
            converged = False
            iteration = 0
            
            # Training loop
            for epoch in range(self.max_iterations):
                epoch_loss = 0.0
                
                # Shuffle data for each epoch if using mini-batches
                if n_batches > 1:
                    indices = torch.randperm(n_samples)
                    X_shuffled = X_tensor[indices]
                    y_shuffled = y_tensor[indices]
                else:
                    X_shuffled = X_tensor
                    y_shuffled = y_tensor
                
                # Mini-batch training
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, n_samples)
                    
                    X_batch = X_shuffled[start_idx:end_idx]
                    y_batch = y_shuffled[start_idx:end_idx]
                    
                    # Zero gradients
                    self.external_optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)
                    
                    # Add regularization for Lasso (L1)
                    if self.loss_type == 'lasso' and self.regularization > 0:
                        l1_penalty = self.regularization * torch.norm(self.model.linear.weight, p=1)
                        loss = loss + l1_penalty
                    
                    # Backward pass
                    loss.backward()
                    
                    # Update weights
                    self.external_optimizer.step()\n                    \n                    epoch_loss += loss.item()\n                    iteration += 1\n                \n                # Check convergence every epoch\n                if epoch % self.convergence_check_freq == 0 or epoch == self.max_iterations - 1:\n                    # Get current weights\n                    current_weights = self.model.linear.weight.data.cpu().numpy().flatten()\n                    \n                    # Compute gradient for convergence check\n                    self.external_optimizer.zero_grad()\n                    outputs = self.model(X_tensor)\n                    loss = self.criterion(outputs, y_tensor)\n                    \n                    # Add regularization to loss\n                    if self.loss_type == 'lasso' and self.regularization > 0:\n                        l1_penalty = self.regularization * torch.norm(self.model.linear.weight, p=1)\n                        loss = loss + l1_penalty\n                    \n                    loss.backward()\n                    \n                    # Get gradient norm\n                    grad_norm = torch.norm(self.model.linear.weight.grad).item()\n                    \n                    # Store in callback history\n                    self.callback_history['losses'].append(float(loss.item()))\n                    self.callback_history['gradient_norms'].append(float(grad_norm))\n                    self.callback_history['weights'].append(current_weights.copy())\n                    self.callback_history['iterations'].append(epoch)\n                    \n                    # Check convergence\n                    if grad_norm < self.convergence_tolerance:\n                        converged = True\n                        print(f\"PyTorch SGD converged at epoch {epoch} (gradient norm: {grad_norm:.6f})\")\n                        break\n            \n            # Extract final weights\n            final_weights = self.model.linear.weight.data.cpu().numpy().flatten()\n            \n            return {\n                'final_weights': final_weights,\n                'converged': converged,\n                'iterations': epoch + 1,\n                'final_loss': float(epoch_loss / n_batches),\n                'batch_size_used': batch_size,\n                'n_batches': n_batches\n            }\n            \n        except Exception as e:\n            print(f\"PyTorch SGD optimization failed: {str(e)}\")\n            return {\n                'final_weights': self.weights,\n                'converged': False,\n                'iterations': 0,\n                'error': str(e)\n            }\n    \n    def _get_algorithm_specific_results(self) -> Dict[str, Any]:\n        \"\"\"\n        Get PyTorch SGD-specific results.\n        \n        Returns:\n            Dictionary containing PyTorch-specific results\n        \"\"\"\n        base_results = super()._get_algorithm_specific_results()\n        \n        pytorch_specific = {\n            'pytorch_sgd_params': {\n                'learning_rate': self.learning_rate,\n                'momentum': self.momentum,\n                'dampening': self.dampening,\n                'weight_decay': self.weight_decay,\n                'nesterov': self.nesterov,\n                'batch_size': self.batch_size,\n                'device': str(self.device)\n            }\n        }\n        \n        # Add model information if available\n        if self.model is not None:\n            pytorch_specific['pytorch_model'] = {\n                'model_type': type(self.model).__name__,\n                'n_parameters': sum(p.numel() for p in self.model.parameters()),\n                'device': str(next(self.model.parameters()).device)\n            }\n        \n        base_results['algorithm_specific'].update(pytorch_specific)\n        return base_results\n\n\ndef create_pytorch_sgd_optimizer(learning_rate: float = 0.001,\n                                loss_type: str = 'ols',\n                                regularization: float = 0.01,\n                                max_iterations: int = 100000,\n                                random_state: Optional[int] = None,\n                                **kwargs) -> PyTorchSGDWrapper:\n    \"\"\"\n    Factory function to create PyTorch SGD optimizer.\n    \n    Args:\n        learning_rate: Learning rate for SGD\n        loss_type: Type of loss function ('ols', 'ridge', 'lasso')\n        regularization: Regularization parameter\n        max_iterations: Maximum number of iterations\n        random_state: Random seed\n        **kwargs: Additional parameters\n        \n    Returns:\n        PyTorchSGDWrapper instance\n    \"\"\"\n    return PyTorchSGDWrapper(\n        learning_rate=learning_rate,\n        loss_type=loss_type,\n        regularization=regularization,\n        max_iterations=max_iterations,\n        random_state=random_state,\n        **kwargs\n    )