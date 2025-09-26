"""
PyTorch Stochastic Gradient Descent variants wrapper.
Implements multiple SGD configurations from PyTorch for stochastic optimization.
"""

import numpy as np
from typing import Dict, Any, Optional, List
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
sys.path.append(project_root)

from src.optimization.external.base_library_wrapper import BaseLibraryWrapper

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


class PyTorchSGDVariants(BaseLibraryWrapper):
    """
    Wrapper for PyTorch SGD variants with different configurations.
    Provides various SGD optimizations including momentum, Nesterov, weight decay.
    """
    
    def __init__(self,
                 learning_rate: float = 0.01,
                 momentum: float = 0.0,
                 dampening: float = 0.0,
                 weight_decay: float = 0.0,
                 nesterov: bool = False,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 loss_type: str = 'ols',
                 regularization: float = 0.01,
                 convergence_tolerance: float = 1e-3,
                 max_iterations: int = 100000,
                 random_state: Optional[int] = None,
                 device: str = 'cpu',
                 **kwargs):
        """
        Initialize PyTorch SGD variants wrapper.
        
        Args:
            learning_rate: Learning rate
            momentum: Momentum factor (0 = no momentum)
            dampening: Dampening for momentum
            weight_decay: Weight decay (L2 penalty)
            nesterov: Enable Nesterov momentum
            batch_size: Batch size for mini-batch SGD
            shuffle: Shuffle data for each epoch
            loss_type: Loss function type
            regularization: Regularization parameter
            convergence_tolerance: Convergence tolerance
            max_iterations: Maximum iterations (epochs)
            random_state: Random seed
            device: PyTorch device ('cpu' or 'cuda')
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install PyTorch to use PyTorchSGDVariants.")
        
        super().__init__(
            library_name='pytorch',
            algorithm_name='SGD',
            loss_type=loss_type,
            regularization=regularization,
            convergence_tolerance=convergence_tolerance,
            max_iterations=max_iterations,
            random_state=random_state,
            learning_rate=learning_rate,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            batch_size=batch_size,
            shuffle=shuffle,
            device=device,
            **kwargs
        )
        
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = max(weight_decay, regularization)  # Use larger regularization
        self.nesterov = nesterov
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = torch.device(device)
        
        # PyTorch model and components
        self.model = None
        self.criterion = None
        self.data_loader = None
        
        # Training history
        self.epoch_losses = []
        self.epoch_gradient_norms = []
        
    def _create_pytorch_model(self, n_features: int) -> nn.Module:
        """
        Create a simple linear model for PyTorch.
        
        Args:
            n_features: Number of features (including bias)
            
        Returns:
            PyTorch linear model
        """
        # Create simple linear regression model (n_features-1 because bias is handled by PyTorch)
        model = nn.Linear(n_features - 1, 1, bias=True)
        model.to(self.device)
        
        # Initialize weights consistently
        torch.manual_seed(42)
        nn.init.normal_(model.weight, mean=0, std=0.01)
        nn.init.normal_(model.bias, mean=0, std=0.01)
        
        return model
    
    def _create_pytorch_criterion(self) -> nn.Module:
        """
        Create PyTorch loss function.
        
        Returns:
            PyTorch loss criterion
        """
        if self.loss_type in ['ols', 'ridge']:  # Ridge regularization handled by optimizer
            return nn.MSELoss()
        elif self.loss_type == 'huber':
            return nn.HuberLoss()
        else:
            # Default to MSE for unknown loss types
            return nn.MSELoss()
    
    def _create_data_loader(self, X: np.ndarray, y: np.ndarray) -> DataLoader:
        """
        Create PyTorch DataLoader.
        
        Args:
            X: Feature matrix (with bias)
            y: Target vector
            
        Returns:
            PyTorch DataLoader
        """
        # Remove bias column (PyTorch handles bias internally)
        X_no_bias = X[:, :-1]
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_no_bias).to(self.device)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(self.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        data_loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=self.shuffle,
            generator=torch.Generator().manual_seed(self.random_state) if self.random_state else None
        )
        
        return data_loader
    
    def _create_external_optimizer(self, n_features: int) -> optim.Optimizer:
        """
        Create PyTorch SGD optimizer.
        
        Args:
            n_features: Number of features (including bias)
            
        Returns:
            PyTorch SGD optimizer
        """
        # Create model and criterion
        self.model = self._create_pytorch_model(n_features)
        self.criterion = self._create_pytorch_criterion()
        
        # Create SGD optimizer
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            dampening=self.dampening,
            weight_decay=self.weight_decay,
            nesterov=self.nesterov
        )
        
        return optimizer
    
    def _optimize_external(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Run PyTorch SGD optimization.
        
        Args:
            X: Feature matrix (with bias)
            y: Target vector
            
        Returns:
            Dictionary containing optimization results
        """
        # Create data loader
        self.data_loader = self._create_data_loader(X, y)
        
        # Training loop
        self.model.train()
        self.epoch_losses = []
        self.epoch_gradient_norms = []
        
        best_loss = float('inf')
        no_improve_count = 0
        patience = 10
        
        for epoch in range(self.max_iterations):
            epoch_loss = 0.0
            batch_count = 0
            total_grad_norm = 0.0
            
            for batch_X, batch_y in self.data_loader:
                # Zero gradients
                self.external_optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                
                # Calculate gradient norm
                total_norm = 0.0
                for param in self.model.parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                grad_norm = total_norm ** 0.5
                total_grad_norm += grad_norm
                
                # Update parameters
                self.external_optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                
                # Track complexity
                self.track_function_evaluation(batch_X.shape)
                self.track_gradient_evaluation(batch_X.shape)
            
            # Average loss and gradient norm for epoch
            avg_epoch_loss = epoch_loss / batch_count
            avg_grad_norm = total_grad_norm / batch_count
            
            self.epoch_losses.append(avg_epoch_loss)
            self.epoch_gradient_norms.append(avg_grad_norm)
            
            # Store in callback history for consistency
            self.callback_history['losses'].append(avg_epoch_loss)
            self.callback_history['gradient_norms'].append(avg_grad_norm)
            
            # Extract current weights
            current_weights = self._extract_pytorch_weights()
            self.callback_history['weights'].append(current_weights.copy())
            self.callback_history['iterations'].append(epoch)
            
            # Check convergence
            if avg_grad_norm < self.convergence_tolerance:
                print(f\"[CONVERGENCE] PyTorch SGD converged at epoch {epoch + 1}\")\
                return {\
                    'final_weights': current_weights,\
                    'iterations': epoch + 1,\
                    'converged': True,\
                    'final_loss': avg_epoch_loss,\
                    'final_gradient_norm': avg_grad_norm\
                }\
            \
            # Early stopping based on loss improvement\
            if avg_epoch_loss < best_loss - self.convergence_tolerance:\
                best_loss = avg_epoch_loss\
                no_improve_count = 0\
            else:\
                no_improve_count += 1\
                \
            if no_improve_count >= patience:\
                print(f\"[EARLY_STOP] PyTorch SGD early stopping at epoch {epoch + 1}\")\
                break\
        \
        # Final weights\
        final_weights = self._extract_pytorch_weights()\
        \
        return {\
            'final_weights': final_weights,\
            'iterations': len(self.epoch_losses),\
            'converged': False,\
            'final_loss': self.epoch_losses[-1] if self.epoch_losses else float('inf'),\
            'final_gradient_norm': self.epoch_gradient_norms[-1] if self.epoch_gradient_norms else float('inf')\
        }\
    \
    def _extract_pytorch_weights(self) -> np.ndarray:\
        \"\"\"\
        Extract weights from PyTorch model.\
        \
        Returns:\
            Numpy array of weights including bias\
        \"\"\"\
        with torch.no_grad():\
            # Get weight and bias from the linear layer\
            weight = self.model.weight.cpu().numpy().flatten()\
            bias = self.model.bias.cpu().numpy().flatten()\
            \
            # Combine weight and bias (bias goes last to match our convention)\
            return np.concatenate([weight, bias])\
    \
    def _get_algorithm_specific_results(self) -> Dict[str, Any]:\
        \"\"\"\
        Get PyTorch SGD specific results.\
        \
        Returns:\
            Dictionary containing PyTorch-specific results\
        \"\"\"\
        base_results = super()._get_algorithm_specific_results()\
        \
        pytorch_specific = {\
            'pytorch_sgd_specific': {\
                'learning_rate': self.learning_rate,\
                'momentum': self.momentum,\
                'dampening': self.dampening,\
                'weight_decay': self.weight_decay,\
                'nesterov': self.nesterov,\
                'batch_size': self.batch_size,\
                'shuffle': self.shuffle,\
                'device': str(self.device),\
                'model_type': 'Linear',\
                'criterion_type': type(self.criterion).__name__ if self.criterion else 'Unknown',\
                'total_epochs': len(self.epoch_losses),\
                'total_batches_processed': len(self.epoch_losses) * len(self.data_loader) if self.data_loader else 0\
            }\
        }\
        \
        base_results.update(pytorch_specific)\
        return base_results\
\
\
# Convenience functions for specific SGD configurations\
def create_pytorch_sgd_vanilla(learning_rate: float = 0.01, batch_size: int = 32, **kwargs) -> PyTorchSGDVariants:\
    \"\"\"Create vanilla SGD without momentum.\"\"\"\
    return PyTorchSGDVariants(\
        learning_rate=learning_rate,\
        momentum=0.0,\
        batch_size=batch_size,\
        **kwargs\
    )\
\
def create_pytorch_sgd_momentum(learning_rate: float = 0.01, momentum: float = 0.9, batch_size: int = 32, **kwargs) -> PyTorchSGDVariants:\
    \"\"\"Create SGD with momentum.\"\"\"\
    return PyTorchSGDVariants(\
        learning_rate=learning_rate,\
        momentum=momentum,\
        batch_size=batch_size,\
        **kwargs\
    )\
\
def create_pytorch_sgd_nesterov(learning_rate: float = 0.01, momentum: float = 0.9, batch_size: int = 32, **kwargs) -> PyTorchSGDVariants:\
    \"\"\"Create SGD with Nesterov momentum.\"\"\"\
    return PyTorchSGDVariants(\
        learning_rate=learning_rate,\
        momentum=momentum,\
        nesterov=True,\
        batch_size=batch_size,\
        **kwargs\
    )\
\
def create_pytorch_sgd_weight_decay(learning_rate: float = 0.01, weight_decay: float = 0.01, batch_size: int = 32, **kwargs) -> PyTorchSGDVariants:\
    \"\"\"Create SGD with weight decay (L2 regularization).\"\"\"\
    return PyTorchSGDVariants(\
        learning_rate=learning_rate,\
        weight_decay=weight_decay,\
        batch_size=batch_size,\
        **kwargs\
    )\
\
def create_pytorch_sgd_full_batch(learning_rate: float = 0.01, momentum: float = 0.0, **kwargs) -> PyTorchSGDVariants:\
    \"\"\"Create SGD with full batch (batch_size = dataset size).\"\"\"\
    return PyTorchSGDVariants(\
        learning_rate=learning_rate,\
        momentum=momentum,\
        batch_size=10000000,  # Large batch size to approximate full batch\
        shuffle=False,\
        **kwargs\
    )