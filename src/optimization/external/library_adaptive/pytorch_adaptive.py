"""
PyTorch Adaptive Optimization algorithms wrapper.
Implements modern adaptive optimizers from PyTorch including Adam, AdamW, RMSprop, etc.
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


class PyTorchAdaptive(BaseLibraryWrapper):
    """
    Wrapper for PyTorch adaptive optimization algorithms.
    Provides modern adaptive optimizers like Adam, AdamW, RMSprop, Adagrad, etc.
    """
    
    def __init__(self,
                 optimizer_type: str = 'adam',  # 'adam', 'adamw', 'rmsprop', 'adagrad', 'adadelta', 'adamax'
                 learning_rate: float = 0.001,
                 betas: tuple = (0.9, 0.999),      # For Adam-like optimizers
                 alpha: float = 0.99,              # For RMSprop
                 eps: float = 1e-8,
                 weight_decay: float = 0.01,
                 amsgrad: bool = False,            # For Adam variants
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
        Initialize PyTorch adaptive optimizer wrapper.
        
        Args:
            optimizer_type: Type of adaptive optimizer
            learning_rate: Learning rate
            betas: Coefficients for computing running averages (Adam-like)
            alpha: Smoothing constant for RMSprop
            eps: Term for numerical stability
            weight_decay: Weight decay (L2 penalty)
            amsgrad: Use AMSGrad variant
            batch_size: Batch size for mini-batch optimization
            shuffle: Shuffle data for each epoch
            loss_type: Loss function type
            regularization: Regularization parameter
            convergence_tolerance: Convergence tolerance
            max_iterations: Maximum iterations (epochs)
            random_state: Random seed
            device: PyTorch device ('cpu' or 'cuda')
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install PyTorch to use PyTorchAdaptive.")
        
        super().__init__(
            library_name='pytorch',
            algorithm_name=optimizer_type.upper(),
            loss_type=loss_type,
            regularization=regularization,
            convergence_tolerance=convergence_tolerance,
            max_iterations=max_iterations,
            random_state=random_state,
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            betas=betas,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            batch_size=batch_size,
            shuffle=shuffle,
            device=device,
            **kwargs
        )
        
        self.optimizer_type = optimizer_type.lower()
        self.learning_rate = learning_rate
        self.betas = betas
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = max(weight_decay, regularization)  # Use larger regularization
        self.amsgrad = amsgrad
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
        if self.loss_type in ['ols', 'ridge']:
            return nn.MSELoss()
        elif self.loss_type == 'huber':
            return nn.HuberLoss()
        elif self.loss_type == 'mae':
            return nn.L1Loss()
        else:
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
        Create PyTorch adaptive optimizer.
        
        Args:
            n_features: Number of features (including bias)
            
        Returns:
            PyTorch adaptive optimizer
        """
        # Create model and criterion
        self.model = self._create_pytorch_model(n_features)
        self.criterion = self._create_pytorch_criterion()
        
        # Create optimizer based on type
        if self.optimizer_type == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=self.betas,
                eps=self.eps,
                weight_decay=self.weight_decay,
                amsgrad=self.amsgrad
            )
        elif self.optimizer_type == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=self.betas,
                eps=self.eps,
                weight_decay=self.weight_decay,
                amsgrad=self.amsgrad
            )
        elif self.optimizer_type == 'rmsprop':
            optimizer = optim.RMSprop(
                self.model.parameters(),
                lr=self.learning_rate,
                alpha=self.alpha,
                eps=self.eps,
                weight_decay=self.weight_decay,
                momentum=0.0  # Can be made configurable
            )
        elif self.optimizer_type == 'adagrad':
            optimizer = optim.Adagrad(
                self.model.parameters(),
                lr=self.learning_rate,
                lr_decay=0.0,
                weight_decay=self.weight_decay,
                eps=self.eps
            )
        elif self.optimizer_type == 'adadelta':
            optimizer = optim.Adadelta(
                self.model.parameters(),
                lr=self.learning_rate,
                rho=0.9,
                eps=self.eps,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == 'adamax':
            optimizer = optim.Adamax(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=self.betas,
                eps=self.eps,
                weight_decay=self.weight_decay
            )
        else:
            # Default to Adam
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=self.betas,
                eps=self.eps,
                weight_decay=self.weight_decay
            )
        
        return optimizer
    
    def _optimize_external(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Run PyTorch adaptive optimization.
        
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
        patience = 15  # Adaptive optimizers may need more patience
        
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
            
            # Store in callback history
            self.callback_history['losses'].append(avg_epoch_loss)
            self.callback_history['gradient_norms'].append(avg_grad_norm)
            
            # Extract current weights
            current_weights = self._extract_pytorch_weights()
            self.callback_history['weights'].append(current_weights.copy())
            self.callback_history['iterations'].append(epoch)
            
            # Check convergence
            if avg_grad_norm < self.convergence_tolerance:
                print(f\"[CONVERGENCE] PyTorch {self.optimizer_type.upper()} converged at epoch {epoch + 1}\")\
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
                print(f\"[EARLY_STOP] PyTorch {self.optimizer_type.upper()} early stopping at epoch {epoch + 1}\")\
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
        Get PyTorch adaptive optimizer specific results.\
        \
        Returns:\
            Dictionary containing PyTorch-specific results\
        \"\"\"\
        base_results = super()._get_algorithm_specific_results()\
        \
        pytorch_specific = {\
            'pytorch_adaptive_specific': {\
                'optimizer_type': self.optimizer_type,\
                'learning_rate': self.learning_rate,\
                'betas': self.betas,\
                'alpha': self.alpha,\
                'eps': self.eps,\
                'weight_decay': self.weight_decay,\
                'amsgrad': self.amsgrad,\
                'batch_size': self.batch_size,\
                'shuffle': self.shuffle,\
                'device': str(self.device),\
                'model_type': 'Linear',\
                'criterion_type': type(self.criterion).__name__ if self.criterion else 'Unknown',\
                'total_epochs': len(self.epoch_losses),\
                'optimizer_class': type(self.external_optimizer).__name__ if self.external_optimizer else 'Unknown'\
            }\
        }\
        \
        base_results.update(pytorch_specific)\
        return base_results\
\
\
# Convenience functions for specific adaptive optimizers\
def create_pytorch_adam(learning_rate: float = 0.001, betas: tuple = (0.9, 0.999), **kwargs) -> PyTorchAdaptive:\
    \"\"\"Create Adam optimizer.\"\"\"\
    return PyTorchAdaptive(\
        optimizer_type='adam',\
        learning_rate=learning_rate,\
        betas=betas,\
        **kwargs\
    )\
\
def create_pytorch_adamw(learning_rate: float = 0.001, weight_decay: float = 0.01, **kwargs) -> PyTorchAdaptive:\
    \"\"\"Create AdamW optimizer.\"\"\"\
    return PyTorchAdaptive(\
        optimizer_type='adamw',\
        learning_rate=learning_rate,\
        weight_decay=weight_decay,\
        **kwargs\
    )\
\
def create_pytorch_rmsprop(learning_rate: float = 0.01, alpha: float = 0.99, **kwargs) -> PyTorchAdaptive:\
    \"\"\"Create RMSprop optimizer.\"\"\"\
    return PyTorchAdaptive(\
        optimizer_type='rmsprop',\
        learning_rate=learning_rate,\
        alpha=alpha,\
        **kwargs\
    )\
\
def create_pytorch_adagrad(learning_rate: float = 0.01, **kwargs) -> PyTorchAdaptive:\
    \"\"\"Create Adagrad optimizer.\"\"\"\
    return PyTorchAdaptive(\
        optimizer_type='adagrad',\
        learning_rate=learning_rate,\
        **kwargs\
    )\
\
def create_pytorch_adadelta(learning_rate: float = 1.0, **kwargs) -> PyTorchAdaptive:\
    \"\"\"Create Adadelta optimizer.\"\"\"\
    return PyTorchAdaptive(\
        optimizer_type='adadelta',\
        learning_rate=learning_rate,\
        **kwargs\
    )\
\
def create_pytorch_adamax(learning_rate: float = 0.002, **kwargs) -> PyTorchAdaptive:\
    \"\"\"Create Adamax optimizer.\"\"\"\
    return PyTorchAdaptive(\
        optimizer_type='adamax',\
        learning_rate=learning_rate,\
        **kwargs\
    )