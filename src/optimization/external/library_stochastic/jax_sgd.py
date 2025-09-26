"""
JAX Stochastic Gradient Descent wrapper.
Implements SGD variants using JAX and Optax for stochastic optimization.
"""

import numpy as np
from typing import Dict, Any, Optional, Callable, Tuple
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
sys.path.append(project_root)

from src.optimization.external.base_library_wrapper import BaseLibraryWrapper

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap, random
    import optax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class JAXSGDVariants(BaseLibraryWrapper):
    """
    Wrapper for JAX/Optax SGD variants.
    Provides various SGD optimizations using JAX's JIT compilation and Optax optimizers.
    """
    
    def __init__(self,
                 learning_rate: float = 0.01,
                 momentum: float = 0.0,
                 nesterov: bool = False,
                 weight_decay: float = 0.0,
                 batch_size: int = 32,
                 optimizer_type: str = 'sgd',  # 'sgd', 'momentum', 'nesterov_momentum'
                 loss_type: str = 'ols',
                 regularization: float = 0.01,
                 convergence_tolerance: float = 1e-3,
                 max_iterations: int = 100000,
                 random_state: Optional[int] = None,
                 **kwargs):
        """
        Initialize JAX SGD variants wrapper.
        
        Args:
            learning_rate: Learning rate
            momentum: Momentum factor (0 = no momentum)
            nesterov: Enable Nesterov momentum
            weight_decay: Weight decay (L2 penalty)
            batch_size: Batch size for mini-batch SGD
            optimizer_type: Type of SGD optimizer
            loss_type: Loss function type
            regularization: Regularization parameter
            convergence_tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            random_state: Random seed
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is not available. Please install JAX and Optax to use JAXSGDVariants.")
        
        super().__init__(
            library_name='jax',
            algorithm_name=f'SGD_{optimizer_type}',
            loss_type=loss_type,
            regularization=regularization,
            convergence_tolerance=convergence_tolerance,
            max_iterations=max_iterations,
            random_state=random_state,
            learning_rate=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            batch_size=batch_size,
            optimizer_type=optimizer_type,
            **kwargs
        )
        
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = max(weight_decay, regularization)  # Use larger regularization
        self.batch_size = batch_size
        self.optimizer_type = optimizer_type
        
        # JAX components
        self.optimizer = None
        self.opt_state = None
        self.loss_fn = None
        self.grad_fn = None
        self.update_fn = None
        
        # Set random seed for JAX
        self.key = random.PRNGKey(random_state if random_state is not None else 42)
    
    def _create_loss_function(self) -> Callable:
        """
        Create JAX loss function.
        
        Returns:
            JAX loss function
        """
        def loss_fn(params: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray) -> float:
            # Linear prediction: X @ params[:-1] + params[-1] (bias)
            pred = jnp.dot(X, params[:-1]) + params[-1]
            
            # Base loss
            if self.loss_type in ['ols', 'ridge']:
                base_loss = jnp.mean((pred - y) ** 2)
            elif self.loss_type == 'mae':
                base_loss = jnp.mean(jnp.abs(pred - y))
            elif self.loss_type == 'huber':
                delta = 1.0
                residual = pred - y
                huber_loss = jnp.where(
                    jnp.abs(residual) <= delta,
                    0.5 * residual ** 2,
                    delta * jnp.abs(residual) - 0.5 * delta ** 2
                )
                base_loss = jnp.mean(huber_loss)
            else:
                base_loss = jnp.mean((pred - y) ** 2)  # Default to MSE
            
            # Add regularization
            if self.weight_decay > 0:
                if self.loss_type == 'ridge':
                    l2_penalty = self.weight_decay * jnp.sum(params[:-1] ** 2)  # Don't regularize bias
                elif self.loss_type == 'lasso':
                    l1_penalty = self.weight_decay * jnp.sum(jnp.abs(params[:-1]))
                    return base_loss + l1_penalty
                else:
                    l2_penalty = self.weight_decay * jnp.sum(params[:-1] ** 2)
                
                return base_loss + l2_penalty
            
            return base_loss
        
        return loss_fn
    
    def _create_batch_generator(self, X: np.ndarray, y: np.ndarray):
        """
        Create batch generator for mini-batch SGD.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Yields:
            Batches of (X_batch, y_batch)
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        # Shuffle indices
        np.random.shuffle(indices)
        
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            yield X[batch_indices], y[batch_indices]
    
    @jit
    def _update_step(self, params: jnp.ndarray, opt_state, X_batch: jnp.ndarray, y_batch: jnp.ndarray) -> Tuple[jnp.ndarray, Any, float]:
        """
        JIT-compiled update step.
        
        Args:
            params: Model parameters
            opt_state: Optimizer state
            X_batch: Batch features
            y_batch: Batch targets
            
        Returns:
            Updated parameters, optimizer state, and loss value
        """
        loss_value = self.loss_fn(params, X_batch, y_batch)
        grads = self.grad_fn(params, X_batch, y_batch)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value
    
    def _create_external_optimizer(self, n_features: int) -> optax.GradientTransformation:
        """
        Create JAX/Optax SGD optimizer.
        
        Args:
            n_features: Number of features (including bias)
            
        Returns:
            Optax optimizer
        """
        # Create loss and gradient functions
        self.loss_fn = self._create_loss_function()
        self.grad_fn = jit(grad(self.loss_fn))
        
        # Create optimizer based on type
        if self.optimizer_type == 'sgd':
            optimizer = optax.sgd(learning_rate=self.learning_rate)
        elif self.optimizer_type == 'momentum':
            optimizer = optax.sgd(learning_rate=self.learning_rate, momentum=self.momentum)
        elif self.optimizer_type == 'nesterov_momentum':
            optimizer = optax.sgd(learning_rate=self.learning_rate, momentum=self.momentum, nesterov=True)
        else:
            # Default to SGD
            optimizer = optax.sgd(learning_rate=self.learning_rate)
        
        # Add weight decay if specified
        if self.weight_decay > 0:
            optimizer = optax.chain(
                optax.add_decayed_weights(weight_decay=self.weight_decay),
                optimizer
            )
        
        return optimizer
    
    def _optimize_external(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Run JAX SGD optimization.
        
        Args:
            X: Feature matrix (with bias)
            y: Target vector
            
        Returns:
            Dictionary containing optimization results
        """
        # Remove bias column (we handle bias explicitly in loss function)
        X_no_bias = X[:, :-1]
        
        # Convert to JAX arrays
        X_jax = jnp.array(X_no_bias)
        y_jax = jnp.array(y)
        
        # Initialize parameters
        self.key, init_key = random.split(self.key)
        params = random.normal(init_key, (X_jax.shape[1] + 1,)) * 0.01  # +1 for bias
        
        # Initialize optimizer state
        self.opt_state = self.optimizer.init(params)
        
        # Training loop
        epoch_losses = []
        epoch_gradient_norms = []
        
        best_loss = float('inf')
        no_improve_count = 0
        patience = 10
        
        n_epochs = min(self.max_iterations, 1000)  # Reasonable upper bound
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            epoch_grad_norm = 0.0
            batch_count = 0
            
            # Mini-batch SGD
            for X_batch, y_batch in self._create_batch_generator(X_no_bias, y):
                X_batch_jax = jnp.array(X_batch)
                y_batch_jax = jnp.array(y_batch)
                
                # Update parameters
                params, self.opt_state, batch_loss = self._update_step(
                    params, self.opt_state, X_batch_jax, y_batch_jax
                )
                
                # Compute gradient norm for monitoring
                grads = self.grad_fn(params, X_batch_jax, y_batch_jax)
                grad_norm = jnp.linalg.norm(grads)
                
                epoch_loss += float(batch_loss)
                epoch_grad_norm += float(grad_norm)
                batch_count += 1
                
                # Track complexity
                self.track_function_evaluation(X_batch.shape)
                self.track_gradient_evaluation(X_batch.shape)
            
            # Average metrics for epoch
            avg_epoch_loss = epoch_loss / max(batch_count, 1)
            avg_grad_norm = epoch_grad_norm / max(batch_count, 1)
            
            epoch_losses.append(avg_epoch_loss)
            epoch_gradient_norms.append(avg_grad_norm)
            
            # Store in callback history
            self.callback_history['losses'].append(avg_epoch_loss)
            self.callback_history['gradient_norms'].append(avg_grad_norm)
            self.callback_history['weights'].append(np.array(params).copy())
            self.callback_history['iterations'].append(epoch)
            
            # Check convergence
            if avg_grad_norm < self.convergence_tolerance:
                print(f\"[CONVERGENCE] JAX SGD converged at epoch {epoch + 1}\")\
                return {\
                    'final_weights': np.array(params),\
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
                print(f\"[EARLY_STOP] JAX SGD early stopping at epoch {epoch + 1}\")\
                break\
        \
        return {\
            'final_weights': np.array(params),\
            'iterations': len(epoch_losses),\
            'converged': False,\
            'final_loss': epoch_losses[-1] if epoch_losses else float('inf'),\
            'final_gradient_norm': epoch_gradient_norms[-1] if epoch_gradient_norms else float('inf')\
        }\
    \
    def _get_algorithm_specific_results(self) -> Dict[str, Any]:\
        \"\"\"\
        Get JAX SGD specific results.\
        \
        Returns:\
            Dictionary containing JAX-specific results\
        \"\"\"\
        base_results = super()._get_algorithm_specific_results()\
        \
        jax_specific = {\
            'jax_sgd_specific': {\
                'learning_rate': self.learning_rate,\
                'momentum': self.momentum,\
                'nesterov': self.nesterov,\
                'weight_decay': self.weight_decay,\
                'batch_size': self.batch_size,\
                'optimizer_type': self.optimizer_type,\
                'jit_compilation': True,\
                'optax_optimizer': type(self.optimizer).__name__ if self.optimizer else 'Unknown',\
                'parameter_count': len(self.callback_history['weights'][-1]) if self.callback_history['weights'] else 0\
            }\
        }\
        \
        base_results.update(jax_specific)\
        return base_results\
\
\
# Convenience functions for specific SGD configurations\
def create_jax_sgd_vanilla(learning_rate: float = 0.01, batch_size: int = 32, **kwargs) -> JAXSGDVariants:\
    \"\"\"Create vanilla SGD.\"\"\"\
    return JAXSGDVariants(\
        learning_rate=learning_rate,\
        momentum=0.0,\
        optimizer_type='sgd',\
        batch_size=batch_size,\
        **kwargs\
    )\
\
def create_jax_sgd_momentum(learning_rate: float = 0.01, momentum: float = 0.9, batch_size: int = 32, **kwargs) -> JAXSGDVariants:\
    \"\"\"Create SGD with momentum.\"\"\"\
    return JAXSGDVariants(\
        learning_rate=learning_rate,\
        momentum=momentum,\
        optimizer_type='momentum',\
        batch_size=batch_size,\
        **kwargs\
    )\
\
def create_jax_sgd_nesterov(learning_rate: float = 0.01, momentum: float = 0.9, batch_size: int = 32, **kwargs) -> JAXSGDVariants:\
    \"\"\"Create SGD with Nesterov momentum.\"\"\"\
    return JAXSGDVariants(\
        learning_rate=learning_rate,\
        momentum=momentum,\
        nesterov=True,\
        optimizer_type='nesterov_momentum',\
        batch_size=batch_size,\
        **kwargs\
    )\
\
def create_jax_sgd_weight_decay(learning_rate: float = 0.01, weight_decay: float = 0.01, batch_size: int = 32, **kwargs) -> JAXSGDVariants:\
    \"\"\"Create SGD with weight decay.\"\"\"\
    return JAXSGDVariants(\
        learning_rate=learning_rate,\
        weight_decay=weight_decay,\
        optimizer_type='sgd',\
        batch_size=batch_size,\
        **kwargs\
    )