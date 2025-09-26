"""
JAX Adaptive Optimization algorithms wrapper.
Implements modern adaptive optimizers using JAX and Optax including Adam, RMSprop, AdaGrad, etc.
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


class JAXAdaptive(BaseLibraryWrapper):
    """
    Wrapper for JAX/Optax adaptive optimization algorithms.
    Provides modern adaptive optimizers using JAX's JIT compilation and Optax optimizers.
    """
    
    def __init__(self,
                 optimizer_type: str = 'adam',  # 'adam', 'adamw', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam', 'yogi', 'lamb'
                 learning_rate: float = 0.001,
                 b1: float = 0.9,                  # Beta1 for Adam-like optimizers
                 b2: float = 0.999,                # Beta2 for Adam-like optimizers
                 eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 batch_size: int = 32,
                 # Adam-specific
                 eps_root: float = 0.0,            # AdamW epsilon_root
                 mu_dtype: Optional[str] = None,   # Momentum dtype
                 # RMSprop-specific
                 decay: float = 0.9,               # RMSprop decay
                 centered: bool = False,           # RMSprop centered
                 # AdaGrad-specific
                 initial_accumulator_value: float = 0.1,
                 # Advanced optimizers
                 nesterov: bool = False,
                 loss_type: str = 'ols',
                 regularization: float = 0.01,
                 convergence_tolerance: float = 1e-3,
                 max_iterations: int = 100000,
                 random_state: Optional[int] = None,
                 **kwargs):
        """
        Initialize JAX adaptive optimizer wrapper.
        
        Args:
            optimizer_type: Type of adaptive optimizer
            learning_rate: Learning rate
            b1: Exponential decay rate for first moment estimates
            b2: Exponential decay rate for second moment estimates
            eps: Small constant for numerical stability
            weight_decay: Weight decay (L2 penalty)
            batch_size: Batch size for mini-batch optimization
            eps_root: Epsilon for square root (AdamW)
            mu_dtype: Momentum dtype for precision control
            decay: Decay rate for RMSprop
            centered: Use centered RMSprop
            initial_accumulator_value: Initial accumulator for AdaGrad
            nesterov: Use Nesterov momentum where applicable
            loss_type: Loss function type
            regularization: Regularization parameter
            convergence_tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            random_state: Random seed
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is not available. Please install JAX and Optax to use JAXAdaptive.")
        
        super().__init__(
            library_name='jax',
            algorithm_name=optimizer_type.upper(),
            loss_type=loss_type,
            regularization=regularization,
            convergence_tolerance=convergence_tolerance,
            max_iterations=max_iterations,
            random_state=random_state,
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            b1=b1,
            b2=b2,
            eps=eps,
            weight_decay=weight_decay,
            batch_size=batch_size,
            eps_root=eps_root,
            mu_dtype=mu_dtype,
            decay=decay,
            centered=centered,
            initial_accumulator_value=initial_accumulator_value,
            nesterov=nesterov,
            **kwargs
        )
        
        self.optimizer_type = optimizer_type.lower()
        self.learning_rate = learning_rate
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.weight_decay = max(weight_decay, regularization)
        self.batch_size = batch_size
        self.eps_root = eps_root
        self.mu_dtype = mu_dtype
        self.decay = decay
        self.centered = centered
        self.initial_accumulator_value = initial_accumulator_value
        self.nesterov = nesterov
        
        # JAX components
        self.optimizer = None
        self.opt_state = None
        self.loss_fn = None
        self.grad_fn = None
        
        # Set random seed for JAX
        self.key = random.PRNGKey(random_state if random_state is not None else 42)
    
    def _create_loss_function(self) -> Callable:
        """
        Create JAX loss function.
        
        Returns:
            JAX loss function
        """
        def loss_fn(params: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray) -> float:
            # Linear prediction
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
                base_loss = jnp.mean((pred - y) ** 2)
            
            # Add regularization
            if self.weight_decay > 0:
                if self.loss_type == 'ridge':
                    l2_penalty = self.weight_decay * jnp.sum(params[:-1] ** 2)
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
        Create batch generator for mini-batch optimization.
        
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
        Create JAX/Optax adaptive optimizer.
        
        Args:
            n_features: Number of features (including bias)
            
        Returns:
            Optax optimizer
        """
        # Create loss and gradient functions
        self.loss_fn = self._create_loss_function()
        self.grad_fn = jit(grad(self.loss_fn))
        
        # Create optimizer based on type
        if self.optimizer_type == 'adam':
            optimizer = optax.adam(
                learning_rate=self.learning_rate,
                b1=self.b1,
                b2=self.b2,
                eps=self.eps,
                eps_root=self.eps_root,
                mu_dtype=self.mu_dtype
            )
        elif self.optimizer_type == 'adamw':
            optimizer = optax.adamw(
                learning_rate=self.learning_rate,
                b1=self.b1,
                b2=self.b2,
                eps=self.eps,
                eps_root=self.eps_root,
                weight_decay=self.weight_decay,
                mu_dtype=self.mu_dtype
            )
        elif self.optimizer_type == 'rmsprop':
            optimizer = optax.rmsprop(
                learning_rate=self.learning_rate,
                decay=self.decay,
                eps=self.eps,
                centered=self.centered
            )
        elif self.optimizer_type == 'adagrad':
            optimizer = optax.adagrad(
                learning_rate=self.learning_rate,
                initial_accumulator_value=self.initial_accumulator_value,
                eps=self.eps
            )
        elif self.optimizer_type == 'adadelta':
            optimizer = optax.adadelta(
                learning_rate=self.learning_rate,
                rho=self.decay,  # Using decay as rho
                eps=self.eps
            )
        elif self.optimizer_type == 'adamax':
            # Adamax is not directly available in Optax, use Adam as approximation
            optimizer = optax.adam(
                learning_rate=self.learning_rate,
                b1=self.b1,
                b2=0.999,  # Fixed beta2 for Adamax-like behavior
                eps=self.eps
            )
        elif self.optimizer_type == 'nadam':
            # NAdam (Nesterov Adam)
            optimizer = optax.nadam(
                learning_rate=self.learning_rate,
                b1=self.b1,
                b2=self.b2,
                eps=self.eps,
                eps_root=self.eps_root
            )
        elif self.optimizer_type == 'yogi':
            # Yogi optimizer
            optimizer = optax.yogi(
                learning_rate=self.learning_rate,
                b1=self.b1,
                b2=self.b2,
                eps=self.eps
            )
        elif self.optimizer_type == 'lamb':
            # LAMB optimizer (Layer-wise Adaptive Moments optimizer for Batch training)
            optimizer = optax.lamb(
                learning_rate=self.learning_rate,
                b1=self.b1,
                b2=self.b2,
                eps=self.eps,
                weight_decay=self.weight_decay
            )
        else:
            # Default to Adam
            optimizer = optax.adam(
                learning_rate=self.learning_rate,
                b1=self.b1,
                b2=self.b2,
                eps=self.eps
            )
        
        # Add weight decay if specified and not handled by optimizer
        if self.weight_decay > 0 and self.optimizer_type not in ['adamw', 'lamb']:
            optimizer = optax.chain(
                optax.add_decayed_weights(weight_decay=self.weight_decay),
                optimizer
            )
        
        return optimizer
    
    def _optimize_external(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Run JAX adaptive optimization.
        
        Args:
            X: Feature matrix (with bias)
            y: Target vector
            
        Returns:
            Dictionary containing optimization results
        """
        # Remove bias column
        X_no_bias = X[:, :-1]
        
        # Convert to JAX arrays
        X_jax = jnp.array(X_no_bias)
        y_jax = jnp.array(y)
        
        # Initialize parameters
        self.key, init_key = random.split(self.key)
        params = random.normal(init_key, (X_jax.shape[1] + 1,)) * 0.01
        
        # Initialize optimizer state
        self.opt_state = self.optimizer.init(params)
        
        # Training loop
        epoch_losses = []
        epoch_gradient_norms = []
        
        best_loss = float('inf')
        no_improve_count = 0
        patience = 20  # Adaptive optimizers may need more patience
        
        n_epochs = min(self.max_iterations, 1000)
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            epoch_grad_norm = 0.0
            batch_count = 0
            
            # Mini-batch optimization
            for X_batch, y_batch in self._create_batch_generator(X_no_bias, y):
                X_batch_jax = jnp.array(X_batch)
                y_batch_jax = jnp.array(y_batch)
                
                # Update parameters
                params, self.opt_state, batch_loss = self._update_step(
                    params, self.opt_state, X_batch_jax, y_batch_jax
                )
                
                # Compute gradient norm
                grads = self.grad_fn(params, X_batch_jax, y_batch_jax)
                grad_norm = jnp.linalg.norm(grads)
                
                epoch_loss += float(batch_loss)
                epoch_grad_norm += float(grad_norm)
                batch_count += 1
                
                # Track complexity
                self.track_function_evaluation(X_batch.shape)
                self.track_gradient_evaluation(X_batch.shape)
            
            # Average metrics
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
                print(f\"[CONVERGENCE] JAX {self.optimizer_type.upper()} converged at epoch {epoch + 1}\")\
                return {\
                    'final_weights': np.array(params),\
                    'iterations': epoch + 1,\
                    'converged': True,\
                    'final_loss': avg_epoch_loss,\
                    'final_gradient_norm': avg_grad_norm\
                }\
            \
            # Early stopping\
            if avg_epoch_loss < best_loss - self.convergence_tolerance:\
                best_loss = avg_epoch_loss\
                no_improve_count = 0\
            else:\
                no_improve_count += 1\
            \
            if no_improve_count >= patience:\
                print(f\"[EARLY_STOP] JAX {self.optimizer_type.upper()} early stopping at epoch {epoch + 1}\")\
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
        Get JAX adaptive optimizer specific results.\
        \
        Returns:\
            Dictionary containing JAX-specific results\
        \"\"\"\
        base_results = super()._get_algorithm_specific_results()\
        \
        jax_specific = {\
            'jax_adaptive_specific': {\
                'optimizer_type': self.optimizer_type,\
                'learning_rate': self.learning_rate,\
                'b1': self.b1,\
                'b2': self.b2,\
                'eps': self.eps,\
                'weight_decay': self.weight_decay,\
                'batch_size': self.batch_size,\
                'eps_root': self.eps_root,\
                'mu_dtype': self.mu_dtype,\
                'decay': self.decay,\
                'centered': self.centered,\
                'initial_accumulator_value': self.initial_accumulator_value,\
                'nesterov': self.nesterov,\
                'jit_compilation': True,\
                'optax_optimizer': type(self.optimizer).__name__ if self.optimizer else 'Unknown'\
            }\
        }\
        \
        base_results.update(jax_specific)\
        return base_results\
\
\
# Convenience functions for specific adaptive optimizers\
def create_jax_adam(learning_rate: float = 0.001, b1: float = 0.9, b2: float = 0.999, **kwargs) -> JAXAdaptive:\
    \"\"\"Create Adam optimizer.\"\"\"\
    return JAXAdaptive(\
        optimizer_type='adam',\
        learning_rate=learning_rate,\
        b1=b1,\
        b2=b2,\
        **kwargs\
    )\
\
def create_jax_adamw(learning_rate: float = 0.001, weight_decay: float = 0.01, **kwargs) -> JAXAdaptive:\
    \"\"\"Create AdamW optimizer.\"\"\"\
    return JAXAdaptive(\
        optimizer_type='adamw',\
        learning_rate=learning_rate,\
        weight_decay=weight_decay,\
        **kwargs\
    )\
\
def create_jax_rmsprop(learning_rate: float = 0.001, decay: float = 0.9, **kwargs) -> JAXAdaptive:\
    \"\"\"Create RMSprop optimizer.\"\"\"\
    return JAXAdaptive(\
        optimizer_type='rmsprop',\
        learning_rate=learning_rate,\
        decay=decay,\
        **kwargs\
    )\
\
def create_jax_adagrad(learning_rate: float = 0.01, **kwargs) -> JAXAdaptive:\
    \"\"\"Create AdaGrad optimizer.\"\"\"\
    return JAXAdaptive(\
        optimizer_type='adagrad',\
        learning_rate=learning_rate,\
        **kwargs\
    )\
\
def create_jax_nadam(learning_rate: float = 0.002, **kwargs) -> JAXAdaptive:\
    \"\"\"Create NAdam optimizer.\"\"\"\
    return JAXAdaptive(\
        optimizer_type='nadam',\
        learning_rate=learning_rate,\
        **kwargs\
    )\
\
def create_jax_yogi(learning_rate: float = 0.01, **kwargs) -> JAXAdaptive:\
    \"\"\"Create Yogi optimizer.\"\"\"\
    return JAXAdaptive(\
        optimizer_type='yogi',\
        learning_rate=learning_rate,\
        **kwargs\
    )\
\
def create_jax_lamb(learning_rate: float = 0.001, weight_decay: float = 0.01, **kwargs) -> JAXAdaptive:\
    \"\"\"Create LAMB optimizer.\"\"\"\
    return JAXAdaptive(\
        optimizer_type='lamb',\
        learning_rate=learning_rate,\
        weight_decay=weight_decay,\
        **kwargs\
    )