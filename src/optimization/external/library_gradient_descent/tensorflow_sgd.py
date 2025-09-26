"""
TensorFlow/Keras SGD wrapper for gradient descent comparison.
"""

import numpy as np
from typing import Dict, Any, Optional
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
sys.path.append(project_root)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, optimizers
    # Suppress TensorFlow warnings
    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    tf.get_logger().setLevel(logging.ERROR)
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None
    layers = None
    optimizers = None

from src.optimization.external.base_library_wrapper import BaseLibraryWrapper


class TensorFlowSGDWrapper(BaseLibraryWrapper):
    """
    Wrapper for TensorFlow/Keras SGD optimizer.
    
    Provides gradient descent functionality using TensorFlow's implementation
    with support for momentum, learning rate schedules, and various configurations.
    """
    
    def __init__(self,
                 learning_rate: float = 0.001,
                 loss_type: str = 'ols',
                 regularization: float = 0.01,
                 max_iterations: int = 100000,
                 convergence_tolerance: float = 1e-3,
                 random_state: Optional[int] = None,
                 # TensorFlow SGD-specific parameters
                 momentum: float = 0.0,
                 nesterov: bool = False,
                 batch_size: Optional[int] = None,
                 learning_rate_schedule: Optional[str] = None,
                 **kwargs):
        """
        Initialize TensorFlow SGD wrapper.
        
        Args:
            learning_rate: Learning rate for SGD
            loss_type: Type of loss function ('ols', 'ridge', 'lasso')
            regularization: Regularization parameter
            max_iterations: Maximum number of iterations (epochs)
            convergence_tolerance: Convergence tolerance
            random_state: Random seed
            momentum: Momentum factor
            nesterov: Enable Nesterov momentum
            batch_size: Batch size for mini-batch SGD (None for full batch)
            learning_rate_schedule: Learning rate schedule ('exponential', 'polynomial', etc.)
            **kwargs: Additional parameters
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for TensorFlowSGDWrapper")
        
        # Store TensorFlow-specific parameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.batch_size = batch_size
        self.learning_rate_schedule = learning_rate_schedule
        self.tensorflow_kwargs = kwargs
        
        super().__init__(
            library_name='tensorflow',
            algorithm_name='SGD',
            loss_type=loss_type,
            regularization=regularization,
            max_iterations=max_iterations,
            convergence_tolerance=convergence_tolerance,
            random_state=random_state,
            learning_rate=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
            batch_size=batch_size,
            learning_rate_schedule=learning_rate_schedule,
            **kwargs
        )
        
        # TensorFlow model components
        self.model = None
        self.compiled_loss = None
        self.lr_scheduler = None
    
    def _create_learning_rate_schedule(self) -> tf.keras.optimizers.schedules.LearningRateSchedule:
        """
        Create learning rate schedule if specified.
        
        Returns:
            TensorFlow learning rate schedule or scalar learning rate
        """
        if self.learning_rate_schedule is None:
            return self.learning_rate
        
        if self.learning_rate_schedule == 'exponential':
            return tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.learning_rate,
                decay_steps=1000,
                decay_rate=0.96,
                staircase=True
            )
        elif self.learning_rate_schedule == 'polynomial':
            return tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=self.learning_rate,
                decay_steps=self.max_iterations,
                end_learning_rate=self.learning_rate * 0.01,
                power=1.0
            )
        elif self.learning_rate_schedule == 'cosine':
            return tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=self.learning_rate,
                decay_steps=self.max_iterations
            )
        else:
            return self.learning_rate
    
    def _create_tensorflow_model(self, n_features: int) -> tf.keras.Model:
        \"\"\"
        Create simple linear model for TensorFlow.
        
        Args:
            n_features: Number of features (including bias)
            
        Returns:
            TensorFlow/Keras model
        \"\"\"
        # Set random seed for TensorFlow
        if self.random_state is not None:
            tf.random.set_seed(self.random_state)
        
        model = tf.keras.Sequential([
            layers.Dense(1, use_bias=False, input_shape=(n_features,),
                        kernel_initializer=tf.keras.initializers.Constant(self.weights.reshape(-1, 1)))
        ])
        
        return model
    
    def _create_loss_function(self):\n        \"\"\"\n        Create TensorFlow loss function.\n        \n        Returns:\n            TensorFlow loss function\n        \"\"\"\n        if self.loss_type == 'ols':\n            return tf.keras.losses.MeanSquaredError()\n        elif self.loss_type == 'ridge':\n            # Use MSE loss and add L2 regularization via kernel_regularizer\n            return tf.keras.losses.MeanSquaredError()\n        elif self.loss_type == 'lasso':\n            # Use MSE loss and add L1 regularization via kernel_regularizer\n            return tf.keras.losses.MeanSquaredError()\n        else:\n            raise ValueError(f\"Unsupported loss type: {self.loss_type}\")\n    \n    def _create_external_optimizer(self, n_features: int) -> optimizers.SGD:\n        \"\"\"\n        Create TensorFlow SGD optimizer.\n        \n        Args:\n            n_features: Number of features (including bias)\n            \n        Returns:\n            TensorFlow SGD optimizer\n        \"\"\"\n        # Create model\n        self.model = self._create_tensorflow_model(n_features)\n        \n        # Add regularization to the model if needed\n        if self.loss_type == 'ridge' and self.regularization > 0:\n            self.model.layers[0].kernel_regularizer = tf.keras.regularizers.L2(self.regularization)\n        elif self.loss_type == 'lasso' and self.regularization > 0:\n            self.model.layers[0].kernel_regularizer = tf.keras.regularizers.L1(self.regularization)\n        \n        # Create loss function\n        self.compiled_loss = self._create_loss_function()\n        \n        # Create learning rate schedule\n        lr_schedule = self._create_learning_rate_schedule()\n        \n        # Create SGD optimizer\n        sgd_params = {\n            'learning_rate': lr_schedule,\n            'momentum': self.momentum,\n            'nesterov': self.nesterov\n        }\n        \n        # Add any additional parameters\n        sgd_params.update(self.tensorflow_kwargs)\n        \n        optimizer = optimizers.SGD(**sgd_params)\n        \n        # Compile model\n        self.model.compile(\n            optimizer=optimizer,\n            loss=self.compiled_loss,\n            run_eagerly=True  # Enable eager execution for easier debugging\n        )\n        \n        return optimizer\n    \n    def _optimize_external(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:\n        \"\"\"\n        Run TensorFlow SGD optimization.\n        \n        Args:\n            X: Feature matrix (with bias)\n            y: Target vector\n            \n        Returns:\n            Dictionary containing optimization results\n        \"\"\"\n        try:\n            # Convert to TensorFlow format\n            X_tf = tf.constant(X, dtype=tf.float32)\n            y_tf = tf.constant(y.reshape(-1, 1), dtype=tf.float32)\n            \n            n_samples = X.shape[0]\n            \n            # Determine batch size\n            if self.batch_size is None or self.batch_size >= n_samples:\n                batch_size = n_samples  # Full batch\n            else:\n                batch_size = self.batch_size\n            \n            # Create dataset\n            dataset = tf.data.Dataset.from_tensor_slices((X_tf, y_tf))\n            if batch_size < n_samples:\n                dataset = dataset.shuffle(n_samples, seed=self.random_state)\n            dataset = dataset.batch(batch_size)\n            \n            converged = False\n            \n            # Custom training loop for better control\n            for epoch in range(self.max_iterations):\n                epoch_loss = 0.0\n                n_batches = 0\n                \n                for X_batch, y_batch in dataset:\n                    with tf.GradientTape() as tape:\n                        y_pred = self.model(X_batch, training=True)\n                        loss = self.compiled_loss(y_batch, y_pred)\n                        \n                        # Add regularization loss if present\n                        if self.model.losses:\n                            regularization_loss = tf.reduce_sum(self.model.losses)\n                            loss = loss + regularization_loss\n                    \n                    # Compute gradients\n                    gradients = tape.gradient(loss, self.model.trainable_variables)\n                    \n                    # Apply gradients\n                    self.external_optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))\n                    \n                    epoch_loss += loss.numpy()\n                    n_batches += 1\n                \n                # Check convergence every epoch\n                if epoch % self.convergence_check_freq == 0 or epoch == self.max_iterations - 1:\n                    # Get current weights\n                    current_weights = self.model.trainable_variables[0].numpy().flatten()\n                    \n                    # Compute gradient for convergence check\n                    with tf.GradientTape() as tape:\n                        y_pred = self.model(X_tf, training=True)\n                        loss = self.compiled_loss(y_tf, y_pred)\n                        \n                        # Add regularization loss\n                        if self.model.losses:\n                            regularization_loss = tf.reduce_sum(self.model.losses)\n                            loss = loss + regularization_loss\n                    \n                    gradients = tape.gradient(loss, self.model.trainable_variables)\n                    grad_norm = tf.norm(gradients[0]).numpy()\n                    \n                    # Store in callback history\n                    self.callback_history['losses'].append(float(loss.numpy()))\n                    self.callback_history['gradient_norms'].append(float(grad_norm))\n                    self.callback_history['weights'].append(current_weights.copy())\n                    self.callback_history['iterations'].append(epoch)\n                    \n                    # Check convergence\n                    if grad_norm < self.convergence_tolerance:\n                        converged = True\n                        print(f\"TensorFlow SGD converged at epoch {epoch} (gradient norm: {grad_norm:.6f})\")\n                        break\n            \n            # Extract final weights\n            final_weights = self.model.trainable_variables[0].numpy().flatten()\n            \n            return {\n                'final_weights': final_weights,\n                'converged': converged,\n                'iterations': epoch + 1,\n                'final_loss': float(epoch_loss / n_batches) if n_batches > 0 else float('inf'),\n                'batch_size_used': batch_size,\n                'n_batches': n_batches\n            }\n            \n        except Exception as e:\n            print(f\"TensorFlow SGD optimization failed: {str(e)}\")\n            return {\n                'final_weights': self.weights,\n                'converged': False,\n                'iterations': 0,\n                'error': str(e)\n            }\n    \n    def _get_algorithm_specific_results(self) -> Dict[str, Any]:\n        \"\"\"\n        Get TensorFlow SGD-specific results.\n        \n        Returns:\n            Dictionary containing TensorFlow-specific results\n        \"\"\"\n        base_results = super()._get_algorithm_specific_results()\n        \n        tensorflow_specific = {\n            'tensorflow_sgd_params': {\n                'learning_rate': self.learning_rate,\n                'momentum': self.momentum,\n                'nesterov': self.nesterov,\n                'batch_size': self.batch_size,\n                'learning_rate_schedule': self.learning_rate_schedule,\n                'tensorflow_version': tf.__version__ if tf else 'unknown'\n            }\n        }\n        \n        # Add model information if available\n        if self.model is not None:\n            tensorflow_specific['tensorflow_model'] = {\n                'model_type': type(self.model).__name__,\n                'n_parameters': self.model.count_params(),\n                'input_shape': str(self.model.input_shape),\n                'output_shape': str(self.model.output_shape)\n            }\n        \n        base_results['algorithm_specific'].update(tensorflow_specific)\n        return base_results\n\n\ndef create_tensorflow_sgd_optimizer(learning_rate: float = 0.001,\n                                   loss_type: str = 'ols',\n                                   regularization: float = 0.01,\n                                   max_iterations: int = 100000,\n                                   random_state: Optional[int] = None,\n                                   **kwargs) -> TensorFlowSGDWrapper:\n    \"\"\"\n    Factory function to create TensorFlow SGD optimizer.\n    \n    Args:\n        learning_rate: Learning rate for SGD\n        loss_type: Type of loss function ('ols', 'ridge', 'lasso')\n        regularization: Regularization parameter\n        max_iterations: Maximum number of iterations\n        random_state: Random seed\n        **kwargs: Additional parameters\n        \n    Returns:\n        TensorFlowSGDWrapper instance\n    \"\"\"\n    return TensorFlowSGDWrapper(\n        learning_rate=learning_rate,\n        loss_type=loss_type,\n        regularization=regularization,\n        max_iterations=max_iterations,\n        random_state=random_state,\n        **kwargs\n    )