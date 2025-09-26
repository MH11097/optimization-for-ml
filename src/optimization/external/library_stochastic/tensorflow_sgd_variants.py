"""
TensorFlow Stochastic Gradient Descent variants wrapper.
Implements multiple SGD configurations from TensorFlow/Keras for stochastic optimization.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import sys
import os
import warnings

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
sys.path.append(project_root)

from src.optimization.external.base_library_wrapper import BaseLibraryWrapper

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, optimizers, losses
    from tensorflow.keras.callbacks import EarlyStopping, Callback
    
    # Disable GPU if not available to avoid warnings
    try:
        tf.config.experimental.set_visible_devices([], 'GPU')
    except:
        pass
    
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class TensorFlowSGDVariants(BaseLibraryWrapper):
    """
    Wrapper for TensorFlow/Keras SGD variants with different configurations.
    Provides various SGD optimizations including momentum, Nesterov, learning rate schedules.
    """
    
    def __init__(self,
                 learning_rate: float = 0.01,
                 momentum: float = 0.0,
                 nesterov: bool = False,
                 batch_size: int = 32,
                 epochs: int = 100,
                 validation_split: float = 0.1,
                 early_stopping_patience: int = 10,
                 lr_schedule: str = 'constant',  # 'constant', 'exponential_decay', 'polynomial_decay'
                 loss_type: str = 'ols',
                 regularization: float = 0.01,
                 convergence_tolerance: float = 1e-3,
                 max_iterations: int = 100000,
                 random_state: Optional[int] = None,
                 **kwargs):
        """
        Initialize TensorFlow SGD variants wrapper.
        
        Args:
            learning_rate: Learning rate
            momentum: Momentum factor (0 = no momentum)
            nesterov: Enable Nesterov momentum
            batch_size: Batch size for mini-batch SGD
            epochs: Number of training epochs
            validation_split: Fraction of data for validation
            early_stopping_patience: Early stopping patience
            lr_schedule: Learning rate schedule type
            loss_type: Loss function type
            regularization: Regularization parameter
            convergence_tolerance: Convergence tolerance
            max_iterations: Maximum iterations (mapped to epochs)
            random_state: Random seed
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not available. Please install TensorFlow to use TensorFlowSGDVariants.")
        
        super().__init__(
            library_name='tensorflow',
            algorithm_name='SGD',
            loss_type=loss_type,
            regularization=regularization,
            convergence_tolerance=convergence_tolerance,
            max_iterations=max_iterations,
            random_state=random_state,
            learning_rate=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            early_stopping_patience=early_stopping_patience,
            lr_schedule=lr_schedule,
            **kwargs
        )
        
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.batch_size = batch_size
        self.epochs = min(epochs, max_iterations)  # Use the smaller value
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.lr_schedule = lr_schedule
        
        # TensorFlow model and components
        self.model = None
        self.history = None
        self.custom_callback = None
        
        # Set random seed for TensorFlow
        if self.random_state is not None:
            tf.random.set_seed(self.random_state)
    
    def _create_learning_rate_schedule(self) -> tf.keras.optimizers.schedules.LearningRateSchedule:
        """
        Create learning rate schedule.
        
        Returns:
            TensorFlow learning rate schedule
        """
        if self.lr_schedule == 'constant':
            return self.learning_rate
        elif self.lr_schedule == 'exponential_decay':
            return tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.learning_rate,
                decay_steps=100,
                decay_rate=0.96,
                staircase=True
            )
        elif self.lr_schedule == 'polynomial_decay':
            return tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=self.learning_rate,
                decay_steps=1000,
                end_learning_rate=self.learning_rate * 0.01
            )
        else:
            return self.learning_rate
    
    def _create_tensorflow_model(self, n_features: int) -> keras.Model:
        """
        Create a simple linear model for TensorFlow.
        
        Args:
            n_features: Number of features (including bias)
            
        Returns:
            TensorFlow/Keras model
        """
        model = keras.Sequential([
            layers.Dense(1, input_shape=(n_features - 1,), use_bias=True,
                        kernel_initializer=keras.initializers.RandomNormal(mean=0, stddev=0.01, seed=42),
                        bias_initializer=keras.initializers.RandomNormal(mean=0, stddev=0.01, seed=42))
        ])
        
        return model
    
    def _get_tensorflow_loss(self) -> str:
        """
        Get TensorFlow loss function name.
        
        Returns:
            TensorFlow loss function name
        """
        loss_mapping = {
            'ols': 'mse',
            'ridge': 'mse',  # Ridge regularization handled separately
            'lasso': 'mse',  # Lasso regularization handled separately  
            'huber': 'huber',
            'mae': 'mae'
        }
        return loss_mapping.get(self.loss_type, 'mse')
    
    def _create_regularizer(self) -> Optional[keras.regularizers.Regularizer]:
        """
        Create regularizer based on loss type.
        
        Returns:
            TensorFlow regularizer or None
        """
        if self.loss_type == 'ridge':
            return keras.regularizers.l2(self.regularization)
        elif self.loss_type == 'lasso':
            return keras.regularizers.l1(self.regularization)
        elif self.loss_type == 'elastic_net':
            return keras.regularizers.l1_l2(l1=self.regularization * 0.5, l2=self.regularization * 0.5)
        else:
            return None
    
    class TrackingCallback(Callback):
        """Custom callback to track optimization progress."""
        
        def __init__(self, wrapper_instance):
            super().__init__()
            self.wrapper = wrapper_instance
        
        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                logs = {}
            
            # Extract current weights
            current_weights = self.wrapper._extract_tensorflow_weights()
            
            # Compute gradient norm (approximation using validation loss change)
            val_loss = logs.get('val_loss', logs.get('loss', 0.0))
            loss = logs.get('loss', 0.0)
            grad_norm_approx = abs(val_loss - loss) + 1e-8  # Approximation
            
            # Store in callback history
            self.wrapper.callback_history['losses'].append(float(loss))
            self.wrapper.callback_history['gradient_norms'].append(float(grad_norm_approx))
            self.wrapper.callback_history['weights'].append(current_weights.copy())
            self.wrapper.callback_history['iterations'].append(epoch)
    
    def _create_external_optimizer(self, n_features: int) -> optimizers.Optimizer:
        """
        Create TensorFlow SGD optimizer.
        
        Args:
            n_features: Number of features (including bias)
            
        Returns:
            TensorFlow SGD optimizer
        """
        # Create model
        self.model = self._create_tensorflow_model(n_features)
        
        # Create learning rate schedule
        learning_rate_schedule = self._create_learning_rate_schedule()
        
        # Create SGD optimizer
        optimizer = optimizers.SGD(
            learning_rate=learning_rate_schedule,
            momentum=self.momentum,
            nesterov=self.nesterov
        )
        
        # Compile model with regularizer if needed
        regularizer = self._create_regularizer()
        if regularizer:
            # Add regularization to the dense layer
            self.model.layers[0].kernel_regularizer = regularizer
            self.model.layers[0].bias_regularizer = regularizer
        
        self.model.compile(
            optimizer=optimizer,
            loss=self._get_tensorflow_loss(),
            metrics=['mae']
        )
        
        # Create custom callback for tracking
        self.custom_callback = self.TrackingCallback(self)
        
        return optimizer
    
    def _optimize_external(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Run TensorFlow SGD optimization.
        
        Args:
            X: Feature matrix (with bias)
            y: Target vector
            
        Returns:
            Dictionary containing optimization results
        """
        # Remove bias column (TensorFlow handles bias internally)
        X_no_bias = X[:, :-1]
        
        # Prepare callbacks
        callbacks = [self.custom_callback]
        
        # Add early stopping if validation split is used
        if self.validation_split > 0:
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                min_delta=self.convergence_tolerance
            )
            callbacks.append(early_stopping)
        
        # Train the model
        self.history = self.model.fit(
            X_no_bias, y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=0  # Suppress training output
        )
        
        # Extract final weights
        final_weights = self._extract_tensorflow_weights()
        
        # Get training metrics
        train_losses = self.history.history['loss']
        final_loss = train_losses[-1] if train_losses else float('inf')
        
        # Determine if converged (if early stopping was triggered)
        actual_epochs = len(train_losses)
        converged = actual_epochs < self.epochs
        
        # Estimate final gradient norm
        if len(train_losses) > 1:
            # Use loss change as approximation for gradient norm
            final_gradient_norm = abs(train_losses[-1] - train_losses[-2]) + 1e-8
        else:
            final_gradient_norm = 1.0
        
        # Track final complexity
        self.track_function_evaluation((X_no_bias.shape[0], X_no_bias.shape[1]))
        self.track_gradient_evaluation((X_no_bias.shape[0], X_no_bias.shape[1]))
        
        return {
            'final_weights': final_weights,
            'iterations': actual_epochs,
            'converged': converged,
            'final_loss': final_loss,
            'final_gradient_norm': final_gradient_norm,
            'training_history': self.history.history
        }
    
    def _extract_tensorflow_weights(self) -> np.ndarray:
        """
        Extract weights from TensorFlow model.
        
        Returns:
            Numpy array of weights including bias
        """
        # Get weights from the dense layer
        layer_weights = self.model.layers[0].get_weights()
        
        if len(layer_weights) == 2:  # weights and bias
            weights = layer_weights[0].flatten()  # Shape: (n_features, 1) -> (n_features,)
            bias = layer_weights[1].flatten()     # Shape: (1,) -> (1,)
        else:  # Only weights, no bias
            weights = layer_weights[0].flatten()
            bias = np.array([0.0])
        
        # Combine weights and bias (bias goes last)
        return np.concatenate([weights, bias])
    
    def _get_algorithm_specific_results(self) -> Dict[str, Any]:
        """
        Get TensorFlow SGD specific results.
        
        Returns:
            Dictionary containing TensorFlow-specific results
        """
        base_results = super()._get_algorithm_specific_results()
        
        tensorflow_specific = {
            'tensorflow_sgd_specific': {
                'learning_rate': self.learning_rate,
                'momentum': self.momentum,
                'nesterov': self.nesterov,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'validation_split': self.validation_split,
                'early_stopping_patience': self.early_stopping_patience,
                'lr_schedule': self.lr_schedule,
                'loss_function': self._get_tensorflow_loss(),
                'regularizer_applied': self._create_regularizer() is not None,
                'actual_epochs_trained': len(self.history.history['loss']) if self.history else 0,
                'training_history_available': self.history is not None
            }
        }
        
        base_results.update(tensorflow_specific)
        return base_results


# Convenience functions for specific SGD configurations
def create_tensorflow_sgd_vanilla(learning_rate: float = 0.01, batch_size: int = 32, **kwargs) -> TensorFlowSGDVariants:
    """Create vanilla SGD without momentum."""
    return TensorFlowSGDVariants(
        learning_rate=learning_rate,
        momentum=0.0,
        batch_size=batch_size,
        **kwargs
    )

def create_tensorflow_sgd_momentum(learning_rate: float = 0.01, momentum: float = 0.9, batch_size: int = 32, **kwargs) -> TensorFlowSGDVariants:
    """Create SGD with momentum."""
    return TensorFlowSGDVariants(
        learning_rate=learning_rate,
        momentum=momentum,
        batch_size=batch_size,
        **kwargs
    )

def create_tensorflow_sgd_nesterov(learning_rate: float = 0.01, momentum: float = 0.9, batch_size: int = 32, **kwargs) -> TensorFlowSGDVariants:
    """Create SGD with Nesterov momentum."""
    return TensorFlowSGDVariants(
        learning_rate=learning_rate,
        momentum=momentum,
        nesterov=True,
        batch_size=batch_size,
        **kwargs
    )

def create_tensorflow_sgd_exponential_decay(learning_rate: float = 0.01, batch_size: int = 32, **kwargs) -> TensorFlowSGDVariants:
    """Create SGD with exponential learning rate decay."""
    return TensorFlowSGDVariants(
        learning_rate=learning_rate,
        lr_schedule='exponential_decay',
        batch_size=batch_size,
        **kwargs
    )

def create_tensorflow_sgd_polynomial_decay(learning_rate: float = 0.01, batch_size: int = 32, **kwargs) -> TensorFlowSGDVariants:
    """Create SGD with polynomial learning rate decay."""
    return TensorFlowSGDVariants(
        learning_rate=learning_rate,
        lr_schedule='polynomial_decay',
        batch_size=batch_size,
        **kwargs
    )