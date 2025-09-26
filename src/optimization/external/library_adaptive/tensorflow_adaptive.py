"""
TensorFlow Adaptive Optimization algorithms wrapper.
Implements modern adaptive optimizers from TensorFlow/Keras including Adam, RMSprop, Adagrad, etc.
"""

import numpy as np
from typing import Dict, Any, Optional, List
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


class TensorFlowAdaptive(BaseLibraryWrapper):
    """
    Wrapper for TensorFlow/Keras adaptive optimization algorithms.
    Provides modern adaptive optimizers like Adam, RMSprop, Adagrad, Adamax, Nadam, etc.
    """
    
    def __init__(self,
                 optimizer_type: str = 'adam',  # 'adam', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam', 'ftrl'
                 learning_rate: float = 0.001,
                 beta_1: float = 0.9,              # For Adam-like optimizers
                 beta_2: float = 0.999,            # For Adam-like optimizers
                 rho: float = 0.95,                # For RMSprop, Adadelta
                 epsilon: float = 1e-7,
                 decay: float = 0.0,               # Learning rate decay
                 batch_size: int = 32,
                 epochs: int = 100,
                 validation_split: float = 0.1,
                 early_stopping_patience: int = 15,
                 loss_type: str = 'ols',
                 regularization: float = 0.01,
                 convergence_tolerance: float = 1e-3,
                 max_iterations: int = 100000,
                 random_state: Optional[int] = None,
                 **kwargs):
        """
        Initialize TensorFlow adaptive optimizer wrapper.
        
        Args:
            optimizer_type: Type of adaptive optimizer
            learning_rate: Learning rate
            beta_1: Exponential decay rate for first moment estimates
            beta_2: Exponential decay rate for second moment estimates
            rho: Discounting factor for gradient history
            epsilon: Small constant for numerical stability
            decay: Learning rate decay over each update
            batch_size: Batch size for mini-batch optimization
            epochs: Number of training epochs
            validation_split: Fraction of data for validation
            early_stopping_patience: Early stopping patience
            loss_type: Loss function type
            regularization: Regularization parameter
            convergence_tolerance: Convergence tolerance
            max_iterations: Maximum iterations (mapped to epochs)
            random_state: Random seed
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not available. Please install TensorFlow to use TensorFlowAdaptive.")
        
        super().__init__(
            library_name='tensorflow',
            algorithm_name=optimizer_type.upper(),
            loss_type=loss_type,
            regularization=regularization,
            convergence_tolerance=convergence_tolerance,
            max_iterations=max_iterations,
            random_state=random_state,
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            rho=rho,
            epsilon=epsilon,
            decay=decay,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            early_stopping_patience=early_stopping_patience,
            **kwargs
        )
        
        self.optimizer_type = optimizer_type.lower()
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.rho = rho
        self.epsilon = epsilon
        self.decay = decay
        self.batch_size = batch_size
        self.epochs = min(epochs, max_iterations)
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        
        # TensorFlow model and components
        self.model = None
        self.history = None
        self.custom_callback = None
        
        # Set random seed for TensorFlow
        if self.random_state is not None:
            tf.random.set_seed(self.random_state)
    
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
            'ridge': 'mse',
            'lasso': 'mse',
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
            
            # Compute gradient norm approximation
            val_loss = logs.get('val_loss', logs.get('loss', 0.0))
            loss = logs.get('loss', 0.0)
            grad_norm_approx = abs(val_loss - loss) + 1e-8
            
            # Store in callback history
            self.wrapper.callback_history['losses'].append(float(loss))
            self.wrapper.callback_history['gradient_norms'].append(float(grad_norm_approx))
            self.wrapper.callback_history['weights'].append(current_weights.copy())
            self.wrapper.callback_history['iterations'].append(epoch)
    
    def _create_external_optimizer(self, n_features: int) -> optimizers.Optimizer:
        """
        Create TensorFlow adaptive optimizer.
        
        Args:
            n_features: Number of features (including bias)
            
        Returns:
            TensorFlow adaptive optimizer
        """
        # Create model
        self.model = self._create_tensorflow_model(n_features)
        
        # Create optimizer based on type
        if self.optimizer_type == 'adam':
            optimizer = optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=self.beta_1,
                beta_2=self.beta_2,
                epsilon=self.epsilon,
                decay=self.decay
            )
        elif self.optimizer_type == 'rmsprop':
            optimizer = optimizers.RMSprop(
                learning_rate=self.learning_rate,
                rho=self.rho,
                momentum=0.0,
                epsilon=self.epsilon,
                centered=False,
                decay=self.decay
            )
        elif self.optimizer_type == 'adagrad':
            optimizer = optimizers.Adagrad(
                learning_rate=self.learning_rate,
                initial_accumulator_value=0.1,
                epsilon=self.epsilon,
                decay=self.decay
            )
        elif self.optimizer_type == 'adadelta':
            optimizer = optimizers.Adadelta(
                learning_rate=self.learning_rate,
                rho=self.rho,
                epsilon=self.epsilon,
                decay=self.decay
            )
        elif self.optimizer_type == 'adamax':
            optimizer = optimizers.Adamax(
                learning_rate=self.learning_rate,
                beta_1=self.beta_1,
                beta_2=self.beta_2,
                epsilon=self.epsilon,
                decay=self.decay
            )
        elif self.optimizer_type == 'nadam':
            optimizer = optimizers.Nadam(
                learning_rate=self.learning_rate,
                beta_1=self.beta_1,
                beta_2=self.beta_2,
                epsilon=self.epsilon,
                decay=self.decay
            )
        elif self.optimizer_type == 'ftrl':
            # FTRL optimizer is good for large-scale learning
            optimizer = optimizers.Ftrl(
                learning_rate=self.learning_rate,
                learning_rate_power=-0.5,
                initial_accumulator_value=0.1,
                l1_regularization_strength=0.0,
                l2_regularization_strength=0.0
            )
        else:
            # Default to Adam
            optimizer = optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=self.beta_1,
                beta_2=self.beta_2,
                epsilon=self.epsilon
            )
        
        # Add regularizer if needed
        regularizer = self._create_regularizer()
        if regularizer:
            self.model.layers[0].kernel_regularizer = regularizer
            self.model.layers[0].bias_regularizer = regularizer
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss=self._get_tensorflow_loss(),
            metrics=['mae']
        )
        
        # Create custom callback
        self.custom_callback = self.TrackingCallback(self)
        
        return optimizer
    
    def _optimize_external(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Run TensorFlow adaptive optimization.
        
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
            verbose=0
        )
        
        # Extract final weights
        final_weights = self._extract_tensorflow_weights()
        
        # Get training metrics
        train_losses = self.history.history['loss']
        final_loss = train_losses[-1] if train_losses else float('inf')
        
        # Determine convergence
        actual_epochs = len(train_losses)
        converged = actual_epochs < self.epochs
        
        # Estimate final gradient norm
        if len(train_losses) > 1:
            final_gradient_norm = abs(train_losses[-1] - train_losses[-2]) + 1e-8
        else:
            final_gradient_norm = 1.0
        
        # Track complexity
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
            weights = layer_weights[0].flatten()
            bias = layer_weights[1].flatten()
        else:  # Only weights, no bias
            weights = layer_weights[0].flatten()
            bias = np.array([0.0])
        
        # Combine weights and bias (bias goes last)
        return np.concatenate([weights, bias])
    
    def _get_algorithm_specific_results(self) -> Dict[str, Any]:
        """
        Get TensorFlow adaptive optimizer specific results.
        
        Returns:
            Dictionary containing TensorFlow-specific results
        """
        base_results = super()._get_algorithm_specific_results()
        
        tensorflow_specific = {
            'tensorflow_adaptive_specific': {
                'optimizer_type': self.optimizer_type,
                'learning_rate': self.learning_rate,
                'beta_1': self.beta_1,
                'beta_2': self.beta_2,
                'rho': self.rho,
                'epsilon': self.epsilon,
                'decay': self.decay,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'validation_split': self.validation_split,
                'early_stopping_patience': self.early_stopping_patience,
                'loss_function': self._get_tensorflow_loss(),
                'regularizer_applied': self._create_regularizer() is not None,
                'actual_epochs_trained': len(self.history.history['loss']) if self.history else 0,
                'optimizer_class': type(self.external_optimizer).__name__ if self.external_optimizer else 'Unknown'
            }
        }
        
        base_results.update(tensorflow_specific)
        return base_results


# Convenience functions for specific adaptive optimizers
def create_tensorflow_adam(learning_rate: float = 0.001, beta_1: float = 0.9, beta_2: float = 0.999, **kwargs) -> TensorFlowAdaptive:
    """Create Adam optimizer."""
    return TensorFlowAdaptive(
        optimizer_type='adam',
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        **kwargs
    )

def create_tensorflow_rmsprop(learning_rate: float = 0.001, rho: float = 0.9, **kwargs) -> TensorFlowAdaptive:
    """Create RMSprop optimizer."""
    return TensorFlowAdaptive(
        optimizer_type='rmsprop',
        learning_rate=learning_rate,
        rho=rho,
        **kwargs
    )

def create_tensorflow_adagrad(learning_rate: float = 0.01, **kwargs) -> TensorFlowAdaptive:
    """Create Adagrad optimizer."""
    return TensorFlowAdaptive(
        optimizer_type='adagrad',
        learning_rate=learning_rate,
        **kwargs
    )

def create_tensorflow_adadelta(learning_rate: float = 1.0, rho: float = 0.95, **kwargs) -> TensorFlowAdaptive:
    """Create Adadelta optimizer."""
    return TensorFlowAdaptive(
        optimizer_type='adadelta',
        learning_rate=learning_rate,
        rho=rho,
        **kwargs
    )

def create_tensorflow_adamax(learning_rate: float = 0.002, **kwargs) -> TensorFlowAdaptive:
    """Create Adamax optimizer."""
    return TensorFlowAdaptive(
        optimizer_type='adamax',
        learning_rate=learning_rate,
        **kwargs
    )

def create_tensorflow_nadam(learning_rate: float = 0.002, **kwargs) -> TensorFlowAdaptive:
    """Create Nadam optimizer."""
    return TensorFlowAdaptive(
        optimizer_type='nadam',
        learning_rate=learning_rate,
        **kwargs
    )

def create_tensorflow_ftrl(learning_rate: float = 0.001, **kwargs) -> TensorFlowAdaptive:
    """Create FTRL optimizer."""
    return TensorFlowAdaptive(
        optimizer_type='ftrl',
        learning_rate=learning_rate,
        **kwargs
    )