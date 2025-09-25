"""
Optimizer Factory for creating optimizers with unified interface.
This module provides a factory pattern implementation for creating
optimizers with standardized parameter handling and validation.
"""
import numpy as np
from typing import Dict, Any, Type, Optional, Union
import inspect
from .base import BaseOptimizer
from .algorithms.gradient_descent import GradientDescentOptimizer
from .algorithms.stochastic_gd import StochasticGDOptimizer
from .algorithms.newton_method import NewtonOptimizer
from .algorithms.quasi_newton import QuasiNewtonOptimizer
from .algorithms.subgradient import SubgradientOptimizer

class OptimizerFactory:
    """
    Factory class for creating optimizers with unified interface.
    
    Supports:
    - Creating optimizers by name
    - Parameter validation 
    - Default parameter handling
    - Parameter mapping from legacy interfaces
    - Configuration-based creation
    """
    
    # Registry of available optimizers
    _optimizer_registry: Dict[str, Type[BaseOptimizer]] = {
        'gradient_descent': GradientDescentOptimizer,
        'gd': GradientDescentOptimizer,  # Alias
        'stochastic_gd': StochasticGDOptimizer,
        'sgd': StochasticGDOptimizer,  # Alias
        'newton': NewtonOptimizer,
        'newton_method': NewtonOptimizer,  # Alias
        'quasi_newton': QuasiNewtonOptimizer,
        'bfgs': QuasiNewtonOptimizer,  # Alias
        'l_bfgs': QuasiNewtonOptimizer,  # Alias
        'subgradient': SubgradientOptimizer,
        'subgrad': SubgradientOptimizer,  # Alias
    }
    
    # Default parameters for each optimizer type
    _default_parameters: Dict[str, Dict[str, Any]] = {
        'gradient_descent': {
            'ham_loss': 'ols',  # Use legacy parameter names for compatibility
            'learning_rate': 0.01,
            'regularization': 0.01,
            'diem_dung': 1e-3,
            'max_iterations': 500000,
            'convergence_check_freq': 1,
            'random_state': None,
            'step_size_method': 'constant',
            'line_search_method': 'none',
            'momentum_method': 'none',
            'momentum_coefficient': 0.9,
        },
        'stochastic_gd': {
            'loss_type': 'ols',
            'learning_rate': 0.01,
            'regularization': 0.01,
            'convergence_tolerance': 1e-3,
            'max_iterations': 10000,
            'convergence_check_freq': 100,
            'random_state': None,
            'batch_strategy': 'fixed',
            'batch_size': 32,
            'step_size_method': 'constant',
            'momentum_method': 'none',
        },
        'newton': {
            'loss_type': 'ols',
            'regularization': 0.01,
            'convergence_tolerance': 1e-3,
            'max_iterations': 1000,
            'convergence_check_freq': 1,
            'random_state': None,
            'damping_strategy': 'none',
            'line_search_method': 'none',
        },
        'quasi_newton': {
            'loss_type': 'ols',
            'regularization': 0.01,
            'convergence_tolerance': 1e-3,
            'max_iterations': 1000,
            'convergence_check_freq': 1,
            'random_state': None,
            'method': 'bfgs',
            'memory_size': 10,
            'line_search_method': 'backtracking',
        },
        'subgradient': {
            'ham_loss': 'lasso',
            'lambda_penalty': 0.1,
            'regularization': 0.1,
            'diem_dung': 1e-8,
            'max_iterations': 750,
            'convergence_check_freq': 50,
            'random_state': None,
            'step_size_method': 'constant',
            'step_size': 0.02,
        }
    }
    
    # Parameter mapping from new interface to legacy interface (for compatibility)
    _parameter_mappings: Dict[str, Dict[str, str]] = {
        'gradient_descent': {
            # New to legacy mapping
            'loss_type': 'ham_loss',
            'convergence_tolerance': 'diem_dung',
            'so_lan_thu': 'max_iterations',
            
            # Direct mappings (no change needed)
            'ham_loss': 'ham_loss',
            'learning_rate': 'learning_rate', 
            'regularization': 'regularization',
            'diem_dung': 'diem_dung',
            'max_iterations': 'max_iterations',
            'convergence_check_freq': 'convergence_check_freq',
            'step_size_method': 'step_size_method',
            'line_search_method': 'line_search_method',
            'momentum_method': 'momentum_method',
            'momentum_coefficient': 'momentum_coefficient',
            
            # Legacy parameter names
            'backtrack_c1': 'backtrack_c1',
            'backtrack_rho': 'backtrack_rho',
            'adaptive_beta1': 'adaptive_beta1',
            'adaptive_beta2': 'adaptive_beta2',
            'adaptive_eps': 'adaptive_eps',
            'wolfe_c2': 'wolfe_c2',
            'decay_gamma': 'decay_rate',
        },
        'stochastic_gd': {
            'ham_loss': 'loss_type',
            'diem_dung': 'convergence_tolerance',
            'so_lan_thu': 'max_iterations',
            'batch_method': 'batch_strategy',
            # Direct mappings (no change needed)
            'step_size_method': 'step_size_method',
            'momentum_method': 'momentum_method',
        },
        'newton': {
            'ham_loss': 'loss_type',
            'diem_dung': 'convergence_tolerance',
            'so_lan_thu': 'max_iterations',
            'damping_method': 'damping_strategy',
            # Direct mappings (no change needed)
            'line_search_method': 'line_search_method',
        },
        'quasi_newton': {
            'ham_loss': 'loss_type',
            'diem_dung': 'convergence_tolerance',
            'so_lan_thu': 'max_iterations',
            # Direct mappings (no change needed)
            'method': 'method',
            'line_search_method': 'line_search_method',
        },
        'subgradient': {
            'loss_type': 'ham_loss',
            'convergence_tolerance': 'diem_dung',
            'so_lan_thu': 'max_iterations',
            'penalty_lambda': 'lambda_penalty',
        }
    }
    
    @classmethod
    def register_optimizer(cls, 
                          name: str, 
                          optimizer_class: Type[BaseOptimizer],
                          default_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a new optimizer in the factory.
        
        Args:
            name: Name of the optimizer
            optimizer_class: Optimizer class
            default_params: Default parameters (optional)
        """
        if not issubclass(optimizer_class, BaseOptimizer):
            raise ValueError(f"Optimizer class must inherit from BaseOptimizer")
        
        cls._optimizer_registry[name] = optimizer_class
        
        if default_params:
            cls._default_parameters[name] = default_params
    
    @classmethod
    def list_available_optimizers(cls) -> list[str]:
        """
        List available optimizers.
        
        Returns:
            List of optimizer names
        """
        return list(cls._optimizer_registry.keys())
    
    @classmethod
    def get_optimizer_info(cls, optimizer_name: str) -> Dict[str, Any]:
        """
        Get information about an optimizer.
        
        Args:
            optimizer_name: Name of the optimizer
            
        Returns:
            Dictionary containing optimizer information
        """
        if optimizer_name not in cls._optimizer_registry:
            raise ValueError(f"Unknown optimizer: {optimizer_name}. "
                           f"Available: {cls.list_available_optimizers()}")
        
        optimizer_class = cls._optimizer_registry[optimizer_name]
        default_params = cls._default_parameters.get(optimizer_name, {})
        
        # Get constructor signature
        sig = inspect.signature(optimizer_class.__init__)
        parameters = {}
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            param_info = {
                'type': param.annotation if param.annotation != inspect.Parameter.empty else 'Any',
                'default': param.default if param.default != inspect.Parameter.empty else None,
                'required': param.default == inspect.Parameter.empty
            }
            parameters[param_name] = param_info
        
        return {
            'name': optimizer_name,
            'class': optimizer_class.__name__,
            'description': optimizer_class.__doc__,
            'parameters': parameters,
            'default_parameters': default_params
        }
    
    @classmethod
    def _map_legacy_parameters(cls, 
                             optimizer_name: str, 
                             params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map parameters from legacy interface to new interface.
        
        Args:
            optimizer_name: Name of the optimizer
            params: Legacy parameters
            
        Returns:
            Mapped parameters
        """
        if optimizer_name not in cls._parameter_mappings:
            return params
        
        mappings = cls._parameter_mappings[optimizer_name]
        mapped_params = {}
        
        for old_name, value in params.items():
            new_name = mappings.get(old_name, old_name)
            mapped_params[new_name] = value
        
        return mapped_params
    
    @classmethod
    def _validate_parameters(cls, 
                           optimizer_name: str,
                           params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean parameters.
        
        Args:
            optimizer_name: Name of the optimizer
            params: Raw parameters
            
        Returns:
            Validated parameters
        """
        # Get optimizer class and signature
        optimizer_class = cls._optimizer_registry[optimizer_name]
        sig = inspect.signature(optimizer_class.__init__)
        
        # Filter parameters to keep only what the optimizer accepts
        valid_params = {}
        for param_name in sig.parameters:
            if param_name == 'self':
                continue
            if param_name in params:
                valid_params[param_name] = params[param_name]
        
        # Add default parameters for missing ones
        defaults = cls._default_parameters.get(optimizer_name, {})
        for param_name, default_value in defaults.items():
            if param_name not in valid_params:
                valid_params[param_name] = default_value
        
        return valid_params
    
    @classmethod
    def create_optimizer(cls, 
                        optimizer_name: str,
                        **kwargs) -> BaseOptimizer:
        """
        Create an optimizer with specified name and parameters.
        
        Args:
            optimizer_name: Name of the optimizer ('gradient_descent', 'sgd', etc.)
            **kwargs: Parameters for the optimizer
            
        Returns:
            Configured optimizer instance
            
        Example:
            optimizer = OptimizerFactory.create_optimizer(
                'gradient_descent',
                loss_type='ols',
                learning_rate=0.01,
                step_size_method='constant'
            )
        """
        # Normalize optimizer name
        optimizer_name = optimizer_name.lower().replace('-', '_').replace(' ', '_')
        
        if optimizer_name not in cls._optimizer_registry:
            raise ValueError(f"Unknown optimizer: {optimizer_name}. "
                           f"Available: {cls.list_available_optimizers()}")
        
        # Map legacy parameters
        mapped_params = cls._map_legacy_parameters(optimizer_name, kwargs)
        
        # Validate parameters
        valid_params = cls._validate_parameters(optimizer_name, mapped_params)
        
        # Create optimizer instance
        optimizer_class = cls._optimizer_registry[optimizer_name]
        
        try:
            return optimizer_class(**valid_params)
        except Exception as e:
            raise ValueError(f"Error creating {optimizer_name} optimizer: {e}\n"
                           f"Parameters: {valid_params}")
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> BaseOptimizer:
        """
        Create an optimizer from configuration dictionary.
        
        Args:
            config: Configuration dictionary with 'optimizer' key and parameters
            
        Returns:
            Configured optimizer instance
            
        Example:
            config = {
                'optimizer': 'gradient_descent',
                'loss_type': 'ols',
                'learning_rate': 0.01,
                'step_size_method': 'adaptive'
            }
            optimizer = OptimizerFactory.create_from_config(config)
        """
        if 'optimizer' not in config:
            raise ValueError("Config must contain 'optimizer' key")
        
        optimizer_name = config['optimizer']
        params = {k: v for k, v in config.items() if k != 'optimizer'}
        
        return cls.create_optimizer(optimizer_name, **params)
    
    @classmethod
    def create_gradient_descent(cls, **kwargs) -> GradientDescentOptimizer:
        """
        Convenience method for creating Gradient Descent optimizer.
        
        Args:
            **kwargs: Parameters for GradientDescentOptimizer
            
        Returns:
            GradientDescentOptimizer instance
        """
        return cls.create_optimizer('gradient_descent', **kwargs)
    
    @classmethod
    def create_stochastic_gd(cls, **kwargs) -> StochasticGDOptimizer:
        """
        Convenience method for creating Stochastic Gradient Descent optimizer.
        
        Args:
            **kwargs: Parameters for StochasticGDOptimizer
            
        Returns:
            StochasticGDOptimizer instance
        """
        return cls.create_optimizer('stochastic_gd', **kwargs)
    
    @classmethod
    def create_newton(cls, **kwargs) -> NewtonOptimizer:
        """
        Convenience method for creating Newton optimizer.
        
        Args:
            **kwargs: Parameters for NewtonOptimizer
            
        Returns:
            NewtonOptimizer instance
        """
        return cls.create_optimizer('newton', **kwargs)
    
    @classmethod
    def create_quasi_newton(cls, **kwargs) -> QuasiNewtonOptimizer:
        """
        Convenience method for creating Quasi-Newton optimizer.
        
        Args:
            **kwargs: Parameters for QuasiNewtonOptimizer
            
        Returns:
            QuasiNewtonOptimizer instance
        """
        return cls.create_optimizer('quasi_newton', **kwargs)
    
    @classmethod
    def create_subgradient(cls, **kwargs) -> SubgradientOptimizer:
        """
        Convenience method for creating Subgradient optimizer.
        
        Args:
            **kwargs: Parameters for SubgradientOptimizer
            
        Returns:
            SubgradientOptimizer instance
        """
        return cls.create_optimizer('subgradient', **kwargs)
    
    @classmethod
    def create_compatible_optimizer(cls, 
                                  legacy_model_class: str,
                                  **legacy_params) -> BaseOptimizer:
        """
        Create an optimizer compatible with legacy model classes.
        
        Args:
            legacy_model_class: Legacy class name ('GradientDescentModel', etc.)
            **legacy_params: Legacy parameters
            
        Returns:
            Compatible optimizer instance
        """
        # Map legacy class names to new optimizer names
        class_mappings = {
            'GradientDescentModel': 'gradient_descent',
            'gradient_descent_model': 'gradient_descent',
            'MomentumGDModel': 'gradient_descent',
            'momentum_gd_model': 'gradient_descent',
            'NesterovGDModel': 'gradient_descent', 
            'nesterov_gd_model': 'gradient_descent',
            'StochasticGDModel': 'stochastic_gd',
            'stochastic_gd_model': 'stochastic_gd',
            'SGDModel': 'stochastic_gd',
            'sgd_model': 'stochastic_gd',
            'NewtonModel': 'newton',
            'newton_model': 'newton',
            'NewtonMethodModel': 'newton',
            'newton_method_model': 'newton',
            'QuasiNewtonModel': 'quasi_newton',
            'quasi_newton_model': 'quasi_newton',
            'BFGSModel': 'quasi_newton',
            'bfgs_model': 'quasi_newton',
            'LBFGSModel': 'quasi_newton',
            'l_bfgs_model': 'quasi_newton',
            'SubgradientModel': 'subgradient',
            'subgradient_model': 'subgradient',
            'BaseSubgradient': 'subgradient',
            'SubgradientConstantStepSize': 'subgradient',
            'SubgradientSquareSummable': 'subgradient',
        }
        
        optimizer_name = class_mappings.get(legacy_model_class, legacy_model_class.lower())
        
        return cls.create_optimizer(optimizer_name, **legacy_params)

# Convenience functions for easier access
def create_optimizer(optimizer_name: str, **kwargs) -> BaseOptimizer:
    """Convenience function - alias for OptimizerFactory.create_optimizer"""
    return OptimizerFactory.create_optimizer(optimizer_name, **kwargs)

def list_optimizers() -> list[str]:
    """Convenience function - alias for OptimizerFactory.list_available_optimizers"""
    return OptimizerFactory.list_available_optimizers()

def get_optimizer_info(optimizer_name: str) -> Dict[str, Any]:
    """Convenience function - alias for OptimizerFactory.get_optimizer_info"""
    return OptimizerFactory.get_optimizer_info(optimizer_name)