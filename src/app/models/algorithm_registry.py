"""
Algorithm Registry System
Định nghĩa tất cả algorithms, parameters và metadata cho Flask app
"""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class ParameterType(Enum):
    FLOAT = "float"
    INTEGER = "int"  
    STRING = "string"
    BOOLEAN = "bool"
    CHOICE = "choice"

@dataclass
class Parameter:
    name: str
    param_type: ParameterType
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    default_value: Any = None
    choices: Optional[List[str]] = None
    description: str = ""
    step: Optional[float] = None

@dataclass
class AlgorithmVariant:
    name: str
    display_name: str
    loss_function: str
    parameters: List[Parameter]
    description: str = ""
    setup_template: str = ""

@dataclass
class Algorithm:
    name: str
    display_name: str
    category: str
    variants: List[AlgorithmVariant]
    model_class: str
    base_dir: str
    description: str = ""

# Algorithm Registry
ALGORITHMS = {
    "gradient_descent": Algorithm(
        name="gradient_descent",
        display_name="Gradient Descent", 
        category="First-Order Methods",
        model_class="GradientDescentModel",
        base_dir="data/03_algorithms/gradient_descent",
        description="Classical gradient descent with various loss functions",
        variants=[
            AlgorithmVariant(
                name="ols",
                display_name="OLS (Ordinary Least Squares)",
                loss_function="ols", 
                setup_template="setup_gd_ols_lr_{learning_rate}",
                parameters=[
                    Parameter(
                        name="learning_rate",
                        param_type=ParameterType.CHOICE,
                        choices=["0.01", "0.1", "0.5"],
                        default_value="0.1",
                        description="Learning rate for gradient updates"
                    ),
                    Parameter(
                        name="so_lan_thu",
                        param_type=ParameterType.INTEGER,
                        min_value=100,
                        max_value=2000,
                        default_value=500,
                        description="Maximum number of iterations"
                    ),
                    Parameter(
                        name="diem_dung",
                        param_type=ParameterType.CHOICE,
                        choices=["1e-3", "1e-4", "1e-5", "1e-6"],
                        default_value="1e-5",
                        description="Convergence tolerance"
                    )
                ]
            ),
            AlgorithmVariant(
                name="ridge",
                display_name="Ridge Regression",
                loss_function="ridge",
                setup_template="setup_gd_ridge_lr_{learning_rate}_reg_{regularization}",
                parameters=[
                    Parameter(
                        name="learning_rate", 
                        param_type=ParameterType.CHOICE,
                        choices=["0.001", "0.01", "0.1"],
                        default_value="0.001",
                        description="Learning rate for gradient updates"
                    ),
                    Parameter(
                        name="regularization",
                        param_type=ParameterType.CHOICE,
                        choices=["0.001", "0.01", "0.1"],
                        default_value="0.001", 
                        description="L2 regularization strength"
                    ),
                    Parameter(
                        name="so_lan_thu",
                        param_type=ParameterType.INTEGER,
                        min_value=100,
                        max_value=2000,
                        default_value=500,
                        description="Maximum number of iterations"
                    ),
                    Parameter(
                        name="diem_dung",
                        param_type=ParameterType.CHOICE,
                        choices=["1e-3", "1e-4", "1e-5", "1e-6"],
                        default_value="1e-5",
                        description="Convergence tolerance"
                    )
                ]
            ),
            AlgorithmVariant(
                name="lasso",
                display_name="Lasso Regression", 
                loss_function="lasso",
                setup_template="setup_gd_lasso_lr_{learning_rate}_reg_{regularization}",
                parameters=[
                    Parameter(
                        name="learning_rate",
                        param_type=ParameterType.CHOICE, 
                        choices=["0.001", "0.01", "0.1"],
                        default_value="0.001",
                        description="Learning rate for gradient updates"
                    ),
                    Parameter(
                        name="regularization",
                        param_type=ParameterType.CHOICE,
                        choices=["0.01", "0.1", "1.0"],
                        default_value="0.1",
                        description="L1 regularization strength"
                    ),
                    Parameter(
                        name="so_lan_thu", 
                        param_type=ParameterType.INTEGER,
                        min_value=100,
                        max_value=2000,
                        default_value=500,
                        description="Maximum number of iterations"
                    ),
                    Parameter(
                        name="diem_dung",
                        param_type=ParameterType.CHOICE,
                        choices=["1e-3", "1e-4", "1e-5", "1e-6"],
                        default_value="1e-5",
                        description="Convergence tolerance"
                    )
                ]
            )
        ]
    ),

    "momentum_gd": Algorithm(
        name="momentum_gd",
        display_name="Momentum Gradient Descent",
        category="First-Order Methods",
        model_class="MomentumGDModel",
        base_dir="data/03_algorithms/gradient_descent",
        description="Gradient descent with momentum for faster convergence",
        variants=[
            AlgorithmVariant(
                name="ols",
                display_name="Momentum GD - OLS",
                loss_function="ols",
                setup_template="setup_momentum_ols_lr_{learning_rate}_mom_{momentum}",
                parameters=[
                    Parameter(
                        name="learning_rate",
                        param_type=ParameterType.CHOICE,
                        choices=["0.01", "0.1", "0.5"],
                        default_value="0.1",
                        description="Learning rate for gradient updates"
                    ),
                    Parameter(
                        name="momentum",
                        param_type=ParameterType.CHOICE,
                        choices=["0.5", "0.9", "0.95", "0.99"],
                        default_value="0.9",
                        description="Momentum coefficient"
                    ),
                    Parameter(
                        name="so_lan_thu",
                        param_type=ParameterType.INTEGER,
                        min_value=100,
                        max_value=2000, 
                        default_value=500,
                        description="Maximum number of iterations"
                    ),
                    Parameter(
                        name="diem_dung",
                        param_type=ParameterType.CHOICE,
                        choices=["1e-3", "1e-4", "1e-5", "1e-6"],
                        default_value="1e-5",
                        description="Convergence tolerance"
                    )
                ]
            )
        ]
    ),

    "newton_method": Algorithm(
        name="newton_method",
        display_name="Newton Method",
        category="Second-Order Methods", 
        model_class="NewtonModel",
        base_dir="data/03_algorithms/newton_method",
        description="Newton's method using second-order derivatives",
        variants=[
            AlgorithmVariant(
                name="ols_pure",
                display_name="Pure Newton - OLS",
                loss_function="ols",
                setup_template="setup_newton_ols_pure",
                parameters=[
                    Parameter(
                        name="regularization",
                        param_type=ParameterType.CHOICE,
                        choices=["0.0", "0.001", "0.01"],
                        default_value="0.0",
                        description="Regularization parameter"
                    ),
                    Parameter(
                        name="numerical_regularization",
                        param_type=ParameterType.CHOICE,
                        choices=["1e-8", "1e-6", "1e-4"],
                        default_value="1e-6", 
                        description="Numerical regularization for Hessian"
                    ),
                    Parameter(
                        name="so_lan_thu",
                        param_type=ParameterType.INTEGER,
                        min_value=10,
                        max_value=100,
                        default_value=50,
                        description="Maximum number of iterations"
                    ),
                    Parameter(
                        name="diem_dung",
                        param_type=ParameterType.CHOICE,
                        choices=["1e-6", "1e-8", "1e-10"],
                        default_value="1e-8",
                        description="Convergence tolerance"
                    )
                ]
            ),
            AlgorithmVariant(
                name="ols_damped",
                display_name="Damped Newton - OLS", 
                loss_function="ols",
                setup_template="setup_newton_ols_damped",
                parameters=[
                    Parameter(
                        name="regularization",
                        param_type=ParameterType.CHOICE,
                        choices=["0.0", "0.001", "0.01"],
                        default_value="0.0",
                        description="Regularization parameter"
                    ),
                    Parameter(
                        name="numerical_regularization",
                        param_type=ParameterType.CHOICE,
                        choices=["1e-8", "1e-6", "1e-4"],
                        default_value="1e-6",
                        description="Numerical regularization for Hessian"
                    ),
                    Parameter(
                        name="so_lan_thu",
                        param_type=ParameterType.INTEGER,
                        min_value=10,
                        max_value=100,
                        default_value=50,
                        description="Maximum number of iterations"
                    ),
                    Parameter(
                        name="diem_dung", 
                        param_type=ParameterType.CHOICE,
                        choices=["1e-6", "1e-8", "1e-10"],
                        default_value="1e-8",
                        description="Convergence tolerance"
                    )
                ]
            ),
            AlgorithmVariant(
                name="ridge_pure",
                display_name="Pure Newton - Ridge",
                loss_function="ridge",
                setup_template="setup_newton_ridge_pure",
                parameters=[
                    Parameter(
                        name="regularization",
                        param_type=ParameterType.CHOICE,
                        choices=["0.001", "0.01", "0.1"],
                        default_value="0.01",
                        description="Ridge regularization parameter"
                    ),
                    Parameter(
                        name="numerical_regularization", 
                        param_type=ParameterType.CHOICE,
                        choices=["1e-8", "1e-6", "1e-4"],
                        default_value="1e-6",
                        description="Numerical regularization for Hessian"
                    ),
                    Parameter(
                        name="so_lan_thu",
                        param_type=ParameterType.INTEGER,
                        min_value=10,
                        max_value=100,
                        default_value=50,
                        description="Maximum number of iterations"
                    ),
                    Parameter(
                        name="diem_dung",
                        param_type=ParameterType.CHOICE,
                        choices=["1e-6", "1e-8", "1e-10"],
                        default_value="1e-8",
                        description="Convergence tolerance"
                    )
                ]
            ),
            AlgorithmVariant(
                name="ridge_damped",
                display_name="Damped Newton - Ridge",
                loss_function="ridge", 
                setup_template="setup_newton_ridge_damped",
                parameters=[
                    Parameter(
                        name="regularization",
                        param_type=ParameterType.CHOICE,
                        choices=["0.001", "0.01", "0.1"],
                        default_value="0.01",
                        description="Ridge regularization parameter"
                    ),
                    Parameter(
                        name="numerical_regularization",
                        param_type=ParameterType.CHOICE,
                        choices=["1e-8", "1e-6", "1e-4"],
                        default_value="1e-6",
                        description="Numerical regularization for Hessian"
                    ),
                    Parameter(
                        name="so_lan_thu",
                        param_type=ParameterType.INTEGER,
                        min_value=10,
                        max_value=100,
                        default_value=50,
                        description="Maximum number of iterations"
                    ),
                    Parameter(
                        name="diem_dung",
                        param_type=ParameterType.CHOICE,
                        choices=["1e-6", "1e-8", "1e-10"], 
                        default_value="1e-8",
                        description="Convergence tolerance"
                    )
                ]
            )
        ]
    ),

    "stochastic_gd": Algorithm(
        name="stochastic_gd",
        display_name="Stochastic Gradient Descent",
        category="Stochastic Methods",
        model_class="SGDModel",
        base_dir="data/03_algorithms/stochastic_gd",
        description="Stochastic gradient descent with configurable batch sizes",
        variants=[
            AlgorithmVariant(
                name="batch_1",
                display_name="Pure SGD (Batch Size 1)",
                loss_function="mse",
                setup_template="setup_sgd_batch_1",
                parameters=[
                    Parameter(
                        name="learning_rate",
                        param_type=ParameterType.CHOICE,
                        choices=["0.001", "0.01", "0.1"],
                        default_value="0.01",
                        description="Learning rate for gradient updates"
                    ),
                    Parameter(
                        name="so_epochs",
                        param_type=ParameterType.INTEGER,
                        min_value=50,
                        max_value=500,
                        default_value=100,
                        description="Number of epochs"
                    ),
                    Parameter(
                        name="batch_size",
                        param_type=ParameterType.INTEGER,
                        min_value=1,
                        max_value=1,
                        default_value=1,
                        description="Batch size (fixed at 1 for pure SGD)"
                    ),
                    Parameter(
                        name="random_state",
                        param_type=ParameterType.INTEGER,
                        min_value=1,
                        max_value=999,
                        default_value=42,
                        description="Random seed for reproducibility"
                    )
                ]
            ),
            AlgorithmVariant(
                name="batch_16", 
                display_name="Mini-batch SGD (Batch Size 16)",
                loss_function="mse",
                setup_template="setup_sgd_batch_16",
                parameters=[
                    Parameter(
                        name="learning_rate",
                        param_type=ParameterType.CHOICE,
                        choices=["0.001", "0.01", "0.1"],
                        default_value="0.01",
                        description="Learning rate for gradient updates"
                    ),
                    Parameter(
                        name="so_epochs",
                        param_type=ParameterType.INTEGER,
                        min_value=50,
                        max_value=500,
                        default_value=100,
                        description="Number of epochs"
                    ),
                    Parameter(
                        name="batch_size",
                        param_type=ParameterType.INTEGER,
                        min_value=16,
                        max_value=16,
                        default_value=16,
                        description="Batch size (fixed at 16)"
                    ),
                    Parameter(
                        name="random_state",
                        param_type=ParameterType.INTEGER,
                        min_value=1,
                        max_value=999,
                        default_value=42,
                        description="Random seed for reproducibility"
                    )
                ]
            ),
            AlgorithmVariant(
                name="batch_32",
                display_name="Mini-batch SGD (Batch Size 32)",
                loss_function="mse",
                setup_template="setup_sgd_batch_32",
                parameters=[
                    Parameter(
                        name="learning_rate",
                        param_type=ParameterType.CHOICE,
                        choices=["0.001", "0.01", "0.1"],
                        default_value="0.01",
                        description="Learning rate for gradient updates"
                    ),
                    Parameter(
                        name="so_epochs",
                        param_type=ParameterType.INTEGER,
                        min_value=50,
                        max_value=500,
                        default_value=100,
                        description="Number of epochs"
                    ),
                    Parameter(
                        name="batch_size",
                        param_type=ParameterType.INTEGER,
                        min_value=32,
                        max_value=32,
                        default_value=32,
                        description="Batch size (fixed at 32)"
                    ),
                    Parameter(
                        name="random_state",
                        param_type=ParameterType.INTEGER,
                        min_value=1,
                        max_value=999,
                        default_value=42,
                        description="Random seed for reproducibility"
                    )
                ]
            ),
            AlgorithmVariant(
                name="batch_64",
                display_name="Mini-batch SGD (Batch Size 64)",
                loss_function="mse",
                setup_template="setup_sgd_batch_64", 
                parameters=[
                    Parameter(
                        name="learning_rate",
                        param_type=ParameterType.CHOICE,
                        choices=["0.001", "0.01", "0.1"],
                        default_value="0.01",
                        description="Learning rate for gradient updates"
                    ),
                    Parameter(
                        name="so_epochs",
                        param_type=ParameterType.INTEGER,
                        min_value=50,
                        max_value=500,
                        default_value=100,
                        description="Number of epochs"
                    ),
                    Parameter(
                        name="batch_size",
                        param_type=ParameterType.INTEGER,
                        min_value=64,
                        max_value=64,
                        default_value=64,
                        description="Batch size (fixed at 64)"
                    ),
                    Parameter(
                        name="random_state",
                        param_type=ParameterType.INTEGER,
                        min_value=1,
                        max_value=999,
                        default_value=42,
                        description="Random seed for reproducibility"
                    )
                ]
            )
        ]
    ),

    "proximal_gd": Algorithm(
        name="proximal_gd",
        display_name="Proximal Gradient Descent",
        category="Proximal Methods",
        model_class="ProximalGDModel",
        base_dir="data/03_algorithms/proximal_gd",
        description="Proximal gradient descent for sparse optimization with L1 regularization",
        variants=[
            AlgorithmVariant(
                name="lasso",
                display_name="Proximal GD - Lasso",
                loss_function="lasso",
                setup_template="setup_lasso_lr_{learning_rate}_l1_{lambda_l1}",
                parameters=[
                    Parameter(
                        name="learning_rate",
                        param_type=ParameterType.CHOICE,
                        choices=["0.001", "0.01", "0.1"],
                        default_value="0.01",
                        description="Learning rate for gradient updates"
                    ),
                    Parameter(
                        name="lambda_l1", 
                        param_type=ParameterType.CHOICE,
                        choices=["0.001", "0.01", "0.1", "1.0"],
                        default_value="0.01",
                        description="L1 regularization parameter"
                    ),
                    Parameter(
                        name="so_lan_thu",
                        param_type=ParameterType.INTEGER,
                        min_value=500,
                        max_value=2000,
                        default_value=1000,
                        description="Maximum number of iterations"
                    ),
                    Parameter(
                        name="diem_dung",
                        param_type=ParameterType.CHOICE,
                        choices=["1e-4", "1e-5", "1e-6", "1e-7"],
                        default_value="1e-6",
                        description="Convergence tolerance"
                    )
                ]
            )
        ]
    ),

    "quasi_newton": Algorithm(
        name="quasi_newton",
        display_name="Quasi-Newton (BFGS)",
        category="Quasi-Newton Methods",
        model_class="QuasiNewtonModel",
        base_dir="data/03_algorithms/quasi_newton",
        description="BFGS quasi-Newton method for fast convergence",
        variants=[
            AlgorithmVariant(
                name="ols",
                display_name="BFGS - OLS",
                loss_function="ols",
                setup_template="setup_bfgs_ols",
                parameters=[
                    Parameter(
                        name="so_lan_thu",
                        param_type=ParameterType.INTEGER,
                        min_value=50,
                        max_value=500,
                        default_value=200,
                        description="Maximum number of iterations"
                    ),
                    Parameter(
                        name="diem_dung",
                        param_type=ParameterType.CHOICE,
                        choices=["1e-6", "1e-8", "1e-10"],
                        default_value="1e-8",
                        description="Convergence tolerance"
                    ),
                    Parameter(
                        name="armijo_c1",
                        param_type=ParameterType.CHOICE,
                        choices=["1e-4", "1e-3", "1e-2"],
                        default_value="1e-4",
                        description="Armijo condition parameter"
                    ),
                    Parameter(
                        name="wolfe_c2",
                        param_type=ParameterType.CHOICE,
                        choices=["0.1", "0.9", "0.99"],
                        default_value="0.9",
                        description="Wolfe condition parameter"
                    )
                ]
            ),
            AlgorithmVariant(
                name="ridge",
                display_name="BFGS - Ridge",
                loss_function="ridge",
                setup_template="setup_bfgs_ridge",
                parameters=[
                    Parameter(
                        name="regularization",
                        param_type=ParameterType.CHOICE,
                        choices=["0.001", "0.01", "0.1"],
                        default_value="0.01",
                        description="Ridge regularization parameter"
                    ),
                    Parameter(
                        name="so_lan_thu",
                        param_type=ParameterType.INTEGER,
                        min_value=50,
                        max_value=500,
                        default_value=200,
                        description="Maximum number of iterations"
                    ),
                    Parameter(
                        name="diem_dung",
                        param_type=ParameterType.CHOICE,
                        choices=["1e-6", "1e-8", "1e-10"],
                        default_value="1e-8",
                        description="Convergence tolerance"
                    ),
                    Parameter(
                        name="armijo_c1",
                        param_type=ParameterType.CHOICE,
                        choices=["1e-4", "1e-3", "1e-2"],
                        default_value="1e-4",
                        description="Armijo condition parameter"
                    ),
                    Parameter(
                        name="wolfe_c2",
                        param_type=ParameterType.CHOICE,
                        choices=["0.1", "0.9", "0.99"],
                        default_value="0.9",
                        description="Wolfe condition parameter"
                    )
                ]
            )
        ]
    )
}

def get_algorithm(name: str) -> Optional[Algorithm]:
    """Get algorithm by name"""
    return ALGORITHMS.get(name)

def get_all_algorithms() -> Dict[str, Algorithm]:
    """Get all algorithms"""
    return ALGORITHMS

def get_algorithms_by_category() -> Dict[str, List[Algorithm]]:
    """Get algorithms grouped by category"""
    categories = {}
    for algorithm in ALGORITHMS.values():
        category = algorithm.category
        if category not in categories:
            categories[category] = []
        categories[category].append(algorithm)
    return categories

def get_algorithm_variant(algorithm_name: str, variant_name: str) -> Optional[AlgorithmVariant]:
    """Get specific algorithm variant"""
    algorithm = get_algorithm(algorithm_name)
    if algorithm:
        for variant in algorithm.variants:
            if variant.name == variant_name:
                return variant
    return None

def format_parameter_value(value: str, param_type: ParameterType) -> Any:
    """Convert string value to proper type"""
    if param_type == ParameterType.FLOAT:
        return float(value)
    elif param_type == ParameterType.INTEGER:
        return int(value)
    elif param_type == ParameterType.BOOLEAN:
        return value.lower() in ('true', '1', 'yes', 'on')
    else:
        return value