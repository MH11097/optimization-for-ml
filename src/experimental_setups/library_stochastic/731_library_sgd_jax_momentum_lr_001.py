"""
Library Stochastic Experiment 731: JAX SGD with Momentum, Learning Rate 0.001
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(project_root)

from src.optimization.external.library_stochastic.jax_sgd import create_jax_sgd_momentum
from src.algorithm_comparator import AlgorithmComparator

def main():
    """Run JAX SGD with momentum experiment."""
    
    # Algorithm configuration
    optimizer = create_jax_sgd_momentum(
        learning_rate=0.001,
        momentum=0.9,
        batch_size=32,
        loss_type='ols',
        max_iterations=300,
        convergence_tolerance=1e-4,
        random_state=42
    )
    
    # Experiment configuration
    experiment_config = {
        'algorithm_name': 'Library_SGD_JAX_Momentum_LR_001',
        'output_dir': 'data/03_algorithms/library_stochastic/731_library_sgd_jax_momentum_lr_001',
        'description': 'JAX SGD with momentum 0.9, learning rate 0.001, batch size 32, JIT compiled',
        'algorithm_type': 'library_stochastic',
        'library_info': {
            'library': 'jax',
            'algorithm': 'SGD',
            'variant': 'momentum',
            'learning_rate': 0.001,
            'momentum': 0.9,
            'batch_size': 32,
            'jit_compiled': True
        }
    }
    
    # Run experiment
    comparator = AlgorithmComparator()
    comparator.run_single_algorithm_experiment(
        optimizer=optimizer,
        experiment_config=experiment_config,
        save_results=True,
        generate_plots=True,
        print_summary=True
    )

if __name__ == "__main__":
    main()