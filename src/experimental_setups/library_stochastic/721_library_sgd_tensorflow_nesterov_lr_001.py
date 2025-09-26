"""
Library Stochastic Experiment 721: TensorFlow SGD with Nesterov Momentum, Learning Rate 0.001
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(project_root)

from src.optimization.external.library_stochastic.tensorflow_sgd_variants import create_tensorflow_sgd_nesterov
from src.algorithm_comparator import AlgorithmComparator

def main():
    """Run TensorFlow SGD with Nesterov momentum experiment."""
    
    # Algorithm configuration
    optimizer = create_tensorflow_sgd_nesterov(
        learning_rate=0.001,
        momentum=0.9,
        batch_size=32,
        epochs=200,
        loss_type='ols',
        max_iterations=200,
        convergence_tolerance=1e-4,
        random_state=42
    )
    
    # Experiment configuration
    experiment_config = {
        'algorithm_name': 'Library_SGD_TensorFlow_Nesterov_LR_001',
        'output_dir': 'data/03_algorithms/library_stochastic/721_library_sgd_tensorflow_nesterov_lr_001',
        'description': 'TensorFlow SGD with Nesterov momentum 0.9, learning rate 0.001, batch size 32',
        'algorithm_type': 'library_stochastic',
        'library_info': {
            'library': 'tensorflow',
            'algorithm': 'SGD',
            'variant': 'nesterov',
            'learning_rate': 0.001,
            'momentum': 0.9,
            'nesterov': True,
            'batch_size': 32,
            'epochs': 200
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