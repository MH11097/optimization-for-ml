"""
Library Stochastic Experiment 701: Sklearn SGD with Constant Learning Rate 0.001
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(project_root)

from src.optimization.external.library_stochastic.sklearn_sgd_variants import create_sklearn_sgd_constant
from src.algorithm_comparator import AlgorithmComparator

def main():
    """Run Sklearn SGD with constant learning rate experiment."""
    
    # Algorithm configuration
    optimizer = create_sklearn_sgd_constant(
        learning_rate=0.001,
        loss_type='ols',
        max_iterations=1000,
        convergence_tolerance=1e-4,
        random_state=42
    )
    
    # Experiment configuration
    experiment_config = {
        'algorithm_name': 'Library_SGD_Sklearn_Constant_LR_001',
        'output_dir': 'data/03_algorithms/library_stochastic/701_library_sgd_sklearn_constant_lr_001',
        'description': 'Sklearn SGD with constant learning rate 0.001',
        'algorithm_type': 'library_stochastic',
        'library_info': {
            'library': 'sklearn',
            'algorithm': 'SGD',
            'variant': 'constant_lr',
            'learning_rate': 0.001
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