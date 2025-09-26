"""
Library Adaptive Experiment 861: Sklearn Adaptive Learning Rate with Initial LR 0.001
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(project_root)

from src.optimization.external.library_adaptive.sklearn_adaptive import create_sklearn_adaptive_standard
from src.algorithm_comparator import AlgorithmComparator

def main():
    """Run Sklearn Adaptive Learning Rate experiment."""
    
    # Algorithm configuration
    optimizer = create_sklearn_adaptive_standard(
        initial_learning_rate=0.001,
        n_iter_no_change=5,
        validation_fraction=0.1,
        loss_type='ols',
        max_iterations=1000,
        convergence_tolerance=1e-4,
        random_state=42
    )
    
    # Experiment configuration
    experiment_config = {
        'algorithm_name': 'Library_Adaptive_Sklearn_LR_001',
        'output_dir': 'data/03_algorithms/library_adaptive/861_library_adaptive_sklearn_lr_001',
        'description': 'Sklearn Adaptive Learning Rate with initial LR 0.001, early stopping, ensemble behavior',
        'algorithm_type': 'library_adaptive',
        'library_info': {
            'library': 'sklearn',
            'algorithm': 'Adaptive_SGD',
            'variant': 'adaptive',
            'initial_learning_rate': 0.001,
            'n_iter_no_change': 5,
            'validation_fraction': 0.1,
            'ensemble_behavior': True
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