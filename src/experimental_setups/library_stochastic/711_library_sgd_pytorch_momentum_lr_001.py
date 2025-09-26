"""
Library Stochastic Experiment 711: PyTorch SGD with Momentum, Learning Rate 0.001
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(project_root)

from src.optimization.external.library_stochastic.pytorch_sgd_variants import create_pytorch_sgd_momentum
from src.algorithm_comparator import AlgorithmComparator

def main():
    """Run PyTorch SGD with momentum experiment."""
    
    # Algorithm configuration
    optimizer = create_pytorch_sgd_momentum(
        learning_rate=0.001,
        momentum=0.9,
        batch_size=32,
        loss_type='ols',
        max_iterations=500,
        convergence_tolerance=1e-4,
        random_state=42
    )
    
    # Experiment configuration
    experiment_config = {
        'algorithm_name': 'Library_SGD_PyTorch_Momentum_LR_001',
        'output_dir': 'data/03_algorithms/library_stochastic/711_library_sgd_pytorch_momentum_lr_001',
        'description': 'PyTorch SGD with momentum 0.9, learning rate 0.001, batch size 32',
        'algorithm_type': 'library_stochastic',
        'library_info': {
            'library': 'pytorch',
            'algorithm': 'SGD',
            'variant': 'momentum',
            'learning_rate': 0.001,
            'momentum': 0.9,
            'batch_size': 32
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