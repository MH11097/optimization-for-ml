"""
Library Adaptive Experiment 801: PyTorch Adam with Learning Rate 0.001
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(project_root)

from src.optimization.external.library_adaptive.pytorch_adaptive import create_pytorch_adam
from src.algorithm_comparator import AlgorithmComparator

def main():
    """Run PyTorch Adam experiment."""
    
    # Algorithm configuration
    optimizer = create_pytorch_adam(
        learning_rate=0.001,
        betas=(0.9, 0.999),
        batch_size=32,
        loss_type='ols',
        max_iterations=300,
        convergence_tolerance=1e-4,
        random_state=42
    )
    
    # Experiment configuration
    experiment_config = {
        'algorithm_name': 'Library_Adam_PyTorch_LR_001',
        'output_dir': 'data/03_algorithms/library_adaptive/801_library_adam_pytorch_lr_001',
        'description': 'PyTorch Adam with learning rate 0.001, betas (0.9, 0.999), batch size 32',
        'algorithm_type': 'library_adaptive',
        'library_info': {
            'library': 'pytorch',
            'algorithm': 'Adam',
            'variant': 'standard',
            'learning_rate': 0.001,
            'betas': (0.9, 0.999),
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