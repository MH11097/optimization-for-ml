"""
Library Adaptive Experiment 821: TensorFlow Adam with Learning Rate 0.001
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(project_root)

from src.optimization.external.library_adaptive.tensorflow_adaptive import create_tensorflow_adam
from src.algorithm_comparator import AlgorithmComparator

def main():
    """Run TensorFlow Adam experiment."""
    
    # Algorithm configuration
    optimizer = create_tensorflow_adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        batch_size=32,
        epochs=200,
        loss_type='ols',
        max_iterations=200,
        convergence_tolerance=1e-4,
        random_state=42
    )
    
    # Experiment configuration
    experiment_config = {
        'algorithm_name': 'Library_Adam_TensorFlow_LR_001',
        'output_dir': 'data/03_algorithms/library_adaptive/821_library_adam_tensorflow_lr_001',
        'description': 'TensorFlow Adam with learning rate 0.001, beta_1=0.9, beta_2=0.999, batch size 32',
        'algorithm_type': 'library_adaptive',
        'library_info': {
            'library': 'tensorflow',
            'algorithm': 'Adam',
            'variant': 'standard',
            'learning_rate': 0.001,
            'beta_1': 0.9,
            'beta_2': 0.999,
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