"""
Library Adaptive Experiment 841: JAX Adam with Learning Rate 0.001
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(project_root)

from src.optimization.external.library_adaptive.jax_adaptive import create_jax_adam
from src.algorithm_comparator import AlgorithmComparator

def main():
    """Run JAX Adam experiment."""
    
    # Algorithm configuration
    optimizer = create_jax_adam(
        learning_rate=0.001,
        b1=0.9,
        b2=0.999,
        batch_size=32,
        loss_type='ols',
        max_iterations=300,
        convergence_tolerance=1e-4,
        random_state=42
    )
    
    # Experiment configuration
    experiment_config = {
        'algorithm_name': 'Library_Adam_JAX_LR_001',
        'output_dir': 'data/03_algorithms/library_adaptive/841_library_adam_jax_lr_001',
        'description': 'JAX Adam with learning rate 0.001, b1=0.9, b2=0.999, batch size 32, JIT compiled',
        'algorithm_type': 'library_adaptive',
        'library_info': {
            'library': 'jax',
            'algorithm': 'Adam',
            'variant': 'standard',
            'learning_rate': 0.001,
            'b1': 0.9,
            'b2': 0.999,
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