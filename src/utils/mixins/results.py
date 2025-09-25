"""
Results Management Mixins
This module provides mixins for standardized optimization results handling,
providing backward compatibility with the original model_mixins module.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

class ResultsManagementMixin:
    """
    Mixin class for standardized optimization results handling
    Maintains backward compatibility with original utils.model_mixins
    """
    def create_standard_results_dict(self, algorithm_name: str, loss_function: str) -> Dict[str, Any]:
        """
        Create a standardized results dictionary structure
        Args:
            algorithm_name: Name of the algorithm
            loss_function: Loss function used
        Returns:
            Standard results dictionary structure
        """
        if not hasattr(self, 'weights') or self.weights is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        return {
            "algorithm": algorithm_name,
            "loss_function": loss_function.upper(),
            "training_results": {
                "training_time": getattr(self, 'training_time', 0.0),
                "converged": getattr(self, 'converged', False),
                "iterations": getattr(self, 'iterations', 0),
                "final_cost": getattr(self, 'final_cost', None),
                "cost_history": getattr(self, 'cost_history', []),
                "gradient_norm_history": getattr(self, 'gradient_norm_history', []),
            },
            "model_parameters": {
                "weights": self.weights.tolist() if hasattr(self.weights, 'tolist') else self.weights,
                "bias": getattr(self, 'bias', None),
                "num_features": len(self.weights) if self.weights is not None else 0,
            },
            "hyperparameters": self._extract_hyperparameters(),
        }
    def _extract_hyperparameters(self) -> Dict[str, Any]:
        """Extract hyperparameters from the model"""
        hyperparams = {}
        # Common hyperparameters
        if hasattr(self, 'learning_rate'):
            hyperparams['learning_rate'] = self.learning_rate
        if hasattr(self, 'max_iterations'):
            hyperparams['max_iterations'] = self.max_iterations
        if hasattr(self, 'convergence_tolerance'):
            hyperparams['convergence_tolerance'] = self.convergence_tolerance
        if hasattr(self, 'regularization_strength'):
            hyperparams['regularization_strength'] = self.regularization_strength
        if hasattr(self, 'momentum'):
            hyperparams['momentum'] = self.momentum
        if hasattr(self, 'batch_size'):
            hyperparams['batch_size'] = self.batch_size
        return hyperparams
    def save_results_to_file(self, results_data: Dict[str, Any],
                           filename: str, results_dir: Optional[str] = None) -> Path:
        """
        Save results to a JSON file
        Args:
            results_data: Results dictionary to save
            filename: Name of the file (with or without .json extension)
            results_dir: Optional directory to save the file
        Returns:
            Path to the saved file
        """
        if not filename.endswith('.json'):
            filename += '.json'
        if results_dir:
            save_path = Path(results_dir) / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            save_path = Path(filename)
        with open(save_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        return save_path
    def load_results_from_file(self, filepath: str) -> Dict[str, Any]:
        """
        Load results from a JSON file
        Args:
            filepath: Path to the results file
        Returns:
            Loaded results dictionary
        """
        with open(filepath, 'r') as f:
            return json.load(f)
    def add_evaluation_metrics(self, results_data: Dict[str, Any],
                             X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Add evaluation metrics to results data
        Args:
            results_data: Existing results dictionary
            X_test: Test feature matrix
            y_test: Test target vector
        Returns:
            Updated results dictionary with evaluation metrics
        """
        if not hasattr(self, 'predict'):
            return results_data
        try:
            # Make predictions
            y_pred = self.predict(X_test)
            # Calculate metrics
            mse = np.mean((y_test - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_test - y_pred))
            # R-squared
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            results_data["evaluation_metrics"] = {
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "r2_score": float(r2),
                "test_samples": len(y_test)
            }
        except Exception as e:
            print(f"Warning: Could not calculate evaluation metrics: {e}")
        return results_data
    def save_results(self, setup_name: str, algorithm_dir: str = None, base_dir: str = "data/03_algorithms") -> Path:
        """
        Save optimization results to file (backward compatibility method)
        Args:
            setup_name: Setup name (e.g., "130_gd_ols_lr_020")
            algorithm_dir: Algorithm directory name (e.g., "gradient_descent"). If None, uses class name.
            base_dir: Base directory to save results
        Returns:
            Path to the results directory
        Note:
            ML metrics will be automatically included if evaluate() was called before save_results()
        """
        if not hasattr(self, 'weights') or self.weights is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        # Determine algorithm directory name
        if algorithm_dir is None:
            # Fallback to class name for backward compatibility
            algorithm_dir = getattr(self, '__class__', type(self)).__name__.lower()
        # Setup results directory using algorithm_dir/setup_name structure
        results_dir = Path(base_dir) / algorithm_dir / setup_name
        results_dir.mkdir(parents=True, exist_ok=True)
        # Get algorithm name from class name for results metadata
        algorithm_name = self.__class__.__name__
        loss_function = getattr(self, 'loss_type', getattr(self, 'ham_loss', 'unknown'))
        # Get best results if method exists
        best_results = self._get_best_results() if hasattr(self, '_get_best_results') else {}
        # Create comprehensive results dictionary
        results_data = {
            "algorithm": f"{algorithm_name} - {loss_function.upper()}",
            "loss_function": loss_function.upper(),
            "parameters": self._extract_hyperparameters(),
            "training_results": {
                "training_time": getattr(self, 'training_time', 0.0),
                "converged": getattr(self, 'converged', False),
                "final_iteration": getattr(self, 'final_iteration', getattr(self, 'iterations', 0)),
                "total_iterations": getattr(self, 'max_iterations', 0),
                "final_loss": float(self.loss_history[-1]) if hasattr(self, 'loss_history') and self.loss_history else 0.0,
                "final_gradient_norm": float(self.gradient_norms[-1]) if hasattr(self, 'gradient_norms') and self.gradient_norms else 0.0,
            },
            "weights_analysis": {
                "n_features": int(len(self.weights) - 1) if self.weights is not None else 0,
                "n_weights_total": int(len(self.weights)) if self.weights is not None else 0,
                "bias_value": float(self.weights[-1]) if self.weights is not None else 0.0,
                "complete_weight_vector": self.weights.tolist() if self.weights is not None else [],
                "weights_stats": {
                    "min": float(np.min(self.weights[:-1])) if self.weights is not None and len(self.weights) > 1 else 0.0,
                    "max": float(np.max(self.weights[:-1])) if self.weights is not None and len(self.weights) > 1 else 0.0,
                    "mean": float(np.mean(self.weights[:-1])) if self.weights is not None and len(self.weights) > 1 else 0.0,
                    "std": float(np.std(self.weights[:-1])) if self.weights is not None and len(self.weights) > 1 else 0.0
                }
            },
            "convergence_analysis": {
                "iterations_to_converge": getattr(self, 'final_iteration', 0)
                # loss_history and gradient_norm_history moved to training_history.csv
            }
        }
        # Add ML metrics if they were computed by previous evaluate() call
        if hasattr(self, '_latest_ml_metrics') and self._latest_ml_metrics:
            results_data["ml_metrics"] = self._latest_ml_metrics
        elif hasattr(self, 'evaluate'):
            # Add placeholder when evaluate() hasn't been called yet
            results_data["ml_metrics"] = {
                "note": "ML metrics not available - call model.evaluate(X_test, y_test) before save_results()"
            }
        # Add best results if available
        if best_results:
            results_data["training_results"].update({
                "best_iteration": best_results.get('best_iteration', 0),
                "best_loss": float(best_results.get('best_loss', 0.0)),
                "best_gradient_norm": float(best_results.get('best_gradient_norm', 0.0))
            })
        # Add complexity analysis if available
        if hasattr(self, 'get_complexity_summary'):
            complexity_summary = self.get_complexity_summary()
            if complexity_summary:
                results_data["computational_complexity"] = complexity_summary
        # Save results to JSON file
        results_file = results_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        # Save training history if available
        if hasattr(self, 'loss_history') and self.loss_history:
            import pandas as pd
            training_data = {
                'iteration': range(len(self.loss_history)),
                'loss': self.loss_history,
                'gradient_norm': getattr(self, 'gradient_norms', [0] * len(self.loss_history))
            }

            # Add learning rate if available
            if hasattr(self, 'learning_rate_history') and self.learning_rate_history:
                # Ensure learning rate history matches loss history length
                if len(self.learning_rate_history) >= len(self.loss_history):
                    # Sample learning rate at same frequency as loss history
                    lr_sampled = []
                    convergence_freq = getattr(self, 'convergence_check_freq', 1)
                    for i in range(len(self.loss_history)):
                        lr_index = min((i + 1) * convergence_freq - 1, len(self.learning_rate_history) - 1)
                        lr_sampled.append(self.learning_rate_history[lr_index])
                    training_data['learning_rate'] = lr_sampled
                else:
                    # Fallback: use available learning rate data
                    training_data['learning_rate'] = self.learning_rate_history[:len(self.loss_history)]

            training_df = pd.DataFrame(training_data)
            training_df.to_csv(results_dir / "training_history.csv", index=False)
        print(f"[SAVED] Results saved to: {results_dir.absolute()}")
        if best_results:
            best_iter = best_results.get('best_iteration', 0)
            best_grad_norm = best_results.get('best_gradient_norm', 0.0)
            print(f"[BEST] Using best results from iteration {best_iter} (gradient norm: {best_grad_norm:.6f})")
        return results_dir

# Backward compatibility alias
OptimizationResultsMixin = ResultsManagementMixin