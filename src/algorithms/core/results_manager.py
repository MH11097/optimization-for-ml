"""
Standardized results management for optimization algorithms
Used for car price prediction experiments
"""

import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import warnings

class ResultsManager:
    """Manages standardized saving and loading of algorithm results"""
    
    def __init__(self, base_dir: str = "data/03_algorithms"):
        self.base_dir = Path(base_dir)
        
    def create_output_dir(self, algorithm: str, setup: str) -> Path:
        """Create output directory for algorithm/setup"""
        output_dir = self.base_dir / algorithm / setup
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def save_results(self, 
                    algorithm: str,
                    setup: str,
                    config: Dict[str, Any],
                    training_history: pd.DataFrame,
                    predictions: Dict[str, np.ndarray],
                    model_weights: np.ndarray,
                    metrics: Dict[str, float],
                    car_price_metrics: Dict[str, float],
                    training_time: float,
                    memory_usage: Optional[float] = None) -> Path:
        """
        Save all algorithm results in standardized format
        
        Parameters:
        -----------
        algorithm : str
            Algorithm name (e.g., 'gradient_descent')
        setup : str  
            Setup name (e.g., 'fast_setup')
        config : dict
            Algorithm configuration parameters
        training_history : pd.DataFrame
            Training history with columns: iteration, train_loss, test_loss, etc.
        predictions : dict
            Dictionary with 'train' and 'test' prediction arrays
        model_weights : np.ndarray
            Trained model weights
        metrics : dict
            Standard ML metrics (mse, r2, etc.)
        car_price_metrics : dict
            Car price specific metrics (dollar errors, etc.)
        training_time : float
            Training time in seconds
        memory_usage : float, optional
            Peak memory usage in MB
        
        Returns:
        --------
        Path to saved results directory
        """
        output_dir = self.create_output_dir(algorithm, setup)
        
        # 1. Save main results.json
        results_data = {
            "algorithm": algorithm,
            "setup": setup,
            "dataset": "used_cars",
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "results": {
                **metrics,
                "training_time_seconds": training_time,
                "memory_usage_mb": memory_usage,
                "convergence_iteration": len(training_history) if len(training_history) > 0 else None
            },
            "car_price_metrics": car_price_metrics
        }
        
        with open(output_dir / "results.json", "w") as f:
            json.dump(results_data, f, indent=2, default=self._json_serializer)
        
        # 2. Save training history
        training_history.to_csv(output_dir / "training_history.csv", index=False)
        
        # 3. Save predictions
        pred_df = pd.DataFrame({
            'train_actual': predictions.get('y_train_actual', []),
            'train_predicted': predictions.get('y_train_pred', []),
            'test_actual': predictions.get('y_test_actual', []),
            'test_predicted': predictions.get('y_test_pred', [])
        })
        pred_df.to_csv(output_dir / "predictions.csv", index=False)
        
        # 4. Save model weights
        joblib.dump(model_weights, output_dir / "model_weights.pkl")
        
        # 5. Save configuration
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=self._json_serializer)
        
        print(f"✅ Results saved to: {output_dir}")
        return output_dir
    
    def load_results(self, algorithm: str, setup: str) -> Dict[str, Any]:
        """Load algorithm results from saved files"""
        result_dir = self.base_dir / algorithm / setup
        
        if not result_dir.exists():
            raise FileNotFoundError(f"Results not found: {result_dir}")
        
        # Load main results
        with open(result_dir / "results.json", "r") as f:
            results = json.load(f)
        
        # Load training history
        history_file = result_dir / "training_history.csv"
        if history_file.exists():
            results['training_history'] = pd.read_csv(history_file)
        
        # Load predictions
        pred_file = result_dir / "predictions.csv"
        if pred_file.exists():
            results['predictions'] = pd.read_csv(pred_file)
        
        # Load model weights
        weights_file = result_dir / "model_weights.pkl"
        if weights_file.exists():
            results['model_weights'] = joblib.load(weights_file)
        
        return results
    
    def list_available_results(self) -> Dict[str, List[str]]:
        """List all available algorithm results"""
        if not self.base_dir.exists():
            return {}
        
        available = {}
        for algo_dir in self.base_dir.iterdir():
            if not algo_dir.is_dir():
                continue
                
            setups = []
            for setup_dir in algo_dir.iterdir():
                if setup_dir.is_dir() and (setup_dir / "results.json").exists():
                    setups.append(setup_dir.name)
            
            if setups:
                available[algo_dir.name] = sorted(setups)
        
        return available
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# Convenience functions
def save_algorithm_results(algorithm: str, setup: str, **kwargs) -> Path:
    """Convenience function for saving results"""
    manager = ResultsManager()
    return manager.save_results(algorithm, setup, **kwargs)

def load_algorithm_results(algorithm: str, setup: str) -> Dict[str, Any]:
    """Convenience function for loading results"""
    manager = ResultsManager()
    return manager.load_results(algorithm, setup)

def list_available_algorithms() -> Dict[str, List[str]]:
    """Convenience function for listing available results"""
    manager = ResultsManager()
    return manager.list_available_results()