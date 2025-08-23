"""
Model Loader and Prediction System
Load saved models và make predictions với new data
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

class ModelLoader:
    """Load và manage saved models for prediction"""
    
    def __init__(self, base_algorithms_dir: str = "data/03_algorithms"):
        self.base_dir = Path(base_algorithms_dir)
    
    def list_available_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        List all available saved models organized by algorithm
        
        Returns:
            Dict with algorithm names as keys and list of experiments as values
        """
        models = {}
        
        if not self.base_dir.exists():
            return models
        
        for algorithm_dir in self.base_dir.iterdir():
            if algorithm_dir.is_dir():
                algorithm_name = algorithm_dir.name
                models[algorithm_name] = []
                
                for experiment_dir in algorithm_dir.iterdir():
                    if experiment_dir.is_dir():
                        model_info = self._get_model_info(experiment_dir)
                        if model_info:
                            models[algorithm_name].append(model_info)
        
        return models
    
    def _get_model_info(self, experiment_dir: Path) -> Optional[Dict[str, Any]]:
        """Get information about a saved model"""
        results_file = experiment_dir / "results.json"
        state_file = experiment_dir / "model_state.json"
        complete_model = experiment_dir / "model_complete.pkl"
        
        if not (results_file.exists() or state_file.exists()):
            return None
        
        info = {
            'experiment_name': experiment_dir.name,
            'experiment_path': str(experiment_dir),
            'has_complete_model': complete_model.exists(),
            'has_enhanced_state': state_file.exists(),
            'created_time': experiment_dir.stat().st_mtime,
            'prediction_ready': False
        }
        
        # Load basic info từ results.json
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                info.update({
                    'algorithm': results.get('algorithm', 'Unknown'),
                    'loss_function': results.get('loss_function', 'Unknown'),
                    'parameters': results.get('parameters', {}),
                    'training_time': results.get('training_time', 0),
                    'converged': results.get('convergence', {}).get('converged', False),
                    'final_loss': results.get('convergence', {}).get('final_loss', None)
                })
            except Exception as e:
                print(f"Warning: Could not load results.json from {experiment_dir}: {e}")
        
        # Load enhanced state info
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                info.update({
                    'prediction_ready': state.get('training_completed', False),
                    'n_features': state.get('feature_info', {}).get('n_features', None),
                    'has_preprocessing_info': 'preprocessing_info' in state
                })
                
                # Add algorithm-specific info
                if 'sparsity_info' in state.get('regularization_state', {}):
                    info['sparsity_ratio'] = state['regularization_state']['sparsity_info'].get('final_sparsity', 0)
                
            except Exception as e:
                print(f"Warning: Could not load model_state.json from {experiment_dir}: {e}")
        
        return info
    
    def load_model(self, experiment_path: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a saved model for prediction
        
        Args:
            experiment_path: Path to experiment directory
            
        Returns:
            Tuple of (model_instance, model_state)
        """
        exp_dir = Path(experiment_path)
        
        # Try loading complete model first (fastest)
        complete_model_path = exp_dir / "model_complete.pkl"
        if complete_model_path.exists():
            try:
                model_data = joblib.load(complete_model_path)
                return model_data['model_instance'], model_data['model_state']
            except Exception as e:
                print(f"Warning: Could not load complete model, trying reconstruction: {e}")
        
        # Fallback: reconstruct from components
        return self._reconstruct_model(exp_dir)
    
    def _reconstruct_model(self, exp_dir: Path) -> Tuple[Any, Dict[str, Any]]:
        """Reconstruct model from saved components"""
        # Load model state
        state_file = exp_dir / "model_state.json"
        if not state_file.exists():
            raise FileNotFoundError(f"No model state found in {exp_dir}")
        
        with open(state_file, 'r') as f:
            model_state = json.load(f)
        
        # Import the appropriate model class
        algorithm_name = model_state.get('algorithm_name', '').lower()
        model_class = self._get_model_class(algorithm_name)
        
        # Create model instance
        params = model_state.get('parameters', {})
        model = model_class(**params)
        
        # Load weights
        weights_file = exp_dir / "weights.npy"
        if weights_file.exists():
            model.weights = np.load(weights_file)
        elif model_state.get('weights'):
            model.weights = np.array(model_state['weights'])
        
        # Load bias if exists
        bias_file = exp_dir / "bias.npy"
        if bias_file.exists():
            model.bias = np.load(bias_file)
        elif model_state.get('bias') is not None:
            model.bias = model_state['bias']
        
        return model, model_state
    
    def _get_model_class(self, algorithm_name: str):
        """Get model class từ algorithm name"""
        # Import here để avoid circular imports
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        
        if 'gradientdescent' in algorithm_name:
            from algorithms.gradient_descent.gradient_descent_model import GradientDescentModel
            return GradientDescentModel
        elif 'momentumgd' in algorithm_name:
            from algorithms.gradient_descent.momentum_gd_model import MomentumGDModel  
            return MomentumGDModel
        elif 'newton' in algorithm_name:
            from algorithms.newton_method.newton_model import NewtonModel
            return NewtonModel
        elif 'sgd' in algorithm_name:
            from algorithms.stochastic_gd.sgd_model import SGDModel
            return SGDModel
        elif 'proximal' in algorithm_name:
            from algorithms.proximal_gd.proximal_gd_model import ProximalGDModel
            return ProximalGDModel
        elif 'quasi' in algorithm_name:
            from algorithms.quasi_newton.quasi_newton_model import QuasiNewtonModel
            return QuasiNewtonModel
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    def predict_from_file(self, experiment_path: str, input_file: str, 
                         output_file: str = None) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load model và predict từ CSV file
        
        Args:
            experiment_path: Path to saved experiment
            input_file: CSV file with data to predict
            output_file: Output file cho predictions (optional)
            
        Returns:
            Tuple of (predictions array, results DataFrame)
        """
        # Load model
        model, model_state = self.load_model(experiment_path)
        
        # Load new data
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Read CSV
        df = pd.read_csv(input_file)
        X_new = df.values
        
        # Apply preprocessing nếu có
        preprocessing_info = model_state.get('preprocessing_info')
        if preprocessing_info and preprocessing_info.get('requires_preprocessing', True):
            X_new = self._apply_preprocessing(X_new, preprocessing_info)
        
        # Make predictions
        predictions = model.predict(X_new)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'prediction': predictions
        })
        
        # Add original data
        for i, col in enumerate(df.columns):
            results_df[f'feature_{i+1}_{col}'] = df[col]
        
        # Add model info
        results_df['model_algorithm'] = model_state.get('algorithm_name', 'Unknown')
        results_df['experiment_name'] = Path(experiment_path).name
        
        # Save results nếu specified
        if output_file:
            results_df.to_csv(output_file, index=False)
            print(f"Predictions saved to: {output_file}")
        
        return predictions, results_df
    
    def _apply_preprocessing(self, X: np.ndarray, preprocessing_info: Dict[str, Any]) -> np.ndarray:
        """Apply preprocessing từ saved info"""
        X_processed = X.copy()
        
        # Extract preprocessing parameters
        feature_means = np.array(preprocessing_info.get('feature_means', 0))
        feature_stds = np.array(preprocessing_info.get('feature_stds', 1))
        
        # Handle shape mismatch
        if len(feature_means) != X.shape[1]:
            print(f"Warning: Feature count mismatch. Expected {len(feature_means)}, got {X.shape[1]}")
            # Truncate or pad as needed
            min_features = min(len(feature_means), X.shape[1])
            feature_means = feature_means[:min_features]
            feature_stds = feature_stds[:min_features]
            X_processed = X_processed[:, :min_features]
        
        # Normalize
        X_processed = (X_processed - feature_means) / np.maximum(feature_stds, 1e-8)
        
        return X_processed
    
    def predict_single_sample(self, experiment_path: str, 
                             features: Union[List[float], np.ndarray]) -> float:
        """
        Predict single sample
        
        Args:
            experiment_path: Path to saved experiment  
            features: Feature values
            
        Returns:
            Single prediction value
        """
        # Load model
        model, model_state = self.load_model(experiment_path)
        
        # Prepare input
        X_new = np.array(features).reshape(1, -1)
        
        # Apply preprocessing
        preprocessing_info = model_state.get('preprocessing_info')
        if preprocessing_info:
            X_new = self._apply_preprocessing(X_new, preprocessing_info)
        
        # Make prediction
        prediction = model.predict(X_new)
        
        return float(prediction[0])
    
    def get_model_summary(self, experiment_path: str) -> Dict[str, Any]:
        """Get comprehensive summary of a saved model"""
        exp_dir = Path(experiment_path)
        
        # Basic info
        model_info = self._get_model_info(exp_dir)
        if not model_info:
            raise ValueError(f"Invalid model path: {experiment_path}")
        
        # Try to load model state for more details
        try:
            _, model_state = self.load_model(experiment_path)
            model_info['detailed_state'] = model_state
            
            # Add summary statistics
            if model_state.get('preprocessing_info'):
                prep_info = model_state['preprocessing_info']
                model_info['data_summary'] = {
                    'input_shape': prep_info.get('input_shape', 'Unknown'),
                    'feature_statistics': {
                        'means_range': f"{min(prep_info.get('feature_means', [0])):.4f} - {max(prep_info.get('feature_means', [0])):.4f}",
                        'stds_range': f"{min(prep_info.get('feature_stds', [1])):.4f} - {max(prep_info.get('feature_stds', [1])):.4f}"
                    }
                }
                
                if 'target_info' in prep_info:
                    target_info = prep_info['target_info']
                    model_info['target_summary'] = {
                        'mean': target_info.get('target_mean', 'Unknown'),
                        'std': target_info.get('target_std', 'Unknown'),
                        'range': f"{target_info.get('target_min', 'Unknown')} - {target_info.get('target_max', 'Unknown')}"
                    }
        
        except Exception as e:
            model_info['load_error'] = str(e)
        
        return model_info
    
    def compare_models(self, experiment_paths: List[str]) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            experiment_paths: List of paths to experiments
            
        Returns:
            DataFrame comparing models
        """
        comparisons = []
        
        for path in experiment_paths:
            try:
                summary = self.get_model_summary(path)
                comparison = {
                    'experiment_name': summary['experiment_name'],
                    'algorithm': summary.get('algorithm', 'Unknown'),
                    'loss_function': summary.get('loss_function', 'Unknown'),
                    'converged': summary.get('converged', False),
                    'final_loss': summary.get('final_loss', None),
                    'training_time': summary.get('training_time', 0),
                    'prediction_ready': summary.get('prediction_ready', False),
                    'n_features': summary.get('n_features', None)
                }
                
                # Add algorithm-specific metrics
                if 'sparsity_ratio' in summary:
                    comparison['sparsity_ratio'] = summary['sparsity_ratio']
                
                comparisons.append(comparison)
                
            except Exception as e:
                print(f"Warning: Could not process {path}: {e}")
        
        return pd.DataFrame(comparisons)

# Convenience functions
def load_model_for_prediction(experiment_path: str):
    """Convenience function để load model"""
    loader = ModelLoader()
    return loader.load_model(experiment_path)

def predict_from_csv(experiment_path: str, csv_file: str, output_file: str = None):
    """Convenience function để predict từ CSV"""
    loader = ModelLoader()
    return loader.predict_from_file(experiment_path, csv_file, output_file)

def list_all_models():
    """Convenience function để list models"""
    loader = ModelLoader()
    return loader.list_available_models()