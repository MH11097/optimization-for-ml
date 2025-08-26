"""
Enhanced Model Base Class
Base class với enhanced saving capabilities cho tất cả models
"""

import json
import pickle
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

class EnhancedModelMixin:
    """
    Mixin class để add enhanced saving capabilities cho existing models
    """
    
    def get_model_state_for_prediction(self) -> Dict[str, Any]:
        """
        Get complete model state needed for prediction
        Override này trong từng model class nếu cần
        """
        state = {
            'weights': self.weights.tolist() if hasattr(self, 'weights') and self.weights is not None else None,
            'bias': getattr(self, 'bias', None),
            'algorithm_name': self.__class__.__name__,
            'loss_function': getattr(self, 'ham_loss', None),
            'parameters': self._get_model_parameters(),
            'training_completed': hasattr(self, 'weights') and self.weights is not None,
            'feature_info': {
                'n_features': len(self.weights) if hasattr(self, 'weights') and self.weights is not None else None,
                'requires_preprocessing': True  # Most models need preprocessing
            }
        }
        
        # Add algorithm-specific state
        if hasattr(self, 'momentum') and hasattr(self, 'velocity'):
            state['momentum_state'] = {
                'momentum': self.momentum,
                'velocity': self.velocity.tolist() if self.velocity is not None else None
            }
        
        if hasattr(self, 'lambda_l1'):
            state['regularization_state'] = {
                'lambda_l1': self.lambda_l1,
                'sparsity_info': {
                    'final_sparsity': self.sparsity_history[-1] if hasattr(self, 'sparsity_history') and self.sparsity_history else 0
                }
            }
        
        return state
    
    def _get_model_parameters(self) -> Dict[str, Any]:
        """Get model parameters cho reconstruction"""
        params = {}
        
        # Common parameters
        common_attrs = [
            'learning_rate', 'so_lan_thu', 'diem_dung', 'regularization',
            'momentum', 'lambda_l1', 'lambda_l2', 'batch_size', 'so_epochs',
            'random_state', 'numerical_regularization', 'armijo_c1', 'wolfe_c2'
        ]
        
        for attr in common_attrs:
            if hasattr(self, attr):
                params[attr] = getattr(self, attr)
        
        return params
    
    def enhanced_save_results(self, ten_file: str, base_dir: str = None, 
                            X_train: np.ndarray = None, y_train: np.ndarray = None) -> Path:
        """
        Enhanced save results with complete model state for prediction
        
        Args:
            ten_file: Tên file để lưu
            base_dir: Base directory, nếu None sẽ dùng default từ algorithm
            X_train, y_train: Training data để lưu preprocessing info
        """
        if self.weights is None:
            raise ValueError("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        
        # Setup results directory
        if base_dir is None:
            # Default base dir based on algorithm name
            algorithm_name = self.__class__.__name__.lower().replace('model', '')
            base_dir = f"data/03_algorithms/{algorithm_name}"
        
        results_dir = Path(base_dir) / ten_file
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Call original save_results if exists
        if hasattr(super(), 'save_results'):
            super().save_results(ten_file, base_dir)
        else:
            # Basic save nếu không có original save_results
            self._basic_save_results(results_dir)
        
        # Enhanced saving
        print(f"   Saving enhanced model state to {results_dir}/model_state.json")
        
        # 1. Save complete model state
        model_state = self.get_model_state_for_prediction()
        
        # Add preprocessing info nếu có training data
        if X_train is not None:
            model_state['preprocessing_info'] = {
                'input_shape': X_train.shape,
                'feature_means': np.mean(X_train, axis=0).tolist(),
                'feature_stds': np.std(X_train, axis=0).tolist(),
                'feature_mins': np.min(X_train, axis=0).tolist(), 
                'feature_maxs': np.max(X_train, axis=0).tolist()
            }
            
            if y_train is not None:
                model_state['preprocessing_info']['target_info'] = {
                    'target_mean': float(np.mean(y_train)),
                    'target_std': float(np.std(y_train)),
                    'target_min': float(np.min(y_train)),
                    'target_max': float(np.max(y_train))
                }
        
        with open(results_dir / "model_state.json", 'w') as f:
            json.dump(model_state, f, indent=2, default=str)
        
        # 2. Save complete model object với joblib
        print(f"   Saving complete model to {results_dir}/model_complete.pkl")
        model_data = {
            'model_instance': self,  # Complete model instance
            'model_state': model_state,
            'class_name': self.__class__.__name__,
            'module_name': self.__class__.__module__
        }
        
        joblib.dump(model_data, results_dir / "model_complete.pkl")
        
        # 3. Save weights separately cho easy loading
        if self.weights is not None:
            print(f"   Saving weights to {results_dir}/weights.npy")
            np.save(results_dir / "weights.npy", self.weights)
        
        if hasattr(self, 'bias') and self.bias is not None:
            print(f"   Saving bias to {results_dir}/bias.npy")  
            np.save(results_dir / "bias.npy", self.bias)
        
        # 4. Create prediction script
        self._create_prediction_script(results_dir, ten_file)
        
        print(f"\n ✅ Enhanced model state saved to: {results_dir.absolute()}")
        return results_dir
    
    def _basic_save_results(self, results_dir: Path):
        """Basic save results nếu model không có save_results method"""
        print(f"   Saving basic results to {results_dir}/results.json")
        
        results_data = {
            "algorithm": self.__class__.__name__,
            "loss_function": getattr(self, 'ham_loss', 'Unknown'),
            "parameters": self._get_model_parameters(),
            "training_time": getattr(self, 'training_time', 0),
            "converged": getattr(self, 'converged', False),
            "final_iteration": getattr(self, 'final_iteration', 0)
        }
        
        with open(results_dir / "results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save training history nếu có
        if hasattr(self, 'loss_history') and self.loss_history:
            training_df = pd.DataFrame({
                'iteration': range(len(self.loss_history)),
                'loss': self.loss_history
            })
            
            if hasattr(self, 'gradient_norms') and self.gradient_norms:
                training_df['gradient_norm'] = self.gradient_norms
            
            training_df.to_csv(results_dir / "training_history.csv", index=False)
    
    def _create_prediction_script(self, results_dir: Path, experiment_name: str):
        """Tạo prediction script để dễ dàng load và predict"""
        script_content = f'''#!/usr/bin/env python3
"""
Prediction script cho {experiment_name}
Auto-generated từ Flask app

Usage:
    python predict_{experiment_name}.py data.csv output.csv
"""

import sys
import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def load_model():
    """Load complete model từ saved state"""
    model_path = Path(__file__).parent / "model_complete.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {{model_path}}")
    
    model_data = joblib.load(model_path)
    return model_data['model_instance'], model_data['model_state']

def predict_from_file(input_file: str, output_file: str = None):
    """
    Load model và predict từ CSV file
    
    Args:
        input_file: Path to CSV file with data to predict
        output_file: Path to save predictions (optional)
    """
    # Load model
    model, model_state = load_model()
    
    # Load new data
    if not Path(input_file).exists():
        raise FileNotFoundError(f"Input file not found: {{input_file}}")
    
    X_new = pd.read_csv(input_file)
    
    # Convert to numpy nếu cần
    if isinstance(X_new, pd.DataFrame):
        X_new = X_new.values
    
    # Make predictions
    predictions = model.predict(X_new)
    
    # Create results DataFrame
    results_df = pd.DataFrame({{
        'prediction': predictions
    }})
    
    # Add original data nếu muốn
    original_df = pd.read_csv(input_file)
    for col in original_df.columns:
        results_df[f'original_{{col}}'] = original_df[col]
    
    # Save results
    if output_file is None:
        output_file = f"predictions_{{Path(input_file).stem}}.csv"
    
    results_df.to_csv(output_file, index=False)
    print(f"Predictions saved to: {{output_file}}")
    
    return predictions, results_df

def main():
    """Main function for command line usage"""
    if len(sys.argv) < 2:
        print("Usage: python predict_{experiment_name}.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        predictions, results_df = predict_from_file(input_file, output_file)
        print(f"Successfully predicted {{len(predictions)}} samples")
        print(f"Prediction stats: Mean={{np.mean(predictions):.4f}}, Std={{np.std(predictions):.4f}}")
        
    except Exception as e:
        print(f"Error during prediction: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        script_path = results_dir / f"predict_{experiment_name}.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        print(f"   Created prediction script: {script_path}")

class PredictionCapableMixin:
    """
    Mixin để add prediction capabilities cho models
    """
    
    @classmethod 
    def load_from_saved(cls, results_dir: str):
        """
        Class method để load model từ saved results
        
        Args:
            results_dir: Path to results directory
            
        Returns:
            Loaded model instance
        """
        results_path = Path(results_dir)
        
        # Try loading complete model first
        complete_model_path = results_path / "model_complete.pkl"
        if complete_model_path.exists():
            model_data = joblib.load(complete_model_path)
            return model_data['model_instance']
        
        # Fallback: reconstruct từ state
        state_path = results_path / "model_state.json"
        if state_path.exists():
            with open(state_path, 'r') as f:
                model_state = json.load(f)
            
            # Reconstruct model
            return cls._reconstruct_from_state(model_state, results_path)
        
        raise FileNotFoundError(f"No saved model found in {results_dir}")
    
    @classmethod
    def _reconstruct_from_state(cls, model_state: Dict[str, Any], results_path: Path):
        """Reconstruct model từ saved state"""
        # Get parameters
        params = model_state.get('parameters', {})
        
        # Create new instance
        instance = cls(**params)
        
        # Restore weights
        if model_state.get('weights'):
            instance.weights = np.array(model_state['weights'])
        
        if model_state.get('bias') is not None:
            instance.bias = model_state['bias']
        
        # Set training completed flag
        instance.training_completed = model_state.get('training_completed', False)
        
        return instance
    
    def predict_with_preprocessing(self, X_new: np.ndarray, 
                                  preprocessing_info: Dict[str, Any] = None) -> np.ndarray:
        """
        Predict với automatic preprocessing
        
        Args:
            X_new: New data to predict
            preprocessing_info: Preprocessing parameters (nếu None sẽ skip preprocessing)
            
        Returns:
            Predictions
        """
        X_processed = X_new.copy()
        
        # Apply preprocessing nếu có info
        if preprocessing_info and preprocessing_info.get('requires_preprocessing', True):
            feature_means = np.array(preprocessing_info.get('feature_means', 0))
            feature_stds = np.array(preprocessing_info.get('feature_stds', 1))
            
            # Normalize
            X_processed = (X_processed - feature_means) / np.maximum(feature_stds, 1e-8)
        
        # Make prediction
        return self.predict(X_processed)