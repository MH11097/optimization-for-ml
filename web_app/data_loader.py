import os
import json
import pandas as pd
import re
from typing import Dict, List, Any, Optional
from pathlib import Path

class AlgorithmDataLoader:
    """Loads and parses optimization algorithm results for web visualization."""
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.algorithms_path = self.data_root / "03_algorithms"
        self._cache = {}
    
    def get_available_algorithms(self) -> List[str]:
        """Get list of available algorithm types."""
        if not self.algorithms_path.exists():
            return []
        
        algorithms = []
        for item in self.algorithms_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                algorithms.append(item.name)
        return sorted(algorithms)
    
    def get_algorithm_setups(self, algorithm: str) -> List[Dict[str, Any]]:
        """Get all setups for a specific algorithm with parsed parameters."""
        algorithm_path = self.algorithms_path / algorithm
        if not algorithm_path.exists():
            return []
        
        setups = []
        for setup_dir in algorithm_path.iterdir():
            if setup_dir.is_dir() and not setup_dir.name.startswith('.'):
                setup_info = self._parse_setup_directory(setup_dir)
                if setup_info:
                    setups.append(setup_info)
        
        return sorted(setups, key=lambda x: x['setup_name'])
    
    def _parse_setup_directory(self, setup_dir: Path) -> Optional[Dict[str, Any]]:
        """Parse a single setup directory to extract metadata and data."""
        results_file = setup_dir / "results.json"
        history_file = setup_dir / "training_history.csv"
        
        if not results_file.exists():
            return None
        
        try:
            # Load results JSON with special handling for Infinity/NaN values
            with open(results_file, 'r') as f:
                content = f.read()
                # Replace problematic values
                content = content.replace('Infinity', 'null')
                content = content.replace('-Infinity', 'null')
                content = content.replace('NaN', 'null')
                results = json.loads(content)
            
            # Parse setup name to extract parameters
            setup_name = setup_dir.name
            parsed_params = self._parse_setup_name(setup_name)
            
            # Load training history if available
            training_history = None
            if history_file.exists():
                training_history = pd.read_csv(history_file).to_dict('records')
            
            return {
                'setup_name': setup_name,
                'setup_path': str(setup_dir),
                'parsed_parameters': parsed_params,
                'results': results,
                'training_history': training_history,
                'has_history': history_file.exists()
            }
        except Exception as e:
            print(f"Error parsing setup {setup_dir.name}: {e}")
            return None
    
    def _parse_setup_name(self, setup_name: str) -> Dict[str, Any]:
        """Parse setup name to extract parameter values."""
        params = {}
        
        # Common parameter patterns
        patterns = {
            'learning_rate': r'lr_([0-9.]+)',
            'regularization': r'reg_([0-9.]+)',
            'momentum': r'mom_([0-9.]+)', 
            'c1': r'c1_([0-9.]+)',
            'rho': r'rho_([0-9.]+)',
            'damping': r'damp_([0-9.]+)',
            'beta1': r'beta1_([0-9.]+)',
            'beta2': r'beta2_([0-9.]+)',
            'epsilon': r'eps_([0-9.]+)'
        }
        
        # Extract parameter values
        for param_name, pattern in patterns.items():
            match = re.search(pattern, setup_name)
            if match:
                params[param_name] = float(match.group(1))
        
        # Extract algorithm type
        if 'gd' in setup_name:
            params['algorithm_type'] = 'gradient_descent'
        elif 'newton' in setup_name:
            params['algorithm_type'] = 'newton_method'
        elif 'quasi' in setup_name:
            params['algorithm_type'] = 'quasi_newton'
        elif 'sgd' in setup_name or 'stochastic' in setup_name:
            params['algorithm_type'] = 'stochastic_gd'
        elif 'subgrad' in setup_name:
            params['algorithm_type'] = 'subgradient'
        
        # Extract loss function
        if 'ols' in setup_name:
            params['loss_function'] = 'ols'
        elif 'ridge' in setup_name:
            params['loss_function'] = 'ridge'
        elif 'lasso' in setup_name:
            params['loss_function'] = 'lasso'
        
        # Extract special methods
        if 'backtracking' in setup_name:
            params['step_size_method'] = 'backtracking'
        elif 'schedule' in setup_name:
            params['step_size_method'] = 'scheduled'
        elif 'momentum' in setup_name:
            params['step_size_method'] = 'momentum'
        elif 'damped' in setup_name:
            params['step_size_method'] = 'damped'
        else:
            params['step_size_method'] = 'constant'
        
        return params
    
    def get_parameter_ranges(self, algorithm: str) -> Dict[str, Dict[str, Any]]:
        """Get parameter ranges for an algorithm to create sliders."""
        setups = self.get_algorithm_setups(algorithm)
        if not setups:
            return {}
        
        # Collect all parameter values
        param_values = {}
        for setup in setups:
            for param, value in setup['parsed_parameters'].items():
                if isinstance(value, (int, float)):
                    if param not in param_values:
                        param_values[param] = []
                    param_values[param].append(value)
        
        # Calculate ranges
        param_ranges = {}
        for param, values in param_values.items():
            if values:
                unique_values = sorted(list(set(values)))
                param_ranges[param] = {
                    'min': min(unique_values),
                    'max': max(unique_values),
                    'values': unique_values,
                    'step': self._calculate_step(unique_values)
                }
        
        return param_ranges
    
    def _calculate_step(self, values: List[float]) -> float:
        """Calculate appropriate step size for slider."""
        if len(values) <= 1:
            return 0.001
        
        diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
        min_diff = min(diffs)
        
        # Use smaller step for fine control
        if min_diff >= 1:
            return 0.1
        elif min_diff >= 0.1:
            return 0.01
        elif min_diff >= 0.01:
            return 0.001
        else:
            return min_diff / 10
    
    def get_setup_by_parameters(self, algorithm: str, target_params: Dict[str, float], 
                               tolerance: float = 0.001) -> Optional[Dict[str, Any]]:
        """Find setup that matches given parameters within tolerance."""
        setups = self.get_algorithm_setups(algorithm)
        
        for setup in setups:
            parsed_params = setup['parsed_parameters']
            match = True
            
            for param, target_value in target_params.items():
                if param in parsed_params:
                    if isinstance(parsed_params[param], (int, float)):
                        if abs(parsed_params[param] - target_value) > tolerance:
                            match = False
                            break
                    else:
                        if parsed_params[param] != target_value:
                            match = False
                            break
            
            if match:
                return setup
        
        return None
    
    def get_grouped_setups(self, algorithm: str) -> Dict[str, List[Dict[str, Any]]]:
        """Group setups by parameter type for organized display."""
        setups = self.get_algorithm_setups(algorithm)
        if not setups:
            return {}
        
        groups = {
            'fixed_lr': [],
            'scheduled_lr': [],
            'momentum': [],
            'regularized': [],
            'backtracking': [],
            'damped': [],
            'other': []
        }
        
        for setup in setups:
            params = setup['parsed_parameters']
            step_method = params.get('step_size_method', 'constant')
            
            if 'momentum' in params or step_method == 'momentum':
                groups['momentum'].append(setup)
            elif 'regularization' in params or params.get('loss_function') in ['ridge', 'lasso']:
                groups['regularized'].append(setup)
            elif step_method == 'backtracking':
                groups['backtracking'].append(setup)
            elif step_method == 'damped':
                groups['damped'].append(setup)
            elif step_method == 'scheduled':
                groups['scheduled_lr'].append(setup)
            elif step_method == 'constant':
                groups['fixed_lr'].append(setup)
            else:
                groups['other'].append(setup)
        
        # Remove empty groups
        return {k: v for k, v in groups.items() if v}