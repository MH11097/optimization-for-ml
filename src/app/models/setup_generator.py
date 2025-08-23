"""
Setup File Generator
Tạo setup files tự động từ algorithm registry và user parameters
"""

import os
from pathlib import Path
from typing import Dict, Any
from .algorithm_registry import Algorithm, AlgorithmVariant, Parameter, ParameterType, format_parameter_value

class SetupFileGenerator:
    """Generator để tạo setup files từ templates và parameters"""
    
    def __init__(self):
        self.templates = {
            "gradient_descent": self._gd_template,
            "momentum_gd": self._momentum_gd_template,
            "newton_method": self._newton_template,
            "stochastic_gd": self._sgd_template,
            "proximal_gd": self._proximal_template,
            "quasi_newton": self._quasi_newton_template
        }
    
    def generate_setup_file(self, algorithm: Algorithm, variant: AlgorithmVariant, 
                           parameters: Dict[str, Any], output_dir: str = None) -> str:
        """
        Generate setup file từ algorithm, variant và parameters
        
        Returns: Path to generated setup file
        """
        # Format parameters với đúng type
        formatted_params = {}
        for param in variant.parameters:
            if param.name in parameters:
                formatted_params[param.name] = format_parameter_value(
                    parameters[param.name], param.param_type
                )
            else:
                formatted_params[param.name] = param.default_value
        
        # Generate filename từ template
        filename = self._generate_filename(algorithm, variant, formatted_params)
        
        # Get template content
        template_func = self.templates.get(algorithm.name)
        if not template_func:
            raise ValueError(f"No template found for algorithm: {algorithm.name}")
        
        content = template_func(algorithm, variant, formatted_params)
        
        # Determine output directory
        if output_dir is None:
            output_dir = f"src/algorithms/{algorithm.name}"
        
        # Create output directory if needed
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Write setup file
        setup_path = Path(output_dir) / f"{filename}.py"
        with open(setup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(setup_path)
    
    def _generate_filename(self, algorithm: Algorithm, variant: AlgorithmVariant, 
                          parameters: Dict[str, Any]) -> str:
        """Generate filename từ template pattern"""
        filename = variant.setup_template
        
        # Replace parameter placeholders
        for param_name, param_value in parameters.items():
            # Convert value to string suitable for filename
            if isinstance(param_value, float):
                if param_value < 1:
                    # Convert 0.01 -> 001, 0.1 -> 01
                    value_str = f"{param_value:g}".replace("0.", "0").replace(".", "")
                else:
                    value_str = f"{param_value:g}".replace(".", "_")
            else:
                value_str = str(param_value)
            
            filename = filename.replace(f"{{{param_name}}}", value_str)
        
        return filename
    
    def _gd_template(self, algorithm: Algorithm, variant: AlgorithmVariant, 
                     parameters: Dict[str, Any]) -> str:
        """Template cho Gradient Descent"""
        lr = parameters.get('learning_rate', 0.1)
        iterations = parameters.get('so_lan_thu', 500)
        tolerance = parameters.get('diem_dung', 1e-5)
        regularization = parameters.get('regularization', None)
        
        setup_name = f"GD {variant.loss_function.upper()}"
        if regularization:
            setup_name += f" (λ={regularization})"
        
        reg_comment = f"\n- Regularization: {regularization}" if regularization else ""
        reg_param = f",\\n        regularization={regularization}" if regularization else ""
        
        return f"""#!/usr/bin/env python3
\"\"\"
Setup script for {setup_name}
- Loss Function: {variant.loss_function.upper()}
- Learning Rate: {lr}
- Max Iterations: {iterations}
- Tolerance: {tolerance}{reg_comment}
\"\"\"

import sys
import os
from pathlib import Path

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from algorithms.gradient_descent.gradient_descent_model import GradientDescentModel
from utils.data_process_utils import load_du_lieu


def get_experiment_name():
    \"\"\"Lấy tên experiment từ tên file hiện tại\"\"\"
    import inspect
    frame = inspect.currentframe()
    filename = frame.f_back.f_globals['__file__']
    return Path(filename).stem

def main():
    \"\"\"Chạy {setup_name}\"\"\"
    print("{setup_name.upper()} - SETUP")
    
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Khởi tạo model
    model = GradientDescentModel(
        ham_loss='{variant.loss_function}',
        learning_rate={lr},
        so_lan_thu={iterations},
        diem_dung={tolerance}{reg_param}
    )
    
    # Huấn luyện model
    results = model.fit(X_train, y_train)
    
    # Đánh giá model
    metrics = model.evaluate(X_test, y_test)
    
    # Lưu kết quả với tên file tự động
    ten_file = get_experiment_name()
    results_dir = model.save_results(ten_file)
    
    # Tạo biểu đồ
    model.plot_results(X_test, y_test, ten_file)
    
    print("\\n\\nTraining and visualization completed!")
    print(f"Results saved to: {{results_dir.absolute()}}")
    
    return model, results, metrics

if __name__ == "__main__":
    model, results, metrics = main()
"""
    
    def _momentum_gd_template(self, algorithm: Algorithm, variant: AlgorithmVariant, 
                             parameters: Dict[str, Any]) -> str:
        """Template cho Momentum Gradient Descent"""
        lr = parameters.get('learning_rate', 0.1)
        momentum = parameters.get('momentum', 0.9)
        iterations = parameters.get('so_lan_thu', 500)
        tolerance = parameters.get('diem_dung', 1e-5)
        
        return f"""#!/usr/bin/env python3
\"\"\"
Setup script for Momentum Gradient Descent - {variant.loss_function.upper()}
- Loss Function: {variant.loss_function.upper()}
- Learning Rate: {lr}
- Momentum: {momentum}
- Max Iterations: {iterations}
- Tolerance: {tolerance}
\"\"\"

import sys
import os
from pathlib import Path

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from algorithms.gradient_descent.momentum_gd_model import MomentumGDModel
from utils.data_process_utils import load_du_lieu


def get_experiment_name():
    \"\"\"Lấy tên experiment từ tên file hiện tại\"\"\"
    import inspect
    frame = inspect.currentframe()
    filename = frame.f_back.f_globals['__file__']
    return Path(filename).stem

def main():
    \"\"\"Chạy Momentum Gradient Descent\"\"\"
    print("MOMENTUM GRADIENT DESCENT - {variant.loss_function.upper()} SETUP")
    
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Khởi tạo model
    model = MomentumGDModel(
        ham_loss='{variant.loss_function}',
        learning_rate={lr},
        momentum={momentum},
        so_lan_thu={iterations},
        diem_dung={tolerance}
    )
    
    # Huấn luyện model
    results = model.fit(X_train, y_train)
    
    # Đánh giá model
    metrics = model.evaluate(X_test, y_test)
    
    # Lưu kết quả với tên file tự động
    ten_file = get_experiment_name()
    results_dir = model.save_results(ten_file)
    
    # Tạo biểu đồ
    model.plot_results(X_test, y_test, ten_file)
    
    print("\\n\\nTraining and visualization completed!")
    print(f"Results saved to: {{results_dir.absolute()}}")
    
    return model, results, metrics

if __name__ == "__main__":
    model, results, metrics = main()
"""
    
    def _newton_template(self, algorithm: Algorithm, variant: AlgorithmVariant, 
                        parameters: Dict[str, Any]) -> str:
        """Template cho Newton Method"""
        reg = parameters.get('regularization', 0.0)
        num_reg = parameters.get('numerical_regularization', 1e-6)
        iterations = parameters.get('so_lan_thu', 50)
        tolerance = parameters.get('diem_dung', 1e-8)
        
        method_type = "Pure" if "pure" in variant.name else "Damped"
        
        return f"""#!/usr/bin/env python3
\"\"\"
Setup script for {method_type} Newton Method - {variant.loss_function.upper()}
- Method: {method_type} Newton
- Loss Function: {variant.loss_function.upper()}
- Regularization: {reg}
- Numerical Regularization: {num_reg}
- Max Iterations: {iterations}
- Tolerance: {tolerance}
\"\"\"

import sys
import os
from pathlib import Path

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from algorithms.newton_method.newton_model import NewtonModel
from utils.data_process_utils import load_du_lieu


def get_experiment_name():
    \"\"\"Lấy tên experiment từ tên file hiện tại\"\"\"
    import inspect
    frame = inspect.currentframe()
    filename = frame.f_back.f_globals['__file__']
    return Path(filename).stem

def main():
    \"\"\"Chạy {method_type} Newton Method\"\"\"
    print("{method_type.upper()} NEWTON METHOD - {variant.loss_function.upper()} SETUP")
    
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Khởi tạo model
    model = NewtonModel(
        ham_loss='{variant.loss_function}',
        regularization={reg},
        numerical_regularization={num_reg},
        so_lan_thu={iterations},
        diem_dung={tolerance}
    )
    
    # Huấn luyện model
    results = model.fit(X_train, y_train)
    
    # Đánh giá model
    metrics = model.evaluate(X_test, y_test)
    
    # Lưu kết quả với tên file tự động
    ten_file = get_experiment_name()
    results_dir = model.save_results(ten_file)
    
    # Tạo biểu đồ
    model.plot_results(X_test, y_test, ten_file)
    
    print("\\n\\nTraining and visualization completed!")
    print(f"Results saved to: {{results_dir.absolute()}}")
    
    return model, results, metrics

if __name__ == "__main__":
    model, results, metrics = main()
"""
    
    def _sgd_template(self, algorithm: Algorithm, variant: AlgorithmVariant, 
                     parameters: Dict[str, Any]) -> str:
        """Template cho Stochastic Gradient Descent"""
        lr = parameters.get('learning_rate', 0.01)
        epochs = parameters.get('so_epochs', 100)
        batch_size = parameters.get('batch_size', 1)
        random_state = parameters.get('random_state', 42)
        
        batch_desc = f"Batch Size {batch_size}" if batch_size > 1 else "Pure SGD"
        
        return f"""#!/usr/bin/env python3
\"\"\"
Setup script for Stochastic Gradient Descent - {batch_desc}
- Learning Rate: {lr}
- Epochs: {epochs}
- Batch Size: {batch_size}
- Random State: {random_state}
\"\"\"

import sys
import os
from pathlib import Path

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from algorithms.stochastic_gd.sgd_model import SGDModel
from utils.data_process_utils import load_sampled_data


def get_experiment_name():
    \"\"\"Lấy tên experiment từ tên file hiện tại\"\"\"
    import inspect
    frame = inspect.currentframe()
    filename = frame.f_back.f_globals['__file__']
    return Path(filename).stem

def main():
    \"\"\"Chạy Stochastic Gradient Descent\"\"\"
    print("STOCHASTIC GRADIENT DESCENT - {batch_desc.upper()} SETUP")
    
    # Load sampled data
    X_train, X_test, y_train, y_test = load_sampled_data()
    
    # Khởi tạo model
    model = SGDModel(
        learning_rate={lr},
        so_epochs={epochs},
        random_state={random_state},
        batch_size={batch_size},
        ham_loss='mse'
    )
    
    # Huấn luyện model
    results = model.fit(X_train, y_train)
    
    # Đánh giá model
    metrics = model.evaluate(X_test, y_test)
    
    # Lưu kết quả với tên file tự động
    ten_file = get_experiment_name()
    results_dir = model.save_results(ten_file)
    
    # Tạo biểu đồ
    model.plot_results(X_test, y_test, ten_file)
    
    print("\\n\\nTraining and visualization completed!")
    print(f"Results saved to: {{results_dir.absolute()}}")
    
    return model, results, metrics

if __name__ == "__main__":
    model, results, metrics = main()
"""
    
    def _proximal_template(self, algorithm: Algorithm, variant: AlgorithmVariant, 
                          parameters: Dict[str, Any]) -> str:
        """Template cho Proximal Gradient Descent"""
        lr = parameters.get('learning_rate', 0.01)
        lambda_l1 = parameters.get('lambda_l1', 0.01)
        iterations = parameters.get('so_lan_thu', 1000)
        tolerance = parameters.get('diem_dung', 1e-6)
        
        return f"""#!/usr/bin/env python3
\"\"\"
Setup script for Proximal Gradient Descent - Lasso
- Learning Rate: {lr}
- L1 Lambda: {lambda_l1}
- Max Iterations: {iterations}
- Tolerance: {tolerance}
\"\"\"

import sys
import os
from pathlib import Path

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from algorithms.proximal_gd.proximal_gd_model import ProximalGDModel
from utils.data_process_utils import load_sampled_data


def get_experiment_name():
    \"\"\"Lấy tên experiment từ tên file hiện tại\"\"\"
    import inspect
    frame = inspect.currentframe()
    filename = frame.f_back.f_globals['__file__']
    return Path(filename).stem

def main():
    \"\"\"Chạy Proximal Gradient Descent - Lasso\"\"\"
    print("PROXIMAL GRADIENT DESCENT - LASSO SETUP")
    
    # Load sampled data
    X_train, X_test, y_train, y_test = load_sampled_data()
    
    # Khởi tạo model
    model = ProximalGDModel(
        ham_loss='lasso',
        learning_rate={lr},
        lambda_l1={lambda_l1},
        so_lan_thu={iterations},
        diem_dung={tolerance}
    )
    
    # Huấn luyện model
    results = model.fit(X_train, y_train)
    
    # Đánh giá model
    metrics = model.evaluate(X_test, y_test)
    
    # Lưu kết quả với tên file tự động
    ten_file = get_experiment_name()
    results_dir = model.save_results(ten_file)
    
    # Tạo biểu đồ
    model.plot_results(X_test, y_test, ten_file)
    
    print("\\n\\nTraining and visualization completed!")
    print(f"Results saved to: {{results_dir.absolute()}}")
    
    return model, results, metrics

if __name__ == "__main__":
    model, results, metrics = main()
"""
    
    def _quasi_newton_template(self, algorithm: Algorithm, variant: AlgorithmVariant, 
                              parameters: Dict[str, Any]) -> str:
        """Template cho Quasi-Newton BFGS"""
        iterations = parameters.get('so_lan_thu', 200)
        tolerance = parameters.get('diem_dung', 1e-8)
        c1 = parameters.get('armijo_c1', 1e-4)
        c2 = parameters.get('wolfe_c2', 0.9)
        reg = parameters.get('regularization', None)
        
        reg_comment = f"\\n- Regularization: {reg}" if reg else ""
        reg_param = f",\\n        regularization={reg}" if reg else ""
        
        return f"""#!/usr/bin/env python3
\"\"\"
Setup script for BFGS Quasi-Newton - {variant.loss_function.upper()}
- Loss Function: {variant.loss_function.upper()}
- Max Iterations: {iterations}
- Tolerance: {tolerance}
- Armijo C1: {c1}
- Wolfe C2: {c2}{reg_comment}
\"\"\"

import sys
import os
from pathlib import Path

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from algorithms.quasi_newton.quasi_newton_model import QuasiNewtonModel
from utils.data_process_utils import load_du_lieu


def get_experiment_name():
    \"\"\"Lấy tên experiment từ tên file hiện tại\"\"\"
    import inspect
    frame = inspect.currentframe()
    filename = frame.f_back.f_globals['__file__']
    return Path(filename).stem

def main():
    \"\"\"Chạy BFGS Quasi-Newton\"\"\"
    print("BFGS QUASI-NEWTON - {variant.loss_function.upper()} SETUP")
    
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    # Khởi tạo model
    model = QuasiNewtonModel(
        ham_loss='{variant.loss_function}',
        so_lan_thu={iterations},
        diem_dung={tolerance},
        armijo_c1={c1},
        wolfe_c2={c2}{reg_param}
    )
    
    # Huấn luyện model
    results = model.fit(X_train, y_train)
    
    # Đánh giá model
    metrics = model.evaluate(X_test, y_test)
    
    # Lưu kết quả với tên file tự động
    ten_file = get_experiment_name()
    results_dir = model.save_results(ten_file)
    
    # Tạo biểu đồ
    model.plot_results(X_test, y_test, ten_file)
    
    print("\\n\\nTraining and visualization completed!")
    print(f"Results saved to: {{results_dir.absolute()}}")
    
    return model, results, metrics

if __name__ == "__main__":
    model, results, metrics = main()
"""