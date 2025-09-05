#!/usr/bin/env python3
"""
Algorithm Comparator - Đơn giản hóa để so sánh các optimization algorithms

Chức năng chính:
1. Thu thập kết quả từ array các setup paths
2. Tạo 3 file chính: bảng markdown, convergence plot, trajectory plot
3. Gọn gàng, dễ sử dụng
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
import os
from datetime import datetime

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from typing import List, Dict, Any
from utils.visualization_utils import (
    tao_bang_so_sanh_markdown, 
    ve_duong_hoi_tu_so_sanh, 
    ve_duong_dong_muc_optimization
)


class AlgorithmComparator:
    """
    Class đơn giản để so sánh các optimization algorithms
    """
    
    def __init__(self, setup_paths: List[str], data_dir="data/03_algorithms", output_dir="data/04_comparison"):
        """
        Initialize AlgorithmComparator
        
        Parameters:
        -----------
        setup_paths : List[str]
            Array các đường dẫn setup để so sánh
            Có thể là relative paths: ["gradient_descent/setup1", "newton/setup2"]
            Hoặc absolute paths
        data_dir : str
            Base directory chứa algorithm results
        output_dir : str  
            Directory lưu kết quả comparison
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.setup_paths = setup_paths
        self.validated_paths = self._validate_setup_paths(setup_paths)
        self.results_data = []
        
        print(f"Validated {len(self.validated_paths)}/{len(setup_paths)} setup paths")
    
    def collect_results(self):
        """Thu thập kết quả từ các setup paths đã chỉ định"""
        print(f"Thu thập kết quả từ {len(self.validated_paths)} setup paths...")
        
        for path in self.validated_paths:
            path_obj = Path(path)
            alg_family = path_obj.parent.name if path_obj.parent.name != "03_algorithms" else "unknown"
            self._process_setup_folder(path_obj, alg_family)
        
        print(f"Thu thập được {len(self.results_data)} experiments")
    
    def _process_setup_folder(self, exp_folder, alg_name):
        """Xử lý một setup folder cụ thể"""
        results_file = exp_folder / "results.json"
        if results_file.exists():
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                result_info = self._extract_key_metrics(data, exp_folder, alg_name)
                self.results_data.append(result_info)
                print(f"      Found: {exp_folder.name}")
                
            except Exception as e:
                print(f"      Không đọc được {results_file}: {e}")
    
    def _extract_key_metrics(self, data: Dict[Any, Any], exp_folder: Path, alg_name: str) -> Dict[str, Any]:
        """Trích xuất các metrics chính cần thiết bao gồm complexity metrics"""
        training_results = data.get('training_results', {})
        
        result_info = {
            'algorithm_name': data.get('algorithm', 'Unknown'),
            'loss_function': data.get('loss_function', 'Unknown'),
            'training_time': training_results.get('training_time', 0),
            'converged': training_results.get('converged', False),
            'iterations': training_results.get('final_iteration', 
                                            training_results.get('final_epoch', 0)),
            'final_loss': training_results.get('final_loss', 
                                             training_results.get('final_cost', float('inf'))),
            'full_path': str(exp_folder),
        }
        
        # Thêm learning rate nếu có
        params = data.get('parameters', {})
        if 'learning_rate' in params:
            result_info['learning_rate'] = params['learning_rate']
        
        # Extract complexity metrics if available
        complexity_data = data.get('computational_complexity', {})
        if complexity_data:
            basic_metrics = complexity_data.get('basic_metrics', {})
            per_iter_metrics = complexity_data.get('per_iteration_averages', {})
            efficiency_metrics = complexity_data.get('efficiency_metrics', {})
            scalability_metrics = complexity_data.get('scalability_metrics', {})
            
            # Add key complexity metrics
            result_info.update({
                'total_operations': basic_metrics.get('total_operations', 0),
                'function_evaluations': basic_metrics.get('function_evaluations', 0),
                'gradient_evaluations': basic_metrics.get('gradient_evaluations', 0),
                'operations_per_iter': per_iter_metrics.get('operations_per_iter', 0),
                'convergence_efficiency': efficiency_metrics.get('convergence_efficiency', 0),
                'operations_to_convergence': efficiency_metrics.get('operations_to_convergence', 0),
                'peak_memory': basic_metrics.get('peak_memory_size', 0),
                'memory_efficiency': scalability_metrics.get('memory_efficiency', 0),
                'ops_per_problem_unit': scalability_metrics.get('operations_per_problem_unit', 0),
                'has_complexity_data': True
            })
        else:
            # Mark as missing complexity data
            result_info.update({
                'total_operations': 0,
                'function_evaluations': 0,
                'gradient_evaluations': 0,
                'operations_per_iter': 0,
                'convergence_efficiency': 0,
                'operations_to_convergence': 0,
                'peak_memory': 0,
                'memory_efficiency': 0,
                'ops_per_problem_unit': 0,
                'has_complexity_data': False
            })
        
        return result_info
    
    def _validate_setup_paths(self, setup_paths: List[str]) -> List[str]:
        """Validate và resolve setup paths"""
        validated = []
        
        for path in setup_paths:
            path_obj = Path(path)
            
            # Handle relative paths
            if not path_obj.is_absolute():
                path_obj = self.data_dir / path_obj
            
            # Check if path exists and has results.json
            results_file = path_obj / "results.json"
            if path_obj.exists() and results_file.exists():
                validated.append(str(path_obj))
                print(f"   Valid: {path}")
            else:
                print(f"   Invalid: {path} (missing results.json)")
        
        return validated
    
    def _collect_convergence_data(self) -> Dict[str, Dict]:
        """Thu thập dữ liệu convergence từ training history files"""
        convergence_data = {}
        
        for result in self.results_data:
            exp_path = Path(result['full_path'])
            history_file = exp_path / "training_history.csv"
            
            if history_file.exists():
                try:
                    history = pd.read_csv(history_file)
                    algorithm_name = result['algorithm_name']
                    
                    convergence_data[algorithm_name] = {
                        'cost_history': history['loss'].tolist() if 'loss' in history.columns else [],
                        'gradient_norms': history.get('gradient_norm', []).tolist() if 'gradient_norm' in history.columns else None
                    }
                except Exception as e:
                    print(f"      Không đọc được history: {history_file}: {e}")
        
        return convergence_data
    
    def run_comparison(self):
        """Chạy quy trình so sánh và tạo files bao gồm complexity analysis"""
        print("ALGORITHM COMPARATOR - WITH COMPLEXITY ANALYSIS")
        print("=" * 50)
        
        # Step 1: Thu thập kết quả
        self.collect_results()
        
        if len(self.results_data) == 0:
            print("Không tìm thấy kết quả nào để so sánh!")
            return
        
        # Check if any experiments have complexity data
        complexity_available = any(result.get('has_complexity_data', False) for result in self.results_data)
        
        # Step 2: Tạo bảng so sánh markdown (bao gồm complexity metrics)
        print("Tạo bảng so sánh markdown...")
        markdown_file = self.output_dir / "algorithm_comparison.md"
        tao_bang_so_sanh_markdown(self.results_data, str(markdown_file))
        
        # Step 3: Tạo convergence comparison plot
        print("Tạo convergence comparison plot...")
        convergence_data = self._collect_convergence_data()
        if convergence_data:
            convergence_file = self.output_dir / "convergence_comparison.png"
            ve_duong_hoi_tu_so_sanh(convergence_data, str(convergence_file))
        else:
            print("   Không tìm thấy dữ liệu convergence")
        
        # Step 4: Tạo optimization trajectory plot (cho algorithm đầu tiên có dữ liệu)
        print("Tạo optimization trajectory plot...")
        self._create_trajectory_plot()
        
        # Step 5: Tạo complexity comparison plots nếu có dữ liệu
        if complexity_available:
            print("Tạo computational complexity analysis...")
            self._create_complexity_comparison()
        else:
            print("   Không có dữ liệu complexity để phân tích")
        
        print("\n" + "=" * 50)
        print("COMPARISON COMPLETED!")
        print(f"Kết quả đã lưu vào: {self.output_dir.absolute()}")
        print("Files được tạo:")
        print("  - algorithm_comparison.md")
        print("  - convergence_comparison.png")
        print("  - optimization_trajectory.png")
        if complexity_available:
            print("  - complexity_comparison.png")
            print("  - complexity_summary.csv")
            print("  - operation_distribution.png (for first algorithm)")
        print("=" * 50)
        
        return {
            'total_experiments': len(self.results_data),
            'complexity_available': complexity_available,
            'output_dir': str(self.output_dir)
        }
    
    def _create_trajectory_plot(self):
        """Tạo optimization trajectory plot cho algorithm đầu tiên có dữ liệu phù hợp"""
        for result in self.results_data[:3]:  # Thử 3 algorithm đầu
            exp_path = Path(result['full_path'])
            history_file = exp_path / "training_history.csv"
            
            if history_file.exists():
                try:
                    history = pd.read_csv(history_file)
                    
                    # Check if có weight columns
                    weight_cols = [col for col in history.columns if col.startswith('weight_') or col.startswith('w_')]
                    
                    if weight_cols and 'loss' in history.columns and len(history) > 5:
                        # Extract weights history
                        weights_history = []
                        for _, row in history.iterrows():
                            weights = [row[col] for col in weight_cols]
                            weights_history.append(weights)
                        
                        if len(weights_history) >= 2:
                            # Tạo dummy data cho loss function
                            n_features = len(weight_cols)
                            X = np.random.randn(50, n_features)
                            y = np.random.randn(50)
                            
                            def simple_loss_function(X, y, weights):
                                return np.sum((X @ weights - y) ** 2)
                            
                            trajectory_file = self.output_dir / "optimization_trajectory.png"
                            ve_duong_dong_muc_optimization(
                                loss_function=simple_loss_function,
                                weights_history=weights_history[:15],
                                X=X,
                                y=y,
                                title=f'Optimization Trajectory - {result["algorithm_name"][:20]}',
                                save_path=str(trajectory_file)
                            )
                            print(f"   Trajectory plot đã lưu: {trajectory_file}")
                            return
                            
                except Exception as e:
                    print(f"      Error processing trajectory for {result['algorithm_name']}: {e}")
                    continue
        
        print("   Không tìm thấy dữ liệu trajectory phù hợp")

    def _create_complexity_comparison(self):
        """Tạo các plots so sánh computational complexity"""
        from ..utils.complexity_visualization import (
            plot_complexity_comparison, create_complexity_summary_table, plot_operation_distribution
        )
        
        # Collect complexity data from results
        complexity_results = []
        algorithm_names = []
        
        for result in self.results_data:
            if result.get('has_complexity_data', False):
                # Load full complexity analysis from the results file
                result_path = Path(result['full_path']) / "results.json"
                if result_path.exists():
                    import json
                    with open(result_path, 'r') as f:
                        full_data = json.load(f)
                    
                    complexity_data = full_data.get('computational_complexity', {})
                    if complexity_data:
                        complexity_results.append(complexity_data)
                        algorithm_names.append(result['algorithm_name'])
        
        if not complexity_results:
            print("   Không có dữ liệu complexity chi tiết để tạo plots")
            return
        
        print(f"   Tạo complexity comparison cho {len(complexity_results)} algorithms...")
        
        # Create comparison plot
        comparison_file = self.output_dir / "complexity_comparison.png"
        plot_complexity_comparison(
            complexity_results,
            save_path=str(comparison_file),
            title="Computational Complexity Comparison"
        )
        
        # Create summary table
        summary_file = self.output_dir / "complexity_summary.csv"
        summary_df = create_complexity_summary_table(
            complexity_results,
            algorithm_names,
            save_path=str(summary_file)
        )
        
        # Create operation distribution for first algorithm
        if complexity_results:
            distribution_file = self.output_dir / "operation_distribution.png"
            plot_operation_distribution(
                complexity_results[0],
                save_path=str(distribution_file),
                title=f"Operation Distribution - {algorithm_names[0]}"
            )
        
        print(f"   ✅ Complexity analysis completed với {len(complexity_results)} algorithms")


def main():
    """Main function để demo sử dụng"""
    print("🚀 ALGORITHM COMPARATOR - SIMPLIFIED VERSION")
    print("=" * 60)
    
    # Example usage với setup paths
    setup_paths = [
        "gradient_descent/01_setup_gd_ols_lr_0001",
        "gradient_descent/02_setup_gd_ols_lr_001",
        "gradient_descent/03_setup_gd_ols_lr_01",
        "gradient_descent/04_setup_gd_ols_lr_03",
        "gradient_descent/05_setup_gd_ols_lr_02",
    ]
    
    try:
        comparator = AlgorithmComparator(setup_paths)
        results = comparator.run_comparison()
        
        if results:
            print(f"\n📊 Summary:")
            print(f"   Total Experiments: {results['total_experiments']}")
            print(f"   Output Directory: {results['output_dir']}")
            print(f"\n✅ Check {results['output_dir']} for results!")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Usage example:")
        print("comparator = AlgorithmComparator(['path1', 'path2', 'path3'])")
        print("comparator.run_comparison()")
    
    print("=" * 60)


if __name__ == "__main__":
    main()