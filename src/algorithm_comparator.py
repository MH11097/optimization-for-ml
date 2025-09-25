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
import argparse
from datetime import datetime
# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from typing import List, Dict, Any
from utils.visualization_utils import (
    tao_bang_so_sanh_markdown, 
    ve_duong_hoi_tu_so_sanh, 
    ve_duong_dong_muc_optimization
)
# Example usage removed - now uses dynamic discovery only
class AlgorithmComparator:
    """
    Class đơn giản để so sánh các optimization algorithms
    """
    
    def __init__(self, folder_name: str, start_number: int, end_number: int,
                 data_dir="data/03_algorithms", output_dir="data/04_comparison"):
        """
        Initialize AlgorithmComparator

        Parameters:
        -----------
        folder_name : str
            Name of folder to scan (e.g., "gradientdescentoptimizer", "gradient_descent")
        start_number : int
            Starting number for folder range (e.g., 111)
        end_number : int
            Ending number for folder range (e.g., 140)
        data_dir : str
            Base directory chứa algorithm results
        output_dir : str
            Directory lưu kết quả comparison
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Use dynamic discovery method
        print(f"Scanning: {folder_name} from {start_number} to {end_number}")
        self.setup_paths = self._discover_setup_folders(folder_name, start_number, end_number)
        self.validated_paths = self.setup_paths  # Already validated in discovery
        self.results_data = []

        print(f"Found {len(self.validated_paths)} valid setup folders")
    
    def collect_results(self):
        """Thu thập kết quả từ các setup paths đã chỉ định"""
        print(f"Collecting {len(self.validated_paths)} results...")
        
        for path in self.validated_paths:
            path_obj = Path(path)
            alg_family = path_obj.parent.name if path_obj.parent.name != "03_algorithms" else "unknown"
            self._process_setup_folder(path_obj, alg_family)
        
        print(f"Found {len(self.results_data)} experiments")
    
    def _process_setup_folder(self, exp_folder, alg_name):
        """Xử lý một setup folder cụ thể"""
        results_file = exp_folder / "results.json"
        if results_file.exists():
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                result_info = self._extract_key_metrics(data, exp_folder, alg_name)
                self.results_data.append(result_info)
        # Removed verbose output
                
            except Exception as e:
                print(f"      Không đọc được {results_file}: {e}")
    
    def _extract_key_metrics(self, data: Dict[Any, Any], exp_folder: Path, alg_name: str) -> Dict[str, Any]:
        """Trích xuất các metrics chính cần thiết bao gồm complexity metrics và best results"""
        training_results = data.get('training_results', {})
        params = data.get('parameters', {})
        
        # Setup name để làm key riêng biệt
        setup_name = exp_folder.name
        
        result_info = {
            'setup_name': setup_name,  # Unique identifier cho mỗi setup
            'algorithm_name': data.get('algorithm', 'Unknown'),
            'loss_function': data.get('loss_function', 'Unknown'),
            'training_time': training_results.get('training_time', 0),
            'converged': training_results.get('converged', False),
            'iterations': training_results.get('final_iteration', 
                                            training_results.get('final_epoch', 0)),
            'final_loss': training_results.get('final_loss', 
                                             training_results.get('final_cost', float('inf'))),
            'final_gradient_norm': training_results.get('final_gradient_norm', 0),
            'full_path': str(exp_folder),
        }
        
        # Extract best results if available (ưu tiên best results)
        best_loss = training_results.get('best_loss', result_info['final_loss'])
        best_iteration = training_results.get('best_iteration', result_info['iterations'])
        best_gradient_norm = training_results.get('best_gradient_norm', result_info['final_gradient_norm'])
        
        result_info.update({
            'best_loss': best_loss,
            'best_iteration': best_iteration, 
            'best_gradient_norm': best_gradient_norm,
            'used_best_results': 'best_loss' in training_results  # Flag to indicate if best results available
        })
        
        # Extract parameters
        result_info['learning_rate'] = params.get('learning_rate', 0)
        result_info['max_iterations'] = params.get('max_iterations', params.get('so_lan_thu', 0))
        result_info['tolerance'] = params.get('tolerance', params.get('diem_dung', 0))
        
        # Algorithm-specific parameters
        result_info['momentum'] = params.get('momentum', 0)  # For momentum/nesterov GD
        result_info['step_size_method'] = params.get('step_size_method', 'constant')  # For enhanced GD
        result_info['regularization'] = params.get('regularization', 0)  # For Ridge/Lasso
        result_info['batch_size'] = params.get('batch_size', 0)  # For SGD
        result_info['learning_rate_schedule'] = params.get('learning_rate_schedule', 'constant')  # For SGD
        
        # Algorithm-specific results
        algorithm_specific = data.get('algorithm_specific', {})
        result_info['gradient_descent_type'] = algorithm_specific.get('gradient_descent_type', 'unknown')
        result_info['momentum_used'] = algorithm_specific.get('momentum_used', False)
        result_info['acceleration_used'] = algorithm_specific.get('acceleration_used', False)
        
        # Final algorithm-specific metrics
        if 'final_velocity_norm' in training_results:
            result_info['final_velocity_norm'] = training_results['final_velocity_norm']
        if 'condition_number' in training_results:
            result_info['condition_number'] = training_results['condition_number']
        
        # Extract ML metrics if available
        ml_metrics = data.get('ml_metrics', {})
        result_info.update({
            'mse': ml_metrics.get('mse', 0),
            'rmse': ml_metrics.get('rmse', 0),
            'mae': ml_metrics.get('mae', 0),
            'r2': ml_metrics.get('r2', 0),
            'adjusted_r2': ml_metrics.get('adjusted_r2', 0),
            'mape': ml_metrics.get('mape', 0),
            'smape': ml_metrics.get('smape', 0),
            'has_ml_metrics': bool(ml_metrics)
        })

        # Extract complexity metrics if available
        complexity_data = data.get('computational_complexity', {})
        if complexity_data:
            basic_metrics = complexity_data.get('basic_metrics', {})
            per_iter_metrics = complexity_data.get('per_iteration_averages', {})
            efficiency_metrics = complexity_data.get('efficiency_metrics', {})
            scalability_metrics = complexity_data.get('scalability_metrics', {})

            # Add key complexity metrics
            result_info.update({
                'total_operations': complexity_data.get('total_operations', basic_metrics.get('total_operations', 0)),
                'function_evaluations': complexity_data.get('function_evaluations', basic_metrics.get('function_evaluations', 0)),
                'gradient_evaluations': complexity_data.get('gradient_evaluations', basic_metrics.get('gradient_evaluations', 0)),
                'operations_per_iter': per_iter_metrics.get('operations_per_iter', 0),
                'convergence_efficiency': efficiency_metrics.get('convergence_efficiency', 0),
                'operations_to_convergence': efficiency_metrics.get('operations_to_convergence', 0),
                'peak_memory': complexity_data.get('peak_memory', basic_metrics.get('peak_memory_size', 0)),
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

    def _discover_setup_folders(self, folder_name: str, start_number: int, end_number: int) -> List[str]:
        """Auto discover setup folders in specified range"""
        discovered_paths = []
        folder_path = self.data_dir / folder_name

        if not folder_path.exists():
            print(f"Warning: Folder {folder_path} does not exist")
            return []

        print(f"Discovering setups in {folder_path} from {start_number} to {end_number}...")

        # Scan for folders matching pattern {number}_*
        for folder in folder_path.iterdir():
            if folder.is_dir():
                folder_name_str = folder.name
                # Extract number from folder name (handle different patterns)
                number_str = ""
                for char in folder_name_str:
                    if char.isdigit():
                        number_str += char
                    elif number_str:  # Stop at first non-digit after finding digits
                        break

                if number_str:
                    try:
                        folder_number = int(number_str)
                        if start_number <= folder_number <= end_number:
                            # Check if has results.json
                            results_file = folder / "results.json"
                            if results_file.exists():
                                discovered_paths.append(str(folder))
                                print(f"   Found: {folder_name_str}")
                            else:
                                print(f"   Skip: {folder_name_str} (no results.json)")
                    except ValueError:
                        continue

        # Sort by folder number for consistent ordering
        discovered_paths.sort(key=lambda x: int(''.join(filter(str.isdigit, Path(x).name))))

        print(f"Discovered {len(discovered_paths)} valid setup folders")
        return discovered_paths

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

    def export_comprehensive_csv(self) -> str:
        """Export all results data to a comprehensive CSV file"""
        if not self.results_data:
            print("No results data to export")
            return ""

        csv_file_path = self.output_dir / "comprehensive_results.csv"

        # Prepare data for CSV export
        csv_rows = []

        # Define all possible columns in order
        columns = [
            # Setup Info
            'setup_name', 'algorithm_name', 'loss_function', 'full_path',

            # Parameters
            'learning_rate', 'momentum', 'step_size_method', 'regularization',
            'batch_size', 'max_iterations', 'tolerance', 'learning_rate_schedule',

            # ML Results
            'final_loss', 'best_loss', 'final_gradient_norm', 'best_gradient_norm',
            'training_time', 'converged', 'iterations', 'best_iteration', 'result_type',

            # ML Performance Metrics
            'mse', 'rmse', 'mae', 'r2', 'adjusted_r2', 'mape', 'smape', 'has_ml_metrics',

            # Complexity Metrics
            'total_operations', 'function_evaluations', 'gradient_evaluations',
            'operations_per_iter', 'convergence_efficiency', 'operations_to_convergence',
            'peak_memory', 'memory_efficiency', 'ops_per_problem_unit', 'has_complexity_data',

            # Algorithm Specific
            'gradient_descent_type', 'momentum_used', 'acceleration_used',
            'final_velocity_norm', 'condition_number'
        ]

        # Extract data for each setup
        for result in self.results_data:
            row = {}

            # Fill row with data, using safe defaults for missing keys
            for col in columns:
                if col == 'result_type':
                    # Determine result type based on available data
                    row[col] = 'Best' if result.get('used_best_results', False) else 'Final'
                else:
                    # Get value with appropriate default
                    default_value = 0 if col in [
                        'learning_rate', 'momentum', 'regularization', 'batch_size',
                        'max_iterations', 'tolerance', 'final_loss', 'best_loss',
                        'final_gradient_norm', 'best_gradient_norm', 'training_time',
                        'iterations', 'best_iteration', 'total_operations',
                        'function_evaluations', 'gradient_evaluations', 'operations_per_iter',
                        'convergence_efficiency', 'operations_to_convergence', 'peak_memory',
                        'memory_efficiency', 'ops_per_problem_unit', 'final_velocity_norm',
                        'condition_number', 'mse', 'rmse', 'mae', 'r2', 'adjusted_r2',
                        'mape', 'smape'
                    ] else False if col in [
                        'converged', 'has_complexity_data', 'momentum_used', 'acceleration_used',
                        'has_ml_metrics'
                    ] else 'Unknown'

                    row[col] = result.get(col, default_value)

            csv_rows.append(row)

        # Write to CSV
        try:
            import csv
            with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=columns)
                writer.writeheader()
                writer.writerows(csv_rows)

            print(f"CSV exported: {csv_file_path} ({len(csv_rows)} rows, {len(columns)} columns)")
            return str(csv_file_path)

        except Exception as e:
            print(f"Error exporting CSV: {e}")
            return ""

    def _collect_convergence_data(self) -> Dict[str, Dict]:
        """Thu thập dữ liệu convergence từ training history files - chỉ bao gồm setup hoàn toàn hợp lệ"""
        convergence_data = {}
        filtered_count = 0
        total_setups = 0
        
        for result in self.results_data:
            exp_path = Path(result['full_path'])
            history_file = exp_path / "training_history.csv"
            total_setups += 1
            setup_key = result['setup_name']
            
            if history_file.exists():
                try:
                    history = pd.read_csv(history_file)
                    
                    # Extract raw data
                    raw_loss_history = history['loss'].tolist() if 'loss' in history.columns else []
                    raw_iterations = history['iteration'].tolist() if 'iteration' in history.columns else None
                    raw_gradient_norms = history.get('gradient_norm', []).tolist() if 'gradient_norm' in history.columns else None
                    
                    # Check if ALL data points are valid - if any invalid, skip entire setup
                    setup_valid = True
                    
                    # Check loss history
                    for loss in raw_loss_history:
                        if loss == float('inf') or loss == float('-inf') or np.isnan(loss) or not np.isfinite(loss):
                            setup_valid = False
                            break
                    
                    # Check iterations if available
                    if setup_valid and raw_iterations:
                        for iteration in raw_iterations:
                            if not np.isfinite(iteration) or np.isnan(iteration):
                                setup_valid = False
                                break
                    
                    # Check gradient norms if available
                    if setup_valid and raw_gradient_norms:
                        for grad_norm in raw_gradient_norms:
                            if grad_norm == float('inf') or grad_norm == float('-inf') or np.isnan(grad_norm) or not np.isfinite(grad_norm):
                                setup_valid = False
                                break
                    
                    if setup_valid and raw_loss_history:
                        display_name = f"{setup_key}"
                        
                        convergence_data[display_name] = {
                            'loss_history': raw_loss_history,
                            'iterations': raw_iterations if raw_iterations else None,
                            'gradient_norms': raw_gradient_norms if raw_gradient_norms else None,
                            'setup_info': {
                                'algorithm': result['algorithm_name'],
                                'learning_rate': result['learning_rate'],
                                'momentum': result['momentum'],
                                'step_size_method': result['step_size_method'],
                                'loss_function': result['loss_function']
                            }
                        }
                        # Removed verbose output
                    else:
                        filtered_count += 1
                        print(f"SKIP {setup_key}: Invalid data")
                        
                except Exception as e:
                    print(f"ERROR {setup_key}: Cannot read history")
                    filtered_count += 1
            else:
                filtered_count += 1
                print(f"SKIP {result['setup_name']}: No history file")
        
        if filtered_count > 0:
            print(f"Valid setups: {len(convergence_data)}/{total_setups}")
        
        return convergence_data

    def _create_four_convergence_plots(self, convergence_data: Dict[str, Dict]) -> bool:
        """Create 4 separate convergence plots instead of subplots"""
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
        from utils.visualization_utils import tao_color_palette

        if not convergence_data:
            print("No convergence data available for plotting")
            return False

        try:
            # Separate algorithms by convergence status
            converged_setups = []
            non_converged_setups = []

            for setup_name, data in convergence_data.items():
                # Check if this setup converged by looking up in results_data
                setup_converged = False
                for result in self.results_data:
                    if result['setup_name'] == setup_name:
                        setup_converged = result.get('converged', False)
                        break

                if setup_converged:
                    converged_setups.append(setup_name)
                else:
                    non_converged_setups.append(setup_name)

            # Create simple 2-color scheme
            colors = {}

            # Blue gradient for converged algorithms (light blue to dark blue)
            if converged_setups:
                blue_colors = cm.Blues(np.linspace(0.4, 0.9, len(converged_setups)))
                for i, setup in enumerate(converged_setups):
                    colors[setup] = blue_colors[i]

            # Red gradient for non-converged algorithms (light red to dark red)
            if non_converged_setups:
                red_colors = cm.Reds(np.linspace(0.4, 0.9, len(non_converged_setups)))
                for i, setup in enumerate(non_converged_setups):
                    colors[setup] = red_colors[i]

            setup_names = list(convergence_data.keys())

            # Plot 1: Loss Convergence (Linear Scale)
            plt.figure(figsize=(12, 8))
            for setup_name, data in convergence_data.items():
                loss_history = data['loss_history']
                iterations = data.get('iterations') or list(range(len(loss_history)))
                plt.plot(iterations, loss_history, color=colors[setup_name], label=setup_name, linewidth=2, alpha=0.8)

            plt.title('Loss Convergence (Linear Scale)', fontsize=14, fontweight='bold')
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Loss Value', fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            loss_file = self.output_dir / "loss_convergence.png"
            plt.savefig(loss_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Plot: {loss_file}")

            # Plot 2: Gradient Norm Convergence (Linear Scale)
            plt.figure(figsize=(12, 8))
            gradient_plots_created = 0
            for setup_name, data in convergence_data.items():
                gradient_norms = data.get('gradient_norms')
                if gradient_norms:
                    iterations = data.get('iterations') or list(range(len(gradient_norms)))
                    plt.plot(iterations, gradient_norms, color=colors[setup_name], label=setup_name, linewidth=2, alpha=0.8)
                    gradient_plots_created += 1

            if gradient_plots_created > 0:
                plt.title('Gradient Norm Convergence (Linear Scale)', fontsize=14, fontweight='bold')
                plt.xlabel('Iteration', fontsize=12)
                plt.ylabel('Gradient Norm', fontsize=12)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                grad_file = self.output_dir / "gradient_norm_convergence.png"
                plt.savefig(grad_file, dpi=300, bbox_inches='tight')
                print(f"Plot: {grad_file}")
            plt.close()

            # Plot 3: Log Loss Convergence (Log Scale)
            plt.figure(figsize=(12, 8))
            for setup_name, data in convergence_data.items():
                loss_history = data['loss_history']
                iterations = data.get('iterations') or list(range(len(loss_history)))
                # Filter out non-positive values for log scale
                valid_indices = [j for j, loss in enumerate(loss_history) if loss > 0]
                if valid_indices:
                    valid_iterations = [iterations[j] for j in valid_indices]
                    valid_losses = [loss_history[j] for j in valid_indices]
                    plt.plot(valid_iterations, valid_losses, color=colors[setup_name], label=setup_name, linewidth=2, alpha=0.8)

            plt.title('Loss Convergence (Log Scale)', fontsize=14, fontweight='bold')
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Loss Value (Log Scale)', fontsize=12)
            plt.yscale('log')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            log_loss_file = self.output_dir / "log_loss_convergence.png"
            plt.savefig(log_loss_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Plot: {log_loss_file}")

            # Plot 4: Log Gradient Norm Convergence (Log Scale)
            plt.figure(figsize=(12, 8))
            log_gradient_plots_created = 0
            for setup_name, data in convergence_data.items():
                gradient_norms = data.get('gradient_norms')
                if gradient_norms:
                    iterations = data.get('iterations') or list(range(len(gradient_norms)))
                    # Filter out non-positive values for log scale
                    valid_indices = [j for j, norm in enumerate(gradient_norms) if norm > 0]
                    if valid_indices:
                        valid_iterations = [iterations[j] for j in valid_indices]
                        valid_norms = [gradient_norms[j] for j in valid_indices]
                        plt.plot(valid_iterations, valid_norms, color=colors[setup_name], label=setup_name, linewidth=2, alpha=0.8)
                        log_gradient_plots_created += 1

            if log_gradient_plots_created > 0:
                plt.title('Gradient Norm Convergence (Log Scale)', fontsize=14, fontweight='bold')
                plt.xlabel('Iteration', fontsize=12)
                plt.ylabel('Gradient Norm (Log Scale)', fontsize=12)
                plt.yscale('log')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                log_grad_file = self.output_dir / "log_gradient_norm_convergence.png"
                plt.savefig(log_grad_file, dpi=300, bbox_inches='tight')
                print(f"Plot: {log_grad_file}")
            plt.close()

            return True

        except Exception as e:
            print(f"Error creating convergence plots: {e}")
            return False

    def run_comparison(self):
        """Chạy quy trình so sánh và tạo files bao gồm complexity analysis"""
        print("ALGORITHM COMPARATOR")
        print("=" * 20)
        
        # Step 1: Thu thập kết quả
        self.collect_results()
        
        if len(self.results_data) == 0:
            print("No results found to compare!")
            return
        
        # Check if any experiments have complexity data
        complexity_available = any(result.get('has_complexity_data', False) for result in self.results_data)

        # Step 2: Export comprehensive CSV
        print("Creating CSV export...")
        csv_file = self.export_comprehensive_csv()

        # Step 3: Print summary table to console
        from utils.visualization_utils import tao_bang_so_sanh_markdown
        tao_bang_so_sanh_markdown(self.results_data)

        # Step 4: Create 4 convergence plots
        print("Creating convergence plots...")
        convergence_data = self._collect_convergence_data()
        plots_created = False

        if convergence_data:
            plots_created = self._create_four_convergence_plots(convergence_data)
        else:
            print("No valid convergence data for plotting")

        # Step 5: Create optimization trajectory plot (optional)
        print("Creating trajectory plot...")
        self._create_trajectory_plot()


        print("\n" + "=" * 50)
        print("COMPARISON COMPLETED!")
        print(f"Results saved to: {self.output_dir.absolute()}")
        print("Files generated:")
        if csv_file:
            print(f"  - comprehensive_results.csv (Full data export)")
        if plots_created:
            print("  - loss_convergence.png (Linear scale)")
            print("  - gradient_norm_convergence.png (Linear scale)")
            print("  - log_loss_convergence.png (Log scale)")
            print("  - log_gradient_norm_convergence.png (Log scale)")
        print("  - optimization_trajectory.png (Contour plot with trajectories)")
    
        
        return {
            'total_experiments': len(self.results_data),
            'complexity_available': complexity_available,
            'output_dir': str(self.output_dir)
        }
    
    def _create_trajectory_plot(self):
        """Tạo contour plot với convergence paths của các setup khác nhau"""
        valid_setups = []
        
        # Thu thập dữ liệu loss history từ các setup
        for result in self.results_data:
            exp_path = Path(result['full_path'])
            history_file = exp_path / "training_history.csv"
            
            if history_file.exists():
                try:
                    history = pd.read_csv(history_file)
                    
                    if 'loss' in history.columns and len(history) > 5:
                        loss_history = history['loss'].tolist()
                        
                        # Kiểm tra data hợp lệ
                        if all(np.isfinite(loss) for loss in loss_history):
                            valid_setups.append({
                                'name': result['setup_name'],
                                'loss_history': loss_history,  # Sử dụng toàn bộ history
                                'learning_rate': result.get('learning_rate', 0.001),
                                'algorithm': result['algorithm_name']
                            })
                            
                except Exception as e:
                    continue
        
        if len(valid_setups) >= 2:
            try:
                trajectory_file = self.output_dir / "optimization_trajectory.png"
                self._create_multi_setup_contour_plot(valid_setups, str(trajectory_file))
                print(f"Trajectory: {trajectory_file}")
                return
            except Exception as e:
                print(f"Trajectory error: {e}")
        
        print("Not enough valid data for trajectory plot")
    
    def _create_multi_setup_contour_plot(self, setups_data, save_path):
        """Tạo contour plot theo kiểu ve_duong_dong_muc_optimization nhưng cho multiple setups"""
        import matplotlib.pyplot as plt
        import numpy as np
        from utils.visualization_utils import tao_color_palette
        
        # Use same style as ve_duong_dong_muc_optimization
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Extract ranges để show toàn bộ trajectory
        max_iterations = 0
        all_losses = []

        for setup in setups_data:
            loss_history = setup['loss_history']
            all_losses.extend(loss_history)
            max_iterations = max(max_iterations, len(loss_history))

        iter_min, iter_max = 0, max_iterations
        loss_min, loss_max = min(all_losses), max(all_losses)

        # Sử dụng toàn bộ range với padding thay vì zoom vào convergence area
        iter_padding = max_iterations * 0.05  # 5% padding
        loss_range = loss_max - loss_min
        loss_padding = loss_range * 0.1 if loss_range > 0 else loss_max * 0.1  # 10% padding

        # Define final_loss for loss surface calculation
        final_loss = loss_min  # Best (minimum) loss achieved

        # Set visualization limits to show full trajectories
        iter_viz_min = max(0, iter_min - iter_padding)
        iter_viz_max = iter_max + iter_padding
        loss_viz_min = max(loss_min * 0.8, loss_min - loss_padding)  # Ensure positive for log scale
        loss_viz_max = loss_max + loss_padding
        
        # Fixed grid size for full trajectory view
        grid_size = 50
        n_contour_levels = 25
        
        # Create meshgrid
        iter_grid = np.linspace(iter_viz_min, iter_viz_max, grid_size)
        loss_grid = np.linspace(loss_viz_min, loss_viz_max, grid_size)
        ITER, LOSS = np.meshgrid(iter_grid, loss_grid)
        
        # Compute loss surface (synthetic ideal convergence)
        loss_surface = np.zeros_like(ITER)
        initial_loss = max(all_losses)
        
        for i in range(grid_size):
            for j in range(grid_size):
                iter_val = ITER[i, j]
                loss_val = LOSS[i, j]
                
                # Ideal exponential decay curve
                if iter_val <= 0:
                    ideal_loss = initial_loss
                else:
                    decay_rate = np.log(initial_loss / final_loss) / max_iterations
                    ideal_loss = initial_loss * np.exp(-decay_rate * iter_val)
                
                # Distance from ideal path + iteration penalty
                loss_surface[i, j] = abs(loss_val - ideal_loss) / ideal_loss + iter_val * 0.001
        
        # Handle NaN/Inf in loss surface (same as utils)
        loss_min_surf, loss_max_surf = np.nanmin(loss_surface), np.nanmax(loss_surface)
        if not np.isfinite(loss_min_surf) or not np.isfinite(loss_max_surf):
            levels = n_contour_levels
        elif loss_max_surf / loss_min_surf > 100:
            try:
                levels = np.logspace(np.log10(loss_min_surf), np.log10(loss_max_surf), n_contour_levels)
                if not np.all(np.isfinite(levels)):
                    levels = n_contour_levels
            except:
                levels = n_contour_levels
        else:
            levels = n_contour_levels
        
        # Plot contour (same style as utils)
        contour = ax.contour(ITER, LOSS, loss_surface, levels=levels, colors='black', linewidths=0.8, alpha=0.7)
        contourf = ax.contourf(ITER, LOSS, loss_surface, levels=levels, cmap='viridis', alpha=0.4)
        
        # Add colorbar
        cbar = plt.colorbar(contourf, ax=ax, shrink=0.8)
        cbar.set_label('Loss Value', rotation=270, labelpad=15, fontsize=10)
        
        # Plot optimization paths (each setup as trajectory)
        colors = tao_color_palette(len(setups_data), 'deep')
        
        for i, setup in enumerate(setups_data):
            loss_history = setup['loss_history']
            name = setup['name']
            lr = setup['learning_rate']
            iterations = list(range(len(loss_history)))
            
            # Plot trajectory as path
            ax.plot(iterations, loss_history, 'r-' if i == 0 else f'C{i}-', linewidth=3, alpha=0.9, 
                   label=f'{name}', zorder=5)
            
            # Mark start and end (same style as utils)
            ax.plot(iterations[0], loss_history[0], 'go' if i == 0 else 'o', 
                   color=colors[i], markersize=10, markeredgecolor='black', zorder=6)
            ax.plot(iterations[-1], loss_history[-1], 'r*' if i == 0 else '*', 
                   color=colors[i], markersize=15, markeredgecolor='black', zorder=6)
            
            # Add iteration annotations (same style as utils)
            n_annotations = min(4, len(iterations))
            annotation_indices = np.linspace(0, len(iterations)-1, n_annotations, dtype=int)
            
            for ann_idx in annotation_indices:
                actual_iter = iterations[ann_idx]
                ax.annotate(f'Iter {actual_iter}', 
                           (iterations[ann_idx], loss_history[ann_idx]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Customize plot (same style as utils)
        ax.legend(fontsize=10)
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Loss Value', fontsize=12)
        ax.set_title(f'Optimization Trajectories - Multiple Setups\n({len(setups_data)} different configurations)',
                    fontsize=14, fontweight='bold')

        # Use linear scale for Y-axis for better loss value visualization
        # ax.set_yscale('log')  # Commented out - using linear scale now

        # Set limits to show full trajectory range
        ax.set_xlim(iter_viz_min, iter_viz_max)
        # Linear scale - use full range with padding
        ax.set_ylim(loss_viz_min, loss_viz_max)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    def _create_enhanced_markdown_report(self, save_path: str):
        """Tạo báo cáo markdown với thông tin best results và setup-specific details"""
        # Saving report...
        
        # Import datetime để có timestamp
        from datetime import datetime
        
        # Check if any experiments have complexity data
        has_complexity = any(result.get('has_complexity_data', False) for result in self.results_data)
        
        # Create enhanced results data with best results prioritized
        enhanced_data = []
        for result in self.results_data:
            enhanced_result = result.copy()
            
            # Use best results if available, otherwise use final
            if result.get('used_best_results', False):
                enhanced_result['display_loss'] = result['best_loss']
                enhanced_result['display_gradient_norm'] = result['best_gradient_norm']
                enhanced_result['display_iteration'] = result['best_iteration']
                enhanced_result['result_type'] = 'Best'
            else:
                enhanced_result['display_loss'] = result['final_loss']
                enhanced_result['display_gradient_norm'] = result['final_gradient_norm']
                enhanced_result['display_iteration'] = result['iterations']
                enhanced_result['result_type'] = 'Final'
            
            enhanced_data.append(enhanced_result)
        
        # Create markdown content
        markdown_content = f"# Algorithm Comparison Report - Setup-Based Analysis\n\n"
        if has_complexity:
            markdown_content += "🔬 **With Computational Complexity Analysis**\n\n"
        markdown_content += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        markdown_content += f"**Total Setups Compared:** {len(enhanced_data)}\n\n"
        
        # Summary statistics
        converged_count = sum(1 for r in enhanced_data if r['converged'])
        convergence_rate = (converged_count / len(enhanced_data)) * 100
        avg_time = sum(r['training_time'] for r in enhanced_data) / len(enhanced_data)
        
        markdown_content += "## Summary Statistics\n\n"
        markdown_content += f"- **Converged Setups:** {converged_count}/{len(enhanced_data)} ({convergence_rate:.1f}%)\n"
        markdown_content += f"- **Average Training Time:** {avg_time:.4f} seconds\n"
        
        # Best performers
        best_loss_setup = min(enhanced_data, key=lambda x: x['display_loss'] if x['display_loss'] != float('inf') else float('inf'))
        fastest_setup = min(enhanced_data, key=lambda x: x['training_time'])
        
        # Add more detailed best performers
        fastest_converged_setup = None
        if converged_count > 0:
            converged_setups_analysis = [r for r in enhanced_data if r['converged']]
            fastest_converged_setup = min(converged_setups_analysis, key=lambda x: x.get('display_iteration', x.get('iterations', float('inf'))))
        
        markdown_content += f"- **Best Loss:** {best_loss_setup['setup_name']} ({best_loss_setup['display_loss']:.6f})\n"
        markdown_content += f"- **Fastest Training Time:** {fastest_setup['setup_name']} ({fastest_setup['training_time']:.4f}s)\n"
        if fastest_converged_setup:
            iterations = fastest_converged_setup.get('display_iteration', fastest_converged_setup.get('iterations', 0))
            markdown_content += f"- **Fastest Convergence:** {fastest_converged_setup['setup_name']} ({iterations} iterations)\n"
        markdown_content += "\n"
        
        # Main comparison table
        markdown_content += "## Setup Comparison Table\n\n"
        markdown_content += "| Setup Name | Algorithm | Loss Func | LR | Step Method | Momentum | Loss | Gradient | Iterations | Time | Converged | Result |\n"
        markdown_content += "|------------|-----------|-----------|----|----|----------|------|----------|-----------|------|-----------|--------|\n"
        
        # Sort by performance (converged first, then by loss)
        sorted_data = sorted(enhanced_data, key=lambda x: (not x['converged'], x['display_loss'] if x['display_loss'] != float('inf') else 999999))
        
        for result in sorted_data:
            converged_icon = "✅" if result['converged'] else "❌"
            loss_display = f"{result['display_loss']:.6f}" if result['display_loss'] != float('inf') else "∞"
            
            # Safely get values with defaults
            lr = result.get('learning_rate', 0)
            momentum = result.get('momentum', 0) 
            step_method = result.get('step_size_method', 'constant')
            gradient_norm = result.get('display_gradient_norm', 0)
            iterations = result.get('display_iteration', result.get('iterations', 0))
            
            markdown_content += f"| {result['setup_name']} | {result['algorithm_name']} | {result['loss_function']} |"
            markdown_content += f" {lr:.4f} | {step_method} | {momentum:.2f} |"
            markdown_content += f" {loss_display} | {gradient_norm:.6f} | {iterations} |"  
            markdown_content += f" {result['training_time']:.4f} | {converged_icon} | {result['result_type']} |\n"
        
        # Algorithm family breakdown
        markdown_content += "\n## Algorithm Family Analysis\n\n"
        
        # Group by algorithm type
        algorithm_families = {}
        for result in enhanced_data:
            family = result['algorithm_name'].split(' - ')[0] if ' - ' in result['algorithm_name'] else result['algorithm_name']
            if family not in algorithm_families:
                algorithm_families[family] = []
            algorithm_families[family].append(result)
        
        for family, setups in algorithm_families.items():
            converged_in_family = sum(1 for s in setups if s['converged'])
            avg_loss = sum(s['display_loss'] for s in setups if s['display_loss'] != float('inf')) / len([s for s in setups if s['display_loss'] != float('inf')]) if any(s['display_loss'] != float('inf') for s in setups) else float('inf')
            
            # Calculate average iterations for converged setups in this family
            converged_setups_in_family = [s for s in setups if s['converged']]
            avg_iterations = None
            if converged_setups_in_family:
                total_iterations = sum(s.get('display_iteration', s.get('iterations', 0)) for s in converged_setups_in_family)
                avg_iterations = total_iterations / len(converged_setups_in_family)
            
            markdown_content += f"### {family}\n"
            markdown_content += f"- **Setups:** {len(setups)}\n"
            markdown_content += f"- **Convergence Rate:** {converged_in_family}/{len(setups)} ({(converged_in_family/len(setups)*100):.1f}%)\n"
            if avg_loss != float('inf'):
                markdown_content += f"- **Average Loss:** {avg_loss:.6f}\n"
            if avg_iterations is not None:
                markdown_content += f"- **Average Iterations to Converge:** {avg_iterations:.0f}\n"
            markdown_content += "\n"
        
        # Best setup recommendations
        markdown_content += "## 🏆 Recommended Setups\n\n"
        
        # Best overall (converged + lowest loss)
        converged_setups = [r for r in enhanced_data if r['converged']]
        if converged_setups:
            best_overall = min(converged_setups, key=lambda x: x['display_loss'])
            best_iterations = best_overall.get('display_iteration', best_overall.get('iterations', 0))
            markdown_content += f"- **🎯 Best Overall:** `{best_overall['setup_name']}` - {best_overall['algorithm_name']} with loss {best_overall['display_loss']:.6f} (converged in {best_iterations} iterations)\n"
        
        # Fastest converged by time
        if converged_setups:
            fastest_converged = min(converged_setups, key=lambda x: x['training_time'])
            fastest_time_iterations = fastest_converged.get('display_iteration', fastest_converged.get('iterations', 0))
            markdown_content += f"- **⚡ Fastest by Time:** `{fastest_converged['setup_name']}` - {fastest_converged['training_time']:.4f}s ({fastest_time_iterations} iterations)\n"
        
        # Fastest converged by iterations
        if converged_setups:
            fastest_iterations = min(converged_setups, key=lambda x: x.get('display_iteration', x.get('iterations', float('inf'))))
            fastest_iter_count = fastest_iterations.get('display_iteration', fastest_iterations.get('iterations', 0))
            markdown_content += f"- **🏃 Fastest by Iterations:** `{fastest_iterations['setup_name']}` - {fastest_iter_count} iterations ({fastest_iterations['training_time']:.4f}s)\n"
        
        # Most robust (best gradient norm)
        if converged_setups:
            most_robust = min(converged_setups, key=lambda x: x.get('display_gradient_norm', float('inf')))
            robust_iterations = most_robust.get('display_iteration', most_robust.get('iterations', 0))
            markdown_content += f"- **🔒 Most Robust:** `{most_robust['setup_name']}` - gradient norm {most_robust.get('display_gradient_norm', 0):.6f} ({robust_iterations} iterations)\n"
        
        # Add data quality information
        markdown_content += "\n## 📊 Data Quality & Visualization Notes\n\n"
        
        # Get convergence data to check for filtering
        convergence_data = self._collect_convergence_data()
        
        if convergence_data:
            total_setups_with_data = len(convergence_data)
            filtered_setups = len(enhanced_data) - total_setups_with_data
            total_filtered_points = sum(data.get('data_quality', {}).get('filtered_points', 0) for data in convergence_data.values())
            
            markdown_content += f"### Convergence Data Processing\n"
            markdown_content += f"- **Total Setups:** {len(enhanced_data)}\n"
            markdown_content += f"- **Setups with Valid Convergence Data:** {total_setups_with_data}\n"
            if filtered_setups > 0:
                markdown_content += f"- **⚠️ Setups Filtered (No Valid Data):** {filtered_setups}\n"
            if total_filtered_points > 0:
                markdown_content += f"- **⚠️ Invalid Data Points Filtered:** {total_filtered_points} (infinity/NaN values)\n"
            
            markdown_content += f"\n### Visualization Notes\n"
            markdown_content += f"- Each setup is displayed as a **separate colored line** in convergence plots\n"
            markdown_content += f"- Invalid data points (∞, -∞, NaN) are automatically filtered out\n"
            markdown_content += f"- Only setups with valid convergence data are included in plots\n"
            if total_filtered_points > 0:
                markdown_content += f"- **Data Quality Warning:** Some algorithms produced infinite/invalid loss values during training\n"
            
            # Add per-setup data quality details if there are filtered points
            if any(data.get('data_quality', {}).get('filtered_points', 0) > 0 for data in convergence_data.values()):
                markdown_content += f"\n#### Setup-specific Data Filtering\n"
                for setup_name, data in convergence_data.items():
                    quality = data.get('data_quality', {})
                    filtered_points = quality.get('filtered_points', 0)
                    if filtered_points > 0:
                        original_points = quality.get('original_points', 0)
                        valid_points = quality.get('valid_points', 0)
                        markdown_content += f"- **{setup_name}:** {filtered_points}/{original_points} points filtered ({valid_points} valid)\n"
        else:
            markdown_content += f"### ⚠️ No Valid Convergence Data\n"
            markdown_content += f"- All setups produced invalid convergence data (infinity/NaN values)\n"
            markdown_content += f"- Convergence plots could not be generated\n"
            markdown_content += f"- This may indicate algorithm convergence issues or numerical instability\n"
        
        markdown_content += "\n---\n"
        markdown_content += f"*Report generated by Enhanced Algorithm Comparator on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*\n"
        markdown_content += "*Each setup represents a unique configuration and is displayed as a separate line in visualizations.*\n"
        if total_filtered_points > 0 or filtered_setups > 0:
            markdown_content += "*⚠️ Some invalid data points were automatically filtered for better visualization.*\n"
        
        # Save file with explicit UTF-8 encoding
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            print(f"Report: {save_path}")
        except Exception as e:
            print(f"Error: {e}")
            # Try fallback encoding
            try:
                with open(save_path, 'w', encoding='utf-8', errors='replace') as f:
                    f.write(markdown_content)
                print(f"Report saved (fallback): {save_path}")
            except Exception as e2:
                print(f"Save failed: {e2}")

def main():
    """Main function with command line arguments support"""
    parser = argparse.ArgumentParser(description='Algorithm Comparator - Compare optimization algorithms')

    # Positional arguments
    parser.add_argument('folder', nargs='?', type=str, default='gradientdescentoptimizer',
                        help='Folder name to scan (default: gradientdescentoptimizer)')
    parser.add_argument('start', nargs='?', type=int, default=111,
                        help='Starting number for folder range (default: 111)')
    parser.add_argument('end', nargs='?', type=int, default=140,
                        help='Ending number for folder range (default: 140)')

    # Optional named arguments (for backward compatibility)
    parser.add_argument('--folder_name', type=str, help='Folder name to scan (overrides positional)')
    parser.add_argument('--start_number', type=int, help='Starting number (overrides positional)')
    parser.add_argument('--end_number', type=int, help='Ending number (overrides positional)')

    args = parser.parse_args()

    # Named arguments override positional if provided
    folder = args.folder_name if args.folder_name else args.folder
    start = args.start_number if args.start_number is not None else args.start
    end = args.end_number if args.end_number is not None else args.end

    print("ALGORITHM COMPARATOR - ENHANCED VERSION")
    print("=" * 70)

    try:
        print(f"Analyzing: {folder} from {start} to {end}")
        comparator = AlgorithmComparator(folder, start, end)

        print("Running comparison analysis...")
        results = comparator.run_comparison()

        if results:
            print(f"\nResults saved to: {results['output_dir']}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 70)

if __name__ == "__main__":
    main()