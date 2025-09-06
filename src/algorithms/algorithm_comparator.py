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
# Example usage với đa dạng setup paths từ different algorithms
setup_paths = [
    "quasi_newton/01a_setup_bfgs_ols",
    "quasi_newton/01b_setup_bfgs_ridge",
    "quasi_newton/02a_setup_lbfgs_ols_m5",
    "quasi_newton/02b_setup_lbfgs_ols_m10",
    "quasi_newton/02c_setup_lbfgs_ridge_m5",
]

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
        
        print(f"Valid: {len(self.validated_paths)}/{len(setup_paths)}")
    
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
        
        # Step 2: Tạo bảng so sánh markdown (bao gồm complexity metrics và setup-specific info)
        # Creating report...
        markdown_file = self.output_dir / "algorithm_comparison.md"
        self._create_enhanced_markdown_report(str(markdown_file))
        
        # Step 3: Tạo convergence comparison plot
        # Creating plots...
        convergence_data = self._collect_convergence_data()
        convergence_plot_created = False
        
        if convergence_data:
            try:
                convergence_file = self.output_dir / "convergence_comparison.png"
                ve_duong_hoi_tu_so_sanh(convergence_data, str(convergence_file))
                convergence_plot_created = True
                print(f"Plot: {convergence_file}")
                    
            except Exception as e:
                print(f"Plot error: {e}")
        else:
            print("No valid data")
        
        # Step 4: Tạo optimization trajectory plot (cho algorithm đầu tiên có dữ liệu)
        # Creating trajectory...
        self._create_trajectory_plot()
        
        
        print("\n" + "=" * 50)
        print("COMPARISON COMPLETED!")
        print(f"Results saved to: {self.output_dir.absolute()}")
        print("Files generated:")
        print("  - algorithm_comparison.md (Enhanced with setup-based analysis)")
        if convergence_plot_created:
            print("  - convergence_comparison.png (Each setup = separate line)")
        else:
            print("  - convergence_data_summary.txt (Plot failed, text summary)")
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
                                'loss_history': loss_history[:50],  # Giới hạn 50 điểm
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
        
        # Extract ranges tương tự như trong utils function
        max_iterations = 0
        all_losses = []
        
        for setup in setups_data:
            loss_history = setup['loss_history']
            all_losses.extend(loss_history)
            max_iterations = max(max_iterations, len(loss_history))
        
        iter_min, iter_max = 0, max_iterations
        loss_min, loss_max = min(all_losses), max(all_losses)
        
        # Calculate ranges with special attention to convergence area (same logic as utils)
        iter_range = iter_max - iter_min if iter_max != iter_min else 100
        loss_range = loss_max - loss_min if loss_max != loss_min else 0.1
        
        # Focus on convergence point (final values)
        final_iter = max_iterations
        final_loss = min(all_losses)
        
        # Smart adaptive zooming (same as utils function)
        min_viz_size = max_iterations * 0.1  # Minimum visualization range
        trajectory_span = max(iter_range, loss_range)
        
        if trajectory_span < max_iterations * 0.1:
            expansion = 2.0
            zoom_level = "tight"
        elif trajectory_span < max_iterations * 0.5:
            expansion = 1.5
            zoom_level = "moderate"
        else:
            expansion = 1.0
            zoom_level = "normal"
        
        # Center visualization around convergence area
        iter_center = final_iter
        loss_center = final_loss
        iter_radius = max(iter_range * (0.5 + expansion * 0.3), min_viz_size)
        loss_radius = max(loss_range * (0.5 + expansion * 0.3), loss_range * 0.1)
        
        iter_viz_min = max(0, iter_center - iter_radius)
        iter_viz_max = iter_center + iter_radius
        loss_viz_min = max(loss_min * 0.8, loss_center - loss_radius)
        loss_viz_max = loss_center + loss_radius
        
        # Adaptive grid size (same pattern as utils)
        if zoom_level == "tight":
            grid_size = 35
            n_contour_levels = 30
        elif zoom_level == "moderate":
            grid_size = 30
            n_contour_levels = 25
        else:
            grid_size = 25
            n_contour_levels = 20
        
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
        colors = tao_color_palette(len(setups_data))
        
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
        ax.set_ylabel('Loss Value (log scale)', fontsize=12) 
        ax.set_title(f'Optimization Trajectories - Multiple Setups\n({len(setups_data)} different configurations)', 
                    fontsize=14, fontweight='bold')
        
        # Use log scale for Y-axis to handle large loss ranges
        ax.set_yscale('log')
        
        # Set limits to focus on interesting region
        ax.set_xlim(iter_viz_min, iter_viz_max)
        # For log scale, ensure positive values
        loss_viz_min_safe = max(loss_viz_min, min(all_losses) * 0.5)
        loss_viz_max_safe = max(loss_viz_max, max(all_losses) * 1.5)
        ax.set_ylim(loss_viz_min_safe, loss_viz_max_safe)
        
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
    """Main function để demo sử dụng - Updated với enhanced comparator"""
    print("ALGORITHM COMPARATOR - ENHANCED VERSION with Setup-based Analysis")
    print("=" * 70)
    
    try:
        print(f"Initializing comparator with {len(setup_paths)} setup paths...")
        comparator = AlgorithmComparator(setup_paths)
        
        print(f"Running comparison analysis...")
        results = comparator.run_comparison()
        
        if results:
            print(f"\nCheck results at: {results['output_dir']}")
    
    except Exception as e:
        print(f"Error during comparison: {e}")
        print(f"   Stack trace: {str(e)}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()