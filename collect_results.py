#!/usr/bin/env python3
"""
Results Collection Script
Systematically collect all experimental results for report validation
"""

import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

def collect_all_results():
    """Collect all experimental results from data directories"""
    
    base_path = Path("data/03_algorithms")
    results = {
        "gradient_descent": [],
        "newton_method": [],
        "stochastic_gd": [],
        "quasi_newton": []
    }
    
    # Collect Gradient Descent results
    gd_path = base_path / "gradient_descent"
    if gd_path.exists():
        for setup_dir in gd_path.iterdir():
            if setup_dir.is_dir():
                results_file = setup_dir / "results.json"
                if results_file.exists():
                    try:
                        with open(results_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            data['setup_name'] = setup_dir.name
                            results["gradient_descent"].append(data)
                    except Exception as e:
                        print(f"Error reading {results_file}: {e}")
    
    # Collect Newton Method results
    newton_path = base_path / "newton_method"
    if newton_path.exists():
        for setup_dir in newton_path.iterdir():
            if setup_dir.is_dir():
                results_file = setup_dir / "results.json"
                if results_file.exists():
                    try:
                        with open(results_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            data['setup_name'] = setup_dir.name
                            results["newton_method"].append(data)
                    except Exception as e:
                        print(f"Error reading {results_file}: {e}")
    
    # Collect Stochastic GD results
    sgd_path = base_path / "stochastic_gd"
    if sgd_path.exists():
        for setup_dir in sgd_path.iterdir():
            if setup_dir.is_dir():
                results_file = setup_dir / "results.json"
                if results_file.exists():
                    try:
                        with open(results_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            data['setup_name'] = setup_dir.name
                            results["stochastic_gd"].append(data)
                    except Exception as e:
                        print(f"Error reading {results_file}: {e}")
    
    return results

def analyze_gradient_descent_results(results: List[Dict]):
    """Analyze gradient descent results and extract key metrics"""
    
    analysis = []
    
    for result in results:
        setup_name = result['setup_name']
        algorithm = result.get('algorithm', 'Unknown')
        
        # Extract key parameters
        params = result.get('parameters', {})
        lr = params.get('learning_rate', 'N/A')
        regularization = params.get('regularization_lambda', 'N/A')
        momentum = params.get('momentum', 'N/A')
        
        # Extract performance metrics
        training = result.get('training_results', {})
        iterations = training.get('final_iteration', 'N/A')
        converged = training.get('converged', False)
        final_loss = training.get('final_loss', 'N/A')
        grad_norm = training.get('final_gradient_norm', 'N/A')
        training_time = training.get('training_time', 'N/A')
        
        # Extract convergence analysis
        convergence = result.get('convergence_analysis', {})
        convergence_rate = convergence.get('convergence_rate', 'N/A')
        
        analysis.append({
            'setup_name': setup_name,
            'algorithm': algorithm,
            'learning_rate': lr,
            'regularization': regularization,
            'momentum': momentum,
            'iterations': iterations,
            'converged': converged,
            'final_loss': final_loss,
            'gradient_norm': grad_norm,
            'training_time': training_time,
            'convergence_rate': convergence_rate
        })
    
    return analysis

def analyze_newton_results(results: List[Dict]):
    """Analyze Newton method results"""
    
    analysis = []
    
    for result in results:
        setup_name = result['setup_name']
        algorithm = result.get('algorithm', 'Unknown')
        
        # Extract key parameters
        params = result.get('parameters', {})
        regularization = params.get('regularization', 'N/A')
        
        # Extract performance metrics
        training = result.get('training_results', {})
        iterations = training.get('final_iteration', 'N/A')
        converged = training.get('converged', False)
        final_loss = training.get('final_loss', 'N/A')
        grad_norm = training.get('final_gradient_norm', 'N/A')
        
        # Extract numerical analysis
        numerical = result.get('numerical_analysis', {})
        condition_number = numerical.get('hessian_condition_number', 'N/A')
        
        # Extract line search analysis
        line_search = result.get('line_search_analysis', {})
        avg_backtracks = line_search.get('average_backtracks', 'N/A')
        
        analysis.append({
            'setup_name': setup_name,
            'algorithm': algorithm,
            'regularization': regularization,
            'iterations': iterations,
            'converged': converged,
            'final_loss': final_loss,
            'gradient_norm': grad_norm,
            'condition_number': condition_number,
            'avg_backtracks': avg_backtracks
        })
    
    return analysis

def analyze_sgd_results(results: List[Dict]):
    """Analyze SGD results"""
    
    analysis = []
    
    for result in results:
        setup_name = result['setup_name']
        algorithm = result.get('algorithm', 'Unknown')
        
        # Extract key parameters
        params = result.get('parameters', {})
        lr = params.get('learning_rate', 'N/A')
        batch_size = params.get('batch_size', 'N/A')
        lr_schedule = params.get('learning_rate_schedule', 'N/A')
        
        # Extract performance metrics
        training = result.get('training_results', {})
        epochs = training.get('final_epoch', 'N/A')
        converged = training.get('converged', False)
        final_cost = training.get('final_cost', 'N/A')
        grad_norm = training.get('final_gradient_norm', 'N/A')
        training_time = training.get('training_time', 'N/A')
        
        analysis.append({
            'setup_name': setup_name,
            'algorithm': algorithm,
            'learning_rate': lr,
            'batch_size': batch_size,
            'lr_schedule': lr_schedule,
            'epochs': epochs,
            'converged': converged,
            'final_cost': final_cost,
            'gradient_norm': grad_norm,
            'training_time': training_time
        })
    
    return analysis

def generate_summary_report(all_results: Dict):
    """Generate a comprehensive summary report"""
    
    print("="*80)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("="*80)
    
    # Gradient Descent Analysis
    print("\n" + "="*40)
    print("GRADIENT DESCENT METHODS")
    print("="*40)
    
    gd_analysis = analyze_gradient_descent_results(all_results["gradient_descent"])
    
    print(f"\nTotal setups: {len(gd_analysis)}")
    
    # Sort by iterations for ranking
    converged_setups = [s for s in gd_analysis if s['converged']]
    non_converged_setups = [s for s in gd_analysis if not s['converged']]
    
    print(f"Converged setups: {len(converged_setups)}")
    print(f"Non-converged setups: {len(non_converged_setups)}")
    
    if converged_setups:
        print("\nTop 5 performers (by iterations):")
        sorted_converged = sorted(converged_setups, key=lambda x: x['iterations'] if isinstance(x['iterations'], (int, float)) else float('inf'))
        for i, setup in enumerate(sorted_converged[:5], 1):
            print(f"{i}. {setup['setup_name']}: {setup['iterations']} iterations")
            print(f"   Algorithm: {setup['algorithm']}")
            print(f"   LR: {setup['learning_rate']}, Reg: {setup['regularization']}")
    
    if non_converged_setups:
        print("\nNon-converged setups:")
        for setup in non_converged_setups:
            print(f"- {setup['setup_name']}: {setup['iterations']} iterations")
            print(f"  Final loss: {setup['final_loss']}, Grad norm: {setup['gradient_norm']}")
    
    # Newton Methods Analysis
    print("\n" + "="*40)
    print("NEWTON METHODS")
    print("="*40)
    
    newton_analysis = analyze_newton_results(all_results["newton_method"])
    
    print(f"\nTotal setups: {len(newton_analysis)}")
    
    if newton_analysis:
        print("\nNewton method performance:")
        sorted_newton = sorted(newton_analysis, key=lambda x: x['iterations'] if isinstance(x['iterations'], (int, float)) else float('inf'))
        for setup in sorted_newton:
            print(f"- {setup['setup_name']}: {setup['iterations']} iterations")
            print(f"  Algorithm: {setup['algorithm']}")
            print(f"  Condition number: {setup['condition_number']}")
            print(f"  Converged: {setup['converged']}")
    
    # SGD Analysis
    print("\n" + "="*40)
    print("STOCHASTIC GRADIENT DESCENT")
    print("="*40)
    
    sgd_analysis = analyze_sgd_results(all_results["stochastic_gd"])
    
    print(f"\nTotal setups: {len(sgd_analysis)}")
    
    if sgd_analysis:
        print("\nSGD performance:")
        for setup in sgd_analysis:
            print(f"- {setup['setup_name']}: {setup['epochs']} epochs")
            print(f"  Batch size: {setup['batch_size']}, LR: {setup['learning_rate']}")
            print(f"  Converged: {setup['converged']}, Final cost: {setup['final_cost']}")
    
    return {
        'gradient_descent': gd_analysis,
        'newton_method': newton_analysis,
        'stochastic_gd': sgd_analysis
    }

def main():
    """Main function to collect and analyze all results"""
    
    print("Collecting experimental results...")
    all_results = collect_all_results()
    
    print("Generating analysis...")
    analysis = generate_summary_report(all_results)
    
    # Save detailed analysis to JSON
    output_file = "experimental_results_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed analysis saved to: {output_file}")
    print("Analysis complete!")

if __name__ == "__main__":
    main()