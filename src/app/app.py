#!/usr/bin/env python3
"""
Master Optimization Framework - Flask Web Application
Main application file with routes and configurations
"""

import os
import sys
import json
import tempfile
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from flask import (
    Flask, render_template, request, redirect, url_for, 
    flash, jsonify, send_file, send_from_directory, abort
)
from werkzeug.utils import secure_filename

# Import our modules
try:
    from .forms import ExperimentForm, PredictionForm, ComparisonForm, ModelExportForm
    from .models.algorithm_registry import get_all_algorithms, get_algorithms_by_category, get_algorithm, get_algorithm_variant
    from .models.setup_generator import SetupFileGenerator
    from .models.model_loader import ModelLoader, list_all_models
    from .models.comparison_engine import WebComparisonEngine, get_quick_comparison, get_full_comparison_data
except ImportError:
    from forms import ExperimentForm, PredictionForm, ComparisonForm, ModelExportForm
    from models.algorithm_registry import get_all_algorithms, get_algorithms_by_category, get_algorithm, get_algorithm_variant
    from models.setup_generator import SetupFileGenerator
    from models.model_loader import ModelLoader, list_all_models
    from models.comparison_engine import WebComparisonEngine, get_quick_comparison, get_full_comparison_data

# Create Flask app
app = Flask(__name__)
try:
    from .config import Config
    app.config.from_object(Config)
except ImportError:
    from config import Config
    app.config.from_object(Config)

# Set secret key (should be in environment variable in production)
app.secret_key = os.environ.get('SECRET_KEY', 'master-optimization-dev-key-change-in-production')

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'json', 'txt'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize components
setup_generator = SetupFileGenerator()
model_loader = ModelLoader()
comparison_engine = WebComparisonEngine()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('errors/500.html'), 500

# Main routes
@app.route('/')
def index():
    """Dashboard - Main page"""
    try:
        # Get quick comparison data for dashboard
        summary = get_quick_comparison()
        
        # Get recent experiments and top performers
        all_models = list_all_models()
        recent_experiments = []
        top_performers = []
        
        # Process models data
        all_experiments = []
        for algorithm_name, experiments in all_models.items():
            for exp in experiments:
                exp['algorithm_family'] = algorithm_name
                all_experiments.append(exp)
        
        # Sort by creation time for recent
        recent_experiments = sorted(all_experiments, 
                                  key=lambda x: x.get('created_time', 0), 
                                  reverse=True)[:5]
        
        # Sort by performance for top performers (mock for now)
        top_performers = sorted([exp for exp in all_experiments if exp.get('converged', False)], 
                               key=lambda x: x.get('training_time', float('inf')))[:5]
        
        return render_template('index.html', 
                             summary=summary,
                             recent_experiments=recent_experiments,
                             top_performers=top_performers)
    except Exception as e:
        app.logger.error(f"Error loading dashboard: {e}")
        return render_template('index.html', 
                             summary={}, 
                             recent_experiments=[], 
                             top_performers=[])

@app.route('/create-experiment', methods=['GET', 'POST'])
def create_experiment():
    """Create new experiment page"""
    form = ExperimentForm()
    algorithms = get_all_algorithms()
    algorithms_by_category = get_algorithms_by_category()
    
    # Convert algorithms to JSON for JavaScript
    algorithms_json = {}
    for name, algorithm in algorithms.items():
        algorithms_json[name] = {
            'name': algorithm.name,
            'display_name': algorithm.display_name,
            'description': algorithm.description,
            'variants': []
        }
        
        for variant in algorithm.variants:
            variant_data = {
                'name': variant.name,
                'display_name': variant.display_name,
                'loss_function': variant.loss_function,
                'setup_template': variant.setup_template,
                'parameters': []
            }
            
            for param in variant.parameters:
                param_data = {
                    'name': param.name,
                    'param_type': param.param_type.value,
                    'min_value': param.min_value,
                    'max_value': param.max_value,
                    'default_value': param.default_value,
                    'choices': param.choices,
                    'description': param.description,
                    'step': param.step
                }
                variant_data['parameters'].append(param_data)
            
            algorithms_json[name]['variants'].append(variant_data)
    
    if request.method == 'POST':
        try:
            # Get form data
            algorithm_name = request.form.get('algorithm')
            variant_name = request.form.get('variant')
            parameters_json = request.form.get('parameters')
            action = request.form.get('action')
            
            if not all([algorithm_name, variant_name, parameters_json]):
                flash('Please complete all required fields.', 'error')
                return redirect(url_for('create_experiment'))
            
            # Parse parameters
            parameters = json.loads(parameters_json)
            
            # Get algorithm and variant objects
            algorithm = get_algorithm(algorithm_name)
            variant = get_algorithm_variant(algorithm_name, variant_name)
            
            if not algorithm or not variant:
                flash('Invalid algorithm or variant selected.', 'error')
                return redirect(url_for('create_experiment'))
            
            # Generate setup file
            setup_path = setup_generator.generate_setup_file(
                algorithm, variant, parameters
            )
            
            flash(f'Setup file generated successfully: {setup_path}', 'success')
            
            # Handle different actions
            if action == 'generate_and_run':
                # TODO: Implement background execution
                flash('Experiment setup generated. You can now run it manually.', 'info')
            elif action == 'download':
                # Send file for download
                return send_file(setup_path, as_attachment=True)
            
            return redirect(url_for('model_browser'))
            
        except Exception as e:
            app.logger.error(f"Error creating experiment: {e}")
            flash(f'Error creating experiment: {str(e)}', 'error')
    
    return render_template('create_experiment.html', 
                         form=form,
                         algorithms_by_category=algorithms_by_category,
                         algorithms_json=json.dumps(algorithms_json))

@app.route('/models')
def model_browser():
    """Browse saved models"""
    try:
        models = list_all_models()
        
        # Get available algorithms for filtering
        available_algorithms = list(models.keys()) if models else []
        
        return render_template('model_browser.html', 
                             models=models,
                             available_algorithms=available_algorithms)
    except Exception as e:
        app.logger.error(f"Error loading models: {e}")
        flash('Error loading models. Please try again.', 'error')
        return render_template('model_browser.html', 
                             models={}, 
                             available_algorithms=[])

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Make predictions with trained models"""
    form = PredictionForm()
    
    # Populate model choices
    try:
        all_models = list_all_models()
        model_choices = []
        
        for algorithm_name, experiments in all_models.items():
            for exp in experiments:
                if exp.get('prediction_ready', False):
                    label = f"{exp['experiment_name']} ({exp.get('algorithm', 'Unknown')})"
                    model_choices.append((exp['experiment_path'], label))
        
        form.model_path.choices = model_choices
        
        # Pre-select model from URL parameter
        selected_model = request.args.get('model')
        if selected_model:
            form.model_path.data = selected_model
            
    except Exception as e:
        app.logger.error(f"Error loading models for prediction: {e}")
        model_choices = []
        form.model_path.choices = model_choices
    
    if request.method == 'POST' and form.validate_on_submit():
        try:
            model_path = form.model_path.data
            data_input_method = form.data_input_method.data
            
            if not model_path:
                flash('Please select a model for prediction.', 'error')
                return redirect(url_for('predict'))
            
            predictions = None
            results_df = None
            
            if data_input_method == 'upload' and form.data_file.data:
                # Handle file upload
                file = form.data_file.data
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    temp_path = os.path.join(UPLOAD_FOLDER, filename)
                    file.save(temp_path)
                    
                    # Make predictions
                    predictions, results_df = model_loader.predict_from_file(
                        model_path, temp_path
                    )
                    
                    # Clean up temp file
                    os.remove(temp_path)
                else:
                    flash('Please upload a valid CSV file.', 'error')
                    return redirect(url_for('predict'))
            
            elif data_input_method == 'manual' and form.manual_features.data:
                # Handle manual input
                try:
                    import numpy as np
                    lines = form.manual_features.data.strip().split('\n')
                    features_list = []
                    
                    for line in lines:
                        if line.strip():
                            features = [float(x.strip()) for x in line.split(',')]
                            features_list.append(features)
                    
                    if features_list:
                        # Make predictions for each sample
                        predictions = []
                        for features in features_list:
                            pred = model_loader.predict_single_sample(model_path, features)
                            predictions.append(pred)
                        
                        # Create results dataframe
                        import pandas as pd
                        results_df = pd.DataFrame({
                            'prediction': predictions
                        })
                        
                        # Add input features
                        for i, features in enumerate(features_list):
                            for j, feature in enumerate(features):
                                results_df.loc[i, f'feature_{j+1}'] = feature
                    
                except Exception as e:
                    flash(f'Error parsing manual input: {str(e)}', 'error')
                    return redirect(url_for('predict'))
            
            else:
                flash('Please provide input data for prediction.', 'error')
                return redirect(url_for('predict'))
            
            if predictions is not None and results_df is not None:
                # Save results to temp file for download
                if form.download_results.data:
                    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
                    results_df.to_csv(temp_file.name, index=False)
                    temp_file.close()
                    
                    return send_file(temp_file.name, 
                                   as_attachment=True, 
                                   download_name='predictions.csv',
                                   mimetype='text/csv')
                
                # Display results on page
                flash(f'Predictions completed successfully. {len(predictions)} samples processed.', 'success')
                return render_template('prediction_results.html', 
                                     predictions=predictions, 
                                     results_df=results_df)
            
        except Exception as e:
            app.logger.error(f"Error during prediction: {e}")
            flash(f'Error during prediction: {str(e)}', 'error')
    
    return render_template('predict.html', form=form)

@app.route('/compare')
def compare_algorithms():
    """Algorithm comparison page"""
    try:
        comparison_data = get_full_comparison_data()
        
        return render_template('compare_algorithms.html',
                             dataframe=comparison_data['dataframe'],
                             plots=comparison_data['plots'],
                             summary=comparison_data['summary'],
                             top_performers=comparison_data['top_performers'])
    except Exception as e:
        app.logger.error(f"Error loading comparison data: {e}")
        flash('Error loading comparison data. Please try again.', 'error')
        return render_template('compare_algorithms.html',
                             dataframe=None,
                             plots={},
                             summary={},
                             top_performers=[])

# API routes
@app.route('/api/model-details')
def api_model_details():
    """API endpoint for model details"""
    model_path = request.args.get('path')
    if not model_path:
        return jsonify({'error': 'Model path is required'}), 400
    
    try:
        model_info = model_loader.get_model_summary(model_path)
        return jsonify(model_info)
    except Exception as e:
        app.logger.error(f"Error getting model details: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/download-model')
def api_download_model():
    """API endpoint for downloading model files"""
    model_path = request.args.get('path')
    if not model_path:
        return jsonify({'error': 'Model path is required'}), 400
    
    try:
        # Create a temporary zip file with model contents
        model_dir = Path(model_path)
        if not model_dir.exists():
            return jsonify({'error': 'Model not found'}), 404
        
        temp_zip = tempfile.NamedTemporaryFile(mode='w+b', suffix='.zip', delete=False)
        
        with zipfile.ZipFile(temp_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in model_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(model_dir)
                    zipf.write(file_path, arcname)
        
        temp_zip.close()
        
        return send_file(temp_zip.name,
                        as_attachment=True,
                        download_name=f"{model_dir.name}.zip",
                        mimetype='application/zip')
    
    except Exception as e:
        app.logger.error(f"Error downloading model: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete-model', methods=['POST'])
def api_delete_model():
    """API endpoint for deleting models"""
    data = request.get_json()
    if not data or 'path' not in data:
        return jsonify({'error': 'Model path is required'}), 400
    
    try:
        import shutil
        model_path = Path(data['path'])
        
        if not model_path.exists():
            return jsonify({'error': 'Model not found'}), 404
        
        # Remove the entire experiment directory
        shutil.rmtree(model_path)
        
        return jsonify({'success': True, 'message': 'Model deleted successfully'})
    
    except Exception as e:
        app.logger.error(f"Error deleting model: {e}")
        return jsonify({'error': str(e)}), 500

# Utility routes
@app.route('/experiment-runner')
def experiment_runner():
    """Experiment runner interface (placeholder)"""
    return render_template('experiment_runner.html')

@app.route('/export-data')
def export_data():
    """Data export interface (placeholder)"""
    return render_template('export_data.html')

@app.route('/help')
def help():
    """Help and documentation page"""
    return render_template('help.html')

# Template filters and functions
@app.template_filter('datetime')
def datetime_filter(timestamp):
    """Format timestamp to readable datetime"""
    if timestamp:
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    return 'Unknown'

@app.template_global()
def moment():
    """Placeholder for moment.js functionality"""
    return None

# Development server
if __name__ == '__main__':
    # Enable debug mode for development
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("Starting Master Optimization Framework Web Interface...")
    print(f"Debug mode: {debug_mode}")
    print("Available at: http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)