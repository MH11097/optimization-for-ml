from flask import Flask, render_template, jsonify, request
import os
import sys
from pathlib import Path
# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from web_app.data_loader import AlgorithmDataLoader
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
# Initialize data loader
DATA_ROOT = project_root / "data"
data_loader = AlgorithmDataLoader(str(DATA_ROOT))

@app.route('/')
def index():
    """Main visualization page."""
    return render_template('index.html')

@app.route('/api/algorithms')
def get_algorithms():
    """Get list of available algorithms."""
    try:
        algorithms = data_loader.get_available_algorithms()
        return jsonify({
            'success': True,
            'algorithms': algorithms
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/algorithms/<algorithm>/setups')
def get_algorithm_setups(algorithm):
    """Get all setups for a specific algorithm."""
    try:
        setups = data_loader.get_algorithm_setups(algorithm)
        return jsonify({
            'success': True,
            'setups': setups
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/algorithms/<algorithm>/parameter-ranges')
def get_parameter_ranges(algorithm):
    """Get parameter ranges for creating sliders."""
    try:
        ranges = data_loader.get_parameter_ranges(algorithm)
        return jsonify({
            'success': True,
            'parameter_ranges': ranges
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/algorithms/<algorithm>/grouped-setups')
def get_grouped_setups(algorithm):
    """Get setups grouped by parameter type."""
    try:
        grouped = data_loader.get_grouped_setups(algorithm)
        return jsonify({
            'success': True,
            'grouped_setups': grouped
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/algorithms/<algorithm>/setup-by-params')
def get_setup_by_params(algorithm):
    """Find setup matching given parameters."""
    try:
        # Get parameters from query string
        target_params = {}
        tolerance = float(request.args.get('tolerance', 0.001))
        
        for key, value in request.args.items():
            if key != 'tolerance':
                try:
                    target_params[key] = float(value)
                except ValueError:
                    target_params[key] = value
        
        setup = data_loader.get_setup_by_parameters(algorithm, target_params, tolerance)
        
        if setup:
            return jsonify({
                'success': True,
                'setup': setup
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No setup found matching the given parameters'
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/setup/<path:setup_path>/history')
def get_training_history(setup_path):
    """Get training history for a specific setup."""
    try:
        # Reconstruct full path
        full_path = DATA_ROOT / "03_algorithms" / setup_path
        
        if not full_path.exists():
            return jsonify({
                'success': False,
                'error': 'Setup path not found'
            }), 404
        
        # Load training history
        history_file = full_path / "training_history.csv"
        if not history_file.exists():
            return jsonify({
                'success': False,
                'error': 'Training history not available'
            }), 404
        
        import pandas as pd
        df = pd.read_csv(history_file)
        history = df.to_dict('records')
        
        return jsonify({
            'success': True,
            'history': history,
            'columns': list(df.columns)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/comparison')
def compare_setups():
    """Compare multiple setups."""
    try:
        setup_paths = request.args.getlist('setups')
        if not setup_paths:
            return jsonify({
                'success': False,
                'error': 'No setups specified for comparison'
            }), 400
        
        comparison_data = []
        
        for setup_path in setup_paths:
            full_path = DATA_ROOT / "03_algorithms" / setup_path
            
            if full_path.exists():
                # Load results
                results_file = full_path / "results.json"
                history_file = full_path / "training_history.csv"
                
                setup_data = {
                    'setup_path': setup_path,
                    'setup_name': full_path.name
                }
                
                if results_file.exists():
                    import json
                    with open(results_file, 'r') as f:
                        setup_data['results'] = json.load(f)
                
                if history_file.exists():
                    import pandas as pd
                    df = pd.read_csv(history_file)
                    setup_data['history'] = df.to_dict('records')
                
                comparison_data.append(setup_data)
        
        return jsonify({
            'success': True,
            'comparison': comparison_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Check if data directory exists
    if not DATA_ROOT.exists():
        print(f"Warning: Data directory not found at {DATA_ROOT}")
        print("Please ensure the optimization algorithm data is available.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)