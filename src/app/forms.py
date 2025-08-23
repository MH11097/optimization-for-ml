"""
Flask-WTF Forms for Master Optimization Framework
"""

from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import (
    StringField, SelectField, FloatField, IntegerField, 
    BooleanField, TextAreaField, HiddenField, SubmitField
)
from wtforms.validators import DataRequired, NumberRange, Length, Optional
from wtforms.widgets import TextArea

class ExperimentForm(FlaskForm):
    """Form for creating new experiments"""
    
    # Algorithm selection
    algorithm = SelectField(
        'Algorithm',
        choices=[],  # Will be populated dynamically
        validators=[DataRequired()]
    )
    
    variant = SelectField(
        'Variant',
        choices=[],  # Will be populated dynamically
        validators=[DataRequired()]
    )
    
    # Parameters (will be added dynamically based on algorithm)
    parameters = HiddenField('Parameters')  # JSON string of all parameters
    
    # Action buttons
    action = HiddenField('Action')
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class PredictionForm(FlaskForm):
    """Form for making predictions with trained models"""
    
    model_path = SelectField(
        'Select Model',
        choices=[],  # Will be populated with available models
        validators=[DataRequired()],
        description='Choose a trained model for prediction'
    )
    
    # Data input options
    data_input_method = SelectField(
        'Data Input Method',
        choices=[
            ('upload', 'Upload CSV File'),
            ('manual', 'Manual Input'),
            ('sample', 'Use Sample Data')
        ],
        default='upload',
        validators=[DataRequired()]
    )
    
    # File upload
    data_file = FileField(
        'Upload Data File',
        validators=[
            FileAllowed(['csv'], 'CSV files only!'),
            Optional()
        ],
        description='Upload a CSV file with features for prediction'
    )
    
    # Manual input
    manual_features = TextAreaField(
        'Manual Feature Input',
        validators=[Optional()],
        description='Enter feature values separated by commas, one sample per line',
        widget=TextArea()
    )
    
    # Output options
    include_confidence = BooleanField(
        'Include Confidence Intervals',
        default=False,
        description='Include prediction confidence (if supported by model)'
    )
    
    download_results = BooleanField(
        'Download Results as CSV',
        default=True,
        description='Automatically download prediction results'
    )
    
    submit = SubmitField('Make Predictions')

class ComparisonForm(FlaskForm):
    """Form for algorithm comparison"""
    
    # Models to compare (checkboxes will be added dynamically)
    selected_models = HiddenField('Selected Models')  # JSON array of model paths
    
    # Comparison options
    comparison_type = SelectField(
        'Comparison Type',
        choices=[
            ('performance', 'Performance Comparison'),
            ('efficiency', 'Training Efficiency'),
            ('convergence', 'Convergence Analysis'),
            ('detailed', 'Detailed Analysis')
        ],
        default='performance',
        validators=[DataRequired()]
    )
    
    # Visualization options
    include_plots = BooleanField(
        'Generate Visualization',
        default=True,
        description='Generate comparison plots and charts'
    )
    
    export_report = BooleanField(
        'Export Detailed Report',
        default=False,
        description='Generate and download comprehensive comparison report'
    )
    
    submit = SubmitField('Compare Models')

class ParameterTuningForm(FlaskForm):
    """Form for hyperparameter tuning (future feature)"""
    
    base_algorithm = SelectField(
        'Base Algorithm',
        choices=[],
        validators=[DataRequired()]
    )
    
    tuning_method = SelectField(
        'Tuning Method',
        choices=[
            ('grid', 'Grid Search'),
            ('random', 'Random Search'),
            ('bayesian', 'Bayesian Optimization')
        ],
        default='grid',
        validators=[DataRequired()]
    )
    
    max_iterations = IntegerField(
        'Maximum Iterations',
        default=50,
        validators=[DataRequired(), NumberRange(min=10, max=500)]
    )
    
    submit = SubmitField('Start Tuning')

class ModelExportForm(FlaskForm):
    """Form for exporting models and results"""
    
    model_paths = HiddenField('Model Paths')  # JSON array
    
    export_format = SelectField(
        'Export Format',
        choices=[
            ('zip', 'ZIP Archive'),
            ('tar', 'TAR Archive'),
            ('json', 'JSON Report'),
            ('csv', 'CSV Data')
        ],
        default='zip',
        validators=[DataRequired()]
    )
    
    include_data = BooleanField(
        'Include Training Data',
        default=False,
        description='Include original training data in export'
    )
    
    include_plots = BooleanField(
        'Include Visualization',
        default=True,
        description='Include generated plots and charts'
    )
    
    include_code = BooleanField(
        'Include Setup Code',
        default=True,
        description='Include generated setup files'
    )
    
    submit = SubmitField('Export')

class DataUploadForm(FlaskForm):
    """Form for uploading training data"""
    
    data_file = FileField(
        'Training Data File',
        validators=[
            FileRequired(),
            FileAllowed(['csv', 'json'], 'CSV and JSON files only!')
        ],
        description='Upload your training dataset'
    )
    
    target_column = StringField(
        'Target Column Name',
        validators=[DataRequired(), Length(min=1, max=100)],
        description='Name of the target/output column'
    )
    
    test_size = FloatField(
        'Test Split Ratio',
        default=0.2,
        validators=[DataRequired(), NumberRange(min=0.1, max=0.5)],
        description='Fraction of data to use for testing (0.1 - 0.5)'
    )
    
    random_state = IntegerField(
        'Random Seed',
        default=42,
        validators=[Optional(), NumberRange(min=0, max=9999)],
        description='Seed for reproducible train/test split'
    )
    
    preprocessing_options = SelectField(
        'Preprocessing',
        choices=[
            ('standard', 'Standard Scaling'),
            ('minmax', 'Min-Max Scaling'), 
            ('robust', 'Robust Scaling'),
            ('none', 'No Scaling')
        ],
        default='standard',
        validators=[DataRequired()]
    )
    
    submit = SubmitField('Upload and Process')

class ExperimentRunnerForm(FlaskForm):
    """Form for running experiments directly through web interface"""
    
    setup_file_path = HiddenField('Setup File Path', validators=[DataRequired()])
    
    run_options = SelectField(
        'Execution Mode',
        choices=[
            ('foreground', 'Run in Foreground'),
            ('background', 'Run in Background'),
            ('scheduled', 'Schedule for Later')
        ],
        default='foreground',
        validators=[DataRequired()]
    )
    
    notifications = BooleanField(
        'Email Notifications',
        default=False,
        description='Send email when experiment completes'
    )
    
    save_intermediate = BooleanField(
        'Save Intermediate Results',
        default=True,
        description='Save progress checkpoints during training'
    )
    
    submit = SubmitField('Run Experiment')

# Dynamic form field generators
def create_parameter_field(parameter):
    """Create appropriate form field for a parameter"""
    from .models.algorithm_registry import ParameterType
    
    field_kwargs = {
        'label': parameter.name.replace('_', ' ').title(),
        'description': parameter.description,
        'default': parameter.default_value
    }
    
    if parameter.param_type == ParameterType.FLOAT:
        validators = [DataRequired()]
        if parameter.min_value is not None and parameter.max_value is not None:
            validators.append(NumberRange(min=parameter.min_value, max=parameter.max_value))
        
        return FloatField(
            validators=validators,
            **field_kwargs
        )
    
    elif parameter.param_type == ParameterType.INTEGER:
        validators = [DataRequired()]
        if parameter.min_value is not None and parameter.max_value is not None:
            validators.append(NumberRange(min=int(parameter.min_value), max=int(parameter.max_value)))
        
        return IntegerField(
            validators=validators,
            **field_kwargs
        )
    
    elif parameter.param_type == ParameterType.CHOICE:
        choices = [(choice, choice) for choice in parameter.choices]
        return SelectField(
            choices=choices,
            validators=[DataRequired()],
            **field_kwargs
        )
    
    elif parameter.param_type == ParameterType.BOOLEAN:
        return BooleanField(**field_kwargs)
    
    else:  # STRING or default
        return StringField(
            validators=[DataRequired()],
            **field_kwargs
        )

def create_dynamic_form(algorithm, variant):
    """Create a dynamic form class for specific algorithm variant"""
    
    class DynamicParameterForm(FlaskForm):
        pass
    
    # Add parameter fields
    for parameter in variant.parameters:
        field = create_parameter_field(parameter)
        setattr(DynamicParameterForm, f'param_{parameter.name}', field)
    
    # Add submit button
    setattr(DynamicParameterForm, 'submit', SubmitField('Generate Setup'))
    
    return DynamicParameterForm