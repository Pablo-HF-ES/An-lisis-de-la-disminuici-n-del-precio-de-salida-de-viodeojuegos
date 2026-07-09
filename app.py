"""
Flask API for Video Game Price Degradation Prediction Model

This module provides REST endpoints for making predictions using trained ML models.
Users can send game data and receive price degradation predictions.
"""

import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DEFAULT_MODEL = 'ridge'

# Global model and pipeline cache
models_cache = {}
pipeline_cache = None


def load_pipeline():
    """Load preprocessing pipeline."""
    global pipeline_cache
    if pipeline_cache is None:
        pipeline_path = os.path.join(MODELS_DIR, 'preprocessing_pipeline.joblib')
        if os.path.exists(pipeline_path):
            pipeline_cache = joblib.load(pipeline_path)
            logger.info("Preprocessing pipeline loaded successfully")
        else:
            logger.warning(f"Pipeline not found at {pipeline_path}")
    return pipeline_cache


def load_model(model_type='ridge'):
    """Load a trained model from cache or disk."""
    if model_type not in models_cache:
        model_path = os.path.join(MODELS_DIR, f'modelo_{model_type}.joblib')
        if os.path.exists(model_path):
            models_cache[model_type] = joblib.load(model_path)
            logger.info(f"Model {model_type} loaded successfully")
        else:
            logger.error(f"Model not found at {model_path}")
            return None
    return models_cache[model_type]


def validate_input_data(data):
    """Validate input data structure."""
    if not isinstance(data, (list, dict)):
        return False, "Data must be a dictionary or list of dictionaries"
    
    if isinstance(data, dict):
        data = [data]
    
    if len(data) == 0:
        return False, "Data list cannot be empty"
    
    return True, ""


def preprocess_data(data):
    """Preprocess input data for prediction."""
    try:
        # Convert to DataFrame if necessary
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
        
        # Fill NaN values with 0
        df.fillna(value=0, inplace=True)
        
        # Select only numeric columns
        df_numeric = df.select_dtypes(exclude=['object'])
        
        # Remove target column if present
        if 'launch_price_decay' in df_numeric.columns:
            df_numeric = df_numeric.drop(columns=['launch_price_decay'])
        
        return df_numeric, None
    except Exception as e:
        return None, str(e)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    try:
        pipeline = load_pipeline()
        if pipeline is None:
            return jsonify({'status': 'unhealthy', 'message': 'Pipeline not loaded'}), 503
        
        return jsonify({
            'status': 'healthy',
            'available_models': ['linear', 'ridge', 'lasso', 'elasticnet']
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 503


@app.route('/models', methods=['GET'])
def get_models():
    """Get list of available models."""
    available_models = []
    for model_name in ['linear', 'ridge', 'lasso', 'elasticnet']:
        model_path = os.path.join(MODELS_DIR, f'modelo_{model_name}.joblib')
        if os.path.exists(model_path):
            available_models.append(model_name)
    
    return jsonify({
        'available_models': available_models,
        'default_model': DEFAULT_MODEL
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.
    
    Expected JSON format:
    {
        "data": {
            "feature1": value1,
            "feature2": value2,
            ...
        },
        "model": "ridge"  # optional, defaults to 'ridge'
    }
    
    Or for batch predictions:
    {
        "data": [
            {"feature1": value1, "feature2": value2, ...},
            {"feature1": value1, "feature2": value2, ...}
        ],
        "model": "ridge"  # optional
    }
    """
    try:
        # Parse request JSON
        request_data = request.get_json()
        if request_data is None:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Extract data and model type
        data = request_data.get('data')
        model_type = request_data.get('model', DEFAULT_MODEL)
        
        # Validate input
        is_valid, error_msg = validate_input_data(data)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        # Validate model type
        valid_models = ['linear', 'ridge', 'lasso', 'elasticnet']
        if model_type not in valid_models:
            return jsonify({
                'error': f'Invalid model type. Must be one of: {valid_models}'
            }), 400
        
        # Load pipeline and model
        pipeline = load_pipeline()
        model = load_model(model_type)
        
        if pipeline is None or model is None:
            return jsonify({'error': 'Model or pipeline not available'}), 503
        
        # Preprocess data
        df_processed, error = preprocess_data(data)
        if error:
            return jsonify({'error': f'Data preprocessing error: {error}'}), 400
        
        # Transform data using pipeline
        X_transformed = pipeline.transform(df_processed)
        
        # Make predictions
        predictions = model.predict(X_transformed)
        
        # Format response
        if isinstance(data, dict):
            # Single prediction
            response = {
                'prediction': float(predictions[0]),
                'model': model_type,
                'status': 'success'
            }
        else:
            # Batch predictions
            response = {
                'predictions': [float(pred) for pred in predictions],
                'count': len(predictions),
                'model': model_type,
                'status': 'success'
            }
        
        logger.info(f"Prediction successful using model: {model_type}")
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint for multiple samples.
    
    Expected JSON format:
    {
        "data": [
            {"feature1": value1, "feature2": value2, ...},
            {"feature1": value1, "feature2": value2, ...}
        ],
        "model": "ridge"  # optional
    }
    """
    try:
        request_data = request.get_json()
        if request_data is None:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        data = request_data.get('data', [])
        model_type = request_data.get('model', DEFAULT_MODEL)
        
        # Ensure data is a list
        if not isinstance(data, list):
            data = [data]
        
        if len(data) == 0:
            return jsonify({'error': 'No data provided'}), 400
        
        # Load pipeline and model
        pipeline = load_pipeline()
        model = load_model(model_type)
        
        if pipeline is None or model is None:
            return jsonify({'error': 'Model or pipeline not available'}), 503
        
        # Process all samples
        all_predictions = []
        errors = []
        
        for idx, sample in enumerate(data):
            try:
                df_processed, error = preprocess_data(sample)
                if error:
                    errors.append({'index': idx, 'error': error})
                    continue
                
                X_transformed = pipeline.transform(df_processed)
                prediction = model.predict(X_transformed)[0]
                all_predictions.append({
                    'index': idx,
                    'prediction': float(prediction)
                })
            except Exception as e:
                errors.append({'index': idx, 'error': str(e)})
        
        response = {
            'predictions': all_predictions,
            'total_samples': len(data),
            'successful_predictions': len(all_predictions),
            'errors': errors if errors else None,
            'model': model_type,
            'status': 'partial_success' if errors else 'success'
        }
        
        logger.info(f"Batch prediction completed: {len(all_predictions)} successful")
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/info', methods=['GET'])
def info():
    """Get API information and metadata."""
    return jsonify({
        'service': 'Video Game Price Degradation Predictor',
        'version': '1.0.0',
        'description': 'ML API for predicting video game launch price degradation',
        'endpoints': {
            '/health': 'Health check endpoint',
            '/models': 'List available models',
            '/predict': 'Single or batch prediction (POST)',
            '/predict-batch': 'Batch prediction (POST)',
            '/info': 'API information'
        },
        'available_models': ['linear', 'ridge', 'lasso', 'elasticnet'],
        'default_model': DEFAULT_MODEL
    }), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': [
            '/health',
            '/models',
            '/predict',
            '/predict-batch',
            '/info'
        ]
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500


def main():
    """Run the Flask app."""
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == '__main__':
    main()
