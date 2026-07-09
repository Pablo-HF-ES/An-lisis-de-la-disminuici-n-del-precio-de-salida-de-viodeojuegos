"""
Unit tests for model training and prediction modules.
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.train import train_model
from src.predict import predict


class TestModelTraining:
    """Test cases for model training module."""
    
    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic training data."""
        np.random.seed(42)
        n_samples = 100
        
        X = np.random.randn(n_samples, 10) * 100
        y = (
            2 * X[:, 0] + 
            1.5 * X[:, 1] - 
            3 * X[:, 2] + 
            np.random.randn(n_samples) * 10
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return (
            pd.DataFrame(X_train),
            pd.DataFrame(X_test),
            pd.Series(y_train),
            pd.Series(y_test)
        )
    
    def test_train_linear_model(self, synthetic_data):
        """Test linear regression model training."""
        X_train, X_test, y_train, y_test = synthetic_data
        
        model, metrics = train_model(X_train, y_train, X_test, y_test, model_type='linear')
        
        assert model is not None, "Model was not created"
        assert 'RMSE' in metrics, "RMSE metric not found"
        assert 'R2' in metrics, "R2 metric not found"
        assert metrics['RMSE'] >= 0, "RMSE should be non-negative"
        assert -1 <= metrics['R2'] <= 1, "R2 should be between -1 and 1"
    
    def test_train_ridge_model(self, synthetic_data):
        """Test Ridge regression model training."""
        X_train, X_test, y_train, y_test = synthetic_data
        
        model, metrics = train_model(X_train, y_train, X_test, y_test, model_type='ridge')
        
        assert model is not None, "Ridge model was not created"
        assert 'RMSE' in metrics
        assert 'R2' in metrics
        assert metrics['RMSE'] >= 0
    
    def test_train_lasso_model(self, synthetic_data):
        """Test Lasso regression model training."""
        X_train, X_test, y_train, y_test = synthetic_data
        
        model, metrics = train_model(X_train, y_train, X_test, y_test, model_type='lasso')
        
        assert model is not None, "Lasso model was not created"
        assert 'RMSE' in metrics
        assert 'R2' in metrics
    
    def test_train_elasticnet_model(self, synthetic_data):
        """Test ElasticNet regression model training."""
        X_train, X_test, y_train, y_test = synthetic_data
        
        model, metrics = train_model(
            X_train, y_train, X_test, y_test, model_type='elasticnet'
        )
        
        assert model is not None, "ElasticNet model was not created"
        assert 'RMSE' in metrics
        assert 'R2' in metrics
    
    def test_train_invalid_model_type(self, synthetic_data):
        """Test that invalid model type raises error."""
        X_train, X_test, y_train, y_test = synthetic_data
        
        with pytest.raises(ValueError):
            train_model(X_train, y_train, X_test, y_test, model_type='invalid')
    
    def test_model_makes_predictions(self, synthetic_data):
        """Test that trained model can make predictions."""
        X_train, X_test, y_train, y_test = synthetic_data
        
        model, _ = train_model(X_train, y_train, X_test, y_test, model_type='linear')
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test), "Prediction count mismatch"
        assert not np.isnan(predictions).any(), "Predictions contain NaN"
    
    def test_model_metrics_are_numbers(self, synthetic_data):
        """Test that model metrics are valid numbers."""
        X_train, X_test, y_train, y_test = synthetic_data
        
        model, metrics = train_model(X_train, y_train, X_test, y_test, model_type='ridge')
        
        assert isinstance(metrics['RMSE'], (int, float, np.number)), \
            "RMSE is not a number"
        assert isinstance(metrics['R2'], (int, float, np.number)), \
            "R2 is not a number"


class TestModelSerialization:
    """Test cases for model saving and loading."""
    
    @pytest.fixture
    def trained_model(self):
        """Create and return a trained model."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.randn(50)
        X_test = np.random.randn(10, 5)
        y_test = np.random.randn(10)
        
        model, _ = train_model(
            pd.DataFrame(X), pd.Series(y), 
            pd.DataFrame(X_test), pd.Series(y_test),
            model_type='ridge'
        )
        return model
    
    def test_save_and_load_model(self, trained_model):
        """Test saving and loading model with joblib."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.joblib')
            
            # Save model
            joblib.dump(trained_model, model_path)
            assert os.path.exists(model_path), "Model file was not saved"
            
            # Load model
            loaded_model = joblib.load(model_path)
            assert loaded_model is not None, "Model could not be loaded"
    
    def test_loaded_model_makes_same_predictions(self, trained_model):
        """Test that loaded model makes identical predictions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.joblib')
            joblib.dump(trained_model, model_path)
            loaded_model = joblib.load(model_path)
            
            # Test data
            test_X = np.random.randn(5, 5)
            
            # Predictions
            original_pred = trained_model.predict(test_X)
            loaded_pred = loaded_model.predict(test_X)
            
            np.testing.assert_array_almost_equal(
                original_pred, loaded_pred,
                err_msg="Loaded model predictions differ from original"
            )


class TestModelPrediction:
    """Test cases for prediction module."""
    
    @pytest.fixture
    def setup_prediction_env(self):
        """Setup environment for prediction testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sample data
            np.random.seed(42)
            train_data = pd.DataFrame(
                np.random.randn(100, 5),
                columns=[f'feature_{i}' for i in range(5)]
            )
            train_data['launch_price_decay'] = np.random.rand(100) * 0.5
            
            # Save training data
            interim_path = os.path.join(tmpdir, 'interim.csv')
            train_data.to_csv(interim_path, index=False)
            
            # Train and save model
            from src.features import build_features
            models_dir = os.path.join(tmpdir, 'models')
            processed_dir = os.path.join(tmpdir, 'processed')
            
            build_features(interim_path, processed_dir, models_dir)
            
            # Train model
            X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
            y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).values.ravel()
            X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
            y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv')).values.ravel()
            
            model, _ = train_model(X_train, y_train, X_test, y_test, model_type='ridge')
            joblib.dump(model, os.path.join(models_dir, 'modelo_ridge.joblib'))
            
            yield {
                'tmpdir': tmpdir,
                'models_dir': models_dir,
                'data': train_data
            }
    
    def test_prediction_creates_output(self, setup_prediction_env):
        """Test that predict function creates output file."""
        env = setup_prediction_env
        
        # Create new data for prediction
        new_data_path = os.path.join(env['tmpdir'], 'new_data.csv')
        new_data = pd.DataFrame(
            np.random.randn(10, 5),
            columns=[f'feature_{i}' for i in range(5)]
        )
        new_data.to_csv(new_data_path, index=False)
        
        # Make predictions
        output_path = os.path.join(env['tmpdir'], 'predictions.csv')
        predict(new_data_path, env['models_dir'], output_path, tipo_modelo='ridge')
        
        # Verify output
        assert os.path.exists(output_path), "Prediction output file was not created"
    
    def test_prediction_output_has_predictions(self, setup_prediction_env):
        """Test that prediction output contains prediction column."""
        env = setup_prediction_env
        
        new_data_path = os.path.join(env['tmpdir'], 'new_data.csv')
        new_data = pd.DataFrame(
            np.random.randn(5, 5),
            columns=[f'feature_{i}' for i in range(5)]
        )
        new_data.to_csv(new_data_path, index=False)
        
        output_path = os.path.join(env['tmpdir'], 'predictions.csv')
        predict(new_data_path, env['models_dir'], output_path, tipo_modelo='ridge')
        
        # Check output
        result = pd.read_csv(output_path)
        assert 'prediccion_launch_price_decay' in result.columns, \
            "Prediction column not found in output"
        assert len(result) == 5, "Output should have same number of rows as input"


class TestModelPerformance:
    """Test cases for model performance metrics."""
    
    def test_model_r2_score_reasonable(self):
        """Test that R2 score is reasonable for synthetic data."""
        np.random.seed(42)
        
        # Create data where y is highly correlated with X
        X = np.random.randn(100, 3)
        y = X[:, 0] * 2 + X[:, 1] * 3 - X[:, 2]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model, metrics = train_model(
            pd.DataFrame(X_train), pd.Series(y_train),
            pd.DataFrame(X_test), pd.Series(y_test),
            model_type='linear'
        )
        
        # For highly correlated data, R2 should be high
        assert metrics['R2'] > 0.5, f"R2 score too low: {metrics['R2']}"
    
    def test_all_models_produce_metrics(self):
        """Test that all model types produce valid metrics."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.randn(50)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model_types = ['linear', 'ridge', 'lasso', 'elasticnet']
        
        for model_type in model_types:
            model, metrics = train_model(
                pd.DataFrame(X_train), pd.Series(y_train),
                pd.DataFrame(X_test), pd.Series(y_test),
                model_type=model_type
            )
            
            assert 'RMSE' in metrics, f"{model_type} missing RMSE"
            assert 'R2' in metrics, f"{model_type} missing R2"
            assert metrics['RMSE'] >= 0, f"{model_type} has negative RMSE"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
