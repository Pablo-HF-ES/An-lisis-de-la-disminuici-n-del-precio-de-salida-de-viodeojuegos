"""
Unit tests for data ingestion and preprocessing modules.
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from src.data_ingestion import data_ingestion
from src.features import build_features


class TestDataIngestion:
    """Test cases for data ingestion module."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        data = {
            'game_name': ['Game1', 'Game2', 'Game3'],
            'release_year': [2020, 2021, 2022],
            'price': [59.99, 49.99, 39.99],
            'launch_price_decay': [0.1, 0.15, 0.2]
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def temp_paths(self):
        """Create temporary paths for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, 'input.csv')
            output_path = os.path.join(tmpdir, 'output.csv')
            yield input_path, output_path
    
    def test_data_ingestion_reads_csv(self, sample_data, temp_paths):
        """Test that data_ingestion reads CSV file correctly."""
        input_path, output_path = temp_paths
        
        # Write sample data to file
        sample_data.to_csv(input_path, index=False)
        
        # Run data ingestion
        data_ingestion(input_path, output_path)
        
        # Verify output file exists
        assert os.path.exists(output_path), "Output file was not created"
        
        # Verify data was read and processed
        result_data = pd.read_csv(output_path)
        assert len(result_data) > 0, "Output data is empty"
    
    def test_data_ingestion_creates_output_directory(self, sample_data, temp_paths):
        """Test that data_ingestion creates output directory if not exists."""
        input_path, output_path = temp_paths
        nested_output = os.path.join(output_path, 'nested', 'path', 'output.csv')
        
        # Write sample data
        sample_data.to_csv(input_path, index=False)
        
        # Run data ingestion with nested path
        data_ingestion(input_path, nested_output)
        
        # Verify directories were created
        assert os.path.exists(nested_output), "Nested output file was not created"
    
    def test_data_ingestion_handles_missing_file(self, temp_paths):
        """Test that data_ingestion handles missing input file gracefully."""
        input_path, output_path = temp_paths
        non_existent_file = os.path.join(input_path, 'non_existent.csv')
        
        # Should raise exception (tested via output)
        with pytest.raises(Exception):
            data_ingestion(non_existent_file, output_path)
    
    def test_data_ingestion_preserves_columns(self, sample_data, temp_paths):
        """Test that data_ingestion preserves column names."""
        input_path, output_path = temp_paths
        
        sample_data.to_csv(input_path, index=False)
        data_ingestion(input_path, output_path)
        
        result_data = pd.read_csv(output_path)
        assert list(result_data.columns) == list(sample_data.columns), \
            "Column names were not preserved"


class TestFeatureEngineering:
    """Test cases for feature engineering module."""
    
    @pytest.fixture
    def sample_processed_data(self):
        """Create sample processed data for feature engineering."""
        data = {
            'revenue': np.random.rand(100) * 1000,
            'marketing_spend': np.random.rand(100) * 500,
            'playtime_hours': np.random.rand(100) * 100,
            'user_score': np.random.rand(100) * 10,
            'launch_price_decay': np.random.rand(100) * 0.5
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def temp_feature_paths(self):
        """Create temporary paths for feature engineering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, 'interim.csv')
            output_dir = os.path.join(tmpdir, 'processed')
            models_dir = os.path.join(tmpdir, 'models')
            yield input_path, output_dir, models_dir
    
    def test_build_features_creates_output_files(self, sample_processed_data, temp_feature_paths):
        """Test that build_features creates all expected output files."""
        input_path, output_dir, models_dir = temp_feature_paths
        
        # Save sample data
        sample_processed_data.to_csv(input_path, index=False)
        
        # Run feature engineering
        build_features(input_path, output_dir, models_dir)
        
        # Verify all expected files were created
        expected_files = ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']
        for file in expected_files:
            file_path = os.path.join(output_dir, file)
            assert os.path.exists(file_path), f"{file} was not created"
    
    def test_build_features_creates_pipeline(self, sample_processed_data, temp_feature_paths):
        """Test that build_features creates preprocessing pipeline."""
        input_path, output_dir, models_dir = temp_feature_paths
        
        sample_processed_data.to_csv(input_path, index=False)
        build_features(input_path, output_dir, models_dir)
        
        pipeline_path = os.path.join(models_dir, 'preprocessing_pipeline.joblib')
        assert os.path.exists(pipeline_path), "Preprocessing pipeline was not created"
    
    def test_build_features_train_test_split(self, sample_processed_data, temp_feature_paths):
        """Test that build_features correctly splits data."""
        input_path, output_dir, models_dir = temp_feature_paths
        
        sample_processed_data.to_csv(input_path, index=False)
        build_features(input_path, output_dir, models_dir)
        
        # Load train and test sets
        X_train = pd.read_csv(os.path.join(output_dir, 'X_train.csv'))
        X_test = pd.read_csv(os.path.join(output_dir, 'X_test.csv'))
        
        # Verify split ratio (80-20)
        total_rows = len(X_train) + len(X_test)
        expected_train_rows = int(len(sample_processed_data) * 0.8)
        
        # Allow small deviation due to integer division
        assert abs(len(X_train) - expected_train_rows) <= 1, \
            f"Train set size incorrect. Expected ~{expected_train_rows}, got {len(X_train)}"
    
    def test_build_features_handles_nan_values(self, temp_feature_paths):
        """Test that build_features handles NaN values correctly."""
        input_path, output_dir, models_dir = temp_feature_paths
        
        # Create data with NaN values
        data = {
            'feature1': [1.0, np.nan, 3.0, 4.0] * 25,
            'feature2': [5.0, 6.0, np.nan, 8.0] * 25,
            'launch_price_decay': np.random.rand(100) * 0.5
        }
        df = pd.DataFrame(data)
        df.to_csv(input_path, index=False)
        
        # Should handle NaN without error
        build_features(input_path, output_dir, models_dir)
        
        # Verify output files exist
        assert os.path.exists(os.path.join(output_dir, 'X_train.csv'))


class TestDataQuality:
    """Test cases for data quality checks."""
    
    def test_no_all_null_columns(self, sample_processed_data):
        """Test that no columns are completely null."""
        for col in sample_processed_data.columns:
            assert sample_processed_data[col].notna().any(), \
                f"Column {col} contains all null values"
    
    def test_numeric_columns_are_numeric(self):
        """Test that numeric columns have correct dtype."""
        data = {
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
        }
        df = pd.DataFrame(data)
        
        assert df['int_col'].dtype in [np.int64, np.int32]
        assert df['float_col'].dtype == np.float64
    
    @pytest.fixture
    def sample_processed_data(self):
        """Create sample processed data for feature engineering."""
        data = {
            'revenue': np.random.rand(100) * 1000,
            'marketing_spend': np.random.rand(100) * 500,
            'playtime_hours': np.random.rand(100) * 100,
            'user_score': np.random.rand(100) * 10,
            'launch_price_decay': np.random.rand(100) * 0.5
        }
        return pd.DataFrame(data)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
