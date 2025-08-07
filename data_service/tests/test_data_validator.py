# data_service/tests/test_data_validator.py

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data_service.data_validator import DataValidator, ValidationResult

class TestDataValidator:
    """Test suite for Data Validator"""
    
    def test_valid_data_validation(self, sample_ohlcv_data):
        """Test validation of valid data"""
        validator = DataValidator()
        result = validator.validate_data(sample_ohlcv_data, "EURUSD", "M15")
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert result.quality_score >= 0.8
        assert len(result.errors) == 0
    
    def test_invalid_data_validation(self, invalid_ohlcv_data):
        """Test validation of invalid data"""
        validator = DataValidator()
        result = validator.validate_data(invalid_ohlcv_data, "EURUSD", "M15")
        
        assert result.is_valid is False
        assert result.quality_score < 0.5
        assert len(result.errors) > 0
    
    def test_empty_dataframe_validation(self):
        """Test validation of empty DataFrame"""
        validator = DataValidator()
        empty_df = pd.DataFrame()
        
        result = validator.validate_data(empty_df, "EURUSD", "M15")
        
        assert result.is_valid is False
        assert result.quality_score == 0.0
        assert "DataFrame is empty" in result.errors
    
    def test_ohlc_integrity_validation(self):
        """Test OHLC integrity validation"""
        validator = DataValidator()
        
        # Create data with OHLC violations
        dates = pd.date_range(start='2024-01-01', periods=3, freq='15min')
        bad_data = pd.DataFrame({
            'open': [1.1000, 1.1010, 1.1020],
            'high': [1.0900, 1.1030, 1.1040],  # First high < open
            'low': [1.0980, 1.0990, 1.1000],
            'close': [1.1010, 1.1020, 1.1015],
            'volume': [100, 200, 150]
        }, index=dates)
        
        result = validator._validate_ohlc_integrity(bad_data, "EURUSD", "M15")
        
        assert len(result['errors']) > 0 or len(result['warnings']) > 0
        assert result['metrics']['invalid_ohlc_count'] > 0
    
    def test_volume_validation(self):
        """Test volume data validation"""
        validator = DataValidator()
        
        dates = pd.date_range(start='2024-01-01', periods=3, freq='15min')
        volume_data = pd.DataFrame({
            'open': [1.1000, 1.1010, 1.1020],
            'high': [1.1020, 1.1030, 1.1040],
            'low': [1.0980, 1.0990, 1.1000],
            'close': [1.1010, 1.1020, 1.1015],
            'volume': [-100, 0, 150]  # Negative and zero volume
        }, index=dates)
        
        result = validator._validate_volume_data(volume_data, "EURUSD", "M15")
        
        assert result['metrics']['negative_volume_count'] == 1
        assert result['metrics']['zero_volume_count'] == 1
    
    def test_time_consistency_validation(self):
        """Test time series consistency validation"""
        validator = DataValidator()
        
        # Create data with duplicate timestamps
        dates = pd.date_range(start='2024-01-01', periods=3, freq='15min')
        dates = dates.insert(1, dates[1])  # Duplicate timestamp
        
        time_data = pd.DataFrame({
            'open': [1.1000, 1.1010, 1.1010, 1.1020],
            'high': [1.1020, 1.1030, 1.1030, 1.1040],
            'low': [1.0980, 1.0990, 1.0990, 1.1000],
            'close': [1.1010, 1.1020, 1.1020, 1.1015],
            'volume': [100, 200, 200, 150]
        }, index=dates)
        
        result = validator._validate_time_consistency(time_data, "EURUSD", "M15")
        
        assert result['metrics']['duplicate_times'] == 1
    
    def test_completeness_validation(self):
        """Test data completeness validation"""
        validator = DataValidator()
        
        # Create data with missing values
        dates = pd.date_range(start='2024-01-01', periods=3, freq='15min')
        incomplete_data = pd.DataFrame({
            'open': [1.1000, np.nan, 1.1020],
            'high': [1.1020, 1.1030, 1.1040],
            'low': [1.0980, 1.0990, np.nan],
            'close': [1.1010, 1.1020, 1.1015],
            'volume': [100, 200, 150]
        }, index=dates)
        
        result = validator._validate_completeness(incomplete_data, "EURUSD", "M15")
        
        assert result['metrics']['missing_values'] == 2


    