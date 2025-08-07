# data_service/tests/test_data_preprocessor.py

import pytest
import pandas as pd
import numpy as np

from data_service.data_preprocessor import DataPreprocessor

class TestDataPreprocessor:
    """Test suite for Data Preprocessor"""
    
    def test_basic_preprocessing(self, sample_ohlcv_data):
        """Test basic data preprocessing"""
        preprocessor = DataPreprocessor()
        
        processed_df = preprocessor.preprocess_data(sample_ohlcv_data, "EURUSD")
        
        assert not processed_df.empty
        assert len(processed_df) <= len(sample_ohlcv_data)  # May remove invalid data
        assert 'typical_price' in processed_df.columns
        assert 'session' in processed_df.columns
    
    def test_ohlc_cleaning(self):
        """Test OHLC data cleaning"""
        preprocessor = DataPreprocessor()
        
        # Create data with invalid OHLC relationships
        dates = pd.date_range(start='2024-01-01', periods=3, freq='15min')
        dirty_data = pd.DataFrame({
            'open': [1.1000, 1.1010, 1.1020],
            'high': [1.0900, 1.1030, 1.1040],  # First high < open (invalid)
            'low': [1.0980, 1.0990, 1.1000],
            'close': [1.1010, 1.1020, 1.1015],
            'volume': [100, 200, 150]
        }, index=dates)
        
        cleaned_data = preprocessor._clean_ohlc_data(dirty_data, "EURUSD")
        
        # Should have fewer rows after cleaning
        assert len(cleaned_data) < len(dirty_data)
        
        # Remaining data should be valid
        for idx, row in cleaned_data.iterrows():
            assert row['high'] >= max(row['open'], row['close'])
            assert row['low'] <= min(row['open'], row['close'])
    
    def test_volume_normalization(self):
        """Test volume normalization"""
        preprocessor = DataPreprocessor()
        
        dates = pd.date_range(start='2024-01-01', periods=5, freq='15min')
        volume_data = pd.DataFrame({
            'open': [1.1000] * 5,
            'high': [1.1020] * 5,
            'low': [1.0980] * 5,
            'close': [1.1010] * 5,
            'volume': [100, -50, 0, 200, 150]  # Contains invalid volumes
        }, index=dates)
        
        normalized_data = preprocessor._normalize_volume(volume_data, "EURUSD")
        
        # Should have no negative or zero volumes
        assert (normalized_data['volume'] > 0).all()
    
    def test_technical_indicators_addition(self, sample_ohlcv_data):
        """Test addition of technical indicators"""
        preprocessor = DataPreprocessor()
        
        enhanced_data = preprocessor._add_basic_indicators(sample_ohlcv_data, "EURUSD")
        
        # Check for added indicators
        expected_indicators = [
            'typical_price', 'price_range', 'body_size',
            'upper_wick', 'lower_wick', 'sma_20', 'sma_50'
        ]
        
        for indicator in expected_indicators:
            assert indicator in enhanced_data.columns
    
    def test_trading_session_detection(self, sample_ohlcv_data):
        """Test trading session detection"""
        preprocessor = DataPreprocessor()
        
        session_data = preprocessor._detect_trading_sessions(sample_ohlcv_data, "EURUSD")
        
        assert 'session' in session_data.columns
        assert 'hour_utc' in session_data.columns
        
        # Check that sessions are properly assigned
        valid_sessions = ['ASIAN', 'LONDON', 'NEW_YORK']
        assert session_data['session'].isin(valid_sessions).all()
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrame"""
        preprocessor = DataPreprocessor()
        empty_df = pd.DataFrame()
        
        result = preprocessor.preprocess_data(empty_df, "EURUSD")
        
        assert result.empty
        assert isinstance(result, pd.DataFrame)

# Run the tests with coverage
if __name__ == "__main__":
    pytest.main([
        "data_service/tests/",
        "-v",
        "--cov=data_service",
        "--cov-report=html",
        "--cov-report=term-missing"
    ])