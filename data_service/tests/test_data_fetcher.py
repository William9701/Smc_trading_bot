# data_service/tests/test_data_fetcher.py

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from data_service.data_fetcher import EnhancedMT5DataFetcher, DataRequest
from utils.exceptions import DataFetchError, MT5ConnectionError

class TestEnhancedMT5DataFetcher:
    """Test suite for Enhanced MT5 Data Fetcher"""
    
    def test_data_request_validation(self):
        """Test DataRequest validation"""
        # Valid request with num_candles
        request1 = DataRequest(symbol="EURUSD", timeframe="M15", num_candles=100)
        assert request1.symbol == "EURUSD"
        assert request1.num_candles == 100
        
        # Valid request with date range
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 2)
        request2 = DataRequest(symbol="GBPUSD", timeframe="H1", start_date=start, end_date=end)
        assert request2.start_date == start
        assert request2.end_date == end
        
        # Invalid request - neither num_candles nor date range
        with pytest.raises(ValueError):
            DataRequest(symbol="USDJPY", timeframe="H4")
    
    @patch('data_service.data_fetcher.mt5')
    def test_initialization_success(self, mock_mt5):
        """Test successful MT5 initialization"""
        # Mock successful initialization
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = Mock(login=12345, server="TestServer", balance=10000.0)
        
        fetcher = EnhancedMT5DataFetcher()
        result = fetcher.initialize_mt5()
        
        assert result is True
        assert fetcher.mt5_initialized is True
        assert fetcher.connection_health is True
        mock_mt5.initialize.assert_called_once()
    
    @patch('data_service.data_fetcher.mt5')
    def test_initialization_failure(self, mock_mt5):
        """Test failed MT5 initialization"""
        # Mock failed initialization
        mock_mt5.initialize.return_value = False
        mock_mt5.last_error.return_value = "Login failed"
        
        fetcher = EnhancedMT5DataFetcher()
        
        with pytest.raises(MT5ConnectionError):
            fetcher.initialize_mt5()
        
        assert fetcher.mt5_initialized is False
        assert fetcher.connection_health is False
    
    @patch('data_service.data_fetcher.mt5')
    def test_symbol_selection(self, mock_mt5):
        """Test symbol selection"""
        fetcher = EnhancedMT5DataFetcher()
        fetcher.mt5_initialized = True
        
        # Mock successful symbol selection
        mock_mt5.symbol_select.return_value = True
        mock_symbol_info = Mock()
        mock_symbol_info.visible = True
        mock_mt5.symbol_info.return_value = mock_symbol_info
        
        result = fetcher.ensure_symbol_selected("EURUSD")
        assert result is True
        
        # Mock failed symbol selection
        mock_mt5.symbol_select.return_value = False
        result = fetcher.ensure_symbol_selected("INVALID")
        assert result is False
    
    @patch('data_service.data_fetcher.mt5')
    def test_data_fetching_success(self, mock_mt5, mock_mt5_rates):
        """Test successful data fetching"""
        fetcher = EnhancedMT5DataFetcher()
        fetcher.mt5_initialized = True
        fetcher.connection_health = True
        
        # Mock successful data fetch
        mock_mt5.symbol_select.return_value = True
        mock_symbol_info = Mock()
        mock_symbol_info.visible = True
        mock_mt5.symbol_info.return_value = mock_symbol_info
        mock_mt5.copy_rates_from_pos.return_value = mock_mt5_rates
        mock_mt5.account_info.return_value = Mock(login=12345)
        mock_mt5.terminal_info.return_value = Mock(connected=True)
        
        request = DataRequest(symbol="EURUSD", timeframe="M15", num_candles=3)
        
        with patch.object(fetcher, 'check_connection_health', return_value=True):
            df = fetcher.fetch_data_with_retry(request)
        
        assert not df.empty
        assert len(df) == 3
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        assert isinstance(df.index, pd.DatetimeIndex)
    
    @patch('data_service.data_fetcher.mt5')
    def test_data_quality_assessment(self, mock_mt5, sample_ohlcv_data):
        """Test data quality assessment"""
        fetcher = EnhancedMT5DataFetcher()
        
        quality = fetcher.assess_data_quality(sample_ohlcv_data, "EURUSD", "M15")
        
        assert quality.total_candles == len(sample_ohlcv_data)
        assert quality.quality_score >= 0.8  # Should be high quality
        assert quality.invalid_candles == 0   # Sample data is valid
    
    def test_data_quality_with_invalid_data(self, invalid_ohlcv_data):
        """Test data quality assessment with invalid data"""
        fetcher = EnhancedMT5DataFetcher()
        
        quality = fetcher.assess_data_quality(invalid_ohlcv_data, "EURUSD", "M15")
        
        assert quality.total_candles == len(invalid_ohlcv_data)
        assert quality.quality_score < 0.5  # Should be low quality
        assert quality.invalid_candles > 0   # Contains invalid data
    
    @patch('data_service.data_fetcher.mt5')
    def test_multi_timeframe_data_fetching(self, mock_mt5, mock_mt5_rates):
        """Test multi-timeframe data fetching"""
        fetcher = EnhancedMT5DataFetcher()
        fetcher.mt5_initialized = True
        fetcher.connection_health = True
        
        # Mock setup
        mock_mt5.symbol_select.return_value = True
        mock_symbol_info = Mock()
        mock_symbol_info.visible = True
        mock_mt5.symbol_info.return_value = mock_symbol_info
        mock_mt5.copy_rates_from_pos.return_value = mock_mt5_rates
        mock_mt5.account_info.return_value = Mock(login=12345)
        mock_mt5.terminal_info.return_value = Mock(connected=True)
        
        with patch.object(fetcher, 'check_connection_health', return_value=True):
            data = fetcher.get_multi_timeframe_data(
                symbol="EURUSD",
                primary_timeframe="M15", 
                higher_timeframes=["H1", "H4"],
                num_candles=100
            )
        
        assert "M15" in data
        assert "H1" in data
        assert "H4" in data
        assert all(not df.empty for df in data.values())
    
    def test_caching_functionality(self, sample_ohlcv_data):
        """Test data caching functionality"""
        fetcher = EnhancedMT5DataFetcher()
        
        # Test cache storage
        cache_key = "EURUSD_M15"
        fetcher.data_cache[cache_key] = sample_ohlcv_data
        
        cached_data = fetcher.get_cached_data("EURUSD", "M15")
        assert cached_data is not None
        assert len(cached_data) == len(sample_ohlcv_data)
        
        # Test cache miss
        missing_data = fetcher.get_cached_data("GBPUSD", "H1")
        assert missing_data is None
    
    @patch('data_service.data_fetcher.mt5')
    def test_context_manager(self, mock_mt5):
        """Test context manager functionality"""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = Mock(login=12345, server="TestServer", balance=10000.0)
        
        with EnhancedMT5DataFetcher() as fetcher:
            assert fetcher.mt5_initialized is True
        
        mock_mt5.shutdown.assert_called_once()

