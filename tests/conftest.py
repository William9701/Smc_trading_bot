# tests/conftest.py - Test Configuration

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='15min')
    
    # Generate realistic OHLCV data
    np.random.seed(42)
    base_price = 1.1000
    
    data = []
    current_price = base_price
    
    for i, date in enumerate(dates):
        # Generate realistic OHLC with proper relationships
        open_price = current_price + np.random.normal(0, 0.0005)
        
        # High and low around open
        high = open_price + abs(np.random.normal(0.0010, 0.0005))
        low = open_price - abs(np.random.normal(0.0010, 0.0005))
        
        # Close between high and low
        close = low + (high - low) * np.random.uniform(0.1, 0.9)
        
        # Ensure OHLC relationships are valid
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        volume = np.random.randint(100, 1000)
        spread = np.random.uniform(1, 5)
        
        data.append({
            'open': round(open_price, 5),
            'high': round(high, 5),
            'low': round(low, 5),
            'close': round(close, 5),
            'volume': volume,
            'spread': spread,
            'real_volume': volume * np.random.uniform(0.8, 1.2)
        })
        
        current_price = close
    
    df = pd.DataFrame(data, index=dates)
    return df

@pytest.fixture
def invalid_ohlcv_data():
    """Create invalid OHLCV data for testing validation"""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='15min')
    
    data = []
    for date in dates:
        data.append({
            'open': 1.1000,
            'high': 1.0900,  # Invalid: high < open
            'low': 1.1100,   # Invalid: low > open
            'close': 1.1050,
            'volume': -10,   # Invalid: negative volume
            'spread': 2.0,
            'real_volume': 100
        })
    
    df = pd.DataFrame(data, index=dates)
    return df

@pytest.fixture
def mock_mt5_rates():
    """Mock MT5 rates data"""
    return np.array([
        (1704067200, 1.1000, 1.1020, 1.0980, 1.1010, 500, 2, 600),
        (1704067800, 1.1010, 1.1030, 1.0990, 1.1020, 450, 2, 550),
        (1704068400, 1.1020, 1.1040, 1.1000, 1.1015, 380, 2, 420)
    ], dtype=[
        ('time', '<u4'), ('open', '<f8'), ('high', '<f8'), ('low', '<f8'),
        ('close', '<f8'), ('tick_volume', '<u8'), ('spread', '<i4'), ('real_volume', '<u8')
    ])

