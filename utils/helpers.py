# utils/helpers.py

"""
SMC Trading Bot Helper Functions
Common utility functions used throughout the system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta, time
import re
from loguru import logger
import pytz

from .constants import MT5_TIMEFRAMES, TIMEFRAME_NAMES, DataQuality

def validate_symbol(symbol: str) -> bool:
    """
    Validate trading symbol format
    """
    if not symbol or not isinstance(symbol, str):
        return False
    
    # Remove any whitespace
    symbol = symbol.strip().upper()
    
    # Basic forex pair validation (6-8 characters)
    if len(symbol) < 4 or len(symbol) > 12:
        return False
    
    # Check for valid characters (letters and numbers only)
    if not re.match(r'^[A-Z0-9]+$', symbol):
        return False
    
    return True

def format_symbol(symbol: str) -> str:
    """
    Format symbol to standard format
    """
    if not symbol:
        return ""
    
    return symbol.strip().upper()

def validate_timeframe(timeframe: str) -> bool:
    """
    Validate if timeframe is supported
    """
    return timeframe in MT5_TIMEFRAMES

def get_mt5_timeframe(timeframe: str):
    """
    Get MT5 timeframe constant from string
    """
    return MT5_TIMEFRAMES.get(timeframe)

def get_timeframe_name(mt5_timeframe) -> str:
    """
    Get timeframe name from MT5 constant
    """
    return TIMEFRAME_NAMES.get(mt5_timeframe, "UNKNOWN")

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate DataFrame structure and content for OHLCV data
    """
    if df is None or df.empty:
        return False
    
    # Check required columns
    required_columns = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_columns):
        return False
    
    # Check for valid OHLC relationships
    invalid_ohlc = (
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    )
    
    if invalid_ohlc.any():
        logger.warning(f"Found {invalid_ohlc.sum()} invalid OHLC relationships")
        return False
    
    # Check for null values in critical columns
    if df[required_columns].isnull().any().any():
        logger.warning("Found null values in OHLC data")
        return False
    
    return True

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR)
    """
    if df.empty or len(df) < period:
        return pd.Series(dtype=float)
    
    try:
        # Calculate True Range components
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close'].shift(1))
        low_close_prev = abs(df['low'] - df['close'].shift(1))
        
        # True Range is the maximum of the three
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # Calculate ATR as moving average of True Range
        atr = true_range.rolling(window=period, min_periods=1).mean()
        
        return atr
        
    except Exception as e:
        logger.error(f"Error calculating ATR: {e}")
        return pd.Series(dtype=float)

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI)
    """
    if df.empty or len(df) < period:
        return pd.Series(dtype=float)
    
    try:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return pd.Series(dtype=float)

def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average
    """
    try:
        return series.rolling(window=period, min_periods=1).mean()
    except Exception as e:
        logger.error(f"Error calculating SMA: {e}")
        return pd.Series(dtype=float)

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average
    """
    try:
        return series.ewm(span=period, adjust=False).mean()
    except Exception as e:
        logger.error(f"Error calculating EMA: {e}")
        return pd.Series(dtype=float)

def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
    """
    Calculate Bollinger Bands
    """
    try:
        sma = calculate_sma(df['close'], period)
        std = df['close'].rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return {
            'middle': sma,
            'upper': upper_band,
            'lower': lower_band
        }
        
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {e}")
        return {'middle': pd.Series(dtype=float), 'upper': pd.Series(dtype=float), 'lower': pd.Series(dtype=float)}

def find_pivots(series: pd.Series, lookback: int = 5) -> Dict[str, List[int]]:
    """
    Find pivot highs and lows in a price series
    """
    try:
        highs = []
        lows = []
        
        for i in range(lookback, len(series) - lookback):
            # Check for pivot high
            is_high = True
            for j in range(1, lookback + 1):
                if series.iloc[i] <= series.iloc[i - j] or series.iloc[i] <= series.iloc[i + j]:
                    is_high = False
                    break
            
            if is_high:
                highs.append(i)
            
            # Check for pivot low
            is_low = True
            for j in range(1, lookback + 1):
                if series.iloc[i] >= series.iloc[i - j] or series.iloc[i] >= series.iloc[i + j]:
                    is_low = False
                    break
            
            if is_low:
                lows.append(i)
        
        return {'highs': highs, 'lows': lows}
        
    except Exception as e:
        logger.error(f"Error finding pivots: {e}")
        return {'highs': [], 'lows': []}

def calculate_price_change_percentage(start_price: float, end_price: float) -> float:
    """
    Calculate percentage change between two prices
    """
    if start_price == 0:
        return 0.0
    
    return ((end_price - start_price) / start_price) * 100

def normalize_price_to_pips(price_change: float, symbol: str) -> float:
    """
    Convert price change to pips for forex pairs
    """
    # Common pip values for major pairs
    pip_values = {
        'EURUSD': 10000, 'GBPUSD': 10000, 'AUDUSD': 10000, 'NZDUSD': 10000,
        'USDJPY': 100, 'USDCHF': 10000, 'USDCAD': 10000,
        'EURJPY': 100, 'GBPJPY': 100, 'AUDJPY': 100,
        'GOLD': 10, 'XAUUSD': 10, 'SILVER': 1000, 'XAGUSD': 1000
    }
    
    # Default to 10000 for most forex pairs
    multiplier = pip_values.get(symbol.upper(), 10000)
    
    return price_change * multiplier

def calculate_fibonacci_levels(high: float, low: float) -> Dict[str, float]:
    """
    Calculate Fibonacci retracement levels
    """
    if high <= low:
        return {}
    
    diff = high - low
    
    levels = {
        '0.0': high,
        '23.6': high - (diff * 0.236),
        '38.2': high - (diff * 0.382),
        '50.0': high - (diff * 0.500),
        '61.8': high - (diff * 0.618),
        '78.6': high - (diff * 0.786),
        '100.0': low
    }
    
    return levels

def is_within_range(value: float, target: float, tolerance: float) -> bool:
    """
    Check if value is within tolerance range of target
    """
    return abs(value - target) <= tolerance

def round_to_decimals(value: float, decimals: int) -> float:
    """
    Round value to specified decimal places
    """
    return round(value, decimals)

def format_timestamp(timestamp: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format timestamp to string
    """
    try:
        return timestamp.strftime(format_str)
    except Exception:
        return str(timestamp)

def parse_timestamp(timestamp_str: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> Optional[datetime]:
    """
    Parse timestamp string to datetime object
    """
    try:
        return datetime.strptime(timestamp_str, format_str)
    except Exception as e:
        logger.error(f"Error parsing timestamp {timestamp_str}: {e}")
        return None

def time_difference_minutes(start: datetime, end: datetime) -> float:
    """
    Calculate time difference in minutes
    """
    try:
        return (end - start).total_seconds() / 60.0
    except Exception:
        return 0.0

def filter_outliers(series: pd.Series, method: str = 'iqr', factor: float = 1.5) -> pd.Series:
    """
    Filter outliers from a pandas Series
    """
    try:
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            return series[(series >= lower_bound) & (series <= upper_bound)]
        
        elif method == 'std':
            mean = series.mean()
            std = series.std()
            
            lower_bound = mean - factor * std
            upper_bound = mean + factor * std
            
            return series[(series >= lower_bound) & (series <= upper_bound)]
        
        else:
            return series
            
    except Exception as e:
        logger.error(f"Error filtering outliers: {e}")
        return series

def detect_gaps(df: pd.DataFrame, gap_threshold: float = 0.001) -> List[Dict]:
    """
    Detect price gaps in OHLC data
    """
    gaps = []
    
    try:
        for i in range(1, len(df)):
            prev_close = df['close'].iloc[i-1]
            current_open = df['open'].iloc[i]
            
            # Calculate gap size as percentage
            gap_size = abs(current_open - prev_close) / prev_close
            
            if gap_size > gap_threshold:
                gap_type = 'gap_up' if current_open > prev_close else 'gap_down'
                
                gaps.append({
                    'index': i,
                    'timestamp': df.index[i],
                    'type': gap_type,
                    'prev_close': prev_close,
                    'current_open': current_open,
                    'gap_size': gap_size,
                    'gap_points': abs(current_open - prev_close)
                })
        
        return gaps
        
    except Exception as e:
        logger.error(f"Error detecting gaps: {e}")
        return []

def calculate_correlation(series1: pd.Series, series2: pd.Series) -> float:
    """
    Calculate correlation between two price series
    """
    try:
        return series1.corr(series2)
    except Exception as e:
        logger.error(f"Error calculating correlation: {e}")
        return 0.0

def find_support_resistance(df: pd.DataFrame, window: int = 20, min_touches: int = 2) -> Dict[str, List[float]]:
    """
    Find support and resistance levels
    """
    try:
        highs = df['high'].rolling(window=window, center=True).max()
        lows = df['low'].rolling(window=window, center=True).min()
        
        # Find levels that were touched multiple times
        resistance_levels = []
        support_levels = []
        
        tolerance = df['close'].std() * 0.01  # 1% of price volatility
        
        for i in range(len(df)):
            if pd.notna(highs.iloc[i]) and highs.iloc[i] == df['high'].iloc[i]:
                level = df['high'].iloc[i]
                touches = sum(1 for price in df['high'] if is_within_range(price, level, tolerance))
                
                if touches >= min_touches:
                    resistance_levels.append(level)
            
            if pd.notna(lows.iloc[i]) and lows.iloc[i] == df['low'].iloc[i]:
                level = df['low'].iloc[i]
                touches = sum(1 for price in df['low'] if is_within_range(price, level, tolerance))
                
                if touches >= min_touches:
                    support_levels.append(level)
        
        return {
            'resistance': list(set(resistance_levels)),  # Remove duplicates
            'support': list(set(support_levels))
        }
        
    except Exception as e:
        logger.error(f"Error finding support/resistance: {e}")
        return {'resistance': [], 'support': []}

def calculate_volatility(series: pd.Series, method: str = 'std', period: int = 20) -> float:
    """
    Calculate volatility using different methods
    """
    try:
        if method == 'std':
            return series.rolling(window=period).std().iloc[-1]
        elif method == 'atr':
            # Approximate ATR using close prices
            true_range = abs(series.diff())
            return true_range.rolling(window=period).mean().iloc[-1]
        else:
            return series.rolling(window=period).std().iloc[-1]
            
    except Exception as e:
        logger.error(f"Error calculating volatility: {e}")
        return 0.0

def determine_market_session(timestamp: datetime) -> str:
    """
    Determine which trading session is active
    """
    try:
        hour = timestamp.hour
        
        # Approximate session times (UTC)
        if 21 <= hour or hour < 6:
            return "SYDNEY"
        elif 23 <= hour or hour < 8:
            return "TOKYO" 
        elif 7 <= hour < 16:
            return "LONDON"
        elif 12 <= hour < 21:
            return "NEW_YORK"
        else:
            return "UNKNOWN"
            
    except Exception:
        return "UNKNOWN"

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division that handles division by zero
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except Exception:
        return default

def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp value between min and max
    """
    return max(min_value, min(value, max_value))

def interpolate_missing_values(series: pd.Series, method: str = 'linear') -> pd.Series:
    """
    Interpolate missing values in a pandas Series
    """
    try:
        return series.interpolate(method=method)
    except Exception as e:
        logger.error(f"Error interpolating missing values: {e}")
        return series

def detect_price_patterns(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """
    Basic price pattern detection
    """
    patterns = {
        'double_tops': [],
        'double_bottoms': [],
        'triangles': [],
        'channels': []
    }
    
    try:
        # Simple pattern detection logic
        pivots = find_pivots(df['close'], lookback=5)
        
        # Double top detection (simplified)
        highs = pivots['highs']
        for i in range(len(highs) - 1):
            idx1, idx2 = highs[i], highs[i+1]
            price1, price2 = df['close'].iloc[idx1], df['close'].iloc[idx2]
            
            if is_within_range(price1, price2, price1 * 0.02):  # Within 2%
                patterns['double_tops'].append({
                    'start_idx': idx1,
                    'end_idx': idx2,
                    'level': (price1 + price2) / 2,
                    'confidence': 0.7
                })
        
        # Double bottom detection (simplified)
        lows = pivots['lows']
        for i in range(len(lows) - 1):
            idx1, idx2 = lows[i], lows[i+1]
            price1, price2 = df['close'].iloc[idx1], df['close'].iloc[idx2]
            
            if is_within_range(price1, price2, price1 * 0.02):  # Within 2%
                patterns['double_bottoms'].append({
                    'start_idx': idx1,
                    'end_idx': idx2,
                    'level': (price1 + price2) / 2,
                    'confidence': 0.7
                })
        
        return patterns
        
    except Exception as e:
        logger.error(f"Error detecting price patterns: {e}")
        return patterns

def calculate_swing_failure(df: pd.DataFrame, swing_high_idx: int, swing_low_idx: int) -> Dict:
    """
    Calculate swing failure characteristics
    """
    try:
        if swing_high_idx >= len(df) or swing_low_idx >= len(df):
            return {}
        
        swing_high = df['high'].iloc[swing_high_idx]
        swing_low = df['low'].iloc[swing_low_idx]
        
        # Look for failure to make new high/low
        recent_data = df.iloc[max(swing_high_idx, swing_low_idx):]
        
        if swing_high_idx > swing_low_idx:
            # Look for failure to break above swing high
            max_attempt = recent_data['high'].max()
            failure = max_attempt < swing_high
        else:
            # Look for failure to break below swing low
            min_attempt = recent_data['low'].min()
            failure = min_attempt > swing_low
        
        return {
            'failure_detected': failure,
            'swing_high': swing_high,
            'swing_low': swing_low,
            'attempt_level': max_attempt if swing_high_idx > swing_low_idx else min_attempt
        }
        
    except Exception as e:
        logger.error(f"Error calculating swing failure: {e}")
        return {}

def assess_data_completeness(df: pd.DataFrame, expected_frequency: str = None) -> Dict:
    """
    Assess data completeness and quality
    """
    try:
        assessment = {
            'total_candles': len(df),
            'date_range': {
                'start': df.index[0] if not df.empty else None,
                'end': df.index[-1] if not df.empty else None
            },
            'missing_values': df.isnull().sum().to_dict(),
            'duplicated_timestamps': df.index.duplicated().sum(),
            'quality_score': 1.0
        }
        
        # Calculate quality score
        total_possible_missing = len(df) * len(df.columns)
        total_missing = df.isnull().sum().sum()
        
        if total_possible_missing > 0:
            missing_ratio = total_missing / total_possible_missing
            assessment['quality_score'] = max(0.0, 1.0 - missing_ratio)
        
        # Check for data gaps if frequency is provided
        if expected_frequency and len(df) > 1:
            expected_index = pd.date_range(
                start=df.index[0],
                end=df.index[-1],
                freq=expected_frequency
            )
            
            missing_timestamps = len(expected_index) - len(df)
            assessment['missing_timestamps'] = missing_timestamps
            
            if len(expected_index) > 0:
                timestamp_completeness = len(df) / len(expected_index)
                assessment['quality_score'] *= timestamp_completeness
        
        # Classify quality
        if assessment['quality_score'] >= 0.95:
            assessment['quality_level'] = DataQuality.EXCELLENT
        elif assessment['quality_score'] >= 0.85:
            assessment['quality_level'] = DataQuality.GOOD
        elif assessment['quality_score'] >= 0.70:
            assessment['quality_level'] = DataQuality.FAIR
        elif assessment['quality_score'] >= 0.50:
            assessment['quality_level'] = DataQuality.POOR
        else:
            assessment['quality_level'] = DataQuality.UNUSABLE
        
        return assessment
        
    except Exception as e:
        logger.error(f"Error assessing data completeness: {e}")
        return {
            'total_candles': 0,
            'quality_score': 0.0,
            'quality_level': DataQuality.UNUSABLE,
            'error': str(e)
        }

def calculate_performance_metrics(start_time: datetime, end_time: datetime, operations_count: int = 1) -> Dict:
    """
    Calculate performance metrics for operations
    """
    try:
        duration = (end_time - start_time).total_seconds()
        duration_ms = duration * 1000
        
        metrics = {
            'duration_seconds': duration,
            'duration_ms': duration_ms,
            'operations_count': operations_count,
            'operations_per_second': operations_count / duration if duration > 0 else 0,
            'ms_per_operation': duration_ms / operations_count if operations_count > 0 else 0
        }
        
        # Performance classification
        if duration_ms < 100:
            metrics['performance_level'] = 'OPTIMAL'
        elif duration_ms < 500:
            metrics['performance_level'] = 'GOOD'
        elif duration_ms < 2000:
            metrics['performance_level'] = 'ACCEPTABLE'
        elif duration_ms < 5000:
            metrics['performance_level'] = 'SLOW'
        else:
            metrics['performance_level'] = 'UNACCEPTABLE'
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {e}")
        return {
            'duration_seconds': 0,
            'duration_ms': 0,
            'performance_level': 'ERROR'
        }

def format_number(number: float, decimals: int = 2, use_thousands_separator: bool = True) -> str:
    """
    Format number for display
    """
    try:
        if use_thousands_separator:
            return f"{number:,.{decimals}f}"
        else:
            return f"{number:.{decimals}f}"
    except Exception:
        return str(number)

def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format value as percentage
    """
    try:
        return f"{value:.{decimals}f}%"
    except Exception:
        return f"{value}%"

def create_summary_stats(series: pd.Series) -> Dict:
    """
    Create summary statistics for a pandas Series
    """
    try:
        stats = {
            'count': len(series),
            'mean': series.mean(),
            'median': series.median(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'q1': series.quantile(0.25),
            'q3': series.quantile(0.75),
            'skewness': series.skew(),
            'kurtosis': series.kurtosis()
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error creating summary stats: {e}")
        return {}

def validate_price_level(price: float, df: pd.DataFrame, tolerance: float = 0.001) -> bool:
    """
    Validate if a price level is reasonable given historical data
    """
    try:
        if df.empty:
            return False
        
        price_range = df['high'].max() - df['low'].min()
        mean_price = df['close'].mean()
        
        # Check if price is within reasonable range
        if price < df['low'].min() * (1 - tolerance) or price > df['high'].max() * (1 + tolerance):
            return False
        
        # Check if price is not too far from mean
        if abs(price - mean_price) > price_range * 2:
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating price level: {e}")
        return False

def merge_overlapping_zones(zones: List[Dict], overlap_threshold: float = 0.1) -> List[Dict]:
    """
    Merge overlapping price zones
    """
    try:
        if len(zones) <= 1:
            return zones
        
        # Sort zones by start price
        sorted_zones = sorted(zones, key=lambda x: x.get('start', x.get('low', 0)))
        merged = []
        
        current_zone = sorted_zones[0].copy()
        
        for zone in sorted_zones[1:]:
            zone_start = zone.get('start', zone.get('low', 0))
            zone_end = zone.get('end', zone.get('high', 0))
            
            current_end = current_zone.get('end', current_zone.get('high', 0))
            
            # Check for overlap
            overlap = max(0, min(current_end, zone_end) - max(current_zone.get('start', current_zone.get('low', 0)), zone_start))
            zone_size = zone_end - zone_start
            
            if zone_size > 0 and overlap / zone_size > overlap_threshold:
                # Merge zones
                current_zone['start'] = current_zone.get('start', current_zone.get('low', 0))
                current_zone['end'] = max(current_end, zone_end)
                current_zone['low'] = min(current_zone.get('low', current_zone['start']), zone.get('low', zone_start))
                current_zone['high'] = max(current_zone.get('high', current_zone['end']), zone.get('high', zone_end))
                
                # Merge other properties if available
                if 'strength' in current_zone and 'strength' in zone:
                    current_zone['strength'] = max(current_zone['strength'], zone['strength'])
            else:
                # No overlap, add current zone and start new one
                merged.append(current_zone)
                current_zone = zone.copy()
        
        # Add the last zone
        merged.append(current_zone)
        
        return merged
        
    except Exception as e:
        logger.error(f"Error merging overlapping zones: {e}")
        return zones

def calculate_zone_strength(zone: Dict, df: pd.DataFrame) -> float:
    """
    Calculate the strength of a price zone based on historical reactions
    """
    try:
        zone_start = zone.get('start', zone.get('low', 0))
        zone_end = zone.get('end', zone.get('high', 0))
        
        if zone_start >= zone_end:
            return 0.0
        
        # Count touches and reactions
        touches = 0
        reactions = 0
        
        for i in range(len(df)):
            candle_low = df['low'].iloc[i]
            candle_high = df['high'].iloc[i]
            
            # Check if candle touched the zone
            if (candle_low <= zone_end and candle_high >= zone_start):
                touches += 1
                
                # Check for reaction (reversal within next few candles)
                if i < len(df) - 3:
                    future_prices = df['close'].iloc[i+1:i+4]
                    current_close = df['close'].iloc[i]
                    
                    # Simple reaction detection
                    if zone_start < zone_end:  # Support zone
                        if any(future_prices > current_close * 1.001):  # 0.1% bounce
                            reactions += 1
                    else:  # Resistance zone
                        if any(future_prices < current_close * 0.999):  # 0.1% drop
                            reactions += 1
        
        # Calculate strength as reaction rate
        if touches > 0:
            strength = reactions / touches
        else:
            strength = 0.0
        
        # Boost strength for multiple touches
        if touches >= 3:
            strength *= 1.2
        elif touches >= 2:
            strength *= 1.1
        
        return min(1.0, strength)
        
    except Exception as e:
        logger.error(f"Error calculating zone strength: {e}")
        return 0.0

def export_data_to_csv(data: Union[pd.DataFrame, Dict], filename: str, include_timestamp: bool = True) -> bool:
    """
    Export data to CSV file
    """
    try:
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{timestamp}.csv"
        
        if isinstance(data, pd.DataFrame):
            data.to_csv(filename)
        elif isinstance(data, dict):
            pd.DataFrame([data]).to_csv(filename, index=False)
        else:
            logger.error("Unsupported data type for CSV export")
            return False
        
        logger.info(f"Data exported to {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting data to CSV: {e}")
        return False

def import_data_from_csv(filename: str, parse_dates: bool = True, index_col: Optional[str] = None) -> pd.DataFrame:
    """
    Import data from CSV file
    """
    try:
        df = pd.read_csv(
            filename,
            parse_dates=parse_dates,
            index_col=index_col
        )
        
        logger.info(f"Data imported from {filename}: {len(df)} rows")
        return df
        
    except Exception as e:
        logger.error(f"Error importing data from CSV: {e}")
        return pd.DataFrame()

# Decorator functions for common operations
def retry_on_failure(max_attempts: int = 3, delay: float = 1.0):
    """
    Decorator to retry function on failure
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying in {delay}s")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_attempts} attempts failed: {e}")
            
            raise last_exception
        
        return wrapper
    return decorator

def measure_performance(func):
    """
    Decorator to measure function performance
    """
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        logger.debug(f"{func.__name__} executed in {duration:.4f} seconds")
        
        return result
    
    return wrapper

def find_swing_highs(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Find swing high points in price data"""
    highs = df['high']
    swing_highs = pd.Series(False, index=df.index)
    
    for i in range(window, len(highs) - window):
        if all(highs.iloc[i] > highs.iloc[i-j] for j in range(1, window+1)) and \
           all(highs.iloc[i] > highs.iloc[i+j] for j in range(1, window+1)):
            swing_highs.iloc[i] = True
    
    return swing_highs

def find_swing_lows(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Find swing low points in price data"""
    lows = df['low']
    swing_lows = pd.Series(False, index=df.index)
    
    for i in range(window, len(lows) - window):
        if all(lows.iloc[i] < lows.iloc[i-j] for j in range(1, window+1)) and \
           all(lows.iloc[i] < lows.iloc[i+j] for j in range(1, window+1)):
            swing_lows.iloc[i] = True
    
    return swing_lows

def calculate_pips(price1: float, price2: float, symbol: str) -> float:
    """Calculate pip difference between two prices"""
    from config.mt5_config import MT5_SYMBOL_CONFIG
    
    if symbol not in MT5_SYMBOL_CONFIG:
        raise ValueError(f"Symbol {symbol} not configured")
    
    digits = MT5_SYMBOL_CONFIG[symbol]["digits"]
    pip_value = 0.0001 if digits == 5 else 0.01 if digits == 3 else 0.0001
    
    return abs(price1 - price2) / pip_value

def is_market_open() -> bool:
    """Check if forex market is open"""
    now = datetime.now(pytz.UTC)
    weekday = now.weekday()
    
    # Forex market is closed on weekends
    if weekday == 5:  # Saturday
        return now.hour >= 22  # Opens Sunday 22:00 UTC
    elif weekday == 6:  # Sunday
        return True
    elif weekday == 4:  # Friday
        return now.hour < 22  # Closes Friday 22:00 UTC
    else:
        return True