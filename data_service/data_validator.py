# data_service/data_validator.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger

from utils.exceptions import DataFetchError
from utils.constants import MarketStructure
from config.mt5_config import MT5_SYMBOL_CONFIG

@dataclass
class ValidationResult:
    """Data validation result"""
    is_valid: bool
    errors: List[str]
    warnings: List[str] 
    quality_score: float
    metrics: Dict[str, Any]

class DataValidator:
    """Professional data validation service"""
    
    def __init__(self):
        self.validation_rules = {
            'ohlc_integrity': self._validate_ohlc_integrity,
            'volume_validity': self._validate_volume_data,
            'time_consistency': self._validate_time_consistency,
            'data_completeness': self._validate_completeness,
            'outlier_detection': self._detect_outliers,
            'spread_validation': self._validate_spreads
        }
    
    def validate_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> ValidationResult:
        """Comprehensive data validation"""
        if df.empty:
            return ValidationResult(
                is_valid=False,
                errors=["DataFrame is empty"],
                warnings=[],
                quality_score=0.0,
                metrics={}
            )
        
        errors = []
        warnings = []
        metrics = {}
        
        # Run all validation rules
        for rule_name, rule_func in self.validation_rules.items():
            try:
                result = rule_func(df, symbol, timeframe)
                
                if result['errors']:
                    errors.extend(result['errors'])
                if result['warnings']:
                    warnings.extend(result['warnings'])
                if result['metrics']:
                    metrics.update(result['metrics'])
                    
            except Exception as e:
                logger.error(f"Validation rule {rule_name} failed: {e}")
                errors.append(f"Validation rule {rule_name} failed: {str(e)}")
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(df, errors, warnings, metrics)
        
        is_valid = len(errors) == 0 and quality_score >= 0.8
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            quality_score=quality_score,
            metrics=metrics
        )
    
    def _validate_ohlc_integrity(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """Validate OHLC data integrity"""
        errors = []
        warnings = []
        metrics = {}
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return {'errors': errors, 'warnings': warnings, 'metrics': metrics}
        
        # Validate OHLC relationships
        invalid_high = df['high'] < df[['open', 'low', 'close']].max(axis=1)
        invalid_low = df['low'] > df[['open', 'high', 'close']].min(axis=1)
        
        invalid_candles = invalid_high | invalid_low
        invalid_count = invalid_candles.sum()
        
        if invalid_count > 0:
            error_pct = (invalid_count / len(df)) * 100
            if error_pct > 1.0:
                errors.append(f"High percentage of invalid OHLC candles: {error_pct:.2f}%")
            else:
                warnings.append(f"Found {invalid_count} invalid OHLC candles ({error_pct:.2f}%)")
        
        # Check for zero/negative prices
        zero_prices = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1).sum()
        if zero_prices > 0:
            errors.append(f"Found {zero_prices} candles with zero/negative prices")
        
        # Check for extreme price movements
        price_changes = df['close'].pct_change().abs()
        extreme_moves = (price_changes > 0.10).sum()  # >10% moves
        if extreme_moves > len(df) * 0.01:  # More than 1% of candles
            warnings.append(f"High number of extreme price movements: {extreme_moves}")
        
        metrics.update({
            'invalid_ohlc_count': invalid_count,
            'invalid_ohlc_pct': (invalid_count / len(df)) * 100,
            'zero_price_count': zero_prices,
            'extreme_move_count': extreme_moves
        })
        
        return {'errors': errors, 'warnings': warnings, 'metrics': metrics}
    
    def _validate_volume_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """Validate volume data"""
        errors = []
        warnings = []
        metrics = {}
        
        if 'volume' not in df.columns:
            warnings.append("Volume data not available")
            return {'errors': errors, 'warnings': warnings, 'metrics': metrics}
        
        # Check for negative volume
        negative_volume = (df['volume'] < 0).sum()
        if negative_volume > 0:
            errors.append(f"Found {negative_volume} candles with negative volume")
        
        # Check for zero volume
        zero_volume = (df['volume'] == 0).sum()
        zero_volume_pct = (zero_volume / len(df)) * 100
        
        if zero_volume_pct > 10:  # More than 10% zero volume
            warnings.append(f"High percentage of zero volume candles: {zero_volume_pct:.2f}%")
        
        # Volume consistency check
        volume_mean = df['volume'].mean()
        volume_std = df['volume'].std()
        volume_outliers = (df['volume'] > volume_mean + 5 * volume_std).sum()
        
        metrics.update({
            'negative_volume_count': negative_volume,
            'zero_volume_count': zero_volume,
            'zero_volume_pct': zero_volume_pct,
            'volume_outliers': volume_outliers,
            'avg_volume': volume_mean
        })
        
        return {'errors': errors, 'warnings': warnings, 'metrics': metrics}
    
    def _validate_time_consistency(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """Validate time series consistency"""
        errors = []
        warnings = []
        metrics = {}
        
        if not isinstance(df.index, pd.DatetimeIndex):
            errors.append("Index is not a DatetimeIndex")
            return {'errors': errors, 'warnings': warnings, 'metrics': metrics}
        
        # Check for duplicate timestamps
        duplicate_times = df.index.duplicated().sum()
        if duplicate_times > 0:
            errors.append(f"Found {duplicate_times} duplicate timestamps")
        
        # Check for correct chronological order
        if not df.index.is_monotonic_increasing:
            errors.append("Data is not in chronological order")
        
        # Check for gaps in time series
        time_diffs = df.index.to_series().diff()
        expected_interval = self._get_expected_interval(timeframe)
        
        if expected_interval:
            # Find gaps larger than expected
            large_gaps = time_diffs > expected_interval * 3  # Allow for weekends/holidays
            gap_count = large_gaps.sum()
            
            if gap_count > 0:
                warnings.append(f"Found {gap_count} large gaps in time series")
            
            metrics['large_gaps'] = gap_count
        
        metrics.update({
            'duplicate_times': duplicate_times,
            'chronological_order': df.index.is_monotonic_increasing,
            'total_timespan': str(df.index[-1] - df.index[0]),
            'avg_interval': str(time_diffs.mean())
        })
        
        return {'errors': errors, 'warnings': warnings, 'metrics': metrics}
    
    def _validate_completeness(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """Validate data completeness"""
        errors = []
        warnings = []
        metrics = {}
        
        # Check for missing values
        missing_data = df.isnull().sum()
        total_missing = missing_data.sum()
        
        if total_missing > 0:
            missing_pct = (total_missing / (len(df) * len(df.columns))) * 100
            if missing_pct > 0.1:  # More than 0.1% missing
                errors.append(f"High percentage of missing data: {missing_pct:.2f}%")
            else:
                warnings.append(f"Found {total_missing} missing values ({missing_pct:.2f}%)")
        
        # Check data density
        expected_candles = self._estimate_expected_candles(df, timeframe)
        if expected_candles and len(df) < expected_candles * 0.9:  # Less than 90% of expected
            completeness_pct = (len(df) / expected_candles) * 100
            warnings.append(f"Data completeness: {completeness_pct:.1f}% of expected candles")
        
        metrics.update({
            'missing_values': total_missing,
            'missing_pct': (total_missing / (len(df) * len(df.columns))) * 100,
            'actual_candles': len(df),
            'expected_candles': expected_candles or 'unknown'
        })
        
        return {'errors': errors, 'warnings': warnings, 'metrics': metrics}
    
    def _detect_outliers(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """Detect price outliers"""
        errors = []
        warnings = []
        metrics = {}
        
        # Calculate z-scores for prices
        price_cols = ['open', 'high', 'low', 'close']
        outlier_counts = {}
        
        for col in price_cols:
            if col in df.columns:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = (z_scores > 4).sum()  # More than 4 standard deviations
                outlier_counts[col] = outliers
                
                if outliers > len(df) * 0.01:  # More than 1% outliers
                    warnings.append(f"High number of {col} outliers: {outliers}")
        
        # Check for price spikes
        close_changes = df['close'].pct_change().abs()
        price_spikes = (close_changes > 0.05).sum()  # >5% moves
        
        if price_spikes > len(df) * 0.02:  # More than 2% spikes
            warnings.append(f"High number of price spikes: {price_spikes}")
        
        metrics.update({
            'outlier_counts': outlier_counts,
            'price_spikes': price_spikes,
            'max_price_change': close_changes.max()
        })
        
        return {'errors': errors, 'warnings': warnings, 'metrics': metrics}
    
    def _validate_spreads(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """Validate spread data if available"""
        errors = []
        warnings = []
        metrics = {}
        
        if 'spread' not in df.columns:
            return {'errors': errors, 'warnings': warnings, 'metrics': metrics}
        
        # Check for negative spreads
        negative_spreads = (df['spread'] < 0).sum()
        if negative_spreads > 0:
            errors.append(f"Found {negative_spreads} negative spreads")
        
        # Check for extreme spreads
        spread_mean = df['spread'].mean()
        spread_std = df['spread'].std()
        extreme_spreads = (df['spread'] > spread_mean + 5 * spread_std).sum()
        
        if extreme_spreads > len(df) * 0.01:
            warnings.append(f"High number of extreme spreads: {extreme_spreads}")
        
        metrics.update({
            'negative_spreads': negative_spreads,
            'extreme_spreads': extreme_spreads,
            'avg_spread': spread_mean,
            'max_spread': df['spread'].max()
        })
        
        return {'errors': errors, 'warnings': warnings, 'metrics': metrics}
    
    def _get_expected_interval(self, timeframe: str) -> Optional[timedelta]:
        """Get expected time interval for timeframe"""
        intervals = {
            'M1': timedelta(minutes=1), 'M5': timedelta(minutes=5),
            'M15': timedelta(minutes=15), 'M30': timedelta(minutes=30),
            'H1': timedelta(hours=1), 'H4': timedelta(hours=4),
            'D1': timedelta(days=1), 'W1': timedelta(weeks=1)
        }
        return intervals.get(timeframe)
    
    def _estimate_expected_candles(self, df: pd.DataFrame, timeframe: str) -> Optional[int]:
        """Estimate expected number of candles"""
        if len(df) < 2:
            return None
        
        total_time = df.index[-1] - df.index[0]
        interval = self._get_expected_interval(timeframe)
        
        if interval:
            # Account for weekends in forex (5/7 days)
            expected = int(total_time / interval * 5/7) if timeframe in ['D1', 'W1'] else int(total_time / interval)
            return expected
        
        return None
    
    def _calculate_quality_score(self, df: pd.DataFrame, errors: List[str], warnings: List[str], metrics: Dict) -> float:
        """Calculate overall data quality score (0-1)"""
        if not df.empty:
            base_score = 1.0
            
            # Deduct for errors (major issues)
            error_penalty = len(errors) * 0.2
            
            # Deduct for warnings (minor issues)
            warning_penalty = len(warnings) * 0.05
            
            # Deduct for data quality issues
            quality_penalties = 0.0
            
            if 'invalid_ohlc_pct' in metrics:
                quality_penalties += metrics['invalid_ohlc_pct'] / 100 * 0.3
            
            if 'missing_pct' in metrics:
                quality_penalties += metrics['missing_pct'] / 100 * 0.2
            
            final_score = max(0.0, base_score - error_penalty - warning_penalty - quality_penalties)
            return final_score
        
        return 0.0

