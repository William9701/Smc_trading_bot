# data_service/data_preprocessor.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger

from utils.exceptions import DataFetchError
from utils.constants import MarketStructure
from config.mt5_config import MT5_SYMBOL_CONFIG


class DataPreprocessor:
    """Professional data preprocessing service"""
    
    def __init__(self):
        self.processing_steps = {
            'clean_ohlc': self._clean_ohlc_data,
            'handle_gaps': self._handle_data_gaps,
            'normalize_volume': self._normalize_volume,
            'add_technical_indicators': self._add_basic_indicators,
            'detect_sessions': self._detect_trading_sessions
        }
    
    def preprocess_data(self, df: pd.DataFrame, symbol: str, options: Dict[str, bool] = None) -> pd.DataFrame:
        """Comprehensive data preprocessing"""
        if df.empty:
            logger.warning("Empty DataFrame passed to preprocessor")
            return df
        
        processed_df = df.copy()
        options = options or {}
        
        logger.info(f"Preprocessing {len(processed_df)} candles for {symbol}")
        
        # Apply preprocessing steps based on options
        for step_name, step_func in self.processing_steps.items():
            if options.get(step_name, True):  # Default to True
                try:
                    processed_df = step_func(processed_df, symbol)
                    logger.debug(f"Applied preprocessing step: {step_name}")
                except Exception as e:
                    logger.error(f"Preprocessing step {step_name} failed: {e}")
        
        logger.info(f"Preprocessing completed - {len(processed_df)} candles output")
        return processed_df
    
    def _clean_ohlc_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean OHLC data"""
        cleaned_df = df.copy()
        
        # Remove candles with invalid OHLC relationships
        valid_mask = (
            (cleaned_df['high'] >= cleaned_df[['open', 'low', 'close']].max(axis=1)) &
            (cleaned_df['low'] <= cleaned_df[['open', 'high', 'close']].min(axis=1)) &
            (cleaned_df[['open', 'high', 'low', 'close']] > 0).all(axis=1)
        )
        
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            logger.warning(f"Removing {invalid_count} invalid OHLC candles for {symbol}")
            cleaned_df = cleaned_df[valid_mask]
        
        return cleaned_df
    
    def _handle_data_gaps(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Handle gaps in data"""
        # Fix: Use ffill() instead of fillna(method='ffill')
        filled_df = df.ffill(limit=3)
        
        remaining_na = filled_df.isnull().sum().sum()
        if remaining_na > 0:
            logger.warning(f"Dropping {remaining_na} remaining NaN values for {symbol}")
            filled_df = filled_df.dropna()
        
        return filled_df
    
    def _normalize_volume(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Normalize volume data"""
        if 'volume' not in df.columns:
            return df
        
        normalized_df = df.copy()
        
        # Replace zero/negative volumes with median
        volume_median = normalized_df['volume'][normalized_df['volume'] > 0].median()
        invalid_volume_mask = normalized_df['volume'] <= 0
        
        if invalid_volume_mask.any():
            normalized_df.loc[invalid_volume_mask, 'volume'] = volume_median
            logger.debug(f"Normalized {invalid_volume_mask.sum()} invalid volume values")
        
        return normalized_df
    
    def _add_basic_indicators(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add basic technical indicators"""
        enhanced_df = df.copy()
        
        # Add basic price-based indicators
        enhanced_df['typical_price'] = (enhanced_df['high'] + enhanced_df['low'] + enhanced_df['close']) / 3
        enhanced_df['price_range'] = enhanced_df['high'] - enhanced_df['low']
        enhanced_df['body_size'] = abs(enhanced_df['close'] - enhanced_df['open'])
        enhanced_df['upper_wick'] = enhanced_df['high'] - enhanced_df[['open', 'close']].max(axis=1)
        enhanced_df['lower_wick'] = enhanced_df[['open', 'close']].min(axis=1) - enhanced_df['low']
        
        # Add simple moving averages
        enhanced_df['sma_20'] = enhanced_df['close'].rolling(window=20).mean()
        enhanced_df['sma_50'] = enhanced_df['close'].rolling(window=50).mean()
        
        return enhanced_df
    
    def _detect_trading_sessions(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Detect trading sessions"""
        session_df = df.copy()
        
        # Add hour column for session detection
        session_df['hour_utc'] = session_df.index.hour
        
        # Define trading sessions (UTC times)
        def get_session(hour):
            if 0 <= hour < 7:
                return 'ASIAN'
            elif 7 <= hour < 15:
                return 'LONDON'
            elif 15 <= hour < 22:
                return 'NEW_YORK'
            else:
                return 'ASIAN'
        
        session_df['session'] = session_df['hour_utc'].apply(get_session)
        
        return session_df

def create_data_preprocessor() -> DataPreprocessor:
    """Factory function to create data preprocessor"""
    return DataPreprocessor()