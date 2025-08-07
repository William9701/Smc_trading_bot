# market_structure/swing_analyzer.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

from utils.constants import MarketStructure
from utils.helpers import calculate_atr

class SwingPoint(NamedTuple):
    """Represents a swing high or low point"""
    timestamp: datetime
    price: float
    swing_type: str  # 'HIGH' or 'LOW'
    strength: float  # 0.0 to 1.0
    candle_index: int
    confirmed: bool = False

@dataclass
class SwingAnalysisConfig:
    """Configuration for swing analysis"""
    lookback_period: int = 5
    min_swing_size_atr: float = 0.5  # Minimum swing size in ATR multiples
    confirmation_bars: int = 2
    use_dynamic_period: bool = True
    min_strength_threshold: float = 0.3

class SwingAnalyzer:
    """
    Professional swing point detection for SMC analysis
    Uses multiple algorithms to identify significant swing highs and lows
    """
    
    def __init__(self, config: SwingAnalysisConfig = None):
        self.config = config or SwingAnalysisConfig()
        logger.info(f"SwingAnalyzer initialized with lookback: {self.config.lookback_period}")
    
    def analyze_swings(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> Dict:
        """
        Comprehensive swing analysis
        Returns dictionary with swing points and analysis metrics
        """
        if df.empty or len(df) < self.config.lookback_period * 2:
            logger.warning(f"Insufficient data for swing analysis: {len(df)} candles")
            return self._empty_result()
        
        logger.info(f"Analyzing swings for {symbol} - {len(df)} candles")
        
        try:
            # Calculate ATR for swing size validation
            atr = calculate_atr(df, period=14)
            
            # Detect swing points using multiple methods
            primary_swings = self._detect_primary_swings(df, atr)
            
            # Validate and filter swings
            validated_swings = self._validate_swings(df, primary_swings, atr)
            
            # Calculate swing strength
            swing_points = self._calculate_swing_strength(df, validated_swings)
            
            # Generate analysis metrics
            analysis_metrics = self._generate_swing_metrics(swing_points, df)
            
            logger.info(f"Detected {len(swing_points)} valid swing points for {symbol}")
            
            return {
                'swing_points': swing_points,
                'swing_highs': [sp for sp in swing_points if sp.swing_type == 'HIGH'],
                'swing_lows': [sp for sp in swing_points if sp.swing_type == 'LOW'],
                'analysis_metrics': analysis_metrics,
                'success': True,
                'symbol': symbol,
                'candles_analyzed': len(df)
            }
            
        except Exception as e:
            logger.error(f"Swing analysis failed for {symbol}: {e}")
            return self._empty_result(error=str(e))
    
    def _detect_primary_swings(self, df: pd.DataFrame, atr: pd.Series) -> List[Dict]:
        """Detect primary swing points using pivots method"""
        swings = []
        lookback = self.config.lookback_period
        
        # Adjust lookback based on volatility if dynamic mode enabled
        if self.config.use_dynamic_period:
            volatility = atr.mean() / df['close'].mean()
            lookback = max(3, min(15, int(lookback * (1 + volatility * 10))))
        
        # Detect swing highs
        for i in range(lookback, len(df) - lookback):
            current_high = df['high'].iloc[i]
            
            # Check if current candle is higher than surrounding candles
            is_swing_high = True
            for j in range(1, lookback + 1):
                if (current_high <= df['high'].iloc[i - j] or 
                    current_high <= df['high'].iloc[i + j]):
                    is_swing_high = False
                    break
            
            if is_swing_high:
                # Validate swing size using ATR
                if self._is_significant_swing(df, i, atr, 'HIGH'):
                    swings.append({
                        'index': i,
                        'timestamp': df.index[i],
                        'price': current_high,
                        'type': 'HIGH',
                        'lookback_used': lookback
                    })
        
        # Detect swing lows
        for i in range(lookback, len(df) - lookback):
            current_low = df['low'].iloc[i]
            
            # Check if current candle is lower than surrounding candles
            is_swing_low = True
            for j in range(1, lookback + 1):
                if (current_low >= df['low'].iloc[i - j] or 
                    current_low >= df['low'].iloc[i + j]):
                    is_swing_low = False
                    break
            
            if is_swing_low:
                # Validate swing size using ATR
                if self._is_significant_swing(df, i, atr, 'LOW'):
                    swings.append({
                        'index': i,
                        'timestamp': df.index[i],
                        'price': current_low,
                        'type': 'LOW',
                        'lookback_used': lookback
                    })
        
        logger.debug(f"Detected {len(swings)} primary swing points")
        return swings
    
    def _is_significant_swing(self, df: pd.DataFrame, index: int, atr: pd.Series, swing_type: str) -> bool:
        """Check if swing meets minimum size requirements"""
        try:
            current_atr = atr.iloc[index]
            if pd.isna(current_atr) or current_atr <= 0:
                return True  # If ATR not available, assume significant
            
            min_size = current_atr * self.config.min_swing_size_atr
            
            if swing_type == 'HIGH':
                # Check swing size from nearby lows
                lookback = self.config.lookback_period
                nearby_lows = df['low'].iloc[max(0, index-lookback):index+lookback+1]
                swing_size = df['high'].iloc[index] - nearby_lows.min()
            else:  # LOW
                # Check swing size from nearby highs
                lookback = self.config.lookback_period
                nearby_highs = df['high'].iloc[max(0, index-lookback):index+lookback+1]
                swing_size = nearby_highs.max() - df['low'].iloc[index]
            
            return swing_size >= min_size
            
        except Exception as e:
            logger.debug(f"Swing size validation error: {e}")
            return True  # Default to significant if validation fails
    
    def _validate_swings(self, df: pd.DataFrame, swings: List[Dict], atr: pd.Series) -> List[Dict]:
        """Validate and filter swing points"""
        if not swings:
            return []
        
        validated = []
        
        # Sort swings by timestamp
        swings_sorted = sorted(swings, key=lambda x: x['index'])
        
        # Remove swings that are too close together
        for i, swing in enumerate(swings_sorted):
            is_valid = True
            
            # Check minimum distance from other swings
            for j, other_swing in enumerate(swings_sorted):
                if i != j:
                    time_distance = abs(swing['index'] - other_swing['index'])
                    
                    # Same type swings should be separated
                    if swing['type'] == other_swing['type'] and time_distance < self.config.lookback_period:
                        # Keep the stronger swing
                        if swing['type'] == 'HIGH':
                            if swing['price'] < other_swing['price']:
                                is_valid = False
                                break
                        else:  # LOW
                            if swing['price'] > other_swing['price']:
                                is_valid = False
                                break
            
            if is_valid:
                validated.append(swing)
        
        logger.debug(f"Validated {len(validated)} swing points from {len(swings)} detected")
        return validated
    
    def _calculate_swing_strength(self, df: pd.DataFrame, swings: List[Dict]) -> List[SwingPoint]:
        """Calculate strength scores for swing points"""
        swing_points = []
        
        for swing in swings:
            try:
                # Calculate strength based on multiple factors
                strength_factors = []
                
                # 1. Price dominance (how much it stands out)
                lookback = swing.get('lookback_used', self.config.lookback_period)
                start_idx = max(0, swing['index'] - lookback)
                end_idx = min(len(df), swing['index'] + lookback + 1)
                
                if swing['type'] == 'HIGH':
                    local_prices = df['high'].iloc[start_idx:end_idx]
                    dominance = (swing['price'] - local_prices.mean()) / local_prices.std()
                else:
                    local_prices = df['low'].iloc[start_idx:end_idx]
                    dominance = (local_prices.mean() - swing['price']) / local_prices.std()
                
                strength_factors.append(min(1.0, dominance / 2.0))
                
                # 2. Volume confirmation (if available)
                if 'volume' in df.columns:
                    local_volumes = df['volume'].iloc[start_idx:end_idx]
                    volume_strength = df['volume'].iloc[swing['index']] / local_volumes.mean()
                    strength_factors.append(min(1.0, volume_strength / 2.0))
                
                # 3. Candle structure strength
                candle = df.iloc[swing['index']]
                body_size = abs(candle['close'] - candle['open'])
                total_range = candle['high'] - candle['low']
                
                if total_range > 0:
                    if swing['type'] == 'HIGH':
                        wick_ratio = (candle['high'] - max(candle['open'], candle['close'])) / total_range
                    else:
                        wick_ratio = (min(candle['open'], candle['close']) - candle['low']) / total_range
                    
                    strength_factors.append(wick_ratio)
                
                # Calculate final strength (average of factors)
                final_strength = np.mean(strength_factors) if strength_factors else 0.5
                final_strength = max(0.0, min(1.0, final_strength))
                
                # Create SwingPoint
                swing_point = SwingPoint(
                    timestamp=swing['timestamp'],
                    price=swing['price'],
                    swing_type=swing['type'],
                    strength=final_strength,
                    candle_index=swing['index'],
                    confirmed=final_strength >= self.config.min_strength_threshold
                )
                
                swing_points.append(swing_point)
                
            except Exception as e:
                logger.warning(f"Error calculating swing strength: {e}")
                # Create basic swing point without strength calculation
                swing_point = SwingPoint(
                    timestamp=swing['timestamp'],
                    price=swing['price'],
                    swing_type=swing['type'],
                    strength=0.5,
                    candle_index=swing['index'],
                    confirmed=True
                )
                swing_points.append(swing_point)
        
        # Sort by timestamp
        swing_points.sort(key=lambda x: x.timestamp)
        
        return swing_points
    
    def _generate_swing_metrics(self, swing_points: List[SwingPoint], df: pd.DataFrame) -> Dict:
        """Generate comprehensive swing analysis metrics"""
        if not swing_points:
            return {'total_swings': 0, 'avg_strength': 0, 'confirmed_swings': 0}
        
        highs = [sp for sp in swing_points if sp.swing_type == 'HIGH']
        lows = [sp for sp in swing_points if sp.swing_type == 'LOW']
        
        metrics = {
            'total_swings': len(swing_points),
            'swing_highs': len(highs),
            'swing_lows': len(lows),
            'avg_strength': np.mean([sp.strength for sp in swing_points]),
            'confirmed_swings': len([sp for sp in swing_points if sp.confirmed]),
            'swing_density': len(swing_points) / len(df) * 100,  # swings per 100 candles
            'strongest_swing': max(swing_points, key=lambda x: x.strength) if swing_points else None,
            'weakest_swing': min(swing_points, key=lambda x: x.strength) if swing_points else None
        }
        
        # Calculate average swing size
        if len(swing_points) >= 2:
            swing_sizes = []
            for i in range(len(swing_points) - 1):
                size = abs(swing_points[i+1].price - swing_points[i].price)
                swing_sizes.append(size)
            
            metrics['avg_swing_size'] = np.mean(swing_sizes) if swing_sizes else 0
            metrics['max_swing_size'] = np.max(swing_sizes) if swing_sizes else 0
            metrics['min_swing_size'] = np.min(swing_sizes) if swing_sizes else 0
        
        return metrics
    
    def _empty_result(self, error: str = None) -> Dict:
        """Return empty result structure"""
        return {
            'swing_points': [],
            'swing_highs': [],
            'swing_lows': [],
            'analysis_metrics': {'total_swings': 0, 'error': error},
            'success': False,
            'symbol': 'UNKNOWN',
            'candles_analyzed': 0
        }
    
    def get_latest_swings(self, df: pd.DataFrame, count: int = 10) -> Dict:
        """Get the most recent swing points"""
        analysis = self.analyze_swings(df)
        
        if not analysis['success']:
            return analysis
        
        # Get latest swings
        latest_swings = sorted(
            analysis['swing_points'], 
            key=lambda x: x.timestamp, 
            reverse=True
        )[:count]
        
        return {
            **analysis,
            'latest_swings': latest_swings,
            'latest_high': next((sp for sp in latest_swings if sp.swing_type == 'HIGH'), None),
            'latest_low': next((sp for sp in latest_swings if sp.swing_type == 'LOW'), None)
        }
    
    def update_swing_config(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated swing config: {key} = {value}")

# Factory function
def create_swing_analyzer(lookback_period: int = 5, min_swing_size: float = 0.5) -> SwingAnalyzer:
    """Factory function to create swing analyzer"""
    config = SwingAnalysisConfig(
        lookback_period=lookback_period,
        min_swing_size_atr=min_swing_size
    )
    return SwingAnalyzer(config)