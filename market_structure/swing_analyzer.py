# market_structure/swing_analyzer.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

from utils.constants import MarketStructure

class SwingPoint(NamedTuple):
    """Represents a swing point (high or low)"""
    timestamp: datetime
    price: float
    candle_index: int
    swing_type: str  # 'HIGH' or 'LOW'
    strength: float  # 0.0 to 1.0, strength of the swing point
    volume: Optional[float] = None
    confirmed: bool = False  # Add confirmed field with default False

@dataclass 
class SwingAnalysisConfig:
    """Configuration for swing analysis"""
    lookback_period: int = 5  # Periods to look back for swing detection
    min_swing_strength: float = 0.3  # Minimum strength for valid swing
    use_volume_confirmation: bool = True  # Use volume for swing validation
    filter_weak_swings: bool = True  # Filter out weak swing points
    atr_multiplier: float = 0.5  # ATR multiplier for swing validation

class SwingAnalyzer:
    """
    Professional swing point analyzer for SMC trading
    Detects and validates swing highs and lows with quality scoring
    """
    
    def __init__(self, config: SwingAnalysisConfig = None):
        self.config = config or SwingAnalysisConfig()
        logger.info(f"SwingAnalyzer initialized with lookback: {self.config.lookback_period}")
    
    def analyze_swings(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> Dict:
        """
        Comprehensive swing point analysis
        Returns detected swing points with quality assessment
        """
        if df.empty:
            logger.warning("Empty DataFrame passed to swing analyzer")
            return self._empty_result()
        
        logger.info(f"Analyzing swings for {symbol} - {len(df)} candles")
        
        try:
            # Detect primary swing points
            swing_points = self._detect_primary_swings(df)
            
            if not swing_points:
                logger.warning(f"No swing points detected for {symbol}")
                return self._empty_result()
            
            # Validate and filter swing points
            validated_swings = self._validate_swings(swing_points, df)
            
            # Calculate swing quality metrics
            quality_metrics = self._calculate_swing_quality(validated_swings, df)
            
            # Generate swing analysis summary
            analysis_summary = self._generate_swing_summary(validated_swings, df)
            
            logger.info(f"Detected {len(validated_swings)} valid swing points for {symbol}")
            
            return {
                'swing_points': validated_swings,
                'quality_metrics': quality_metrics,
                'analysis_summary': analysis_summary,
                'success': True,
                'symbol': symbol,
                'candles_analyzed': len(df),
                'swing_count': len(validated_swings)
            }
            
        except Exception as e:
            logger.error(f"Swing analysis failed for {symbol}: {e}")
            return self._empty_result(error=str(e))
    
    def _detect_primary_swings(self, df: pd.DataFrame) -> List[SwingPoint]:
        """Detect primary swing points using enhanced pivot detection"""
        
        swing_points = []
        lookback = self.config.lookback_period
        
        # Ensure we have enough data
        if len(df) < lookback * 2 + 1:
            logger.debug(f"Insufficient data for swing detection: {len(df)} candles")
            return swing_points
        
        # Calculate ATR for swing validation
        atr = self._calculate_atr(df)
        
        # Detect swing highs and lows
        for i in range(lookback, len(df) - lookback):
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            current_time = df.index[i]
            
            # Check for swing high
            if self._is_swing_high(df, i, lookback):
                strength = self._calculate_swing_strength(df, i, 'HIGH', atr)
                
                if strength >= self.config.min_swing_strength:
                    volume = df.iloc[i].get('volume', None)
                    # Confirm swing based on volume or price action
                    confirmed = self._confirm_swing(df, i, 'HIGH') if self.config.use_volume_confirmation else True
                    
                    swing_point = SwingPoint(
                        timestamp=current_time,
                        price=current_high,
                        candle_index=i,
                        swing_type='HIGH',
                        strength=strength,
                        volume=volume,
                        confirmed=confirmed
                    )
                    swing_points.append(swing_point)
            
            # Check for swing low
            if self._is_swing_low(df, i, lookback):
                strength = self._calculate_swing_strength(df, i, 'LOW', atr)
                
                if strength >= self.config.min_swing_strength:
                    volume = df.iloc[i].get('volume', None)
                    
                    swing_point = SwingPoint(
                        timestamp=current_time,
                        price=current_low,
                        candle_index=i,
                        swing_type='LOW',
                        strength=strength,
                        volume=volume,
                        confirmed=self._confirm_swing(df, i, 'LOW') if self.config.use_volume_confirmation else True

                    )
                    swing_points.append(swing_point)
        
        logger.debug(f"Detected {len(swing_points)} primary swing points")
        return swing_points
    
    def _is_swing_high(self, df: pd.DataFrame, index: int, lookback: int) -> bool:
        """Check if the candle at index is a swing high"""
        try:
            current_high = df.iloc[index]['high']
            
            # Check left side (previous candles)
            for i in range(max(0, index - lookback), index):
                if df.iloc[i]['high'] >= current_high:
                    return False
            
            # Check right side (next candles)
            for i in range(index + 1, min(len(df), index + lookback + 1)):
                if df.iloc[i]['high'] >= current_high:
                    return False
            
            return True
        except Exception as e:
            logger.debug(f"Error checking swing high at index {index}: {e}")
            return False
    
    def _is_swing_low(self, df: pd.DataFrame, index: int, lookback: int) -> bool:
        """Check if the candle at index is a swing low"""
        try:
            current_low = df.iloc[index]['low']
            
            # Check left side (previous candles)
            for i in range(max(0, index - lookback), index):
                if df.iloc[i]['low'] <= current_low:
                    return False
            
            # Check right side (next candles)
            for i in range(index + 1, min(len(df), index + lookback + 1)):
                if df.iloc[i]['low'] <= current_low:
                    return False
            
            return True
        except Exception as e:
            logger.debug(f"Error checking swing low at index {index}: {e}")
            return False
    
    def _calculate_swing_strength(self, df: pd.DataFrame, index: int, swing_type: str, atr: float) -> float:
        """Calculate the strength/quality of a swing point"""
        try:
            strength_factors = []
            
            # Factor 1: Price movement relative to ATR
            if swing_type == 'HIGH':
                current_price = df.iloc[index]['high']
                # Compare to nearby lows
                nearby_low = df.iloc[max(0, index-10):min(len(df), index+10)]['low'].min()
                price_movement = current_price - nearby_low
            else:  # LOW
                current_price = df.iloc[index]['low']
                # Compare to nearby highs
                nearby_high = df.iloc[max(0, index-10):min(len(df), index+10)]['high'].max()
                price_movement = nearby_high - current_price
            
            if atr > 0:
                movement_strength = min(price_movement / (atr * self.config.atr_multiplier), 1.0)
                strength_factors.append(movement_strength)
            else:
                strength_factors.append(0.5)  # Default when ATR unavailable
            
            # Factor 2: Volume confirmation (if available)
            if self.config.use_volume_confirmation and 'volume' in df.columns:
                current_volume = df.iloc[index].get('volume', 0)
                if current_volume > 0:
                    avg_volume = df['volume'].rolling(20).mean().iloc[index]
                    if avg_volume > 0:
                        volume_strength = min(current_volume / avg_volume / 2.0, 1.0)
                        strength_factors.append(volume_strength)
            
            # Factor 3: Candle body size relative to range
            candle_range = df.iloc[index]['high'] - df.iloc[index]['low']
            if candle_range > 0:
                body_size = abs(df.iloc[index]['close'] - df.iloc[index]['open'])
                body_strength = body_size / candle_range
                strength_factors.append(body_strength)
            
            # Factor 4: Time since last swing of same type
            time_factor = self._calculate_time_factor(df, index, swing_type)
            strength_factors.append(time_factor)
            
            # Calculate weighted average
            if strength_factors:
                return min(np.mean(strength_factors), 1.0)
            else:
                return 0.5
                
        except Exception as e:
            logger.debug(f"Error calculating swing strength: {e}")
            return 0.5
    
    def _confirm_swing(self, df: pd.DataFrame, index: int, swing_type: str) -> bool:
        """Confirm swing point based on volume or price action."""
        try:
            if 'volume' not in df.columns:
                return True  # Default to True if volume not available
            
            current_volume = df.iloc[index]['volume']
            avg_volume = df['volume'].rolling(20).mean().iloc[index]
            
            if pd.isna(avg_volume) or avg_volume == 0:
                return True
            
            # Consider swing confirmed if volume is above average
            volume_confirmation = current_volume > avg_volume
            
            # Additional price action confirmation (e.g., candle body size)
            candle_range = df.iloc[index]['high'] - df.iloc[index]['low']
            if candle_range > 0:
                body_size = abs(df.iloc[index]['close'] - df.iloc[index]['open'])
                body_confirmation = body_size / candle_range >= 0.5
            else:
                body_confirmation = True
            
            return volume_confirmation or body_confirmation
        except Exception as e:
            logger.debug(f"Error confirming swing at index {index}: {e}")
            return True  # Default to True on error to avoid blocking analysis



    def _calculate_time_factor(self, df: pd.DataFrame, index: int, swing_type: str) -> float:
        """Calculate time factor based on spacing between swings"""
        try:
            # Look for previous swing of same type
            lookback_range = min(50, index)
            
            for i in range(index - 1, max(0, index - lookback_range), -1):
                if swing_type == 'HIGH' and self._is_swing_high(df, i, self.config.lookback_period):
                    distance = index - i
                    # Optimal distance is around 10-20 candles
                    if distance >= 10:
                        return min(distance / 20.0, 1.0)
                    else:
                        return distance / 20.0
                elif swing_type == 'LOW' and self._is_swing_low(df, i, self.config.lookback_period):
                    distance = index - i
                    if distance >= 10:
                        return min(distance / 20.0, 1.0)
                    else:
                        return distance / 20.0
            
            # If no previous swing found, return medium strength
            return 0.7
            
        except Exception as e:
            logger.debug(f"Error calculating time factor: {e}")
            return 0.5
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            if len(df) < period:
                return 0.0
            
            high = df['high']
            low = df['low']
            close = df['close'].shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean().iloc[-1]
            
            return atr if not pd.isna(atr) else 0.0
            
        except Exception as e:
            logger.debug(f"Error calculating ATR: {e}")
            return 0.0
    
    def _validate_swings(self, swing_points: List[SwingPoint], df: pd.DataFrame) -> List[SwingPoint]:
        """Validate and filter swing points"""
        if not swing_points:
            return []
        
        validated = []
        
        # Sort by timestamp
        sorted_swings = sorted(swing_points, key=lambda x: x.candle_index)
        
        # Filter out weak swings if enabled
        if self.config.filter_weak_swings:
            for swing in sorted_swings:
                if swing.strength >= self.config.min_swing_strength:
                    validated.append(swing)
        else:
            validated = sorted_swings.copy()
        
        # Remove duplicate swings at same index
        final_swings = []
        used_indices = set()
        
        for swing in validated:
            if swing.candle_index not in used_indices:
                final_swings.append(swing)
                used_indices.add(swing.candle_index)
        
        # Ensure alternating highs and lows
        alternating_swings = self._ensure_alternating_swings(final_swings)
        
        logger.debug(f"Validated {len(alternating_swings)} swing points from {len(swing_points)} detected")
        return alternating_swings
    
    def _ensure_alternating_swings(self, swing_points: List[SwingPoint]) -> List[SwingPoint]:
        """Ensure swing points alternate between highs and lows"""
        if len(swing_points) <= 1:
            return swing_points
        
        alternating = [swing_points[0]]
        
        for i in range(1, len(swing_points)):
            current = swing_points[i]
            previous = alternating[-1]
            
            # Only add if it's different type from previous
            if current.swing_type != previous.swing_type:
                alternating.append(current)
            else:
                # Keep the stronger swing of the same type
                if current.strength > previous.strength:
                    alternating[-1] = current
        
        return alternating
    
    def _calculate_swing_quality(self, swing_points: List[SwingPoint], df: pd.DataFrame) -> Dict:
        """Calculate overall quality metrics for swing analysis"""
        if not swing_points:
            return {
                'overall_quality': 0.0,
                'avg_strength': 0.0,
                'swing_density': 0.0,
                'alternation_score': 0.0
            }
        
        # Average swing strength
        avg_strength = np.mean([sp.strength for sp in swing_points])
        
        # Swing density (swings per 100 candles)
        swing_density = len(swing_points) / len(df) * 100
        
        # Alternation score (how well swings alternate)
        alternation_score = self._calculate_alternation_score(swing_points)
        
        # Overall quality score
        quality_factors = [avg_strength, min(swing_density / 10, 1.0), alternation_score]
        overall_quality = np.mean(quality_factors)
        
        return {
            'overall_quality': overall_quality,
            'avg_strength': avg_strength,
            'swing_density': swing_density,
            'alternation_score': alternation_score,
            'high_count': len([sp for sp in swing_points if sp.swing_type == 'HIGH']),
            'low_count': len([sp for sp in swing_points if sp.swing_type == 'LOW'])
        }
    
    def _calculate_alternation_score(self, swing_points: List[SwingPoint]) -> float:
        """Calculate how well swings alternate between highs and lows"""
        if len(swing_points) <= 1:
            return 1.0
        
        alternations = 0
        for i in range(1, len(swing_points)):
            if swing_points[i].swing_type != swing_points[i-1].swing_type:
                alternations += 1
        
        max_alternations = len(swing_points) - 1
        return alternations / max_alternations if max_alternations > 0 else 1.0
    
    def _generate_swing_summary(self, swing_points: List[SwingPoint], df: pd.DataFrame) -> Dict:
        """Generate comprehensive swing analysis summary"""
        if not swing_points:
            return {
                'trend_direction': MarketStructure.SIDEWAYS,
                'recent_swings': [],
                'swing_progression': 'UNCLEAR'
            }
        
        # Determine trend from recent swings
        trend_direction = self._determine_trend_from_swings(swing_points)
        
        # Get recent swings (last 5)
        recent_swings = swing_points[-5:] if len(swing_points) >= 5 else swing_points
        
        # Analyze swing progression
        swing_progression = self._analyze_swing_progression(swing_points)
        
        return {
            'trend_direction': trend_direction,
            'recent_swings': recent_swings,
            'swing_progression': swing_progression,
            'first_swing': swing_points[0] if swing_points else None,
            'last_swing': swing_points[-1] if swing_points else None,
            'swing_range': self._calculate_swing_range(swing_points)
        }
    
    def _determine_trend_from_swings(self, swing_points: List[SwingPoint]) -> str:
        """Determine trend direction from swing progression"""
        if len(swing_points) < 4:
            return MarketStructure.SIDEWAYS
        
        # Get recent highs and lows
        recent_swings = swing_points[-6:] if len(swing_points) >= 6 else swing_points
        highs = [sp for sp in recent_swings if sp.swing_type == 'HIGH']
        lows = [sp for sp in recent_swings if sp.swing_type == 'LOW']
        
        if len(highs) >= 2 and len(lows) >= 2:
            # Sort by time
            highs.sort(key=lambda x: x.candle_index)
            lows.sort(key=lambda x: x.candle_index)
            
            # Check for higher highs and higher lows (bullish)
            if len(highs) >= 2:
                higher_highs = highs[-1].price > highs[-2].price
            else:
                higher_highs = False
                
            if len(lows) >= 2:
                higher_lows = lows[-1].price > lows[-2].price
            else:
                higher_lows = False
            
            # Check for lower highs and lower lows (bearish)
            if len(highs) >= 2:
                lower_highs = highs[-1].price < highs[-2].price
            else:
                lower_highs = False
                
            if len(lows) >= 2:
                lower_lows = lows[-1].price < lows[-2].price
            else:
                lower_lows = False
            
            # Determine trend
            if higher_highs and higher_lows:
                return MarketStructure.BULLISH
            elif lower_highs and lower_lows:
                return MarketStructure.BEARISH
            else:
                return MarketStructure.SIDEWAYS
        
        return MarketStructure.SIDEWAYS
    
    def _analyze_swing_progression(self, swing_points: List[SwingPoint]) -> str:
        """Analyze the progression/quality of swings"""
        if len(swing_points) < 3:
            return 'INSUFFICIENT_DATA'
        
        # Check strength progression
        recent_strengths = [sp.strength for sp in swing_points[-3:]]
        avg_recent_strength = np.mean(recent_strengths)
        
        if avg_recent_strength >= 0.7:
            return 'STRONG_PROGRESSION'
        elif avg_recent_strength >= 0.5:
            return 'MODERATE_PROGRESSION'
        else:
            return 'WEAK_PROGRESSION'
    
    def _calculate_swing_range(self, swing_points: List[SwingPoint]) -> Dict:
        """Calculate price range covered by swings"""
        if not swing_points:
            return {'high': 0.0, 'low': 0.0, 'range': 0.0}
        
        prices = [sp.price for sp in swing_points]
        swing_high = max(prices)
        swing_low = min(prices)
        swing_range = swing_high - swing_low
        
        return {
            'high': swing_high,
            'low': swing_low,
            'range': swing_range
        }
    
    def get_swing_levels(self, df: pd.DataFrame, lookback_swings: int = 10) -> Dict:
        """Get key swing levels for trading decisions"""
        analysis = self.analyze_swings(df)
        
        if not analysis['success']:
            return {'support_levels': [], 'resistance_levels': []}
        
        swing_points = analysis['swing_points']
        recent_swings = swing_points[-lookback_swings:] if len(swing_points) >= lookback_swings else swing_points
        
        # Extract support and resistance levels
        resistance_levels = [sp.price for sp in recent_swings if sp.swing_type == 'HIGH']
        support_levels = [sp.price for sp in recent_swings if sp.swing_type == 'LOW']
        
        # Sort levels
        resistance_levels.sort(reverse=True)
        support_levels.sort()
        
        return {
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'key_levels': resistance_levels + support_levels
        }
    
    def _empty_result(self, error: str = None) -> Dict:
        """Return empty result structure"""
        return {
            'swing_points': [],
            'quality_metrics': {
                'overall_quality': 0.0,
                'avg_strength': 0.0,
                'swing_density': 0.0,
                'alternation_score': 0.0
            },
            'analysis_summary': {
                'trend_direction': MarketStructure.SIDEWAYS,
                'recent_swings': [],
                'swing_progression': 'UNCLEAR'
            },
            'success': False,
            'error': error,
            'swing_count': 0
        }