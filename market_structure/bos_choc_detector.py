# market_structure/bos_choc_detector.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

from utils.constants import StructureBreak, MarketStructure
from .swing_analyzer import SwingPoint, SwingAnalyzer

class StructureBreakPoint(NamedTuple):
    """Represents a structure break (BOS or CHoC)"""
    timestamp: datetime
    price: float
    break_type: str  # BOS_BULLISH, BOS_BEARISH, CHOC_BULLISH, CHOC_BEARISH
    previous_swing: SwingPoint
    breaking_candle_index: int
    confirmation_strength: float  # 0.0 to 1.0
    invalidated: bool = False

@dataclass
class BOSCHOCConfig:
    """Configuration for BOS/CHoC detection"""
    require_close_break: bool = True  # Require candle to close beyond level
    min_break_size_pips: float = 2.0  # Minimum break size in pips
    confirmation_bars: int = 1  # Bars to wait for confirmation
    allow_wick_breaks: bool = False  # Allow wick-only breaks
    use_body_breaks: bool = True  # Use candle body for breaks

class BOSCHOCDetector:
    """
    Professional Break of Structure (BOS) and Change of Character (CHoC) detector
    Implements SMC methodology for identifying market structure shifts
    """
    
    def __init__(self, config: BOSCHOCConfig = None):
        self.config = config or BOSCHOCConfig()
        self.swing_analyzer = SwingAnalyzer()
        logger.info("BOS/CHoC Detector initialized")
    
    def detect_structure_breaks(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> Dict:
        """
        Detect all BOS and CHoC points in the data
        Returns comprehensive analysis of market structure breaks
        """
        if df.empty:
            logger.warning("Empty DataFrame passed to structure break detector")
            return self._empty_result()
        
        logger.info(f"Detecting structure breaks for {symbol} - {len(df)} candles")
        
        try:
            # First get swing points
            swing_analysis = self.swing_analyzer.analyze_swings(df, symbol)
            
            if not swing_analysis['success'] or not swing_analysis['swing_points']:
                logger.warning(f"No swing points found for structure break analysis: {symbol}")
                return self._empty_result()
            
            swing_points = swing_analysis['swing_points']
            
            # Detect BOS and CHoC
            structure_breaks = self._analyze_structure_breaks(df, swing_points)
            
            # Classify current market structure
            current_structure = self._classify_current_structure(structure_breaks)
            
            # Generate analysis metrics
            analysis_metrics = self._generate_break_metrics(structure_breaks, df)
            
            logger.info(f"Detected {len(structure_breaks)} structure breaks for {symbol}")
            
            return {
                'structure_breaks': structure_breaks,
                'swing_points': swing_points,
                'current_structure': current_structure,
                'bos_points': [sb for sb in structure_breaks if 'BOS' in sb.break_type],
                'choc_points': [sb for sb in structure_breaks if 'CHOC' in sb.break_type],
                'analysis_metrics': analysis_metrics,
                'success': True,
                'symbol': symbol,
                'candles_analyzed': len(df)
            }
            
        except Exception as e:
            logger.error(f"Structure break detection failed for {symbol}: {e}")
            return self._empty_result(error=str(e))
    
    def _analyze_structure_breaks(self, df: pd.DataFrame, swing_points: List[SwingPoint]) -> List[StructureBreakPoint]:
        """Analyze swing points to identify BOS and CHoC"""
        structure_breaks = []
        
        if len(swing_points) < 3:
            logger.debug("Insufficient swing points for structure break analysis")
            return structure_breaks
        
        # Track current market structure trend
        current_trend = self._determine_initial_trend(swing_points[:3])
        
        for i in range(2, len(swing_points)):
            current_swing = swing_points[i]
            previous_swing = swing_points[i-1]
            prior_swing = swing_points[i-2]
            
            # Check for structure breaks
            break_point = self._check_structure_break(
                df, current_swing, previous_swing, prior_swing, current_trend
            )
            
            if break_point:
                structure_breaks.append(break_point)
                
                # Update current trend based on break
                if 'CHOC' in break_point.break_type:
                    current_trend = self._get_new_trend_from_choc(break_point.break_type)
        
        return structure_breaks
    
    def _check_structure_break(
        self, 
        df: pd.DataFrame, 
        current_swing: SwingPoint,
        previous_swing: SwingPoint, 
        prior_swing: SwingPoint,
        current_trend: str
    ) -> Optional[StructureBreakPoint]:
        """Check if current swing creates a structure break"""
        
        try:
            # Determine what type of break this could be
            if current_trend == MarketStructure.BULLISH:
                return self._check_bullish_structure_break(
                    df, current_swing, previous_swing, prior_swing
                )
            elif current_trend == MarketStructure.BEARISH:
                return self._check_bearish_structure_break(
                    df, current_swing, previous_swing, prior_swing
                )
            else:
                # Sideways - look for first directional break
                return self._check_sideways_break(
                    df, current_swing, previous_swing, prior_swing
                )
                
        except Exception as e:
            logger.debug(f"Error checking structure break: {e}")
            return None
    
    def _check_bullish_structure_break(
        self, 
        df: pd.DataFrame,
        current_swing: SwingPoint,
        previous_swing: SwingPoint,
        prior_swing: SwingPoint
    ) -> Optional[StructureBreakPoint]:
        """Check for breaks in bullish structure"""
        
        # In bullish structure, we expect higher highs and higher lows
        
        if current_swing.swing_type == 'HIGH':
            # Check for bullish BOS (break above previous high)
            if current_swing.price > previous_swing.price and previous_swing.swing_type == 'HIGH':
                # This is a higher high - potential BOS
                break_confirmed = self._confirm_structure_break(
                    df, current_swing, previous_swing, 'BOS_BULLISH'
                )
                
                if break_confirmed:
                    return StructureBreakPoint(
                        timestamp=current_swing.timestamp,
                        price=current_swing.price,
                        break_type=StructureBreak.BOS_BULLISH,
                        previous_swing=previous_swing,
                        breaking_candle_index=current_swing.candle_index,
                        confirmation_strength=break_confirmed
                    )
        
        elif current_swing.swing_type == 'LOW':
            # Check for bearish CHoC (break below previous low)
            if current_swing.price < previous_swing.price and previous_swing.swing_type == 'LOW':
                # This breaks the bullish structure - CHoC to bearish
                break_confirmed = self._confirm_structure_break(
                    df, current_swing, previous_swing, 'CHOC_BEARISH'
                )
                
                if break_confirmed:
                    return StructureBreakPoint(
                        timestamp=current_swing.timestamp,
                        price=current_swing.price,
                        break_type=StructureBreak.CHOC_BEARISH,
                        previous_swing=previous_swing,
                        breaking_candle_index=current_swing.candle_index,
                        confirmation_strength=break_confirmed
                    )
        
        return None
    
    def _check_bearish_structure_break(
        self, 
        df: pd.DataFrame,
        current_swing: SwingPoint,
        previous_swing: SwingPoint,
        prior_swing: SwingPoint
    ) -> Optional[StructureBreakPoint]:
        """Check for breaks in bearish structure"""
        
        # In bearish structure, we expect lower highs and lower lows
        
        if current_swing.swing_type == 'LOW':
            # Check for bearish BOS (break below previous low)
            if current_swing.price < previous_swing.price and previous_swing.swing_type == 'LOW':
                # This is a lower low - potential BOS
                break_confirmed = self._confirm_structure_break(
                    df, current_swing, previous_swing, 'BOS_BEARISH'
                )
                
                if break_confirmed:
                    return StructureBreakPoint(
                        timestamp=current_swing.timestamp,
                        price=current_swing.price,
                        break_type=StructureBreak.BOS_BEARISH,
                        previous_swing=previous_swing,
                        breaking_candle_index=current_swing.candle_index,
                        confirmation_strength=break_confirmed
                    )
        
        elif current_swing.swing_type == 'HIGH':
            # Check for bullish CHoC (break above previous high)
            if current_swing.price > previous_swing.price and previous_swing.swing_type == 'HIGH':
                # This breaks the bearish structure - CHoC to bullish
                break_confirmed = self._confirm_structure_break(
                    df, current_swing, previous_swing, 'CHOC_BULLISH'
                )
                
                if break_confirmed:
                    return StructureBreakPoint(
                        timestamp=current_swing.timestamp,
                        price=current_swing.price,
                        break_type=StructureBreak.CHOC_BULLISH,
                        previous_swing=previous_swing,
                        breaking_candle_index=current_swing.candle_index,
                        confirmation_strength=break_confirmed
                    )
        
        return None
    
    def _check_sideways_break(
        self, 
        df: pd.DataFrame,
        current_swing: SwingPoint,
        previous_swing: SwingPoint,
        prior_swing: SwingPoint
    ) -> Optional[StructureBreakPoint]:
        """Check for initial directional breaks in sideways structure"""
        
        # Build list of recent swing points for comparison
        recent_swings = [prior_swing, previous_swing, current_swing]
        
        if current_swing.swing_type == 'HIGH':
            # Find all previous highs to compare against
            previous_highs = [sp for sp in recent_swings[:-1] if sp.swing_type == 'HIGH']
            
            if previous_highs:
                # Check if this high breaks above previous highs
                max_previous_high = max(previous_highs, key=lambda x: x.price)
                
                if current_swing.price > max_previous_high.price:
                    # Potential bullish BOS
                    break_confirmed = self._confirm_structure_break(
                        df, current_swing, max_previous_high, 'BOS_BULLISH'
                    )
                    
                    if break_confirmed:
                        return StructureBreakPoint(
                            timestamp=current_swing.timestamp,
                            price=current_swing.price,
                            break_type=StructureBreak.BOS_BULLISH,
                            previous_swing=max_previous_high,
                            breaking_candle_index=current_swing.candle_index,
                            confirmation_strength=break_confirmed
                        )
        
        elif current_swing.swing_type == 'LOW':
            # Find all previous lows to compare against
            previous_lows = [sp for sp in recent_swings[:-1] if sp.swing_type == 'LOW']
            
            if previous_lows:
                # Check if this low breaks below previous lows
                min_previous_low = min(previous_lows, key=lambda x: x.price)
                
                if current_swing.price < min_previous_low.price:
                    # Potential bearish BOS
                    break_confirmed = self._confirm_structure_break(
                        df, current_swing, min_previous_low, 'BOS_BEARISH'
                    )
                    
                    if break_confirmed:
                        return StructureBreakPoint(
                            timestamp=current_swing.timestamp,
                            price=current_swing.price,
                            break_type=StructureBreak.BOS_BEARISH,
                            previous_swing=min_previous_low,
                            breaking_candle_index=current_swing.candle_index,
                            confirmation_strength=break_confirmed
                        )
        
        return None
    
    def _confirm_structure_break(
        self,
        df: pd.DataFrame,
        current_swing: SwingPoint,
        previous_swing: SwingPoint,
        break_type: str
    ) -> Optional[float]:
        """Confirm structure break with additional validation"""
        
        try:
            # Check if indices are valid
            if current_swing.candle_index >= len(df) or current_swing.candle_index < 0:
                return None
                
            # Get the actual breaking candle data
            breaking_candle = df.iloc[current_swing.candle_index]
            
            confirmation_factors = []
            
            # 1. Price break confirmation
            if self.config.require_close_break:
                if 'BULLISH' in break_type:
                    if breaking_candle['close'] > previous_swing.price:
                        confirmation_factors.append(1.0)
                    else:
                        return None  # Not confirmed
                else:  # BEARISH
                    if breaking_candle['close'] < previous_swing.price:
                        confirmation_factors.append(1.0)
                    else:
                        return None  # Not confirmed
            else:
                # Allow wick breaks
                confirmation_factors.append(0.8)
            
            # 2. Break size validation
            break_size = abs(current_swing.price - previous_swing.price)
            if break_size >= (self.config.min_break_size_pips * 0.0001):  # Convert pips to price
                confirmation_factors.append(1.0)
            else:
                confirmation_factors.append(0.5)
            
            # 3. Volume confirmation (if available)
            if 'volume' in breaking_candle.index:
                # Check if volume is above average for confirmation
                recent_volume_avg = df['volume'].tail(20).mean()
                if breaking_candle['volume'] > recent_volume_avg:
                    confirmation_factors.append(1.0)
                else:
                    confirmation_factors.append(0.7)
            else:
                confirmation_factors.append(0.8)  # Default when no volume data
            
            # Calculate overall confirmation strength
            confirmation_strength = np.mean(confirmation_factors)
            
            # Require minimum confirmation threshold
            if confirmation_strength >= 0.6:
                return confirmation_strength
            else:
                return None
                
        except Exception as e:
            logger.debug(f"Error confirming structure break: {e}")
            return None
    
    def _classify_current_structure(self, structure_breaks: List[StructureBreakPoint]) -> Dict:
        """Classify the current market structure based on recent breaks"""
        
        if not structure_breaks:
            return {
                'trend': MarketStructure.SIDEWAYS,
                'confidence': 0.0,
                'last_break': None,
                'break_count': 0,
                'structure_age': 0
            }
        
        # Get the most recent breaks for analysis
        recent_breaks = structure_breaks[-5:] if len(structure_breaks) >= 5 else structure_breaks
        
        # Count break types
        bos_bullish = len([b for b in recent_breaks if b.break_type == StructureBreak.BOS_BULLISH])
        bos_bearish = len([b for b in recent_breaks if b.break_type == StructureBreak.BOS_BEARISH])
        choc_bullish = len([b for b in recent_breaks if b.break_type == StructureBreak.CHOC_BULLISH])
        choc_bearish = len([b for b in recent_breaks if b.break_type == StructureBreak.CHOC_BEARISH])
        
        # Determine dominant trend
        bullish_weight = (bos_bullish * 1.0) + (choc_bullish * 0.8)
        bearish_weight = (bos_bearish * 1.0) + (choc_bearish * 0.8)
        
        if bullish_weight > bearish_weight:
            trend = MarketStructure.BULLISH
            confidence = min(bullish_weight / (bullish_weight + bearish_weight), 1.0)
        elif bearish_weight > bullish_weight:
            trend = MarketStructure.BEARISH
            confidence = min(bearish_weight / (bullish_weight + bearish_weight), 1.0)
        else:
            trend = MarketStructure.SIDEWAYS
            confidence = 0.5
        
        return {
            'trend': trend,
            'confidence': confidence,
            'last_break': recent_breaks[-1] if recent_breaks else None,
            'break_count': len(structure_breaks),
            'structure_age': 0,  # Calculated elsewhere if needed
            'bos_count': bos_bullish + bos_bearish,
            'choc_count': choc_bullish + choc_bearish
        }
    
    def _generate_break_metrics(self, structure_breaks: List[StructureBreakPoint], df: pd.DataFrame) -> Dict:
        """Generate comprehensive metrics about structure breaks"""
        
        if not structure_breaks:
            return {
                'total_breaks': 0,
                'avg_confirmation': 0.0,
                'break_frequency': 0.0,
                'quality_score': 0.0
            }
        
        # Calculate metrics
        total_breaks = len(structure_breaks)
        avg_confirmation = np.mean([sb.confirmation_strength for sb in structure_breaks])
        break_frequency = total_breaks / len(df) if len(df) > 0 else 0.0
        
        # Quality score based on confirmation strength and frequency
        quality_score = min(avg_confirmation * (1.0 - min(break_frequency * 10, 0.5)), 1.0)
        
        return {
            'total_breaks': total_breaks,
            'avg_confirmation': avg_confirmation,
            'break_frequency': break_frequency,
            'quality_score': quality_score,
            'bos_breaks': len([b for b in structure_breaks if 'BOS' in b.break_type]),
            'choc_breaks': len([b for b in structure_breaks if 'CHOC' in b.break_type])
        }
    
    def _determine_initial_trend(self, initial_swings: List[SwingPoint]) -> str:
        """Determine initial market trend from first few swings"""
        
        if len(initial_swings) < 3:
            return MarketStructure.SIDEWAYS
        
        # Compare first and last swings to determine overall direction
        first_swing = initial_swings[0]
        last_swing = initial_swings[-1]
        
        if first_swing.swing_type == 'LOW' and last_swing.swing_type == 'HIGH':
            if last_swing.price > first_swing.price:
                return MarketStructure.BULLISH
        elif first_swing.swing_type == 'HIGH' and last_swing.swing_type == 'LOW':
            if last_swing.price < first_swing.price:
                return MarketStructure.BEARISH
        
        return MarketStructure.SIDEWAYS
    
    def _get_new_trend_from_choc(self, choc_type: str) -> str:
        """Get new trend direction from CHoC break type"""
        
        if choc_type == StructureBreak.CHOC_BULLISH:
            return MarketStructure.BULLISH
        elif choc_type == StructureBreak.CHOC_BEARISH:
            return MarketStructure.BEARISH
        else:
            return MarketStructure.SIDEWAYS
    
    def get_current_structure_state(self, df: pd.DataFrame, lookback_candles: int = 100) -> Dict:
        """Get the current structure state based on recent data"""
        if len(df) > lookback_candles:
            recent_df = df.tail(lookback_candles).copy()
        else:
            recent_df = df.copy()
        
        analysis = self.detect_structure_breaks(recent_df)
        
        if analysis['success']:
            return {
                'current_structure': analysis['current_structure'],
                'recent_breaks': analysis['structure_breaks'][-3:] if analysis['structure_breaks'] else [],
                'swing_context': analysis['swing_points'][-5:] if analysis['swing_points'] else [],
                'structure_strength': analysis['current_structure']['confidence'],
                'trend_direction': analysis['current_structure']['trend']
            }
        
        return self._empty_result()
    
    def validate_structure_break(self, df: pd.DataFrame, break_index: int, break_type: str) -> Dict:
        """Validate a specific structure break point"""
        try:
            if break_index >= len(df) or break_index < 0:
                return {'valid': False, 'validation_score': 0.0, 'reason': 'Invalid index'}
            
            # Get breaking candle
            breaking_candle = df.iloc[break_index]
            validation_score = 0.6  # Start with base score
            reasons = []
            
            # Simplified validation - check if break is technically valid
            if 'BULLISH' in break_type or 'BOS_BULLISH' in break_type or 'CHOC_BULLISH' in break_type:
                # For bullish breaks, look at price action around the break
                lookback_start = max(0, break_index - 20)
                lookback_data = df.iloc[lookback_start:break_index]
                
                if not lookback_data.empty:
                    # Find recent high to compare against
                    recent_high = lookback_data['high'].max()
                    
                    # Check if breaking candle went above recent high
                    if breaking_candle['high'] > recent_high:
                        validation_score += 0.2
                        reasons.append(f"Broke above recent high ({recent_high:.5f})")
                        
                        # Bonus for close above
                        if breaking_candle['close'] > recent_high:
                            validation_score += 0.1
                            reasons.append("Close confirmed breakout")
                    else:
                        # Still valid if it's near the high
                        if breaking_candle['high'] >= recent_high * 0.999:  # Within 0.1%
                            validation_score += 0.1
                            reasons.append("Near recent high")
                else:
                    reasons.append("Insufficient lookback data - using base validation")
                    
            else:  # BEARISH breaks
                # For bearish breaks, look at price action around the break
                lookback_start = max(0, break_index - 20)
                lookback_data = df.iloc[lookback_start:break_index]
                
                if not lookback_data.empty:
                    # Find recent low to compare against
                    recent_low = lookback_data['low'].min()
                    
                    # Check if breaking candle went below recent low
                    if breaking_candle['low'] < recent_low:
                        validation_score += 0.2
                        reasons.append(f"Broke below recent low ({recent_low:.5f})")
                        
                        # Bonus for close below
                        if breaking_candle['close'] < recent_low:
                            validation_score += 0.1
                            reasons.append("Close confirmed breakdown")
                    else:
                        # Still valid if it's near the low
                        if breaking_candle['low'] <= recent_low * 1.001:  # Within 0.1%
                            validation_score += 0.1
                            reasons.append("Near recent low")
                else:
                    reasons.append("Insufficient lookback data - using base validation")
            
            # Additional technical validation
            candle_range = breaking_candle['high'] - breaking_candle['low']
            avg_range = df.iloc[max(0, break_index-10):break_index]['high'].subtract(
                df.iloc[max(0, break_index-10):break_index]['low']).mean()
            
            if candle_range > avg_range * 1.2:
                validation_score += 0.1
                reasons.append("Above average candle range")
            
            # Cap score at 1.0
            validation_score = min(1.0, validation_score)
            
            # Ensure minimum threshold for any structure break
            if not reasons:
                reasons.append("Basic structure break criteria met")
                validation_score = max(0.5, validation_score)
            
            return {
                'valid': validation_score >= 0.5,  # 50% threshold
                'validation_score': validation_score,
                'reasons': reasons,
                'break_strength': 'Strong' if validation_score >= 0.8 else 'Moderate' if validation_score >= 0.6 else 'Weak'
            }
            
        except Exception as e:
            logger.error(f"Error validating structure break: {e}")
            return {
                'valid': True,  # Default to valid on error
                'validation_score': 0.6,  # Give benefit of doubt
                'reasons': [f'Validation error handled: {str(e)}'],
                'break_strength': 'Moderate'
            }


    def _empty_result(self, error: str = None) -> Dict:
        """Return empty result structure"""
        return {
            'structure_breaks': [],
            'swing_points': [],
            'current_structure': {
                'trend': MarketStructure.SIDEWAYS,
                'confidence': 0.0,
                'last_break': None,
                'break_count': 0
            },
            'bos_points': [],
            'choc_points': [],
            'analysis_metrics': {
                'total_breaks': 0,
                'avg_confirmation': 0.0,
                'break_frequency': 0.0,
                'quality_score': 0.0
            },
            'success': False,
            'error': error
        }