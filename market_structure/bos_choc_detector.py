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
                # This is a higher high - potential bullish BOS
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
            # Check for potential CHoC (break below previous low)
            if current_swing.price < previous_swing.price and previous_swing.swing_type == 'LOW':
                # This breaks the higher low pattern - potential bearish CHoC
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
                # This is a lower low - potential bearish BOS
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
            # Check for potential CHoC (break above previous high)
            if current_swing.price > previous_swing.price and previous_swing.swing_type == 'HIGH':
                # This breaks the lower high pattern - potential bullish CHoC
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
        """Check for initial directional break from sideways movement"""
        
        # Look for significant breaks that establish trend direction
        if current_swing.swing_type != previous_swing.swing_type:
            # Different swing types - potential structure formation
            
            if (current_swing.swing_type == 'HIGH' and 
                current_swing.price > max(p.price for p in [previous_swing, prior_swing] if p.swing_type == 'HIGH')):
                
                # Potential bullish structure start
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
            
            elif (current_swing.swing_type == 'LOW' and 
                  current_swing.price < min(p.price for p in [previous_swing, prior_swing] if p.swing_type == 'LOW')):
                
                # Potential bearish structure start
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
            if 'volume' in df.columns:
                avg_range = df['high'].rolling(20).mean() - df['low'].rolling(20).mean()
                relative_size = break_size / avg_range.iloc[current_swing.candle_index]
                size_strength = min(1.0, relative_size / 0.5)  # Normalize
                confirmation_factors.append(size_strength)
            
            # 3. Volume confirmation (if available)
            if 'volume' in df.columns:
                current_volume = breaking_candle['volume']
                avg_volume = df['volume'].rolling(20).mean().iloc[current_swing.candle_index]
                if pd.notna(avg_volume) and avg_volume > 0:
                    volume_strength = min(2.0, current_volume / avg_volume) / 2.0
                    confirmation_factors.append(volume_strength)
            
            # 4. Candle structure strength
            body_size = abs(breaking_candle['close'] - breaking_candle['open'])
            total_range = breaking_candle['high'] - breaking_candle['low']
            
            if total_range > 0:
                body_strength = body_size / total_range
                confirmation_factors.append(body_strength)
            
            # Calculate final confirmation strength
            if confirmation_factors:
                final_strength = np.mean(confirmation_factors)
                return max(0.0, min(1.0, final_strength))
            
            return 0.5  # Default moderate confirmation
            
        except Exception as e:
            logger.debug(f"Error confirming structure break: {e}")
            return None
    
    def _determine_initial_trend(self, swing_points: List[SwingPoint]) -> str:
        """Determine initial trend from first few swing points"""
        if len(swing_points) < 3:
            return MarketStructure.SIDEWAYS
        
        try:
            # Look at first few swings to determine trend
            highs = [sp for sp in swing_points[:3] if sp.swing_type == 'HIGH']
            lows = [sp for sp in swing_points[:3] if sp.swing_type == 'LOW']
            
            if len(highs) >= 2 and len(lows) >= 2:
                # Check if we have higher highs and higher lows
                if (highs[-1].price > highs[0].price and 
                    lows[-1].price > lows[0].price):
                    return MarketStructure.BULLISH
                
                # Check if we have lower highs and lower lows
                elif (highs[-1].price < highs[0].price and 
                      lows[-1].price < lows[0].price):
                    return MarketStructure.BEARISH
            
            return MarketStructure.SIDEWAYS
            
        except Exception as e:
            logger.debug(f"Error determining initial trend: {e}")
            return MarketStructure.SIDEWAYS
    
    def _get_new_trend_from_choc(self, choc_type: str) -> str:
        """Get new trend direction from CHoC type"""
        if choc_type == StructureBreak.CHOC_BULLISH:
            return MarketStructure.BULLISH
        elif choc_type == StructureBreak.CHOC_BEARISH:
            return MarketStructure.BEARISH
        else:
            return MarketStructure.SIDEWAYS
    
    def _classify_current_structure(self, structure_breaks: List[StructureBreakPoint]) -> Dict:
        """Classify the current market structure based on recent breaks"""
        if not structure_breaks:
            return {
                'trend': MarketStructure.SIDEWAYS,
                'confidence': 0.0,
                'last_break': None,
                'breaks_in_direction': 0
            }
        
        # Get the most recent structure break
        latest_break = max(structure_breaks, key=lambda x: x.timestamp)
        
        # Determine current trend from latest break
        if 'BULLISH' in latest_break.break_type:
            current_trend = MarketStructure.BULLISH
        elif 'BEARISH' in latest_break.break_type:
            current_trend = MarketStructure.BEARISH
        else:
            current_trend = MarketStructure.SIDEWAYS
        
        # Count consecutive breaks in same direction
        breaks_in_direction = 1
        for i in range(len(structure_breaks) - 2, -1, -1):
            break_point = structure_breaks[i]
            
            if ((current_trend == MarketStructure.BULLISH and 'BULLISH' in break_point.break_type) or
                (current_trend == MarketStructure.BEARISH and 'BEARISH' in break_point.break_type)):
                breaks_in_direction += 1
            else:
                break
        
        # Calculate confidence based on multiple factors
        confidence_factors = []
        
        # 1. Strength of latest break
        confidence_factors.append(latest_break.confirmation_strength)
        
        # 2. Number of consecutive breaks
        consecutive_strength = min(1.0, breaks_in_direction / 3.0)
        confidence_factors.append(consecutive_strength)
        
        # 3. Recency of break (more recent = higher confidence)
        # This would need timestamp comparison with current time
        confidence_factors.append(0.8)  # Default high recency
        
        final_confidence = np.mean(confidence_factors)
        
        return {
            'trend': current_trend,
            'confidence': final_confidence,
            'last_break': latest_break,
            'breaks_in_direction': breaks_in_direction,
            'break_type': latest_break.break_type
        }
    
    def _generate_break_metrics(self, structure_breaks: List[StructureBreakPoint], df: pd.DataFrame) -> Dict:
        """Generate comprehensive structure break analysis metrics"""
        if not structure_breaks:
            return {
                'total_breaks': 0,
                'bos_count': 0,
                'choc_count': 0,
                'avg_confirmation_strength': 0
            }
        
        bos_breaks = [sb for sb in structure_breaks if 'BOS' in sb.break_type]
        choc_breaks = [sb for sb in structure_breaks if 'CHOC' in sb.break_type]
        bullish_breaks = [sb for sb in structure_breaks if 'BULLISH' in sb.break_type]
        bearish_breaks = [sb for sb in structure_breaks if 'BEARISH' in sb.break_type]
        
        metrics = {
            'total_breaks': len(structure_breaks),
            'bos_count': len(bos_breaks),
            'choc_count': len(choc_breaks),
            'bullish_breaks': len(bullish_breaks),
            'bearish_breaks': len(bearish_breaks),
            'avg_confirmation_strength': np.mean([sb.confirmation_strength for sb in structure_breaks]),
            'break_frequency': len(structure_breaks) / len(df) * 100,  # breaks per 100 candles
            'strongest_break': max(structure_breaks, key=lambda x: x.confirmation_strength) if structure_breaks else None,
            'latest_break': max(structure_breaks, key=lambda x: x.timestamp) if structure_breaks else None
        }
        
        # Calculate average time between breaks
        if len(structure_breaks) >= 2:
            sorted_breaks = sorted(structure_breaks, key=lambda x: x.timestamp)
            time_diffs = []
            for i in range(1, len(sorted_breaks)):
                # Calculate time difference in terms of candle indices
                time_diff = sorted_breaks[i].breaking_candle_index - sorted_breaks[i-1].breaking_candle_index
                time_diffs.append(time_diff)
            
            metrics['avg_time_between_breaks'] = np.mean(time_diffs) if time_diffs else 0
            metrics['min_time_between_breaks'] = np.min(time_diffs) if time_diffs else 0
            metrics['max_time_between_breaks'] = np.max(time_diffs) if time_diffs else 0
        
        return metrics
    
    def _empty_result(self, error: str = None) -> Dict:
        """Return empty result structure"""
        return {
            'structure_breaks': [],
            'swing_points': [],
            'current_structure': {
                'trend': MarketStructure.UNKNOWN,
                'confidence': 0.0,
                'last_break': None,
                'breaks_in_direction': 0
            },
            'bos_points': [],
            'choc_points': [],
            'analysis_metrics': {'total_breaks': 0, 'error': error},
            'success': False,
            'symbol': 'UNKNOWN',
            'candles_analyzed': 0
        }
    
    def get_latest_structure_state(self, df: pd.DataFrame, lookback_candles: int = 50) -> Dict:
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
                return {'valid': False, 'reason': 'Invalid index'}
            
            # Get swing analysis for context
            swing_analysis = self.swing_analyzer.analyze_swings(df)
            
            if not swing_analysis['success']:
                return {'valid': False, 'reason': 'Could not analyze swings'}
            
            # Find relevant swing points around the break
            relevant_swings = [
                sp for sp in swing_analysis['swing_points']
                if abs(sp.candle_index - break_index) <= 20
            ]
            
            if len(relevant_swings) < 2:
                return {'valid': False, 'reason': 'Insufficient swing context'}
            
            # Validate the break based on SMC rules
            validation_score = 0.0
            validation_factors = []
            
            # Check if break follows SMC structure rules
            candle = df.iloc[break_index]
            
            # Add validation logic here based on break_type
            if 'BOS' in break_type:
                # BOS should extend previous trend
                validation_factors.append(0.8)  # Placeholder
            elif 'CHOC' in break_type:
                # CHoC should reverse previous trend
                validation_factors.append(0.9)  # Placeholder
            
            validation_score = np.mean(validation_factors) if validation_factors else 0.5
            
            return {
                'valid': validation_score >= 0.6,
                'validation_score': validation_score,
                'break_strength': validation_score,
                'surrounding_swings': len(relevant_swings),
                'candle_data': {
                    'timestamp': df.index[break_index],
                    'open': candle['open'],
                    'high': candle['high'],
                    'low': candle['low'],
                    'close': candle['close']
                }
            }
            
        except Exception as e:
            logger.error(f"Structure break validation error: {e}")
            return {'valid': False, 'reason': f'Validation error: {str(e)}'}

# Factory function
def create_bos_choc_detector(require_close_break: bool = True) -> BOSCHOCDetector:
    """Factory function to create BOS/CHoC detector"""
    config = BOSCHOCConfig(require_close_break=require_close_break)
    return BOSCHOCDetector(config)