# market_structure/structure_detector.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

from utils.constants import MarketStructure, StructureBreak
from .swing_analyzer import SwingPoint, SwingAnalyzer
from .bos_choc_detector import StructureBreakPoint, BOSCHOCDetector

class MarketStructureState(NamedTuple):
    """Represents the overall market structure state"""
    primary_trend: str
    secondary_trend: str  # Counter-trend or internal structure
    trend_strength: float  # 0.0 to 1.0
    structure_quality: float  # Overall structure clarity
    last_major_break: Optional[StructureBreakPoint]
    swing_count: int
    structure_age: int  # Candles since last structure change

@dataclass
class StructureDetectionConfig:
    """Configuration for structure detection"""
    min_swing_count: int = 4  # Minimum swings needed for structure
    trend_confirmation_breaks: int = 2  # Breaks needed to confirm trend
    structure_timeout_candles: int = 100  # Max age before structure is stale
    quality_threshold: float = 0.6  # Minimum quality for reliable structure
    use_multi_timeframe: bool = True  # Consider multiple timeframes

class StructureDetector:
    """
    Professional market structure detector that combines swing analysis
    and BOS/CHoC detection to provide comprehensive structure assessment
    """
    
    def __init__(self, config: StructureDetectionConfig = None):
        self.config = config or StructureDetectionConfig()
        self.swing_analyzer = SwingAnalyzer()
        self.bos_choc_detector = BOSCHOCDetector()
        logger.info("Structure Detector initialized")
    
    def detect_market_structure(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> Dict:
        """
        Comprehensive market structure detection
        Returns complete structure analysis with quality assessment
        """
        if df.empty:
            logger.warning("Empty DataFrame passed to structure detector")
            return self._empty_result()
        
        logger.info(f"Detecting market structure for {symbol} - {len(df)} candles")
        
        try:
            # Get comprehensive structure analysis
            structure_analysis = self.bos_choc_detector.detect_structure_breaks(df, symbol)
            
            if not structure_analysis['success']:
                logger.warning(f"BOS/CHoC detection failed for {symbol}")
                return self._empty_result()
            
            # Analyze structure quality and reliability
            structure_state = self._analyze_structure_state(
                structure_analysis['swing_points'],
                structure_analysis['structure_breaks'],
                structure_analysis['current_structure'],
                df
            )
            
            # Detect internal structure patterns
            internal_structure = self._detect_internal_structure(
                structure_analysis['swing_points'],
                structure_analysis['current_structure']
            )
            
            # Calculate structure reliability metrics
            reliability_metrics = self._calculate_reliability_metrics(
                structure_state, structure_analysis, df
            )
            
            # Generate trading context
            trading_context = self._generate_trading_context(
                structure_state, internal_structure, reliability_metrics
            )
            
            logger.info(f"Structure detection completed for {symbol}")
            
            return {
                'structure_state': structure_state,
                'internal_structure': internal_structure,
                'reliability_metrics': reliability_metrics,
                'trading_context': trading_context,
                'raw_analysis': structure_analysis,
                'success': True,
                'symbol': symbol,
                'candles_analyzed': len(df)
            }
            
        except Exception as e:
            logger.error(f"Structure detection failed for {symbol}: {e}")
            return self._empty_result(error=str(e))
    
    def _analyze_structure_state(
        self,
        swing_points: List[SwingPoint],
        structure_breaks: List[StructureBreakPoint],
        current_structure: Dict,
        df: pd.DataFrame
    ) -> MarketStructureState:
        """Analyze the overall market structure state"""
        
        try:
            # Primary trend from current structure
            primary_trend = current_structure.get('trend', MarketStructure.UNKNOWN)
            
            # Detect secondary/internal trend
            secondary_trend = self._detect_secondary_trend(swing_points, primary_trend)
            
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(
                swing_points, structure_breaks, current_structure
            )
            
            # Calculate structure quality
            structure_quality = self._calculate_structure_quality(
                swing_points, structure_breaks, current_structure
            )
            
            # Get last major break
            last_major_break = None
            if structure_breaks:
                # Find the most recent significant break
                major_breaks = [sb for sb in structure_breaks if sb.confirmation_strength >= 0.7]
                if major_breaks:
                    last_major_break = max(major_breaks, key=lambda x: x.timestamp)
            
            # Calculate structure age
            structure_age = 0
            if last_major_break:
                structure_age = len(df) - last_major_break.breaking_candle_index - 1
            
            return MarketStructureState(
                primary_trend=primary_trend,
                secondary_trend=secondary_trend,
                trend_strength=trend_strength,
                structure_quality=structure_quality,
                last_major_break=last_major_break,
                swing_count=len(swing_points),
                structure_age=structure_age
            )
            
        except Exception as e:
            logger.error(f"Error analyzing structure state: {e}")
            return MarketStructureState(
                primary_trend=MarketStructure.UNKNOWN,
                secondary_trend=MarketStructure.UNKNOWN,
                trend_strength=0.0,
                structure_quality=0.0,
                last_major_break=None,
                swing_count=0,
                structure_age=0
            )
    
    def _detect_secondary_trend(self, swing_points: List[SwingPoint], primary_trend: str) -> str:
        """Detect secondary or counter-trend within primary structure"""
        
        if len(swing_points) < 6:
            return MarketStructure.UNKNOWN
        
        try:
            # Look at recent swing points for internal structure
            recent_swings = swing_points[-6:]
            
            # Analyze recent swing progression
            highs = [sp for sp in recent_swings if sp.swing_type == 'HIGH']
            lows = [sp for sp in recent_swings if sp.swing_type == 'LOW']
            
            if len(highs) >= 2 and len(lows) >= 2:
                # Check internal trend direction
                recent_high_trend = 'RISING' if highs[-1].price > highs[0].price else 'FALLING'
                recent_low_trend = 'RISING' if lows[-1].price > lows[0].price else 'FALLING'
                
                # Determine secondary trend
                if recent_high_trend == 'RISING' and recent_low_trend == 'RISING':
                    secondary_trend = MarketStructure.BULLISH
                elif recent_high_trend == 'FALLING' and recent_low_trend == 'FALLING':
                    secondary_trend = MarketStructure.BEARISH
                else:
                    secondary_trend = MarketStructure.SIDEWAYS
                
                # If secondary contradicts primary, it might be a correction
                if secondary_trend != primary_trend and primary_trend != MarketStructure.UNKNOWN:
                    return f"CORRECTION_{secondary_trend}"
                
                return secondary_trend
            
            return MarketStructure.UNKNOWN
            
        except Exception as e:
            logger.debug(f"Error detecting secondary trend: {e}")
            return MarketStructure.UNKNOWN
    
    def _calculate_trend_strength(
        self, 
        swing_points: List[SwingPoint], 
        structure_breaks: List[StructureBreakPoint],
        current_structure: Dict
    ) -> float:
        """Calculate the strength of the current trend"""
        
        try:
            strength_factors = []
            
            # 1. Structure break confirmation strength
            if structure_breaks:
                recent_breaks = structure_breaks[-3:]  # Last 3 breaks
                avg_confirmation = np.mean([sb.confirmation_strength for sb in recent_breaks])
                strength_factors.append(avg_confirmation)
            
            # 2. Swing progression consistency
            if len(swing_points) >= 4:
                trend = current_structure.get('trend', MarketStructure.UNKNOWN)
                
                if trend == MarketStructure.BULLISH:
                    # Check for higher highs and higher lows consistency
                    highs = [sp for sp in swing_points[-6:] if sp.swing_type == 'HIGH']
                    lows = [sp for sp in swing_points[-6:] if sp.swing_type == 'LOW']
                    
                    if len(highs) >= 2:
                        higher_highs = sum(1 for i in range(1, len(highs)) if highs[i].price > highs[i-1].price)
                        hh_consistency = higher_highs / max(1, len(highs) - 1)
                        strength_factors.append(hh_consistency)
                    
                    if len(lows) >= 2:
                        higher_lows = sum(1 for i in range(1, len(lows)) if lows[i].price > lows[i-1].price)
                        hl_consistency = higher_lows / max(1, len(lows) - 1)
                        strength_factors.append(hl_consistency)
                
                elif trend == MarketStructure.BEARISH:
                    # Check for lower highs and lower lows consistency
                    highs = [sp for sp in swing_points[-6:] if sp.swing_type == 'HIGH']
                    lows = [sp for sp in swing_points[-6:] if sp.swing_type == 'LOW']
                    
                    if len(highs) >= 2:
                        lower_highs = sum(1 for i in range(1, len(highs)) if highs[i].price < highs[i-1].price)
                        lh_consistency = lower_highs / max(1, len(highs) - 1)
                        strength_factors.append(lh_consistency)
                    
                    if len(lows) >= 2:
                        lower_lows = sum(1 for i in range(1, len(lows)) if lows[i].price < lows[i-1].price)
                        ll_consistency = lower_lows / max(1, len(lows) - 1)
                        strength_factors.append(ll_consistency)
            
            # 3. Current structure confidence
            confidence = current_structure.get('confidence', 0.0)
            strength_factors.append(confidence)
            
            # 4. Consecutive breaks in same direction
            breaks_in_direction = current_structure.get('breaks_in_direction', 0)
            directional_strength = min(1.0, breaks_in_direction / 3.0)
            strength_factors.append(directional_strength)
            
            return np.mean(strength_factors) if strength_factors else 0.0
            
        except Exception as e:
            logger.debug(f"Error calculating trend strength: {e}")
            return 0.0
    
    def _calculate_structure_quality(
        self,
        swing_points: List[SwingPoint],
        structure_breaks: List[StructureBreakPoint], 
        current_structure: Dict
    ) -> float:
        """Calculate overall structure quality and clarity"""
        
        try:
            quality_factors = []
            
            # 1. Swing point quality
            if swing_points:
                avg_swing_strength = np.mean([sp.strength for sp in swing_points])
                quality_factors.append(avg_swing_strength)
                
                # Confirmed swings ratio
                confirmed_ratio = sum(1 for sp in swing_points if sp.confirmed) / len(swing_points)
                quality_factors.append(confirmed_ratio)
            
            # 2. Structure break quality
            if structure_breaks:
                avg_break_strength = np.mean([sb.confirmation_strength for sb in structure_breaks])
                quality_factors.append(avg_break_strength)
            
            # 3. Structure consistency
            trend = current_structure.get('trend', MarketStructure.UNKNOWN)
            if trend != MarketStructure.UNKNOWN:
                # Structure has clear direction
                quality_factors.append(0.8)
                
                # Confidence level
                confidence = current_structure.get('confidence', 0.0)
                quality_factors.append(confidence)
            else:
                quality_factors.append(0.2)  # Unknown trend reduces quality
            
            # 4. Swing density (not too sparse, not too dense)
            if swing_points:
                # Optimal swing density is around 2-4 per 100 candles
                swing_density = len(swing_points) / 100  # Assuming 100 candles context
                optimal_density = 3.0
                density_score = 1.0 - min(1.0, abs(swing_density - optimal_density) / optimal_density)
                quality_factors.append(density_score)
            
            return np.mean(quality_factors) if quality_factors else 0.0
            
        except Exception as e:
            logger.debug(f"Error calculating structure quality: {e}")
            return 0.0
    
    def _detect_internal_structure(
        self, 
        swing_points: List[SwingPoint], 
        current_structure: Dict
    ) -> Dict:
        """Detect internal structure patterns within the major trend"""
        
        try:
            internal_patterns = {
                'pullback_active': False,
                'continuation_pattern': None,
                'correction_depth': 0.0,
                'internal_swings': [],
                'micro_structure': MarketStructure.UNKNOWN
            }
            
            if len(swing_points) < 4:
                return internal_patterns
            
            # Get recent swing points for internal analysis
            recent_swings = swing_points[-8:]  # Last 8 swings
            primary_trend = current_structure.get('trend', MarketStructure.UNKNOWN)
            
            # Detect active pullback
            internal_patterns['pullback_active'] = self._detect_active_pullback(
                recent_swings, primary_trend
            )
            
            # Identify continuation patterns
            internal_patterns['continuation_pattern'] = self._identify_continuation_pattern(
                recent_swings, primary_trend
            )
            
            # Calculate correction depth
            internal_patterns['correction_depth'] = self._calculate_correction_depth(
                recent_swings, primary_trend
            )
            
            # Filter internal swings (smaller degree swings)
            internal_patterns['internal_swings'] = self._filter_internal_swings(recent_swings)
            
            # Determine micro structure
            internal_patterns['micro_structure'] = self._determine_micro_structure(
                recent_swings[-4:] if len(recent_swings) >= 4 else recent_swings
            )
            
            return internal_patterns
            
        except Exception as e:
            logger.debug(f"Error detecting internal structure: {e}")
            return {
                'pullback_active': False,
                'continuation_pattern': None,
                'correction_depth': 0.0,
                'internal_swings': [],
                'micro_structure': MarketStructure.UNKNOWN
            }
    
    def _detect_active_pullback(self, swing_points: List[SwingPoint], primary_trend: str) -> bool:
        """Detect if there's an active pullback against the primary trend"""
        
        if len(swing_points) < 3:
            return False
        
        try:
            # Get the last few swings
            recent_swings = swing_points[-3:]
            
            if primary_trend == MarketStructure.BULLISH:
                # In bullish trend, pullback = recent lower high or lower low
                highs = [sp for sp in recent_swings if sp.swing_type == 'HIGH']
                if len(highs) >= 2:
                    return highs[-1].price < highs[-2].price
            
            elif primary_trend == MarketStructure.BEARISH:
                # In bearish trend, pullback = recent higher high or higher low
                lows = [sp for sp in recent_swings if sp.swing_type == 'LOW']
                if len(lows) >= 2:
                    return lows[-1].price > lows[-2].price
            
            return False
            
        except Exception as e:
            logger.debug(f"Error detecting pullback: {e}")
            return False
    
    def _identify_continuation_pattern(self, swing_points: List[SwingPoint], primary_trend: str) -> Optional[str]:
        """Identify continuation patterns within the structure"""
        
        if len(swing_points) < 4:
            return None
        
        try:
            # Look for common continuation patterns
            patterns = []
            
            # Flag pattern detection
            if self._detect_flag_pattern(swing_points, primary_trend):
                patterns.append("FLAG")
            
            # Pennant pattern detection
            if self._detect_pennant_pattern(swing_points, primary_trend):
                patterns.append("PENNANT")
            
            # Triangle pattern detection
            if self._detect_triangle_pattern(swing_points):
                patterns.append("TRIANGLE")
            
            return patterns[0] if patterns else None
            
        except Exception as e:
            logger.debug(f"Error identifying continuation pattern: {e}")
            return None
    
    def _detect_flag_pattern(self, swing_points: List[SwingPoint], primary_trend: str) -> bool:
        """Detect flag continuation pattern"""
        
        if len(swing_points) < 4:
            return False
        
        try:
            # Flag pattern: rectangular consolidation after strong move
            recent_highs = [sp for sp in swing_points[-4:] if sp.swing_type == 'HIGH']
            recent_lows = [sp for sp in swing_points[-4:] if sp.swing_type == 'LOW']
            
            if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                # Check if highs and lows are relatively equal (flag characteristics)
                high_range = max(h.price for h in recent_highs) - min(h.price for h in recent_highs)
                low_range = max(l.price for l in recent_lows) - min(l.price for l in recent_lows)
                
                avg_price = np.mean([sp.price for sp in swing_points[-4:]])
                
                # Flag pattern if ranges are small relative to average price
                if high_range / avg_price < 0.02 and low_range / avg_price < 0.02:
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Error detecting flag pattern: {e}")
            return False
    
    def _detect_pennant_pattern(self, swing_points: List[SwingPoint], primary_trend: str) -> bool:
        """Detect pennant continuation pattern"""
        
        if len(swing_points) < 5:
            return False
        
        try:
            # Pennant: converging highs and lows
            recent_points = swing_points[-5:]
            highs = [sp for sp in recent_points if sp.swing_type == 'HIGH']
            lows = [sp for sp in recent_points if sp.swing_type == 'LOW']
            
            if len(highs) >= 2 and len(lows) >= 2:
                # Check for convergence
                high_trend = 'FALLING' if highs[-1].price < highs[0].price else 'RISING'
                low_trend = 'RISING' if lows[-1].price > lows[0].price else 'FALLING'
                
                # Pennant if highs falling and lows rising (convergence)
                if high_trend == 'FALLING' and low_trend == 'RISING':
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Error detecting pennant pattern: {e}")
            return False
    
    def _detect_triangle_pattern(self, swing_points: List[SwingPoint]) -> bool:
        """Detect triangle continuation pattern"""
        
        if len(swing_points) < 5:
            return False
        
        try:
            # Triangle: series of lower highs and higher lows
            recent_points = swing_points[-6:]
            highs = sorted([sp for sp in recent_points if sp.swing_type == 'HIGH'], key=lambda x: x.timestamp)
            lows = sorted([sp for sp in recent_points if sp.swing_type == 'LOW'], key=lambda x: x.timestamp)
            
            if len(highs) >= 3 and len(lows) >= 3:
                # Check for triangle formation
                descending_highs = all(highs[i].price > highs[i+1].price for i in range(len(highs)-1))
                ascending_lows = all(lows[i].price < lows[i+1].price for i in range(len(lows)-1))
                
                if descending_highs and ascending_lows:
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Error detecting triangle pattern: {e}")
            return False
    
    def _calculate_correction_depth(self, swing_points: List[SwingPoint], primary_trend: str) -> float:
        """Calculate the depth of current correction"""
        
        if len(swing_points) < 3:
            return 0.0
        
        try:
            if primary_trend == MarketStructure.BULLISH:
                # Find last major high and current low
                recent_highs = [sp for sp in swing_points if sp.swing_type == 'HIGH']
                recent_lows = [sp for sp in swing_points if sp.swing_type == 'LOW']
                
                if recent_highs and recent_lows:
                    last_high = max(recent_highs, key=lambda x: x.price)
                    current_low = min([sp for sp in recent_lows if sp.timestamp > last_high.timestamp], 
                                    key=lambda x: x.price, default=recent_lows[-1])
                    
                    correction_depth = (last_high.price - current_low.price) / last_high.price
                    return max(0.0, min(1.0, correction_depth))
            
            elif primary_trend == MarketStructure.BEARISH:
                # Find last major low and current high
                recent_highs = [sp for sp in swing_points if sp.swing_type == 'HIGH']
                recent_lows = [sp for sp in swing_points if sp.swing_type == 'LOW']
                
                if recent_highs and recent_lows:
                    last_low = min(recent_lows, key=lambda x: x.price)
                    current_high = max([sp for sp in recent_highs if sp.timestamp > last_low.timestamp],
                                     key=lambda x: x.price, default=recent_highs[-1])
                    
                    correction_depth = (current_high.price - last_low.price) / last_low.price
                    return max(0.0, min(1.0, correction_depth))
            
            return 0.0
            
        except Exception as e:
            logger.debug(f"Error calculating correction depth: {e}")
            return 0.0
    
    def _filter_internal_swings(self, swing_points: List[SwingPoint]) -> List[SwingPoint]:
        """Filter out internal/minor swings, keeping only significant ones"""
        
        if len(swing_points) <= 4:
            return swing_points
        
        try:
            # Sort by strength and keep top 70%
            sorted_swings = sorted(swing_points, key=lambda x: x.strength, reverse=True)
            keep_count = max(4, int(len(sorted_swings) * 0.7))
            
            significant_swings = sorted_swings[:keep_count]
            
            # Sort back by timestamp
            significant_swings.sort(key=lambda x: x.timestamp)
            
            return significant_swings
            
        except Exception as e:
            logger.debug(f"Error filtering internal swings: {e}")
            return swing_points
    
    def _determine_micro_structure(self, swing_points: List[SwingPoint]) -> str:
        """Determine micro structure from recent swing points"""
        
        if len(swing_points) < 3:
            return MarketStructure.UNKNOWN
        
        try:
            highs = [sp for sp in swing_points if sp.swing_type == 'HIGH']
            lows = [sp for sp in swing_points if sp.swing_type == 'LOW']
            
            if len(highs) >= 2 and len(lows) >= 2:
                # Check micro trend
                higher_highs = highs[-1].price > highs[0].price
                higher_lows = lows[-1].price > lows[0].price
                
                if higher_highs and higher_lows:
                    return MarketStructure.BULLISH
                elif not higher_highs and not higher_lows:
                    return MarketStructure.BEARISH
            
            return MarketStructure.SIDEWAYS
            
        except Exception as e:
            logger.debug(f"Error determining micro structure: {e}")
            return MarketStructure.UNKNOWN
    
    def _calculate_reliability_metrics(
        self, 
        structure_state: MarketStructureState, 
        structure_analysis: Dict,
        df: pd.DataFrame
    ) -> Dict:
        """Calculate structure reliability metrics"""
        
        try:
            metrics = {}
            
            # Structure age factor
            max_age = self.config.structure_timeout_candles
            age_factor = max(0.0, 1.0 - (structure_state.structure_age / max_age))
            metrics['age_reliability'] = age_factor
            
            # Structure quality factor
            metrics['quality_reliability'] = structure_state.structure_quality
            
            # Trend strength factor
            metrics['strength_reliability'] = structure_state.trend_strength
            
            # Data sufficiency factor
            min_candles = 50
            data_sufficiency = min(1.0, len(df) / min_candles)
            metrics['data_reliability'] = data_sufficiency
            
            # Break consistency factor
            breaks = structure_analysis.get('structure_breaks', [])
            if breaks:
                consistent_breaks = 0
                total_breaks = len(breaks)
                primary_trend = structure_state.primary_trend
                
                for break_point in breaks[-5:]:  # Last 5 breaks
                    if ((primary_trend == MarketStructure.BULLISH and 'BULLISH' in break_point.break_type) or
                        (primary_trend == MarketStructure.BEARISH and 'BEARISH' in break_point.break_type)):
                        consistent_breaks += 1
                
                consistency_rate = consistent_breaks / min(5, total_breaks)
                metrics['consistency_reliability'] = consistency_rate
            else:
                metrics['consistency_reliability'] = 0.5
            
            # Overall reliability score
            weights = {
                'age_reliability': 0.15,
                'quality_reliability': 0.25,
                'strength_reliability': 0.25,
                'data_reliability': 0.15,
                'consistency_reliability': 0.20
            }
            
            overall_score = sum(metrics[key] * weight for key, weight in weights.items())
            metrics['overall_reliability'] = overall_score
            
            # Reliability classification
            if overall_score >= 0.8:
                metrics['reliability_class'] = 'HIGH'
            elif overall_score >= 0.6:
                metrics['reliability_class'] = 'MEDIUM'
            else:
                metrics['reliability_class'] = 'LOW'
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating reliability metrics: {e}")
            return {
                'age_reliability': 0.5,
                'quality_reliability': 0.5,
                'strength_reliability': 0.5,
                'data_reliability': 0.5,
                'consistency_reliability': 0.5,
                'overall_reliability': 0.5,
                'reliability_class': 'LOW'
            }
    
    def _generate_trading_context(
        self,
        structure_state: MarketStructureState,
        internal_structure: Dict,
        reliability_metrics: Dict
    ) -> Dict:
        """Generate trading context and recommendations"""
        
        try:
            context = {
                'trade_bias': MarketStructure.UNKNOWN,
                'confidence_level': 'LOW',
                'entry_context': 'WAIT',
                'structure_notes': [],
                'risk_factors': [],
                'opportunities': []
            }
            
            # Determine trade bias
            primary_trend = structure_state.primary_trend
            trend_strength = structure_state.trend_strength
            reliability = reliability_metrics['overall_reliability']
            
            if reliability >= 0.7 and trend_strength >= 0.6:
                context['trade_bias'] = primary_trend
                context['confidence_level'] = 'HIGH'
                context['entry_context'] = 'FAVORABLE'
            elif reliability >= 0.5 and trend_strength >= 0.4:
                context['trade_bias'] = primary_trend  
                context['confidence_level'] = 'MEDIUM'
                context['entry_context'] = 'CAUTIOUS'
            else:
                context['trade_bias'] = MarketStructure.UNKNOWN
                context['confidence_level'] = 'LOW'
                context['entry_context'] = 'WAIT'
            
            # Generate structure notes
            if structure_state.structure_quality >= 0.7:
                context['structure_notes'].append("High quality structure formation")
            
            if internal_structure['pullback_active']:
                context['structure_notes'].append("Active pullback detected - potential entry zone")
            
            if internal_structure['continuation_pattern']:
                pattern = internal_structure['continuation_pattern']
                context['structure_notes'].append(f"Continuation pattern: {pattern}")
            
            # Identify risk factors
            if structure_state.structure_age > self.config.structure_timeout_candles * 0.7:
                context['risk_factors'].append("Structure aging - may be losing relevance")
            
            if structure_state.trend_strength < 0.4:
                context['risk_factors'].append("Weak trend strength - potential reversal risk")
            
            if internal_structure['correction_depth'] > 0.5:
                context['risk_factors'].append("Deep correction - trend weakening signal")
            
            # Identify opportunities
            if (structure_state.trend_strength >= 0.6 and 
                internal_structure['pullback_active']):
                context['opportunities'].append("High probability continuation setup")
            
            if structure_state.last_major_break and structure_state.structure_age < 20:
                context['opportunities'].append("Fresh structure break - momentum opportunity")
            
            if internal_structure['continuation_pattern']:
                context['opportunities'].append("Pattern completion trade setup")
            
            return context
            
        except Exception as e:
            logger.error(f"Error generating trading context: {e}")
            return {
                'trade_bias': MarketStructure.UNKNOWN,
                'confidence_level': 'LOW', 
                'entry_context': 'WAIT',
                'structure_notes': ['Error in analysis'],
                'risk_factors': ['Analysis reliability compromised'],
                'opportunities': []
            }
    
    def _empty_result(self, error: str = None) -> Dict:
        """Return empty result structure"""
        return {
            'structure_state': MarketStructureState(
                primary_trend=MarketStructure.UNKNOWN,
                secondary_trend=MarketStructure.UNKNOWN,
                trend_strength=0.0,
                structure_quality=0.0,
                last_major_break=None,
                swing_count=0,
                structure_age=0
            ),
            'internal_structure': {
                'pullback_active': False,
                'continuation_pattern': None,
                'correction_depth': 0.0,
                'internal_swings': [],
                'micro_structure': MarketStructure.UNKNOWN
            },
            'reliability_metrics': {
                'overall_reliability': 0.0,
                'reliability_class': 'LOW'
            },
            'trading_context': {
                'trade_bias': MarketStructure.UNKNOWN,
                'confidence_level': 'LOW',
                'entry_context': 'WAIT',
                'structure_notes': ['Analysis failed'],
                'risk_factors': ['Insufficient data'],
                'opportunities': []
            },
            'raw_analysis': {},
            'success': False,
            'symbol': 'UNKNOWN',
            'candles_analyzed': 0,
            'error': error
        }
    
    def get_structure_summary(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> Dict:
        """Get concise structure summary for quick analysis"""
        try:
            analysis = self.detect_market_structure(df, symbol)
            
            if not analysis['success']:
                return {'error': 'Structure detection failed'}
            
            structure_state = analysis['structure_state']
            reliability = analysis['reliability_metrics']
            context = analysis['trading_context']
            
            return {
                'symbol': symbol,
                'trend': structure_state.primary_trend,
                'strength': structure_state.trend_strength,
                'quality': structure_state.structure_quality,
                'reliability': reliability['overall_reliability'],
                'trade_bias': context['trade_bias'],
                'confidence': context['confidence_level'],
                'last_update': datetime.now(),
                'actionable': context['entry_context'] in ['FAVORABLE', 'CAUTIOUS']
            }
            
        except Exception as e:
            logger.error(f"Error getting structure summary: {e}")
            return {'error': str(e)}
    
    def validate_structure_integrity(self, df: pd.DataFrame) -> Dict:
        """Validate the integrity and consistency of detected structure"""
        try:
            analysis = self.detect_market_structure(df)
            
            if not analysis['success']:
                return {'valid': False, 'reason': 'Analysis failed'}
            
            validation_results = {
                'valid': True,
                'integrity_score': 0.0,
                'validation_checks': [],
                'warnings': []
            }
            
            structure_state = analysis['structure_state']
            reliability = analysis['reliability_metrics']
            
            # Check 1: Structure quality threshold
            if structure_state.structure_quality >= self.config.quality_threshold:
                validation_results['validation_checks'].append('Quality threshold met')
            else:
                validation_results['warnings'].append('Structure quality below threshold')
            
            # Check 2: Sufficient swing points
            if structure_state.swing_count >= self.config.min_swing_count:
                validation_results['validation_checks'].append('Sufficient swing points')
            else:
                validation_results['warnings'].append('Insufficient swing points for reliable analysis')
            
            # Check 3: Structure age
            if structure_state.structure_age < self.config.structure_timeout_candles:
                validation_results['validation_checks'].append('Structure is current')
            else:
                validation_results['warnings'].append('Structure may be outdated')
            
            # Check 4: Overall reliability
            overall_reliability = reliability['overall_reliability']
            if overall_reliability >= 0.6:
                validation_results['validation_checks'].append('Reliability acceptable')
            else:
                validation_results['warnings'].append('Low overall reliability')
            
            # Calculate integrity score
            integrity_factors = [
                structure_state.structure_quality,
                structure_state.trend_strength,
                min(1.0, structure_state.swing_count / self.config.min_swing_count),
                max(0.0, 1.0 - (structure_state.structure_age / self.config.structure_timeout_candles)),
                overall_reliability
            ]
            
            validation_results['integrity_score'] = np.mean(integrity_factors)
            
            # Final validation
            if (validation_results['integrity_score'] < 0.5 or 
                len(validation_results['warnings']) > len(validation_results['validation_checks'])):
                validation_results['valid'] = False
                validation_results['reason'] = 'Failed integrity checks'
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Structure validation error: {e}")
            return {
                'valid': False,
                'reason': f'Validation error: {str(e)}',
                'integrity_score': 0.0
            }

# Factory function
def create_structure_detector(min_swing_count: int = 4, quality_threshold: float = 0.6) -> StructureDetector:
    """Factory function to create structure detector"""
    config = StructureDetectionConfig(
        min_swing_count=min_swing_count,
        quality_threshold=quality_threshold
    )
    return StructureDetector(config)