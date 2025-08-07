# market_structure/trend_classifier.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, NamedTuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger

from utils.constants import MarketStructure, TrendStrength
from utils.helpers import calculate_atr
from .swing_analyzer import SwingPoint, SwingAnalyzer
from .bos_choc_detector import StructureBreakPoint, BOSCHOCDetector

class TrendClassification(NamedTuple):
    """Represents a complete trend classification"""
    primary_trend: str
    trend_strength: str  # WEAK, MODERATE, STRONG, VERY_STRONG
    confidence: float  # 0.0 to 1.0
    trend_age: int  # Candles since trend started
    trend_momentum: float  # Rate of trend development
    reversal_probability: float  # 0.0 to 1.0
    support_factors: List[str]  # Factors supporting the trend
    warning_signals: List[str]  # Signals suggesting weakness

@dataclass
class TrendClassificationConfig:
    """Configuration for trend classification"""
    trend_confirmation_bars: int = 10  # Bars needed to confirm trend
    momentum_period: int = 20  # Period for momentum calculation
    volatility_period: int = 14  # Period for volatility analysis
    strength_thresholds: Dict[str, float] = None  # Custom strength thresholds
    
    def __post_init__(self):
        if self.strength_thresholds is None:
            self.strength_thresholds = {
                'WEAK': 0.3,
                'MODERATE': 0.5,
                'STRONG': 0.7,
                'VERY_STRONG': 0.85
            }

class TrendClassifier:
    """
    Professional trend classification system for SMC analysis
    Classifies market trends with strength analysis and reversal prediction
    """
    
    def __init__(self, config: TrendClassificationConfig = None):
        self.config = config or TrendClassificationConfig()
        self.swing_analyzer = SwingAnalyzer()
        self.bos_choc_detector = BOSCHOCDetector()
        logger.info("Trend Classifier initialized")
    
    def classify_trend(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> Dict:
        """
        Comprehensive trend classification
        Returns detailed trend analysis with strength and reversal signals
        """
        if df.empty or len(df) < self.config.trend_confirmation_bars:
            logger.warning(f"Insufficient data for trend classification: {len(df)} candles")
            return self._empty_result()
        
        logger.info(f"Classifying trend for {symbol} - {len(df)} candles")
        
        try:
            # Get structure analysis
            structure_analysis = self.bos_choc_detector.detect_structure_breaks(df, symbol)
            
            if not structure_analysis['success']:
                logger.warning(f"Structure analysis failed for trend classification: {symbol}")
                return self._empty_result()
            
            # Perform trend classification
            trend_classification = self._perform_trend_classification(
                df, structure_analysis, symbol
            )
            
            # Analyze trend momentum
            momentum_analysis = self._analyze_trend_momentum(
                df, trend_classification, structure_analysis['swing_points']
            )
            
            # Calculate reversal probability
            reversal_analysis = self._analyze_reversal_probability(
                df, trend_classification, structure_analysis
            )
            
            # Generate trend strength analysis
            strength_analysis = self._analyze_trend_strength(
                df, trend_classification, structure_analysis
            )
            
            # Create comprehensive result
            result = {
                'trend_classification': trend_classification,
                'momentum_analysis': momentum_analysis,
                'reversal_analysis': reversal_analysis,
                'strength_analysis': strength_analysis,
                'structure_context': structure_analysis,
                'success': True,
                'symbol': symbol,
                'candles_analyzed': len(df),
                'classification_time': datetime.now()
            }
            
            logger.info(f"Trend classification completed for {symbol}: {trend_classification.primary_trend}")
            return result
            
        except Exception as e:
            logger.error(f"Trend classification failed for {symbol}: {e}")
            return self._empty_result(error=str(e))
    
    def _perform_trend_classification(
        self, 
        df: pd.DataFrame, 
        structure_analysis: Dict,
        symbol: str
    ) -> TrendClassification:
        """Perform comprehensive trend classification"""
        
        try:
            swing_points = structure_analysis['swing_points']
            structure_breaks = structure_analysis['structure_breaks']
            current_structure = structure_analysis['current_structure']
            
            # Get primary trend from structure analysis
            primary_trend = current_structure.get('trend', MarketStructure.UNKNOWN)
            
            # Calculate trend strength factors
            strength_factors = self._calculate_trend_strength_factors(
                df, swing_points, structure_breaks, primary_trend
            )
            
            # Determine trend strength classification
            trend_strength = self._classify_trend_strength(strength_factors)
            
            # Calculate confidence
            confidence = self._calculate_trend_confidence(
                strength_factors, current_structure, swing_points
            )
            
            # Calculate trend age
            trend_age = self._calculate_trend_age(structure_breaks, df)
            
            # Calculate momentum
            trend_momentum = self._calculate_trend_momentum(df, primary_trend)
            
            # Calculate reversal probability
            reversal_probability = self._calculate_reversal_probability(
                df, swing_points, structure_breaks, primary_trend
            )
            
            # Identify support factors
            support_factors = self._identify_support_factors(
                df, swing_points, structure_breaks, primary_trend
            )
            
            # Identify warning signals
            warning_signals = self._identify_warning_signals(
                df, swing_points, structure_breaks, primary_trend
            )
            
            return TrendClassification(
                primary_trend=primary_trend,
                trend_strength=trend_strength,
                confidence=confidence,
                trend_age=trend_age,
                trend_momentum=trend_momentum,
                reversal_probability=reversal_probability,
                support_factors=support_factors,
                warning_signals=warning_signals
            )
            
        except Exception as e:
            logger.error(f"Error performing trend classification: {e}")
            return TrendClassification(
                primary_trend=MarketStructure.UNKNOWN,
                trend_strength=TrendStrength.WEAK,
                confidence=0.0,
                trend_age=0,
                trend_momentum=0.0,
                reversal_probability=0.5,
                support_factors=[],
                warning_signals=['Classification error']
            )
    
    def _calculate_trend_strength_factors(
        self, 
        df: pd.DataFrame,
        swing_points: List[SwingPoint],
        structure_breaks: List[StructureBreakPoint],
        primary_trend: str
    ) -> Dict[str, float]:
        """Calculate various factors that contribute to trend strength"""
        
        factors = {}
        
        try:
            # 1. Structure break strength
            if structure_breaks:
                recent_breaks = structure_breaks[-3:]
                avg_break_strength = np.mean([sb.confirmation_strength for sb in recent_breaks])
                factors['break_strength'] = avg_break_strength
            else:
                factors['break_strength'] = 0.0
            
            # 2. Swing progression consistency
            if len(swing_points) >= 4:
                consistency = self._calculate_swing_consistency(swing_points, primary_trend)
                factors['swing_consistency'] = consistency
            else:
                factors['swing_consistency'] = 0.0
            
            # 3. Price momentum
            if len(df) >= self.config.momentum_period:
                momentum = self._calculate_price_momentum(df, self.config.momentum_period)
                factors['price_momentum'] = abs(momentum)  # Absolute momentum strength
            else:
                factors['price_momentum'] = 0.0
            
            # 4. Volume confirmation (if available)
            if 'volume' in df.columns:
                volume_trend = self._calculate_volume_trend_alignment(df, primary_trend)
                factors['volume_confirmation'] = volume_trend
            else:
                factors['volume_confirmation'] = 0.5  # Neutral if no volume data
            
            # 5. Volatility factor
            atr = calculate_atr(df, self.config.volatility_period)
            if not atr.empty:
                volatility_strength = self._calculate_volatility_strength(atr)
                factors['volatility_strength'] = volatility_strength
            else:
                factors['volatility_strength'] = 0.5
            
            # 6. Trend duration factor
            trend_duration = self._calculate_trend_duration_strength(structure_breaks)
            factors['trend_duration'] = trend_duration
            
            # 7. Higher highs/Lower lows factor
            hl_factor = self._calculate_hl_ll_factor(swing_points, primary_trend)
            factors['hl_ll_factor'] = hl_factor
            
            return factors
            
        except Exception as e:
            logger.debug(f"Error calculating trend strength factors: {e}")
            return {
                'break_strength': 0.0,
                'swing_consistency': 0.0,
                'price_momentum': 0.0,
                'volume_confirmation': 0.5,
                'volatility_strength': 0.5,
                'trend_duration': 0.0,
                'hl_ll_factor': 0.0
            }
    
    def _calculate_swing_consistency(self, swing_points: List[SwingPoint], primary_trend: str) -> float:
        """Calculate how consistently swings follow the trend"""
        
        if len(swing_points) < 4:
            return 0.0
        
        try:
            if primary_trend == MarketStructure.BULLISH:
                # Check for higher highs and higher lows
                highs = [sp for sp in swing_points[-6:] if sp.swing_type == 'HIGH']
                lows = [sp for sp in swing_points[-6:] if sp.swing_type == 'LOW']
                
                hh_score = 0.0
                if len(highs) >= 2:
                    higher_highs = sum(1 for i in range(1, len(highs)) if highs[i].price > highs[i-1].price)
                    hh_score = higher_highs / max(1, len(highs) - 1)
                
                hl_score = 0.0
                if len(lows) >= 2:
                    higher_lows = sum(1 for i in range(1, len(lows)) if lows[i].price > lows[i-1].price)
                    hl_score = higher_lows / max(1, len(lows) - 1)
                
                return (hh_score + hl_score) / 2
            
            elif primary_trend == MarketStructure.BEARISH:
                # Check for lower highs and lower lows
                highs = [sp for sp in swing_points[-6:] if sp.swing_type == 'HIGH']
                lows = [sp for sp in swing_points[-6:] if sp.swing_type == 'LOW']
                
                lh_score = 0.0
                if len(highs) >= 2:
                    lower_highs = sum(1 for i in range(1, len(highs)) if highs[i].price < highs[i-1].price)
                    lh_score = lower_highs / max(1, len(highs) - 1)
                
                ll_score = 0.0
                if len(lows) >= 2:
                    lower_lows = sum(1 for i in range(1, len(lows)) if lows[i].price < lows[i-1].price)
                    ll_score = lower_lows / max(1, len(lows) - 1)
                
                return (lh_score + ll_score) / 2
            
            return 0.0
            
        except Exception as e:
            logger.debug(f"Error calculating swing consistency: {e}")
            return 0.0
    
    def _calculate_price_momentum(self, df: pd.DataFrame, period: int) -> float:
        """Calculate price momentum over specified period"""
        
        try:
            if len(df) < period:
                return 0.0
            
            # Calculate rate of change
            current_price = df['close'].iloc[-1]
            past_price = df['close'].iloc[-period]
            
            if past_price == 0:
                return 0.0
            
            momentum = (current_price - past_price) / past_price
            
            # Normalize momentum (typical range -0.1 to 0.1 for most timeframes)
            normalized_momentum = np.tanh(momentum * 10) # Tanh to keep in -1 to 1 range
            
            return normalized_momentum
            
        except Exception as e:
            logger.debug(f"Error calculating price momentum: {e}")
            return 0.0
    
    def _calculate_volume_trend_alignment(self, df: pd.DataFrame, primary_trend: str) -> float:
        """Calculate how well volume aligns with trend direction"""
        
        try:
            if len(df) < 20:
                return 0.5
            
            # Calculate price change and volume for recent periods
            recent_df = df.tail(20)
            price_changes = recent_df['close'].diff()
            volumes = recent_df['volume']
            
            if primary_trend == MarketStructure.BULLISH:
                # In bullish trend, expect higher volume on up days
                up_days = price_changes > 0
                avg_up_volume = volumes[up_days].mean() if up_days.any() else 0
                avg_down_volume = volumes[~up_days].mean() if (~up_days).any() else 0
                
                if avg_down_volume == 0:
                    return 1.0
                
                volume_ratio = avg_up_volume / (avg_up_volume + avg_down_volume)
                return volume_ratio
            
            elif primary_trend == MarketStructure.BEARISH:
                # In bearish trend, expect higher volume on down days
                down_days = price_changes < 0
                avg_down_volume = volumes[down_days].mean() if down_days.any() else 0
                avg_up_volume = volumes[~down_days].mean() if (~down_days).any() else 0
                
                if avg_up_volume == 0:
                    return 1.0
                
                volume_ratio = avg_down_volume / (avg_down_volume + avg_up_volume)
                return volume_ratio
            
            return 0.5
            
        except Exception as e:
            logger.debug(f"Error calculating volume trend alignment: {e}")
            return 0.5
    
    def _calculate_volatility_strength(self, atr: pd.Series) -> float:
        """Calculate trend strength based on volatility patterns"""
        
        try:
            if atr.empty or len(atr) < 10:
                return 0.5
            
            # Compare recent ATR to longer-term average
            recent_atr = atr.tail(5).mean()
            long_term_atr = atr.tail(20).mean()
            
            if long_term_atr == 0:
                return 0.5
            
            # Higher recent volatility can indicate strong trending
            volatility_ratio = recent_atr / long_term_atr
            
            # Optimal volatility for trending is slightly elevated (1.2-1.5x normal)
            if 1.2 <= volatility_ratio <= 1.8:
                return min(1.0, volatility_ratio / 1.5)
            elif volatility_ratio > 1.8:
                return max(0.3, 2.0 - volatility_ratio)  # Too much volatility reduces trend quality
            else:
                return volatility_ratio / 1.2  # Too low volatility
            
        except Exception as e:
            logger.debug(f"Error calculating volatility strength: {e}")
            return 0.5
    
    def _calculate_trend_duration_strength(self, structure_breaks: List[StructureBreakPoint]) -> float:
        """Calculate trend strength based on how long it has been developing"""
        
        try:
            if not structure_breaks:
                return 0.0
            
            # Count consecutive breaks in same direction
            if len(structure_breaks) < 2:
                return 0.3
            
            # Look at recent breaks to see trend persistence
            recent_breaks = structure_breaks[-5:]
            
            # Group by trend direction
            bullish_breaks = [sb for sb in recent_breaks if 'BULLISH' in sb.break_type]
            bearish_breaks = [sb for sb in recent_breaks if 'BEARISH' in sb.break_type]
            
            max_consecutive = max(len(bullish_breaks), len(bearish_breaks))
            
            # Strength increases with consecutive breaks, but plateaus
            if max_consecutive >= 4:
                return 1.0
            elif max_consecutive >= 3:
                return 0.8
            elif max_consecutive >= 2:
                return 0.6
            else:
                return 0.3
                
        except Exception as e:
            logger.debug(f"Error calculating trend duration strength: {e}")
            return 0.0
    
    def _calculate_hl_ll_factor(self, swing_points: List[SwingPoint], primary_trend: str) -> float:
        """Calculate higher highs/lower lows factor"""
        
        try:
            if len(swing_points) < 4:
                return 0.0
            
            # Get recent swing points
            recent_swings = swing_points[-8:]
            highs = [sp for sp in recent_swings if sp.swing_type == 'HIGH']
            lows = [sp for sp in recent_swings if sp.swing_type == 'LOW']
            
            if primary_trend == MarketStructure.BULLISH:
                hh_factor = 0.0
                if len(highs) >= 2:
                    # Calculate how much each high exceeds the previous
                    high_improvements = []
                    for i in range(1, len(highs)):
                        if highs[i].price > highs[i-1].price:
                            improvement = (highs[i].price - highs[i-1].price) / highs[i-1].price
                            high_improvements.append(improvement)
                    
                    if high_improvements:
                        hh_factor = min(1.0, np.mean(high_improvements) * 50)  # Scale factor
                
                hl_factor = 0.0
                if len(lows) >= 2:
                    # Calculate how much each low exceeds the previous
                    low_improvements = []
                    for i in range(1, len(lows)):
                        if lows[i].price > lows[i-1].price:
                            improvement = (lows[i].price - lows[i-1].price) / lows[i-1].price
                            low_improvements.append(improvement)
                    
                    if low_improvements:
                        hl_factor = min(1.0, np.mean(low_improvements) * 50)  # Scale factor
                
                return (hh_factor + hl_factor) / 2
            
            elif primary_trend == MarketStructure.BEARISH:
                lh_factor = 0.0
                if len(highs) >= 2:
                    # Calculate how much each high is lower than previous
                    high_declines = []
                    for i in range(1, len(highs)):
                        if highs[i].price < highs[i-1].price:
                            decline = (highs[i-1].price - highs[i].price) / highs[i-1].price
                            high_declines.append(decline)
                    
                    if high_declines:
                        lh_factor = min(1.0, np.mean(high_declines) * 50)  # Scale factor
                
                ll_factor = 0.0
                if len(lows) >= 2:
                    # Calculate how much each low is lower than previous
                    low_declines = []
                    for i in range(1, len(lows)):
                        if lows[i].price < lows[i-1].price:
                            decline = (lows[i-1].price - lows[i].price) / lows[i-1].price
                            low_declines.append(decline)
                    
                    if low_declines:
                        ll_factor = min(1.0, np.mean(low_declines) * 50)  # Scale factor
                
                return (lh_factor + ll_factor) / 2
            
            return 0.0
            
        except Exception as e:
            logger.debug(f"Error calculating HL/LL factor: {e}")
            return 0.0
    
    def _classify_trend_strength(self, strength_factors: Dict[str, float]) -> str:
        """Classify trend strength based on calculated factors"""
        
        try:
            # Weight different factors
            weights = {
                'break_strength': 0.20,
                'swing_consistency': 0.20,
                'price_momentum': 0.15,
                'volume_confirmation': 0.10,
                'volatility_strength': 0.10,
                'trend_duration': 0.15,
                'hl_ll_factor': 0.10
            }
            
            # Calculate weighted strength score
            weighted_score = sum(
                strength_factors.get(factor, 0.0) * weight 
                for factor, weight in weights.items()
            )
            
            # Classify based on thresholds
            thresholds = self.config.strength_thresholds
            
            if weighted_score >= thresholds['VERY_STRONG']:
                return TrendStrength.VERY_STRONG
            elif weighted_score >= thresholds['STRONG']:
                return TrendStrength.STRONG
            elif weighted_score >= thresholds['MODERATE']:
                return TrendStrength.MODERATE
            else:
                return TrendStrength.WEAK
                
        except Exception as e:
            logger.debug(f"Error classifying trend strength: {e}")
            return TrendStrength.WEAK
    
    def _calculate_trend_confidence(
        self, 
        strength_factors: Dict[str, float], 
        current_structure: Dict,
        swing_points: List[SwingPoint]
    ) -> float:
        """Calculate confidence in trend classification"""
        
        try:
            confidence_factors = []
            
            # Structure confidence
            structure_confidence = current_structure.get('confidence', 0.0)
            confidence_factors.append(structure_confidence)
            
            # Factor consistency (how well factors agree)
            factor_values = [v for v in strength_factors.values() if 0.0 <= v <= 1.0]
            if factor_values:
                factor_std = np.std(factor_values)
                consistency_score = max(0.0, 1.0 - factor_std * 2)  # Lower std = higher consistency
                confidence_factors.append(consistency_score)
            
            # Data sufficiency
            if len(swing_points) >= 6:
                confidence_factors.append(1.0)
            elif len(swing_points) >= 4:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.5)
            
            # Average strength of factors
            avg_factor_strength = np.mean(factor_values) if factor_values else 0.0
            confidence_factors.append(avg_factor_strength)
            
            return np.mean(confidence_factors) if confidence_factors else 0.0
            
        except Exception as e:
            logger.debug(f"Error calculating trend confidence: {e}")
            return 0.0
    
    def _calculate_trend_age(self, structure_breaks: List[StructureBreakPoint], df: pd.DataFrame) -> int:
        """Calculate how long the current trend has been in place"""
        
        try:
            if not structure_breaks:
                return 0
            
            # Find the most recent significant trend-defining break
            trend_start_break = None
            current_trend_type = None
            
            # Go through breaks from newest to oldest
            for break_point in reversed(structure_breaks):
                if 'CHoC' in break_point.break_type:  # Change of character defines new trend
                    trend_start_break = break_point
                    current_trend_type = 'BULLISH' if 'BULLISH' in break_point.break_type else 'BEARISH'
                    break
            
            if trend_start_break:
                # Calculate age from trend start
                trend_age = len(df) - trend_start_break.breaking_candle_index - 1
                return max(0, trend_age)
            
            # If no CHoC found, use the oldest break of current trend type
            if structure_breaks:
                latest_break = structure_breaks[-1]
                trend_age = len(df) - latest_break.breaking_candle_index - 1
                return max(0, trend_age)
            
            return 0
            
        except Exception as e:
            logger.debug(f"Error calculating trend age: {e}")
            return 0
    
    def _calculate_trend_momentum(self, df: pd.DataFrame, primary_trend: str) -> float:
        """Calculate trend momentum (rate of trend development)"""
        
        try:
            if len(df) < self.config.momentum_period:
                return 0.0
            
            # Calculate price momentum over different periods
            short_period = min(5, len(df) // 4)
            medium_period = min(10, len(df) // 2)
            long_period = min(self.config.momentum_period, len(df))
            
            periods = [short_period, medium_period, long_period]
            momentum_values = []
            
            for period in periods:
                if len(df) >= period:
                    current_price = df['close'].iloc[-1]
                    past_price = df['close'].iloc[-period]
                    
                    if past_price != 0:
                        momentum = (current_price - past_price) / past_price
                        momentum_values.append(momentum)
            
            if not momentum_values:
                return 0.0
            
            # Calculate accelerating/decelerating momentum
            if len(momentum_values) >= 2:
                # Check if momentum is accelerating (short > medium > long)
                if primary_trend == MarketStructure.BULLISH:
                    accelerating = all(momentum_values[i] >= momentum_values[i+1] 
                                     for i in range(len(momentum_values)-1))
                elif primary_trend == MarketStructure.BEARISH:
                    accelerating = all(momentum_values[i] <= momentum_values[i+1] 
                                     for i in range(len(momentum_values)-1))
                else:
                    accelerating = False
                
                if accelerating:
                    return abs(momentum_values[0])  # Return strongest momentum
                else:
                    return abs(np.mean(momentum_values)) * 0.7  # Reduced for decelerating
            
            return abs(np.mean(momentum_values)) if momentum_values else 0.0
            
        except Exception as e:
            logger.debug(f"Error calculating trend momentum: {e}")
            return 0.0
    
    def _calculate_reversal_probability(
        self, 
        df: pd.DataFrame,
        swing_points: List[SwingPoint],
        structure_breaks: List[StructureBreakPoint],
        primary_trend: str
    ) -> float:
        """Calculate probability of trend reversal"""
        
        try:
            reversal_factors = []
            
            # 1. Trend maturity factor (older trends more likely to reverse)
            if structure_breaks:
                trend_age = self._calculate_trend_age(structure_breaks, df)
                max_trend_age = 100  # Assume trends weaken after 100 candles
                maturity_factor = min(1.0, trend_age / max_trend_age)
                reversal_factors.append(maturity_factor * 0.3)  # Weight 30%
            
            # 2. Momentum divergence
            if len(df) >= 20:
                recent_momentum = self._calculate_price_momentum(df.tail(10), 5)
                older_momentum = self._calculate_price_momentum(df.tail(20).head(10), 5)
                
                # Divergence occurs when momentum weakens while price continues
                if primary_trend == MarketStructure.BULLISH:
                    divergence = max(0, older_momentum - recent_momentum) if older_momentum > 0 else 0
                elif primary_trend == MarketStructure.BEARISH:
                    divergence = max(0, recent_momentum - older_momentum) if older_momentum < 0 else 0
                else:
                    divergence = 0
                
                reversal_factors.append(min(1.0, divergence * 5) * 0.25)  # Weight 25%
            
            # 3. Structure break failure rate
            if len(structure_breaks) >= 3:
                recent_breaks = structure_breaks[-3:]
                avg_break_strength = np.mean([sb.confirmation_strength for sb in recent_breaks])
                weakness_factor = max(0, 1.0 - avg_break_strength)
                reversal_factors.append(weakness_factor * 0.2)  # Weight 20%
            
            # 4. Swing weakening
            if len(swing_points) >= 4:
                recent_swings = swing_points[-4:]
                avg_recent_strength = np.mean([sp.strength for sp in recent_swings])
                
                if len(swing_points) >= 8:
                    older_swings = swing_points[-8:-4]
                    avg_older_strength = np.mean([sp.strength for sp in older_swings])
                    
                    if avg_older_strength > 0:
                        strength_decline = max(0, avg_older_strength - avg_recent_strength) / avg_older_strength
                        reversal_factors.append(strength_decline * 0.15)  # Weight 15%
            
            # 5. Extreme price levels (if we can determine them)
            if len(df) >= 50:
                current_price = df['close'].iloc[-1]
                high_50 = df['high'].tail(50).max()
                low_50 = df['low'].tail(50).min()
                price_range = high_50 - low_50
                
                if price_range > 0:
                    if primary_trend == MarketStructure.BULLISH:
                        # How close to recent highs
                        proximity_to_extreme = (current_price - low_50) / price_range
                    else:
                        # How close to recent lows  
                        proximity_to_extreme = (high_50 - current_price) / price_range
                    
                    extreme_factor = max(0, proximity_to_extreme - 0.8) * 5  # Kicks in at 80% of range
                    reversal_factors.append(min(1.0, extreme_factor) * 0.1)  # Weight 10%
            
            # Calculate final reversal probability
            base_probability = 0.2  # Base 20% chance of reversal
            additional_risk = sum(reversal_factors)
            
            final_probability = min(0.9, base_probability + additional_risk)  # Cap at 90%
            return final_probability
            
        except Exception as e:
            logger.debug(f"Error calculating reversal probability: {e}")
            return 0.5  # Default neutral probability
    
    def _identify_support_factors(
        self,
        df: pd.DataFrame,
        swing_points: List[SwingPoint],
        structure_breaks: List[StructureBreakPoint], 
        primary_trend: str
    ) -> List[str]:
        """Identify factors that support the current trend"""
        
        support_factors = []
        
        try:
            # Strong recent breaks
            if structure_breaks:
                recent_breaks = [sb for sb in structure_breaks[-3:] 
                               if sb.confirmation_strength >= 0.7]
                if recent_breaks:
                    support_factors.append(f"Strong structure breaks ({len(recent_breaks)})")
            
            # High quality swing points
            if swing_points:
                high_quality_swings = [sp for sp in swing_points[-6:] 
                                     if sp.strength >= 0.7]
                if len(high_quality_swings) >= 3:
                    support_factors.append("High quality swing formation")
            
            # Momentum alignment
            if len(df) >= 10:
                momentum = self._calculate_price_momentum(df, 10)
                if primary_trend == MarketStructure.BULLISH and momentum > 0.02:
                    support_factors.append("Strong bullish momentum")
                elif primary_trend == MarketStructure.BEARISH and momentum < -0.02:
                    support_factors.append("Strong bearish momentum")
            
            # Volume confirmation (if available)
            if 'volume' in df.columns and len(df) >= 10:
                volume_alignment = self._calculate_volume_trend_alignment(df, primary_trend)
                if volume_alignment >= 0.6:
                    support_factors.append("Volume supporting trend")
            
            # Consistent swing progression
            consistency = self._calculate_swing_consistency(swing_points, primary_trend)
            if consistency >= 0.7:
                support_factors.append("Consistent swing progression")
            
            # Recent trend development
            if structure_breaks:
                trend_age = self._calculate_trend_age(structure_breaks, df)
                if 5 <= trend_age <= 50:  # Not too new, not too old
                    support_factors.append("Trend in optimal age range")
            
        except Exception as e:
            logger.debug(f"Error identifying support factors: {e}")
        
        return support_factors
    
    def _identify_warning_signals(
        self,
        df: pd.DataFrame,
        swing_points: List[SwingPoint],
        structure_breaks: List[StructureBreakPoint],
        primary_trend: str
    ) -> List[str]:
        """Identify warning signals that suggest trend weakness"""
        
        warning_signals = []
        
        try:
            # Weakening structure breaks
            if structure_breaks:
                recent_breaks = structure_breaks[-3:]
                weak_breaks = [sb for sb in recent_breaks if sb.confirmation_strength < 0.5]
                if weak_breaks:
                    warning_signals.append(f"Weakening structure breaks ({len(weak_breaks)})")
            
            # Declining swing quality
            if len(swing_points) >= 6:
                recent_swings = swing_points[-3:]
                older_swings = swing_points[-6:-3]
                
                recent_avg_strength = np.mean([sp.strength for sp in recent_swings])
                older_avg_strength = np.mean([sp.strength for sp in older_swings])
                
                if recent_avg_strength < older_avg_strength * 0.8:
                    warning_signals.append("Declining swing quality")
            
            # Momentum divergence
            if len(df) >= 20:
                recent_momentum = abs(self._calculate_price_momentum(df.tail(10), 5))
                older_momentum = abs(self._calculate_price_momentum(df.tail(20).head(10), 5))
                
                if recent_momentum < older_momentum * 0.7:
                    warning_signals.append("Momentum divergence detected")
            
            # Trend aging
            if structure_breaks:
                trend_age = self._calculate_trend_age(structure_breaks, df)
                if trend_age > 80:
                    warning_signals.append("Trend showing signs of maturity")
            
            # Lack of follow-through
            if structure_breaks and len(structure_breaks) >= 2:
                latest_break = structure_breaks[-1]
                candles_since_break = len(df) - latest_break.breaking_candle_index - 1
                
                if candles_since_break > 20:  # No new break in 20+ candles
                    warning_signals.append("Lack of trend follow-through")
            
            # Volume divergence (if available)
            if 'volume' in df.columns and len(df) >= 10:
                volume_alignment = self._calculate_volume_trend_alignment(df, primary_trend)
                if volume_alignment < 0.4:
                    warning_signals.append("Volume not supporting trend")
            
            # Inconsistent swing progression
            consistency = self._calculate_swing_consistency(swing_points, primary_trend)
            if consistency < 0.4:
                warning_signals.append("Inconsistent swing progression")
            
        except Exception as e:
            logger.debug(f"Error identifying warning signals: {e}")
        
        return warning_signals
    
    def _analyze_trend_momentum(
        self,
        df: pd.DataFrame,
        trend_classification: TrendClassification,
        swing_points: List[SwingPoint]
    ) -> Dict:
        """Analyze trend momentum characteristics"""
        
        try:
            momentum_analysis = {
                'current_momentum': trend_classification.trend_momentum,
                'momentum_direction': 'UNKNOWN',
                'momentum_strength': 'WEAK',
                'momentum_sustainability': 0.0,
                'momentum_signals': []
            }
            
            # Determine momentum direction
            if trend_classification.trend_momentum > 0.02:
                momentum_analysis['momentum_direction'] = 'ACCELERATING'
            elif trend_classification.trend_momentum < -0.02:
                momentum_analysis['momentum_direction'] = 'DECELERATING'
            else:
                momentum_analysis['momentum_direction'] = 'STABLE'
            
            # Classify momentum strength
            momentum_value = abs(trend_classification.trend_momentum)
            if momentum_value >= 0.05:
                momentum_analysis['momentum_strength'] = 'VERY_STRONG'
            elif momentum_value >= 0.03:
                momentum_analysis['momentum_strength'] = 'STRONG'
            elif momentum_value >= 0.01:
                momentum_analysis['momentum_strength'] = 'MODERATE'
            else:
                momentum_analysis['momentum_strength'] = 'WEAK'
            
            # Calculate sustainability
            if len(df) >= 20:
                # Compare short vs long term momentum
                short_momentum = abs(self._calculate_price_momentum(df.tail(5), 3))
                long_momentum = abs(self._calculate_price_momentum(df, 20))
                
                if long_momentum > 0:
                    sustainability = min(1.0, short_momentum / long_momentum)
                else:
                    sustainability = 0.5
                
                momentum_analysis['momentum_sustainability'] = sustainability
            
            # Generate momentum signals
            if momentum_analysis['momentum_direction'] == 'ACCELERATING':
                momentum_analysis['momentum_signals'].append("Momentum building")
            elif momentum_analysis['momentum_direction'] == 'DECELERATING':
                momentum_analysis['momentum_signals'].append("Momentum weakening")
            
            if momentum_analysis['momentum_strength'] in ['STRONG', 'VERY_STRONG']:
                momentum_analysis['momentum_signals'].append("Strong momentum present")
            
            if momentum_analysis['momentum_sustainability'] >= 0.7:
                momentum_analysis['momentum_signals'].append("Sustainable momentum")
            elif momentum_analysis['momentum_sustainability'] <= 0.3:
                momentum_analysis['momentum_signals'].append("Momentum may be exhausting")
            
            return momentum_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing trend momentum: {e}")
            return {
                'current_momentum': 0.0,
                'momentum_direction': 'UNKNOWN',
                'momentum_strength': 'WEAK', 
                'momentum_sustainability': 0.0,
                'momentum_signals': ['Analysis error']
            }
    
    def _analyze_reversal_probability(
        self,
        df: pd.DataFrame,
        trend_classification: TrendClassification,
        structure_analysis: Dict
    ) -> Dict:
        """Analyze trend reversal probability and signals"""
        
        try:
            reversal_probability = trend_classification.reversal_probability
            
            reversal_analysis = {
                'reversal_probability': reversal_probability,
                'reversal_risk': 'LOW',
                'reversal_timeframe': 'UNKNOWN',
                'reversal_signals': [],
                'reversal_triggers': []
            }
            
            # Classify reversal risk
            if reversal_probability >= 0.7:
                reversal_analysis['reversal_risk'] = 'HIGH'
            elif reversal_probability >= 0.5:
                reversal_analysis['reversal_risk'] = 'MODERATE'
            elif reversal_probability >= 0.3:
                reversal_analysis['reversal_risk'] = 'LOW_MODERATE'
            else:
                reversal_analysis['reversal_risk'] = 'LOW'
            
            # Estimate reversal timeframe based on trend characteristics
            if trend_classification.trend_age > 50:
                if reversal_probability >= 0.6:
                    reversal_analysis['reversal_timeframe'] = 'SHORT_TERM'
                else:
                    reversal_analysis['reversal_timeframe'] = 'MEDIUM_TERM'
            else:
                reversal_analysis['reversal_timeframe'] = 'LONG_TERM'
            
            # Identify reversal signals
            if len(trend_classification.warning_signals) >= 3:
                reversal_analysis['reversal_signals'].append("Multiple warning signals present")
            
            if trend_classification.trend_momentum < 0.01:
                reversal_analysis['reversal_signals'].append("Momentum weakening")
            
            if trend_classification.trend_age > 80:
                reversal_analysis['reversal_signals'].append("Mature trend - higher reversal risk")
            
            # Identify potential reversal triggers
            swing_points = structure_analysis.get('swing_points', [])
            if swing_points:
                recent_swing = swing_points[-1]
                if recent_swing.strength < 0.5:
                    reversal_analysis['reversal_triggers'].append("Weak recent swing formation")
            
            structure_breaks = structure_analysis.get('structure_breaks', [])
            if structure_breaks:
                recent_break = structure_breaks[-1]
                if recent_break.confirmation_strength < 0.6:
                    reversal_analysis['reversal_triggers'].append("Weak structure break confirmation")
            
            # Add specific reversal level if identifiable
            if swing_points and len(swing_points) >= 2:
                if trend_classification.primary_trend == MarketStructure.BULLISH:
                    # Key support level for reversal
                    recent_lows = [sp for sp in swing_points[-4:] if sp.swing_type == 'LOW']
                    if recent_lows:
                        key_support = min(recent_lows, key=lambda x: x.price)
                        reversal_analysis['key_reversal_level'] = key_support.price
                else:
                    # Key resistance level for reversal
                    recent_highs = [sp for sp in swing_points[-4:] if sp.swing_type == 'HIGH']
                    if recent_highs:
                        key_resistance = max(recent_highs, key=lambda x: x.price)
                        reversal_analysis['key_reversal_level'] = key_resistance.price
            
            return reversal_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing reversal probability: {e}")
            return {
                'reversal_probability': 0.5,
                'reversal_risk': 'UNKNOWN',
                'reversal_timeframe': 'UNKNOWN',
                'reversal_signals': ['Analysis error'],
                'reversal_triggers': []
            }
    
    def _analyze_trend_strength(
        self,
        df: pd.DataFrame,
        trend_classification: TrendClassification,
        structure_analysis: Dict
    ) -> Dict:
        """Analyze trend strength characteristics"""
        
        try:
            strength_analysis = {
                'strength_classification': trend_classification.trend_strength,
                'strength_score': 0.0,
                'strength_components': {},
                'strength_outlook': 'NEUTRAL'
            }
            
            # Calculate individual strength components
            swing_points = structure_analysis.get('swing_points', [])
            structure_breaks = structure_analysis.get('structure_breaks', [])
            
            strength_factors = self._calculate_trend_strength_factors(
                df, swing_points, structure_breaks, trend_classification.primary_trend
            )
            
            strength_analysis['strength_components'] = strength_factors
            
            # Calculate overall strength score
            weights = {
                'break_strength': 0.20,
                'swing_consistency': 0.20, 
                'price_momentum': 0.15,
                'volume_confirmation': 0.10,
                'volatility_strength': 0.10,
                'trend_duration': 0.15,
                'hl_ll_factor': 0.10
            }
            
            strength_score = sum(
                strength_factors.get(factor, 0.0) * weight 
                for factor, weight in weights.items()
            )
            
            strength_analysis['strength_score'] = strength_score
            
            # Determine strength outlook
            if len(trend_classification.support_factors) > len(trend_classification.warning_signals):
                strength_analysis['strength_outlook'] = 'STRENGTHENING'
            elif len(trend_classification.warning_signals) > len(trend_classification.support_factors):
                strength_analysis['strength_outlook'] = 'WEAKENING'
            else:
                strength_analysis['strength_outlook'] = 'STABLE'
            
            return strength_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing trend strength: {e}")
            return {
                'strength_classification': TrendStrength.WEAK,
                'strength_score': 0.0,
                'strength_components': {},
                'strength_outlook': 'UNKNOWN'
            }
    
    def _empty_result(self, error: str = None) -> Dict:
        """Return empty result structure"""
        return {
            'trend_classification': TrendClassification(
                primary_trend=MarketStructure.UNKNOWN,
                trend_strength=TrendStrength.WEAK,
                confidence=0.0,
                trend_age=0,
                trend_momentum=0.0,
                reversal_probability=0.5,
                support_factors=[],
                warning_signals=['Analysis failed'] if error else []
            ),
            'momentum_analysis': {
                'current_momentum': 0.0,
                'momentum_direction': 'UNKNOWN',
                'momentum_strength': 'WEAK',
                'momentum_sustainability': 0.0,
                'momentum_signals': []
            },
            'reversal_analysis': {
                'reversal_probability': 0.5,
                'reversal_risk': 'UNKNOWN', 
                'reversal_signals': [],
                'reversal_triggers': []
            },
            'strength_analysis': {
                'strength_classification': TrendStrength.WEAK,
                'strength_score': 0.0,
                'strength_components': {},
                'strength_outlook': 'UNKNOWN'
            },
            'structure_context': {},
            'success': False,
            'symbol': 'UNKNOWN',
            'candles_analyzed': 0,
            'classification_time': datetime.now(),
            'error': error
        }
    
    def get_trend_summary(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> Dict:
        """Get concise trend summary for quick analysis"""
        try:
            analysis = self.classify_trend(df, symbol)
            
            if not analysis['success']:
                return {'error': 'Trend classification failed'}
            
            classification = analysis['trend_classification']
            
            return {
                'symbol': symbol,
                'trend': classification.primary_trend,
                'strength': classification.trend_strength,
                'confidence': classification.confidence,
                'momentum': classification.trend_momentum,
                'reversal_risk': analysis['reversal_analysis']['reversal_risk'],
                'age': classification.trend_age,
                'last_update': datetime.now(),
                'actionable': classification.confidence >= 0.6
            }
            
        except Exception as e:
            logger.error(f"Error getting trend summary: {e}")
            return {'error': str(e)}

# Factory function
def create_trend_classifier(momentum_period: int = 20) -> TrendClassifier:
    """Factory function to create trend classifier"""
    config = TrendClassificationConfig(momentum_period=momentum_period)
    return TrendClassifier(config)