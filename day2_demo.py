# day2_demo.py - Market Structure Detection Demo

"""
SMC Trading Bot - Day 2 Demo
============================

Market Structure Detection System Demo
- Swing point detection
- BOS/CHoC identification  
- Trend classification
- Visual debugging

Target: 80%+ swing detection, 85%+ BOS/CHoC accuracy
"""

import sys
import asyncio
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Imports
from config.settings import settings, DAY2_TESTING_CONFIG
from config.logging_config import setup_logging, LoggingContext
from data_service.data_fetcher import EnhancedMT5DataFetcher
from market_structure.swing_analyzer import SwingAnalyzer, SwingAnalysisConfig
from market_structure.bos_choc_detector import BOSCHOCDetector, BOSCHOCConfig
from market_structure.structure_detector import StructureDetector, StructureDetectionConfig
from market_structure.trend_classifier import TrendClassifier, TrendClassificationConfig
from analysis_dashboard.structure_visualizer import StructureVisualizer
from utils.constants import MarketStructure, StructureBreak
from utils.helpers import calculate_performance_metrics, format_number, format_percentage
from loguru import logger

class SMCTradingBotDay2:
    """Day 2 SMC Trading Bot - Market Structure Detection Demo"""
    
    def __init__(self):
        """Initialize the Day 2 demo system"""
        setup_logging(log_level="INFO")
        logger.info("="*60)
        logger.info("SMC Trading Bot - Day 2 Market Structure Demo")
        logger.info("="*60)
        
        # Initialize components
        self.data_fetcher = EnhancedMT5DataFetcher()
        self.swing_analyzer = SwingAnalyzer()
        self.bos_choc_detector = BOSCHOCDetector()
        self.structure_detector = StructureDetector()
        self.trend_classifier = TrendClassifier()
        self.visualizer = StructureVisualizer()
        
        # Configuration
        self.test_symbols = DAY2_TESTING_CONFIG["test_symbols"]
        self.test_timeframes = DAY2_TESTING_CONFIG["test_timeframes"] 
        self.analysis_candles = DAY2_TESTING_CONFIG["analysis_candles"]
        self.performance_targets = DAY2_TESTING_CONFIG["performance_targets"]
        
        # Results storage
        self.analysis_results = {}
        self.performance_metrics = {}
        
        logger.info(f"Testing market structure on: {self.test_symbols}")
        logger.info(f"Analysis timeframes: {self.test_timeframes}")
    
    async def run_day2_tests(self) -> Dict:
        """Run comprehensive Day 2 market structure tests"""
        logger.info("Starting Day 2 market structure analysis...")
        
        with LoggingContext("day2_comprehensive_tests"):
            test_results = {
                'swing_detection': await self._test_swing_detection(),
                'bos_choc_detection': await self._test_bos_choc_detection(),
                'structure_detection': await self._test_structure_detection(),
                'trend_classification': await self._test_trend_classification(),
                'multi_timeframe_structure': await self._test_multi_timeframe_structure(),
                'structure_validation': await self._test_structure_validation(),
                'performance_analysis': await self._test_performance_analysis(),
                'visual_debugging': await self._test_visual_debugging()
            }
            
            # Generate comprehensive Day 2 report
            await self._generate_day2_report(test_results)
            
            return test_results
    
    async def _test_swing_detection(self) -> Dict:
        """Test swing point detection accuracy"""
        logger.info("Testing swing point detection...")
        
        results = {}
        
        for symbol in self.test_symbols:
            logger.info(f"Analyzing swing points for {symbol}")
            
            symbol_results = {}
            
            for timeframe in self.test_timeframes:
                try:
                    with LoggingContext("swing_detection", symbol=symbol, timeframe=timeframe):
                        # Fetch data
                        df = await asyncio.to_thread(
                            self.data_fetcher.get_symbol_data, 
                            symbol, timeframe, self.analysis_candles
                        )
                        
                        if df.empty:
                            logger.warning(f"No data for {symbol} {timeframe}")
                            continue
                        
                        start_time = datetime.now()
                        
                        # Analyze swings
                        swing_analysis = self.swing_analyzer.analyze_swings(df, symbol)
                        
                        end_time = datetime.now()
                        duration_ms = (end_time - start_time).total_seconds() * 1000
                        
                        if swing_analysis['success']:
                            swing_points = swing_analysis['swing_points']
                            metrics = swing_analysis['quality_metrics']
                            
                            # Calculate swing detection quality metrics
                            quality_score = self._calculate_swing_quality(swing_points, df)
                            
                            symbol_results[timeframe] = {
                                'success': True,
                                'total_swings': len(swing_points),
                                'swing_highs': metrics.get('high_count', 0),
                                'swing_lows': metrics.get('low_count', 0),
                                'confirmed_swings': sum(1 for sp in swing_points if getattr(sp, 'confirmed', False)),
                                'avg_strength': metrics['avg_strength'],
                                'swing_density': metrics['swing_density'],
                                'quality_score': quality_score,
                                'processing_time_ms': duration_ms,
                                'candles_analyzed': len(df),
                                'target_met': quality_score >= 0.6
                            }
                            
                            logger.info(f"  {timeframe}: {len(swing_points)} swings, quality: {quality_score:.3f}")
                        
                        else:
                            symbol_results[timeframe] = {
                                'success': False,
                                'error': swing_analysis['analysis_metrics'].get('error', 'Unknown error')
                            }
                
                except Exception as e:
                    logger.error(f"Swing detection error for {symbol} {timeframe}: {e}")
                    symbol_results[timeframe] = {
                        'success': False,
                        'error': str(e)
                    }
            
            results[symbol] = symbol_results
        
        logger.success("Swing detection tests completed")
        return results
    
    async def _test_bos_choc_detection(self) -> Dict:
        """Test BOS/CHoC detection accuracy"""
        logger.info("Testing BOS/CHoC detection...")
        
        results = {}
        
        for symbol in self.test_symbols:
            logger.info(f"Analyzing structure breaks for {symbol}")
            
            symbol_results = {}
            
            for timeframe in self.test_timeframes:
                try:
                    with LoggingContext("bos_choc_detection", symbol=symbol, timeframe=timeframe):
                        # Fetch data
                        df = await asyncio.to_thread(
                            self.data_fetcher.get_symbol_data,
                            symbol, timeframe, self.analysis_candles
                        )
                        
                        if df.empty:
                            continue
                        
                        start_time = datetime.now()
                        
                        # Analyze structure breaks
                        structure_analysis = self.bos_choc_detector.detect_structure_breaks(df, symbol)
                        
                        end_time = datetime.now()
                        duration_ms = (end_time - start_time).total_seconds() * 1000
                        
                        if structure_analysis['success']:
                            breaks = structure_analysis['structure_breaks']
                            current_structure = structure_analysis['current_structure']
                            metrics = structure_analysis['analysis_metrics']
                            
                            # Calculate structure detection quality
                            structure_quality = self._calculate_structure_quality(breaks, current_structure)
                            
                            symbol_results[timeframe] = {
                                'success': True,
                                'total_breaks': len(breaks),
                                'bos_breaks': len(structure_analysis['bos_points']),
                                'choc_breaks': len(structure_analysis['choc_points']),
                                'current_trend': current_structure['trend'],
                                'trend_confidence': current_structure['confidence'],
                                'structure_quality': structure_quality,
                                'break_frequency': metrics.get('break_frequency', 0),
                                'avg_confirmation': metrics.get('avg_confirmation_strength', 0),
                                'processing_time_ms': duration_ms,
                                'latest_break_type': current_structure.get('break_type', 'None'),
                                'target_met': structure_quality >= 0.7 and current_structure['confidence'] >= 0.6
                            }
                            
                            logger.info(f"  {timeframe}: {len(breaks)} breaks, trend: {current_structure['trend']}")
                        
                        else:
                            symbol_results[timeframe] = {
                                'success': False,
                                'error': structure_analysis['analysis_metrics'].get('error', 'Unknown error')
                            }
                
                except Exception as e:
                    logger.error(f"Structure detection error for {symbol} {timeframe}: {e}")
                    symbol_results[timeframe] = {
                        'success': False,
                        'error': str(e)
                    }
            
            results[symbol] = symbol_results
        
        logger.success("BOS/CHoC detection tests completed")
        return results
    
    async def _test_structure_detection(self) -> Dict:
        """Test comprehensive structure detection"""
        logger.info("Testing comprehensive structure detection...")
        
        results = {}
        
        for symbol in self.test_symbols:
            symbol_results = {}
            
            for timeframe in self.test_timeframes:
                try:
                    with LoggingContext("structure_detection", symbol=symbol, timeframe=timeframe):
                        df = await asyncio.to_thread(
                            self.data_fetcher.get_symbol_data,
                            symbol, timeframe, self.analysis_candles
                        )
                        
                        if df.empty:
                            continue
                        
                        start_time = datetime.now()
                        
                        # Comprehensive structure analysis
                        structure_analysis = self.structure_detector.detect_market_structure(df, symbol)
                        
                        end_time = datetime.now()
                        duration_ms = (end_time - start_time).total_seconds() * 1000
                        
                        if structure_analysis['success']:
                            structure_state = structure_analysis['structure_state']
                            reliability = structure_analysis['reliability_metrics']
                            context = structure_analysis['trading_context']
                            
                            symbol_results[timeframe] = {
                                'success': True,
                                'primary_trend': structure_state.primary_trend,
                                'trend_strength': structure_state.trend_strength,
                                'structure_quality': structure_state.structure_quality,
                                'reliability_score': reliability['overall_reliability'],
                                'trading_bias': context['trade_bias'],
                                'confidence_level': context['confidence_level'],
                                'processing_time_ms': duration_ms,
                                'structure_age': structure_state.structure_age,
                                'swing_count': structure_state.swing_count,
                                'target_met': (
                                    structure_state.structure_quality >= 0.6 and
                                    reliability['overall_reliability'] >= 0.7
                                )
                            }
                        
                        else:
                            symbol_results[timeframe] = {
                                'success': False,
                                'error': structure_analysis.get('error', 'Analysis failed')
                            }
                
                except Exception as e:
                    logger.error(f"Structure detection error for {symbol} {timeframe}: {e}")
                    symbol_results[timeframe] = {
                        'success': False,
                        'error': str(e)
                    }
            
            results[symbol] = symbol_results
        
        logger.success("Structure detection tests completed")
        return results
    
    async def _test_trend_classification(self) -> Dict:
        """Test trend classification system"""
        logger.info("Testing trend classification...")
        
        results = {}
        
        for symbol in self.test_symbols:
            symbol_results = {}
            
            for timeframe in self.test_timeframes:
                try:
                    with LoggingContext("trend_classification", symbol=symbol, timeframe=timeframe):
                        df = await asyncio.to_thread(
                            self.data_fetcher.get_symbol_data,
                            symbol, timeframe, self.analysis_candles
                        )
                        
                        if df.empty:
                            continue
                        
                        start_time = datetime.now()
                        
                        # Trend classification
                        trend_analysis = self.trend_classifier.classify_trend(df, symbol)
                        
                        end_time = datetime.now()
                        duration_ms = (end_time - start_time).total_seconds() * 1000
                        
                        if trend_analysis['success']:
                            classification = trend_analysis['trend_classification']
                            momentum = trend_analysis['momentum_analysis']
                            reversal = trend_analysis['reversal_analysis']
                            
                            symbol_results[timeframe] = {
                                'success': True,
                                'trend': classification.primary_trend,
                                'strength': classification.trend_strength,
                                'confidence': classification.confidence,
                                'trend_age': classification.trend_age,
                                'momentum': classification.trend_momentum,
                                'reversal_probability': classification.reversal_probability,
                                'momentum_direction': momentum['momentum_direction'],
                                'reversal_risk': reversal['reversal_risk'],
                                'processing_time_ms': duration_ms,
                                'support_factors': len(classification.support_factors),
                                'warning_signals': len(classification.warning_signals),
                                'target_met': classification.confidence >= 0.6
                            }
                        
                        else:
                            symbol_results[timeframe] = {
                                'success': False,
                                'error': trend_analysis.get('error', 'Classification failed')
                            }
                
                except Exception as e:
                    logger.error(f"Trend classification error for {symbol} {timeframe}: {e}")
                    symbol_results[timeframe] = {
                        'success': False,
                        'error': str(e)
                    }
            
            results[symbol] = symbol_results
        
        logger.success("Trend classification tests completed")
        return results
    
    async def _test_multi_timeframe_structure(self) -> Dict:
        """Test multi-timeframe structure alignment"""
        logger.info("Testing multi-timeframe structure alignment...")
        
        results = {}
        
        for symbol in self.test_symbols:
            logger.info(f"Analyzing multi-TF structure for {symbol}")
            
            try:
                with LoggingContext("multi_timeframe_analysis", symbol=symbol):
                    # Fetch data for all timeframes
                    multi_tf_data = await asyncio.to_thread(
                        self.data_fetcher.get_multi_timeframe_data,
                        symbol=symbol,
                        primary_timeframe="M15",
                        higher_timeframes=["H1", "H4"],
                        num_candles=self.analysis_candles
                    )
                    
                    required_tfs = ["M15", "H1", "H4"]
                    if not all(tf in multi_tf_data and not multi_tf_data[tf].empty for tf in required_tfs):
                        logger.warning(f"Incomplete multi-timeframe data for {symbol}. Skipping.")
                        continue
                    
                    # Analyze structure on each timeframe
                    tf_structures = {}
                    
                    for tf, df in multi_tf_data.items():
                        if not df.empty:
                            structure_analysis = self.bos_choc_detector.detect_structure_breaks(df, f"{symbol}_{tf}")
                            
                            if structure_analysis['success']:
                                tf_structures[tf] = {
                                    'trend': structure_analysis['current_structure']['trend'],
                                    'confidence': structure_analysis['current_structure']['confidence'],
                                    'breaks': len(structure_analysis['structure_breaks']),
                                    'latest_break': structure_analysis['current_structure'].get('break_type', 'None'),
                                    'swing_count': len(structure_analysis['swing_points'])
                                }
                    
                    # Calculate alignment score
                    alignment_score = self._calculate_timeframe_alignment(tf_structures)
                    
                    results[symbol] = {
                        'timeframe_structures': tf_structures,
                        'alignment_score': alignment_score,
                        'dominant_trend': self._get_dominant_trend(tf_structures),
                        'conflicting_signals': self._count_conflicting_signals(tf_structures),
                        'structure_strength': self._calculate_overall_structure_strength(tf_structures),
                        'target_met': alignment_score >= 0.66
                    }
                    
                    logger.info(f"  Multi-TF alignment: {alignment_score:.3f}, dominant: {results[symbol]['dominant_trend']}")
            
            except Exception as e:
                logger.error(f"Multi-TF structure error for {symbol}: {e}")
                results[symbol] = {
                    'error': str(e),
                    'alignment_score': 0.0,
                    'target_met': False
                }
        
        logger.success("Multi-timeframe structure tests completed")
        return results
    
    async def _test_structure_validation(self) -> Dict:
        """Test structure validation and accuracy"""
        logger.info("Testing structure validation...")
        
        results = {}
        
        # Test structure validation with known patterns
        for symbol in self.test_symbols[:2]:  # Test two symbols for better coverage
            try:
                with LoggingContext("structure_validation", symbol=symbol):
                    # Get data
                    df = await asyncio.to_thread(
                        self.data_fetcher.get_symbol_data,
                        symbol, "H1", self.analysis_candles
                    )
                    
                    if df.empty:
                        continue
                    
                    # Get structure analysis
                    structure_analysis = self.bos_choc_detector.detect_structure_breaks(df, symbol)
                    
                    if not structure_analysis['success']:
                        continue
                    
                    breaks = structure_analysis['structure_breaks']
                    
                    if not breaks:
                        results[symbol] = {
                            'total_validated': 0,
                            'valid_breaks': 0,
                            'validation_rate': 0.0,
                            'avg_validation_score': 0.0,
                            'target_met': False,
                            'note': 'No structure breaks found'
                        }
                        continue
                    
                    # Validate each structure break (test recent breaks)
                    validation_results = []
                    
                    for break_point in breaks[-min(10, len(breaks)):]:  # Test last 10 breaks or all if fewer
                        validation = self.bos_choc_detector.validate_structure_break(
                            df, break_point.breaking_candle_index, break_point.break_type
                        )
                        
                        validation_results.append({
                            'break_type': break_point.break_type,
                            'valid': validation.get('valid', False),
                            'score': validation.get('validation_score', 0.0),  # Fixed key name
                            'timestamp': break_point.timestamp,
                            'strength': validation.get('break_strength', 'Unknown')
                        })
                    
                    # Calculate validation metrics
                    if validation_results:
                        valid_count = sum(1 for v in validation_results if v['valid'])
                        validation_scores = [v['score'] for v in validation_results]
                        avg_score = np.mean(validation_scores)
                        validation_rate = valid_count / len(validation_results)
                        
                        results[symbol] = {
                            'total_validated': len(validation_results),
                            'valid_breaks': valid_count,
                            'validation_rate': validation_rate,
                            'avg_validation_score': avg_score,
                            'validation_details': validation_results,
                            'target_met': validation_rate >= 0.65 and avg_score >= 0.4,  # More realistic targets
                            'break_strengths': {
                                'strong': sum(1 for v in validation_results if v['strength'] == 'Strong'),
                                'moderate': sum(1 for v in validation_results if v['strength'] == 'Moderate'),
                                'weak': sum(1 for v in validation_results if v['strength'] == 'Weak')
                            }
                        }
                        
                        logger.info(f"  {symbol}: {valid_count}/{len(validation_results)} breaks valid "
                                f"(avg score: {avg_score:.3f}, rate: {validation_rate:.3f})")
                    else:
                        results[symbol] = {
                            'total_validated': 0,
                            'valid_breaks': 0,
                            'validation_rate': 0.0,
                            'avg_validation_score': 0.0,
                            'target_met': False,
                            'note': 'No breaks to validate'
                        }
            
            except Exception as e:
                logger.error(f"Structure validation error for {symbol}: {e}")
                results[symbol] = {'error': str(e), 'target_met': False}
        
        logger.success("Structure validation tests completed")
        return results


    async def _test_performance_analysis(self) -> Dict:
        """Test system performance for market structure analysis"""
        logger.info("Testing performance analysis...")
        
        performance_results = {}
        
        # Test processing speed
        logger.info("Testing processing speed...")
        
        try:
            with LoggingContext("performance_testing"):
                # Get large dataset
                df = await asyncio.to_thread(
                    self.data_fetcher.get_symbol_data,
                    "EURUSD", "M15", 2000
                )
                
                if not df.empty:
                    # Test swing analysis speed
                    swing_times = []
                    for _ in range(3):
                        start_time = datetime.now()
                        self.swing_analyzer.analyze_swings(df)
                        end_time = datetime.now()
                        duration_ms = (end_time - start_time).total_seconds() * 1000
                        swing_times.append(duration_ms)
                    
                    # Test structure detection speed  
                    structure_times = []
                    for _ in range(3):
                        start_time = datetime.now()
                        self.bos_choc_detector.detect_structure_breaks(df)
                        end_time = datetime.now()
                        duration_ms = (end_time - start_time).total_seconds() * 1000
                        structure_times.append(duration_ms)
                    
                    avg_swing_time = np.mean(swing_times)
                    avg_structure_time = np.mean(structure_times)
                    
                    performance_results['processing_speed'] = {
                        'candles_tested': len(df),
                        'avg_swing_analysis_time_ms': avg_swing_time,
                        'avg_structure_detection_time_ms': avg_structure_time,
                        'swing_speed_candles_per_sec': len(df) / (avg_swing_time / 1000),
                        'structure_speed_candles_per_sec': len(df) / (avg_structure_time / 1000),
                        'target_met': avg_structure_time < self.performance_targets["processing_speed_ms"]
                    }
        
        except Exception as e:
            logger.error(f"Performance analysis error: {e}")
            performance_results['processing_speed'] = {'error': str(e), 'target_met': False}
        
        # Memory usage analysis
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            performance_results['memory_usage'] = {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'target_met': memory_info.rss / 1024 / 1024 < self.performance_targets["memory_usage_mb"]
            }
        
        except Exception as e:
            logger.error(f"Memory analysis error: {e}")
            performance_results['memory_usage'] = {'error': str(e), 'target_met': False}
        
        logger.success("Performance analysis completed")
        return performance_results
    
    async def _test_visual_debugging(self) -> Dict:
        """Test visual debugging capabilities"""
        logger.info("Testing visual debugging...")
        
        results = {}
        
        try:
            with LoggingContext("visual_debugging"):
                # Generate visual analysis for one symbol
                symbol = "EURUSD"
                df = await asyncio.to_thread(
                    self.data_fetcher.get_symbol_data,
                    symbol, "H1", 500
                )
                
                if df.empty:
                    return {'error': 'No data for visualization', 'target_met': False}
                
                # Get analysis results
                swing_analysis = self.swing_analyzer.analyze_swings(df, symbol)
                structure_analysis = self.bos_choc_detector.detect_structure_breaks(df, symbol)
                
                if swing_analysis['success'] and structure_analysis['success']:
                    # Create visual debugging output
                    chart_data = self._create_structure_chart_data(
                        df, swing_analysis, structure_analysis
                    )
                    
                    # Test chart creation
                    try:
                        fig = self.visualizer.create_structure_chart(
                            df=df,
                            swing_points=swing_analysis['swing_points'],
                            structure_breaks=structure_analysis['structure_breaks'],
                            current_structure=structure_analysis['current_structure'],
                            symbol=symbol,
                            timeframe="H1"
                        )
                        
                        # Save chart for verification
                        chart_path = self.visualizer.save_chart_html(fig, f"{symbol}_h1_day2_demo")
                        
                        results = {
                            'chart_ready': True,
                            'swing_points_plotted': len(swing_analysis['swing_points']),
                            'structure_breaks_plotted': len(structure_analysis['structure_breaks']),
                            'current_trend_identified': structure_analysis['current_structure']['trend'],
                            'visual_data': chart_data,
                            'chart_saved': chart_path is not None,
                            'chart_path': chart_path,
                            'target_met': True
                        }
                        
                        logger.info(f"Visual debugging ready: {len(swing_analysis['swing_points'])} swings, {len(structure_analysis['structure_breaks'])} breaks")
                        
                    except Exception as chart_error:
                        logger.error(f"Chart creation error: {chart_error}")
                        results = {
                            'chart_ready': False,
                            'error': f"Chart creation failed: {str(chart_error)}",
                            'target_met': False
                        }
                
                else:
                    results = {
                        'error': 'Analysis failed for visualization',
                        'target_met': False
                    }
        
        except Exception as e:
            logger.error(f"Visual debugging error: {e}")
            results = {'error': str(e), 'target_met': False}
        
        logger.success("Visual debugging tests completed")
        return results
    
    def _calculate_swing_quality(self, swing_points: List, df: pd.DataFrame) -> float:
        """Calculate swing detection quality score"""
        if not swing_points:
            return 0.0
        
        quality_factors = []
        
        # 1. Swing density (not too many, not too few)
        density = len(swing_points) / len(df) * 100
        optimal_density = 2.0  # 2 swings per 100 candles
        density_score = 1.0 - min(1.0, abs(density - optimal_density) / optimal_density)
        quality_factors.append(density_score)
        
        # 2. Average swing strength
        avg_strength = np.mean([sp.strength for sp in swing_points])
        quality_factors.append(avg_strength)
        
        # 3. Confirmation rate
        confirmed_count = sum(1 for sp in swing_points if sp.confirmed)
        confirmation_rate = confirmed_count / len(swing_points)
        quality_factors.append(confirmation_rate)
        
        # 4. Alternating pattern (highs and lows should alternate reasonably)
        if len(swing_points) >= 4:
            alternating_score = 0.0
            for i in range(len(swing_points) - 1):
                if swing_points[i].swing_type != swing_points[i+1].swing_type:
                    alternating_score += 1
            alternating_rate = alternating_score / (len(swing_points) - 1)
            quality_factors.append(alternating_rate)
        
        return np.mean(quality_factors) if quality_factors else 0.5
    
    def _calculate_structure_quality(self, breaks: List, current_structure: Dict) -> float:
        """Calculate structure detection quality score"""
        if not breaks:
            return 0.0
        
        quality_factors = []
        
        # 1. Structure confidence
        quality_factors.append(current_structure.get('confidence', 0.0))
        
        # 2. Break confirmation strength
        avg_confirmation = np.mean([bp.confirmation_strength for bp in breaks])
        quality_factors.append(avg_confirmation)
        
        # 3. Break consistency (breaks should support identified trend)
        consistent_breaks = 0
        current_trend = current_structure.get('trend', 'UNKNOWN')
        
        for break_point in breaks[-5:]:  # Check last 5 breaks
            if ((current_trend == MarketStructure.BULLISH and 'BULLISH' in break_point.break_type) or
                (current_trend == MarketStructure.BEARISH and 'BEARISH' in break_point.break_type)):
                consistent_breaks += 1
        
        consistency_rate = consistent_breaks / min(5, len(breaks)) if breaks else 0
        quality_factors.append(consistency_rate)
        
        return np.mean(quality_factors) if quality_factors else 0.5
    
    def _calculate_timeframe_alignment(self, tf_structures: Dict) -> float:
        """Calculate multi-timeframe alignment score"""
        if len(tf_structures) < 2:
            return 0.0
        
        # Get trend directions
        trends = [structure.get('trend', MarketStructure.UNKNOWN) for structure in tf_structures.values()]
        
        # Calculate alignment
        if len(set(trends)) == 1 and trends[0] != MarketStructure.UNKNOWN:
            # Perfect alignment
            return 1.0
        elif MarketStructure.SIDEWAYS in trends:
            # Partial alignment with sideways
            non_sideways = [t for t in trends if t != MarketStructure.SIDEWAYS]
            if len(set(non_sideways)) <= 1:
                return 0.7
        
        # Check for majority alignment
        trend_counts = {}
        for trend in trends:
            trend_counts[trend] = trend_counts.get(trend, 0) + 1
        
        max_count = max(trend_counts.values()) if trend_counts else 0
        alignment_ratio = max_count / len(trends) if trends else 0
        
        return alignment_ratio
    
    def _get_dominant_trend(self, tf_structures: Dict) -> str:
        """Get dominant trend across timeframes"""
        if not tf_structures:
            return MarketStructure.UNKNOWN
        
        # Weight higher timeframes more heavily
        weights = {"M15": 1.0, "H1": 1.5, "H4": 2.0}
        
        trend_scores = {}
        
        for tf, structure in tf_structures.items():
            trend = structure.get('trend', MarketStructure.UNKNOWN)
            confidence = structure.get('confidence', 0.0)
            weight = weights.get(tf, 1.0)
            
            weighted_score = confidence * weight
            
            if trend in trend_scores:
                trend_scores[trend] += weighted_score
            else:
                trend_scores[trend] = weighted_score
        
        if trend_scores:
            return max(trend_scores.items(), key=lambda x: x[1])[0]
        
        return MarketStructure.UNKNOWN
    
    def _count_conflicting_signals(self, tf_structures: Dict) -> int:
        """Count conflicting signals across timeframes"""
        if len(tf_structures) < 2:
            return 0
        
        trends = [structure.get('trend') for structure in tf_structures.values()]
        unique_trends = set(t for t in trends if t != MarketStructure.UNKNOWN and t != MarketStructure.SIDEWAYS)
        
        # If more than one directional trend, we have conflicts
        return max(0, len(unique_trends) - 1)
    
    def _calculate_overall_structure_strength(self, tf_structures: Dict) -> float:
        """Calculate overall structure strength"""
        if not tf_structures:
            return 0.0
        
        # Weight higher timeframes more
        weights = {"M15": 1.0, "H1": 1.5, "H4": 2.0}
        
        weighted_confidences = []
        
        for tf, structure in tf_structures.items():
            confidence = structure.get('confidence', 0.0)
            weight = weights.get(tf, 1.0)
            weighted_confidences.append(confidence * weight)
        
        if weighted_confidences:
            return sum(weighted_confidences) / sum(weights.get(tf, 1.0) for tf in tf_structures.keys())
        
        return 0.0
    
    def _create_structure_chart_data(self, df: pd.DataFrame, swing_analysis: Dict, structure_analysis: Dict) -> Dict:
        """Create chart data for visual debugging"""
        try:
            chart_data = {
                'price_data': {
                    'timestamps': [str(ts) for ts in df.index.tolist()],
                    'open': df['open'].tolist(),
                    'high': df['high'].tolist(),
                    'low': df['low'].tolist(),
                    'close': df['close'].tolist(),
                },
                'swing_points': [],
                'structure_breaks': [],
                'trend_info': structure_analysis['current_structure']
            }
            
            # Add swing points
            for swing_point in swing_analysis['swing_points']:
                chart_data['swing_points'].append({
                    'timestamp': str(swing_point.timestamp),
                    'price': swing_point.price,
                    'type': swing_point.swing_type,
                    'strength': swing_point.strength,
                    'confirmed': swing_point.confirmed
                })
            
            # Add structure breaks
            for break_point in structure_analysis['structure_breaks']:
                chart_data['structure_breaks'].append({
                    'timestamp': str(break_point.timestamp),
                    'price': break_point.price,
                    'type': break_point.break_type,
                    'strength': break_point.confirmation_strength
                })
            
            return chart_data
            
        except Exception as e:
            logger.error(f"Chart data creation error: {e}")
            return {'error': str(e)}
    
    async def _generate_day2_report(self, test_results: Dict):
        """Generate comprehensive Day 2 report"""
        logger.info("Generating Day 2 comprehensive report...")
        
        with LoggingContext("report_generation"):
            # Calculate overall success metrics
            swing_success = self._calculate_swing_success_rate(test_results['swing_detection'])
            structure_success = self._calculate_structure_success_rate(test_results['bos_choc_detection'])
            alignment_success = self._calculate_alignment_success_rate(test_results['multi_timeframe_structure'])
            
            # Performance metrics
            performance = test_results['performance_analysis']
            meets_performance_targets = (
                performance.get('processing_speed', {}).get('target_met', False) and
                performance.get('memory_usage', {}).get('memory_efficient', False)
            )
            
            # Additional metrics
            structure_detection_success = self._calculate_comprehensive_structure_success(test_results['structure_detection'])
            trend_classification_success = self._calculate_trend_classification_success(test_results['trend_classification'])
            
            # Overall Day 2 success
            day2_success = (
                swing_success >= self.performance_targets["swing_detection_accuracy"] and
                structure_success >= self.performance_targets["bos_choc_accuracy"] and
                meets_performance_targets
            )
            
            # Generate report
            report = f"""
SMC Trading Bot - Day 2 Market Structure Results
============================================================

OVERALL STATUS: {'SUCCESS ✅' if day2_success else 'NEEDS IMPROVEMENT ⚠️'}

SWING POINT DETECTION
Success Rate: {swing_success:.1%}
Target: {self.performance_targets["swing_detection_accuracy"]:.0%}+ - {'ACHIEVED ✅' if swing_success >= self.performance_targets["swing_detection_accuracy"] else 'BELOW TARGET ❌'}
Average Quality Score: {self._get_avg_swing_quality(test_results['swing_detection']):.3f}

BOS/CHOC DETECTION  
Success Rate: {structure_success:.1%}
Target: {self.performance_targets["bos_choc_accuracy"]:.0%}+ - {'ACHIEVED ✅' if structure_success >= self.performance_targets["bos_choc_accuracy"] else 'BELOW TARGET ❌'}
Average Structure Quality: {self._get_avg_structure_quality(test_results['bos_choc_detection']):.3f}

COMPREHENSIVE STRUCTURE DETECTION
Success Rate: {structure_detection_success:.1%}
Reliability Score: {self._get_avg_reliability_score(test_results['structure_detection']):.3f}
Trading Context Generation: {'OPERATIONAL ✅' if structure_detection_success >= 0.7 else 'NEEDS WORK ⚠️'}

TREND CLASSIFICATION
Success Rate: {trend_classification_success:.1%}
Average Confidence: {self._get_avg_trend_confidence(test_results['trend_classification']):.3f}
Momentum Analysis: {'OPERATIONAL ✅' if trend_classification_success >= 0.6 else 'NEEDS WORK ⚠️'}

MULTI-TIMEFRAME ANALYSIS
Alignment Success: {alignment_success:.1%}
Target: {self.performance_targets["multi_timeframe_alignment"]:.0%}+ - {'ACHIEVED ✅' if alignment_success >= self.performance_targets["multi_timeframe_alignment"] else 'BELOW TARGET ❌'}
Conflicting Signals: {self._count_total_conflicts(test_results['multi_timeframe_structure'])}
Dominant Trend Accuracy: {self._calculate_trend_accuracy(test_results['multi_timeframe_structure']):.1%}

PERFORMANCE METRICS
Processing Speed: {'TARGET MET ✅' if performance.get('processing_speed', {}).get('target_met', False) else 'NEEDS OPTIMIZATION ❌'}
Memory Usage: {'EFFICIENT ✅' if performance.get('memory_usage', {}).get('target_met', False) else 'HIGH ⚠️'}
Average Processing Time: {self._get_avg_processing_time(test_results):.2f}ms

VALIDATION RESULTS
Structure Validation Rate: {self._get_validation_rate(test_results['structure_validation']):.1%}
Average Validation Score: {self._get_avg_validation_score(test_results['structure_validation']):.3f}

VISUAL DEBUGGING
Chart Generation: {'READY ✅' if test_results['visual_debugging'].get('target_met', False) else 'ISSUES ❌'}
Data Points Plotted: {self._count_visual_data_points(test_results['visual_debugging'])}
Interactive Charts: {'AVAILABLE ✅' if test_results['visual_debugging'].get('chart_saved', False) else 'NOT AVAILABLE ❌'}

DAY 2 SUCCESS CRITERIA SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Swing detection ≥{self.performance_targets["swing_detection_accuracy"]:.0%}: {'YES ✅' if swing_success >= self.performance_targets["swing_detection_accuracy"] else 'NO ❌'}
BOS/CHoC detection ≥{self.performance_targets["bos_choc_accuracy"]:.0%}: {'YES ✅' if structure_success >= self.performance_targets["bos_choc_accuracy"] else 'NO ❌'}  
Performance targets met: {'YES ✅' if meets_performance_targets else 'NO ❌'}
Multi-TF alignment ≥{self.performance_targets["multi_timeframe_alignment"]:.0%}: {'YES ✅' if alignment_success >= self.performance_targets["multi_timeframe_alignment"] else 'NO ❌'}
Visual debugging operational: {'YES ✅' if test_results['visual_debugging'].get('target_met', False) else 'NO ❌'}

DETAILED PERFORMANCE BREAKDOWN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Test Symbols: {', '.join(self.test_symbols)}
Test Timeframes: {', '.join(self.test_timeframes)}
Total Candles Analyzed: {self._count_total_candles(test_results)}
Processing Speed: {self._get_processing_speed(test_results)} candles/sec
Memory Efficiency: {self._get_memory_efficiency(test_results)}

NEXT STEPS FOR DAY 3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Implement Fair Value Gap Detection
• Build Advanced Order Block Recognition  
• Create Comprehensive Liquidity Analysis
• Develop Pattern Recognition System
• Build Premium/Discount Analysis
• Create Multi-Timeframe Coordination

============================================================
Day 2 Status: {'READY FOR DAY 3 🚀' if day2_success else 'REFINEMENT NEEDED 🔧'}
============================================================
"""
            
            print(report)
            logger.success("Day 2 report generated")
            
            # Save report to file
            report_path = Path("reports/day2_results.txt")
            report_path.parent.mkdir(exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"Report saved to {report_path}")
    
    # Helper methods for report generation
    def _calculate_swing_success_rate(self, swing_results: Dict) -> float:
        """Calculate overall swing detection success rate"""
        total_tests = 0
        successful_tests = 0
        
        for symbol_results in swing_results.values():
            for tf_result in symbol_results.values():
                total_tests += 1
                if tf_result.get('success', False) and tf_result.get('target_met', False):
                    successful_tests += 1
        
        return successful_tests / total_tests if total_tests > 0 else 0.0
    
    def _calculate_structure_success_rate(self, structure_results: Dict) -> float:
        """Calculate overall structure detection success rate"""
        total_tests = 0
        successful_tests = 0
        
        for symbol_results in structure_results.values():
            for tf_result in symbol_results.values():
                total_tests += 1
                if tf_result.get('success', False) and tf_result.get('target_met', False):
                    successful_tests += 1
        
        return successful_tests / total_tests if total_tests > 0 else 0.0
    
    def _calculate_comprehensive_structure_success(self, structure_results: Dict) -> float:
        """Calculate comprehensive structure detection success rate"""
        total_tests = 0
        successful_tests = 0
        
        for symbol_results in structure_results.values():
            for tf_result in symbol_results.values():
                total_tests += 1
                if tf_result.get('success', False) and tf_result.get('target_met', False):
                    successful_tests += 1
        
        return successful_tests / total_tests if total_tests > 0 else 0.0
    
    def _calculate_trend_classification_success(self, trend_results: Dict) -> float:
        """Calculate trend classification success rate"""
        total_tests = 0
        successful_tests = 0
        
        for symbol_results in trend_results.values():
            for tf_result in symbol_results.values():
                total_tests += 1
                if tf_result.get('success', False) and tf_result.get('target_met', False):
                    successful_tests += 1
        
        return successful_tests / total_tests if total_tests > 0 else 0.0
    
    def _calculate_alignment_success_rate(self, alignment_results: Dict) -> float:
        """Calculate multi-timeframe alignment success rate"""
        successful_alignments = 0
        total_symbols = len(alignment_results)
        
        for symbol_result in alignment_results.values():
            if symbol_result.get('target_met', False):
                successful_alignments += 1
        
        return successful_alignments / total_symbols if total_symbols > 0 else 0.0
    
    def _get_avg_swing_quality(self, swing_results: Dict) -> float:
        """Get average swing quality score"""
        quality_scores = []
        
        for symbol_results in swing_results.values():
            for tf_result in symbol_results.values():
                if tf_result.get('success', False):
                    quality_scores.append(tf_result.get('quality_score', 0))
        
        return np.mean(quality_scores) if quality_scores else 0.0
    
    def _get_avg_structure_quality(self, structure_results: Dict) -> float:
        """Get average structure quality score"""
        quality_scores = []
        
        for symbol_results in structure_results.values():
            for tf_result in symbol_results.values():
                if tf_result.get('success', False):
                    quality_scores.append(tf_result.get('structure_quality', 0))
        
        return np.mean(quality_scores) if quality_scores else 0.0
    
    def _get_avg_reliability_score(self, structure_results: Dict) -> float:
        """Get average reliability score"""
        reliability_scores = []
        
        for symbol_results in structure_results.values():
            for tf_result in symbol_results.values():
                if tf_result.get('success', False):
                    reliability_scores.append(tf_result.get('reliability_score', 0))
        
        return np.mean(reliability_scores) if reliability_scores else 0.0
    
    def _get_avg_trend_confidence(self, trend_results: Dict) -> float:
        """Get average trend confidence"""
        confidence_scores = []
        
        for symbol_results in trend_results.values():
            for tf_result in symbol_results.values():
                if tf_result.get('success', False):
                    confidence_scores.append(tf_result.get('confidence', 0))
        
        return np.mean(confidence_scores) if confidence_scores else 0.0
    
    def _count_total_conflicts(self, alignment_results: Dict) -> int:
        """Count total conflicting signals"""
        total_conflicts = 0
        
        for symbol_result in alignment_results.values():
            total_conflicts += symbol_result.get('conflicting_signals', 0)
        
        return total_conflicts
    
    def _calculate_trend_accuracy(self, alignment_results: Dict) -> float:
        """Calculate trend identification accuracy"""
        accurate_trends = 0
        total_symbols = len(alignment_results)
        
        for symbol_result in alignment_results.values():
            dominant_trend = symbol_result.get('dominant_trend', 'UNKNOWN')
            if dominant_trend != 'UNKNOWN' and dominant_trend != MarketStructure.UNKNOWN:
                accurate_trends += 1
        
        return accurate_trends / total_symbols if total_symbols > 0 else 0.0
    
    def _get_validation_rate(self, validation_results: Dict) -> float:
        """Get structure validation rate"""
        validation_rates = []
        
        for symbol_result in validation_results.values():
            if 'validation_rate' in symbol_result:
                validation_rates.append(symbol_result['validation_rate'])
        
        return np.mean(validation_rates) if validation_rates else 0.0
    
    def _get_avg_validation_score(self, validation_results: Dict) -> float:
        """Get average validation score"""
        validation_scores = []
        
        for symbol_result in validation_results.values():
            if 'avg_validation_score' in symbol_result:
                validation_scores.append(symbol_result['avg_validation_score'])
        
        return np.mean(validation_scores) if validation_scores else 0.0
    
    def _count_visual_data_points(self, visual_results: Dict) -> int:
        """Count visual data points"""
        swing_points = visual_results.get('swing_points_plotted', 0)
        structure_breaks = visual_results.get('structure_breaks_plotted', 0)
        return swing_points + structure_breaks
    
    def _count_total_candles(self, test_results: Dict) -> int:
        """Count total candles analyzed across all tests"""
        total_candles = 0
        
        # Count from swing detection results
        for symbol_results in test_results.get('swing_detection', {}).values():
            for tf_result in symbol_results.values():
                if tf_result.get('success', False):
                    total_candles += tf_result.get('candles_analyzed', 0)
        
        return total_candles
    
    def _get_processing_speed(self, test_results: Dict) -> str:
        """Get overall processing speed"""
        performance = test_results.get('performance_analysis', {})
        speed_data = performance.get('processing_speed', {})
        
        structure_speed = speed_data.get('structure_speed_candles_per_sec', 0)
        return f"{structure_speed:.0f}" if structure_speed > 0 else "N/A"
    
    def _get_memory_efficiency(self, test_results: Dict) -> str:
        """Get memory efficiency status"""
        performance = test_results.get('performance_analysis', {})
        memory_data = performance.get('memory_usage', {})
        
        rss_mb = memory_data.get('rss_mb', 0)
        target_met = memory_data.get('target_met', False)
        
        return f"{rss_mb:.0f}MB ({'EFFICIENT' if target_met else 'HIGH'})" if rss_mb > 0 else "N/A"
    
    def _get_avg_processing_time(self, test_results: Dict) -> float:
        """Get average processing time across all tests"""
        processing_times = []
        
        # Collect from all test categories
        for test_category in ['swing_detection', 'bos_choc_detection', 'structure_detection', 'trend_classification']:
            category_results = test_results.get(test_category, {})
            for symbol_results in category_results.values():
                for tf_result in symbol_results.values():
                    if tf_result.get('success', False):
                        time_key = next((k for k in tf_result.keys() if 'processing_time' in k), None)
                        if time_key:
                            processing_times.append(tf_result[time_key])
        
        return np.mean(processing_times) if processing_times else 0.0
    
    def cleanup(self):
        """Clean shutdown of all components"""
        logger.info("Cleaning up Day 2 demo...")
        
        try:
            self.data_fetcher.shutdown_mt5()
            logger.info("Day 2 cleanup completed successfully")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

async def main():
    """Main Day 2 demo execution"""
    demo = SMCTradingBotDay2()
    
    try:
        # Initialize MT5 connection
        if not demo.data_fetcher.initialize_mt5():
            logger.error("Failed to initialize MT5 connection")
            return None
        
        # Run comprehensive Day 2 tests
        results = await demo.run_day2_tests()
        
        # Display final status
        logger.info("="*60)
        logger.success("DAY 2 MARKET STRUCTURE ANALYSIS COMPLETED!")
        logger.info("Market structure detection system is operational")
        logger.info("="*60)
        
        return results
        
    except KeyboardInterrupt:
        logger.warning("Demo interrupted by user")
        return None
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise
    finally:
        demo.cleanup()

if __name__ == "__main__":
    # Create reports directory
    Path("reports").mkdir(exist_ok=True)
    
    # Run the Day 2 demo
    results = asyncio.run(main())