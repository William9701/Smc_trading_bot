# utils/constants.py

"""
SMC Trading Bot Constants
Contains all constants used throughout the SMC trading system
"""

import MetaTrader5 as mt5

# Market Structure Constants
class MarketStructure:
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"
    UNKNOWN = "UNKNOWN"

# Structure Break Types
class StructureBreak:
    BOS_BULLISH = "BOS_BULLISH"
    BOS_BEARISH = "BOS_BEARISH"
    CHOC_BULLISH = "CHOC_BULLISH"
    CHOC_BEARISH = "CHOC_BEARISH"

# Trend Strength Classifications
class TrendStrength:
    WEAK = "WEAK"
    MODERATE = "MODERATE" 
    STRONG = "STRONG"
    VERY_STRONG = "VERY_STRONG"

# Pattern Types
class PatternType:
    FAIR_VALUE_GAP = "FAIR_VALUE_GAP"
    ORDER_BLOCK = "ORDER_BLOCK"
    INDUCEMENT = "INDUCEMENT"
    REJECTION_BLOCK = "REJECTION_BLOCK"
    INSTITUTIONAL_CANDLE = "INSTITUTIONAL_CANDLE"
    LIQUIDITY_VOID = "LIQUIDITY_VOID"
    VACUUM_BLOCK = "VACUUM_BLOCK"

# Order Block Types
class OrderBlockType:
    BULLISH_OB = "BULLISH_OB"
    BEARISH_OB = "BEARISH_OB"
    BREAKER_BLOCK = "BREAKER_BLOCK"
    MITIGATION_BLOCK = "MITIGATION_BLOCK"

# Liquidity Types
class LiquidityType:
    BUY_SIDE = "BUY_SIDE"
    SELL_SIDE = "SELL_SIDE"
    EQUAL_HIGHS = "EQUAL_HIGHS"
    EQUAL_LOWS = "EQUAL_LOWS"
    EXTERNAL = "EXTERNAL"
    INTERNAL = "INTERNAL"

# Premium/Discount Zones
class PremiumDiscount:
    PREMIUM = "PREMIUM"
    DISCOUNT = "DISCOUNT"
    EQUILIBRIUM = "EQUILIBRIUM"

# MT5 Timeframe Mappings
MT5_TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M2": mt5.TIMEFRAME_M2,
    "M3": mt5.TIMEFRAME_M3,
    "M4": mt5.TIMEFRAME_M4,
    "M5": mt5.TIMEFRAME_M5,
    "M6": mt5.TIMEFRAME_M6,
    "M10": mt5.TIMEFRAME_M10,
    "M12": mt5.TIMEFRAME_M12,
    "M15": mt5.TIMEFRAME_M15,
    "M20": mt5.TIMEFRAME_M20,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H2": mt5.TIMEFRAME_H2,
    "H3": mt5.TIMEFRAME_H3,
    "H4": mt5.TIMEFRAME_H4,
    "H6": mt5.TIMEFRAME_H6,
    "H8": mt5.TIMEFRAME_H8,
    "H12": mt5.TIMEFRAME_H12,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}

# Reverse mapping for display
TIMEFRAME_NAMES = {v: k for k, v in MT5_TIMEFRAMES.items()}

# Trading Session Constants
class TradingSessions:
    SYDNEY = "SYDNEY"
    TOKYO = "TOKYO"
    LONDON = "LONDON"
    NEW_YORK = "NEW_YORK"

# Session Times (UTC)
SESSION_TIMES = {
    TradingSessions.SYDNEY: {"start": "21:00", "end": "06:00"},
    TradingSessions.TOKYO: {"start": "23:00", "end": "08:00"},
    TradingSessions.LONDON: {"start": "07:00", "end": "16:00"},
    TradingSessions.NEW_YORK: {"start": "12:00", "end": "21:00"}
}

# Analysis Confidence Levels
class ConfidenceLevel:
    VERY_HIGH = "VERY_HIGH"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"

# Trade Entry Types
class EntryType:
    AGGRESSIVE = "AGGRESSIVE"
    CONSERVATIVE = "CONSERVATIVE"
    CONFIRMATION = "CONFIRMATION"

# Risk Levels
class RiskLevel:
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"

# SMC Reference Points (PD Arrays)
class PDArrayType:
    OLD_HIGH = "OLD_HIGH"
    OLD_LOW = "OLD_LOW"
    ORDER_BLOCK = "ORDER_BLOCK"
    REJECTION_BLOCK = "REJECTION_BLOCK"
    MITIGATION_BLOCK = "MITIGATION_BLOCK"
    BREAKER_BLOCK = "BREAKER_BLOCK"
    FAIR_VALUE_GAP = "FAIR_VALUE_GAP"
    LIQUIDITY_VOID = "LIQUIDITY_VOID"

# PD Array Hierarchy (Order of Importance)
BULLISH_DISCOUNT_HIERARCHY = [
    PDArrayType.OLD_LOW,
    PDArrayType.REJECTION_BLOCK,
    PDArrayType.ORDER_BLOCK,
    PDArrayType.FAIR_VALUE_GAP,
    PDArrayType.LIQUIDITY_VOID,
    PDArrayType.BREAKER_BLOCK,
    PDArrayType.MITIGATION_BLOCK
]

BEARISH_PREMIUM_HIERARCHY = [
    PDArrayType.OLD_HIGH,
    PDArrayType.REJECTION_BLOCK,
    PDArrayType.ORDER_BLOCK,
    PDArrayType.FAIR_VALUE_GAP,
    PDArrayType.LIQUIDITY_VOID,
    PDArrayType.BREAKER_BLOCK,
    PDArrayType.MITIGATION_BLOCK
]

# Fibonacci Levels
FIBONACCI_LEVELS = {
    "0.0": 0.0,
    "23.6": 0.236,
    "38.2": 0.382,
    "50.0": 0.500,
    "61.8": 0.618,
    "78.6": 0.786,
    "88.6": 0.886,
    "100.0": 1.0
}

# Common Retracement Levels
RETRACEMENT_LEVELS = [0.382, 0.5, 0.618]

# Volume Analysis Constants
class VolumeType:
    TICK_VOLUME = "TICK_VOLUME"
    REAL_VOLUME = "REAL_VOLUME"

# Candle Patterns
class CandlePattern:
    DOJI = "DOJI"
    HAMMER = "HAMMER"
    HANGING_MAN = "HANGING_MAN"
    SHOOTING_STAR = "SHOOTING_STAR"
    ENGULFING_BULLISH = "ENGULFING_BULLISH"
    ENGULFING_BEARISH = "ENGULFING_BEARISH"

# Market Phases (Wyckoff)
class MarketPhase:
    ACCUMULATION = "ACCUMULATION"
    MARKUP = "MARKUP"
    DISTRIBUTION = "DISTRIBUTION"
    MARKDOWN = "MARKDOWN"

# SMC Concepts Status
class ConceptStatus:
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    MITIGATED = "MITIGATED"
    INVALIDATED = "INVALIDATED"

# Analysis Results Status
class AnalysisStatus:
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"
    PENDING = "PENDING"

# Data Quality Levels
class DataQuality:
    EXCELLENT = "EXCELLENT"  # 95-100%
    GOOD = "GOOD"           # 85-94%
    FAIR = "FAIR"           # 70-84%
    POOR = "POOR"           # 50-69%
    UNUSABLE = "UNUSABLE"   # <50%

# Performance Metrics
class PerformanceLevel:
    OPTIMAL = "OPTIMAL"     # <100ms
    GOOD = "GOOD"          # 100-500ms
    ACCEPTABLE = "ACCEPTABLE"  # 500ms-2s
    SLOW = "SLOW"          # 2s-5s
    UNACCEPTABLE = "UNACCEPTABLE"  # >5s

# Color Scheme for Visualizations
CHART_COLORS = {
    "bullish": "#00ff88",
    "bearish": "#ff4444",
    "neutral": "#888888",
    "swing_high": "#ff6b6b",
    "swing_low": "#4ecdc4",
    "bos": "#ffd93d",
    "choc": "#ff6348",
    "order_block_bullish": "#26de81",
    "order_block_bearish": "#fc5c65",
    "fvg_bullish": "#45aaf2",
    "fvg_bearish": "#fd79a8",
    "background": "#1e1e1e",
    "grid": "#333333",
    "text": "#ffffff"
}

# Default Analysis Parameters
DEFAULT_SWING_LOOKBACK = 5
DEFAULT_ATR_PERIOD = 14
DEFAULT_MOMENTUM_PERIOD = 20
DEFAULT_VOLATILITY_PERIOD = 14

# Minimum Data Requirements
MIN_CANDLES_FOR_SWING_ANALYSIS = 20
MIN_CANDLES_FOR_STRUCTURE_ANALYSIS = 50
MIN_CANDLES_FOR_TREND_ANALYSIS = 100
MIN_CANDLES_FOR_PATTERN_ANALYSIS = 30

# Quality Thresholds
SWING_STRENGTH_THRESHOLD = 0.3
STRUCTURE_QUALITY_THRESHOLD = 0.6
TREND_CONFIDENCE_THRESHOLD = 0.6
PATTERN_CONFIDENCE_THRESHOLD = 0.7

# Performance Targets (Day 2)
DAY2_TARGETS = {
    "swing_detection_accuracy": 0.80,
    "bos_choc_accuracy": 0.85,
    "processing_speed_ms": 100,
    "memory_usage_mb": 800,
    "multi_timeframe_alignment": 0.90
}

# Error Messages
class ErrorMessages:
    INSUFFICIENT_DATA = "Insufficient data for analysis"
    INVALID_TIMEFRAME = "Invalid timeframe specified"
    CONNECTION_FAILED = "Failed to establish MT5 connection"
    DATA_QUALITY_LOW = "Data quality below acceptable threshold"
    ANALYSIS_FAILED = "Analysis failed to complete successfully"
    INVALID_SYMBOL = "Invalid or unavailable symbol"

# Success Messages
class SuccessMessages:
    ANALYSIS_COMPLETE = "Analysis completed successfully"
    DATA_FETCHED = "Data fetched successfully"
    CONNECTION_ESTABLISHED = "MT5 connection established"
    STRUCTURE_DETECTED = "Market structure detected"
    PATTERNS_IDENTIFIED = "Patterns identified successfully"

# File Extensions
SUPPORTED_EXPORT_FORMATS = [".csv", ".xlsx", ".json", ".html", ".png", ".pdf"]

# Cache Settings
CACHE_TIMEOUT_SECONDS = 300  # 5 minutes
MAX_CACHE_SIZE_MB = 500
MAX_CACHED_ITEMS = 1000

# Logging Levels
class LogLevel:
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

# Development Flags
DEBUG_MODE = False
ENABLE_PERFORMANCE_MONITORING = True
ENABLE_DATA_VALIDATION = True
ENABLE_CACHING = True

PD_LEVELS = [0.0, 0.5, 1.0]  # Simplified for P&D

# Trade Directions
class TradeDirection:
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"