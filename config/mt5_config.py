# config/mt5_config.py
import MetaTrader5 as mt5
from enum import Enum
from typing import Dict, Any

class MT5Timeframes(Enum):
    """MT5 timeframe constants with readable names"""
    M1 = mt5.TIMEFRAME_M1
    M2 = mt5.TIMEFRAME_M2  
    M3 = mt5.TIMEFRAME_M3
    M4 = mt5.TIMEFRAME_M4
    M5 = mt5.TIMEFRAME_M5
    M6 = mt5.TIMEFRAME_M6
    M10 = mt5.TIMEFRAME_M10
    M12 = mt5.TIMEFRAME_M12
    M15 = mt5.TIMEFRAME_M15
    M20 = mt5.TIMEFRAME_M20
    M30 = mt5.TIMEFRAME_M30
    H1 = mt5.TIMEFRAME_H1
    H2 = mt5.TIMEFRAME_H2
    H3 = mt5.TIMEFRAME_H3
    H4 = mt5.TIMEFRAME_H4
    H6 = mt5.TIMEFRAME_H6
    H8 = mt5.TIMEFRAME_H8
    H12 = mt5.TIMEFRAME_H12
    D1 = mt5.TIMEFRAME_D1
    W1 = mt5.TIMEFRAME_W1
    MN1 = mt5.TIMEFRAME_MN1

def get_mt5_timeframe(timeframe_str: str) -> int:
    """Convert string timeframe to MT5 constant"""
    try:
        return MT5Timeframes[timeframe_str].value
    except KeyError:
        raise ValueError(f"Invalid timeframe: {timeframe_str}")

def get_timeframe_name(timeframe_code: int) -> str:
    """Convert MT5 timeframe code to readable name"""
    for tf in MT5Timeframes:
        if tf.value == timeframe_code:
            return tf.name
    return f"UNKNOWN_TF_{timeframe_code}"

# MT5 symbol specifications for major pairs
MT5_SYMBOL_CONFIG: Dict[str, Dict[str, Any]] = {
    "EURUSD": {"digits": 5, "tick_size": 0.00001, "tick_value": 1.0},
    "GBPUSD": {"digits": 5, "tick_size": 0.00001, "tick_value": 1.0},
    "USDJPY": {"digits": 3, "tick_size": 0.001, "tick_value": 1.0},
    "AUDUSD": {"digits": 5, "tick_size": 0.00001, "tick_value": 1.0},
    "USDCAD": {"digits": 5, "tick_size": 0.00001, "tick_value": 1.0},
    "NZDUSD": {"digits": 5, "tick_size": 0.00001, "tick_value": 1.0},
    "USDCHF": {"digits": 5, "tick_size": 0.00001, "tick_value": 1.0},
    "EURJPY": {"digits": 3, "tick_size": 0.001, "tick_value": 1.0},
}