# config/settings.py

"""
SMC Trading Bot Configuration Settings
Centralized configuration management for the trading system
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import timedelta

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
CHARTS_DIR = BASE_DIR / "charts"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "ml_models" / "models"

# Create directories if they don't exist
for directory in [DATA_DIR, LOGS_DIR, CHARTS_DIR, REPORTS_DIR, MODELS_DIR]:
    directory.mkdir(exist_ok=True)

@dataclass
class SMCTradingBotSettings:
    """Main configuration class for SMC Trading Bot"""
    
    # MT5 Connection Settings
    MT5_LOGIN: int = int(os.getenv('MT5_LOGIN', '0'))
    MT5_PASSWORD: str = os.getenv('MT5_PASSWORD', '')
    MT5_SERVER: str = os.getenv('MT5_SERVER', '')
    
    # Trading Pairs
    TRADING_PAIRS: List[str] = None
    
    # Timeframe Settings
    PRIMARY_TIMEFRAMES: List[str] = None
    HIGHER_TIMEFRAMES: List[str] = None
    ANALYSIS_TIMEFRAMES: List[str] = None
    
    # Analysis Parameters
    DEFAULT_CANDLE_COUNT: int = 1000
    SWING_LOOKBACK_PERIOD: int = 5
    ATR_PERIOD: int = 14
    MOMENTUM_PERIOD: int = 20
    VOLATILITY_PERIOD: int = 14
    
    # Structure Detection Settings
    MIN_SWING_COUNT: int = 4
    SWING_STRENGTH_THRESHOLD: float = 0.3
    STRUCTURE_QUALITY_THRESHOLD: float = 0.6
    TREND_CONFIDENCE_THRESHOLD: float = 0.6
    
    # BOS/CHoC Detection Settings
    REQUIRE_CLOSE_BREAK: bool = True
    MIN_BREAK_SIZE_PIPS: float = 2.0
    CONFIRMATION_BARS: int = 1
    
    # Performance Settings
    MAX_CACHE_SIZE_MB: int = 500
    CACHE_TIMEOUT_MINUTES: int = 5
    MAX_RETRIES: int = 3
    RETRY_DELAY_SECONDS: float = 1.0
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    LOG_ROTATION: str = "10 MB"
    LOG_RETENTION: str = "30 days"
    
    # Data Quality Settings
    MIN_DATA_QUALITY_SCORE: float = 0.85
    ALLOW_DATA_GAPS: bool = False
    MAX_MISSING_CANDLES_PERCENT: float = 5.0
    
    # Visualization Settings
    CHART_WIDTH: int = 1200
    CHART_HEIGHT: int = 800
    CHART_THEME: str = "plotly_dark"
    
    # Development Settings
    DEBUG_MODE: bool = False
    ENABLE_PERFORMANCE_MONITORING: bool = True
    SAVE_ANALYSIS_RESULTS: bool = True
    
    def __post_init__(self):
        """Initialize default values after dataclass creation"""
        if self.TRADING_PAIRS is None:
            self.TRADING_PAIRS = [
                "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", 
                "NZDUSD", "USDCHF", "EURJPY", "GBPJPY", "AUDJPY"
            ]
        
        if self.PRIMARY_TIMEFRAMES is None:
            self.PRIMARY_TIMEFRAMES = ["M15", "M30", "H1"]
        
        if self.HIGHER_TIMEFRAMES is None:
            self.HIGHER_TIMEFRAMES = ["H4", "D1", "W1"]
        
        if self.ANALYSIS_TIMEFRAMES is None:
            self.ANALYSIS_TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]
    
    def validate_mt5_config(self) -> bool:
        """Validate MT5 configuration"""
        if not self.MT5_LOGIN or not self.MT5_PASSWORD or not self.MT5_SERVER:
            return False
        return True
    
    def get_cache_timeout_seconds(self) -> int:
        """Get cache timeout in seconds"""
        return self.CACHE_TIMEOUT_MINUTES * 60
    
    def get_retry_delay_ms(self) -> int:
        """Get retry delay in milliseconds"""
        return int(self.RETRY_DELAY_SECONDS * 1000)

# Create global settings instance
settings = SMCTradingBotSettings()

# Day 2 Specific Settings
DAY2_TESTING_CONFIG = {
    "test_symbols": ["EURUSD", "GBPUSD"],
    "test_timeframes": ["M15", "H1", "H4"],
    "analysis_candles": 1000,
    "performance_targets": {
        "swing_detection_accuracy": 0.80,
        "bos_choc_accuracy": 0.85,
        "processing_speed_ms": 100,
        "memory_usage_mb": 800,
        "multi_timeframe_alignment": 0.90
    },
    "quality_thresholds": {
        "swing_quality_min": 0.6,
        "structure_quality_min": 0.7,
        "trend_confidence_min": 0.6,
        "validation_rate_min": 0.75
    }
}

# Pattern Detection Settings
PATTERN_DETECTION_CONFIG = {
    "fair_value_gaps": {
        "min_gap_size": 0.001,  # 0.1% minimum gap size
        "max_age_candles": 50,
        "require_imbalance": True
    },
    "order_blocks": {
        "validation_factors": 7,
        "min_confirmation_strength": 0.6,
        "require_liquidity_sweep": True,
        "require_imbalance": True
    },
    "inducements": {
        "use_first_pullback_method": True,
        "min_pullback_size": 0.005,  # 0.5%
        "require_bos_confirmation": True
    },
    "liquidity_zones": {
        "equal_levels_threshold": 0.001,  # 0.1%
        "min_touches": 2,
        "volume_confirmation": True
    }
}

# Risk Management Settings
RISK_MANAGEMENT_CONFIG = {
    "max_risk_per_trade": 0.02,  # 2%
    "max_account_risk": 0.06,    # 6%
    "min_risk_reward_ratio": 2.0,
    "max_correlation": 0.7,
    "position_sizing_method": "fixed_fractional"
}

# Backtesting Settings
BACKTESTING_CONFIG = {
    "start_date": "2022-01-01",
    "end_date": "2024-12-31",
    "initial_balance": 10000,
    "commission_per_lot": 7.0,
    "slippage_pips": 1.0,
    "use_spread": True,
    "account_currency": "USD"
}

# ML Model Settings
ML_MODEL_CONFIG = {
    "model_types": ["random_forest", "xgboost", "neural_network"],
    "feature_engineering": {
        "lookback_periods": [5, 10, 20, 50],
        "technical_indicators": True,
        "price_patterns": True,
        "market_structure_features": True
    },
    "training": {
        "validation_split": 0.2,
        "test_split": 0.1,
        "cross_validation_folds": 5,
        "hyperparameter_optimization": True
    },
    "model_selection": {
        "primary_metric": "accuracy",
        "secondary_metrics": ["precision", "recall", "f1_score"],
        "min_accuracy_threshold": 0.65
    }
}

# Database Settings (for future use)
DATABASE_CONFIG = {
    "type": "sqlite",  # or postgresql, mysql
    "name": "smc_trading_bot.db",
    "host": "localhost",
    "port": 5432,
    "username": "",
    "password": "",
    "pool_size": 5,
    "max_overflow": 10
}

# API Settings (for future integrations)
API_CONFIG = {
    "telegram": {
        "bot_token": os.getenv('TELEGRAM_BOT_TOKEN', ''),
        "chat_id": os.getenv('TELEGRAM_CHAT_ID', ''),
        "enable_notifications": False
    },
    "discord": {
        "webhook_url": os.getenv('DISCORD_WEBHOOK_URL', ''),
        "enable_notifications": False
    },
    "email": {
        "smtp_server": os.getenv('SMTP_SERVER', ''),
        "smtp_port": int(os.getenv('SMTP_PORT', '587')),
        "username": os.getenv('EMAIL_USERNAME', ''),
        "password": os.getenv('EMAIL_PASSWORD', ''),
        "enable_notifications": False
    }
}

# Export Settings
EXPORT_CONFIG = {
    "supported_formats": ["csv", "xlsx", "json", "html", "png", "pdf"],
    "default_format": "csv",
    "include_metadata": True,
    "compress_exports": False,
    "export_directory": str(DATA_DIR / "exports")
}

# Security Settings
SECURITY_CONFIG = {
    "enable_encryption": False,
    "encryption_key": os.getenv('ENCRYPTION_KEY', ''),
    "session_timeout_minutes": 60,
    "max_login_attempts": 5,
    "lockout_duration_minutes": 30
}

# Performance Monitoring
PERFORMANCE_CONFIG = {
    "enable_profiling": settings.DEBUG_MODE,
    "log_slow_operations": True,
    "slow_operation_threshold_ms": 1000,
    "memory_monitoring": True,
    "memory_alert_threshold_mb": 1024,
    "cpu_monitoring": False
}

# Feature Flags
FEATURE_FLAGS = {
    "enable_ai_models": True,
    "enable_advanced_patterns": True,
    "enable_multi_timeframe_analysis": True,
    "enable_volume_analysis": True,
    "enable_session_analysis": False,
    "enable_news_integration": False,
    "enable_social_sentiment": False,
    "enable_market_scanner": True,
    "enable_backtesting": True,
    "enable_paper_trading": True,
    "enable_live_trading": False  # Disabled by default for safety
}

def get_environment() -> str:
    """Get current environment"""
    return os.getenv('ENVIRONMENT', 'development').lower()

def is_development() -> bool:
    """Check if running in development environment"""
    return get_environment() == 'development'

def is_production() -> bool:
    """Check if running in production environment"""
    return get_environment() == 'production'

def is_testing() -> bool:
    """Check if running in testing environment"""
    return get_environment() == 'testing'

def get_config_for_environment(config_name: str) -> Dict[str, Any]:
    """Get configuration for specific environment"""
    environment = get_environment()
    
    configs = {
        'development': {
            'log_level': 'DEBUG',
            'enable_caching': False,
            'save_debug_data': True,
            'performance_monitoring': True
        },
        'testing': {
            'log_level': 'WARNING',
            'enable_caching': False,
            'use_mock_data': True,
            'fast_mode': True
        },
        'production': {
            'log_level': 'INFO',
            'enable_caching': True,
            'save_debug_data': False,
            'performance_monitoring': True,
            'strict_validation': True
        }
    }
    
    return configs.get(environment, {})

def update_settings_from_env():
    """Update settings from environment variables"""
    # Update MT5 settings
    if os.getenv('MT5_LOGIN'):
        settings.MT5_LOGIN = int(os.getenv('MT5_LOGIN'))
    
    if os.getenv('MT5_PASSWORD'):
        settings.MT5_PASSWORD = os.getenv('MT5_PASSWORD')
    
    if os.getenv('MT5_SERVER'):
        settings.MT5_SERVER = os.getenv('MT5_SERVER')
    
    # Update debug mode
    if os.getenv('DEBUG_MODE'):
        settings.DEBUG_MODE = os.getenv('DEBUG_MODE').lower() == 'true'
    
    # Update log level
    if os.getenv('LOG_LEVEL'):
        settings.LOG_LEVEL = os.getenv('LOG_LEVEL').upper()

# Initialize environment-specific settings
update_settings_from_env()

# Validation functions
def validate_configuration() -> Dict[str, bool]:
    """Validate all configuration settings"""
    validation_results = {
        'mt5_config': settings.validate_mt5_config(),
        'directories_exist': all(d.exists() for d in [DATA_DIR, LOGS_DIR, CHARTS_DIR, REPORTS_DIR]),
        'trading_pairs_valid': len(settings.TRADING_PAIRS) > 0,
        'timeframes_valid': len(settings.PRIMARY_TIMEFRAMES) > 0,
        'thresholds_valid': all([
            0 < settings.SWING_STRENGTH_THRESHOLD < 1,
            0 < settings.STRUCTURE_QUALITY_THRESHOLD < 1,
            0 < settings.TREND_CONFIDENCE_THRESHOLD < 1
        ])
    }
    
    return validation_results

def get_configuration_summary() -> Dict[str, Any]:
    """Get summary of current configuration"""
    return {
        'environment': get_environment(),
        'debug_mode': settings.DEBUG_MODE,
        'mt5_configured': settings.validate_mt5_config(),
        'trading_pairs_count': len(settings.TRADING_PAIRS),
        'primary_timeframes': settings.PRIMARY_TIMEFRAMES,
        'feature_flags': {k: v for k, v in FEATURE_FLAGS.items() if v},
        'directories': {
            'data': str(DATA_DIR),
            'logs': str(LOGS_DIR),
            'charts': str(CHARTS_DIR),
            'reports': str(REPORTS_DIR)
        }
    }

# Export commonly used settings
__all__ = [
    'settings',
    'DAY2_TESTING_CONFIG',
    'PATTERN_DETECTION_CONFIG', 
    'RISK_MANAGEMENT_CONFIG',
    'BACKTESTING_CONFIG',
    'ML_MODEL_CONFIG',
    'FEATURE_FLAGS',
    'BASE_DIR',
    'DATA_DIR',
    'LOGS_DIR',
    'REPORTS_DIR',
    'validate_configuration',
    'get_configuration_summary',
    'get_environment',
    'is_development',
    'is_production',
    'is_testing'
]