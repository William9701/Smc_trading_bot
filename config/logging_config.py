# config/logging_config.py

"""
Logging Configuration for SMC Trading Bot
Professional logging setup with multiple handlers and formatters
"""

import sys
from pathlib import Path
from loguru import logger
from loguru._logger import Logger  

from datetime import datetime
from typing import Dict, Any, Optional

from .settings import settings, LOGS_DIR, get_environment, is_development

def setup_logging(
    log_level: str = None,
    enable_file_logging: bool = True,
    enable_console_logging: bool = True,
    log_to_separate_files: bool = True
) -> None:
    """
    Setup comprehensive logging configuration for SMC Trading Bot
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_file_logging: Enable logging to files
        enable_console_logging: Enable console output
        log_to_separate_files: Create separate files for different log levels
    """
    
    # Remove default handler
    logger.remove()
    
    # Use provided log level or get from settings
    log_level = log_level or settings.LOG_LEVEL
    
    # Ensure logs directory exists
    LOGS_DIR.mkdir(exist_ok=True)
    
    # Console logging setup
    if enable_console_logging:
        console_format = _get_console_format()
        console_level = "DEBUG" if is_development() else "INFO"
        
        logger.add(
            sys.stdout,
            format=console_format,
            level=console_level,
            colorize=True,
            backtrace=True,
            diagnose=is_development()
        )
    
    # File logging setup
    if enable_file_logging:
        _setup_file_logging(log_level, log_to_separate_files)
    
    # Performance logging for development
    if is_development():
        _setup_performance_logging()
    
    # Error logging with enhanced details
    _setup_error_logging()
    
    logger.info("Logging configuration initialized successfully")
    logger.info(f"Environment: {get_environment()}")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Logs directory: {LOGS_DIR}")

def _get_console_format() -> str:
    """Get console log format based on environment"""
    if is_development():
        return (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
    else:
        return (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} - "
            "{message}"
        )

def _get_file_format() -> str:
    """Get file log format"""
    return (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{process} | "
        "{thread} | "
        "{name}:{function}:{line} - "
        "{message}"
    )

def _setup_file_logging(log_level: str, separate_files: bool) -> None:
    """Setup file-based logging handlers"""
    
    file_format = _get_file_format()
    today = datetime.now().strftime("%Y%m%d")
    
    if separate_files:
        # Separate log files by level
        log_files = {
            "DEBUG": LOGS_DIR / f"debug_{today}.log",
            "INFO": LOGS_DIR / f"info_{today}.log", 
            "WARNING": LOGS_DIR / f"warning_{today}.log",
            "ERROR": LOGS_DIR / f"error_{today}.log",
            "CRITICAL": LOGS_DIR / f"critical_{today}.log"
        }
        
        for level, filepath in log_files.items():
            logger.add(
                str(filepath),
                format=file_format,
                level=level,
                rotation=settings.LOG_ROTATION,
                retention=settings.LOG_RETENTION,
                compression="zip",
                backtrace=True,
                diagnose=True,
                filter=lambda record, lvl=level: record["level"].name == lvl
            )
    else:
        # Single combined log file
        combined_log = LOGS_DIR / f"smc_bot_{today}.log"
        logger.add(
            str(combined_log),
            format=file_format,
            level=log_level,
            rotation=settings.LOG_ROTATION,
            retention=settings.LOG_RETENTION,
            compression="zip",
            backtrace=True,
            diagnose=True
        )

def _setup_performance_logging() -> None:
    """Setup performance-specific logging"""
    
    performance_log = LOGS_DIR / "performance.log"
    performance_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "PERF | "
        "{extra[operation]} | "
        "{extra[duration_ms]}ms | "
        "{extra[memory_mb]}MB | "
        "{message}"
    )
    
    logger.add(
        str(performance_log),
        format=performance_format,
        level="INFO",
        rotation="10 MB",
        retention="7 days",
        filter=lambda record: "performance" in record["extra"]
    )

def _setup_error_logging() -> None:
    """Setup enhanced error logging"""
    
    error_log = LOGS_DIR / "errors.log"
    error_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{process} | "
        "{thread} | "
        "{name}:{function}:{line} | "
        "Exception: {exception} | "
        "{message}"
    )
    
    logger.add(
        str(error_log),
        format=error_format,
        level="ERROR",
        rotation="50 MB",
        retention="30 days",
        backtrace=True,
        diagnose=True,
        catch=True
    )

def _setup_trading_logging() -> None:
    """Setup trading-specific logging (for future use)"""
    
    trading_log = LOGS_DIR / "trading.log"
    trading_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "TRADE | "
        "{extra[symbol]} | "
        "{extra[action]} | "
        "{extra[price]} | "
        "{extra[volume]} | "
        "{message}"
    )
    
    logger.add(
        str(trading_log),
        format=trading_format,
        level="INFO",
        rotation="daily",
        retention="90 days",
        filter=lambda record: "trading" in record["extra"]
    )

def _setup_analysis_logging() -> None:
    """Setup analysis-specific logging"""
    
    analysis_log = LOGS_DIR / "analysis.log"
    analysis_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "ANALYSIS | "
        "{extra[symbol]} | "
        "{extra[timeframe]} | "
        "{extra[analysis_type]} | "
        "{extra[duration_ms]}ms | "
        "{message}"
    )
    
    logger.add(
        str(analysis_log),
        format=analysis_format,
        level="INFO",
        rotation="20 MB",
        retention="14 days",
        filter=lambda record: "analysis" in record["extra"]
    )

def log_performance(operation: str, duration_ms: float, memory_mb: float = 0, **kwargs):
    """
    Log performance metrics
    
    Args:
        operation: Name of the operation
        duration_ms: Duration in milliseconds  
        memory_mb: Memory usage in MB
        **kwargs: Additional context
    """
    
    extra_data = {
        "performance": True,
        "operation": operation,
        "duration_ms": round(duration_ms, 2),
        "memory_mb": round(memory_mb, 2),
        **kwargs
    }
    
    message = f"Performance: {operation}"
    if duration_ms > 1000:  # Log slow operations
        logger.warning(message, extra=extra_data)
    else:
        logger.info(message, extra=extra_data)

def log_analysis(symbol: str, timeframe: str, analysis_type: str, duration_ms: float, success: bool, **kwargs):
    """
    Log analysis operations
    
    Args:
        symbol: Trading symbol
        timeframe: Analysis timeframe  
        analysis_type: Type of analysis performed
        duration_ms: Duration in milliseconds
        success: Whether analysis was successful
        **kwargs: Additional context
    """
    
    extra_data = {
        "analysis": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "analysis_type": analysis_type,
        "duration_ms": round(duration_ms, 2),
        "success": success,
        **kwargs
    }
    
    level = "info" if success else "warning"
    message = f"Analysis completed: {analysis_type} for {symbol} {timeframe}"
    
    logger.log(level.upper(), message, extra=extra_data)

def log_trading_action(symbol: str, action: str, price: float, volume: float = 0, **kwargs):
    """
    Log trading actions (for future use)
    
    Args:
        symbol: Trading symbol
        action: Trading action (BUY, SELL, CLOSE, etc.)
        price: Execution price
        volume: Trade volume
        **kwargs: Additional context
    """
    
    extra_data = {
        "trading": True,
        "symbol": symbol,
        "action": action,
        "price": price,
        "volume": volume,
        **kwargs
    }
    
    message = f"Trading action: {action} {volume} {symbol} @ {price}"
    logger.info(message, extra=extra_data)

def log_structure_detection(
    symbol: str, 
    timeframe: str, 
    structure_type: str, 
    confidence: float,
    swing_count: int,
    **kwargs
):
    """
    Log market structure detection
    
    Args:
        symbol: Trading symbol
        timeframe: Analysis timeframe
        structure_type: Type of structure detected
        confidence: Detection confidence (0-1)
        swing_count: Number of swing points detected
        **kwargs: Additional context
    """
    
    extra_data = {
        "analysis": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "analysis_type": "structure_detection",
        "structure_type": structure_type,
        "confidence": round(confidence, 3),
        "swing_count": swing_count,
        **kwargs
    }
    
    message = f"Structure detected: {structure_type} ({confidence:.1%} confidence) - {swing_count} swings"
    logger.info(message, extra=extra_data)

def log_pattern_detection(
    symbol: str,
    timeframe: str, 
    pattern_type: str,
    pattern_count: int,
    confidence: float,
    **kwargs
):
    """
    Log pattern detection results
    
    Args:
        symbol: Trading symbol
        timeframe: Analysis timeframe
        pattern_type: Type of pattern detected
        pattern_count: Number of patterns found
        confidence: Average confidence level
        **kwargs: Additional context
    """
    
    extra_data = {
        "analysis": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "analysis_type": "pattern_detection",
        "pattern_type": pattern_type,
        "pattern_count": pattern_count,
        "confidence": round(confidence, 3),
        **kwargs
    }
    
    message = f"Patterns detected: {pattern_count} {pattern_type} ({confidence:.1%} confidence)"
    logger.info(message, extra=extra_data)

def create_custom_logger(name: str, log_file: str = None, level: str = "INFO") -> 'Logger':
    """
    Create a custom logger instance
    
    Args:
        name: Logger name
        log_file: Optional custom log file
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    
    custom_logger = logger.bind(logger_name=name)
    
    if log_file:
        log_path = LOGS_DIR / log_file
        custom_logger.add(
            str(log_path),
            format=_get_file_format(),
            level=level,
            rotation="10 MB",
            retention="7 days",
            filter=lambda record: record["extra"].get("logger_name") == name
        )
    
    return custom_logger

def get_log_files() -> Dict[str, Path]:
    """
    Get list of current log files
    
    Returns:
        Dictionary of log file types and paths
    """
    
    log_files = {}
    
    if LOGS_DIR.exists():
        for log_file in LOGS_DIR.glob("*.log"):
            log_type = log_file.stem.split("_")[0]  # Get prefix before first underscore
            log_files[log_type] = log_file
    
    return log_files

def clean_old_logs(days_to_keep: int = 30) -> int:
    """
    Clean old log files
    
    Args:
        days_to_keep: Number of days of logs to keep
        
    Returns:
        Number of files deleted
    """
    
    if not LOGS_DIR.exists():
        return 0
    
    cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
    deleted_count = 0
    
    for log_file in LOGS_DIR.glob("*.log*"):
        if log_file.stat().st_mtime < cutoff_time:
            try:
                log_file.unlink()
                deleted_count += 1
                logger.debug(f"Deleted old log file: {log_file}")
            except Exception as e:
                logger.error(f"Failed to delete log file {log_file}: {e}")
    
    if deleted_count > 0:
        logger.info(f"Cleaned up {deleted_count} old log files")
    
    return deleted_count

def get_log_stats() -> Dict[str, Any]:
    """
    Get logging statistics
    
    Returns:
        Dictionary with logging statistics
    """
    
    stats = {
        "log_directory": str(LOGS_DIR),
        "total_log_files": 0,
        "total_size_mb": 0,
        "log_files": {},
        "oldest_log": None,
        "newest_log": None
    }
    
    if not LOGS_DIR.exists():
        return stats
    
    log_files = list(LOGS_DIR.glob("*.log*"))
    stats["total_log_files"] = len(log_files)
    
    oldest_time = None
    newest_time = None
    
    for log_file in log_files:
        try:
            file_stats = log_file.stat()
            file_size_mb = file_stats.st_size / (1024 * 1024)
            file_time = datetime.fromtimestamp(file_stats.st_mtime)
            
            stats["total_size_mb"] += file_size_mb
            stats["log_files"][log_file.name] = {
                "size_mb": round(file_size_mb, 2),
                "modified": file_time.isoformat()
            }
            
            if oldest_time is None or file_time < oldest_time:
                oldest_time = file_time
                stats["oldest_log"] = log_file.name
            
            if newest_time is None or file_time > newest_time:
                newest_time = file_time
                stats["newest_log"] = log_file.name
                
        except Exception as e:
            logger.error(f"Error getting stats for {log_file}: {e}")
    
    stats["total_size_mb"] = round(stats["total_size_mb"], 2)
    
    return stats

# Context managers for structured logging
class LoggingContext:
    """Context manager for structured logging with automatic performance tracking"""
    
    def __init__(self, operation: str, symbol: str = None, timeframe: str = None, **kwargs):
        self.operation = operation
        self.symbol = symbol
        self.timeframe = timeframe
        self.extra_data = kwargs
        self.start_time = None
        
    def __enter__(self):
        self.start_time = datetime.now()
        
        context_data = {
            "operation": self.operation,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            **self.extra_data
        }
        
        logger.info(f"Starting {self.operation}", extra=context_data)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds() * 1000
        
        context_data = {
            "operation": self.operation,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "duration_ms": round(duration, 2),
            "success": exc_type is None,
            **self.extra_data
        }
        
        if exc_type is None:
            logger.success(f"Completed {self.operation} in {duration:.2f}ms", extra=context_data)
        else:
            logger.error(f"Failed {self.operation} after {duration:.2f}ms: {exc_val}", extra=context_data)

# Initialize logging for different modules
def setup_module_logging():
    """Setup logging for specific modules"""
    _setup_analysis_logging()
    _setup_trading_logging()
    
    logger.info("Module-specific logging configured")

# Export commonly used functions
__all__ = [
    'setup_logging',
    'log_performance', 
    'log_analysis',
    'log_trading_action',
    'log_structure_detection',
    'log_pattern_detection',
    'create_custom_logger',
    'LoggingContext',
    'get_log_files',
    'clean_old_logs',
    'get_log_stats'
]