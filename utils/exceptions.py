# utils/exceptions.py

"""
SMC Trading Bot Custom Exceptions
Comprehensive exception hierarchy for error handling
"""

class SMCTradingBotError(Exception):
    """Base exception for all SMC Trading Bot errors"""
    
    def __init__(self, message: str, error_code: str = None, context: dict = None):
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        super().__init__(self.message)
    
    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message

# Data-related exceptions
class DataError(SMCTradingBotError):
    """Base class for data-related errors"""
    pass

class DataFetchError(DataError):
    """Error occurred while fetching data"""
    pass

class DataValidationError(DataError):
    """Data validation failed"""
    pass

class DataQualityError(DataError):
    """Data quality below acceptable threshold"""
    pass

class InsufficientDataError(DataError):
    """Not enough data for analysis"""
    pass

# Connection-related exceptions
class ConnectionError(SMCTradingBotError):
    """Base class for connection errors"""
    pass

class MT5ConnectionError(ConnectionError):
    """MT5 connection failed"""
    pass

class NetworkError(ConnectionError):
    """Network connectivity issues"""
    pass

# Analysis-related exceptions
class AnalysisError(SMCTradingBotError):
    """Base class for analysis errors"""
    pass

class SwingAnalysisError(AnalysisError):
    """Swing point analysis failed"""
    pass

class StructureAnalysisError(AnalysisError):
    """Market structure analysis failed"""
    pass

class PatternDetectionError(AnalysisError):
    """Pattern detection failed"""
    pass

class TrendAnalysisError(AnalysisError):
    """Trend classification failed"""
    pass

# Configuration-related exceptions
class ConfigurationError(SMCTradingBotError):
    """Base class for configuration errors"""
    pass

class InvalidConfigurationError(ConfigurationError):
    """Invalid configuration detected"""
    pass

class MissingConfigurationError(ConfigurationError):
    """Required configuration missing"""
    pass

# Trading-related exceptions
class TradingError(SMCTradingBotError):
    """Base class for trading errors"""
    pass

class OrderExecutionError(TradingError):
    """Order execution failed"""
    pass

class RiskManagementError(TradingError):
    """Risk management violation"""
    pass

class InsufficientFundsError(TradingError):
    """Insufficient account funds"""
    pass

# Validation-related exceptions
class ValidationError(SMCTradingBotError):
    """Base class for validation errors"""
    pass

class SymbolValidationError(ValidationError):
    """Invalid trading symbol"""
    pass

class TimeframeValidationError(ValidationError):
    """Invalid timeframe"""
    pass

class ParameterValidationError(ValidationError):
    """Invalid parameter value"""
    pass

# Performance-related exceptions
class PerformanceError(SMCTradingBotError):
    """Base class for performance errors"""
    pass

class TimeoutError(PerformanceError):
    """Operation timed out"""
    pass

class MemoryError(PerformanceError):
    """Memory usage exceeded limits"""
    pass

class ProcessingError(PerformanceError):
    """Processing performance degraded"""
    pass

# System-related exceptions
class SystemError(SMCTradingBotError):
    """Base class for system errors"""
    pass

class FileSystemError(SystemError):
    """File system operation failed"""
    pass

class DatabaseError(SystemError):
    """Database operation failed"""
    pass

class CacheError(SystemError):
    """Cache operation failed"""
    pass

# Model-related exceptions (for ML components)
class ModelError(SMCTradingBotError):
    """Base class for ML model errors"""
    pass

class ModelTrainingError(ModelError):
    """Model training failed"""
    pass

class ModelPredictionError(ModelError):
    """Model prediction failed"""
    pass

class ModelLoadError(ModelError):
    """Model loading failed"""
    pass

# Visualization-related exceptions
class VisualizationError(SMCTradingBotError):
    """Base class for visualization errors"""
    pass

class ChartGenerationError(VisualizationError):
    """Chart generation failed"""
    pass

class ExportError(VisualizationError):
    """Data export failed"""
    pass

# Utility functions for exception handling
def handle_mt5_error(mt5_error_code: int, mt5_error_message: str) -> MT5ConnectionError:
    """Convert MT5 error to custom exception"""
    error_map = {
        1: "No error",
        2: "Common error", 
        3: "Invalid trade parameters",
        4: "Trade server is busy",
        5: "Old version of the client terminal",
        6: "No connection with trade server",
        7: "Not enough rights",
        8: "Too frequent requests",
        9: "Malfunctional trade operation",
        10: "Account disabled",
        11: "Invalid account",
        12: "Trade is prohibited",
        64: "Account disconnected"
    }
    
    error_description = error_map.get(mt5_error_code, "Unknown MT5 error")
    return MT5ConnectionError(
        f"MT5 Error {mt5_error_code}: {error_description} - {mt5_error_message}",
        error_code=f"MT5_{mt5_error_code}"
    )

def create_analysis_error(analysis_type: str, symbol: str, timeframe: str, error_message: str) -> AnalysisError:
    """Create specific analysis error with context"""
    context = {
        'analysis_type': analysis_type,
        'symbol': symbol,
        'timeframe': timeframe
    }
    
    error_classes = {
        'swing': SwingAnalysisError,
        'structure': StructureAnalysisError, 
        'pattern': PatternDetectionError,
        'trend': TrendAnalysisError
    }
    
    error_class = error_classes.get(analysis_type.lower(), AnalysisError)
    
    return error_class(
        f"{analysis_type.title()} analysis failed for {symbol} {timeframe}: {error_message}",
        error_code=f"ANALYSIS_{analysis_type.upper()}_FAILED",
        context=context
    )

def create_data_error(operation: str, symbol: str, timeframe: str, error_message: str) -> DataError:
    """Create specific data error with context"""
    context = {
        'operation': operation,
        'symbol': symbol,
        'timeframe': timeframe
    }
    
    return DataFetchError(
        f"Data {operation} failed for {symbol} {timeframe}: {error_message}",
        error_code=f"DATA_{operation.upper()}_FAILED",
        context=context
    )

def validate_and_raise(condition: bool, exception_class: type, message: str, error_code: str = None):
    """Validate condition and raise exception if false"""
    if not condition:
        raise exception_class(message, error_code=error_code)

# Context managers for error handling
class ErrorContext:
    """Context manager for structured error handling"""
    
    def __init__(self, operation: str, reraise_as: type = None):
        self.operation = operation
        self.reraise_as = reraise_as or SMCTradingBotError
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and not isinstance(exc_val, SMCTradingBotError):
            # Convert generic exceptions to SMC-specific exceptions
            raise self.reraise_as(
                f"Error in {self.operation}: {str(exc_val)}",
                error_code=f"{self.operation.upper()}_ERROR"
            ) from exc_val

# Exception logging helper
def log_exception(logger, exception: Exception, context: dict = None):
    """Log exception with appropriate level and context"""
    context = context or {}
    
    if isinstance(exception, SMCTradingBotError):
        logger.error(f"SMC Error: {exception.message}")
        if exception.error_code:
            logger.error(f"Error Code: {exception.error_code}")
        if exception.context:
            logger.error(f"Context: {exception.context}")
    else:
        logger.error(f"Unexpected Error: {str(exception)}")
    
    if context:
        logger.error(f"Additional Context: {context}")

# Error recovery strategies
class ErrorRecovery:
    """Error recovery strategies"""
    
    @staticmethod
    def retry_with_backoff(func, max_attempts: int = 3, base_delay: float = 1.0):
        """Retry function with exponential backoff"""
        import time
        
        for attempt in range(max_attempts):
            try:
                return func()
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise
                
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
    
    @staticmethod
    def fallback_on_error(primary_func, fallback_func, fallback_exceptions: tuple = None):
        """Execute fallback function on specific errors"""
        fallback_exceptions = fallback_exceptions or (Exception,)
        
        try:
            return primary_func()
        except fallback_exceptions as e:
            return fallback_func()

# Export all exception classes
__all__ = [
    'SMCTradingBotError',
    'DataError', 'DataFetchError', 'DataValidationError', 'DataQualityError', 'InsufficientDataError',
    'ConnectionError', 'MT5ConnectionError', 'NetworkError',
    'AnalysisError', 'SwingAnalysisError', 'StructureAnalysisError', 'PatternDetectionError', 'TrendAnalysisError',
    'ConfigurationError', 'InvalidConfigurationError', 'MissingConfigurationError',
    'TradingError', 'OrderExecutionError', 'RiskManagementError', 'InsufficientFundsError',
    'ValidationError', 'SymbolValidationError', 'TimeframeValidationError', 'ParameterValidationError',
    'PerformanceError', 'TimeoutError', 'MemoryError', 'ProcessingError',
    'SystemError', 'FileSystemError', 'DatabaseError', 'CacheError',
    'ModelError', 'ModelTrainingError', 'ModelPredictionError', 'ModelLoadError',
    'VisualizationError', 'ChartGenerationError', 'ExportError',
    'handle_mt5_error', 'create_analysis_error', 'create_data_error', 'validate_and_raise',
    'ErrorContext', 'log_exception', 'ErrorRecovery'
]