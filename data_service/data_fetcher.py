# data_service/data_fetcher.py - Enhanced Professional Version

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pytz
from pathlib import Path
import sys
import time
from dataclasses import dataclass
from loguru import logger

# Import our configuration
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import settings
from config.mt5_config import get_mt5_timeframe, get_timeframe_name, MT5_SYMBOL_CONFIG
from utils.exceptions import DataFetchError, MT5ConnectionError
from utils.helpers import validate_dataframe

@dataclass
class DataRequest:
    """Data request specification"""
    symbol: str
    timeframe: str
    num_candles: Optional[int] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.num_candles and not (self.start_date and self.end_date):
            raise ValueError("Either num_candles or date range must be specified")

@dataclass 
class DataQuality:
    """Data quality metrics"""
    total_candles: int
    missing_candles: int
    invalid_candles: int
    quality_score: float
    gaps: List[Tuple[datetime, datetime]]

class EnhancedMT5DataFetcher:
    """
    Professional MT5 data fetcher with enhanced error handling,
    validation, and monitoring capabilities.
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.mt5_initialized = False
        self.connection_health = False
        self.last_health_check = None
        self.data_cache: Dict[str, pd.DataFrame] = {}
        
        logger.info("Enhanced MT5 Data Fetcher initialized")
    
    def initialize_mt5(self) -> bool:
        """
        Initialize MT5 with enhanced error handling and validation
        """
        if self.mt5_initialized and self.connection_health:
            logger.info("MT5 already initialized and healthy")
            return True
        
        logger.info("Initializing MT5 connection...")
        
        try:
            # Initialize MT5
            if not mt5.initialize(
                login=settings.MT5_LOGIN,
                password=settings.MT5_PASSWORD, 
                server=settings.MT5_SERVER
            ):
                error_msg = f"Failed to initialize MT5: {mt5.last_error()}"
                logger.error(error_msg)
                raise MT5ConnectionError(error_msg)
            
            # Verify connection
            account_info = mt5.account_info()
            if not account_info:
                error_msg = "Failed to get account info after initialization"
                logger.error(error_msg)
                raise MT5ConnectionError(error_msg)
            
            logger.info(f"MT5 initialized successfully - Account: {account_info.login}")
            logger.info(f"Server: {account_info.server}, Balance: {account_info.balance}")
            
            self.mt5_initialized = True
            self.connection_health = True
            self.last_health_check = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"MT5 initialization failed: {e}")
            self.mt5_initialized = False
            self.connection_health = False
            return False
    
    def check_connection_health(self) -> bool:
        """Check MT5 connection health"""
        try:
            # Check if we need to verify health
            if (self.last_health_check and 
                datetime.now() - self.last_health_check < timedelta(minutes=5)):
                return self.connection_health
            
            # Verify connection is still active
            account_info = mt5.account_info()
            terminal_info = mt5.terminal_info()
            
            if not account_info or not terminal_info:
                logger.warning("MT5 connection health check failed")
                self.connection_health = False
                return False
            
            self.connection_health = True
            self.last_health_check = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"Connection health check error: {e}")
            self.connection_health = False
            return False
    
    def ensure_symbol_selected(self, symbol: str) -> bool:
        """Ensure symbol is selected and available"""
        try:
            # Try to select symbol
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select symbol {symbol}")
                return False
            
            # Verify symbol info
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Failed to get symbol info for {symbol}")
                return False
            
            if not symbol_info.visible:
                logger.warning(f"Symbol {symbol} is not visible, attempting to make visible")
                if not mt5.symbol_select(symbol, True):
                    return False
            
            logger.debug(f"Symbol {symbol} selected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Symbol selection error for {symbol}: {e}")
            return False
    
    def fetch_data_with_retry(self, request: DataRequest) -> pd.DataFrame:
        """Fetch data with retry logic"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Data fetch attempt {attempt + 1}/{self.max_retries} for {request.symbol}")
                
                # Ensure connection is healthy
                if not self.check_connection_health():
                    if not self.initialize_mt5():
                        raise MT5ConnectionError("Failed to establish MT5 connection")
                
                # Ensure symbol is selected
                if not self.ensure_symbol_selected(request.symbol):
                    raise DataFetchError(f"Failed to select symbol {request.symbol}")
                
                # Fetch the data
                df = self._fetch_raw_data(request)
                
                if df.empty:
                    raise DataFetchError(f"No data returned for {request.symbol}")
                
                # Validate data quality
                if not validate_dataframe(df):
                    raise DataFetchError(f"Invalid data structure for {request.symbol}")
                
                logger.info(f"Successfully fetched {len(df)} candles for {request.symbol}")
                return df
                
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
        
        # All retries failed
        error_msg = f"Failed to fetch data after {self.max_retries} attempts: {last_error}"
        logger.error(error_msg)
        raise DataFetchError(error_msg)
    
    def _fetch_raw_data(self, request: DataRequest) -> pd.DataFrame:
        """Internal method to fetch raw data from MT5"""
        timeframe = get_mt5_timeframe(request.timeframe)
        
        # Fetch data based on request type
        if request.num_candles:
            rates = mt5.copy_rates_from_pos(request.symbol, timeframe, 0, request.num_candles)
        else:
            # Ensure dates are timezone-aware
            start_dt = request.start_date
            end_dt = request.end_date
            
            if start_dt.tzinfo is None:
                start_dt = pytz.utc.localize(start_dt)
            if end_dt.tzinfo is None:
                end_dt = pytz.utc.localize(end_dt)
            
            rates = mt5.copy_rates_range(
                request.symbol, timeframe,
                start_dt.astimezone(pytz.utc),
                end_dt.astimezone(pytz.utc)
            )
        
        if rates is None:
            mt5_error = mt5.last_error()
            raise DataFetchError(f"MT5 error: {mt5_error}")
        
        if len(rates) == 0:
            raise DataFetchError("No rates returned from MT5")
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Standardize column names
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        
        # Select required columns in correct order
        df = df[['open', 'high', 'low', 'close', 'volume', 'spread', 'real_volume']]
        
        return df
    
    def get_symbol_data(self, symbol: str, timeframe: str, num_candles: int) -> pd.DataFrame:
        """
        Simplified interface for getting symbol data
        Compatible with existing code but with enhanced capabilities
        """
        request = DataRequest(
            symbol=symbol,
            timeframe=timeframe, 
            num_candles=num_candles
        )
        
        try:
            return self.fetch_data_with_retry(request)
        except Exception as e:
            logger.error(f"Error in get_symbol_data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_multi_timeframe_data(
        self, 
        symbol: str, 
        primary_timeframe: str,
        higher_timeframes: List[str], 
        num_candles: int
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple timeframes with enhanced error handling
        """
        all_data = {}
        
        # Combine all timeframes
        all_timeframes = [primary_timeframe] + higher_timeframes
        
        for tf in all_timeframes:
            try:
                logger.info(f"Fetching {tf} data for {symbol}")
                
                request = DataRequest(
                    symbol=symbol,
                    timeframe=tf,
                    num_candles=num_candles
                )
                
                df = self.fetch_data_with_retry(request)
                all_data[tf] = df
                
                # Add data quality assessment
                quality = self.assess_data_quality(df, symbol, tf)
                logger.info(f"Data quality for {symbol} {tf}: {quality.quality_score:.2f}")
                
            except Exception as e:
                logger.error(f"Failed to fetch {tf} data for {symbol}: {e}")
                all_data[tf] = pd.DataFrame()
        
        return all_data
    
    def assess_data_quality(self, df: pd.DataFrame, symbol: str, timeframe: str) -> DataQuality:
        """Assess the quality of fetched data"""
        if df.empty:
            return DataQuality(0, 0, 0, 0.0, [])
        
        total_candles = len(df)
        
        # Check for invalid OHLC relationships
        invalid_mask = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        invalid_candles = invalid_mask.sum()
        
        # Check for missing data (gaps in time series)
        expected_frequency = self._get_expected_frequency(timeframe)
        if expected_frequency:
            expected_index = pd.date_range(
                start=df.index[0],
                end=df.index[-1], 
                freq=expected_frequency
            )
            missing_candles = len(expected_index) - total_candles
            gaps = self._find_gaps(df.index, expected_frequency)
        else:
            missing_candles = 0
            gaps = []
        
        # Calculate quality score
        quality_score = max(0.0, 1.0 - (invalid_candles + missing_candles) / total_candles)
        
        return DataQuality(
            total_candles=total_candles,
            missing_candles=missing_candles,
            invalid_candles=invalid_candles,
            quality_score=quality_score,
            gaps=gaps
        )
    
    def _get_expected_frequency(self, timeframe: str) -> Optional[str]:
        """Get pandas frequency string for timeframe"""
        frequency_map = {
            'M1': '1min', 'M2': '2min', 'M3': '3min', 'M4': '4min', 'M5': '5min',
            'M6': '6min', 'M10': '10min', 'M12': '12min', 'M15': '15min',
            'M20': '20min', 'M30': '30min',
            'H1': '1h', 'H2': '2h', 'H3': '3h', 'H4': '4h', 'H6': '6h',  # Fixed: 'H' -> 'h'
            'H8': '8h', 'H12': '12h',
            'D1': '1D', 'W1': '1W', 'MN1': '1M'
        }
        return frequency_map.get(timeframe)
    
    def _find_gaps(self, time_index: pd.DatetimeIndex, frequency: str) -> List[Tuple[datetime, datetime]]:
        """Find gaps in time series data"""
        gaps = []
        
        for i in range(1, len(time_index)):
            expected_time = time_index[i-1] + pd.Timedelta(frequency)
            if time_index[i] > expected_time:
                gaps.append((time_index[i-1], time_index[i]))
        
        return gaps
    
    def get_historical_data_range(
        self,
        symbol: str,
        timeframe: str, 
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Get historical data for specific date range"""
        request = DataRequest(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        try:
            return self.fetch_data_with_retry(request)
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def preload_data(self, symbols: List[str], timeframes: List[str], num_candles: int = 1000):
        """Preload data for multiple symbols and timeframes"""
        logger.info(f"Preloading data for {len(symbols)} symbols across {len(timeframes)} timeframes")
        
        total_requests = len(symbols) * len(timeframes)
        completed = 0
        
        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    cache_key = f"{symbol}_{timeframe}"
                    df = self.get_symbol_data(symbol, timeframe, num_candles)
                    
                    if not df.empty:
                        self.data_cache[cache_key] = df
                        logger.debug(f"Cached data for {cache_key}")
                    
                    completed += 1
                    logger.info(f"Preload progress: {completed}/{total_requests}")
                    
                except Exception as e:
                    logger.error(f"Failed to preload {symbol} {timeframe}: {e}")
                    completed += 1
        
        logger.info(f"Preload completed: {len(self.data_cache)} datasets cached")
    
    def get_cached_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get data from cache if available"""
        cache_key = f"{symbol}_{timeframe}"
        return self.data_cache.get(cache_key)
    
    def update_cache(self, symbol: str, timeframe: str, new_candles: int = 100):
        """Update cached data with latest candles"""
        cache_key = f"{symbol}_{timeframe}"
        
        if cache_key not in self.data_cache:
            logger.warning(f"No cached data found for {cache_key}")
            return
        
        try:
            # Fetch latest data
            latest_df = self.get_symbol_data(symbol, timeframe, new_candles)
            
            if latest_df.empty:
                logger.warning(f"No new data available for {cache_key}")
                return
            
            # Merge with existing cache
            existing_df = self.data_cache[cache_key]
            
            # Combine and remove duplicates
            combined_df = pd.concat([existing_df, latest_df])
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df = combined_df.sort_index()
            
            # Keep last 5000 candles to manage memory
            if len(combined_df) > 5000:
                combined_df = combined_df.tail(5000)
            
            self.data_cache[cache_key] = combined_df
            logger.debug(f"Updated cache for {cache_key} - {len(combined_df)} candles")
            
        except Exception as e:
            logger.error(f"Failed to update cache for {cache_key}: {e}")
    
    def get_market_info(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive market information for a symbol"""
        try:
            if not self.ensure_symbol_selected(symbol):
                return {}
            
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return {}
            
            # Get current tick
            tick = mt5.symbol_info_tick(symbol)
            
            market_info = {
                'symbol': symbol_info.name,
                'description': symbol_info.description,
                'currency_base': symbol_info.currency_base,
                'currency_profit': symbol_info.currency_profit,
                'digits': symbol_info.digits,
                'point': symbol_info.point,
                'spread': symbol_info.spread,
                'trade_mode': symbol_info.trade_mode,
                'min_lot': symbol_info.volume_min,
                'max_lot': symbol_info.volume_max,
                'lot_step': symbol_info.volume_step,
                'swap_long': symbol_info.swap_long,
                'swap_short': symbol_info.swap_short,
                'current_bid': tick.bid if tick else None,
                'current_ask': tick.ask if tick else None,
                'current_spread': (tick.ask - tick.bid) if tick else None,
                'last_update': tick.time if tick else None
            }
            
            return market_info
            
        except Exception as e:
            logger.error(f"Error getting market info for {symbol}: {e}")
            return {}
    
    def shutdown_mt5(self):
        """Clean shutdown of MT5 connection"""
        if self.mt5_initialized:
            # Clear cache
            self.data_cache.clear()
            
            # Shutdown MT5
            mt5.shutdown()
            logger.info("MT5 connection shut down cleanly")
            
            self.mt5_initialized = False
            self.connection_health = False
        else:
            logger.info("MT5 not initialized, nothing to shut down")
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize_mt5()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown_mt5()

# Factory function for easy instantiation
def create_mt5_data_fetcher(max_retries: int = 3, retry_delay: float = 1.0) -> EnhancedMT5DataFetcher:
    """Factory function to create MT5 data fetcher"""
    return EnhancedMT5DataFetcher(max_retries=max_retries, retry_delay=retry_delay)