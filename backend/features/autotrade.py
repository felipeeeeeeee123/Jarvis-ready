# autotrade.py

from dotenv import load_dotenv
import os
from datetime import datetime
from typing import Tuple, Dict
import pandas as pd
from alpaca_trade_api import REST, TimeFrame

from utils.memory_db import DatabaseMemoryManager
from utils.logging_config import get_logger
from config.settings import settings
from .telegram_alerts import send_telegram_alert
from .strategies import rsi_strategy, ema_strategy, macd_strategy
from .risk_manager import risk_manager
from .strategy_manager import strategy_manager

load_dotenv()

# Initialize logger
logger = get_logger(__name__)

# === Alpaca REST client ===
aip = None
if settings.ALPACA_API_KEY and settings.ALPACA_SECRET_KEY:
    try:
        aip = REST(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY, base_url=settings.ALPACA_BASE_URL)
        logger.info("Alpaca API client initialized", extra={"base_url": settings.ALPACA_BASE_URL})
    except Exception as e:
        logger.error("Failed to initialize Alpaca API client", extra={"error": str(e)})
else:
    logger.warning("Alpaca API credentials not configured - trading will be disabled")

memory = DatabaseMemoryManager()

# === Strategy definitions ===
STRATEGIES = {
    "RSI": rsi_strategy,
    "EMA": ema_strategy,
    "MACD": macd_strategy,
}

def choose_strategy():
    return STRATEGIES.get(strategy_manager.current_strategy, rsi_strategy)

def position_size(price: float, cash: float) -> int:
    """Calculate position size using risk management."""
    return risk_manager.calculate_position_size("", price, cash)

def trade_signal(symbol: str) -> Tuple[str, str, Dict]:
    """Get enhanced trade signal with strategy management."""
    if not aip:
        logger.warning("Alpaca API not configured - returning hold signal")
        return "hold", "none", {}
    
    try:
        end = datetime.utcnow()
        start = end - pd.Timedelta(days=10)
        bars = aip.get_bars(symbol, TimeFrame.Hour, start, end).df
        if bars.empty:
            return "hold", "none", {}
        
        prices = bars.close
        volume = bars.volume if 'volume' in bars.columns else None
        
        # Get signal with dynamic strategy switching
        signal, strategy_used, metadata = strategy_manager.get_strategy_signal(symbol, prices, volume)
        
        return signal, strategy_used, metadata
        
    except Exception as e:
        logger.error(f"Failed to get trade signal for {symbol}: {e}")
        return "hold", "error", {"error": str(e)}

def execute_trade(symbol: str) -> None:
    """Execute a trade with enhanced risk management and strategy switching."""
    try:
        if not aip:
            logger.warning(f"Trade skipped for {symbol} - Alpaca API not configured")
            return
        
        logger.info(f"Executing trade analysis for {symbol}")
        
        # Get account information
        account = aip.get_account()
        cash = float(account.cash)
        portfolio_value = float(account.portfolio_value)
        
        # Get current price
        last_price = float(aip.get_latest_trade(symbol).price)
        
        # Check for stop loss / take profit triggers first
        stop_loss_trigger = risk_manager.check_stop_loss_trigger(symbol, last_price)
        take_profit_trigger = risk_manager.check_take_profit_trigger(symbol, last_price)
        
        if stop_loss_trigger:
            logger.warning(f"Stop loss triggered for {symbol}", extra=stop_loss_trigger)
            # Execute stop loss sell
            qty = abs(stop_loss_trigger.get("quantity", 1))  # Would need to track position size
            action = "sell"
            strategy_used = "stop_loss"
            metadata = stop_loss_trigger
        elif take_profit_trigger:
            logger.info(f"Take profit triggered for {symbol}", extra=take_profit_trigger)
            # Execute take profit sell
            qty = abs(take_profit_trigger.get("quantity", 1))  # Would need to track position size
            action = "sell"
            strategy_used = "take_profit"
            metadata = take_profit_trigger
        else:
            # Regular trading signal
            action, strategy_used, metadata = trade_signal(symbol)
            
            if action == "hold":
                logger.info(f"No action for {symbol} (signal: hold)")
                return
            
            # Calculate position size with risk management
            qty = risk_manager.calculate_position_size(symbol, last_price, portfolio_value)
            
            if qty <= 0:
                logger.info(f"No position to take for {symbol} (calculated quantity: {qty})")
                return
            
            # Validate trade against risk management rules
            valid, validation_message = risk_manager.validate_trade(
                symbol, action, qty, last_price, portfolio_value
            )
            
            if not valid:
                logger.warning(f"Trade rejected by risk management: {validation_message}")
                return
        
        logger.info(f"Trade signal for {symbol}: {action} (strategy: {strategy_used})")
        
        # Execute the trade
        if action == "buy":
            order = aip.submit_order(symbol, qty, "buy", "market", "gtc")
            order_id = order.id if hasattr(order, 'id') else None
            
            # Calculate stop loss and take profit
            stop_loss_price = risk_manager.calculate_stop_loss(last_price, "buy")
            take_profit_price = risk_manager.calculate_take_profit(last_price, "buy")
            
            # Log trade with enhanced metadata
            enhanced_metadata = {
                **metadata,
                "stop_loss_price": stop_loss_price,
                "take_profit_price": take_profit_price,
                "portfolio_value": portfolio_value,
                "signal_strength": metadata.get("signal_strength", 0.5)
            }
            
            memory.log_trade(symbol, "buy", qty, last_price, strategy_used, order_id)
            
            message = f"📈 Bought {qty} {symbol} @ ${last_price:.2f} (SL: ${stop_loss_price:.2f}, TP: ${take_profit_price:.2f})"
            logger.info(message, extra={
                "symbol": symbol, "action": "buy", "quantity": qty, "price": last_price,
                "strategy": strategy_used, "metadata": enhanced_metadata
            })
            send_telegram_alert(message)
            
        elif action == "sell":
            order = aip.submit_order(symbol, qty, "sell", "market", "gtc")
            order_id = order.id if hasattr(order, 'id') else None
            
            memory.log_trade(symbol, "sell", qty, last_price, strategy_used, order_id)
            
            # Calculate P&L if this is closing a position
            pnl_info = ""
            if "profit_amount" in metadata:
                pnl_info = f" (P&L: ${metadata['profit_amount']:.2f})"
            elif "loss_amount" in metadata:
                pnl_info = f" (P&L: -${metadata['loss_amount']:.2f})"
            
            message = f"📉 Sold {qty} {symbol} @ ${last_price:.2f}{pnl_info}"
            logger.info(message, extra={
                "symbol": symbol, "action": "sell", "quantity": qty, "price": last_price,
                "strategy": strategy_used, "metadata": metadata
            })
            send_telegram_alert(message)
        
        # Update strategy performance tracking
        strategy_manager.update_strategy_performance(strategy_used, symbol)
        
    except Exception as e:
        logger.error(f"Trade execution failed for {symbol}", extra={"error": str(e), "symbol": symbol})
        raise

def run_autotrader(symbols=None):
    """Run the autotrader for specified symbols."""
    symbols = symbols or ["AAPL"]
    logger.info(f"Running autotrader for symbols: {symbols}")
    
    for sym in symbols:
        try:
            execute_trade(sym)
        except Exception as exc:
            logger.error(f"Autotrade error for {sym}: {exc}", extra={"symbol": sym, "error": str(exc)})
            # Continue with other symbols even if one fails
