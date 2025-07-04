from flask import Flask, request, jsonify
from functools import wraps
import logging
from features.autotrade import run_autotrader
from utils.security import validate_api_key
from utils.logging_config import setup_logging, get_logger
from database.config import init_database
from config.settings import settings

app = Flask(__name__)

# Initialize logging and database
setup_logging()
logger = get_logger(__name__)

try:
    init_database()
    logger.info("Database initialized for API server")
except Exception as e:
    logger.error(f"Database initialization failed: {e}")

# Load API key from settings
API_KEY = settings.API_KEY
if not API_KEY:
    logger.warning("JARVIS_API_KEY not set. API endpoints will be unprotected!")


def require_api_key(f):
    """Decorator to require API key authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Skip authentication if no API key is configured
        if not API_KEY:
            logger.warning("API endpoint accessed without authentication (no API key configured)")
            return f(*args, **kwargs)
        
        # Check for API key in header
        provided_key = request.headers.get('X-API-Key')
        if not provided_key:
            # Check for API key in query parameters (less secure but convenient)
            provided_key = request.args.get('api_key')
        
        if not provided_key:
            logger.warning(f"API endpoint accessed without API key from {request.remote_addr}")
            return jsonify({"error": "API key required"}), 401
        
        if not validate_api_key(provided_key) or provided_key != API_KEY:
            logger.warning(f"API endpoint accessed with invalid API key from {request.remote_addr}")
            return jsonify({"error": "Invalid API key"}), 401
        
        return f(*args, **kwargs)
    return decorated_function


@app.route("/trade")
@require_api_key
def trade():
    """Execute a trade for the specified symbol."""
    try:
        symbol = request.args.get("symbol", "AAPL")
        logger.info(f"Trade request received for symbol: {symbol}")
        run_autotrader([symbol])
        return jsonify({"status": "ok", "symbol": symbol})
    except Exception as e:
        logger.error(f"Trade execution failed: {e}")
        return jsonify({"error": "Trade execution failed"}), 500


@app.route("/health")
def health():
    """Health check endpoint with system status."""
    try:
        from database.config import engine
        from database.services import memory_service, qa_service, trade_service
        from config.settings import settings
        import time
        
        start_time = time.time()
        
        # Test database connectivity
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        
        # Get system stats
        memory_count = len(memory_service.get_by_category("general"))
        recent_trades = len(trade_service.get_recent_trades(limit=10))
        
        response_time = time.time() - start_time
        
        return jsonify({
            "status": "healthy",
            "service": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "database": "connected",
            "memory_entries": memory_count,
            "recent_trades": recent_trades,
            "response_time_ms": round(response_time * 1000, 2),
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "service": settings.APP_NAME,
            "error": str(e),
            "timestamp": time.time()
        }), 503


@app.route("/")
def index():
    """API information endpoint."""
    return jsonify({
        "service": "JARVIS v3.0 API",
        "version": "3.0.0",
        "endpoints": {
            "/health": "Health check (no auth required)",
            "/trade": "Execute trades (requires API key)",
        }
    })


if __name__ == "__main__":
    app.run(host=settings.FLASK_HOST, port=settings.FLASK_PORT, debug=settings.FLASK_DEBUG)
