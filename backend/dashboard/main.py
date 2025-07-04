"""JARVIS v3.0 Comprehensive Dashboard - FastAPI Backend."""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import json
import asyncio
from datetime import datetime, timedelta
import uvicorn
import os
from pathlib import Path

# Import JARVIS components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from features.system_monitor import system_monitor
from features.self_healing import self_healing_daemon
from features.model_monitor import model_monitor
from features.parameter_tuner import parameter_tuner
from features.plugin_manager import plugin_manager
from features.performance_analyzer import performance_analyzer
from features.autonomous_scheduler import autonomous_scheduler
from features.learning_engine import learning_engine, LearningType, ActionType
from features.portfolio_manager import portfolio_manager
from features.backtester import backtester
from features.strategy_manager import strategy_manager
from features.risk_manager import risk_manager
from database.services import config_service, trade_service, memory_service
from utils.logging_config import get_logger

logger = get_logger(__name__)

# FastAPI app
app = FastAPI(
    title="JARVIS v3.0 Dashboard",
    description="Comprehensive control and monitoring dashboard for autonomous AI trading system",
    version="3.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

# Pydantic models for API
class UserRating(BaseModel):
    interaction_id: str
    rating: int = Field(..., ge=1, le=5)
    category: str
    comments: Optional[str] = None

class ActionRequest(BaseModel):
    action_type: str
    parameters: Dict[str, Any] = {}

class ConfigUpdate(BaseModel):
    key: str
    value: Any
    description: Optional[str] = None

class BacktestRequest(BaseModel):
    symbol: str
    strategy: str
    start_date: datetime
    end_date: datetime

class TradeRequest(BaseModel):
    symbol: str
    action: str
    quantity: Optional[int] = None

# Background task for real-time updates
async def broadcast_updates():
    """Background task to broadcast system updates to connected clients."""
    while True:
        try:
            if manager.active_connections:
                # Get system status
                status = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "system": system_monitor.get_current_status(),
                    "autonomous": autonomous_scheduler.get_status(),
                    "learning": learning_engine.get_status()
                }
                
                await manager.broadcast(json.dumps({
                    "type": "system_update",
                    "data": status
                }))
            
            await asyncio.sleep(30)  # Update every 30 seconds
            
        except Exception as e:
            logger.error(f"Error in broadcast updates: {e}")
            await asyncio.sleep(30)

@app.on_event("startup")
async def startup_event():
    """Initialize JARVIS components on startup."""
    logger.info("Starting JARVIS v3.0 Dashboard")
    
    # Start all autonomous systems
    try:
        system_monitor.start()
        self_healing_daemon.start()
        model_monitor.start_monitoring()
        parameter_tuner.start_auto_tuning()
        plugin_manager.start_monitoring()
        autonomous_scheduler.start()
        learning_engine.start()
        
        logger.info("All autonomous systems started successfully")
        
        # Start background task for real-time updates
        asyncio.create_task(broadcast_updates())
        
    except Exception as e:
        logger.error(f"Error starting autonomous systems: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down JARVIS v3.0 Dashboard")
    
    try:
        system_monitor.stop()
        self_healing_daemon.stop()
        model_monitor.stop_monitoring()
        parameter_tuner.stop_auto_tuning()
        plugin_manager.stop_monitoring()
        autonomous_scheduler.stop()
        learning_engine.stop()
        
        logger.info("All autonomous systems stopped successfully")
        
    except Exception as e:
        logger.error(f"Error stopping autonomous systems: {e}")

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back for now - could handle commands here
            await manager.send_personal_message(f"Echo: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# API Routes

@app.get("/")
async def read_root():
    """Serve the main dashboard."""
    return FileResponse("backend/dashboard/static/index.html")

@app.get("/api/status")
async def get_system_status():
    """Get comprehensive system status."""
    try:
        return {
            "system": system_monitor.get_current_status(),
            "self_healing": self_healing_daemon.get_health_status(),
            "model_monitor": model_monitor.get_status(),
            "parameter_tuner": parameter_tuner.get_tuning_status(),
            "plugin_manager": plugin_manager.get_status(),
            "autonomous_scheduler": autonomous_scheduler.get_status(),
            "learning_engine": learning_engine.get_status(),
            "performance": performance_analyzer.calculate_comprehensive_metrics().__dict__,
            "portfolio": portfolio_manager.get_portfolio_summary(),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/performance")
async def get_performance_metrics():
    """Get detailed performance metrics."""
    try:
        metrics = performance_analyzer.calculate_comprehensive_metrics()
        report = performance_analyzer.generate_performance_report(metrics)
        opportunities = performance_analyzer.identify_optimization_opportunities(metrics)
        
        return {
            "metrics": metrics.__dict__,
            "report": report,
            "optimization_opportunities": opportunities,
            "strategy_comparison": performance_analyzer.get_strategy_comparison()
        }
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trades")
async def get_trades(limit: int = 100):
    """Get recent trades."""
    try:
        trades = trade_service.get_recent_trades(limit=limit)
        return {
            "trades": [
                {
                    "id": trade.id,
                    "symbol": trade.symbol,
                    "action": trade.action,
                    "quantity": trade.quantity,
                    "price": trade.price,
                    "total_value": trade.total_value,
                    "strategy_used": trade.strategy_used,
                    "executed_at": trade.executed_at.isoformat(),
                    "status": trade.status
                }
                for trade in trades
            ]
        }
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/trades")
async def execute_trade(trade_request: TradeRequest):
    """Execute a trade."""
    try:
        from features.autotrade import execute_trade
        
        # Execute trade
        execute_trade(trade_request.symbol)
        
        return {"status": "success", "message": f"Trade executed for {trade_request.symbol}"}
        
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolio")
async def get_portfolio():
    """Get portfolio information."""
    try:
        summary = portfolio_manager.get_portfolio_summary()
        return summary
    except Exception as e:
        logger.error(f"Error getting portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/autonomous")
async def get_autonomous_status():
    """Get autonomous system status."""
    try:
        return {
            "scheduler": autonomous_scheduler.get_status(),
            "learning": learning_engine.get_status(),
            "self_healing": self_healing_daemon.get_health_status(),
            "model_monitor": model_monitor.get_status()
        }
    except Exception as e:
        logger.error(f"Error getting autonomous status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/autonomous/actions/{action_id}/execute")
async def execute_autonomous_action(action_id: str):
    """Force execute an autonomous action."""
    try:
        success = autonomous_scheduler.force_execute_action(action_id)
        if success:
            return {"status": "success", "message": f"Action {action_id} executed"}
        else:
            raise HTTPException(status_code=404, detail="Action not found")
    except Exception as e:
        logger.error(f"Error executing autonomous action: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/learning/rating")
async def submit_user_rating(rating: UserRating):
    """Submit user rating/feedback."""
    try:
        success = learning_engine.record_user_rating(
            interaction_id=rating.interaction_id,
            rating=rating.rating,
            category=rating.category,
            comments=rating.comments
        )
        
        if success:
            return {"status": "success", "message": "Rating recorded"}
        else:
            raise HTTPException(status_code=500, detail="Failed to record rating")
            
    except Exception as e:
        logger.error(f"Error recording user rating: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/learning/insights")
async def get_learning_insights():
    """Get learning insights and analytics."""
    try:
        insights = learning_engine.get_learning_insights()
        return insights
    except Exception as e:
        logger.error(f"Error getting learning insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/backtest")
async def run_backtest(backtest_request: BacktestRequest):
    """Run a backtest."""
    try:
        result = backtester.run_backtest(
            symbol=backtest_request.symbol,
            strategy_name=backtest_request.strategy,
            start_date=backtest_request.start_date,
            end_date=backtest_request.end_date
        )
        
        report = backtester.generate_backtest_report(result)
        
        return {
            "result": result.__dict__,
            "report": report
        }
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/strategies")
async def get_strategies():
    """Get strategy information."""
    try:
        return {
            "current_strategy": strategy_manager.current_strategy,
            "performance": strategy_manager.get_performance_summary(),
            "available_strategies": list(strategy_manager.strategies.keys())
        }
    except Exception as e:
        logger.error(f"Error getting strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/strategies/{strategy_name}/switch")
async def switch_strategy(strategy_name: str):
    """Switch to a different strategy."""
    try:
        success = strategy_manager.execute_strategy_switch(strategy_name, "Manual switch via dashboard")
        if success:
            return {"status": "success", "message": f"Switched to {strategy_name}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to switch strategy")
    except Exception as e:
        logger.error(f"Error switching strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/plugins")
async def get_plugins():
    """Get plugin information."""
    try:
        status = plugin_manager.get_status()
        return status
    except Exception as e:
        logger.error(f"Error getting plugins: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/plugins/{plugin_name}/install")
async def install_plugin(plugin_name: str):
    """Install a plugin."""
    try:
        success = plugin_manager.install_plugin(plugin_name)
        if success:
            return {"status": "success", "message": f"Plugin {plugin_name} installed"}
        else:
            raise HTTPException(status_code=500, detail="Failed to install plugin")
    except Exception as e:
        logger.error(f"Error installing plugin: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def get_models():
    """Get model information."""
    try:
        status = model_monitor.get_status()
        return status
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/{model_name}/update")
async def update_model(model_name: str):
    """Force update a model."""
    try:
        success = model_monitor.force_update(model_name)
        if success:
            return {"status": "success", "message": f"Model {model_name} updated"}
        else:
            raise HTTPException(status_code=500, detail="Failed to update model")
    except Exception as e:
        logger.error(f"Error updating model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config")
async def get_config():
    """Get system configuration."""
    try:
        # Get key configuration values
        config_keys = [
            "trading.current_strategy",
            "trading.risk_management_enabled",
            "scheduler.check_interval",
            "learning.enabled",
            "model_monitor.auto_update",
            "plugins.auto_update"
        ]
        
        config = {}
        for key in config_keys:
            try:
                config[key] = config_service.get_config(key, None)
            except:
                config[key] = None
        
        return config
        
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/config")
async def update_config(config_update: ConfigUpdate):
    """Update system configuration."""
    try:
        config_service.set_config(
            config_update.key, 
            config_update.value, 
            config_update.description or "Updated via dashboard"
        )
        
        return {"status": "success", "message": f"Config {config_update.key} updated"}
        
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/logs")
async def get_logs(limit: int = 100):
    """Get recent log entries."""
    try:
        # This would integrate with your logging system
        # For now, return a placeholder
        return {
            "logs": [
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": "INFO",
                    "message": "JARVIS v3.0 Dashboard serving logs",
                    "component": "dashboard"
                }
            ]
        }
    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "3.0.0"
    }

# Serve static files
app.mount("/static", StaticFiles(directory="backend/dashboard/static"), name="static")

if __name__ == "__main__":
    # Create static directory if it doesn't exist
    static_dir = Path("backend/dashboard/static")
    static_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )