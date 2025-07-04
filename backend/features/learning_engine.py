"""Training feedback loop and self-learning system for JARVIS v3.0."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
from collections import defaultdict, deque
import threading
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from database.services import config_service, memory_service, trade_service
from utils.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class LearningType(Enum):
    """Types of learning experiences."""
    TRADE_OUTCOME = "trade_outcome"
    USER_FEEDBACK = "user_feedback"
    STRATEGY_PERFORMANCE = "strategy_performance"
    PARAMETER_OPTIMIZATION = "parameter_optimization"
    SYSTEM_ADAPTATION = "system_adaptation"
    USER_INTERACTION = "user_interaction"


class ActionType(Enum):
    """Types of actions that can be learned from."""
    TRADE_DECISION = "trade_decision"
    STRATEGY_SELECTION = "strategy_selection"
    PARAMETER_CHOICE = "parameter_choice"
    RESPONSE_GENERATION = "response_generation"
    RISK_ASSESSMENT = "risk_assessment"


@dataclass
class LearningExperience:
    """Represents a learning experience with feedback."""
    id: str
    timestamp: datetime
    learning_type: LearningType
    action_type: ActionType
    context: Dict[str, Any]           # Situation context (market conditions, system state, etc.)
    action_taken: Dict[str, Any]      # What action was taken
    outcome: Dict[str, Any]           # What happened as a result
    feedback_score: float             # Numerical feedback (-1.0 to 1.0)
    feedback_details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed: bool = False


@dataclass
class LearningModel:
    """Represents a trained learning model."""
    name: str
    model_type: str
    action_type: ActionType
    features: List[str]
    model_object: Any = None
    scaler: Any = None
    accuracy: float = 0.0
    last_trained: Optional[datetime] = None
    training_samples: int = 0
    feature_importance: Dict[str, float] = field(default_factory=dict)


@dataclass
class UserRating:
    """User rating for responses or actions."""
    timestamp: datetime
    interaction_id: str
    rating: int                       # 1-5 scale
    category: str                     # "response_quality", "trade_decision", etc.
    comments: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


class FeatureExtractor:
    """Extracts features from context for machine learning."""
    
    def __init__(self):
        self.feature_extractors = {
            ActionType.TRADE_DECISION: self._extract_trade_features,
            ActionType.STRATEGY_SELECTION: self._extract_strategy_features,
            ActionType.PARAMETER_CHOICE: self._extract_parameter_features,
            ActionType.RESPONSE_GENERATION: self._extract_response_features,
            ActionType.RISK_ASSESSMENT: self._extract_risk_features
        }
    
    def extract_features(self, experience: LearningExperience) -> Dict[str, float]:
        """Extract numerical features from a learning experience."""
        try:
            if experience.action_type in self.feature_extractors:
                return self.feature_extractors[experience.action_type](experience)
            else:
                return self._extract_generic_features(experience)
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {}
    
    def _extract_trade_features(self, experience: LearningExperience) -> Dict[str, float]:
        """Extract features for trade decisions."""
        features = {}
        context = experience.context
        
        # Market features
        features.update({
            "price_change_1d": context.get("price_change_1d", 0.0),
            "price_change_5d": context.get("price_change_5d", 0.0),
            "volume_ratio": context.get("volume_ratio", 1.0),
            "volatility": context.get("volatility", 0.0),
            "rsi": context.get("rsi", 50.0),
            "macd_signal": context.get("macd_signal", 0.0),
            "ema_crossover": context.get("ema_crossover", 0.0)
        })
        
        # Portfolio features
        features.update({
            "portfolio_value": context.get("portfolio_value", 10000.0),
            "cash_ratio": context.get("cash_ratio", 0.1),
            "position_count": context.get("position_count", 0),
            "daily_pnl": context.get("daily_pnl", 0.0),
            "drawdown": context.get("drawdown", 0.0)
        })
        
        # Time features
        timestamp = experience.timestamp
        features.update({
            "hour_of_day": timestamp.hour,
            "day_of_week": timestamp.weekday(),
            "market_session": 1.0 if 9 <= timestamp.hour < 16 else 0.0
        })
        
        # Strategy features
        features.update({
            "strategy_performance": context.get("strategy_performance", 0.5),
            "signal_strength": context.get("signal_strength", 0.5),
            "confidence_score": context.get("confidence_score", 0.5)
        })
        
        return features
    
    def _extract_strategy_features(self, experience: LearningExperience) -> Dict[str, float]:
        """Extract features for strategy selection."""
        features = {}
        context = experience.context
        
        # Market condition features
        features.update({
            "trend_strength": context.get("trend_strength", 0.0),
            "volatility_regime": context.get("volatility_regime", 0.0),
            "market_regime": context.get("market_regime", 0.0),  # 0=bear, 0.5=sideways, 1=bull
            "correlation_environment": context.get("correlation", 0.0)
        })
        
        # Strategy performance features
        for strategy in ["RSI", "EMA", "MACD"]:
            strategy_key = strategy.lower()
            features.update({
                f"{strategy_key}_recent_performance": context.get(f"{strategy_key}_performance", 0.5),
                f"{strategy_key}_win_rate": context.get(f"{strategy_key}_win_rate", 0.5),
                f"{strategy_key}_sharpe": context.get(f"{strategy_key}_sharpe", 0.0)
            })
        
        return features
    
    def _extract_parameter_features(self, experience: LearningExperience) -> Dict[str, float]:
        """Extract features for parameter choices."""
        features = {}
        context = experience.context
        
        # Current parameter values
        params = context.get("parameters", {})
        for param_name, param_value in params.items():
            if isinstance(param_value, (int, float)):
                features[f"param_{param_name}"] = float(param_value)
        
        # Performance context
        features.update({
            "recent_performance": context.get("recent_performance", 0.0),
            "parameter_age_days": context.get("parameter_age_days", 0),
            "optimization_score": context.get("optimization_score", 0.0)
        })
        
        return features
    
    def _extract_response_features(self, experience: LearningExperience) -> Dict[str, float]:
        """Extract features for response generation."""
        features = {}
        context = experience.context
        
        # Response characteristics
        features.update({
            "response_length": context.get("response_length", 0),
            "confidence_score": context.get("confidence_score", 0.5),
            "complexity_score": context.get("complexity_score", 0.5),
            "relevance_score": context.get("relevance_score", 0.5)
        })
        
        # User context
        features.update({
            "user_experience_level": context.get("user_experience", 0.5),
            "session_length": context.get("session_length", 0),
            "previous_satisfaction": context.get("previous_satisfaction", 0.5)
        })
        
        return features
    
    def _extract_risk_features(self, experience: LearningExperience) -> Dict[str, float]:
        """Extract features for risk assessment."""
        features = {}
        context = experience.context
        
        # Risk metrics
        features.update({
            "position_size": context.get("position_size", 0.0),
            "portfolio_concentration": context.get("portfolio_concentration", 0.0),
            "correlation_risk": context.get("correlation_risk", 0.0),
            "volatility": context.get("volatility", 0.0),
            "liquidity_risk": context.get("liquidity_risk", 0.0)
        })
        
        return features
    
    def _extract_generic_features(self, experience: LearningExperience) -> Dict[str, float]:
        """Extract generic features when specific extractor not available."""
        features = {}
        context = experience.context
        
        # Extract any numerical values from context
        for key, value in context.items():
            if isinstance(value, (int, float)):
                features[key] = float(value)
        
        return features


class LearningEngine:
    """Main learning and adaptation engine."""
    
    def __init__(self):
        self.experiences: deque = deque(maxlen=10000)
        self.user_ratings: deque = deque(maxlen=5000)
        self.models: Dict[ActionType, LearningModel] = {}
        self.feature_extractor = FeatureExtractor()
        
        self.learning_enabled = True
        self.min_samples_for_training = 100
        self.retrain_interval_hours = 24
        self.adaptation_rate = 0.1
        
        # Learning thread
        self.learning_thread = None
        self.running = False
        
        self.load_configuration()
        logger.info("Learning engine initialized")
    
    def load_configuration(self):
        """Load learning engine configuration."""
        try:
            self.learning_enabled = config_service.get_config("learning.enabled", True)
            self.min_samples_for_training = config_service.get_config("learning.min_samples", 100)
            self.retrain_interval_hours = config_service.get_config("learning.retrain_interval", 24)
            self.adaptation_rate = config_service.get_config("learning.adaptation_rate", 0.1)
            
            # Load saved models
            self._load_saved_models()
            
            logger.info("Learning engine configuration loaded")
        except Exception as e:
            logger.warning(f"Failed to load learning configuration: {e}")
    
    def save_configuration(self):
        """Save learning engine configuration."""
        try:
            config_service.set_config("learning.enabled", self.learning_enabled, "Enable learning engine")
            config_service.set_config("learning.min_samples", self.min_samples_for_training, "Minimum samples for training")
            config_service.set_config("learning.retrain_interval", self.retrain_interval_hours, "Model retrain interval")
            config_service.set_config("learning.adaptation_rate", self.adaptation_rate, "Learning adaptation rate")
            
            logger.info("Learning engine configuration saved")
        except Exception as e:
            logger.error(f"Failed to save learning configuration: {e}")
    
    def _load_saved_models(self):
        """Load previously trained models."""
        try:
            # This would load models from disk
            # For now, initialize empty models
            for action_type in ActionType:
                self.models[action_type] = LearningModel(
                    name=f"{action_type.value}_model",
                    model_type="random_forest",
                    action_type=action_type,
                    features=[]
                )
            
            logger.info("Learning models initialized")
        except Exception as e:
            logger.error(f"Error loading saved models: {e}")
    
    def _save_models(self):
        """Save trained models to disk."""
        try:
            for action_type, model in self.models.items():
                if model.model_object is not None:
                    model_path = f"models/{action_type.value}_model.pkl"
                    scaler_path = f"models/{action_type.value}_scaler.pkl"
                    
                    # Create directory if needed
                    import os
                    os.makedirs("models", exist_ok=True)
                    
                    # Save model and scaler
                    joblib.dump(model.model_object, model_path)
                    if model.scaler:
                        joblib.dump(model.scaler, scaler_path)
                    
                    logger.info(f"Saved model for {action_type.value}")
                    
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def record_experience(self, learning_type: LearningType, action_type: ActionType,
                         context: Dict[str, Any], action_taken: Dict[str, Any],
                         outcome: Dict[str, Any], feedback_score: float,
                         feedback_details: Dict[str, Any] = None) -> str:
        """Record a learning experience."""
        try:
            experience_id = f"{action_type.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
            
            experience = LearningExperience(
                id=experience_id,
                timestamp=datetime.utcnow(),
                learning_type=learning_type,
                action_type=action_type,
                context=context,
                action_taken=action_taken,
                outcome=outcome,
                feedback_score=feedback_score,
                feedback_details=feedback_details or {}
            )
            
            self.experiences.append(experience)
            
            # Store in database for persistence
            memory_service.store_memory(
                "learning_experience",
                f"Learning experience: {learning_type.value}",
                tags=["learning", action_type.value],
                metadata={
                    "experience_id": experience_id,
                    "learning_type": learning_type.value,
                    "action_type": action_type.value,
                    "feedback_score": feedback_score,
                    "context": context,
                    "action_taken": action_taken,
                    "outcome": outcome
                }
            )
            
            logger.info(f"Recorded learning experience: {experience_id}")
            return experience_id
            
        except Exception as e:
            logger.error(f"Error recording learning experience: {e}")
            return ""
    
    def record_user_rating(self, interaction_id: str, rating: int, category: str,
                          comments: Optional[str] = None, context: Dict[str, Any] = None) -> bool:
        """Record user rating/feedback."""
        try:
            user_rating = UserRating(
                timestamp=datetime.utcnow(),
                interaction_id=interaction_id,
                rating=rating,
                category=category,
                comments=comments,
                context=context or {}
            )
            
            self.user_ratings.append(user_rating)
            
            # Store in database
            memory_service.store_memory(
                "user_rating",
                f"User rating: {rating}/5 for {category}",
                tags=["user_feedback", category],
                metadata={
                    "interaction_id": interaction_id,
                    "rating": rating,
                    "category": category,
                    "comments": comments,
                    "context": context
                }
            )
            
            # Convert to learning experience
            feedback_score = (rating - 3) / 2  # Convert 1-5 scale to -1 to 1
            
            if category == "trade_decision":
                action_type = ActionType.TRADE_DECISION
            elif category == "response_quality":
                action_type = ActionType.RESPONSE_GENERATION
            elif category == "strategy_choice":
                action_type = ActionType.STRATEGY_SELECTION
            else:
                action_type = ActionType.RESPONSE_GENERATION
            
            self.record_experience(
                learning_type=LearningType.USER_FEEDBACK,
                action_type=action_type,
                context=context or {},
                action_taken={"interaction_id": interaction_id},
                outcome={"user_rating": rating},
                feedback_score=feedback_score,
                feedback_details={"comments": comments, "category": category}
            )
            
            # Update user satisfaction tracking
            self._update_user_satisfaction(rating, category)
            
            logger.info(f"Recorded user rating: {rating}/5 for {category}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording user rating: {e}")
            return False
    
    def _update_user_satisfaction(self, rating: int, category: str):
        """Update running user satisfaction metrics."""
        try:
            # Get current averages
            current_avg = config_service.get_config("user.average_feedback_score", 3.0)
            current_count = config_service.get_config("user.feedback_count", 0)
            
            # Calculate new average
            new_count = current_count + 1
            new_avg = (current_avg * current_count + rating) / new_count
            
            # Update configuration
            config_service.set_config("user.average_feedback_score", new_avg, "Average user feedback score")
            config_service.set_config("user.feedback_count", new_count, "Total feedback count")
            config_service.set_config("user.last_interaction", datetime.utcnow().isoformat(), "Last user interaction")
            
            # Category-specific tracking
            category_avg_key = f"user.{category}_average"
            category_count_key = f"user.{category}_count"
            
            category_avg = config_service.get_config(category_avg_key, 3.0)
            category_count = config_service.get_config(category_count_key, 0)
            
            new_category_count = category_count + 1
            new_category_avg = (category_avg * category_count + rating) / new_category_count
            
            config_service.set_config(category_avg_key, new_category_avg, f"Average {category} score")
            config_service.set_config(category_count_key, new_category_count, f"Total {category} count")
            
        except Exception as e:
            logger.error(f"Error updating user satisfaction: {e}")
    
    def train_model(self, action_type: ActionType) -> bool:
        """Train a model for a specific action type."""
        try:
            if not self.learning_enabled:
                return False
            
            # Get experiences for this action type
            action_experiences = [
                exp for exp in self.experiences 
                if exp.action_type == action_type and not exp.processed
            ]
            
            if len(action_experiences) < self.min_samples_for_training:
                logger.info(f"Insufficient samples for {action_type.value}: {len(action_experiences)} < {self.min_samples_for_training}")
                return False
            
            logger.info(f"Training model for {action_type.value} with {len(action_experiences)} samples")
            
            # Extract features and labels
            X_data = []
            y_data = []
            feature_names = set()
            
            for exp in action_experiences:
                features = self.feature_extractor.extract_features(exp)
                if features:
                    X_data.append(features)
                    y_data.append(exp.feedback_score)
                    feature_names.update(features.keys())
            
            if len(X_data) < self.min_samples_for_training:
                logger.warning(f"Insufficient valid samples after feature extraction: {len(X_data)}")
                return False
            
            # Convert to consistent feature matrix
            feature_names = sorted(list(feature_names))
            X_matrix = []
            
            for features_dict in X_data:
                row = [features_dict.get(fname, 0.0) for fname in feature_names]
                X_matrix.append(row)
            
            X = np.array(X_matrix)
            y = np.array(y_data)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            # Feature importance
            feature_importance = dict(zip(feature_names, model.feature_importances_))
            
            # Update model
            learning_model = self.models[action_type]
            learning_model.model_object = model
            learning_model.scaler = scaler
            learning_model.features = feature_names
            learning_model.accuracy = test_score
            learning_model.last_trained = datetime.utcnow()
            learning_model.training_samples = len(X_data)
            learning_model.feature_importance = feature_importance
            
            # Mark experiences as processed
            for exp in action_experiences:
                exp.processed = True
            
            logger.info(f"Model trained for {action_type.value}: accuracy={test_score:.3f}, samples={len(X_data)}")
            
            # Save model
            self._save_models()
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model for {action_type.value}: {e}")
            return False
    
    def predict_outcome(self, action_type: ActionType, context: Dict[str, Any],
                       action: Dict[str, Any]) -> Tuple[float, float]:
        """Predict outcome for a proposed action."""
        try:
            if action_type not in self.models:
                return 0.0, 0.0  # No prediction available
            
            model = self.models[action_type]
            if model.model_object is None:
                return 0.0, 0.0  # Model not trained
            
            # Create mock experience for feature extraction
            mock_experience = LearningExperience(
                id="prediction",
                timestamp=datetime.utcnow(),
                learning_type=LearningType.SYSTEM_ADAPTATION,
                action_type=action_type,
                context=context,
                action_taken=action,
                outcome={},
                feedback_score=0.0
            )
            
            # Extract features
            features = self.feature_extractor.extract_features(mock_experience)
            
            if not features:
                return 0.0, 0.0
            
            # Convert to feature vector
            feature_vector = [features.get(fname, 0.0) for fname in model.features]
            X = np.array([feature_vector])
            
            # Scale features
            if model.scaler:
                X_scaled = model.scaler.transform(X)
            else:
                X_scaled = X
            
            # Make prediction
            prediction = model.model_object.predict(X_scaled)[0]
            
            # Calculate confidence (simplified)
            confidence = min(1.0, model.accuracy)
            
            return float(prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"Error making prediction for {action_type.value}: {e}")
            return 0.0, 0.0
    
    def adapt_parameters(self, action_type: ActionType, current_params: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt parameters based on learning."""
        try:
            if not self.learning_enabled:
                return current_params
            
            # Get recent experiences
            recent_experiences = [
                exp for exp in self.experiences
                if (exp.action_type == action_type and 
                    exp.timestamp > datetime.utcnow() - timedelta(days=7))
            ]
            
            if len(recent_experiences) < 10:
                return current_params
            
            # Calculate average feedback for different parameter ranges
            param_feedback = defaultdict(list)
            
            for exp in recent_experiences:
                action_params = exp.action_taken.get("parameters", {})
                for param_name, param_value in action_params.items():
                    if isinstance(param_value, (int, float)):
                        param_feedback[param_name].append((param_value, exp.feedback_score))
            
            # Adapt parameters based on feedback
            adapted_params = current_params.copy()
            
            for param_name, feedback_data in param_feedback.items():
                if param_name in current_params and len(feedback_data) >= 5:
                    # Find parameter values with highest feedback
                    feedback_data.sort(key=lambda x: x[1], reverse=True)
                    best_values = [x[0] for x in feedback_data[:3]]  # Top 3
                    
                    # Move towards better values
                    current_value = current_params[param_name]
                    target_value = np.mean(best_values)
                    
                    # Gradual adaptation
                    new_value = current_value + (target_value - current_value) * self.adaptation_rate
                    adapted_params[param_name] = new_value
                    
                    logger.info(f"Adapted parameter {param_name}: {current_value:.3f} -> {new_value:.3f}")
            
            return adapted_params
            
        except Exception as e:
            logger.error(f"Error adapting parameters: {e}")
            return current_params
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from learning data."""
        try:
            insights = {
                "total_experiences": len(self.experiences),
                "total_user_ratings": len(self.user_ratings),
                "learning_enabled": self.learning_enabled,
                "model_status": {},
                "user_satisfaction": {},
                "recent_trends": {},
                "top_features": {}
            }
            
            # Model status
            for action_type, model in self.models.items():
                insights["model_status"][action_type.value] = {
                    "trained": model.model_object is not None,
                    "accuracy": model.accuracy,
                    "training_samples": model.training_samples,
                    "last_trained": model.last_trained.isoformat() if model.last_trained else None,
                    "features_count": len(model.features)
                }
            
            # User satisfaction
            recent_ratings = [r for r in self.user_ratings if r.timestamp > datetime.utcnow() - timedelta(days=7)]
            if recent_ratings:
                by_category = defaultdict(list)
                for rating in recent_ratings:
                    by_category[rating.category].append(rating.rating)
                
                for category, ratings in by_category.items():
                    insights["user_satisfaction"][category] = {
                        "average": np.mean(ratings),
                        "count": len(ratings),
                        "trend": "improving" if len(ratings) > 5 and np.mean(ratings[-5:]) > np.mean(ratings[:-5]) else "stable"
                    }
            
            # Recent learning trends
            recent_experiences = [e for e in self.experiences if e.timestamp > datetime.utcnow() - timedelta(days=7)]
            if recent_experiences:
                by_action_type = defaultdict(list)
                for exp in recent_experiences:
                    by_action_type[exp.action_type].append(exp.feedback_score)
                
                for action_type, scores in by_action_type.items():
                    insights["recent_trends"][action_type.value] = {
                        "average_feedback": np.mean(scores),
                        "count": len(scores),
                        "improving": len(scores) > 5 and np.mean(scores[-5:]) > np.mean(scores[:-5])
                    }
            
            # Top features by importance
            for action_type, model in self.models.items():
                if model.feature_importance:
                    top_features = sorted(model.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                    insights["top_features"][action_type.value] = top_features
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting learning insights: {e}")
            return {"error": str(e)}
    
    def learning_loop(self):
        """Main learning loop."""
        logger.info("Learning engine started")
        
        while self.running:
            try:
                if not self.learning_enabled:
                    time.sleep(3600)  # Sleep 1 hour if learning disabled
                    continue
                
                # Train models that need retraining
                for action_type in ActionType:
                    model = self.models[action_type]
                    
                    # Check if model needs retraining
                    needs_training = (
                        model.last_trained is None or
                        datetime.utcnow() - model.last_trained > timedelta(hours=self.retrain_interval_hours)
                    )
                    
                    if needs_training:
                        logger.info(f"Attempting to train model for {action_type.value}")
                        self.train_model(action_type)
                
                # Sleep before next check
                time.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Learning loop error: {e}")
                time.sleep(3600)
    
    def start(self):
        """Start the learning engine."""
        if self.running:
            logger.warning("Learning engine already running")
            return
        
        self.running = True
        self.learning_thread = threading.Thread(target=self.learning_loop, daemon=True)
        self.learning_thread.start()
        
        logger.info("Learning engine started")
    
    def stop(self):
        """Stop the learning engine."""
        if not self.running:
            logger.warning("Learning engine not running")
            return
        
        self.running = False
        if self.learning_thread:
            self.learning_thread.join(timeout=10)
        
        # Save models before stopping
        self._save_models()
        
        logger.info("Learning engine stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get learning engine status."""
        try:
            return {
                "running": self.running,
                "learning_enabled": self.learning_enabled,
                "total_experiences": len(self.experiences),
                "total_user_ratings": len(self.user_ratings),
                "models": {
                    action_type.value: {
                        "trained": model.model_object is not None,
                        "accuracy": model.accuracy,
                        "training_samples": model.training_samples,
                        "last_trained": model.last_trained.isoformat() if model.last_trained else None
                    }
                    for action_type, model in self.models.items()
                },
                "recent_activity": {
                    "experiences_last_24h": len([e for e in self.experiences 
                                                if e.timestamp > datetime.utcnow() - timedelta(days=1)]),
                    "ratings_last_24h": len([r for r in self.user_ratings 
                                           if r.timestamp > datetime.utcnow() - timedelta(days=1)])
                },
                "last_check": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting learning engine status: {e}")
            return {"error": str(e)}


# Global learning engine instance
learning_engine = LearningEngine()