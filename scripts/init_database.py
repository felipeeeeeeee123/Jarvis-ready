#!/usr/bin/env python3
"""Initialize JARVIS v3.0 database and migrate from JSON files."""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from database.config import init_database, get_session
from database.services import memory_service, qa_service, config_service
from config.settings import settings
from utils.logging_config import setup_logging, get_logger

# Initialize logging
setup_logging()
logger = get_logger(__name__)


def migrate_json_memory():
    """Migrate existing JSON memory files to database."""
    json_files = [
        Path(__file__).parent.parent / "memory.json",
        Path(__file__).parent.parent / "data" / "memory.json"
    ]
    
    migrated_count = 0
    for json_file in json_files:
        if json_file.exists():
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                logger.info(f"Migrating memory from {json_file}")
                for key, value in data.items():
                    memory_service.set(key, value, "general")
                    migrated_count += 1
                
                # Backup the original file
                backup_path = json_file.with_suffix('.json.backup')
                json_file.rename(backup_path)
                logger.info(f"Backed up original file to {backup_path}")
                
            except Exception as e:
                logger.error(f"Failed to migrate {json_file}: {e}")
    
    return migrated_count


def migrate_qa_memory():
    """Migrate existing Q&A memory to database."""
    json_files = [
        Path(__file__).parent.parent / "data" / "qa_memory.json"
    ]
    
    migrated_count = 0
    for json_file in json_files:
        if json_file.exists():
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                logger.info(f"Migrating Q&A from {json_file}")
                entries = data.get("entries", [])
                for entry in entries:
                    qa_service.add_entry(
                        question=entry.get("question", ""),
                        answer=entry.get("answer", ""),
                        source=entry.get("source", "Unknown"),
                        confidence_score=entry.get("confidence", 0.0)
                    )
                    migrated_count += 1
                
                # Backup the original file
                backup_path = json_file.with_suffix('.json.backup')
                json_file.rename(backup_path)
                logger.info(f"Backed up original file to {backup_path}")
                
            except Exception as e:
                logger.error(f"Failed to migrate {json_file}: {e}")
    
    return migrated_count


def initialize_system_config():
    """Initialize system configuration in database."""
    logger.info("Initializing system configuration")
    
    # Basic system info
    config_service.set_config("system_version", settings.APP_VERSION, "JARVIS version")
    config_service.set_config("system_start_time", datetime.utcnow().isoformat(), "System start time")
    config_service.set_config("database_initialized", True, "Database initialization flag")
    
    # Migration settings
    config_service.set_config("trading_enabled", bool(settings.ALPACA_API_KEY), "Trading functionality enabled")
    config_service.set_config("web_search_enabled", settings.WEB_SEARCH_ENABLED, "Web search enabled")
    config_service.set_config("autotrain_enabled", settings.AUTOTRAIN_ENABLED, "Auto-training enabled")
    
    logger.info("System configuration initialized")


def main():
    """Main initialization function."""
    logger.info("Starting JARVIS v3.0 database initialization")
    
    try:
        # Initialize database tables
        logger.info("Creating database tables...")
        init_database()
        
        # Migrate existing data
        logger.info("Migrating existing data...")
        memory_migrated = migrate_json_memory()
        qa_migrated = migrate_qa_memory()
        
        # Initialize system configuration
        initialize_system_config()
        
        # Summary
        logger.info("Database initialization completed successfully")
        logger.info(f"Migrated {memory_migrated} memory entries")
        logger.info(f"Migrated {qa_migrated} Q&A entries")
        
        print(f"✅ JARVIS v3.0 database initialized successfully!")
        print(f"📊 Migrated {memory_migrated} memory entries and {qa_migrated} Q&A entries")
        print(f"🗃️ Database location: {settings.DATABASE_URL}")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        print(f"❌ Database initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()