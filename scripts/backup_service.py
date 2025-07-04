#!/usr/bin/env python3
"""JARVIS v3.0 Backup Service."""

import os
import time
import schedule
import subprocess
from datetime import datetime
from pathlib import Path
import sys

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from utils.logging_config import setup_logging, get_logger

# Initialize logging
setup_logging()
logger = get_logger(__name__)


def run_backup():
    """Run the backup script."""
    try:
        backup_script = Path(__file__).parent / "backup.sh"
        logger.info("Starting scheduled backup")
        
        result = subprocess.run(
            ["bash", str(backup_script)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            logger.info("Backup completed successfully", extra={"output": result.stdout})
        else:
            logger.error("Backup failed", extra={"error": result.stderr})
    
    except subprocess.TimeoutExpired:
        logger.error("Backup timed out after 5 minutes")
    except Exception as e:
        logger.error(f"Backup failed with exception: {e}")


def check_disk_space():
    """Check available disk space and warn if low."""
    try:
        backup_dir = Path("/app/backups")
        backup_dir.mkdir(exist_ok=True)
        
        # Get disk usage
        statvfs = os.statvfs(backup_dir)
        free_bytes = statvfs.f_bavail * statvfs.f_frsize
        free_gb = free_bytes / (1024**3)
        
        if free_gb < 1.0:  # Less than 1GB free
            logger.warning(f"Low disk space: {free_gb:.2f}GB available")
        else:
            logger.info(f"Disk space check: {free_gb:.2f}GB available")
            
    except Exception as e:
        logger.error(f"Failed to check disk space: {e}")


def cleanup_old_backups():
    """Clean up old backup files."""
    try:
        backup_dir = Path("/app/backups")
        if not backup_dir.exists():
            return
        
        # Remove backups older than 30 days
        cutoff_time = time.time() - (30 * 24 * 60 * 60)  # 30 days ago
        
        removed_count = 0
        for backup_file in backup_dir.glob("jarvis_*"):
            if backup_file.stat().st_mtime < cutoff_time:
                backup_file.unlink()
                removed_count += 1
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old backup files")
            
    except Exception as e:
        logger.error(f"Failed to cleanup old backups: {e}")


def main():
    """Main backup service loop."""
    logger.info("JARVIS Backup Service v3.0 starting")
    
    # Schedule backups every 12 hours
    schedule.every(12).hours.do(run_backup)
    
    # Schedule disk space check every hour
    schedule.every().hour.do(check_disk_space)
    
    # Schedule cleanup every day at 2 AM
    schedule.every().day.at("02:00").do(cleanup_old_backups)
    
    # Run initial backup
    logger.info("Running initial backup")
    run_backup()
    
    # Run initial disk space check
    check_disk_space()
    
    logger.info("Backup service is running (backups every 12 hours)")
    
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Backup service shutting down")
            break
        except Exception as e:
            logger.error(f"Backup service error: {e}")
            time.sleep(300)  # Wait 5 minutes before retrying


if __name__ == "__main__":
    main()