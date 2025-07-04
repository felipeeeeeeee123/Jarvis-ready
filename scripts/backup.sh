#!/bin/bash
# JARVIS v3.0 Backup Script

set -e

# Configuration
BACKUP_DIR="/app/backups"
DATE=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="jarvis_backup_${DATE}"

# Create backup directory if it doesn't exist
mkdir -p "${BACKUP_DIR}"

echo "Starting JARVIS backup at $(date)"

# Create backup archive
tar -czf "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" \
    -C /app \
    --exclude='logs/*.log' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    data/ \
    backend/database/ \
    config.json \
    .env 2>/dev/null || true

# Backup database separately
if [ -f "/app/data/jarvis.db" ]; then
    cp "/app/data/jarvis.db" "${BACKUP_DIR}/jarvis_db_${DATE}.db"
    echo "Database backed up to jarvis_db_${DATE}.db"
fi

# Keep only last 10 backups
cd "${BACKUP_DIR}"
ls -t jarvis_backup_*.tar.gz | tail -n +11 | xargs -r rm
ls -t jarvis_db_*.db | tail -n +11 | xargs -r rm

echo "Backup completed: ${BACKUP_NAME}.tar.gz"
echo "Backup size: $(du -h ${BACKUP_NAME}.tar.gz | cut -f1)"

# Log backup completion
echo "$(date): Backup completed successfully" >> /app/logs/backup.log