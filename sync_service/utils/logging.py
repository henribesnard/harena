# sync_service/utils/logging.py
import json
import logging
from datetime import datetime

class StructuredLogRecord(logging.LogRecord):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timestamp = datetime.utcnow().isoformat()

class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": getattr(record, "timestamp", datetime.utcnow().isoformat()),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        
        if hasattr(record, "user_id"):
            log_obj["user_id"] = record.user_id
            
        if hasattr(record, "bridge_item_id"):
            log_obj["bridge_item_id"] = record.bridge_item_id
            
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_obj)

def setup_structured_logging():
    """Configure structured logging for the sync service."""
    logging.setLogRecordFactory(StructuredLogRecord)
    
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredFormatter())
    
    logger = logging.getLogger("sync_service")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    return logger