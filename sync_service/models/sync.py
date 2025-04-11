from sqlalchemy import Column, Integer, String, ForeignKey, Boolean, DateTime, JSON, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from user_service.models.base import Base, TimestampMixin

class SyncItem(Base, TimestampMixin):
    __tablename__ = "sync_items"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    bridge_item_id = Column(Integer, nullable=False, unique=True)
    status = Column(Integer, default=0)
    status_code_info = Column(String, nullable=True)
    status_description = Column(String, nullable=True)
    last_successful_refresh = Column(DateTime(timezone=True), nullable=True)
    last_try_refresh = Column(DateTime(timezone=True), nullable=True)
    provider_id = Column(Integer, nullable=True)
    account_types = Column(String, nullable=True)
    needs_user_action = Column(Boolean, default=False)
    
    # Relations
    user = relationship("User", back_populates="sync_items")
    accounts = relationship("SyncAccount", back_populates="item", cascade="all, delete-orphan")

class SyncAccount(Base, TimestampMixin):
    __tablename__ = "sync_accounts"
    
    id = Column(Integer, primary_key=True)
    item_id = Column(Integer, ForeignKey("sync_items.id", ondelete="CASCADE"), nullable=False)
    bridge_account_id = Column(Integer, nullable=False, unique=True)
    account_name = Column(String, nullable=True)
    account_type = Column(String, nullable=True)
    last_sync_timestamp = Column(DateTime(timezone=True), nullable=True)
    last_transaction_date = Column(DateTime(timezone=True), nullable=True)
    
    # Relations
    item = relationship("SyncItem", back_populates="accounts")

class WebhookEvent(Base, TimestampMixin):
    __tablename__ = "webhook_events"
    
    id = Column(Integer, primary_key=True)
    event_type = Column(String, nullable=False)
    event_content = Column(JSON, nullable=False)
    raw_payload = Column(Text, nullable=False)
    signature = Column(String, nullable=True)
    processed = Column(Boolean, default=False)
    processing_timestamp = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(String, nullable=True)