from sqlalchemy import Column, Integer, String, ForeignKey, Boolean, DateTime, Enum, JSON, Text, Float, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from enum import Enum as PyEnum
from datetime import datetime, timezone
import uuid

from sync_service.models.base import Base, TimestampMixin

class ConversationState(PyEnum):
    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"
    DELETED = "DELETED"

class MessageRole(PyEnum):
    USER = "USER"
    ASSISTANT = "ASSISTANT"
    SYSTEM = "SYSTEM"

class WebhookEvent(Base, TimestampMixin):
    __tablename__ = "webhook_events"
    
    id = Column(Integer, primary_key=True)
    event_type = Column(String, nullable=False)
    event_content = Column(JSON, nullable=False)
    raw_payload = Column(Text, nullable=False)
    signature = Column(String, nullable=True)
    processed = Column(Boolean, nullable=True, default=False)
    processing_timestamp = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(String, nullable=True)

class SyncItem(Base, TimestampMixin):
    __tablename__ = "sync_items"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    bridge_item_id = Column(Integer, nullable=False, unique=True)
    status = Column(Integer, nullable=True)
    status_code_info = Column(String, nullable=True)
    status_description = Column(String, nullable=True)
    provider_id = Column(Integer, nullable=True)
    account_types = Column(String, nullable=True)
    needs_user_action = Column(Boolean, nullable=True, default=False)
    last_successful_refresh = Column(DateTime(timezone=True), nullable=True)
    last_try_refresh = Column(DateTime(timezone=True), nullable=True)
    
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
    balance = Column(Float, nullable=True)
    currency_code = Column(String(3), nullable=True)
    
    # Relations
    item = relationship("SyncItem", back_populates="accounts")
    loan_details = relationship("LoanDetail", uselist=False, back_populates="account", cascade="all, delete-orphan")
    raw_transactions = relationship("RawTransaction", back_populates="account", cascade="all, delete-orphan")
    raw_stocks = relationship("RawStock", back_populates="account", cascade="all, delete-orphan")

class LoanDetail(Base, TimestampMixin):
    __tablename__ = "loan_details"
    
    id = Column(Integer, primary_key=True)
    account_id = Column(Integer, ForeignKey("sync_accounts.id", ondelete="CASCADE"), nullable=False, unique=True)
    interest_rate = Column(Float, nullable=True)
    next_payment_date = Column(DateTime(timezone=True), nullable=True)
    next_payment_amount = Column(Float, nullable=True)
    maturity_date = Column(DateTime(timezone=True), nullable=True)
    opening_date = Column(DateTime(timezone=True), nullable=True)
    borrowed_capital = Column(Float, nullable=True)
    repaid_capital = Column(Float, nullable=True)
    remaining_capital = Column(Float, nullable=True)
    total_estimated_interests = Column(Float, nullable=True)
    
    # Relations
    account = relationship("SyncAccount", back_populates="loan_details")

class RawTransaction(Base, TimestampMixin):
    __tablename__ = "raw_transactions"
    
    id = Column(Integer, primary_key=True)
    bridge_transaction_id = Column(Integer, nullable=False, unique=True)
    account_id = Column(Integer, ForeignKey("sync_accounts.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    clean_description = Column(String, nullable=True)
    provider_description = Column(String, nullable=True)
    amount = Column(Float, nullable=False)
    date = Column(DateTime(timezone=True), nullable=False)
    booking_date = Column(DateTime(timezone=True), nullable=True)
    transaction_date = Column(DateTime(timezone=True), nullable=True)
    value_date = Column(DateTime(timezone=True), nullable=True)
    currency_code = Column(String(3), nullable=True)
    category_id = Column(Integer, nullable=True)
    operation_type = Column(String, nullable=True)
    deleted = Column(Boolean, default=False)
    future = Column(Boolean, default=False)
    updated_at_bridge = Column(DateTime(timezone=True), nullable=True)
    
    # Relations
    account = relationship("SyncAccount", back_populates="raw_transactions")
    user = relationship("User", back_populates="raw_transactions")

class BridgeCategory(Base, TimestampMixin):
    __tablename__ = "bridge_categories"
    
    id = Column(Integer, primary_key=True)
    bridge_category_id = Column(Integer, nullable=False, unique=True)
    name = Column(String, nullable=False)
    parent_id = Column(Integer, nullable=True)
    parent_name = Column(String, nullable=True)

class RawStock(Base, TimestampMixin):
    __tablename__ = "raw_stocks"
    
    id = Column(Integer, primary_key=True)
    bridge_stock_id = Column(Integer, nullable=False, unique=True)
    account_id = Column(Integer, ForeignKey("sync_accounts.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    label = Column(String, nullable=True)
    ticker = Column(String, nullable=True)
    isin = Column(String, nullable=True)
    marketplace = Column(String, nullable=True)
    stock_key = Column(String, nullable=True)
    current_price = Column(Float, nullable=True)
    currency_code = Column(String(3), nullable=True)
    quantity = Column(Float, nullable=True)
    total_value = Column(Float, nullable=True)
    average_purchase_price = Column(Float, nullable=True)
    value_date = Column(DateTime(timezone=True), nullable=True)
    deleted = Column(Boolean, default=False)
    
    # Relations
    account = relationship("SyncAccount", back_populates="raw_stocks")
    user = relationship("User", back_populates="raw_stocks")

class AccountInformation(Base, TimestampMixin):
    __tablename__ = "account_information"
    
    id = Column(Integer, primary_key=True)
    item_id = Column(Integer, ForeignKey("sync_items.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    account_details = Column(JSON, nullable=True)
    
    # Relations
    user = relationship("User")
    item = relationship("SyncItem")

class BridgeInsight(Base, TimestampMixin):
    __tablename__ = "bridge_insights"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    global_kpis = Column(JSON, nullable=True)
    monthly_kpis = Column(JSON, nullable=True)
    oldest_existing_transaction = Column(DateTime(timezone=True), nullable=True)
    fully_analyzed_month = Column(Integer, nullable=True)
    fully_analyzed_day = Column(Integer, nullable=True)
    
    # Relations
    user = relationship("User")

class SyncTask(Base, TimestampMixin):
    __tablename__ = "sync_tasks"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    item_id = Column(Integer, ForeignKey("sync_items.id", ondelete="CASCADE"), nullable=True)
    task_type = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending")
    progress = Column(Float, nullable=True)
    parameters = Column(JSON, nullable=True)
    result_summary = Column(JSON, nullable=True)
    error_message = Column(String, nullable=True)
    scheduled_at = Column(DateTime(timezone=True), default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relations
    user = relationship("User")
    item = relationship("SyncItem")

class SyncStat(Base, TimestampMixin):
    __tablename__ = "sync_stats"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    item_id = Column(Integer, ForeignKey("sync_items.id", ondelete="CASCADE"), nullable=True)
    account_id = Column(Integer, ForeignKey("sync_accounts.id", ondelete="CASCADE"), nullable=True)
    stat_type = Column(String, nullable=False)
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)
    transactions_count = Column(Integer, nullable=True)
    metrics = Column(JSON, nullable=False)
    
    # Relations
    user = relationship("User")
    item = relationship("SyncItem")
    account = relationship("SyncAccount")