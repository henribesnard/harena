import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db_service.base import Base
from db_service.models.sync import (
    SyncAccount as SyncAccountModel,
    BridgeCategory as BridgeCategoryModel,
    RawTransaction as RawTransactionModel,
)


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        yield db
    finally:
        db.close()


def test_join_helpers_with_shared_models(session):
    account = SyncAccountModel(
        id=1,
        item_id=1,
        bridge_account_id=123,
        account_name="Main Account",
    )
    category = BridgeCategoryModel(id=1, bridge_category_id=10, name="Food")
    tx = RawTransactionModel(
        id=1,
        bridge_transaction_id=100,
        account_id=1,
        user_id=1,
        amount=-20.0,
        date=datetime.utcnow(),
        currency_code="EUR",
        category_id=10,
    )
    session.add_all([account, category, tx])
    session.commit()

    result = (
        session.query(RawTransactionModel, SyncAccountModel, BridgeCategoryModel)
        .join(SyncAccountModel, RawTransactionModel.account_id == SyncAccountModel.id)
        .join(
            BridgeCategoryModel,
            RawTransactionModel.category_id == BridgeCategoryModel.bridge_category_id,
        )
        .all()
    )

    assert len(result) == 1
    raw_tx, acc, cat = result[0]
    assert acc.account_name == "Main Account"
    assert cat.name == "Food"
    assert raw_tx.account_id == acc.id
    assert raw_tx.category_id == cat.bridge_category_id
