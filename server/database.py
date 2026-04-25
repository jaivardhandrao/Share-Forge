"""
Share-Forge Database Layer.

SQLAlchemy 2.x ORM with sync engine. Three tables:
  - predictions  : every /api/forecast call
  - backtests    : every /api/backtest call
  - actions_log  : every /api/predict call

The app starts even if Postgres is unreachable: get_session() returns None
and the endpoints just skip persistence. Tables are created on the first
successful connection via init_db().
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    desc,
    func,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


DEFAULT_SQLITE_URL = "sqlite:///./share_forge.db"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_SQLITE_URL)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    horizon_days = Column(Integer, nullable=False)
    horizon_label = Column(String(16), nullable=True)
    method = Column(String(64), nullable=False, default="gbm_monte_carlo")
    last_close = Column(Float, nullable=False)
    median_terminal = Column(Float, nullable=False)
    p05_terminal = Column(Float, nullable=False)
    p95_terminal = Column(Float, nullable=False)
    mu_daily = Column(Float, nullable=False)
    sigma_daily = Column(Float, nullable=False)
    payload = Column(JSON, nullable=True)


class Backtest(Base):
    __tablename__ = "backtests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    task_type = Column(String(64), nullable=False)
    grading_mode = Column(String(32), nullable=False)
    score = Column(Float, nullable=False)
    sharpe = Column(Float, nullable=False)
    max_drawdown = Column(Float, nullable=False)
    total_return = Column(Float, nullable=False)
    buy_and_hold_return = Column(Float, nullable=False)
    n_trades = Column(Integer, nullable=False)
    final_value = Column(Float, nullable=False)
    payload = Column(JSON, nullable=True)


class ActionLog(Base):
    __tablename__ = "actions_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    session_id = Column(String(128), nullable=True)
    action = Column(Integer, nullable=False)
    action_name = Column(String(8), nullable=False)
    last_close = Column(Float, nullable=False)
    is_long = Column(Boolean, nullable=False, default=False)
    source = Column(String(32), nullable=False, default="api")


_engine = None
_SessionLocal: Optional[sessionmaker] = None
_db_ready = False
_db_error: Optional[str] = None


def _build_engine():
    connect_args = {}
    if DATABASE_URL.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
    return create_engine(DATABASE_URL, pool_pre_ping=True, future=True, connect_args=connect_args)


def init_db() -> bool:
    """Create the engine + session factory and run create_all. Idempotent."""
    global _engine, _SessionLocal, _db_ready, _db_error
    if _db_ready:
        return True
    try:
        _engine = _build_engine()
        with _engine.connect() as conn:
            conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        Base.metadata.create_all(_engine)
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)
        _db_ready = True
        _db_error = None
        return True
    except SQLAlchemyError as e:
        _db_ready = False
        _db_error = str(e).splitlines()[0]
        return False
    except Exception as e:
        _db_ready = False
        _db_error = str(e).splitlines()[0]
        return False


def is_ready() -> bool:
    return _db_ready


def status() -> Dict[str, Any]:
    return {
        "url": _redacted_url(DATABASE_URL),
        "ready": _db_ready,
        "error": _db_error,
    }


def _redacted_url(url: str) -> str:
    if "@" in url and "://" in url:
        scheme, rest = url.split("://", 1)
        if "@" in rest:
            _, host = rest.split("@", 1)
            return f"{scheme}://***:***@{host}"
    return url


@contextmanager
def get_session() -> Iterator[Optional[Session]]:
    if not _db_ready:
        if not init_db():
            yield None
            return
    assert _SessionLocal is not None
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        yield None
    finally:
        session.close()


def record_prediction(payload: Dict[str, Any], horizon_label: Optional[str]) -> Optional[int]:
    fc = payload.get("forecast", {})
    median = fc.get("median") or [0.0]
    p05 = fc.get("p05") or [0.0]
    p95 = fc.get("p95") or [0.0]

    with get_session() as session:
        if session is None:
            return None
        row = Prediction(
            horizon_days=int(payload.get("horizon_days", 0)),
            horizon_label=horizon_label,
            method=str(payload.get("method", "gbm_monte_carlo")),
            last_close=float(payload.get("last_close", 0.0)),
            median_terminal=float(median[-1]),
            p05_terminal=float(p05[-1]),
            p95_terminal=float(p95[-1]),
            mu_daily=float(payload.get("mu_daily", 0.0)),
            sigma_daily=float(payload.get("sigma_daily", 0.0)),
            payload=json.loads(json.dumps(payload, default=str)),
        )
        session.add(row)
        session.flush()
        return row.id


def record_backtest(task_type: str, grading_mode: str, summary: Dict[str, Any], score: float) -> Optional[int]:
    with get_session() as session:
        if session is None:
            return None
        row = Backtest(
            task_type=task_type,
            grading_mode=grading_mode,
            score=float(score),
            sharpe=float(summary.get("sharpe", 0.0)),
            max_drawdown=float(summary.get("max_drawdown", 0.0)),
            total_return=float(summary.get("total_return", 0.0)),
            buy_and_hold_return=float(summary.get("buy_and_hold_return", 0.0)),
            n_trades=int(summary.get("n_trades", 0)),
            final_value=float(summary.get("final_value", 0.0)),
            payload=json.loads(json.dumps(summary, default=str)),
        )
        session.add(row)
        session.flush()
        return row.id


def record_action(action: int, last_close: float, is_long: bool, session_id: Optional[str], source: str = "api") -> Optional[int]:
    with get_session() as session:
        if session is None:
            return None
        row = ActionLog(
            session_id=session_id,
            action=int(action),
            action_name=["HOLD", "BUY", "SELL"][int(action)] if 0 <= int(action) <= 2 else "UNKNOWN",
            last_close=float(last_close),
            is_long=bool(is_long),
            source=source,
        )
        session.add(row)
        session.flush()
        return row.id


def list_predictions(limit: int = 50) -> List[Dict[str, Any]]:
    with get_session() as session:
        if session is None:
            return []
        rows = session.query(Prediction).order_by(desc(Prediction.created_at)).limit(limit).all()
        return [
            {
                "id": r.id,
                "created_at": r.created_at.isoformat(),
                "horizon_days": r.horizon_days,
                "horizon_label": r.horizon_label,
                "method": r.method,
                "last_close": r.last_close,
                "median_terminal": r.median_terminal,
                "p05_terminal": r.p05_terminal,
                "p95_terminal": r.p95_terminal,
            }
            for r in rows
        ]


def list_backtests(limit: int = 50) -> List[Dict[str, Any]]:
    with get_session() as session:
        if session is None:
            return []
        rows = session.query(Backtest).order_by(desc(Backtest.created_at)).limit(limit).all()
        return [
            {
                "id": r.id,
                "created_at": r.created_at.isoformat(),
                "task_type": r.task_type,
                "grading_mode": r.grading_mode,
                "score": r.score,
                "sharpe": r.sharpe,
                "max_drawdown": r.max_drawdown,
                "total_return": r.total_return,
                "buy_and_hold_return": r.buy_and_hold_return,
                "n_trades": r.n_trades,
                "final_value": r.final_value,
            }
            for r in rows
        ]


def list_actions(limit: int = 100) -> List[Dict[str, Any]]:
    with get_session() as session:
        if session is None:
            return []
        rows = session.query(ActionLog).order_by(desc(ActionLog.created_at)).limit(limit).all()
        return [
            {
                "id": r.id,
                "created_at": r.created_at.isoformat(),
                "session_id": r.session_id,
                "action": r.action,
                "action_name": r.action_name,
                "last_close": r.last_close,
                "is_long": r.is_long,
                "source": r.source,
            }
            for r in rows
        ]


def counts() -> Dict[str, int]:
    with get_session() as session:
        if session is None:
            return {"predictions": 0, "backtests": 0, "actions_log": 0}
        return {
            "predictions": int(session.query(func.count(Prediction.id)).scalar() or 0),
            "backtests": int(session.query(func.count(Backtest.id)).scalar() or 0),
            "actions_log": int(session.query(func.count(ActionLog.id)).scalar() or 0),
        }
