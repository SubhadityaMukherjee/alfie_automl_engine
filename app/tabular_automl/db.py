"""Database models and engine setup for tabular AutoML sessions."""

import datetime
import os

from dotenv import find_dotenv, load_dotenv
from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv(find_dotenv())

DATABASE_URL = os.getenv("TABULAR_DATABASE_CONFIG", "sqlite:///automl_sessions.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class AutoMLSession(Base):
    """SQLAlchemy model representing a tabular AutoML session.

    Stores file paths, task configuration, and creation time for a session.
    """

    __tablename__ = "automl_sessions"

    session_id = Column(String, primary_key=True, index=True)
    train_file_path = Column(String, nullable=False)
    test_file_path = Column(String, nullable=True)
    target_column = Column(String, nullable=False)
    time_stamp_column_name = Column(String, nullable=True)
    task_type = Column(String, nullable=False)
    time_budget = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


Base.metadata.create_all(bind=engine)
