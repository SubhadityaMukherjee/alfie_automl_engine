import os
import datetime
from dotenv import load_dotenv
from sqlalchemy import Column, String, Integer, DateTime, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///automl_sessions.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class AutoMLSession(Base):
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


