"""Database utilities and session management."""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.models.schema import Base
from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Database:
    """Database connection manager."""
    
    def __init__(self, database_url: str = None):
        config = get_config()
        self.database_url = database_url or config.settings.database_url
        
        self.engine = create_engine(
            self.database_url,
            echo=False,
            pool_size=10,
            max_overflow=20,
        )
        
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
            bind=self.engine,
        )
        
        logger.info("Database initialized", database_url=self.database_url)
    
    def create_tables(self):
        """Create all tables in the database."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created")
    
    def drop_tables(self):
        """Drop all tables in the database."""
        Base.metadata.drop_all(bind=self.engine)
        logger.info("Database tables dropped")
    
    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """
        Provide a transactional scope for database operations.
        
        Usage:
            with db.session() as session:
                session.query(...)
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()


# Global database instance
_db = None


def get_database() -> Database:
    """Get or create global database instance."""
    global _db
    if _db is None:
        _db = Database()
    return _db


def init_database():
    """Initialize database and create tables."""
    db = get_database()
    db.create_tables()
