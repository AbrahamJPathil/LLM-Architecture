"""Custom exceptions and error handling utilities."""

from src.utils.logging_config import logger

class ContextEngineeringError(Exception):
    """Base exception for context engineering system."""
    pass

class GraphConnectionError(ContextEngineeringError):
    """Raised when Neo4j connection fails."""
    pass

class GraphOperationError(ContextEngineeringError):
    """Raised when graph operations fail."""
    pass

class ValidationError(ContextEngineeringError):
    """Raised when data validation fails."""
    pass

class ConfigurationError(ContextEngineeringError):
    """Raised when configuration is invalid."""
    pass

def handle_neo4j_error(func):
    """Decorator to handle common Neo4j errors."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Neo4j operation failed in {func.__name__}: {e}")
            if "authentication" in str(e).lower():
                raise GraphConnectionError(f"Authentication failed: {e}")
            elif "connection" in str(e).lower():
                raise GraphConnectionError(f"Connection failed: {e}")
            else:
                raise GraphOperationError(f"Operation failed: {e}")
    return wrapper