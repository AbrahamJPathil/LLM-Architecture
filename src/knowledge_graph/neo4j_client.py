"""Neo4j database client with connection management."""

import asyncio
from typing import Dict, List, Any, Optional
import yaml
import os
from neo4j import GraphDatabase, AsyncGraphDatabase
import deal
from src.utils.error_handling import GraphConnectionError, ConfigurationError, GraphOperationError
from src.utils.logging_config import logger

class Neo4jClient:
    """Neo4j database client with async support."""
    
    def __init__(self, config_path: str = "config/neo4j_config.yaml"):
        """Initialize Neo4j client with configuration."""
        self.config = self._load_config(config_path)
        self.driver = None
        self.database = self.config.get("neo4j", {}).get("database", "neo4j")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Substitute environment variables
            neo4j_config = config.get("neo4j", {})
            password = neo4j_config.get("password", "")
            if password.startswith("${") and password.endswith("}"):
                env_var = password[2:-1]
                password = os.environ.get(env_var)
                if not password:
                    raise ConfigurationError(f"Environment variable {env_var} not set")
                neo4j_config["password"] = password
                
            return config
            
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {
                "neo4j": {
                    "uri": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                    "username": os.environ.get("NEO4J_USERNAME", "neo4j"),
                    "password": os.environ.get("NEO4J_PASSWORD", "password"),
                    "database": "neo4j"
                }
            }
        except Exception as e:
            raise ConfigurationError(f"Failed to load config: {e}")
    
    async def connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            neo4j_config = self.config["neo4j"]
            
            self.driver = AsyncGraphDatabase.driver(
                neo4j_config["uri"],
                auth=(neo4j_config["username"], neo4j_config["password"]),
                max_connection_pool_size=neo4j_config.get("max_connection_pool_size", 50),
                connection_timeout=neo4j_config.get("connection_timeout", 5)
            )
            
            # Test connection
            await self.driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j database")
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise GraphConnectionError(f"Connection failed: {e}")
    

    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results."""
        if not self.driver:
            raise GraphConnectionError("Not connected to database")
            
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(query, params or {})
                records = await result.data()
                logger.debug(f"Query executed successfully, returned {len(records)} records")
                return records
                
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise GraphOperationError(f"Query execution failed: {e}")
    
    async def execute_write_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a write query within a transaction."""
        if not self.driver:
            raise GraphConnectionError("Not connected to database")
            
        try:
            async with self.driver.session(database=self.database) as session:
                async def write_transaction(tx):
                    result = await tx.run(query, params or {})
                    return await result.data()
                
                result = await session.execute_write(write_transaction)
                logger.debug(f"Write query executed successfully")
                return result
                
        except Exception as e:
            logger.error(f"Write query execution failed: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise GraphOperationError(f"Write query execution failed: {e}")
    
    async def close(self) -> None:
        """Close the database connection."""
        if self.driver:
            await self.driver.close()
            self.driver = None
            logger.info("Neo4j connection closed")