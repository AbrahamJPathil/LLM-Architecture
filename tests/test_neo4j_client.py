"""Tests for Neo4j client functionality."""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from src.knowledge_graph.neo4j_client import Neo4jClient
from src.utils.error_handling import GraphConnectionError, ConfigurationError

class TestNeo4jClient:
    """Test cases for Neo4j client."""
    
    @pytest.fixture
    def client(self):
        """Create Neo4j client for testing."""
        with patch('src.knowledge_graph.neo4j_client.yaml.safe_load') as mock_yaml:
            mock_yaml.return_value = {
                "neo4j": {
                    "uri": "bolt://localhost:7687",
                    "username": "neo4j",
                    "password": "test_password",
                    "database": "test_db"
                }
            }
            return Neo4jClient("test_config.yaml")
    
    @pytest.mark.asyncio
    async def test_connect_success(self, client):
        """Test successful connection to Neo4j."""
        with patch('src.knowledge_graph.neo4j_client.AsyncGraphDatabase.driver') as mock_driver:
            mock_driver_instance = AsyncMock()
            mock_driver.return_value = mock_driver_instance
            mock_driver_instance.verify_connectivity = AsyncMock()
            
            await client.connect()
            
            assert client.driver is not None
            mock_driver_instance.verify_connectivity.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_failure(self, client):
        """Test connection failure handling."""
        with patch('src.knowledge_graph.neo4j_client.AsyncGraphDatabase.driver') as mock_driver:
            mock_driver.side_effect = Exception("Connection failed")
            
            with pytest.raises(GraphConnectionError):
                await client.connect()
    
    @pytest.mark.asyncio
    async def test_execute_query_success(self, client):
        """Test successful query execution."""
        # Setup mock driver
        mock_driver = AsyncMock()
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        
        mock_driver.session.return_value.__aenter__.return_value = mock_session
        mock_session.run.return_value = mock_result
        mock_result.data.return_value = [{"test": "data"}]
        
        client.driver = mock_driver
        
        result = await client.execute_query("MATCH (n) RETURN n", {"param": "value"})
        
        assert result == [{"test": "data"}]
        mock_session.run.assert_called_once_with("MATCH (n) RETURN n", {"param": "value"})
    
    @pytest.mark.asyncio
    async def test_execute_query_no_connection(self, client):
        """Test query execution without connection."""
        with pytest.raises(GraphConnectionError, match="Not connected to database"):
            await client.execute_query("MATCH (n) RETURN n")
    
    @pytest.mark.asyncio
    async def test_close_connection(self, client):
        """Test closing database connection."""
        mock_driver = AsyncMock()
        client.driver = mock_driver
        
        await client.close()
        
        mock_driver.close.assert_called_once()
        assert client.driver is None