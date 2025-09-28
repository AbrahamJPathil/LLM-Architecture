"""Tests for graph operations functionality."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from src.knowledge_graph.graph_operations import GraphOperations
from src.models.graph_models import PersonNode, ProjectNode, UserRole, ProjectStatus, Priority
from src.utils.error_handling import GraphOperationError

class TestGraphOperations:
    """Test cases for graph operations."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock Neo4j client."""
        return AsyncMock()
    
    @pytest.fixture
    def graph_ops(self, mock_client):
        """Create graph operations instance."""
        return GraphOperations(mock_client)
    
    @pytest.fixture
    def sample_person(self):
        """Create sample person node."""
        return PersonNode(
            person_id="p001",
            name="John Doe",
            email="john.doe@example.com",
            role=UserRole.DEVELOPER,
            department="Engineering",
            hire_date=datetime(2023, 1, 15),
            permissions=["read", "write"]
        )
    
    @pytest.fixture
    def sample_project(self):
        """Create sample project node."""
        return ProjectNode(
            project_id="proj001",
            name="Test Project",
            description="A test project",
            status=ProjectStatus.ACTIVE,
            start_date=datetime(2023, 6, 1),
            end_date=datetime(2024, 6, 1),
            priority=Priority.HIGH,
            budget=100000.0
        )
    
    @pytest.mark.asyncio
    async def test_create_person_success(self, graph_ops, mock_client, sample_person):
        """Test successful person creation."""
        mock_client.execute_write_query.return_value = [{"n": "person_node"}]
        
        result = await graph_ops.create_person(sample_person)
        
        assert result == "p001"
        mock_client.execute_write_query.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_person_failure(self, graph_ops, mock_client, sample_person):
        """Test person creation failure."""
        mock_client.execute_write_query.return_value = []
        
        with pytest.raises(GraphOperationError):
            await graph_ops.create_person(sample_person)
    
    @pytest.mark.asyncio
    async def test_create_project_success(self, graph_ops, mock_client, sample_project):
        """Test successful project creation."""
        mock_client.execute_write_query.return_value = [{"n": "project_node"}]
        
        result = await graph_ops.create_project(sample_project)
        
        assert result == "proj001"
        mock_client.execute_write_query.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_find_related_entities_success(self, graph_ops, mock_client):
        """Test finding related entities."""
        mock_client.execute_query.return_value = [
            {
                "related": {"id": "rel001", "name": "Related Entity"},
                "node_labels": ["Person"],
                "last_relationship_type": "WORKS_ON"
            }
        ]
        
        result = await graph_ops.find_related_entities("p001", max_depth=2)
        
        assert len(result) == 1
        assert result[0]["related"]["id"] == "rel001"
        mock_client.execute_query.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_find_related_entities_with_relationship_filter(self, graph_ops, mock_client):
        """Test finding related entities with relationship type filter."""
        mock_client.execute_query.return_value = []
        
        result = await graph_ops.find_related_entities(
            "p001", 
            relationship_types=["WORKS_ON", "MANAGES"],
            max_depth=1
        )
        
        assert len(result) == 0
        mock_client.execute_query.assert_called_once()
        
        # Verify query contains relationship filter
        call_args = mock_client.execute_query.call_args
        query = call_args[0][0]
        assert "WORKS_ON|MANAGES" in query