"""Tests for graph models and data validation."""

import pytest
from datetime import datetime
from src.models.graph_models import (
    PersonNode, ProjectNode, TaskNode, DocumentNode, ConversationNode, Relationship,
    UserRole, ProjectStatus, TaskStatus, Priority
)
from src.utils.error_handling import ValidationError

class TestGraphModels:
    """Test cases for graph models."""
    
    def test_person_node_valid_creation(self):
        """Test creating a valid person node."""
        person = PersonNode(
            person_id="p001",
            name="  John Doe  ",  # Test trimming
            email="  JOHN.DOE@EXAMPLE.COM  ",  # Test normalization
            role=UserRole.DEVELOPER,
            department="Engineering",
            hire_date=datetime(2023, 1, 15),
            permissions=["read", "write"],
            phone="+1-555-0123",
            manager_id="p002"
        )
        
        assert person.name == "John Doe"
        assert person.email == "john.doe@example.com"
        assert person.role == UserRole.DEVELOPER
    
    def test_person_node_invalid_email(self):
        """Test person node with invalid email."""
        with pytest.raises(Exception):  # deal.PreContractError in real usage
            PersonNode(
                person_id="p001",
                name="John Doe",
                email="invalid-email",  # No @ symbol
                role=UserRole.DEVELOPER,
                department="Engineering",
                hire_date=datetime(2023, 1, 15)
            )
    
    def test_project_node_valid_creation(self):
        """Test creating a valid project node."""
        project = ProjectNode(
            project_id="proj001",
            name="  Test Project  ",  # Test trimming
            description="A comprehensive test project",
            status=ProjectStatus.ACTIVE,
            start_date=datetime(2023, 6, 1),
            end_date=datetime(2024, 6, 1),
            priority=Priority.HIGH,
            budget=100000.0,
            owner_id="p001",
            tags=["important", "quarterly"]
        )
        
        assert project.name == "Test Project"
        assert project.budget == 100000.0
        assert project.status == ProjectStatus.ACTIVE
    
    def test_project_node_invalid_dates(self):
        """Test project node with invalid date range."""
        with pytest.raises(ValidationError):
            ProjectNode(
                project_id="proj001",
                name="Test Project",
                description="A test project",
                status=ProjectStatus.ACTIVE,
                start_date=datetime(2024, 6, 1),
                end_date=datetime(2023, 6, 1),  # End before start
                priority=Priority.HIGH
            )
    
    def test_project_node_negative_budget(self):
        """Test project node with negative budget."""
        with pytest.raises(Exception):  # deal.PreContractError in real usage
            ProjectNode(
                project_id="proj001",
                name="Test Project",
                description="A test project",
                status=ProjectStatus.ACTIVE,
                start_date=datetime(2023, 6, 1),
                end_date=datetime(2024, 6, 1),
                priority=Priority.HIGH,
                budget=-1000.0  # Negative budget
            )
    
    def test_task_node_valid_creation(self):
        """Test creating a valid task node."""
        task = TaskNode(
            task_id="task001",
            title="  Implement Feature  ",  # Test trimming
            description="Implement the new user authentication feature",
            status=TaskStatus.IN_PROGRESS,
            priority=Priority.MEDIUM,
            created_date=datetime(2023, 7, 1),
            due_date=datetime(2023, 7, 15),
            estimated_hours=8.0,
            assignee_id="p001",
            project_id="proj001"
        )
        
        assert task.title == "Implement Feature"
        assert task.estimated_hours == 8.0
        assert task.status == TaskStatus.IN_PROGRESS
    
    def test_document_node_valid_creation(self):
        """Test creating a valid document node."""
        document = DocumentNode(
            document_id="doc001",
            title="  Project Requirements  ",  # Test trimming
            content_type="application/pdf",
            file_path="/documents/requirements.pdf",
            created_date=datetime(2023, 6, 1),
            last_modified=datetime(2023, 6, 15),
            author_id="p001",
            size_bytes=1024000,
            version="2.1",
            tags=["requirements", "specification"],
            project_ids=["proj001"]
        )
        
        assert document.title == "Project Requirements"
        assert document.size_bytes == 1024000
        assert document.version == "2.1"
    
    def test_conversation_node_valid_creation(self):
        """Test creating a valid conversation node."""
        conversation = ConversationNode(
            conversation_id="conv001",
            title="  Daily Standup  ",  # Test trimming
            start_time=datetime(2023, 7, 3, 9, 0),
            end_time=datetime(2023, 7, 3, 9, 30),
            participant_ids=["p001", "p002", "p003"],
            channel="slack",
            topic="Sprint progress review",
            message_count=15,
            project_ids=["proj001"]
        )
        
        assert conversation.title == "Daily Standup"
        assert len(conversation.participant_ids) == 3
        assert conversation.message_count == 15
    
    def test_conversation_node_no_participants(self):
        """Test conversation node with no participants."""
        with pytest.raises(Exception):  # deal.PreContractError in real usage
            ConversationNode(
                conversation_id="conv001",
                title="Empty Meeting",
                start_time=datetime(2023, 7, 3, 9, 0),
                end_time=datetime(2023, 7, 3, 9, 30),
                participant_ids=[],  # No participants
                channel="slack"
            )
    
    def test_relationship_valid_creation(self):
        """Test creating a valid relationship."""
        relationship = Relationship(
            from_id="p001",
            to_id="proj001",
            relationship_type="works_on",
            properties={"role": "lead_developer", "start_date": "2023-06-01"},
            created_date=datetime(2023, 6, 1)
        )
        
        assert relationship.from_id == "p001"
        assert relationship.to_id == "proj001"
        assert relationship.relationship_type == "WORKS_ON"  # Should be uppercase
        assert relationship.properties["role"] == "lead_developer"
    
    def test_relationship_empty_ids(self):
        """Test relationship with empty IDs."""
        with pytest.raises(Exception):  # deal.PreContractError in real usage
            Relationship(
                from_id="",  # Empty ID
                to_id="proj001",
                relationship_type="WORKS_ON"
            )