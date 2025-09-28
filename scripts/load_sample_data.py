#!/usr/bin/env python3
"""
Load sample data into the Neo4j knowledge graph for testing and demonstration.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge_graph.neo4j_client import Neo4jClient
from src.knowledge_graph.graph_operations import GraphOperations
from src.models.graph_models import (
    PersonNode, ProjectNode, TaskNode, DocumentNode, ConversationNode, Relationship,
    UserRole, ProjectStatus, TaskStatus, Priority
)
from src.utils.logging_config import setup_logging, logger
from src.utils.error_handling import ContextEngineeringError

class SampleDataLoader:
    """Load sample data for demonstration and testing."""
    
    def __init__(self):
        """Initialize the sample data loader."""
        self.client = Neo4jClient()
        self.graph_ops = GraphOperations(self.client)
        
    async def load_all_sample_data(self):
        """Load all sample data into the graph."""
        try:
            logger.info("Starting sample data loading process...")
            
            # Connect to Neo4j
            await self.client.connect()
            
            # Load simple data - only people and relationships
            await self.load_sample_people()
            # Skip complex entities for simplicity
            # await self.load_sample_projects()
            # await self.load_sample_tasks()
            # await self.load_sample_documents()
            # await self.load_sample_conversations()
            await self.load_sample_relationships()
            
            logger.info("Simple sample data loading completed successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load sample data: {e}")
            raise
        finally:
            await self.client.close()
    
    async def load_sample_people(self):
        """Load 5 simple people into the graph."""
        logger.info("Loading 5 simple people...")
        
        people = [
            PersonNode(
                person_id="p001",
                name="Alice Johnson",
                email="alice@company.com",
                role=UserRole.EXECUTIVE,
                department="Leadership",
                hire_date=datetime(2020, 1, 15),
                permissions=["admin"],
                phone="+1-555-0101"
            ),
            PersonNode(
                person_id="p002",
                name="Bob Smith",
                email="bob@company.com",
                role=UserRole.MANAGER,
                department="Engineering",
                hire_date=datetime(2021, 3, 20),
                permissions=["manage"],
                phone="+1-555-0102"
            ),
            PersonNode(
                person_id="p003",
                name="Carol Davis",
                email="carol@company.com",
                role=UserRole.DEVELOPER,
                department="Engineering",
                hire_date=datetime(2022, 6, 10),
                permissions=["code"],
                phone="+1-555-0103"
            ),
            PersonNode(
                person_id="p004",
                name="David Wilson",
                email="david@company.com",
                role=UserRole.DESIGNER,
                department="Design",
                hire_date=datetime(2022, 8, 5),
                permissions=["design"],
                phone="+1-555-0104"
            ),
            PersonNode(
                person_id="p005",
                name="Emma Brown",
                email="emma@company.com",
                role=UserRole.DEVELOPER,
                department="Engineering",
                hire_date=datetime(2023, 1, 12),
                permissions=["code"],
                phone="+1-555-0105"
            )
        ]
        
        for person in people:
            try:
                await self.graph_ops.create_person(person)
            except Exception as e:
                logger.warning(f"Failed to create person {person.name}: {e}")
    
    async def load_sample_projects(self):
        """Load sample projects into the graph."""
        logger.info("Loading sample projects...")
        
        projects = [
            ProjectNode(
                project_id="proj001",
                name="Customer Portal Redesign",
                description="Complete redesign of the customer-facing web portal",
                status=ProjectStatus.ACTIVE,
                start_date=datetime(2023, 6, 1),
                end_date=datetime(2024, 3, 31),
                priority=Priority.HIGH,
                budget=250000.0,
                owner_id="p002",
                department_id="Engineering",
                tags=["web", "ui/ux", "customer-facing"]
            ),
            ProjectNode(
                project_id="proj002",
                name="Data Analytics Platform",
                description="Internal platform for business intelligence and analytics",
                status=ProjectStatus.ACTIVE,
                start_date=datetime(2023, 8, 15),
                end_date=datetime(2024, 6, 30),
                priority=Priority.MEDIUM,
                budget=180000.0,
                owner_id="p003",
                department_id="Engineering",
                tags=["analytics", "data", "internal-tools"]
            ),
            ProjectNode(
                project_id="proj003",
                name="Mobile App Development",
                description="Native mobile applications for iOS and Android",
                status=ProjectStatus.PLANNING,
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 9, 30),
                priority=Priority.HIGH,
                budget=320000.0,
                owner_id="p002",
                department_id="Engineering",
                tags=["mobile", "ios", "android"]
            )
        ]
        
        for project in projects:
            try:
                await self.graph_ops.create_project(project)
            except Exception as e:
                logger.warning(f"Failed to create project {project.name}: {e}")
    
    async def load_sample_tasks(self):
        """Load sample tasks into the graph."""
        logger.info("Loading sample tasks...")
        
        tasks = [
            TaskNode(
                task_id="task001",  
                title="Design new user interface",
                description="Create wireframes and mockups for the redesigned portal",
                status=TaskStatus.DONE,
                priority=Priority.HIGH,
                created_date=datetime(2023, 6, 5),
                due_date=datetime(2023, 7, 15),
                completed_date=datetime(2023, 7, 12),
                estimated_hours=40.0,
                actual_hours=38.5,
                assignee_id="p004",
                project_id="proj001",
                tags=["design", "ui/ux"]
            ),
            TaskNode(
                task_id="task002",
                title="Implement authentication system",
                description="Build secure user authentication and authorization",
                status=TaskStatus.IN_PROGRESS,
                priority=Priority.HIGH,
                created_date=datetime(2023, 7, 1),
                due_date=datetime(2023, 8, 15),
                estimated_hours=60.0,
                actual_hours=25.0,
                assignee_id="p003",
                project_id="proj001",
                tags=["backend", "security"]
            ),
            TaskNode(
                task_id="task003",
                title="Set up data pipeline",
                description="Configure ETL processes for analytics platform",
                status=TaskStatus.TODO,
                priority=Priority.MEDIUM,
                created_date=datetime(2023, 8, 20),
                due_date=datetime(2023, 10, 1),
                estimated_hours=45.0,
                assignee_id="p003",
                project_id="proj002",
                tags=["data", "etl", "pipeline"]
            )
        ]
        
        for task in tasks:
            try:
                await self.graph_ops.create_task(task)
            except Exception as e:
                logger.warning(f"Failed to create task {task.title}: {e}")
    
    async def load_sample_documents(self):
        """Load sample documents into the graph."""
        logger.info("Loading sample documents...")
        
        documents = [
            DocumentNode(
                document_id="doc001",
                title="Project Requirements Document",
                content_type="application/pdf",
                file_path="/documents/proj001/requirements.pdf",
                created_date=datetime(2023, 5, 20),
                last_modified=datetime(2023, 6, 15),
                author_id="p002",
                size_bytes=2048000,
                version="2.1",
                tags=["requirements", "specification"],
                project_ids=["proj001"]
            ),
            DocumentNode(
                document_id="doc002",
                title="API Documentation",
                content_type="text/markdown",
                file_path="/documents/proj001/api-docs.md",
                created_date=datetime(2023, 7, 10),
                last_modified=datetime(2023, 7, 25),
                author_id="p003",
                size_bytes=512000,
                version="1.0",
                tags=["api", "documentation", "technical"],
                project_ids=["proj001", "proj002"]
            ),
            DocumentNode(
                document_id="doc003",
                title="Design System Guidelines",
                content_type="application/pdf",
                file_path="/documents/design/design-system.pdf",
                created_date=datetime(2023, 6, 1),
                last_modified=datetime(2023, 7, 20),
                author_id="p004",
                size_bytes=5120000,
                version="1.5",
                tags=["design", "guidelines", "standards"],
                project_ids=["proj001", "proj003"]
            )
        ]
        
        for document in documents:
            try:
                await self.graph_ops.create_document(document)
            except Exception as e:
                logger.warning(f"Failed to create document {document.title}: {e}")
    
    async def load_sample_conversations(self):
        """Load sample conversations into the graph."""
        logger.info("Loading sample conversations...")
        
        conversations = [
            ConversationNode(
                conversation_id="conv001",
                title="Daily Standup - July 15",
                start_time=datetime(2023, 7, 15, 9, 0),
                end_time=datetime(2023, 7, 15, 9, 30),
                participant_ids=["p002", "p003", "p004"],
                channel="slack",
                topic="Sprint progress and blockers",
                message_count=23,
                project_ids=["proj001"]
            ),
            ConversationNode(
                conversation_id="conv002",
                title="Architecture Review Meeting",
                start_time=datetime(2023, 7, 20, 14, 0),
                end_time=datetime(2023, 7, 20, 15, 30),
                participant_ids=["p001", "p002", "p003"],
                channel="zoom",
                topic="Technical architecture decisions",
                message_count=45,
                project_ids=["proj001", "proj002"]
            )
        ]
        
        for conversation in conversations:
            try:
                # Note: ConversationNode creation not implemented in GraphOperations
                # This would need to be added similar to other create methods
                logger.info(f"Would create conversation: {conversation.title}")
            except Exception as e:
                logger.warning(f"Failed to create conversation {conversation.title}: {e}")
    
    async def load_sample_relationships(self):
        """Load simple relationships between 5 people."""
        logger.info("Loading simple relationships...")
        
        relationships = [
            # Simple management hierarchy
            Relationship("p001", "p002", "MANAGES", {"since": "2021-03-20", "type": "direct_report"}),
            Relationship("p002", "p003", "MANAGES", {"since": "2022-06-10", "type": "direct_report"}),
            Relationship("p002", "p005", "MANAGES", {"since": "2023-01-12", "type": "direct_report"}),
            
            # Team collaboration
            Relationship("p003", "p005", "COLLABORATES_WITH", {"project": "web_app", "since": "2023-02-01"}),
            Relationship("p004", "p003", "WORKS_WITH", {"project": "ui_design", "since": "2023-03-15"}),
        ]
        
        for relationship in relationships:
            try:
                await self.graph_ops.create_relationship(relationship)
            except Exception as e:
                logger.warning(f"Failed to create relationship {relationship.from_id} -> {relationship.to_id}: {e}")

async def main():
    """Main function to load sample data."""
    # Setup logging
    setup_logging("INFO")
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Create and run the loader
    loader = SampleDataLoader()
    
    try:
        await loader.load_all_sample_data()
        print("✅ Simple sample data loaded successfully!")
        print("\nSimple organizational structure created:")
        print("- 5 People: Alice (Executive), Bob (Manager), Carol (Developer), David (Designer), Emma (Developer)")
        print("- 5 Relationships: Management hierarchy and team collaboration")
        print("\nTry these Neo4j Browser queries:")
        print("  MATCH (p:Person) RETURN p")
        print("  MATCH (a)-[r]->(b) RETURN a.name, type(r), b.name")
        
    except ContextEngineeringError as e:
        print(f"❌ Context engineering error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())