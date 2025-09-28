"""Graph operations with comprehensive error handling."""

from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime
from src.models.graph_models import (
    PersonNode, ProjectNode, TaskNode, DocumentNode, ConversationNode, Relationship
)
from src.knowledge_graph.neo4j_client import Neo4jClient
from src.knowledge_graph.query_builder import CypherQueryBuilder
from src.utils.error_handling import GraphOperationError
import logging

logger = logging.getLogger(__name__)


class GraphOperations:
    """High-level graph operations for the knowledge graph."""
    
    def __init__(self, client: Neo4jClient):
        """Initialize with Neo4j client."""
        self.client = client
        self.query_builder = CypherQueryBuilder()


    async def create_person(self, person: PersonNode) -> str:
        """Create a person node in the graph."""
        try:
            properties = {
                "id": person.person_id,  # Generic ID for relationships
                "person_id": person.person_id,
                "name": person.name,
                "email": person.email,
                "role": person.role.value,
                "department": person.department,
                "hire_date": person.hire_date.isoformat(),
                "permissions": person.permissions,
                "phone": person.phone,
                "manager_id": person.manager_id,
                "team_ids": person.team_ids
            }
            
            query, params = self.query_builder.create_node_query("Person", properties)
            result = await self.client.execute_write_query(query, params)
            
            if not result:
                raise GraphOperationError("Failed to create person node")
                
            logger.info(f"Successfully created person: {person.name} ({person.person_id})")
            return person.person_id
            
        except Exception as e:
            logger.error(f"Failed to create person {person.name}: {e}")
            raise GraphOperationError(f"Person creation failed: {e}")


    async def create_project(self, project: ProjectNode) -> str:
        """Create a project node in the graph."""
        try:
            properties = {
                "id": project.project_id,  # Generic ID for relationships
                "project_id": project.project_id,
                "name": project.name,
                "description": project.description,
                "status": project.status.value,
                "start_date": project.start_date.isoformat(),
                "end_date": project.end_date.isoformat() if project.end_date else None,
                "budget": project.budget,
                "owner_id": project.owner_id,
                "team_ids": project.team_ids,
                "technologies": project.technologies,
                "priority": project.priority.value
            }
            
            query, params = self.query_builder.create_node_query("Project", properties)
            result = await self.client.execute_write_query(query, params)
            
            if not result:
                raise GraphOperationError("Failed to create project node")
                
            logger.info(f"Successfully created project: {project.name} ({project.project_id})")
            return project.project_id
            
        except Exception as e:
            logger.error(f"Failed to create project {project.name}: {e}")
            raise GraphOperationError(f"Project creation failed: {e}")


    async def create_task(self, task: TaskNode) -> str:
        """Create a task node in the graph."""
        try:
            properties = {
                "id": task.task_id,  # Generic ID for relationships
                "task_id": task.task_id,
                "name": task.name,
                "description": task.description,
                "status": task.status.value,
                "priority": task.priority.value,
                "created_date": task.created_date.isoformat(),
                "due_date": task.due_date.isoformat() if task.due_date else None,
                "completed_date": task.completed_date.isoformat() if task.completed_date else None,
                "assigned_to": task.assigned_to,
                "project_id": task.project_id,
                "estimated_hours": task.estimated_hours,
                "actual_hours": task.actual_hours,
                "tags": task.tags,
                "dependencies": task.dependencies
            }
            
            query, params = self.query_builder.create_node_query("Task", properties)
            result = await self.client.execute_write_query(query, params)
            
            if not result:
                raise GraphOperationError("Failed to create task node")
                
            logger.info(f"Successfully created task: {task.name} ({task.task_id})")
            return task.task_id
            
        except Exception as e:
            logger.error(f"Failed to create task {task.name}: {e}")
            raise GraphOperationError(f"Task creation failed: {e}")


    async def create_document(self, document: DocumentNode) -> str:
        """Create a document node in the graph."""
        try:
            properties = {
                "id": document.document_id,  # Generic ID for relationships
                "document_id": document.document_id,
                "title": document.title,
                "content": document.content,
                "document_type": document.document_type,
                "created_date": document.created_date.isoformat(),
                "modified_date": document.modified_date.isoformat() if document.modified_date else None,
                "author_id": document.author_id,
                "project_id": document.project_id,
                "version": document.version,
                "file_path": document.file_path,
                "tags": document.tags,
                "metadata": document.metadata
            }
            
            query, params = self.query_builder.create_node_query("Document", properties)
            result = await self.client.execute_write_query(query, params)
            
            if not result:
                raise GraphOperationError("Failed to create document node")
                
            logger.info(f"Successfully created document: {document.title} ({document.document_id})")
            return document.document_id
            
        except Exception as e:
            logger.error(f"Failed to create document {document.title}: {e}")
            raise GraphOperationError(f"Document creation failed: {e}")


    async def create_relationship(self, relationship: Relationship) -> bool:
        """Create a relationship between two nodes."""
        try:
            # Generic relationship creation - assumes nodes exist
            query = f"""
            MATCH (from {{id: $from_id}})
            MATCH (to {{id: $to_id}}) 
            CREATE (from)-[r:{relationship.relationship_type} $properties]->(to)
            RETURN r
            """
            
            params = {
                "from_id": relationship.from_id,
                "to_id": relationship.to_id,
                "properties": {
                    **relationship.properties,
                    "created_date": relationship.created_date.isoformat()
                }
            }
            
            result = await self.client.execute_write_query(query, params)
            
            if not result:
                raise GraphOperationError("Failed to create relationship")
                
            logger.info(f"Successfully created relationship: {relationship.from_id} "
                       f"-[{relationship.relationship_type}]-> {relationship.to_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create relationship: {e}")
            raise GraphOperationError(f"Relationship creation failed: {e}")


    async def find_related_entities(self, entity_id: str, 
                                  relationship_types: Optional[List[str]] = None,
                                  max_depth: int = 2) -> List[Dict[str, Any]]:
        """Find entities related to the given entity."""
        try:
            if relationship_types:
                rel_filter = "|".join(relationship_types)
                query = f"""
                MATCH (start {{id: $entity_id}})-[r:{rel_filter}*1..{max_depth}]-(related)
                RETURN related, r
                """
            else:
                query = f"""
                MATCH (start {{id: $entity_id}})-[r*1..{max_depth}]-(related)
                RETURN related, r
                """
            
            params = {"entity_id": entity_id}
            result = await self.client.execute_query(query, params)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to find related entities: {e}")
            raise GraphOperationError(f"Find related entities failed: {e}")


    async def get_entity_by_id(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get an entity by its ID."""
        try:
            query = "MATCH (n {id: $entity_id}) RETURN n"
            params = {"entity_id": entity_id}
            
            result = await self.client.execute_query(query, params)
            
            if result:
                return result[0]['n']
            return None
            
        except Exception as e:
            logger.error(f"Failed to get entity by ID: {e}")
            raise GraphOperationError(f"Get entity failed: {e}")


    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and all its relationships."""
        try:
            query = """
            MATCH (n {id: $entity_id})
            DETACH DELETE n
            """
            params = {"entity_id": entity_id}
            
            result = await self.client.execute_write_query(query, params)
            
            logger.info(f"Successfully deleted entity: {entity_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete entity: {e}")
            raise GraphOperationError(f"Delete entity failed: {e}")


    async def update_entity_properties(self, entity_id: str, 
                                     properties: Dict[str, Any]) -> bool:
        """Update properties of an entity."""
        try:
            # Build SET clause dynamically
            set_clauses = []
            params = {"entity_id": entity_id}
            
            for key, value in properties.items():
                param_name = f"prop_{key}"
                set_clauses.append(f"n.{key} = ${param_name}")
                params[param_name] = value
            
            set_clause = ", ".join(set_clauses)
            
            query = f"""
            MATCH (n {{id: $entity_id}})
            SET {set_clause}
            RETURN n
            """
            
            result = await self.client.execute_write_query(query, params)
            
            if not result:
                raise GraphOperationError("Failed to update entity")
                
            logger.info(f"Successfully updated entity: {entity_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update entity: {e}")
            raise GraphOperationError(f"Update entity failed: {e}")