#!/usr/bin/env python3
# Context Engineering System - Main Entry Point

import asyncio
import sys
import os
from pathlib import Path
from typing import Optional

# Add src to Python path for module imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.knowledge_graph.neo4j_client import Neo4jClient
from src.knowledge_graph.graph_operations import GraphOperations
from src.models.graph_models import PersonNode, UserRole
from src.utils.logging_config import setup_logging, logger
from src.utils.error_handling import ContextEngineeringError
from datetime import datetime


class ContextEngineeringSystem:
    # Main context engineering system orchestrator
    
    def __init__(self, config_path: Optional[str] = None):
        # Initialize the context engineering system
        self.config_path = config_path or "config/neo4j_config.yaml"
        self.client = Neo4jClient(self.config_path)
        self.graph_ops = GraphOperations(self.client)
        
    async def initialize(self):
        # Initialize the system and establish database connections
        try:
            logger.info("Initializing Context Engineering System...")
            await self.client.connect()
            logger.info("System initialized successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            return False
    
    async def test_basic_operations(self):
        # Test basic system operations to verify functionality
        try:
            logger.info("Testing basic graph operations...")
            
            # Create a test person
            test_person = PersonNode(
                person_id="test001",
                name="Test User",
                email="test@example.com",
                role=UserRole.DEVELOPER,
                department="Engineering",
                hire_date=datetime.now(),
                permissions=["read", "write"],
                phone="+1-555-0123",
                manager_id=None,
                team_ids=["team001"]
            )
            
            # Test person creation
            person_id = await self.graph_ops.create_person(test_person)
            logger.info(f"Created test person with ID: {person_id}")
            
            # Test entity retrieval
            retrieved_person = await self.graph_ops.get_entity_by_id(person_id)
            if retrieved_person:
                logger.info("Successfully retrieved created person")
                return True
            else:
                logger.error("Failed to retrieve created person")
                return False
                
        except Exception as e:
            logger.error(f"Basic operations test failed: {e}")
            return False
    
    async def shutdown(self):
        # Perform clean shutdown of the system
        try:
            logger.info("Shutting down Context Engineering System...")
            await self.client.close()
            logger.info("System shutdown complete.")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


async def main():
    # Main application entry point for the Context Engineering System
    # Set up environment variables for development
    os.environ.setdefault('NEO4J_PASSWORD', 'test1234')
    
    # Configure logging system
    setup_logging("INFO")
    
    # Create system instance
    system = ContextEngineeringSystem()
    
    try:
        print("🚀 Starting Context Engineering System...")
        
        # Initialize the system
        if not await system.initialize():
            print("❌ System initialization failed!")
            return
        
        print("✅ System initialized successfully!")
        
        # Run basic functionality tests
        print("🔍 Running system tests...")
        if await system.test_basic_operations():
            print("✅ All tests passed!")
        else:
            print("⚠️ Some tests failed - check logs for details")
        
        print("📊 System Status:")
        print("   • Neo4j Connection: Active")
        print("   • Graph Operations: Available")
        print("   • Ready for Knowledge Graph Operations")
        
        print("\n💡 Next Steps:")
        print("   • Load sample data: python scripts/load_sample_data.py")
        print("   • View in Neo4j Browser: http://localhost:7474")
        print("   • Credentials: neo4j/test1234")
        
    except ContextEngineeringError as e:
        logger.error(f"Context Engineering Error: {e}")
        print(f"❌ System Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Unexpected Error: {e}")
    finally:
        # Ensure clean shutdown
        await system.shutdown()
        print("👋 Context Engineering System shutdown complete.")


if __name__ == "__main__":
    # Script execution entry point
    asyncio.run(main())
