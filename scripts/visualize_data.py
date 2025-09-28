#!/usr/bin/env python3
"""
Neo4j Data Visualization Script
Shows sample queries to visualize the knowledge graph in Neo4j Browser
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.knowledge_graph.neo4j_client import Neo4jClient
from src.utils.logging_config import setup_logging

# Setup logging
setup_logging()

class DataVisualizer:
    def __init__(self):
        self.client = Neo4jClient()
    
    async def connect(self):
        """Connect to Neo4j database"""
        await self.client.connect()
        logging.info("Connected to Neo4j for data visualization")
    
    async def close(self):
        """Close Neo4j connection"""
        await self.client.close()
    
    async def show_all_nodes(self):
        """Show all nodes in the database"""
        query = """
        MATCH (n)
        RETURN n
        LIMIT 50
        """
        try:
            result = await self.client.execute_query(query)
            logging.info(f"Found {len(result)} nodes")
            return result
        except Exception as e:
            logging.error(f"Failed to retrieve nodes: {e}")
            return []
    
    async def show_people_and_projects(self):
        """Show people and their project relationships"""
        query = """
        MATCH (p:Person)-[r]-(proj:Project)
        RETURN p, r, proj
        """
        try:
            result = await self.client.execute_query(query)
            logging.info(f"Found {len(result)} person-project relationships")
            return result
        except Exception as e:
            logging.error(f"Failed to retrieve person-project relationships: {e}")
            return []
    
    async def show_network_structure(self):
        """Show the complete network structure"""
        query = """
        MATCH (n)-[r]->(m)
        RETURN n, r, m
        LIMIT 100
        """
        try:
            result = await self.client.execute_query(query)
            logging.info(f"Found {len(result)} relationships")
            return result
        except Exception as e:
            logging.error(f"Failed to retrieve network structure: {e}")
            return []
    
    async def get_database_stats(self):
        """Get statistics about the database"""
        queries = {
            "people": "MATCH (p:Person) RETURN count(p) as count",
            "projects": "MATCH (p:Project) RETURN count(p) as count",
            "tasks": "MATCH (t:Task) RETURN count(t) as count",
            "documents": "MATCH (d:Document) RETURN count(d) as count",
            "relationships": "MATCH ()-[r]->() RETURN count(r) as count"
        }
        
        stats = {}
        for entity, query in queries.items():
            try:
                result = await self.client.execute_query(query)
                stats[entity] = result[0]['count'] if result else 0
            except Exception as e:
                logging.error(f"Failed to get {entity} count: {e}")
                stats[entity] = 0
        
        return stats

async def main():
    """Main function to visualize data"""
    visualizer = DataVisualizer()
    
    try:
        await visualizer.connect()
        
        print("\n" + "="*60)
        print("NEO4J KNOWLEDGE GRAPH VISUALIZATION")
        print("="*60)
        
        # Get database statistics
        stats = await visualizer.get_database_stats()
        print(f"\n📊 DATABASE STATISTICS:")
        print(f"   • People: {stats['people']}")
        print(f"   • Projects: {stats['projects']}")
        print(f"   • Tasks: {stats['tasks']}")
        print(f"   • Documents: {stats['documents']}")
        print(f"   • Relationships: {stats['relationships']}")
        
        # Show sample queries
        print(f"\n🔍 SAMPLE QUERIES FOR NEO4J BROWSER:")
        print(f"   Copy and paste these into the Neo4j Browser query box:")
        print(f"\n   1. Show all nodes:")
        print(f"      MATCH (n) RETURN n LIMIT 50")
        
        print(f"\n   2. Show all people:")
        print(f"      MATCH (p:Person) RETURN p")
        
        print(f"\n   3. Show all projects:")
        print(f"      MATCH (proj:Project) RETURN proj")
        
        print(f"\n   4. Show network relationships:")
        print(f"      MATCH (n)-[r]-(m) RETURN n, r, m LIMIT 25")
        
        print(f"\n   5. Show people and their roles:")
        print(f"      MATCH (p:Person) RETURN p.name, p.role, p.email")
        
        print(f"\n   6. Show project details:")
        print(f"      MATCH (proj:Project) RETURN proj.name, proj.status, proj.description")
        
        print(f"\n🌐 NEO4J BROWSER ACCESS:")
        print(f"   URL: http://localhost:7474")
        print(f"   Username: neo4j")
        print(f"   Password: test1234")
        
        print(f"\n💡 TIP: Use the graph visualization in the browser to see")
        print(f"   the knowledge graph structure and relationships!")
        
        print("\n" + "="*60)
        
    except Exception as e:
        logging.error(f"Visualization failed: {e}")
    finally:
        await visualizer.close()

if __name__ == "__main__":
    asyncio.run(main())