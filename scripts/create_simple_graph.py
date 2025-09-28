#!/usr/bin/env python3
"""
Create a simple graph with only 5 people and clear relationships
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.knowledge_graph.neo4j_client import Neo4jClient
from src.knowledge_graph.graph_operations import GraphOperations
from src.models.graph_models import PersonNode, Relationship, UserRole
from src.utils.logging_config import setup_logging

async def create_simple_graph():
    """Create a simple graph with 5 people and basic relationships."""
    
    # Setup
    setup_logging("INFO")
    os.environ['NEO4J_PASSWORD'] = 'test1234'
    
    client = Neo4jClient()
    graph_ops = GraphOperations(client)
    
    try:
        await client.connect()
        
        print("🧹 CLEARING ALL EXISTING DATA")
        print("=" * 50)
        
        # Clear all existing data
        await client.execute_write_query("MATCH (n) DETACH DELETE n", {})
        print("✅ All data cleared")
        
        print("\n👥 CREATING 5 SIMPLE PEOPLE")
        print("=" * 50)
        
        # Create 5 simple people
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
        
        # Create people
        for person in people:
            await graph_ops.create_person(person)
            print(f"   ✅ Created: {person.name} ({person.role.value}) - {person.department}")
        
        print(f"\n🔗 CREATING SIMPLE RELATIONSHIPS")
        print("=" * 50)
        
        # Create simple relationships
        relationships = [
            # Management hierarchy
            Relationship("p001", "p002", "MANAGES", {"since": "2021-03-20", "type": "direct_report"}),
            Relationship("p002", "p003", "MANAGES", {"since": "2022-06-10", "type": "direct_report"}),
            Relationship("p002", "p005", "MANAGES", {"since": "2023-01-12", "type": "direct_report"}),
            
            # Team collaboration
            Relationship("p003", "p005", "COLLABORATES_WITH", {"project": "web_app", "since": "2023-02-01"}),
            Relationship("p004", "p003", "WORKS_WITH", {"project": "ui_design", "since": "2023-03-15"})
        ]
        
        for rel in relationships:
            await graph_ops.create_relationship(rel)
            print(f"   ✅ Created: {rel.from_id} -[{rel.relationship_type}]-> {rel.to_id}")
        
        print(f"\n📊 FINAL STATISTICS")
        print("=" * 50)
        
        # Get final stats
        people_count = await client.execute_query("MATCH (p:Person) RETURN count(p) as count")
        rel_count = await client.execute_query("MATCH ()-[r]->() RETURN count(r) as count")
        
        print(f"   👥 People: {people_count[0]['count']}")
        print(f"   🔗 Relationships: {rel_count[0]['count']}")
        
        print(f"\n🌐 NEO4J BROWSER QUERIES")
        print("=" * 50)
        print("   URL: http://localhost:7474")
        print("   Username: neo4j")
        print("   Password: test1234")
        print()
        print("   Try these simple queries:")
        print("   1. Show all people:")
        print("      MATCH (p:Person) RETURN p")
        print()
        print("   2. Show all relationships:")
        print("      MATCH (a)-[r]->(b) RETURN a.name, type(r), b.name")
        print()
        print("   3. Show management hierarchy:")
        print("      MATCH (manager)-[r:MANAGES]->(employee) RETURN manager.name, employee.name")
        print()
        print("   4. Show network graph:")
        print("      MATCH (n)-[r]-(m) RETURN n, r, m")
        
        print(f"\n✅ SIMPLE GRAPH CREATED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(create_simple_graph())