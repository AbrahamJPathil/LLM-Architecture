#!/usr/bin/env python3
"""
Debug relationship creation issues
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.knowledge_graph.neo4j_client import Neo4jClient

async def debug_relationships():
    """Debug why relationships aren't being created"""
    
    # Set environment variable
    os.environ['NEO4J_PASSWORD'] = 'test1234'
    
    client = Neo4jClient()
    await client.connect()
    
    print("🔍 DEBUGGING RELATIONSHIP CREATION")
    print("=" * 50)
    
    # 1. Check if nodes exist
    print("\n1. Checking existing nodes:")
    nodes_query = "MATCH (n) RETURN labels(n)[0] as type, n.id as id, n.name as name LIMIT 20"
    nodes = await client.execute_query(nodes_query)
    
    for node in nodes:
        print(f"   • {node['type']}: {node['id']} - {node['name']}")
    
    # 2. Try to create a simple relationship manually
    print("\n2. Testing relationship creation:")
    
    # Check if specific nodes exist
    check_nodes = """
    MATCH (p1:Person {id: 'p001'})
    MATCH (p2:Person {id: 'p002'})
    RETURN p1.name as person1, p2.name as person2
    """
    
    result = await client.execute_query(check_nodes)
    if result:
        print(f"   ✅ Found nodes: {result[0]['person1']} and {result[0]['person2']}")
        
        # Try creating a relationship
        create_rel = """
        MATCH (p1:Person {id: 'p001'})
        MATCH (p2:Person {id: 'p002'})
        CREATE (p1)-[r:MANAGES {since: '2021-03-20', created_date: datetime()}]->(p2)
        RETURN r
        """
        
        try:
            rel_result = await client.execute_write_query(create_rel, {})
            if rel_result:
                print("   ✅ Successfully created test relationship!")
            else:
                print("   ❌ Failed to create relationship - no result returned")
        except Exception as e:
            print(f"   ❌ Error creating relationship: {e}")
    else:
        print("   ❌ Could not find the test nodes")
    
    # 3. Check existing relationships
    print("\n3. Checking existing relationships:")
    rel_query = "MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count"
    rels = await client.execute_query(rel_query)
    
    if rels:
        for rel in rels:
            print(f"   • {rel['rel_type']}: {rel['count']} relationships")
    else:
        print("   • No relationships found")
    
    # 4. Show all relationships with nodes
    print("\n4. All relationships with connected nodes:")
    all_rels = """
    MATCH (a)-[r]->(b) 
    RETURN a.name as from_name, type(r) as relationship, b.name as to_name, properties(r) as props
    LIMIT 10
    """
    
    all_rel_results = await client.execute_query(all_rels)
    if all_rel_results:
        for rel in all_rel_results:
            print(f"   • {rel['from_name']} -[{rel['relationship']}]-> {rel['to_name']}")
    else:
        print("   • No relationships found")
    
    await client.close()

if __name__ == "__main__":
    asyncio.run(debug_relationships())