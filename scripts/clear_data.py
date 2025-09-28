#!/usr/bin/env python3
"""
Clear all data and reload simple 5-person graph
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.knowledge_graph.neo4j_client import Neo4jClient

async def clear_all_data():
    """Clear all data from Neo4j"""
    
    os.environ['NEO4J_PASSWORD'] = 'test1234'
    
    client = Neo4jClient()
    await client.connect()
    
    print("🧹 CLEARING ALL DATA FROM NEO4J")
    print("=" * 50)
    
    # Clear everything
    await client.execute_write_query("MATCH (n) DETACH DELETE n", {})
    
    # Verify it's empty
    result = await client.execute_query("MATCH (n) RETURN count(n) as count")
    count = result[0]['count']
    
    if count == 0:
        print("✅ All data cleared successfully!")
    else:
        print(f"⚠️  Warning: {count} nodes still remain")
    
    await client.close()

if __name__ == "__main__":
    asyncio.run(clear_all_data())