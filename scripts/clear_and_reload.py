#!/usr/bin/env python3
"""
Clear existing data and reload with proper IDs for relationships
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.knowledge_graph.neo4j_client import Neo4jClient

async def clear_and_reload():
    """Clear all data and reload with proper IDs"""
    
    # Set environment variable
    os.environ['NEO4J_PASSWORD'] = 'test1234'
    
    client = Neo4jClient()
    await client.connect()
    
    print("🧹 CLEARING ALL DATA")
    print("=" * 50)
    
    # Clear all data
    clear_query = "MATCH (n) DETACH DELETE n"
    await client.execute_write_query(clear_query, {})
    print("✅ All data cleared")
    
    await client.close()
    
    print("\n🔄 RELOADING SAMPLE DATA")
    print("=" * 50)
    
    # Now reload the sample data with proper IDs
    from scripts.load_sample_data import main as load_data
    await load_data()

if __name__ == "__main__":
    asyncio.run(clear_and_reload())