"""Cypher query builder utilities."""

from typing import Dict, List, Any, Optional
import deal

class CypherQueryBuilder:
    """Utility class for building Cypher queries."""
    
    @staticmethod
    def create_node_query(node_type: str, properties: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Build CREATE query for a node."""
        # Create parameter placeholders
        prop_placeholders = ", ".join([f"{key}: ${key}" for key in properties.keys()])
        
        query = f"CREATE (n:{node_type} {{{prop_placeholders}}}) RETURN n"
        
        return query, properties
    
    @staticmethod
    def create_relationship_query(from_node_type: str, from_id_field: str, from_id_value: str,
                                to_node_type: str, to_id_field: str, to_id_value: str,
                                relationship_type: str, 
                                properties: Optional[Dict[str, Any]] = None) -> tuple[str, Dict[str, Any]]:
        """Build CREATE query for a relationship."""
        
        params = {
            "from_id": from_id_value,
            "to_id": to_id_value
        }
        
        if properties:
            rel_props = ", ".join([f"{key}: ${key}" for key in properties.keys()])
            rel_clause = f"[r:{relationship_type} {{{rel_props}}}]"
            params.update(properties)
        else:
            rel_clause = f"[r:{relationship_type}]"
            
        query = f"""
        MATCH (from:{from_node_type} {{{from_id_field}: $from_id}})
        MATCH (to:{to_node_type} {{{to_id_field}: $to_id}})
        CREATE (from)-{rel_clause}->(to)
        RETURN r
        """
        
        return query.strip(), params
    
    @staticmethod
    def find_related_entities_query(node_type: str, id_field: str, 
                                   relationship_types: Optional[List[str]] = None,
                                   max_depth: int = 2) -> tuple[str, Dict[str, Any]]:
        """Build query to find related entities."""
        
        if relationship_types:
            rel_filter = "|".join(relationship_types)
            rel_clause = f"[r:{rel_filter}*1..{max_depth}]"
        else:
            rel_clause = f"[r*1..{max_depth}]"
            
        query = f"""
        MATCH (start:{node_type} {{{id_field}: $entity_id}})
        MATCH (start)-{rel_clause}-(related)
        RETURN DISTINCT related, r, 
               labels(related) as node_labels,
               type(r[-1]) as last_relationship_type
        LIMIT 50
        """
        
        return query.strip(), {"entity_id": None}  # Will be filled in by caller