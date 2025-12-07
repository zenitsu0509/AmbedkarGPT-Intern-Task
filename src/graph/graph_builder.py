"""
Graph Builder - Construct knowledge graph from entities and relationships.

Builds a networkx graph with entities as nodes and relationships as edges.
"""

import pickle
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict

import networkx as nx
import numpy as np


class GraphBuilder:
    """
    Build a knowledge graph from extracted entities and relationships.
    
    Creates a networkx graph where:
    - Nodes = Entities (with attributes: label, frequency, chunk_ids, embedding)
    - Edges = Relationships (with attributes: type, predicate, chunk_ids)
    """
    
    def __init__(self):
        """Initialize the graph builder."""
        self.graph = nx.Graph()  # Undirected graph for community detection
        self.directed_graph = nx.DiGraph()  # Directed graph for relationships
    
    def build_graph(
        self,
        entities: List[Dict],
        relationships: List[Dict],
        chunks: List[Dict],
        entity_embeddings: Optional[Dict[str, np.ndarray]] = None
    ) -> nx.Graph:
        """
        Build the knowledge graph from entities and relationships.
        
        Args:
            entities: List of entity dictionaries
            relationships: List of relationship dictionaries
            chunks: List of chunk dictionaries
            entity_embeddings: Optional mapping of entity text to embeddings
            
        Returns:
            NetworkX graph
        """
        print("Building knowledge graph...")
        
        # Create chunk lookup
        chunk_map = {c["id"]: c for c in chunks}
        
        # Add entity nodes
        for entity in entities:
            node_id = entity["normalized"]
            
            # Get embedding if available
            embedding = None
            if entity_embeddings and node_id in entity_embeddings:
                embedding = entity_embeddings[node_id].tolist()
            
            self.graph.add_node(
                node_id,
                text=entity["text"],
                label=entity["label"],
                frequency=entity.get("frequency", 1),
                chunk_ids=entity.get("chunk_ids", []),
                embedding=embedding
            )
            
            self.directed_graph.add_node(
                node_id,
                text=entity["text"],
                label=entity["label"],
                frequency=entity.get("frequency", 1),
                chunk_ids=entity.get("chunk_ids", []),
                embedding=embedding
            )
        
        # Add relationship edges
        edge_weights = defaultdict(int)
        edge_predicates = defaultdict(list)
        edge_chunks = defaultdict(set)
        
        for rel in relationships:
            subject = rel["subject"].lower().strip()
            obj = rel["object"].lower().strip()
            predicate = rel.get("predicate", "RELATED_TO")
            chunk_id = rel.get("chunk_id")
            
            # Only add edges between existing nodes
            if subject in self.graph.nodes and obj in self.graph.nodes:
                edge_key = (subject, obj) if subject < obj else (obj, subject)
                edge_weights[edge_key] += 1
                edge_predicates[edge_key].append(predicate)
                if chunk_id:
                    edge_chunks[edge_key].add(chunk_id)
                
                # Add to directed graph
                self.directed_graph.add_edge(
                    subject, obj,
                    predicate=predicate,
                    chunk_id=chunk_id
                )
        
        # Add weighted edges to undirected graph
        for (source, target), weight in edge_weights.items():
            predicates = list(set(edge_predicates[(source, target)]))
            chunk_ids = list(edge_chunks[(source, target)])
            
            self.graph.add_edge(
                source, target,
                weight=weight,
                predicates=predicates,
                chunk_ids=chunk_ids
            )
        
        print(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def add_chunk_connections(self, chunks: List[Dict]):
        """
        Add connections between entities that appear in the same chunk.
        
        This ensures entities mentioned together are connected in the graph.
        """
        chunk_entities = defaultdict(list)
        
        # Group entities by chunk
        for node_id, data in self.graph.nodes(data=True):
            for chunk_id in data.get("chunk_ids", []):
                chunk_entities[chunk_id].append(node_id)
        
        # Connect entities within same chunk
        for chunk_id, entities in chunk_entities.items():
            for i, ent1 in enumerate(entities):
                for ent2 in entities[i+1:]:
                    if not self.graph.has_edge(ent1, ent2):
                        self.graph.add_edge(
                            ent1, ent2,
                            weight=1,
                            predicates=["CO_OCCURS_IN_CHUNK"],
                            chunk_ids=[chunk_id]
                        )
                    else:
                        # Increase weight for existing edge
                        self.graph[ent1][ent2]["weight"] += 1
                        if chunk_id not in self.graph[ent1][ent2]["chunk_ids"]:
                            self.graph[ent1][ent2]["chunk_ids"].append(chunk_id)
    
    def compute_node_importance(self) -> Dict[str, float]:
        """
        Compute importance scores for nodes using PageRank.
        
        Returns:
            Dictionary mapping node IDs to importance scores
        """
        if self.graph.number_of_nodes() == 0:
            return {}
        
        # Use PageRank for importance
        pagerank = nx.pagerank(self.graph, weight="weight")
        
        # Normalize scores
        max_score = max(pagerank.values()) if pagerank else 1
        normalized = {node: score / max_score for node, score in pagerank.items()}
        
        # Add importance to node attributes
        for node, importance in normalized.items():
            self.graph.nodes[node]["importance"] = importance
        
        return normalized
    
    def get_entity_neighbors(
        self,
        entity: str,
        max_depth: int = 2
    ) -> Dict[str, List[str]]:
        """
        Get neighbors of an entity up to a certain depth.
        
        Args:
            entity: Entity normalized text
            max_depth: Maximum depth to traverse
            
        Returns:
            Dictionary mapping depth to list of neighbor entities
        """
        entity = entity.lower().strip()
        if entity not in self.graph:
            return {}
        
        neighbors_by_depth = {}
        visited = {entity}
        current_level = [entity]
        
        for depth in range(1, max_depth + 1):
            next_level = []
            for node in current_level:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in visited:
                        next_level.append(neighbor)
                        visited.add(neighbor)
            
            if next_level:
                neighbors_by_depth[depth] = next_level
            current_level = next_level
        
        return neighbors_by_depth
    
    def get_subgraph_for_entities(
        self,
        entities: List[str],
        include_neighbors: bool = True
    ) -> nx.Graph:
        """
        Extract a subgraph containing specified entities.
        
        Args:
            entities: List of entity texts
            include_neighbors: Whether to include direct neighbors
            
        Returns:
            Subgraph containing the entities
        """
        normalized = [e.lower().strip() for e in entities]
        nodes_to_include = set()
        
        for entity in normalized:
            if entity in self.graph:
                nodes_to_include.add(entity)
                if include_neighbors:
                    nodes_to_include.update(self.graph.neighbors(entity))
        
        return self.graph.subgraph(nodes_to_include).copy()
    
    def get_paths_between_entities(
        self,
        source: str,
        target: str,
        max_paths: int = 3
    ) -> List[List[str]]:
        """
        Find shortest paths between two entities.
        
        Args:
            source: Source entity
            target: Target entity
            max_paths: Maximum number of paths to return
            
        Returns:
            List of paths (each path is a list of entity names)
        """
        source = source.lower().strip()
        target = target.lower().strip()
        
        if source not in self.graph or target not in self.graph:
            return []
        
        try:
            # Find all simple paths up to length 5
            paths = list(nx.all_simple_paths(
                self.graph, source, target, cutoff=5
            ))
            
            # Sort by length and return top max_paths
            paths.sort(key=len)
            return paths[:max_paths]
        except nx.NetworkXNoPath:
            return []
    
    def save_graph(self, filepath: str):
        """Save the graph to a pickle file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "graph": self.graph,
            "directed_graph": self.directed_graph
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        
        print(f"Graph saved to {filepath}")
    
    def load_graph(self, filepath: str):
        """Load the graph from a pickle file."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        self.graph = data["graph"]
        self.directed_graph = data.get("directed_graph", nx.DiGraph())
        
        print(f"Graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def get_graph_statistics(self) -> Dict:
        """Get statistics about the graph."""
        stats = {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph) if self.graph.number_of_nodes() > 1 else 0,
            "num_connected_components": nx.number_connected_components(self.graph),
        }
        
        if self.graph.number_of_nodes() > 0:
            degrees = dict(self.graph.degree())
            stats["avg_degree"] = sum(degrees.values()) / len(degrees)
            stats["max_degree"] = max(degrees.values())
            stats["min_degree"] = min(degrees.values())
        
        return stats
