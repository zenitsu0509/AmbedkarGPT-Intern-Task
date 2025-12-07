"""
Community Detector - Detect communities in the knowledge graph.

Uses Louvain or Leiden algorithm for community detection,
then groups related entities together.
"""

import pickle
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict

import networkx as nx
import numpy as np

try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False

try:
    import leidenalg
    import igraph as ig
    HAS_LEIDEN = True
except ImportError:
    HAS_LEIDEN = False


class CommunityDetector:
    """
    Detect communities in the knowledge graph using Louvain or Leiden algorithm.
    
    Communities represent thematically grouped entities that are
    closely related in the graph structure.
    """
    
    def __init__(
        self,
        algorithm: str = "louvain",
        resolution: float = 1.0,
        min_community_size: int = 3
    ):
        """
        Initialize the community detector.
        
        Args:
            algorithm: Algorithm to use ('louvain' or 'leiden')
            resolution: Resolution parameter (higher = more communities)
            min_community_size: Minimum size for a community
        """
        self.algorithm = algorithm
        self.resolution = resolution
        self.min_community_size = min_community_size
        
        # Validate algorithm availability
        if algorithm == "louvain" and not HAS_LOUVAIN:
            print("Warning: python-louvain not installed, will try leiden")
            self.algorithm = "leiden" if HAS_LEIDEN else None
        elif algorithm == "leiden" and not HAS_LEIDEN:
            print("Warning: leidenalg not installed, will try louvain")
            self.algorithm = "louvain" if HAS_LOUVAIN else None
        
        if self.algorithm is None:
            raise ImportError(
                "Neither python-louvain nor leidenalg is installed. "
                "Please install one: pip install python-louvain or pip install leidenalg igraph"
            )
    
    def detect_communities(self, graph: nx.Graph) -> Dict[str, int]:
        """
        Detect communities in the graph.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Dictionary mapping node IDs to community IDs
        """
        if graph.number_of_nodes() == 0:
            return {}
        
        print(f"Detecting communities using {self.algorithm} algorithm...")
        
        if self.algorithm == "louvain":
            partition = self._louvain_detection(graph)
        else:
            partition = self._leiden_detection(graph)
        
        # Count community sizes
        community_sizes = defaultdict(int)
        for node, comm_id in partition.items():
            community_sizes[comm_id] += 1
        
        # Filter small communities
        valid_communities = {
            comm_id for comm_id, size in community_sizes.items()
            if size >= self.min_community_size
        }
        
        # Create filtered partition
        filtered_partition = {}
        for node, comm_id in partition.items():
            if comm_id in valid_communities:
                filtered_partition[node] = comm_id
            else:
                # Assign to nearest valid community or create singleton
                filtered_partition[node] = -1  # Uncategorized
        
        num_communities = len(set(filtered_partition.values()) - {-1})
        print(f"Detected {num_communities} communities")
        
        return filtered_partition
    
    def _louvain_detection(self, graph: nx.Graph) -> Dict[str, int]:
        """Run Louvain community detection."""
        partition = community_louvain.best_partition(
            graph,
            weight="weight",
            resolution=self.resolution
        )
        return partition
    
    def _leiden_detection(self, graph: nx.Graph) -> Dict[str, int]:
        """Run Leiden community detection."""
        # Convert to igraph
        ig_graph = self._nx_to_igraph(graph)
        
        # Run Leiden
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight" if "weight" in ig_graph.es.attributes() else None,
            resolution_parameter=self.resolution
        )
        
        # Convert back to dictionary
        node_names = ig_graph.vs["name"]
        result = {}
        for comm_id, community in enumerate(partition):
            for node_idx in community:
                result[node_names[node_idx]] = comm_id
        
        return result
    
    def _nx_to_igraph(self, nx_graph: nx.Graph) -> 'ig.Graph':
        """Convert NetworkX graph to igraph."""
        # Get nodes and create mapping
        nodes = list(nx_graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        # Get edges
        edges = [(node_to_idx[u], node_to_idx[v]) for u, v in nx_graph.edges()]
        weights = [nx_graph[u][v].get("weight", 1) for u, v in nx_graph.edges()]
        
        # Create igraph
        ig_graph = ig.Graph(edges=edges, directed=False)
        ig_graph.vs["name"] = nodes
        ig_graph.es["weight"] = weights
        
        return ig_graph
    
    def get_community_members(
        self,
        partition: Dict[str, int]
    ) -> Dict[int, List[str]]:
        """
        Get members of each community.
        
        Args:
            partition: Node to community mapping
            
        Returns:
            Dictionary mapping community IDs to member lists
        """
        communities = defaultdict(list)
        for node, comm_id in partition.items():
            communities[comm_id].append(node)
        return dict(communities)
    
    def get_community_info(
        self,
        graph: nx.Graph,
        partition: Dict[str, int],
        chunks: List[Dict]
    ) -> List[Dict]:
        """
        Get detailed information about each community.
        
        Args:
            graph: Knowledge graph
            partition: Node to community mapping
            chunks: List of chunks
            
        Returns:
            List of community information dictionaries
        """
        community_members = self.get_community_members(partition)
        chunk_map = {c["id"]: c for c in chunks}
        
        community_info = []
        
        for comm_id, members in community_members.items():
            if comm_id == -1:  # Skip uncategorized
                continue
            
            # Get entity details
            entities = []
            all_chunk_ids = set()
            entity_labels = defaultdict(int)
            
            for member in members:
                if member in graph.nodes:
                    node_data = graph.nodes[member]
                    entities.append({
                        "name": node_data.get("text", member),
                        "normalized": member,
                        "label": node_data.get("label", "UNKNOWN"),
                        "frequency": node_data.get("frequency", 1),
                        "importance": node_data.get("importance", 0)
                    })
                    entity_labels[node_data.get("label", "UNKNOWN")] += 1
                    all_chunk_ids.update(node_data.get("chunk_ids", []))
            
            # Get representative chunks
            chunk_texts = []
            for chunk_id in list(all_chunk_ids)[:5]:  # Top 5 chunks
                if chunk_id in chunk_map:
                    chunk_texts.append(chunk_map[chunk_id]["text"])
            
            # Get subgraph for community
            subgraph = graph.subgraph(members)
            
            info = {
                "community_id": comm_id,
                "size": len(members),
                "entities": entities,
                "entity_labels": dict(entity_labels),
                "chunk_ids": list(all_chunk_ids),
                "sample_chunks": chunk_texts,
                "internal_edges": subgraph.number_of_edges(),
                "density": nx.density(subgraph) if len(members) > 1 else 0
            }
            
            community_info.append(info)
        
        # Sort by size
        community_info.sort(key=lambda x: x["size"], reverse=True)
        
        return community_info
    
    def compute_community_embeddings(
        self,
        graph: nx.Graph,
        partition: Dict[str, int]
    ) -> Dict[int, np.ndarray]:
        """
        Compute embeddings for each community by averaging member embeddings.
        
        Args:
            graph: Knowledge graph with node embeddings
            partition: Node to community mapping
            
        Returns:
            Dictionary mapping community IDs to embeddings
        """
        community_members = self.get_community_members(partition)
        community_embeddings = {}
        
        for comm_id, members in community_members.items():
            if comm_id == -1:
                continue
            
            # Collect embeddings of members
            embeddings = []
            for member in members:
                if member in graph.nodes:
                    emb = graph.nodes[member].get("embedding")
                    if emb is not None:
                        embeddings.append(np.array(emb))
            
            if embeddings:
                # Average embedding
                community_embeddings[comm_id] = np.mean(embeddings, axis=0)
        
        return community_embeddings
    
    def save_communities(
        self,
        partition: Dict[str, int],
        community_info: List[Dict],
        filepath: str
    ):
        """Save community data to a pickle file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "partition": partition,
            "community_info": community_info
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        
        print(f"Communities saved to {filepath}")
    
    def load_communities(self, filepath: str) -> Tuple[Dict[str, int], List[Dict]]:
        """Load community data from a pickle file."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        return data["partition"], data["community_info"]
