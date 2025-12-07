"""
Global Graph RAG Search - Implementation of SEMRAG Equation 5.

Retrieves relevant community summaries and extracts top chunks.
D_retrieved = Top_k(⋃_{r ∈ R_Top-K(Q)} ⋃_{c_i ∈ C_r} (⋃_{p_j ∈ c_i} (p_j, score(p_j, Q))), score(p_j, Q))
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class GlobalGraphRAGSearch:
    """
    Implements Global Graph RAG Search (Equation 5 from SEMRAG).
    
    Steps:
    1. Find top-K community reports relevant to query
    2. Extract chunks from those communities
    3. Score each chunk within communities
    4. Return top-K chunks based on scores
    """
    
    def __init__(
        self,
        top_k_communities: int = 3,
        top_k_chunks: int = 5
    ):
        """
        Initialize global search.
        
        Args:
            top_k_communities: Number of top communities to retrieve
            top_k_chunks: Number of top chunks to return
        """
        self.top_k_communities = top_k_communities
        self.top_k_chunks = top_k_chunks
    
    def search(
        self,
        query_embedding: np.ndarray,
        community_info: List[Dict],
        community_embeddings: Dict[int, np.ndarray],
        chunk_embeddings: Dict[str, np.ndarray],
        chunks: List[Dict],
        top_k_communities: Optional[int] = None,
        top_k_chunks: Optional[int] = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Perform global graph RAG search.
        
        Args:
            query_embedding: Embedding of the query
            community_info: List of community information dictionaries
            community_embeddings: Dict mapping community IDs to embeddings
            chunk_embeddings: Dict mapping chunk IDs to embeddings
            chunks: List of chunk dictionaries
            top_k_communities: Override default top_k_communities
            top_k_chunks: Override default top_k_chunks
            
        Returns:
            Tuple of (community_results, chunk_results)
        """
        k_comm = top_k_communities or self.top_k_communities
        k_chunks = top_k_chunks or self.top_k_chunks
        
        query_embedding = query_embedding.reshape(1, -1)
        
        # Step 1: Find top-K relevant communities
        relevant_communities = self._find_relevant_communities(
            query_embedding, community_info, community_embeddings, k_comm
        )
        
        if not relevant_communities:
            return [], []
        
        # Step 2: Extract and score chunks from relevant communities
        scored_chunks = self._score_community_chunks(
            query_embedding, relevant_communities, chunk_embeddings, chunks
        )
        
        # Step 3: Return top-K chunks
        return relevant_communities, scored_chunks[:k_chunks]
    
    def _find_relevant_communities(
        self,
        query_embedding: np.ndarray,
        community_info: List[Dict],
        community_embeddings: Dict[int, np.ndarray],
        top_k: int
    ) -> List[Dict]:
        """Find communities most relevant to the query."""
        scored_communities = []
        
        for comm in community_info:
            comm_id = comm.get("community_id")
            
            # Try community embedding first
            if comm_id in community_embeddings:
                emb = np.array(community_embeddings[comm_id]).reshape(1, -1)
                similarity = cosine_similarity(query_embedding, emb)[0][0]
            else:
                # Fallback: use summary text similarity if embeddings not available
                similarity = 0.0
            
            comm_result = comm.copy()
            comm_result["relevance_score"] = float(similarity)
            scored_communities.append(comm_result)
        
        # Sort by relevance
        scored_communities.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored_communities[:top_k]
    
    def _score_community_chunks(
        self,
        query_embedding: np.ndarray,
        communities: List[Dict],
        chunk_embeddings: Dict[str, np.ndarray],
        chunks: List[Dict]
    ) -> List[Dict]:
        """Score and rank chunks from relevant communities."""
        chunk_map = {c["id"]: c for c in chunks}
        scored_chunks = []
        seen_chunks = set()
        
        for comm in communities:
            comm_score = comm.get("relevance_score", 0)
            chunk_ids = comm.get("chunk_ids", [])
            
            for chunk_id in chunk_ids:
                if chunk_id in seen_chunks:
                    continue
                seen_chunks.add(chunk_id)
                
                if chunk_id not in chunk_map:
                    continue
                
                # Score chunk
                if chunk_id in chunk_embeddings:
                    chunk_emb = np.array(chunk_embeddings[chunk_id]).reshape(1, -1)
                    chunk_similarity = cosine_similarity(query_embedding, chunk_emb)[0][0]
                else:
                    chunk_similarity = 0.0
                
                chunk = chunk_map[chunk_id].copy()
                chunk["community_id"] = comm.get("community_id")
                chunk["community_score"] = float(comm_score)
                chunk["chunk_similarity"] = float(chunk_similarity)
                # Combined score: weighted combination
                chunk["global_score"] = 0.4 * comm_score + 0.6 * chunk_similarity
                
                scored_chunks.append(chunk)
        
        # Sort by global score
        scored_chunks.sort(key=lambda x: x["global_score"], reverse=True)
        return scored_chunks
    
    def get_community_summaries(
        self,
        query_embedding: np.ndarray,
        community_info: List[Dict],
        community_embeddings: Dict[int, np.ndarray],
        top_k: int = 3
    ) -> List[Dict]:
        """
        Get relevant community summaries for the query.
        
        Returns community summaries that can be used as context.
        """
        query_embedding = query_embedding.reshape(1, -1)
        
        relevant = self._find_relevant_communities(
            query_embedding, community_info, community_embeddings, top_k
        )
        
        summaries = []
        for comm in relevant:
            summary_info = {
                "community_id": comm.get("community_id"),
                "summary": comm.get("summary", ""),
                "size": comm.get("size", 0),
                "relevance_score": comm.get("relevance_score", 0),
                "key_entities": [e["name"] for e in comm.get("entities", [])[:5]]
            }
            summaries.append(summary_info)
        
        return summaries
    
    def search_by_entity_community(
        self,
        entity: str,
        partition: Dict[str, int],
        community_info: List[Dict],
        chunks: List[Dict]
    ) -> Tuple[Optional[Dict], List[Dict]]:
        """
        Search for chunks related to an entity's community.
        
        Args:
            entity: Entity name to search for
            partition: Entity to community mapping
            community_info: List of community information
            chunks: List of chunks
            
        Returns:
            Tuple of (community_info, related_chunks)
        """
        entity_normalized = entity.lower().strip()
        
        if entity_normalized not in partition:
            return None, []
        
        comm_id = partition[entity_normalized]
        
        # Find community info
        comm = None
        for c in community_info:
            if c.get("community_id") == comm_id:
                comm = c
                break
        
        if comm is None:
            return None, []
        
        # Get chunks
        chunk_map = {c["id"]: c for c in chunks}
        chunk_ids = comm.get("chunk_ids", [])
        related_chunks = [chunk_map[cid] for cid in chunk_ids if cid in chunk_map]
        
        return comm, related_chunks
