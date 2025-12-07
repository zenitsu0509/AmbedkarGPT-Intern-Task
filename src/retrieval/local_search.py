"""
Local Graph RAG Search - Implementation of SEMRAG Equation 4.

Retrieves relevant entities and chunks based on query similarity.
D_retrieved = Top_k({v ∈ V, g ∈ G | sim(v, Q+H) > τ_e ∧ sim(g, v) > τ_d})
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class LocalGraphRAGSearch:
    """
    Implements Local Graph RAG Search (Equation 4 from SEMRAG).
    
    Steps:
    1. Calculate similarity between query and entities
    2. Filter by threshold τ_e (entity_similarity_threshold)
    3. Find chunks related to those entities
    4. Filter by threshold τ_d (chunk_similarity_threshold)
    5. Return top_k results
    """
    
    def __init__(
        self,
        entity_similarity_threshold: float = 0.3,
        chunk_similarity_threshold: float = 0.2,
        top_k: int = 5
    ):
        """
        Initialize local search.
        
        Args:
            entity_similarity_threshold: τ_e - threshold for entity similarity
            chunk_similarity_threshold: τ_d - threshold for chunk similarity
            top_k: Number of top results to return
        """
        self.entity_threshold = entity_similarity_threshold
        self.chunk_threshold = chunk_similarity_threshold
        self.top_k = top_k
    
    def search(
        self,
        query_embedding: np.ndarray,
        entity_embeddings: Dict[str, np.ndarray],
        chunk_embeddings: Dict[str, np.ndarray],
        entity_to_chunks: Dict[str, List[str]],
        chunks: List[Dict],
        graph=None,
        history_embedding: Optional[np.ndarray] = None,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Perform local graph RAG search.
        
        Args:
            query_embedding: Embedding of the query
            entity_embeddings: Dict mapping entity names to embeddings
            chunk_embeddings: Dict mapping chunk IDs to embeddings
            entity_to_chunks: Dict mapping entities to chunk IDs
            chunks: List of chunk dictionaries
            graph: Optional knowledge graph for relationship traversal
            history_embedding: Optional embedding of conversation history
            top_k: Override default top_k
            
        Returns:
            List of retrieved chunk dictionaries with scores
        """
        k = top_k or self.top_k
        
        # Combine query with history if provided (Q + H)
        if history_embedding is not None:
            combined_query = (query_embedding + history_embedding) / 2
        else:
            combined_query = query_embedding
        
        combined_query = combined_query.reshape(1, -1)
        
        # Step 1: Find entities similar to query
        relevant_entities = self._find_relevant_entities(
            combined_query, entity_embeddings
        )
        
        if not relevant_entities:
            # Fallback: direct chunk search
            return self._direct_chunk_search(combined_query, chunk_embeddings, chunks, k)
        
        # Step 2: Get chunks related to relevant entities
        candidate_chunks = self._get_entity_related_chunks(
            relevant_entities, entity_to_chunks
        )
        
        # Step 3: Expand with graph neighbors if available
        if graph is not None:
            candidate_chunks = self._expand_with_neighbors(
                candidate_chunks, relevant_entities, entity_to_chunks, graph
            )
        
        # Step 4: Score and filter chunks
        scored_chunks = self._score_chunks(
            combined_query, candidate_chunks, chunk_embeddings, chunks
        )
        
        # Step 5: Return top-k
        return scored_chunks[:k]
    
    def _find_relevant_entities(
        self,
        query_embedding: np.ndarray,
        entity_embeddings: Dict[str, np.ndarray]
    ) -> List[Tuple[str, float]]:
        """Find entities with similarity above threshold τ_e."""
        relevant = []
        
        for entity, embedding in entity_embeddings.items():
            embedding = np.array(embedding).reshape(1, -1)
            similarity = cosine_similarity(query_embedding, embedding)[0][0]
            
            if similarity > self.entity_threshold:
                relevant.append((entity, similarity))
        
        # Sort by similarity
        relevant.sort(key=lambda x: x[1], reverse=True)
        return relevant
    
    def _get_entity_related_chunks(
        self,
        entities: List[Tuple[str, float]],
        entity_to_chunks: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """Get chunks related to the relevant entities."""
        chunk_scores = {}
        
        for entity, entity_score in entities:
            chunk_ids = entity_to_chunks.get(entity, [])
            for chunk_id in chunk_ids:
                # Aggregate entity scores for each chunk
                if chunk_id not in chunk_scores:
                    chunk_scores[chunk_id] = entity_score
                else:
                    # Take max or combine scores
                    chunk_scores[chunk_id] = max(chunk_scores[chunk_id], entity_score)
        
        return chunk_scores
    
    def _expand_with_neighbors(
        self,
        candidate_chunks: Dict[str, float],
        entities: List[Tuple[str, float]],
        entity_to_chunks: Dict[str, List[str]],
        graph
    ) -> Dict[str, float]:
        """Expand candidates with chunks from neighboring entities."""
        expanded = candidate_chunks.copy()
        
        for entity, entity_score in entities[:10]:  # Top 10 entities
            if entity in graph:
                for neighbor in graph.neighbors(entity):
                    neighbor_chunks = entity_to_chunks.get(neighbor, [])
                    for chunk_id in neighbor_chunks:
                        if chunk_id not in expanded:
                            # Discount score for neighbor-derived chunks
                            expanded[chunk_id] = entity_score * 0.7
        
        return expanded
    
    def _score_chunks(
        self,
        query_embedding: np.ndarray,
        candidate_chunks: Dict[str, float],
        chunk_embeddings: Dict[str, np.ndarray],
        chunks: List[Dict]
    ) -> List[Dict]:
        """Score candidate chunks by direct similarity."""
        chunk_map = {c["id"]: c for c in chunks}
        scored = []
        
        for chunk_id, entity_score in candidate_chunks.items():
            if chunk_id not in chunk_embeddings or chunk_id not in chunk_map:
                continue
            
            chunk_emb = np.array(chunk_embeddings[chunk_id]).reshape(1, -1)
            direct_similarity = cosine_similarity(query_embedding, chunk_emb)[0][0]
            
            # Filter by chunk similarity threshold τ_d
            if direct_similarity > self.chunk_threshold:
                chunk = chunk_map[chunk_id].copy()
                
                # Combined score: entity relevance + direct similarity
                chunk["entity_score"] = entity_score
                chunk["similarity_score"] = float(direct_similarity)
                chunk["combined_score"] = (entity_score + direct_similarity) / 2
                
                scored.append(chunk)
        
        # Sort by combined score
        scored.sort(key=lambda x: x["combined_score"], reverse=True)
        return scored
    
    def _direct_chunk_search(
        self,
        query_embedding: np.ndarray,
        chunk_embeddings: Dict[str, np.ndarray],
        chunks: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """Fallback: Direct similarity search over chunks."""
        chunk_map = {c["id"]: c for c in chunks}
        scored = []
        
        for chunk_id, embedding in chunk_embeddings.items():
            if chunk_id not in chunk_map:
                continue
            
            emb = np.array(embedding).reshape(1, -1)
            similarity = cosine_similarity(query_embedding, emb)[0][0]
            
            if similarity > self.chunk_threshold:
                chunk = chunk_map[chunk_id].copy()
                chunk["similarity_score"] = float(similarity)
                chunk["combined_score"] = float(similarity)
                scored.append(chunk)
        
        scored.sort(key=lambda x: x["combined_score"], reverse=True)
        return scored[:top_k]
    
    def search_with_entities(
        self,
        query_embedding: np.ndarray,
        entity_embeddings: Dict[str, np.ndarray],
        graph,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search for relevant entities only (without chunk retrieval).
        
        Returns entities with their graph context.
        """
        query_embedding = query_embedding.reshape(1, -1)
        relevant_entities = self._find_relevant_entities(query_embedding, entity_embeddings)
        
        results = []
        for entity, score in relevant_entities[:top_k]:
            entity_info = {
                "entity": entity,
                "score": float(score),
                "neighbors": []
            }
            
            if entity in graph:
                # Get neighbors
                for neighbor in list(graph.neighbors(entity))[:5]:
                    edge_data = graph[entity][neighbor]
                    entity_info["neighbors"].append({
                        "entity": neighbor,
                        "relation": edge_data.get("predicates", ["RELATED_TO"])[0]
                    })
            
            results.append(entity_info)
        
        return results
