"""
AmbedkarGPT - Main RAG Pipeline

Complete SEMRAG-based RAG system for Dr. B.R. Ambedkar's works.
Integrates semantic chunking, knowledge graph, retrieval, and LLM.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

# Import our modules (absolute imports)
from chunking import SemanticChunker, BufferMerger
from graph import EntityExtractor, GraphBuilder, CommunityDetector, CommunitySummarizer
from retrieval import LocalGraphRAGSearch, GlobalGraphRAGSearch, ResultRanker
from llm import LLMClient, PromptTemplates, AnswerGenerator


class AmbedkarGPT:
    """
    Main RAG pipeline for answering questions about Dr. B.R. Ambedkar's works.
    
    Implements the complete SEMRAG architecture:
    1. Semantic chunking with buffer merging
    2. Knowledge graph construction with community detection
    3. Local and global graph RAG search
    4. LLM-based answer generation
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize AmbedkarGPT.
        
        Args:
            config_path: Path to config.yaml file
        """
        # Load configuration
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = Path(__file__).parent.parent.parent / "config.yaml"
        
        self.config = self._load_config()
        self.base_path = self.config_path.parent
        
        # Initialize components (lazy loading)
        self._embedding_model = None
        self._llm_client = None
        self._chunker = None
        self._entity_extractor = None
        self._graph_builder = None
        self._community_detector = None
        self._summarizer = None
        self._local_search = None
        self._global_search = None
        self._ranker = None
        self._answer_generator = None
        
        # Data storage
        self.chunks = []
        self.sub_chunks = []
        self.entities = []
        self.relationships = []
        self.entity_to_chunks = {}
        self.graph = None
        self.partition = {}
        self.community_info = []
        
        # Embeddings cache
        self.chunk_embeddings = {}
        self.entity_embeddings = {}
        self.community_embeddings = {}
        
        # Conversation history
        self.history = []
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {}
    
    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy load embedding model."""
        if self._embedding_model is None:
            model_name = self.config.get("chunking", {}).get(
                "embedding_model", "all-MiniLM-L6-v2"
            )
            print(f"Loading embedding model: {model_name}")
            self._embedding_model = SentenceTransformer(model_name)
        return self._embedding_model
    
    @property
    def llm_client(self) -> LLMClient:
        """Lazy load LLM client."""
        if self._llm_client is None:
            llm_config = self.config.get("llm", {})
            self._llm_client = LLMClient(
                model=llm_config.get("model", "mistral:7b"),
                base_url=llm_config.get("base_url", "http://localhost:11434"),
                temperature=llm_config.get("temperature", 0.7),
                max_tokens=llm_config.get("max_tokens", 2048)
            )
        return self._llm_client
    
    @property
    def local_search(self) -> LocalGraphRAGSearch:
        """Lazy load local search."""
        if self._local_search is None:
            retrieval_config = self.config.get("retrieval", {}).get("local", {})
            self._local_search = LocalGraphRAGSearch(
                entity_similarity_threshold=retrieval_config.get("entity_similarity_threshold", 0.3),
                chunk_similarity_threshold=retrieval_config.get("chunk_similarity_threshold", 0.2),
                top_k=retrieval_config.get("top_k", 5)
            )
        return self._local_search
    
    @property
    def global_search(self) -> GlobalGraphRAGSearch:
        """Lazy load global search."""
        if self._global_search is None:
            retrieval_config = self.config.get("retrieval", {}).get("global", {})
            self._global_search = GlobalGraphRAGSearch(
                top_k_communities=retrieval_config.get("top_k_communities", 3),
                top_k_chunks=retrieval_config.get("top_k_chunks", 5)
            )
        return self._global_search
    
    @property
    def ranker(self) -> ResultRanker:
        """Lazy load result ranker."""
        if self._ranker is None:
            self._ranker = ResultRanker()
        return self._ranker
    
    @property
    def answer_generator(self) -> AnswerGenerator:
        """Lazy load answer generator."""
        if self._answer_generator is None:
            self._answer_generator = AnswerGenerator(self.llm_client)
        return self._answer_generator
    
    # ==================== INGESTION PIPELINE ====================
    
    def ingest_document(self, pdf_path: str, save_processed: bool = True):
        """
        Full ingestion pipeline: PDF -> chunks -> graph -> communities.
        
        Args:
            pdf_path: Path to PDF file
            save_processed: Whether to save processed data
        """
        print("=" * 60)
        print("STARTING DOCUMENT INGESTION")
        print("=" * 60)
        
        # Step 1: Semantic Chunking
        print("\n[Step 1/4] Semantic Chunking...")
        self._run_chunking(pdf_path)
        
        # Step 2: Entity Extraction
        print("\n[Step 2/4] Entity Extraction...")
        self._run_entity_extraction()
        
        # Step 3: Knowledge Graph Construction
        print("\n[Step 3/4] Building Knowledge Graph...")
        self._build_knowledge_graph()
        
        # Step 4: Community Detection
        print("\n[Step 4/4] Detecting Communities...")
        self._detect_communities()
        
        # Prepare embeddings for retrieval
        print("\n[Finalizing] Preparing embeddings...")
        self._prepare_embeddings()
        
        # Save processed data
        if save_processed:
            self._save_processed_data()
        
        print("\n" + "=" * 60)
        print("INGESTION COMPLETE!")
        print(f"  Chunks: {len(self.chunks)}")
        print(f"  Sub-chunks: {len(self.sub_chunks)}")
        print(f"  Entities: {len(self.entities)}")
        print(f"  Graph nodes: {self.graph.number_of_nodes() if self.graph else 0}")
        print(f"  Communities: {len(self.community_info)}")
        print("=" * 60)
    
    def _run_chunking(self, pdf_path: str):
        """Run semantic chunking on the document."""
        chunking_config = self.config.get("chunking", {})
        
        chunker = SemanticChunker(
            embedding_model=chunking_config.get("embedding_model", "all-MiniLM-L6-v2"),
            max_tokens=chunking_config.get("max_tokens", 1024),
            sub_chunk_tokens=chunking_config.get("sub_chunk_tokens", 128),
            overlap_tokens=chunking_config.get("overlap_tokens", 32),
            similarity_threshold=chunking_config.get("similarity_threshold", 0.5),
            buffer_size=chunking_config.get("buffer_size", 3)
        )
        
        self.chunks, self.sub_chunks = chunker.process_document(pdf_path)
        
        # Use sub_chunks as primary retrieval units
        print(f"  Created {len(self.chunks)} semantic chunks")
        print(f"  Created {len(self.sub_chunks)} sub-chunks for retrieval")
    
    def _run_entity_extraction(self):
        """Extract entities and relationships from chunks."""
        graph_config = self.config.get("graph", {})
        
        extractor = EntityExtractor(
            model_name=graph_config.get("spacy_model", "en_core_web_sm")
        )
        
        self.entities, self.relationships, self.entity_to_chunks = extractor.process_chunks(
            self.sub_chunks,
            min_entity_freq=graph_config.get("min_entity_freq", 2)
        )
        
        print(f"  Extracted {len(self.entities)} unique entities")
        print(f"  Extracted {len(self.relationships)} relationships")
    
    def _build_knowledge_graph(self):
        """Build the knowledge graph from entities and relationships."""
        builder = GraphBuilder()
        
        # Generate entity embeddings
        entity_texts = [e["text"] for e in self.entities]
        if entity_texts:
            embeddings = self.embedding_model.encode(entity_texts)
            entity_emb_dict = {
                self.entities[i]["normalized"]: embeddings[i]
                for i in range(len(self.entities))
            }
        else:
            entity_emb_dict = {}
        
        self.graph = builder.build_graph(
            self.entities,
            self.relationships,
            self.sub_chunks,
            entity_embeddings=entity_emb_dict
        )
        
        # Add chunk co-occurrence connections
        builder.add_chunk_connections(self.sub_chunks)
        
        # Compute node importance
        builder.compute_node_importance()
        
        self._graph_builder = builder
        
        stats = builder.get_graph_statistics()
        print(f"  Graph statistics: {stats}")
    
    def _detect_communities(self):
        """Detect communities in the knowledge graph."""
        comm_config = self.config.get("community", {})
        
        detector = CommunityDetector(
            algorithm=comm_config.get("algorithm", "louvain"),
            resolution=comm_config.get("resolution", 1.0),
            min_community_size=comm_config.get("min_community_size", 3)
        )
        
        self.partition = detector.detect_communities(self.graph)
        self.community_info = detector.get_community_info(
            self.graph, self.partition, self.sub_chunks
        )
        
        # Generate community summaries
        summarizer = CommunitySummarizer(self.llm_client)
        self.community_info = summarizer.summarize_all_communities(self.community_info)
        
        # Compute community embeddings
        self.community_embeddings = detector.compute_community_embeddings(
            self.graph, self.partition
        )
        
        print(f"  Detected {len(self.community_info)} communities")
    
    def _prepare_embeddings(self):
        """Prepare embeddings for retrieval."""
        # Chunk embeddings (from sub_chunks)
        for chunk in self.sub_chunks:
            if "embedding" in chunk:
                self.chunk_embeddings[chunk["id"]] = np.array(chunk["embedding"])
        
        # Entity embeddings (from graph nodes)
        for node, data in self.graph.nodes(data=True):
            if data.get("embedding") is not None:
                self.entity_embeddings[node] = np.array(data["embedding"])
        
        # Community embeddings (already computed)
        # Convert to numpy arrays
        for comm_id, emb in self.community_embeddings.items():
            if isinstance(emb, list):
                self.community_embeddings[comm_id] = np.array(emb)
        
        print(f"  Prepared {len(self.chunk_embeddings)} chunk embeddings")
        print(f"  Prepared {len(self.entity_embeddings)} entity embeddings")
        print(f"  Prepared {len(self.community_embeddings)} community embeddings")
    
    def _save_processed_data(self):
        """Save all processed data to files."""
        processed_dir = self.base_path / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Save chunks
        chunks_path = processed_dir / "chunks.json"
        with open(chunks_path, "w", encoding="utf-8") as f:
            # Remove embeddings for JSON serialization
            chunks_to_save = [{k: v for k, v in c.items() if k != "embedding"} 
                             for c in self.chunks]
            json.dump(chunks_to_save, f, indent=2, ensure_ascii=False)
        
        # Save sub_chunks
        sub_chunks_path = processed_dir / "sub_chunks.json"
        with open(sub_chunks_path, "w", encoding="utf-8") as f:
            sub_chunks_to_save = [{k: v for k, v in c.items() if k != "embedding"} 
                                  for c in self.sub_chunks]
            json.dump(sub_chunks_to_save, f, indent=2, ensure_ascii=False)
        
        # Save knowledge graph
        graph_path = processed_dir / "knowledge_graph.pkl"
        self._graph_builder.save_graph(str(graph_path))
        
        # Save embeddings
        embeddings_path = processed_dir / "embeddings.pkl"
        with open(embeddings_path, "wb") as f:
            pickle.dump({
                "chunk_embeddings": {k: v.tolist() for k, v in self.chunk_embeddings.items()},
                "entity_embeddings": {k: v.tolist() for k, v in self.entity_embeddings.items()},
                "community_embeddings": {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                        for k, v in self.community_embeddings.items()}
            }, f)
        
        # Save community info
        community_path = processed_dir / "communities.pkl"
        with open(community_path, "wb") as f:
            pickle.dump({
                "partition": self.partition,
                "community_info": self.community_info
            }, f)
        
        print(f"\n  Saved processed data to {processed_dir}")
    
    def load_processed_data(self, processed_dir: Optional[str] = None):
        """
        Load previously processed data.
        
        Args:
            processed_dir: Path to processed data directory
        """
        if processed_dir:
            processed_dir = Path(processed_dir)
        else:
            processed_dir = self.base_path / "data" / "processed"
        
        print(f"Loading processed data from {processed_dir}...")
        
        # Load chunks
        chunks_path = processed_dir / "chunks.json"
        if chunks_path.exists():
            with open(chunks_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
        
        # Load sub_chunks
        sub_chunks_path = processed_dir / "sub_chunks.json"
        if sub_chunks_path.exists():
            with open(sub_chunks_path, "r", encoding="utf-8") as f:
                self.sub_chunks = json.load(f)
        
        # Load knowledge graph
        graph_path = processed_dir / "knowledge_graph.pkl"
        if graph_path.exists():
            builder = GraphBuilder()
            builder.load_graph(str(graph_path))
            self.graph = builder.graph
            self._graph_builder = builder
        
        # Load embeddings
        embeddings_path = processed_dir / "embeddings.pkl"
        if embeddings_path.exists():
            with open(embeddings_path, "rb") as f:
                emb_data = pickle.load(f)
                self.chunk_embeddings = {k: np.array(v) for k, v in emb_data.get("chunk_embeddings", {}).items()}
                self.entity_embeddings = {k: np.array(v) for k, v in emb_data.get("entity_embeddings", {}).items()}
                self.community_embeddings = {k: np.array(v) for k, v in emb_data.get("community_embeddings", {}).items()}
        
        # Load communities
        community_path = processed_dir / "communities.pkl"
        if community_path.exists():
            with open(community_path, "rb") as f:
                comm_data = pickle.load(f)
                self.partition = comm_data.get("partition", {})
                self.community_info = comm_data.get("community_info", [])
        
        # Rebuild entity_to_chunks mapping
        self._rebuild_entity_to_chunks()
        
        print(f"  Loaded {len(self.chunks)} chunks")
        print(f"  Loaded {len(self.sub_chunks)} sub-chunks")
        print(f"  Loaded graph with {self.graph.number_of_nodes() if self.graph else 0} nodes")
        print(f"  Loaded {len(self.community_info)} communities")
    
    def _rebuild_entity_to_chunks(self):
        """Rebuild entity to chunks mapping from graph."""
        if self.graph:
            for node, data in self.graph.nodes(data=True):
                self.entity_to_chunks[node] = data.get("chunk_ids", [])
    
    # ==================== QUERY PIPELINE ====================
    
    def query(
        self,
        question: str,
        use_local: bool = True,
        use_global: bool = True,
        top_k: int = 5
    ) -> Dict:
        """
        Answer a question using the RAG pipeline.
        
        Args:
            question: User's question
            use_local: Whether to use local graph RAG search
            use_global: Whether to use global graph RAG search
            top_k: Number of results to retrieve
            
        Returns:
            Dictionary with answer and metadata
        """
        print(f"\nProcessing query: {question}")
        
        # Encode query
        query_embedding = self.embedding_model.encode([question])[0]
        
        # Local search
        local_results = []
        if use_local:
            local_results = self.local_search.search(
                query_embedding=query_embedding,
                entity_embeddings=self.entity_embeddings,
                chunk_embeddings=self.chunk_embeddings,
                entity_to_chunks=self.entity_to_chunks,
                chunks=self.sub_chunks,
                graph=self.graph,
                top_k=top_k
            )
            print(f"  Local search: {len(local_results)} results")
        
        # Global search
        global_results = None
        if use_global and self.community_info:
            global_results = self.global_search.search(
                query_embedding=query_embedding,
                community_info=self.community_info,
                community_embeddings=self.community_embeddings,
                chunk_embeddings=self.chunk_embeddings,
                chunks=self.sub_chunks,
                top_k_communities=3,
                top_k_chunks=top_k
            )
            print(f"  Global search: {len(global_results[0])} communities, {len(global_results[1])} chunks")
        
        # Get relevant entities
        entities = self.local_search.search_with_entities(
            query_embedding=query_embedding,
            entity_embeddings=self.entity_embeddings,
            graph=self.graph,
            top_k=5
        )
        
        # Generate answer
        result = self.answer_generator.generate_answer(
            query=question,
            local_results=local_results,
            global_results=global_results,
            entities=entities,
            history=self.history[-4:] if self.history else None
        )
        
        # Update history
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": result["answer"]})
        
        return result
    
    def query_simple(self, question: str) -> str:
        """Simple query interface returning just the answer."""
        result = self.query(question)
        return result.get("answer", "Unable to generate answer.")
    
    def clear_history(self):
        """Clear conversation history."""
        self.history = []
    
    # ==================== INTERACTIVE MODE ====================
    
    def interactive_mode(self):
        """Run interactive Q&A session."""
        print("\n" + "=" * 60)
        print("AMBEDKARGPT - Interactive Mode")
        print("Ask questions about Dr. B.R. Ambedkar's works")
        print("Commands: 'quit' to exit, 'clear' to reset history")
        print("=" * 60 + "\n")
        
        while True:
            try:
                question = input("\nYou: ").strip()
                
                if not question:
                    continue
                
                if question.lower() == "quit":
                    print("Goodbye!")
                    break
                
                if question.lower() == "clear":
                    self.clear_history()
                    print("History cleared.")
                    continue
                
                result = self.query(question)
                print(f"\nAmbedkarGPT: {result['answer']}")
                
                # Show citations
                if result.get("citations"):
                    print("\n📚 Sources:")
                    for i, cite in enumerate(result["citations"][:3], 1):
                        print(f"  {i}. [{cite['chunk_id']}] (score: {cite['score']:.2f})")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main entry point for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AmbedkarGPT - SEMRAG RAG System")
    parser.add_argument("--config", type=str, help="Path to config.yaml")
    parser.add_argument("--ingest", type=str, help="Path to PDF to ingest")
    parser.add_argument("--load", action="store_true", help="Load processed data")
    parser.add_argument("--query", type=str, help="Single query to answer")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    
    args = parser.parse_args()
    
    # Initialize
    gpt = AmbedkarGPT(config_path=args.config)
    
    # Ingest or load
    if args.ingest:
        gpt.ingest_document(args.ingest)
    elif args.load:
        gpt.load_processed_data()
    
    # Query or interactive
    if args.query:
        result = gpt.query(args.query)
        print(f"\nAnswer: {result['answer']}")
    elif args.interactive:
        gpt.interactive_mode()


if __name__ == "__main__":
    main()
