"""
Integration tests for AmbedkarGPT pipeline.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import tempfile
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestPipelineComponents:
    """Integration tests for pipeline components."""
    
    def test_chunking_to_graph_flow(self):
        """Test data flows correctly from chunking to graph."""
        from chunking.semantic_chunker import SemanticChunker
        from graph.entity_extractor import EntityExtractor
        
        # Create chunker
        chunker = SemanticChunker(
            embedding_model="all-MiniLM-L6-v2",
            max_tokens=512,
            similarity_threshold=0.5
        )
        
        # Test with sample text
        sample_text = """
        Dr. B.R. Ambedkar was born in Mhow, India. He was the chief architect 
        of the Indian Constitution. Ambedkar fought against caste discrimination 
        throughout his life. He established the Independent Labour Party in 1936.
        """
        
        sentences = chunker.split_into_sentences(sample_text)
        assert len(sentences) > 0
        
        embeddings = chunker.generate_embeddings(sentences)
        assert embeddings.shape[0] == len(sentences)
        
        groups = chunker.semantic_grouping(sentences, embeddings)
        assert len(groups) > 0
        
        # Test entity extraction
        extractor = EntityExtractor(model_name="en_core_web_sm")
        entities = extractor.extract_entities(sample_text)
        
        # Should find some entities
        assert len(entities) > 0
        entity_texts = [e["text"] for e in entities]
        # Should find Ambedkar
        assert any("Ambedkar" in t for t in entity_texts)
    
    def test_graph_to_retrieval_flow(self):
        """Test data flows correctly from graph to retrieval."""
        from graph.graph_builder import GraphBuilder
        from retrieval.local_search import LocalGraphRAGSearch
        
        # Create sample graph data
        entities = [
            {"text": "Ambedkar", "normalized": "ambedkar", "label": "PERSON"},
            {"text": "India", "normalized": "india", "label": "GPE"},
            {"text": "Constitution", "normalized": "constitution", "label": "LAW"}
        ]
        
        relationships = [
            {"subject": "Ambedkar", "predicate": "drafted", "object": "Constitution"},
            {"subject": "Ambedkar", "predicate": "from", "object": "India"}
        ]
        
        chunks = [
            {"id": "chunk_1", "text": "Ambedkar drafted the Constitution of India."}
        ]
        
        # Build graph
        builder = GraphBuilder()
        graph = builder.build_graph(entities, relationships, chunks)
        
        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() > 0
        
        # Test retrieval setup
        local_search = LocalGraphRAGSearch(
            entity_similarity_threshold=0.3,
            chunk_similarity_threshold=0.2,
            top_k=5
        )
        
        assert local_search.entity_threshold == 0.3
    
    def test_end_to_end_query(self):
        """Test a simple end-to-end query flow."""
        from retrieval.ranker import ResultRanker
        from llm.prompt_templates import PromptTemplates
        
        # Test ranking
        ranker = ResultRanker()
        
        local_results = [
            {"id": "c1", "text": "Ambedkar was a reformer.", "combined_score": 0.8}
        ]
        global_results = [
            {"id": "c2", "text": "Social reform in India.", "global_score": 0.6}
        ]
        
        combined = ranker.combine_results(local_results, global_results, top_k=5)
        assert len(combined) > 0
        
        # Test prompt creation
        context = ranker.format_results_for_context(combined)
        prompt = PromptTemplates.qa_simple(
            query="Who was Ambedkar?",
            context=context
        )
        
        assert "Ambedkar" in prompt
        assert "Context" in prompt


class TestDataPersistence:
    """Test data saving and loading."""
    
    def test_chunk_serialization(self):
        """Test chunks can be saved and loaded."""
        chunks = [
            {
                "id": "chunk_1",
                "text": "Sample text here.",
                "token_count": 5,
                "embedding": [0.1, 0.2, 0.3]
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Save without embeddings
            chunks_to_save = [{k: v for k, v in c.items() if k != "embedding"} 
                             for c in chunks]
            json.dump(chunks_to_save, f)
            temp_path = f.name
        
        # Load back
        with open(temp_path, 'r') as f:
            loaded = json.load(f)
        
        assert len(loaded) == 1
        assert loaded[0]["id"] == "chunk_1"
        assert "embedding" not in loaded[0]
        
        # Cleanup
        Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
