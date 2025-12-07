"""
Answer Generator - Generate answers using retrieved context and LLM.

Combines retrieval results with LLM to produce final answers.
"""

from typing import List, Dict, Optional, Tuple
from .llm_client import LLMClient
from .prompt_templates import PromptTemplates


class AnswerGenerator:
    """
    Generate answers by combining retrieved context with LLM.
    
    Implements the final step of the RAG pipeline:
    1. Combine local and global retrieval results
    2. Format context with entities and community summaries
    3. Generate answer using LLM
    4. Include citations to source chunks
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        max_context_tokens: int = 3000,
        include_citations: bool = True
    ):
        """
        Initialize the answer generator.
        
        Args:
            llm_client: LLM client instance
            max_context_tokens: Maximum tokens for context
            include_citations: Whether to include source citations
        """
        self.llm = llm_client
        self.max_context_tokens = max_context_tokens
        self.include_citations = include_citations
        self.templates = PromptTemplates()
    
    def generate_answer(
        self,
        query: str,
        local_results: List[Dict],
        global_results: Optional[Tuple[List[Dict], List[Dict]]] = None,
        entities: Optional[List[Dict]] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Dict:
        """
        Generate an answer for the query using retrieved context.
        
        Args:
            query: User's question
            local_results: Results from local graph RAG search
            global_results: Tuple of (community_results, chunk_results) from global search
            entities: Relevant entities from the knowledge graph
            history: Conversation history
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Prepare context from local results
        local_context = self._format_local_context(local_results)
        
        # Prepare context from global results
        global_context = ""
        community_summaries = []
        if global_results:
            communities, global_chunks = global_results
            global_context = self._format_global_context(global_chunks)
            community_summaries = [
                c.get("summary", "") for c in communities if c.get("summary")
            ]
        
        # Combine contexts
        full_context = self._combine_contexts(local_context, global_context)
        
        # Get entity names
        entity_names = []
        if entities:
            entity_names = [e.get("entity", e.get("name", "")) for e in entities[:10]]
        
        # Create prompt
        prompt = PromptTemplates.qa_with_context(
            query=query,
            context=full_context,
            entities=entity_names if entity_names else None,
            community_summaries=community_summaries if community_summaries else None
        )
        
        # Generate answer
        answer = self.llm.generate(
            prompt=prompt,
            system_prompt=PromptTemplates.SYSTEM_QA,
            max_tokens=1024
        )
        
        # Prepare citations
        citations = self._extract_citations(local_results, global_results)
        
        return {
            "answer": answer,
            "query": query,
            "citations": citations,
            "entities_used": entity_names,
            "community_summaries": community_summaries,
            "context_length": len(full_context)
        }
    
    def _format_local_context(self, results: List[Dict]) -> str:
        """Format local search results as context."""
        if not results:
            return ""
        
        context_parts = []
        total_chars = 0
        max_chars = self.max_context_tokens * 4  # Rough char estimate
        
        for i, result in enumerate(results):
            text = result.get("text", "")
            chunk_id = result.get("id", f"chunk_{i}")
            score = result.get("combined_score", result.get("similarity_score", 0))
            
            if self.include_citations:
                part = f"[{chunk_id}] (relevance: {score:.2f})\n{text}"
            else:
                part = text
            
            if total_chars + len(part) > max_chars:
                break
            
            context_parts.append(part)
            total_chars += len(part)
        
        return "\n\n".join(context_parts)
    
    def _format_global_context(self, results: List[Dict]) -> str:
        """Format global search results as context."""
        if not results:
            return ""
        
        context_parts = []
        total_chars = 0
        max_chars = self.max_context_tokens * 2  # Give less space to global
        
        for result in results:
            text = result.get("text", "")
            comm_id = result.get("community_id", "")
            
            if self.include_citations:
                part = f"[Community {comm_id}]\n{text}"
            else:
                part = text
            
            if total_chars + len(part) > max_chars:
                break
            
            context_parts.append(part)
            total_chars += len(part)
        
        return "\n\n".join(context_parts)
    
    def _combine_contexts(self, local: str, global_ctx: str) -> str:
        """Combine local and global contexts."""
        parts = []
        
        if local:
            parts.append("## Directly Relevant Passages:\n" + local)
        
        if global_ctx:
            parts.append("## Related Topic Context:\n" + global_ctx)
        
        return "\n\n".join(parts)
    
    def _extract_citations(
        self,
        local_results: List[Dict],
        global_results: Optional[Tuple[List[Dict], List[Dict]]]
    ) -> List[Dict]:
        """Extract citation information from results."""
        citations = []
        
        # Local citations
        for result in local_results[:5]:
            citations.append({
                "source": "local",
                "chunk_id": result.get("id"),
                "score": result.get("combined_score", 0),
                "preview": result.get("text", "")[:150] + "..."
            })
        
        # Global citations
        if global_results:
            communities, chunks = global_results
            for result in chunks[:3]:
                citations.append({
                    "source": "global",
                    "chunk_id": result.get("id"),
                    "community_id": result.get("community_id"),
                    "score": result.get("global_score", 0),
                    "preview": result.get("text", "")[:150] + "..."
                })
        
        return citations
    
    def generate_simple_answer(self, query: str, context: str) -> str:
        """
        Generate a simple answer without complex formatting.
        
        Useful for quick responses or testing.
        """
        prompt = PromptTemplates.qa_simple(query, context)
        return self.llm.generate(prompt, max_tokens=512)
    
    def generate_with_followup(
        self,
        query: str,
        context: str,
        history: List[Dict[str, str]]
    ) -> str:
        """
        Generate an answer considering conversation history.
        
        Useful for multi-turn conversations.
        """
        messages = PromptTemplates.create_chat_messages(
            query=query,
            context=context,
            history=history
        )
        
        return self.llm.chat(messages)
    
    def summarize_results(self, results: List[Dict]) -> str:
        """
        Generate a summary of the retrieved results.
        
        Useful for providing an overview before detailed answers.
        """
        if not results:
            return "No relevant information found."
        
        # Combine texts
        texts = [r.get("text", "")[:300] for r in results[:5]]
        combined = "\n\n".join(texts)
        
        prompt = f"""Summarize the following excerpts from Dr. B.R. Ambedkar's works in 2-3 sentences:

{combined}

Summary:"""
        
        return self.llm.generate(prompt, max_tokens=200)
    
    def explain_entities(
        self,
        entities: List[Dict],
        context: str
    ) -> str:
        """
        Generate explanations for key entities.
        
        Helps users understand the important concepts.
        """
        entity_names = [e.get("name", e.get("entity", "")) for e in entities[:5]]
        entity_list = ", ".join(entity_names)
        
        prompt = f"""Based on the following context, briefly explain the significance of these entities in Dr. B.R. Ambedkar's works:

Entities: {entity_list}

Context:
{context[:1500]}

For each entity, provide a one-sentence explanation of its relevance:"""
        
        return self.llm.generate(prompt, max_tokens=400)
