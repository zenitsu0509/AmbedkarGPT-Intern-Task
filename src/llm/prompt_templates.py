"""
Prompt Templates - Templates for LLM prompts in RAG pipeline.

Provides structured prompts for question answering, summarization,
and context integration.
"""

from typing import List, Dict, Optional


class PromptTemplates:
    """
    Collection of prompt templates for the RAG system.
    
    Templates are designed to work with Mistral 7B and similar models.
    """
    
    # System prompts
    SYSTEM_QA = """You are AmbedkarGPT, an expert assistant on Dr. B.R. Ambedkar's works and writings. 
You answer questions based on the provided context from his books and documents.
Always base your answers on the given context. If the context doesn't contain enough information, say so.
Be accurate, respectful, and informative in your responses."""

    SYSTEM_SUMMARIZE = """You are an expert at summarizing and analyzing text.
Provide concise, accurate summaries that capture the key points."""

    @staticmethod
    def qa_with_context(
        query: str,
        context: str,
        entities: Optional[List[str]] = None,
        community_summaries: Optional[List[str]] = None
    ) -> str:
        """
        Create a Q&A prompt with retrieved context.
        
        Args:
            query: User's question
            context: Retrieved text context
            entities: Optional list of relevant entities
            community_summaries: Optional community summaries
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        # Add context
        prompt_parts.append("### Retrieved Context:")
        prompt_parts.append(context)
        prompt_parts.append("")
        
        # Add entities if provided
        if entities:
            prompt_parts.append("### Relevant Entities:")
            prompt_parts.append(", ".join(entities[:10]))
            prompt_parts.append("")
        
        # Add community summaries if provided
        if community_summaries:
            prompt_parts.append("### Topic Summaries:")
            for i, summary in enumerate(community_summaries[:3], 1):
                prompt_parts.append(f"{i}. {summary}")
            prompt_parts.append("")
        
        # Add question
        prompt_parts.append("### Question:")
        prompt_parts.append(query)
        prompt_parts.append("")
        
        # Add instruction
        prompt_parts.append("### Instructions:")
        prompt_parts.append("Based on the context provided above, answer the question thoroughly and accurately.")
        prompt_parts.append("If the context doesn't contain sufficient information, acknowledge this limitation.")
        prompt_parts.append("Cite specific parts of the context when relevant.")
        prompt_parts.append("")
        prompt_parts.append("### Answer:")
        
        return "\n".join(prompt_parts)
    
    @staticmethod
    def qa_simple(query: str, context: str) -> str:
        """Simple Q&A prompt without extra metadata."""
        return f"""Context from Dr. B.R. Ambedkar's works:
{context}

Question: {query}

Please provide a detailed answer based on the context above. If the context doesn't contain enough information to fully answer the question, mention what information is missing.

Answer:"""

    @staticmethod
    def summarize_community(
        entities: List[Dict],
        sample_text: str,
        entity_types: Dict[str, int]
    ) -> str:
        """
        Create a prompt for summarizing a community.
        
        Args:
            entities: List of entity dictionaries
            sample_text: Sample text from the community
            entity_types: Count of entity types
        """
        entity_list = ", ".join([e.get("name", "") for e in entities[:10]])
        type_list = ", ".join([f"{k}: {v}" for k, v in entity_types.items()])
        
        return f"""Analyze this group of related concepts from Dr. B.R. Ambedkar's writings:

**Key entities ({len(entities)} total):** {entity_list}

**Entity types:** {type_list}

**Sample text excerpt:**
{sample_text[:500]}...

Provide a 2-3 sentence summary describing:
1. The main theme or topic this group represents
2. How these concepts relate to Ambedkar's philosophy
3. Key insights about this topic

Summary:"""

    @staticmethod
    def extract_entities(text: str) -> str:
        """Prompt for entity extraction (if using LLM for NER)."""
        return f"""Extract named entities from the following text. Identify:
- People (PERSON)
- Organizations (ORG)
- Locations (GPE/LOC)
- Dates (DATE)
- Events (EVENT)

Text:
{text}

List the entities in format: "entity_text | entity_type"

Entities:"""

    @staticmethod
    def extract_relationships(text: str, entities: List[str]) -> str:
        """Prompt for relationship extraction."""
        entity_list = ", ".join(entities[:20])
        
        return f"""Given the following text and entities, identify relationships between the entities.

Text:
{text}

Entities: {entity_list}

List relationships in format: "entity1 | relationship | entity2"
Focus on meaningful relationships like: works_with, belongs_to, advocates_for, opposes, etc.

Relationships:"""

    @staticmethod
    def refine_answer(
        query: str,
        initial_answer: str,
        additional_context: str
    ) -> str:
        """Prompt to refine an answer with additional context."""
        return f"""Original question: {query}

Initial answer: {initial_answer}

Additional context:
{additional_context}

Please refine the answer by incorporating the additional context. 
Correct any inaccuracies and add relevant details from the new context.

Refined answer:"""

    @staticmethod
    def compare_perspectives(
        topic: str,
        context_chunks: List[str]
    ) -> str:
        """Prompt to compare different perspectives on a topic."""
        contexts = "\n\n".join([
            f"Excerpt {i+1}:\n{chunk}"
            for i, chunk in enumerate(context_chunks[:5])
        ])
        
        return f"""Analyze the following excerpts about "{topic}" from Dr. B.R. Ambedkar's works:

{contexts}

Compare the different aspects or perspectives presented:
1. What are the common themes?
2. Are there any contrasting viewpoints?
3. How does Ambedkar's position evolve or remain consistent?

Analysis:"""

    @staticmethod
    def create_chat_messages(
        query: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """
        Create chat message format for conversational QA.
        
        Args:
            query: Current user query
            context: Retrieved context
            history: Previous conversation history
            
        Returns:
            List of message dictionaries
        """
        messages = [
            {
                "role": "system",
                "content": PromptTemplates.SYSTEM_QA
            }
        ]
        
        # Add history if provided
        if history:
            messages.extend(history[-4:])  # Last 4 turns
        
        # Add current query with context
        user_message = f"""Context from Dr. B.R. Ambedkar's works:
{context}

My question: {query}

Please answer based on the context provided."""
        
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        return messages
    
    @staticmethod
    def format_with_citations(
        query: str,
        context_with_sources: List[Dict]
    ) -> str:
        """
        Create a prompt that encourages citing sources.
        
        Args:
            query: User's question
            context_with_sources: List of context dicts with 'text' and 'source_id'
        """
        context_parts = []
        for i, ctx in enumerate(context_with_sources):
            source_id = ctx.get("source_id", f"Source {i+1}")
            text = ctx.get("text", "")
            context_parts.append(f"[{source_id}]: {text}")
        
        context_str = "\n\n".join(context_parts)
        
        return f"""Answer the question using the numbered sources below. 
When you use information from a source, cite it using [Source ID].

Sources:
{context_str}

Question: {query}

Answer (with citations):"""
