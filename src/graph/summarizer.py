"""
Community Summarizer - Generate summaries for detected communities using LLM.

Creates human-readable summaries of each community's theme and contents.
"""

from typing import List, Dict, Optional
import json


class CommunitySummarizer:
    """
    Generate summaries for communities using an LLM.
    
    Creates concise descriptions of what each community represents,
    its key entities, and main themes.
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize the summarizer.
        
        Args:
            llm_client: LLM client for generating summaries
        """
        self.llm_client = llm_client
    
    def set_llm_client(self, llm_client):
        """Set the LLM client."""
        self.llm_client = llm_client
    
    def generate_community_summary(
        self,
        community_info: Dict,
        max_entities: int = 10,
        max_chunks: int = 3
    ) -> str:
        """
        Generate a summary for a single community.
        
        Args:
            community_info: Community information dictionary
            max_entities: Maximum entities to include in prompt
            max_chunks: Maximum chunk samples to include
            
        Returns:
            Generated summary string
        """
        if self.llm_client is None:
            # Fallback: create a simple template-based summary
            return self._create_template_summary(community_info)
        
        # Prepare entities list
        entities = community_info.get("entities", [])[:max_entities]
        entity_list = ", ".join([e["name"] for e in entities])
        
        # Prepare entity types
        entity_labels = community_info.get("entity_labels", {})
        label_str = ", ".join([f"{k}: {v}" for k, v in entity_labels.items()])
        
        # Prepare sample text
        sample_chunks = community_info.get("sample_chunks", [])[:max_chunks]
        sample_text = "\n".join([f"- {chunk[:300]}..." for chunk in sample_chunks])
        
        # Create prompt
        prompt = f"""Analyze this group of related entities from Dr. B.R. Ambedkar's works and provide a concise summary.

**Entities in this group ({community_info.get('size', 0)} total):**
{entity_list}

**Entity types:**
{label_str}

**Sample text excerpts:**
{sample_text}

Based on the entities and context, write a 2-3 sentence summary describing:
1. The main theme or topic this group represents
2. Key relationships between the entities
3. Why these entities are grouped together

Summary:"""

        try:
            summary = self.llm_client.generate(prompt, max_tokens=256)
            return summary.strip()
        except Exception as e:
            print(f"Error generating summary: {e}")
            return self._create_template_summary(community_info)
    
    def _create_template_summary(self, community_info: Dict) -> str:
        """Create a template-based summary without LLM."""
        entities = community_info.get("entities", [])
        entity_labels = community_info.get("entity_labels", {})
        size = community_info.get("size", 0)
        
        # Get top entities by importance/frequency
        sorted_entities = sorted(
            entities,
            key=lambda x: x.get("importance", x.get("frequency", 0)),
            reverse=True
        )[:5]
        
        top_names = [e["name"] for e in sorted_entities]
        
        # Determine dominant entity type
        dominant_type = max(entity_labels.items(), key=lambda x: x[1])[0] if entity_labels else "ENTITY"
        
        summary = f"This community contains {size} related entities, primarily {dominant_type}s. "
        summary += f"Key entities include: {', '.join(top_names)}. "
        
        # Add context based on entity types
        if "PERSON" in entity_labels:
            summary += "This group focuses on people and their relationships. "
        if "ORG" in entity_labels:
            summary += "Organizations and institutions are central to this theme. "
        if "GPE" in entity_labels or "LOC" in entity_labels:
            summary += "Geographic locations play an important role. "
        if "NORP" in entity_labels:
            summary += "This involves nationalities, religious or political groups. "
        
        return summary
    
    def summarize_all_communities(
        self,
        community_info_list: List[Dict],
        batch_size: int = 5
    ) -> List[Dict]:
        """
        Generate summaries for all communities.
        
        Args:
            community_info_list: List of community info dictionaries
            batch_size: Number of communities to process at once
            
        Returns:
            Updated community info list with summaries
        """
        print(f"Generating summaries for {len(community_info_list)} communities...")
        
        for i, community_info in enumerate(community_info_list):
            print(f"  Processing community {i+1}/{len(community_info_list)}...")
            summary = self.generate_community_summary(community_info)
            community_info["summary"] = summary
        
        return community_info_list
    
    def create_community_report(
        self,
        community_info_list: List[Dict]
    ) -> str:
        """
        Create a formatted report of all communities.
        
        Args:
            community_info_list: List of community info with summaries
            
        Returns:
            Formatted report string
        """
        report_lines = [
            "# Knowledge Graph Community Report",
            "",
            f"Total communities: {len(community_info_list)}",
            "",
        ]
        
        for info in community_info_list:
            comm_id = info.get("community_id", "Unknown")
            size = info.get("size", 0)
            summary = info.get("summary", "No summary available")
            
            report_lines.extend([
                f"## Community {comm_id}",
                f"**Size:** {size} entities",
                f"**Summary:** {summary}",
                "",
                "**Key Entities:**",
            ])
            
            # Add top 5 entities
            entities = info.get("entities", [])[:5]
            for ent in entities:
                report_lines.append(f"- {ent['name']} ({ent['label']})")
            
            report_lines.extend(["", "---", ""])
        
        return "\n".join(report_lines)
    
    def get_community_summary_embeddings(
        self,
        community_info_list: List[Dict],
        embedding_model
    ) -> Dict[int, list]:
        """
        Generate embeddings for community summaries.
        
        Args:
            community_info_list: Communities with summaries
            embedding_model: Sentence transformer model
            
        Returns:
            Dictionary mapping community IDs to summary embeddings
        """
        summary_embeddings = {}
        
        for info in community_info_list:
            comm_id = info.get("community_id")
            summary = info.get("summary", "")
            
            if summary:
                embedding = embedding_model.encode([summary])[0]
                summary_embeddings[comm_id] = embedding.tolist()
        
        return summary_embeddings
