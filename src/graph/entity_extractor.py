"""
Entity Extractor - Extract entities from text chunks using spaCy.

Uses Named Entity Recognition (NER) and dependency parsing to identify
entities and their relationships within chunks.
"""

import re
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict

import spacy
from spacy.tokens import Doc, Span


class EntityExtractor:
    """
    Extract entities and relationships from text chunks using spaCy.
    
    Implements NER-based entity extraction and dependency parsing
    for relationship extraction as described in SEMRAG.
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the entity extractor.
        
        Args:
            model_name: spaCy model to use for NER
        """
        print(f"Loading spaCy model: {model_name}")
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model {model_name} not found. Downloading...")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)
        
        # Entity types to extract
        self.entity_types = {
            "PERSON",      # People, including fictional
            "ORG",         # Organizations
            "GPE",         # Countries, cities, states
            "LOC",         # Non-GPE locations
            "EVENT",       # Named events
            "WORK_OF_ART", # Titles of books, songs, etc.
            "LAW",         # Named documents made into laws
            "NORP",        # Nationalities, religious/political groups
            "DATE",        # Dates or periods
            "FAC",         # Facilities
        }
        
        # Relationship patterns (subject-verb-object)
        self.relation_patterns = [
            "nsubj",   # Nominal subject
            "dobj",    # Direct object
            "pobj",    # Object of preposition
            "attr",    # Attribute
            "prep",    # Prepositional modifier
        ]
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of entity dictionaries with text, label, and positions
        """
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in self.entity_types:
                entity = {
                    "text": ent.text.strip(),
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "normalized": self._normalize_entity(ent.text)
                }
                entities.append(entity)
        
        return entities
    
    def _normalize_entity(self, text: str) -> str:
        """Normalize entity text for matching."""
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', text.strip())
        # Convert to lowercase for matching
        normalized = normalized.lower()
        return normalized
    
    def extract_relationships(self, text: str) -> List[Dict]:
        """
        Extract relationships between entities using dependency parsing.
        
        Args:
            text: Input text
            
        Returns:
            List of relationship dictionaries
        """
        doc = self.nlp(text)
        relationships = []
        
        # Get entity spans for reference
        entity_spans = {(ent.start, ent.end): ent for ent in doc.ents}
        
        for sent in doc.sents:
            # Find subject-verb-object patterns
            relations = self._extract_svo_relations(sent, entity_spans)
            relationships.extend(relations)
            
            # Find entity co-occurrence relationships
            cooccur_relations = self._extract_cooccurrence_relations(sent, doc)
            relationships.extend(cooccur_relations)
        
        return relationships
    
    def _extract_svo_relations(
        self,
        sent: Span,
        entity_spans: Dict
    ) -> List[Dict]:
        """Extract subject-verb-object relationships."""
        relations = []
        
        for token in sent:
            # Look for verbs
            if token.pos_ == "VERB":
                subject = None
                obj = None
                
                # Find subject
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subject = self._get_entity_or_noun_phrase(child, entity_spans)
                    elif child.dep_ in ["dobj", "pobj", "attr"]:
                        obj = self._get_entity_or_noun_phrase(child, entity_spans)
                
                if subject and obj:
                    relation = {
                        "subject": subject,
                        "predicate": token.lemma_,
                        "object": obj,
                        "type": "SVO",
                        "sentence": sent.text
                    }
                    relations.append(relation)
        
        return relations
    
    def _get_entity_or_noun_phrase(
        self,
        token,
        entity_spans: Dict
    ) -> Optional[str]:
        """Get the entity text or noun phrase for a token."""
        # Check if token is part of an entity
        for (start, end), ent in entity_spans.items():
            if start <= token.i < end:
                return ent.text
        
        # Otherwise, get the noun phrase
        if token.pos_ in ["NOUN", "PROPN"]:
            # Get the full noun phrase
            phrase_tokens = [token]
            for child in token.children:
                if child.dep_ in ["compound", "amod", "det"]:
                    phrase_tokens.append(child)
            
            phrase_tokens.sort(key=lambda t: t.i)
            return " ".join([t.text for t in phrase_tokens])
        
        return token.text if token.pos_ in ["NOUN", "PROPN"] else None
    
    def _extract_cooccurrence_relations(
        self,
        sent: Span,
        doc: Doc
    ) -> List[Dict]:
        """Extract relationships based on entity co-occurrence in same sentence."""
        relations = []
        entities_in_sent = [ent for ent in doc.ents if ent.sent == sent]
        
        # Create co-occurrence relations between entities in same sentence
        for i, ent1 in enumerate(entities_in_sent):
            for ent2 in entities_in_sent[i+1:]:
                if ent1.text != ent2.text:
                    relation = {
                        "subject": ent1.text,
                        "predicate": "RELATED_TO",
                        "object": ent2.text,
                        "type": "CO_OCCURRENCE",
                        "subject_label": ent1.label_,
                        "object_label": ent2.label_,
                        "sentence": sent.text
                    }
                    relations.append(relation)
        
        return relations
    
    def process_chunks(
        self,
        chunks: List[Dict],
        min_entity_freq: int = 2
    ) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Process all chunks to extract entities and relationships.
        
        Args:
            chunks: List of chunk dictionaries
            min_entity_freq: Minimum frequency for entity inclusion
            
        Returns:
            Tuple of (entities, relationships, entity_to_chunks mapping)
        """
        all_entities = []
        all_relationships = []
        entity_frequency = defaultdict(int)
        entity_to_chunks = defaultdict(list)
        entity_details = {}
        
        print("Extracting entities and relationships from chunks...")
        
        for chunk in chunks:
            chunk_id = chunk["id"]
            text = chunk.get("full_context", chunk["text"])
            
            # Extract entities
            entities = self.extract_entities(text)
            for entity in entities:
                normalized = entity["normalized"]
                entity_frequency[normalized] += 1
                entity_to_chunks[normalized].append(chunk_id)
                
                # Store entity details (use latest occurrence)
                if normalized not in entity_details:
                    entity_details[normalized] = entity
            
            # Extract relationships
            relationships = self.extract_relationships(text)
            for rel in relationships:
                rel["chunk_id"] = chunk_id
                all_relationships.append(rel)
        
        # Filter entities by frequency
        filtered_entities = []
        for normalized, freq in entity_frequency.items():
            if freq >= min_entity_freq:
                entity = entity_details[normalized].copy()
                entity["frequency"] = freq
                entity["chunk_ids"] = entity_to_chunks[normalized]
                filtered_entities.append(entity)
        
        print(f"Extracted {len(filtered_entities)} unique entities")
        print(f"Extracted {len(all_relationships)} relationships")
        
        return filtered_entities, all_relationships, dict(entity_to_chunks)
    
    def get_entity_context(
        self,
        entity_text: str,
        chunks: List[Dict],
        entity_to_chunks: Dict
    ) -> List[str]:
        """Get all context sentences containing an entity."""
        normalized = self._normalize_entity(entity_text)
        chunk_ids = entity_to_chunks.get(normalized, [])
        
        contexts = []
        chunk_map = {c["id"]: c for c in chunks}
        
        for chunk_id in chunk_ids:
            if chunk_id in chunk_map:
                contexts.append(chunk_map[chunk_id]["text"])
        
        return contexts
