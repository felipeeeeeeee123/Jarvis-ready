"""Knowledge base with RAG (Retrieval-Augmented Generation) using FAISS for JARVIS v3.0."""

import os
import json
import pickle
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

from database.services import qa_service, memory_service
from utils.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class SimpleEmbedding:
    """Simple text embedding using TF-IDF when sentence-transformers is not available."""
    
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=512,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)
        )
        self.is_fitted = False
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit the vectorizer and transform texts."""
        embeddings = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return embeddings.toarray().astype(np.float32)
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts using fitted vectorizer."""
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit_transform first.")
        embeddings = self.vectorizer.transform(texts)
        return embeddings.toarray().astype(np.float32)


class KnowledgeBase:
    """Knowledge base with semantic search using FAISS."""
    
    def __init__(self, embedding_dim: int = 512, max_entries: int = 10000):
        self.embedding_dim = embedding_dim
        self.max_entries = max_entries
        self.index = None
        self.documents = []
        self.embeddings_cache = {}
        self.embedder = None
        self.kb_dir = Path(settings.DATA_DIR) / "knowledge_base"
        self.kb_dir.mkdir(exist_ok=True)
        
        logger.info(f"Knowledge base initialized (FAISS available: {FAISS_AVAILABLE})")
        
        if FAISS_AVAILABLE:
            self.initialize_faiss()
        else:
            logger.warning("FAISS not available, using fallback search")
    
    def initialize_faiss(self):
        """Initialize FAISS index."""
        try:
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
            logger.info("FAISS index initialized")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            self.index = None
    
    def _get_embedder(self):
        """Get or create text embedder."""
        if self.embedder is None:
            try:
                # Try to use sentence-transformers if available
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                self.embedding_dim = 384  # Update dimension for this model
                logger.info("Using SentenceTransformers for embeddings")
            except ImportError:
                # Fallback to TF-IDF
                self.embedder = SimpleEmbedding()
                logger.info("Using TF-IDF for embeddings (fallback)")
        return self.embedder
    
    def _embed_text(self, text: str) -> np.ndarray:
        """Convert text to embedding vector."""
        embedder = self._get_embedder()
        
        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.embeddings_cache:
            return self.embeddings_cache[text_hash]
        
        try:
            if hasattr(embedder, 'encode'):  # SentenceTransformers
                embedding = embedder.encode([text])[0]
            else:  # TF-IDF fallback
                if not embedder.is_fitted:
                    # Fit on existing documents if available
                    if self.documents:
                        texts = [doc['content'] for doc in self.documents[-1000:]]  # Use recent docs
                        texts.append(text)
                        embeddings = embedder.fit_transform(texts)
                        embedding = embeddings[-1]  # Last one is our text
                    else:
                        # Fit on just this text
                        embedding = embedder.fit_transform([text])[0]
                else:
                    embedding = embedder.transform([text])[0]
            
            # Normalize for cosine similarity
            embedding = embedding / np.linalg.norm(embedding)
            
            # Cache the embedding
            self.embeddings_cache[text_hash] = embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            # Return random embedding as fallback
            return np.random.rand(self.embedding_dim).astype(np.float32)
    
    def add_document(self, content: str, metadata: Optional[Dict] = None, source: str = "unknown") -> bool:
        """Add a document to the knowledge base."""
        try:
            # Create document entry
            doc = {
                "id": len(self.documents),
                "content": content,
                "metadata": metadata or {},
                "source": source,
                "timestamp": np.datetime64('now').astype(str)
            }
            
            # Generate embedding
            embedding = self._embed_text(content)
            
            # Add to FAISS index if available
            if self.index is not None and FAISS_AVAILABLE:
                # Ensure embedding has correct dimension
                if embedding.shape[0] != self.embedding_dim:
                    # Update embedding dimension if needed
                    self.embedding_dim = embedding.shape[0]
                    self.index = faiss.IndexFlatIP(self.embedding_dim)
                    logger.info(f"Updated embedding dimension to {self.embedding_dim}")
                
                self.index.add(embedding.reshape(1, -1))
            
            # Add to document store
            self.documents.append(doc)
            
            # Limit size
            if len(self.documents) > self.max_entries:
                self.documents = self.documents[-self.max_entries:]
                # Note: FAISS index doesn't support easy removal, 
                # so we'd need to rebuild it periodically
            
            logger.info(f"Added document to knowledge base (total: {len(self.documents)})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5, min_similarity: float = 0.1) -> List[Dict[str, Any]]:
        """Search for relevant documents."""
        if not self.documents:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self._embed_text(query)
            
            if self.index is not None and FAISS_AVAILABLE:
                # Use FAISS for semantic search
                return self._faiss_search(query_embedding, top_k, min_similarity)
            else:
                # Fallback to basic text search
                return self._fallback_search(query, top_k)
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _faiss_search(self, query_embedding: np.ndarray, top_k: int, min_similarity: float) -> List[Dict[str, Any]]:
        """Perform FAISS-based semantic search."""
        try:
            # Search FAISS index
            similarities, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
            
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if similarity >= min_similarity and idx < len(self.documents):
                    doc = self.documents[idx].copy()
                    doc['similarity'] = float(similarity)
                    doc['rank'] = i + 1
                    results.append(doc)
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []
    
    def _fallback_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback text-based search."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored_docs = []
        for doc in self.documents:
            content_lower = doc['content'].lower()
            content_words = set(content_lower.split())
            
            # Simple scoring based on word overlap
            overlap = len(query_words.intersection(content_words))
            if overlap > 0:
                score = overlap / len(query_words.union(content_words))
                doc_copy = doc.copy()
                doc_copy['similarity'] = score
                scored_docs.append(doc_copy)
        
        # Sort by score and return top_k
        scored_docs.sort(key=lambda x: x['similarity'], reverse=True)
        return scored_docs[:top_k]
    
    def get_context_for_query(self, query: str, max_context_length: int = 1000) -> str:
        """Get relevant context for a query."""
        relevant_docs = self.search(query, top_k=3, min_similarity=0.1)
        
        if not relevant_docs:
            return ""
        
        context_parts = []
        total_length = 0
        
        for doc in relevant_docs:
            content = doc['content']
            if total_length + len(content) <= max_context_length:
                context_parts.append(f"[{doc['source']}] {content}")
                total_length += len(content)
            else:
                # Add truncated content
                remaining_length = max_context_length - total_length
                if remaining_length > 50:  # Only add if reasonable length
                    truncated = content[:remaining_length-3] + "..."
                    context_parts.append(f"[{doc['source']}] {truncated}")
                break
        
        return "\n\n".join(context_parts)
    
    def populate_from_qa_history(self, limit: int = 1000) -> int:
        """Populate knowledge base from Q&A history."""
        try:
            qa_entries = qa_service.get_recent_entries(limit)
            added_count = 0
            
            for entry in qa_entries:
                # Combine question and answer as a knowledge document
                content = f"Q: {entry.question}\nA: {entry.answer}"
                metadata = {
                    "confidence": entry.confidence_score,
                    "token_count": entry.token_count,
                    "created_at": entry.created_at.isoformat()
                }
                
                if self.add_document(content, metadata, f"qa_history_{entry.source}"):
                    added_count += 1
            
            logger.info(f"Populated knowledge base with {added_count} Q&A entries")
            return added_count
            
        except Exception as e:
            logger.error(f"Failed to populate from Q&A history: {e}")
            return 0
    
    def save_to_disk(self) -> bool:
        """Save knowledge base to disk."""
        try:
            # Save documents
            docs_path = self.kb_dir / "documents.json"
            with open(docs_path, 'w') as f:
                json.dump(self.documents, f, indent=2, default=str)
            
            # Save embeddings cache
            cache_path = self.kb_dir / "embeddings_cache.pkl"
            with open(cache_path, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
            
            # Save FAISS index if available
            if self.index is not None and FAISS_AVAILABLE:
                index_path = self.kb_dir / "faiss_index.bin"
                faiss.write_index(self.index, str(index_path))
            
            logger.info("Knowledge base saved to disk")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save knowledge base: {e}")
            return False
    
    def load_from_disk(self) -> bool:
        """Load knowledge base from disk."""
        try:
            # Load documents
            docs_path = self.kb_dir / "documents.json"
            if docs_path.exists():
                with open(docs_path, 'r') as f:
                    self.documents = json.load(f)
            
            # Load embeddings cache
            cache_path = self.kb_dir / "embeddings_cache.pkl"
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
            
            # Load FAISS index if available
            index_path = self.kb_dir / "faiss_index.bin"
            if index_path.exists() and FAISS_AVAILABLE:
                self.index = faiss.read_index(str(index_path))
            
            logger.info(f"Knowledge base loaded from disk ({len(self.documents)} documents)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        return {
            "total_documents": len(self.documents),
            "embedding_dimension": self.embedding_dim,
            "faiss_available": FAISS_AVAILABLE,
            "cached_embeddings": len(self.embeddings_cache),
            "sources": list(set(doc.get('source', 'unknown') for doc in self.documents))
        }


# Global knowledge base instance
knowledge_base = KnowledgeBase()