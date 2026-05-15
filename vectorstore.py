import os
import pickle
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from PIL import Image
import torch
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import re
import argparse
from download_models import download_models

# --- Configuration ---
EMBEDDING_MODEL = 'SIH/clip-model'
RERANKER_MODEL = 'SIH/cross-encoder_ms-marco-MiniLM-L-6-v2'

class UnifiedEmbeddingGenerator:
    """Generates embeddings for all content types using a single CLIP model."""
    def __init__(self):
        """Initializes the unified CLIP model."""
        if not os.path.exists(EMBEDDING_MODEL):
            raise FileNotFoundError(f"CLIP model not found at {EMBEDDING_MODEL}. Run 'python download_models.py'")
        
        print(f"Loading unified CLIP embedding model: {EMBEDDING_MODEL}...")
        try:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = SentenceTransformer(EMBEDDING_MODEL, device=self.device)
            print(f"Model loaded on device: {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model: {e}")

    def embed(self, content: Any) -> np.ndarray:
        """Embeds either text (str) or an image (PIL Image object)."""
        return self.model.encode(content, convert_to_numpy=True, normalize_embeddings=True)

    def process_chunks(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Processes chunks and generates embeddings with FULL content storage."""
        embeddings = []
        metadata = []
        
        print("Generating unified CLIP embeddings for all chunks...")
        for chunk in tqdm(chunks):
            content_type = chunk['metadata']['type']
            embedding = None
            
            if content_type == 'image' and 'image_path' in chunk['metadata']:
                try:
                    if Path(chunk['metadata']['image_path']).is_file():
                        image = Image.open(chunk['metadata']['image_path']).convert('RGB')
                        embedding = self.embed(image)
                    else:
                        raise FileNotFoundError
                except Exception as e:
                    print(f"Warning: Could not process image. Embedding text instead. Error: {e}")
                    embedding = self.embed(chunk['content'])
            else:
                embedding = self.embed(chunk['content'])

            if embedding is not None:
                embeddings.append(embedding)
                # CRITICAL: Store FULL content + additional context
                metadata.append({
                    'chunk_id': chunk.get('id', 'N/A'),
                    'type': content_type,
                    'content': chunk['content'],  # FULL CONTENT
                    'content_preview': chunk['content'][:200] + '...' if len(chunk['content']) > 200 else chunk['content'],
                    'source': chunk['metadata'].get('source', 'unknown'),
                    'page_number': chunk['metadata'].get('page_number', 'N/A'),
                    'chunk_index': chunk['metadata'].get('chunk_index', 0),
                    'total_chunks': chunk['metadata'].get('total_chunks', 1),
                    'token_count': chunk.get('token_count', 0),
                    'original_metadata': chunk['metadata']
                })
        
        return {
            'embeddings': np.array(embeddings),
            'metadata': metadata
        }


class BM25HybridSearchEngine:
    """
    Advanced hybrid search combining:
    1. Semantic search (FAISS + CLIP)
    2. BM25 keyword search
    3. Cross-encoder reranking
    """
    
    def __init__(self, index, metadata: List[Dict], generator: UnifiedEmbeddingGenerator):
        self.index = index
        self.metadata = metadata
        self.generator = generator
        
        # Initialize BM25
        print("Building BM25 index...")
        self.bm25 = self._build_bm25_index()
        
        # Initialize cross-encoder reranker
        if not os.path.exists(RERANKER_MODEL):
            raise FileNotFoundError(f"Reranker model not found at {RERANKER_MODEL}. Run 'python download_models.py'")
        
        print(f"Loading cross-encoder reranker: {RERANKER_MODEL}...")
        try:
            self.reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
            print("Reranker loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to load reranker model: {e}")
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        # Convert to lowercase and extract words
        text = text.lower()
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        # Split and filter out short tokens
        tokens = [token for token in text.split() if len(token) > 2]
        return tokens
    
    def _build_bm25_index(self) -> BM25Okapi:
        """Build BM25 index from document corpus."""
        # Tokenize all documents
        corpus_tokens = []
        for meta in tqdm(self.metadata, desc="Tokenizing corpus for BM25"):
            content = meta['content']
            tokens = self._tokenize(content)
            corpus_tokens.append(tokens)
        
        # Create BM25 index
        bm25 = BM25Okapi(corpus_tokens)
        print(f"BM25 index built with {len(corpus_tokens)} documents")
        return bm25
    
    def bm25_search(self, query: str, k: int = 50) -> List[Tuple[int, float]]:
        """
        Perform BM25 search.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document_index, bm25_score) tuples
        """
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top k indices
        top_indices = np.argsort(scores)[-k:][::-1]
        
        # Normalize scores to [0, 1] range
        max_score = scores[top_indices[0]] if len(top_indices) > 0 and scores[top_indices[0]] > 0 else 1.0
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include documents with non-zero scores
                normalized_score = scores[idx] / max_score
                results.append((int(idx), float(normalized_score)))
        
        return results
    
    def semantic_search(self, query: str, k: int = 50) -> List[Tuple[int, float]]:
        """
        Perform semantic search using FAISS.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document_index, similarity_score) tuples
        """
        query_embedding = self.generator.embed(query)
        if query_embedding.ndim == 1:
            query_embedding = np.array([query_embedding])
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        
        similarities, indices = self.index.search(query_embedding, k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx != -1:
                results.append((int(idx), float(similarities[0][i])))
        return results
    
    def reciprocal_rank_fusion(self, 
                               search_results_list: List[List[Tuple[int, float]]], 
                               k: int = 60) -> List[Tuple[int, float]]:
        """
        Combine multiple ranked lists using Reciprocal Rank Fusion (RRF).
        
        RRF is a simple but effective method that doesn't require score normalization.
        Score for document d = sum over all rankings of 1 / (k + rank(d))
        
        Args:
            search_results_list: List of search results from different methods
            k: Constant for RRF formula (typically 60)
            
        Returns:
            Fused ranked list of (document_index, fused_score) tuples
        """
        # Calculate RRF scores
        rrf_scores = {}
        
        for search_results in search_results_list:
            for rank, (doc_id, _) in enumerate(search_results, start=1):
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0.0
                rrf_scores[doc_id] += 1.0 / (k + rank)
        
        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_results
    
    def rerank_with_cross_encoder(self, 
                                   query: str, 
                                   candidates: List[Tuple[int, float]], 
                                   top_k: int = 15) -> List[Dict]:
        """
        Rerank candidates using cross-encoder for final ranking.
        
        Args:
            query: Search query
            candidates: List of (document_index, score) tuples
            top_k: Number of final results to return
            
        Returns:
            List of reranked results with metadata
        """
        if not candidates:
            return []
        
        # Prepare query-document pairs for cross-encoder
        pairs = []
        candidate_indices = []
        
        for doc_idx, _ in candidates[:15]:  # Rerank top 15 candidates
            content = self.metadata[doc_idx]['content']
            # Truncate very long content for cross-encoder
            content_truncated = content[:1000] if len(content) > 1000 else content
            pairs.append([query, content_truncated])
            candidate_indices.append(doc_idx)
        
        # Get cross-encoder scores
        print(f"Reranking {len(pairs)} candidates with cross-encoder...")
        ce_scores = self.reranker.predict(pairs)
        
        # Combine with original scores and create final ranking
        reranked_results = []
        for idx, ce_score in zip(candidate_indices, ce_scores):
            reranked_results.append({
                'doc_idx': idx,
                'ce_score': float(ce_score),
                'metadata': self.metadata[idx]
            })
        
        # Sort by cross-encoder score
        reranked_results.sort(key=lambda x: x['ce_score'], reverse=True)
        
        # Format final results
        final_results = []
        for i, result in enumerate(reranked_results[:top_k]):
            final_results.append({
                "score": result['ce_score'],
                "rank": i + 1,
                "metadata": result['metadata']
            })
        
        return final_results
    
    def hybrid_search(self, 
                     query: str, 
                     k: int = 15,
                     semantic_weight: float = 0.5,
                     bm25_weight: float = 0.5,
                     use_reranker: bool = True) -> List[Dict]:
        """
        Perform hybrid search combining semantic + BM25 with optional reranking.
        
        Pipeline:
        1. Retrieve candidates from semantic search (top 50)
        2. Retrieve candidates from BM25 (top 50)
        3. Fuse results using Reciprocal Rank Fusion
        4. Rerank top candidates with cross-encoder
        
        Args:
            query: Search query
            k: Number of final results
            semantic_weight: Weight for semantic search (not used in RRF)
            bm25_weight: Weight for BM25 (not used in RRF)
            use_reranker: Whether to use cross-encoder reranking
            
        Returns:
            List of top k results with scores and metadata
        """
        print(f"\nðŸ”  Hybrid Search Pipeline:")
        print(f"   Query: '{query}'")
        
        # Step 1: Semantic search
        print(f"   1️⃣ Semantic search (retrieving top 25)...")
        semantic_results = self.semantic_search(query, k=25)
        print(f"      ✓ Retrieved {len(semantic_results)} semantic results")
        
        # Step 2: BM25 search
        print(f"   2️⃣ BM25 keyword search (retrieving top 25)...")
        bm25_results = self.bm25_search(query, k=25)
        print(f"      ✓ Retrieved {len(bm25_results)} BM25 results")
        
        # Step 3: Reciprocal Rank Fusion
        print(f"   3ï¸ âƒ£ Fusing results with Reciprocal Rank Fusion...")
        fused_results = self.reciprocal_rank_fusion(
            [semantic_results, bm25_results],
            k=60
        )
        print(f"      âœ“ Fused to {len(fused_results)} unique documents")
        
        # Step 4: Cross-encoder reranking
        if use_reranker:
            print(f"   4ï¸âƒ£ Reranking with cross-encoder...")
            final_results = self.rerank_with_cross_encoder(query, fused_results, top_k=k)
            print(f"      âœ“ Reranked to top {len(final_results)} results")
        else:
            # Skip reranking, just format results
            final_results = []
            for i, (doc_idx, score) in enumerate(fused_results[:k]):
                final_results.append({
                    "score": score,
                    "rank": i + 1,
                    "metadata": self.metadata[doc_idx]
                })
        
        print(f"   âœ… Final: {len(final_results)} results ready\n")
        return final_results


def create_and_save_faiss_index(embeddings: np.ndarray, metadata: list, 
                                output_path: str = "faiss_index.bin"):
    """Creates and saves a FAISS index optimized for cosine similarity."""
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    
    print(f"Adding {embeddings.shape[0]} vectors to the FAISS index...")
    index.add(embeddings)

    faiss.write_index(index, output_path)
    
    metadata_path = os.path.splitext(output_path)[0] + "_metadata.json"
    
    # Quick fix: Convert booleans when saving
    import json
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_)):
                return bool(obj)
            elif isinstance(obj, (np.integer)):
                return int(obj)
            elif isinstance(obj, (np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)

    print(f"âœ… FAISS index saved to '{output_path}'")
    print(f"âœ… Metadata saved to '{metadata_path}'")

def load_faiss_index(index_path: str = "faiss_index.bin"):
    """Loads a FAISS index and its corresponding metadata."""
    metadata_path = os.path.splitext(index_path)[0] + "_metadata.json"
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(f"FAISS index or metadata not found.")

    print(f"Loading FAISS index from '{index_path}'...")
    index = faiss.read_index(index_path)
    
    print(f"Loading metadata from '{metadata_path}'...")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    return index, metadata


def main():
    """Main function to run the full indexing and testing pipeline."""
    chunks_file_path = "SIH/audio_chunks.pkl"
    try:
        print(f"Loading raw chunks from: {chunks_file_path}")
        with open(chunks_file_path, 'rb') as f:
            raw_chunks = pickle.load(f)
        print(f"Successfully loaded {len(raw_chunks)} chunks.")
    except FileNotFoundError:
        print(f"Error: The file '{chunks_file_path}' was not found.")
        return

    # Generate embeddings with full content
    generator = UnifiedEmbeddingGenerator()
    embeddings_data = generator.process_chunks(raw_chunks)
    
    embeddings = embeddings_data['embeddings']
    metadata = embeddings_data['metadata']

    # Create and save the FAISS index
    faiss_file_path = "document_index.bin"
    create_and_save_faiss_index(embeddings, metadata, faiss_file_path)

    # Load and test BM25 hybrid search with reranking
    try:
        loaded_index, loaded_metadata = load_faiss_index(faiss_file_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Initialize BM25 hybrid search engine with reranker
    print("\n" + "="*80)
    print("Initializing BM25 Hybrid Search Engine with Cross-Encoder Reranker")
    print("="*80)
    hybrid_engine = BM25HybridSearchEngine(loaded_index, loaded_metadata, generator)
    
    # Test query
    query = "explain about the the contents in the complex tables"
    print(f"\n{'='*80}")
    print(f"Testing: '{query}'")
    print("="*80)
    
    search_results = hybrid_engine.hybrid_search(query, k=10, use_reranker=True)

    print(f"\n{'='*80}")
    print(f"TOP 10 RERANKED RESULTS")
    print("="*80)
    for i, result in enumerate(search_results, 1):
        meta = result['metadata']
        print(f"\n{i}. Rank: {result['rank']} | Cross-Encoder Score: {result['score']:.4f}")
        print(f"   Type: {meta['type']}")
        print(f"   Source: {meta['source']} (Page: {meta.get('page_number', 'N/A')})")
        print(f"   Preview: {meta['content_preview']}")

if __name__ == "__main__":
    main()