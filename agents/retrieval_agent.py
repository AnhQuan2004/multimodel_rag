import os
import logging
import numpy as np
import requests
import base64
import fitz  # PyMuPDF
import uuid
import datetime
import google.generativeai as genai
from qdrant_client import QdrantClient, models
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import (
    REPORTS_DIR, IMAGES_DIR, SIMILARITY_THRESHOLD, JINA_API_KEY, GEMINI_API_KEY,
    TOP_K_RETRIEVAL, QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME,
    CHUNK_SIZE, MIN_CHUNK_SIZE, LAYOUT_ANALYSIS_MODEL
)

logger = logging.getLogger(__name__)

class RetrievalAgent:
    def __init__(self):
        logger.info("Initializing RetrievalAgent for Hybrid Pipeline...")
        
        if not JINA_API_KEY:
            raise ValueError("JINA_API_KEY is not set.")
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set.")
            
        genai.configure(api_key=GEMINI_API_KEY)
            
        self.jina_api_url = 'https://api.jina.ai/v1/embeddings'
        self.jina_headers = {
          'Content-Type': 'application/json',
          'Authorization': f'Bearer {JINA_API_KEY}'
        }
        
        self.layout_model = genai.GenerativeModel(LAYOUT_ANALYSIS_MODEL)
        
        if QDRANT_URL == ":memory:":
            self.qdrant_client = QdrantClient(location=":memory:")
        else:
            self.qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        logger.info("RetrievalAgent initialized successfully.")

    def _ensure_collection_exists(self, vector_size: int, collection_name: str):
        try:
            self.qdrant_client.get_collection(collection_name=collection_name)
            logger.info(f"Collection '{collection_name}' already exists.")
        except Exception:
            logger.info(f"Collection '{collection_name}' not found. Recreating collection.")
            self.qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )

    def _get_jina_embeddings(self, texts: List[str]) -> Tuple[List[np.ndarray], int]:
        if not texts:
            return [], 0
        inputs = [{"text": text} for text in texts]
        data = {
          'input': inputs,
          'model': 'jina-embeddings-v2-base-en',
          'encoding_type': 'float'
        }
        try:
            response = requests.post(self.jina_api_url, headers=self.jina_headers, json=data)
            response.raise_for_status()
            json_response = response.json()
            embeddings = [np.array(item['embedding']) for item in json_response['data']]
            tokens_used = json_response.get('usage', {}).get('total_tokens', 0)
            return embeddings, tokens_used
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Jina API: {e}")
            return [], 0

    def _classify_page_layout(self, page_image_path: str) -> bool:
        try:
            with open(page_image_path, "rb") as f:
                img_bytes = f.read()
            image_part = {"mime_type": "image/jpeg", "data": img_bytes}
            prompt = "Does this page contain any chart, graph, or image besides plain text? Reply with YES or NO only."
            response = self.layout_model.generate_content([prompt, image_part])
            result = response.text.strip().upper()
            logger.info(f"Layout analysis result for {os.path.basename(page_image_path)}: {result}")
            return "YES" in result
        except Exception as e:
            logger.error(f"Error during layout analysis for {os.path.basename(page_image_path)}: {e}")
            return False

    def build_index_from_pdf(self, pdf_path: str):
        pdf_filename = os.path.basename(pdf_path)
        doc_id = str(uuid.uuid4())
        logger.info(f"Processing PDF '{pdf_filename}' with doc_id '{doc_id}'")
        doc = fitz.open(pdf_path)
        
        all_chunks_with_metadata = []
        
        images_dir = os.path.join(REPORTS_DIR, "..", "images")
        os.makedirs(images_dir, exist_ok=True)

        for page_num in range(len(doc)):
            page_index = page_num + 1
            logger.info(f"Extracting content from page {page_index}/{len(doc)}...")
            page = doc.load_page(page_num)
            
            image_path = os.path.join(images_dir, f"{doc_id}_page_{page_index}.jpg")
            pix = page.get_pixmap(dpi=150)
            pix.save(image_path)
            
            is_visual_content = self._classify_page_layout(image_path)
            
            blocks = []
            source_type = "native"
            if is_visual_content:
                try:
                    tp = page.get_textpage_ocr(flags=3, full=True)
                    blocks = tp.extractBLOCKS()
                    source_type = "ocr"
                except Exception as e:
                    logger.error(f"OCR failed on page {page_index}, falling back. Error: {e}")
                    blocks = page.get_text("blocks")
            else:
                blocks = page.get_text("blocks")

            chunk_index = 0
            for b in blocks:
                block_text = b[4]
                if len(block_text) < MIN_CHUNK_SIZE:
                    continue
                
                sub_chunks = [block_text[i:i + CHUNK_SIZE] for i in range(0, len(block_text), CHUNK_SIZE)]
                for sub_chunk_text in sub_chunks:
                    if len(sub_chunk_text) < MIN_CHUNK_SIZE:
                        continue

                    metadata = {
                        "doc_id": doc_id, "page_index": page_index, "chunk_index": chunk_index,
                        "text": sub_chunk_text, "is_visual_content": is_visual_content,
                        "image_path": image_path if is_visual_content else "",
                        "source_type": source_type, "created_at": datetime.datetime.utcnow().isoformat()
                    }
                    all_chunks_with_metadata.append(metadata)
                    chunk_index += 1
        doc.close()
        
        if not all_chunks_with_metadata:
            logger.warning("No text chunks generated from the document.")
            return

        logger.info(f"Starting parallel embedding for {len(all_chunks_with_metadata)} chunks...")
        
        total_jina_tokens = 0
        all_embeddings = [None] * len(all_chunks_with_metadata)
        
        batch_size = 128
        with ThreadPoolExecutor() as executor:
            future_to_batch_index = {}
            for i in range(0, len(all_chunks_with_metadata), batch_size):
                batch_metadata = all_chunks_with_metadata[i:i + batch_size]
                batch_texts = [m["text"] for m in batch_metadata]
                future = executor.submit(self._get_jina_embeddings, batch_texts)
                future_to_batch_index[future] = i

            for future in as_completed(future_to_batch_index):
                start_index = future_to_batch_index[future]
                embeddings, tokens_used = future.result()
                total_jina_tokens += tokens_used
                end_index = start_index + len(embeddings)
                all_embeddings[start_index:end_index] = embeddings

        if any(e is None for e in all_embeddings):
            logger.error("Mismatch between chunks and embeddings count. Aborting.")
            return
            
        logger.info(f"Parallel embedding complete. Total Jina tokens used: {total_jina_tokens}")
        vector_size = len(all_embeddings[0])
        self._ensure_collection_exists(vector_size, QDRANT_COLLECTION_NAME)
        
        qdrant_points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload=p
            )
            for p, embedding in zip(all_chunks_with_metadata, all_embeddings)
        ]
        
        upsert_batch_size = 100
        for i in range(0, len(qdrant_points), upsert_batch_size):
            batch = qdrant_points[i:i + upsert_batch_size]
            self.qdrant_client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=batch, wait=True)
            logger.info(f"Upserted batch {i//upsert_batch_size + 1} to Qdrant.")
        
        logger.info(f"✓ Finished indexing {len(qdrant_points)} chunks.")

    def search(self, query: str) -> Dict:
        results_payload = {"chunks": [], "metrics": {}}
        try:
            self.qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        except Exception:
            logger.error(f"Qdrant collection not found. Please run main.py first.")
            return results_payload

        query_embeddings, tokens_used = self._get_jina_embeddings([query])
        results_payload["metrics"]["jina_query_tokens"] = tokens_used
        
        if not query_embeddings:
            return results_payload
            
        search_result = self.qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_embeddings[0].tolist(),
            limit=TOP_K_RETRIEVAL,
            score_threshold=SIMILARITY_THRESHOLD
        )
        
        results_payload["chunks"] = [point.payload for point in search_result]
        logger.info(f"Retrieved {len(results_payload['chunks'])} chunks from Qdrant.")
        return results_payload
