import logging
import base64
import os
import google.generativeai as genai
from typing import Dict, List
from config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

class ReasoningAgent:
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        logger.info(f"ReasoningAgent initialized with {GEMINI_MODEL}")
    
    def answer(self, query: str, retrieved_chunks: List[Dict], conversation_history: List[Dict]) -> Dict:
        try:
            logger.info(f"Synthesizing final answer based on {len(retrieved_chunks)} retrieved chunks.")

            prompt_parts = [
                "You are an expert AI assistant. Based ONLY on the following context (text chunks and images), answer the user's question.",
                "Provide citations like [doc_id, page X, chunk Y] for text and [Image from page X] for images.",
                "---"
            ]

            # Keep track of pages we've already added the image for
            pages_with_images_added = set()

            for chunk in retrieved_chunks:
                page_index = chunk.get('page_index', 'N/A')
                chunk_index = chunk.get('chunk_index', 'N/A')
                doc_id = chunk.get('doc_id', 'unknown')

                # Add the text part of the chunk
                prompt_parts.append(f"Context from [doc_id: {doc_id}, page {page_index}, chunk {chunk_index}]:")
                prompt_parts.append(chunk.get('text', ''))

                # If the chunk is from a visual page, add the corresponding image (once per page)
                if chunk.get('is_visual_content') and page_index not in pages_with_images_added:
                    image_path = chunk.get('image_path')
                    if image_path and os.path.exists(image_path):
                        logger.info(f"Adding image context from page {page_index} ({image_path})")
                        prompt_parts.append(f"[Image from page {page_index}]:")
                        with open(image_path, "rb") as f:
                            img_bytes = f.read()
                            image_part = {"mime_type": "image/jpeg", "data": img_bytes}
                            prompt_parts.append(image_part)
                        pages_with_images_added.add(page_index)
            
            prompt_parts.append("---")
            prompt_parts.append(f"User Question: {query}")
            prompt_parts.append("Your Detailed Answer:")

            response = self.model.generate_content(prompt_parts)
            answer = response.text.strip()
            
            if not answer:
                answer = "I could not find a relevant answer in the provided context."

            # Extract usage metadata for cost calculation
            usage = response.usage_metadata
            metrics = {
                "gemini_input_tokens": usage.prompt_token_count,
                "gemini_output_tokens": usage.candidates_token_count,
                "gemini_total_tokens": usage.total_token_count
            }
            logger.info(f"Gemini API usage: {metrics}")

            return {"answer": answer, "metrics": metrics, "quality_score": 1.0}
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}", exc_info=True)
            return {"answer": f"Error during answer generation: {e}", "metrics": {}, "quality_score": 0.0}
