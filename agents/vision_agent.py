import logging
from PIL import Image
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL
from typing import Dict, List

logger = logging.getLogger(__name__)

class VisionAgent:
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(
            GEMINI_MODEL,
            generation_config={
                "temperature": 0.2,
                "top_p": 0.95,
                "max_output_tokens": 1024,
            }
        )
        logger.info("VisionAgent initialized with Gemini Vision")
    
    def describe(self, image_path: str, query_context: str = "") -> Dict:
        try:
            image = Image.open(image_path)
            
            context_prompt = ""
            if query_context:
                context_prompt = f"\n\nUser Query Context: {query_context}\nFocus on aspects relevant to this query."
            
            prompt = f"""Analyze this image comprehensively and provide a detailed description.

Include:
1. **Main Content**: What is the primary subject or focus?
2. **Visual Elements**: Charts, graphs, diagrams, text, tables, or other data visualizations
3. **Data & Numbers**: Any numerical data, statistics, percentages, or measurements visible
4. **Text Content**: All readable text, labels, titles, legends, or annotations
5. **Context & Meaning**: What insights or information does this image convey?
6. **Colors & Layout**: Significant colors, layout structure, or design elements
7. **Quality Assessment**: Image clarity, resolution, and readability

Be specific and thorough. Extract all readable information.{context_prompt}"""

            response = self.model.generate_content([prompt, image])
            description = response.text.strip()
            
            logger.info(f"✓ Generated detailed description for {image_path} ({len(description)} chars)")
            
            return {
                "image_path": image_path,
                "description": description,
                "length": len(description)
            }
            
        except Exception as e:
            logger.error(f"Error describing image {image_path}: {e}")
            return {
                "image_path": image_path,
                "description": f"Error processing image: {str(e)}",
                "length": 0
            }
    
    def batch_describe(self, image_paths: List[str], query_context: str = "") -> List[Dict]:
        descriptions = []
        for img_path in image_paths:
            desc = self.describe(img_path, query_context)
            descriptions.append(desc)
        return descriptions
    
    def analyze_visual_data(self, image_path: str, specific_question: str) -> Dict:
        try:
            image = Image.open(image_path)
            
            prompt = f"""You are analyzing a visual document (chart, graph, table, or diagram).

Specific Question: {specific_question}

Please:
1. Answer the question directly based on the visual data
2. Provide specific numbers, trends, or patterns you observe
3. Cite visual elements that support your answer
4. Note any limitations or uncertainties in the data

Be precise and data-driven in your response."""

            response = self.model.generate_content([prompt, image])
            analysis = response.text.strip()
            
            logger.info(f"✓ Generated targeted analysis for {image_path}")
            
            return {
                "image_path": image_path,
                "question": specific_question,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {e}")
            return {
                "image_path": image_path,
                "question": specific_question,
                "analysis": f"Error: {str(e)}"
            }