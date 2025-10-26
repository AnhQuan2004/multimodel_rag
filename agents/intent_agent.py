import json
import logging
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

class IntentAgent:
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(
            GEMINI_MODEL,
            generation_config={
                "temperature": 0.1,
                "top_p": 0.95,
                "max_output_tokens": 256,
            }
        )
        logger.info("IntentAgent initialized with optimized config")
    
    def predict(self, query: str, conversation_history: list = None) -> dict:
        try:
            history_context = ""
            if conversation_history:
                recent = conversation_history[-3:]
                history_context = "\n".join([f"User: {h['query']}\nIntent: {h.get('intent', 'unknown')}" for h in recent])
            
            prompt = f"""You are an expert intent classifier for a document retrieval system. Analyze the query considering conversation history.

Intent Categories:
1. FACT_LOOKUP: Simple factual questions, definitions, specific data points
   Examples: "What is X?", "When did Y happen?", "Define Z"

2. COMPARATIVE_ANALYSIS: Comparing multiple entities, pros/cons, differences
   Examples: "Compare X and Y", "What are differences between A and B?"

3. SUMMARIZATION: Requesting summaries, overviews, key points
   Examples: "Summarize the report", "What are the main findings?"

4. VISUAL_QUERY: Questions about charts, graphs, images, visual data
   Examples: "Show me the chart for X", "What does the graph indicate?"

5. COMPLEX_REASONING: Multi-step reasoning, calculations, inference
   Examples: "Based on X and Y, what can we conclude?", "Calculate the impact of Z"

6. PROCEDURAL: How-to questions, step-by-step processes
   Examples: "How to do X?", "What are the steps for Y?"

Recent Conversation History:
{history_context}

Current Query: {query}

Analyze and return JSON with:
{{
    "intent": "FACT_LOOKUP|COMPARATIVE_ANALYSIS|SUMMARIZATION|VISUAL_QUERY|COMPLEX_REASONING|PROCEDURAL",
    "confidence": 0.XX,
    "reasoning": "Brief explanation of classification",
    "requires_images": true/false,
    "complexity": "low|medium|high"
}}"""

            response = self.model.generate_content(prompt)
            text = response.text.strip()
            
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            data = json.loads(text)
            
            result = {
                "intent": data.get("intent", "FACT_LOOKUP"),
                "confidence": float(data.get("confidence", 0.8)),
                "reasoning": data.get("reasoning", ""),
                "requires_images": data.get("requires_images", False),
                "complexity": data.get("complexity", "medium")
            }
            
            logger.info(f"Intent: {result['intent']} | Confidence: {result['confidence']:.2f} | Complexity: {result['complexity']}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return {
                "intent": "FACT_LOOKUP",
                "confidence": 0.6,
                "reasoning": "Fallback classification",
                "requires_images": False,
                "complexity": "medium"
            }
        except Exception as e:
            logger.error(f"Intent prediction error: {e}")
            return {
                "intent": "FACT_LOOKUP",
                "confidence": 0.5,
                "reasoning": "Error in classification",
                "requires_images": False,
                "complexity": "medium"
            }