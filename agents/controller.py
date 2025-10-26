import logging
from typing import Dict
from agents.intent_agent import IntentAgent
from agents.retrieval_agent import RetrievalAgent
from agents.vision_agent import VisionAgent
from agents.reasoning_agent import ReasoningAgent
from agents.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class Controller:
    def __init__(self):
        logger.info("Initializing Controller with all agents...")
        self.intent = IntentAgent()
        self.retrieval = RetrievalAgent()
        self.vision = VisionAgent()
        self.reasoning = ReasoningAgent()
        self.memory = MemoryManager()
        logger.info("✓ Controller initialized successfully")
    
    def handle_query(self, query: str) -> Dict:
        try:
            logger.info(f"Processing query: {query[:100]}...")
            
            conversation_history = self.memory.get_recent_history()
            
            # Intent agent is disabled for the new pipeline
            intent_data = {"intent": "HYBRID_SEARCH"} # Mock intent data
            
            retrieval_result = self.retrieval.search(query)
            retrieved_chunks = retrieval_result.get("chunks", [])
            
            response = self.reasoning.answer(
                query=query,
                retrieved_chunks=retrieved_chunks,
                conversation_history=conversation_history
            )
            
            combined_metrics = {
                **retrieval_result.get("metrics", {}),
                **response.get("metrics", {}),
                "quality_score": response.get("quality_score", 0.0)
            }
            
            self.memory.add_turn(
                query=query,
                answer=response["answer"],
                intent_data=intent_data, # Still passing mock data
                retrieved_data={"chunks": retrieved_chunks}, # Pass chunks to memory
                metrics=combined_metrics
            )
            
            result = {
                "query": query,
                "answer": response["answer"],
                "intent": intent_data,
                "retrieved_chunks": retrieved_chunks,
                "metrics": combined_metrics,
                "conversation_summary": self.memory.get_conversation_summary()
            }
            
            logger.info(f"✓ Query processed successfully | Quality: {combined_metrics.get('quality_score', 0):.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Critical error in handle_query: {e}", exc_info=True)
            return {
                "query": query,
                "answer": f"I encountered a critical error: {str(e)}. Please try again or contact support.",
                "intent": {"intent": "ERROR", "confidence": 0.0},
                "retrieved_texts": [],
                "retrieved_images": [],
                "image_descriptions": [],
                "metrics": {"error": str(e)},
                "conversation_summary": {}
            }
    
    def reset_conversation(self):
        self.memory.clear_memory()
        logger.info("Conversation reset")
