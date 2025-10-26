import json
import logging
from datetime import datetime
from typing import List, Dict
from config import MEMORY_PATH, MAX_MEMORY_TURNS

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self):
        self.memory = []
        self.load_memory()
        logger.info("MemoryManager initialized")
    
    def load_memory(self):
        try:
            if MEMORY_PATH.exists():
                with open(MEMORY_PATH, 'r', encoding='utf-8') as f:
                    self.memory = json.load(f)
                logger.info(f"Loaded {len(self.memory)} conversation turns from memory")
        except Exception as e:
            logger.error(f"Error loading memory: {e}")
            self.memory = []
    
    def save_memory(self):
        try:
            with open(MEMORY_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.memory[-MAX_MEMORY_TURNS:], f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.memory)} turns to memory")
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
    
    def add_turn(self, query: str, answer: str, intent_data: Dict, retrieved_data: Dict, metrics: Dict):
        turn = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": answer[:500],
            "intent": intent_data.get("intent", "unknown"),
            "confidence": intent_data.get("confidence", 0),
            "num_texts": len(retrieved_data.get("texts", [])),
            "num_images": len(retrieved_data.get("images", [])),
            "metrics": metrics
        }
        self.memory.append(turn)
        
        if len(self.memory) > MAX_MEMORY_TURNS * 2:
            self.memory = self.memory[-MAX_MEMORY_TURNS:]
        
        self.save_memory()
    
    def get_recent_history(self, n: int = MAX_MEMORY_TURNS) -> List[Dict]:
        return self.memory[-n:] if self.memory else []
    
    def get_conversation_summary(self) -> Dict:
        if not self.memory:
            return {"total_turns": 0}
        
        intents = [turn.get("intent", "unknown") for turn in self.memory]
        intent_counts = {}
        for intent in intents:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        avg_confidence = sum(turn.get("confidence", 0) for turn in self.memory) / len(self.memory)
        
        return {
            "total_turns": len(self.memory),
            "intent_distribution": intent_counts,
            "avg_confidence": avg_confidence,
            "first_query": self.memory[0]["query"] if self.memory else "",
            "last_query": self.memory[-1]["query"] if self.memory else ""
        }
    
    def clear_memory(self):
        self.memory = []
        self.save_memory()
        logger.info("Memory cleared")