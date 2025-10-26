import logging
import json
from pathlib import Path
from typing import List, Dict
from agents.controller import Controller
import numpy as np
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

class RAGEvaluator:
    def __init__(self):
        self.controller = Controller()
        self.controller.retrieval.load_indexes()
        self.results = []
        logger.info("RAG Evaluator initialized")
    
    def create_test_queries(self) -> List[Dict]:
        return [
            {
                "query": "What are the main findings in the report?",
                "expected_intent": "SUMMARIZATION",
                "category": "general"
            },
            {
                "query": "Compare the performance metrics between Q1 and Q2",
                "expected_intent": "COMPARATIVE_ANALYSIS",
                "category": "comparative"
            },
            {
                "query": "What does the revenue chart show?",
                "expected_intent": "VISUAL_QUERY",
                "category": "visual"
            },
            {
                "query": "Define machine learning",
                "expected_intent": "FACT_LOOKUP",
                "category": "factual"
            },
            {
                "query": "Based on the trends, what can we predict for next quarter?",
                "expected_intent": "COMPLEX_REASONING",
                "category": "reasoning"
            },
            {
                "query": "How do I implement the recommended strategy?",
                "expected_intent": "PROCEDURAL",
                "category": "procedural"
            }
        ]
    
    def evaluate_query(self, test_case: Dict) -> Dict:
        logger.info(f"Evaluating: {test_case['query']}")
        
        result = self.controller.handle_query(test_case["query"])
        
        intent_correct = result["intent"]["intent"] == test_case["expected_intent"]
        has_answer = len(result["answer"]) > 100
        has_sources = len(result["retrieved_texts"]) > 0
        has_citations = "[Document" in result["answer"] or "[Page" in result["answer"]
        quality_score = result["metrics"].get("quality_score", 0)
        
        evaluation = {
            "query": test_case["query"],
            "category": test_case["category"],
            "expected_intent": test_case["expected_intent"],
            "predicted_intent": result["intent"]["intent"],
            "intent_confidence": result["intent"]["confidence"],
            "intent_correct": intent_correct,
            "has_answer": has_answer,
            "has_sources": has_sources,
            "has_citations": has_citations,
            "quality_score": quality_score,
            "num_sources": len(result["retrieved_texts"]),
            "num_images": len(result["retrieved_images"]),
            "answer_length": len(result["answer"]),
            "word_count": len(result["answer"].split()),
            "avg_similarity": result["metrics"].get("avg_similarity", 0),
            "avg_rerank_score": result["metrics"].get("avg_rerank_score", 0),
            "answer_preview": result["answer"][:200] + "..."
        }
        
        return evaluation
    
    def run_evaluation(self):
        logger.info("Starting comprehensive evaluation...")
        
        test_queries = self.create_test_queries()
        
        for i, test_case in enumerate(test_queries, 1):
            logger.info(f"\nTest {i}/{len(test_queries)}")
            logger.info("="*80)
            
            evaluation = self.evaluate_query(test_case)
            self.results.append(evaluation)
            
            logger.info(f"✓ Intent Correct: {evaluation['intent_correct']}")
            logger.info(f"✓ Quality Score: {evaluation['quality_score']:.2%}")
            logger.info(f"✓ Sources: {evaluation['num_sources']}")
        
        self.generate_report()
    
    def generate_report(self):
        logger.info("\n" + "="*80)
        logger.info("EVALUATION REPORT")
        logger.info("="*80)
        
        intent_accuracy = sum(r["intent_correct"] for r in self.results) / len(self.results)
        avg_quality = np.mean([r["quality_score"] for r in self.results])
        avg_confidence = np.mean([r["intent_confidence"] for r in self.results])
        avg_sources = np.mean([r["num_sources"] for r in self.results])
        avg_word_count = np.mean([r["word_count"] for r in self.results])
        citation_rate = sum(r["has_citations"] for r in self.results) / len(self.results)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(self.results),
            "metrics": {
                "intent_accuracy": float(intent_accuracy),
                "avg_quality_score": float(avg_quality),
                "avg_confidence": float(avg_confidence),
                "avg_sources_used": float(avg_sources),
                "avg_word_count": float(avg_word_count),
                "citation_rate": float(citation_rate)
            },
            "by_category": {},
            "detailed_results": self.results
        }
        
        for result in self.results:
            category = result["category"]
            if category not in report["by_category"]:
                report["by_category"][category] = []
            report["by_category"][category].append(result)
        
        logger.info(f"\n📊 Overall Metrics:")
        logger.info(f"  • Intent Accuracy: {intent_accuracy:.1%}")
        logger.info(f"  • Avg Quality Score: {avg_quality:.1%}")
        logger.info(f"  • Avg Confidence: {avg_confidence:.1%}")
        logger.info(f"  • Avg Sources Used: {avg_sources:.1f}")
        logger.info(f"  • Avg Word Count: {avg_word_count:.0f}")
        logger.info(f"  • Citation Rate: {citation_rate:.1%}")
        
        logger.info(f"\n📁 By Category:")
        for category, results in report["by_category"].items():
            cat_quality = np.mean([r["quality_score"] for r in results])
            cat_intent = sum(r["intent_correct"] for r in results) / len(results)
            logger.info(f"  • {category.upper()}")
            logger.info(f"    - Quality: {cat_quality:.1%}")
            logger.info(f"    - Intent Accuracy: {cat_intent:.1%}")
        
        report_path = Path("evaluation_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✅ Full report saved to: {report_path}")
        
        if avg_quality >= 0.8:
            logger.info("\n🎉 EXCELLENT: System performing at high quality!")
        elif avg_quality >= 0.6:
            logger.info("\n✓ GOOD: System performing adequately")
        else:
            logger.info("\n⚠️ NEEDS IMPROVEMENT: Consider adjusting parameters")
        
        return report

def main():
    print("""
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║     📊 RAG SYSTEM EVALUATION                                   ║
║     Testing accuracy, quality, and performance                 ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
""")
    
    evaluator = RAGEvaluator()
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()