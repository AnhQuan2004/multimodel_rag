import logging
import time
import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self):
        self.metrics = []
        self.start_time = None
    
    def start(self):
        self.start_time = time.time()
    
    def end(self, operation: str, metadata: Dict = None):
        if self.start_time:
            duration = time.time() - self.start_time
            metric = {
                "operation": operation,
                "duration": duration,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }
            self.metrics.append(metric)
            logger.info(f"⏱️ {operation}: {duration:.2f}s")
            self.start_time = None
            return duration
    
    def get_stats(self) -> Dict:
        if not self.metrics:
            return {}
        
        durations = [m["duration"] for m in self.metrics]
        return {
            "total_operations": len(self.metrics),
            "avg_duration": np.mean(durations),
            "min_duration": np.min(durations),
            "max_duration": np.max(durations),
            "total_time": np.sum(durations)
        }
    
    def save_report(self, filename: str = "performance_report.json"):
        report = {
            "timestamp": datetime.now().isoformat(),
            "statistics": self.get_stats(),
            "detailed_metrics": self.metrics
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance report saved to {filename}")

class BatchProcessor:
    def __init__(self, controller):
        self.controller = controller
        self.monitor = PerformanceMonitor()
    
    def process_queries(self, queries: List[str], output_file: str = "batch_results.json"):
        results = []
        
        logger.info(f"Processing {len(queries)} queries in batch...")
        
        for i, query in enumerate(queries, 1):
            logger.info(f"\nProcessing {i}/{len(queries)}: {query[:50]}...")
            
            self.monitor.start()
            result = self.controller.handle_query(query)
            duration = self.monitor.end("query_processing", {"query_index": i})
            
            results.append({
                "query": query,
                "answer": result["answer"],
                "intent": result["intent"]["intent"],
                "confidence": result["intent"]["confidence"],
                "quality_score": result["metrics"].get("quality_score", 0),
                "processing_time": duration,
                "num_sources": len(result["retrieved_texts"])
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✅ Batch processing complete. Results saved to {output_file}")
        
        stats = self.monitor.get_stats()
        logger.info(f"\n📊 Batch Statistics:")
        logger.info(f"  • Total Queries: {stats['total_operations']}")
        logger.info(f"  • Avg Time: {stats['avg_duration']:.2f}s")
        logger.info(f"  • Min Time: {stats['min_duration']:.2f}s")
        logger.info(f"  • Max Time: {stats['max_duration']:.2f}s")
        logger.info(f"  • Total Time: {stats['total_time']:.2f}s")
        
        return results

class DocumentAnalyzer:
    @staticmethod
    def analyze_corpus(reports_dir: Path, images_dir: Path) -> Dict:
        pdf_files = list(reports_dir.glob("*.pdf"))
        image_files = [f for f in images_dir.glob("*") 
                      if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
        
        pdf_sizes = [f.stat().st_size / 1024 / 1024 for f in pdf_files]
        image_sizes = [f.stat().st_size / 1024 / 1024 for f in image_files]
        
        analysis = {
            "documents": {
                "count": len(pdf_files),
                "total_size_mb": sum(pdf_sizes),
                "avg_size_mb": np.mean(pdf_sizes) if pdf_sizes else 0,
                "files": [f.name for f in pdf_files]
            },
            "images": {
                "count": len(image_files),
                "total_size_mb": sum(image_sizes),
                "avg_size_mb": np.mean(image_sizes) if image_sizes else 0,
                "files": [f.name for f in image_files]
            },
            "total_corpus_size_mb": sum(pdf_sizes) + sum(image_sizes)
        }
        
        return analysis
    
    @staticmethod
    def print_corpus_info(analysis: Dict):
        print("\n" + "="*70)
        print("📚 CORPUS ANALYSIS")
        print("="*70)
        print(f"\n📄 Documents:")
        print(f"  • Count: {analysis['documents']['count']}")
        print(f"  • Total Size: {analysis['documents']['total_size_mb']:.2f} MB")
        print(f"  • Avg Size: {analysis['documents']['avg_size_mb']:.2f} MB")
        
        print(f"\n🖼️ Images:")
        print(f"  • Count: {analysis['images']['count']}")
        print(f"  • Total Size: {analysis['images']['total_size_mb']:.2f} MB")
        print(f"  • Avg Size: {analysis['images']['avg_size_mb']:.2f} MB")
        
        print(f"\n💾 Total Corpus Size: {analysis['total_corpus_size_mb']:.2f} MB")
        print("="*70)

def format_metrics_table(metrics: Dict) -> str:
    lines = [
        "\n╔══════════════════════════════════════════════════╗",
        "║              PERFORMANCE METRICS                 ║",
        "╠══════════════════════════════════════════════════╣"
    ]
    
    for key, value in metrics.items():
        if isinstance(value, float):
            if value < 1:
                formatted = f"{value:.2%}"
            else:
                formatted = f"{value:.2f}"
        else:
            formatted = str(value)
        
        label = key.replace("_", " ").title()
        line = f"║  {label:<30} {formatted:>15}  ║"
        lines.append(line)
    
    lines.append("╚══════════════════════════════════════════════════╝")
    
    return "\n".join(lines)

def export_conversation_to_markdown(memory_path: Path, output_file: str = "conversation_export.md"):
    with open(memory_path, 'r', encoding='utf-8') as f:
        memory = json.load(f)
    
    md_lines = [
        "# Conversation History Export",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n**Total Queries:** {len(memory)}",
        "\n---\n"
    ]
    
    for i, turn in enumerate(memory, 1):
        md_lines.append(f"\n## Query {i}")
        md_lines.append(f"\n**Timestamp:** {turn.get('timestamp', 'N/A')}")
        md_lines.append(f"\n**Intent:** {turn.get('intent', 'Unknown')} ({turn.get('confidence', 0):.1%} confidence)")
        md_lines.append(f"\n**Question:**\n> {turn.get('query', '')}")
        md_lines.append(f"\n**Answer:**\n{turn.get('answer', '')}")
        md_lines.append(f"\n**Metrics:**")
        md_lines.append(f"- Sources Used: {turn.get('num_sources', 0)}")
        md_lines.append(f"- Quality Score: {turn.get('metrics', {}).get('quality_score', 0):.1%}")
        md_lines.append("\n---\n")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    logger.info(f"Conversation exported to {output_file}")

if __name__ == "__main__":
    from config import REPORTS_DIR, IMAGES_DIR, MEMORY_PATH
    
    analyzer = DocumentAnalyzer()
    analysis = analyzer.analyze_corpus(REPORTS_DIR, IMAGES_DIR)
    analyzer.print_corpus_info(analysis)
    
    if MEMORY_PATH.exists():
        export_conversation_to_markdown(MEMORY_PATH)