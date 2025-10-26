import logging
import sys
from pathlib import Path
from agents.retrieval_agent import RetrievalAgent
from config import LOG_LEVEL, LOG_FORMAT, REPORTS_DIR

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler("rag_system.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def print_banner():
    banner = """
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║ 🚀 ADVANCED MULTIMODAL RAG SYSTEM (Layout-Aware)              ║
║    Powered by Gemini & Jina                                    ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def find_first_pdf(directory: Path):
    pdfs = list(directory.glob("*.pdf"))
    if not pdfs:
        logger.error(f"No PDF files found in {directory}")
        return None
    logger.info(f"Found {len(pdfs)} PDF(s). Processing the first one: {pdfs[0].name}")
    return pdfs[0]

def main():
    print_banner()
    logger.info("="*80)
    logger.info("Starting RAG System Index Builder (Layout-Aware Mode)")
    logger.info("="*80)
    
    pdf_to_process = find_first_pdf(REPORTS_DIR)
    
    if not pdf_to_process:
        sys.exit(1)
    
    print("\n🔧 Initializing Retrieval Agent...")
    retrieval = RetrievalAgent()
    
    print(f"\n📄 Processing document: {pdf_to_process.name}")
    print("   This will analyze page layouts, classify content, and build indexes.")
    print("   This may take some time depending on the document size...")
    
    retrieval.build_index_from_pdf(str(pdf_to_process))
    
    print("\n" + "="*70)
    print("✅ INDEX BUILDING COMPLETE!")
    print("="*70)
    print("\n📌 Next Steps:")
    print("   Run: streamlit run streamlit_app.py")
    print("   Then open: http://localhost:8501")
    print("\n" + "="*70)
    
    logger.info("Index building completed successfully")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)
