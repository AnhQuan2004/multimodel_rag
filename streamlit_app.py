import logging
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from agents.controller import Controller
from config import LOG_LEVEL, LOG_FORMAT
import time

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler("streamlit_app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="High-Accuracy Multimodal RAG",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_controller():
    logger.info("Loading controller...")
    controller = Controller()
    return controller

def display_metrics_dashboard(metrics: dict):
    st.subheader("📊 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        quality = metrics.get("quality_score", 0)
        st.metric(
            "Quality Score",
            f"{quality:.1%}",
            delta=None,
            help="Overall response quality based on length, citations, and structure"
        )
    
    with col2:
        confidence = metrics.get("confidence", 0)
        st.metric(
            "Intent Confidence",
            f"{confidence:.1%}",
            delta=None,
            help="Confidence in query classification"
        )
    
    with col3:
        avg_sim = metrics.get("avg_similarity", 0)
        st.metric(
            "Avg Similarity",
            f"{avg_sim:.3f}",
            delta=None,
            help="Average similarity score of retrieved documents"
        )
    
    with col4:
        num_sources = metrics.get("num_sources", 0)
        st.metric(
            "Sources Used",
            num_sources,
            delta=None,
            help="Total number of documents and images used"
        )
    
    with st.expander("🔍 Detailed Metrics"):
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.write("**Retrieval Metrics:**")
            st.write(f"- Text Retrieved: {metrics.get('text_retrieved', 0)}")
            st.write(f"- After Reranking: {metrics.get('text_after_rerank', 0)}")
            st.write(f"- Images Retrieved: {metrics.get('images_retrieved', 0)}")
            st.write(f"- Avg Rerank Score: {metrics.get('avg_rerank_score', 0):.3f}")
        
        with col_b:
            st.write("**Response Metrics:**")
            st.write(f"- Word Count: {metrics.get('word_count', 0)}")
            st.write(f"- Has Citations: {'✓' if metrics.get('has_citations') else '✗'}")
            st.write(f"- Context Length: {metrics.get('context_used', 0)} chars")
            st.write(f"- Intent: {metrics.get('intent', 'Unknown')}")
            st.write(f"- Complexity: {metrics.get('complexity', 'Unknown')}")

def display_conversation_stats(conv_summary: dict):
    if conv_summary.get("total_turns", 0) > 0:
        st.sidebar.subheader("💬 Conversation Stats")
        st.sidebar.metric("Total Queries", conv_summary["total_turns"])
        st.sidebar.metric("Avg Confidence", f"{conv_summary.get('avg_confidence', 0):.1%}")
        
        if conv_summary.get("intent_distribution"):
            st.sidebar.write("**Intent Distribution:**")
            for intent, count in conv_summary["intent_distribution"].items():
                st.sidebar.write(f"  • {intent}: {count}")

def create_quality_gauge(quality_score: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=quality_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Response Quality", 'font': {'size': 20}},
        delta={'reference': 80, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffebee'},
                {'range': [50, 75], 'color': '#fff9c4'},
                {'range': [75, 100], 'color': '#c8e6c9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def main():
    st.title("🎯 High-Accuracy Multimodal RAG System")
    st.caption("Powered by Gemini 2.0 Flash with Advanced Retrieval & Reranking")
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        show_metrics = st.checkbox("Show Performance Metrics", value=True)
        show_sources = st.checkbox("Show Retrieved Sources", value=True)
        show_images = st.checkbox("Show Image Analysis", value=True)
        
        st.divider()
        
        if st.button("🔄 Reset Conversation", use_container_width=True):
            if 'controller' in st.session_state:
                st.session_state.controller.reset_conversation()
                st.success("Conversation reset!")
                st.rerun()
        
        st.divider()
        st.caption("System Status: 🟢 Online")
    
    try:
        if 'controller' not in st.session_state:
            with st.spinner("🔧 Loading system components..."):
                st.session_state.controller = load_controller()
            st.success("✅ System loaded successfully!")
        
        controller = st.session_state.controller
        
        if hasattr(controller, 'memory'):
            display_conversation_stats(controller.memory.get_conversation_summary())
        
    except Exception as e:
        st.error(f"❌ Failed to load system: {e}")
        st.info("💡 Please run 'python main.py' first to build indexes.")
        st.stop()
    
    st.divider()
    
    query = st.text_area(
        "🔍 Enter your question:",
        height=120,
        placeholder="Ask anything about your documents and images...\n\nExamples:\n• What are the key findings in the report?\n• Compare the data from Q1 and Q2\n• Explain the chart showing revenue trends",
        help="Enter detailed questions for better results"
    )
    
    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
    
    with col_btn1:
        search_button = st.button("🚀 Search & Analyze", type="primary", use_container_width=True)
    
    if search_button and query:
        start_time = time.time()
        
        with st.spinner("🔄 Processing your query..."):
            progress_bar = st.progress(0)
            
            progress_bar.progress(20)
            st.caption("Analyzing intent...")
            
            result = controller.handle_query(query)
            
            progress_bar.progress(100)
            st.caption("✓ Complete!")
            time.sleep(0.3)
            progress_bar.empty()
        
        processing_time = time.time() - start_time
        
        if show_metrics:
            display_metrics_dashboard(result["metrics"])
            
            col_gauge1, col_gauge2 = st.columns(2)
            with col_gauge1:
                st.plotly_chart(
                    create_quality_gauge(result["metrics"].get("quality_score", 0)),
                    use_container_width=True
                )
            with col_gauge2:
                intent_data = result.get("intent", {})
                st.info(f"""
**Query Analysis:**
- **Intent**: {intent_data.get('intent', 'Unknown')}
- **Confidence**: {intent_data.get('confidence', 0):.1%}
- **Complexity**: {intent_data.get('complexity', 'Unknown')}
- **Reasoning**: {intent_data.get('reasoning', 'N/A')}
- **Processing Time**: {processing_time:.2f}s
                """)
        
        st.divider()
        
        st.subheader("💡 Answer")
        st.markdown(result["answer"])
        
        st.divider()
        
        if show_sources and result.get("retrieved_texts"):
            with st.expander(f"📄 Retrieved Documents ({len(result['retrieved_texts'])})", expanded=False):
                for i, doc in enumerate(result["retrieved_texts"], 1):
                    st.markdown(f"### Document {i}")
                    
                    col_info1, col_info2, col_info3 = st.columns(3)
                    with col_info1:
                        st.write(f"**Source:** {doc.get('source', 'Unknown')}")
                    with col_info2:
                        st.write(f"**Page:** {doc.get('page', 'N/A')}")
                    with col_info3:
                        st.write(f"**Score:** {doc.get('rerank_score', 0):.3f}")
                    
                    with st.container(border=True):
                        st.text(doc.get("text", "")[:800] + ("..." if len(doc.get("text", "")) > 800 else ""))
                    
                    st.divider()
        
        if show_images and result.get("retrieved_images"):
            with st.expander(f"🖼️ Retrieved Images ({len(result['retrieved_images'])})", expanded=False):
                cols = st.columns(min(3, len(result["retrieved_images"])))
                for i, img in enumerate(result["retrieved_images"]):
                    with cols[i % 3]:
                        st.image(
                            img["path"],
                            caption=f"{img['name']} (Score: {img.get('similarity_score', 0):.3f})",
                            use_container_width=True
                        )
        
        if show_images and result.get("image_descriptions"):
            with st.expander("🔍 Detailed Image Analysis", expanded=False):
                for i, desc in enumerate(result["image_descriptions"], 1):
                    st.markdown(f"### Image {i}: {desc.get('image_path', 'Unknown').split('/')[-1]}")
                    
                    if desc.get("question"):
                        st.info(f"**Question:** {desc['question']}")
                        st.markdown(f"**Analysis:**\n\n{desc.get('analysis', '')}")
                    else:
                        st.markdown(desc.get("description", ""))
                    
                    st.divider()
    
    elif search_button:
        st.warning("⚠️ Please enter a question first!")

if __name__ == "__main__":
    main()
