import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime

def run_dashboard(orchestrator):
    st.title("CAG System Monitoring")
    
    # Cache Performance
    st.header("Cache Performance")
    metrics = orchestrator.telemetry.get_metrics()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Queries", metrics["query_count"])
    col2.metric("Cache Hit Ratio", f"{metrics['cache_hit_ratio']:.2%}")
    col3.metric("Avg Response Time", f"{metrics['avg_response_time']:.2f}ms")
    
    # Cache Source Distribution
    st.subheader("Cache Source Distribution")
    source_data = metrics["cache_source_distribution"]
    fig, ax = plt.subplots()
    ax.pie(source_data.values(), labels=source_data.keys(), autopct='%1.1f%%')
    st.pyplot(fig)
    
    # Feedback Quality
    st.header("Response Quality")
    quality_metrics = orchestrator.feedback_system.get_response_quality_metrics()
    
    if quality_metrics["total_feedback"] > 0:
        st.metric("Average Rating", f"{quality_metrics['average_rating']:.1f}/5")
        
        # Rating distribution
        st.subheader("Rating Distribution")
        rating_data = quality_metrics.get("rating_distribution", {})
        rating_df = pd.DataFrame({
            "Rating": list(rating_data.keys()),
            "Count": list(rating_data.values())
        })
        st.bar_chart(rating_df.set_index("Rating"))
    else:
        st.info("No feedback data available yet")
    
    # Recent queries log
    st.header("Recent Queries")
    # This would need to be implemented in the orchestrator
    if hasattr(orchestrator, "query_log"):
        log_df = pd.DataFrame(orchestrator.query_log[-20:])
        st.dataframe(log_df)
    
    # Auto-refresh every 30 seconds
    st.button("Refresh")
    time.sleep(30)
    st.experimental_rerun()

if __name__ == "__main__":
    # This would connect to your orchestrator instance
    # For example: from your_module import orchestrator
    # run_dashboard(orchestrator)
    st.error("Please import your orchestrator and uncomment the run_dashboard line")