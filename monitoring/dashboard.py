# monitoring/dashboard.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time # Keep for potential future use, but remove loop
from datetime import datetime

def run_dashboard(orchestrator): # Accept orchestrator instance
    st.set_page_config(layout="wide") # Optional: Use wide layout
    st.title("🧠 CAG System Monitoring Dashboard")

    # --- Key Metrics Row ---
    st.header("📊 Key Performance Indicators")
    metrics = orchestrator.telemetry.get_metrics()
    feedback_metrics = orchestrator.feedback_system.get_response_quality_metrics()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Queries", metrics.get("query_count", 0))

    # Calculate actual hit ratio from orchestrator stats if available
    orch_stats = orchestrator.get_stats() # Get combined stats
    actual_hit_ratio = orch_stats.get('overall_performance', {}).get('actual_response_hit_ratio', 0)
    col2.metric("Actual Hit Ratio", f"{actual_hit_ratio:.2%}")

    col3.metric("Avg Response Time", f"{metrics.get('avg_response_time_ms', 0):.2f} ms")
    avg_rating = feedback_metrics.get('average_rating', 0)
    total_fb = feedback_metrics.get('total_feedback', 0)
    col4.metric("Avg Feedback Rating", f"{avg_rating:.1f}/5 ({total_fb} ratings)")

    st.divider()

    # --- Cache Performance ---
    col_cache1, col_cache2 = st.columns([1, 2]) # Adjust column ratios as needed

    with col_cache1:
        st.header("💾 Cache Performance")
        cache_stats = orch_stats.get('cache_performance', {})
        st.metric("L1 Size", cache_stats.get('l1_size', 0))
        st.metric("L2 Size", cache_stats.get('l2_size', 0))
        st.metric("L3 Approx. Size", cache_stats.get('l3_approx_size', 'N/A'))
        st.metric("L1 Hits", cache_stats.get('hits_L1', 0))
        st.metric("L2 Hits", cache_stats.get('hits_L2', 0))
        st.metric("L3 Hits (Initial)", cache_stats.get('hits_L3', 0)) # Hits found by TieredCache
        st.metric("Total Misses", cache_stats.get('misses', 0))
        st.metric("Cache Lookup Hit Ratio", f"{cache_stats.get('overall_hit_ratio', 0):.2%}")


    with col_cache2:
        st.header("📈 Response Source Distribution")
        response_sources = orch_stats.get('response_source_distribution', {})
        # Filter out zero values for a cleaner pie chart
        source_data = {k: v for k, v in response_sources.items() if v > 0}
        if source_data:
            fig, ax = plt.subplots()
            ax.pie(source_data.values(), labels=source_data.keys(), autopct='%1.1f%%', startangle=90)
            ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig)
        else:
            st.info("No responses logged yet.")

    st.divider()

    # --- Feedback Quality ---
    col_fb1, col_fb2 = st.columns([1, 2])

    with col_fb1:
        st.header("⭐ Response Quality")
        if total_fb > 0:
            st.metric("Average Rating", f"{avg_rating:.1f}/5")
            st.metric("Total Feedback Entries", total_fb)
            low_rated = orchestrator.feedback_system.get_low_rated_queries(threshold=2, min_feedback=1)
            st.metric("Low Rated Queries (<=2)", len(low_rated))
        else:
            st.info("No feedback data available yet.")

    with col_fb2:
        st.header("📊 Rating Distribution")
        if total_fb > 0:
            rating_data = feedback_metrics.get("rating_distribution", {})
            # Ensure all ratings 1-5 are present, even if count is 0
            full_rating_data = {r: rating_data.get(r, 0) for r in range(1, 6)}
            rating_df = pd.DataFrame({
                "Rating": list(full_rating_data.keys()),
                "Count": list(full_rating_data.values())
            })
            st.bar_chart(rating_df.set_index("Rating"))
        else:
            st.info("No feedback data available yet.")

    st.divider()

    # --- Recent Queries Log ---
    st.header("📜 Recent Queries Log")
    if hasattr(orchestrator, "query_log") and orchestrator.query_log:
        # Create dataframe from list of dicts, select and rename columns
        log_df = pd.DataFrame(orchestrator.query_log)
        log_df['timestamp'] = pd.to_datetime(log_df['timestamp'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
        log_df_display = log_df[[
            'timestamp', 'query', 'cache_source', 'score', 'response_source', 'response_time_ms', 'response'
        ]].rename(columns={
            'cache_source': 'Initial Hit Tier',
            'score': 'Initial Score',
            'response_source': 'Final Source',
            'response_time_ms': 'Time (ms)',
            'response': 'Response Snippet'
        })
        st.dataframe(log_df_display.tail(20), use_container_width=True) # Show last 20
    else:
        st.info("No queries logged yet.")

    # Add a manual refresh button
    st.button("Refresh Data")

    # Removed the auto-refresh loop
    # st.button("Refresh") # Keep manual refresh
    # time.sleep(30)
    # st.experimental_rerun() # Avoid forcing reruns