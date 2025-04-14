import streamlit as st
import pandas as pd
import plotly.express as px
from web_attack_detector_utils import *

# UI Config
st.set_page_config(page_title="Web Attack Detector", layout="wide", page_icon="🛡️")

# Cache the model
@st.cache_resource
def load_cached_model():
    return load_model_pipeline('web_attack_model.pkl')

pipeline = load_cached_model()

st.markdown("""
<style>
@media screen and (max-width: 600px) {
    .stTextInput input {font-size: 16px !important;}
    .stButton>button {width: 100%;}
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("shield_icon.png", width=80)
    st.title("Settings")
    method = st.radio("HTTP Method", ["GET", "POST", "PUT", "DELETE"], index=0)
    st.markdown("---")
    st.markdown("**About**")
    st.info("This tool analyzes URLs for potential web attacks using machine learning.")

# Main Interface
st.title("🛡️ Web Attack Detection System")
st.markdown("Analyze URLs for SQLi, XSS, Path Traversal and other web attacks")

with st.form("analysis_form"):
    url = st.text_input("Enter URL :", placeholder="https://example.com/login.php?--")
    content = st.text_area("Request Content:", height=100,
                         placeholder="Optional for POST/PUT requests...")
    submitted = st.form_submit_button("Analyze Security Risk 🔍", type="primary")

if submitted and url:
    with st.spinner("Scanning for threats..."):
        result = predict_web_attack(url, content if content else None, method, pipeline)
        
        # Results Dashboard
        st.subheader("Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prediction", result['prediction'])
        with col2:
            st.metric("Anomaly Probability", f"{result['probability_anomalous']:.1%}")
        with col3:
            st.metric("Risk Level", 
                     "High" if result['is_anomalous'] else "Low",
                     delta_color="inverse")
        
        # Visualizations
        tab1, tab2, tab3 = st.tabs(["Risk Meter", "Threat Indicators", "Recommendations"])
        
        with tab1:
            fig = px.bar(x=["Normal", "Anomalous"],
                        y=[result['probability_normal'], result['probability_anomalous']],
                        # color=["green", "red"],
                        title="Classification Probabilities")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            if result['is_anomalous']:
                st.error("**Detected Threats:**")
                if "script" in url.lower():
                    st.warning("- Potential XSS (JavaScript detected)")
                if count_per(url) > 2:
                    st.warning(f"- Suspicious character count (%): {count_per(url)}")
                if shortening_service(url):
                    st.warning("- URL shortening service detected")
            else:
                st.success("No obvious threat patterns detected")
        
        with tab3:
            if result['is_anomalous']:
                st.warning("""
                **Recommended Actions:**
                - Block this request pattern
                - Review server logs for similar requests
                - Implement WAF rules for this pattern
                - Scan for possible vulnerabilities
                """)
            else:
                st.info("""
                **Security Best Practices:**
                - Always validate user input
                - Use parameterized queries
                - Implement CSP headers
                - Keep systems patched
                """)
        
        # Raw Data
        with st.expander("Technical Details"):
            st.json(result)

elif submitted:
    st.warning("Please enter a URL to analyze")
