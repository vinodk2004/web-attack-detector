import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

# Suggested URLs as templates
attack_examples = {
    "Normal URLs": "http://localhost:8080/tienda1/publico/entrar.jsp HTTP/1.1" ,
    
    "XSS Examples": "https://example.com/search?q=<script>alert(1)</script>",

    "SQL Injection": "https://example.com/products?category='; DROP TABLE users--", 

    "Path Traversal": "https://example.com/download?file=../../etc/passwd"
}

with st.form("analysis_form"):
    # Attack type selector
    attack_type = st.selectbox("Choose a request type to simulate a potential web attack (or select Custom to enter your own):", 
                               ["Custom URL"] + list(attack_examples.keys()))

    # Automatically populate URL based on selected attack type
    if attack_type == "Custom URL":
        url = st.text_input("Target URL for Analysis:", placeholder="https://example.com/login.php?--")
    else:
        url = st.text_input("Target URL for Analysis:", value=attack_examples[attack_type])

    # Request content input
    content = st.text_area("Request Content:", height=75, placeholder="Optional for POST/PUT requests...")
    
    submitted = st.form_submit_button("Analyze Security Risk 🔍", type="primary")

# with st.form("analysis_form"):
#     # Attack type selector
#     attack_type = st.selectbox("Select attack type to preview:", 
#                              ["Custom URL"] + list(attack_examples.keys()))
    
#     # Dynamic URL selection
#     if attack_type != "Custom URL":
#         example_url = st.selectbox("Select example:", attack_examples[attack_type])
#         url = st.text_input("Enter URL:", value=example_url)
#     else:
#         url = st.text_input("Enter URL:", placeholder="https://example.com/login.php?--")
    
#     # Rest of your form
#     content = st.text_area("Request Content:", height=75, placeholder="Optional for POST/PUT requests...")
#     submitted = st.form_submit_button("Analyze Security Risk 🔍", type="primary")

st.markdown("""
<style>
.url-badge {
    display: inline-block;
    padding: 0.25em 0.4em;
    margin: 0.1em;
    font-size: 75%;
    font-weight: 700;
    line-height: 1;
    text-align: center;
    white-space: nowrap;
    vertical-align: baseline;
    border-radius: 0.25rem;
    cursor: pointer;
}
.badge-normal { color: #fff; background-color: #28a745; }        
.badge-xss { color: #fff; background-color: #dc3545; }
.badge-sqli { color: #fff; background-color: #fd7e14; }
.badge-path { color: #fff; background-color: #6f42c1; }
</style>
""", unsafe_allow_html=True)

examples = {
    "Normal": ("safe-example", "badge-normal"),
    "XSS": ("<script>alert(1)</script>", "badge-xss"),
    "SQLi": ("1' OR 1=1--", "badge-sqli"),
    "Path": ("../../etc/passwd", "badge-path")
}


if submitted and url:
    with st.spinner("Scanning for threats..."):
        result = predict_web_attack(url, content if content else None, method, pipeline)
        
        # Results Dashboard
        st.subheader("Analysis Results")
        
        col1, col2, col3 = st.columns(3)

        with col1:
            color = "red" if result['is_anomalous'] else "green"
            status_text = "Anomalous 🔴" if result['is_anomalous'] else "Normal 🟢"
            st.markdown(f"**Prediction**")
            st.markdown(f"<p style='color:{color}; font-size: 28px; font-weight: bold;'>{status_text}</p>",unsafe_allow_html=True)

        with col2:
            st.metric("Anomaly Probability", f"{result['probability_anomalous']:.1%}")

        with col3:
            risk_level = "High" if result['is_anomalous'] else "Low"
            st.metric("Risk Level", risk_level, delta_color="inverse")


        # Dynamic Tab Header Color
        tab_text_color = "#3c763d" if not result['is_anomalous'] else "#a94442"

        st.markdown(f"""
        <style>
        /* Target the outer tab container */
        .stTabs [data-baseweb="tab"] {{
            font-size: 0px !important; /* Hide default text size to prevent overlap */
        }}

        /* Target the actual text inside the tab */
        .stTabs [data-baseweb="tab"] > div {{
            font-size: 17px !important;
            font-weight: 700 !important;
            color: #fff !important;
        }}

        /* Optional: hover effect */
        .stTabs [data-baseweb="tab"]:hover > div {{
            color: {tab_text_color} !important;
        }}
        </style>
        """, unsafe_allow_html=True)

        # Visualizations
        tab1, tab2, tab3 = st.tabs(["Risk Meter", "Threat Indicators", "Recommendations"])
        
        with tab1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result['probability_anomalous'],
                title={'text': "Threat Probability"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "red" if result['is_anomalous'] else "green"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "#dff0d8"},
                        {'range': [0.5, 1], 'color': "#f2dede"}
                    ]
                },
                domain={'x': [0, 1], 'y': [0.2, 1]}  # Moves the gauge up to make space for custom annotation
            ))

            # Add custom text below the gauge
            fig.add_annotation(
                x=0.5, y=0.05,
                text=f"<b>Status:</b> {'Anomalous 🔴' if result['is_anomalous'] else 'Normal 🟢'}",
                showarrow=False,
                font=dict(size=16),
                xref="paper", yref="paper",
                align="center"
            )

            # Render
            st.plotly_chart(fig, use_container_width=True)


                    
        with tab2:
            if result['is_anomalous']:
                # st.error("**Detected Threats**")

                # SQL Injection Detection
                if any(keyword in url.lower() for keyword in ["'", "--", " or ", "1=1", "union", "select", "drop", "insert"]):
                    st.warning("- **SQL Injection Detected** (e.g., `' OR 1=1`, `UNION SELECT`, etc.)")

                # XSS Detection
                if any(tag in url.lower() for tag in ["<script", "javascript:", "onerror=", "alert(", "onload="]):
                    st.warning("- **Cross Site Scripting (XSS)** Detected")

                # Path Traversal
                if "../" in url or "%2e%2e%2f" in url.lower():
                    st.warning("- **Path Traversal** Detected (e.g., `../etc/passwd`)")

                # Shell Injection
                if any(cmd in url.lower() for cmd in [";ls", "|whoami", "`cat", "$(cat"]):
                    st.warning("- **Command Injection** Detected")

                # Encoding or Obfuscation
                if "%" in url and count_per(url) > 2:
                    st.warning(f"- **Suspicious Encoding Detected** (% count = {count_per(url)})")

                # URL Shortener
                if shortening_service(url):
                    st.warning("- **URL Shortening Service** Detected (can hide true destination)")

                # Simulated Attack Description
                st.markdown("#### 🧪 Simulated Attack Summary")
                st.markdown("""
                This request exhibits behavior matching known **malicious patterns**.  
                It's likely a **simulated** or **attempted attack** involving:
                - Input manipulation
                - Encoding tricks
                - Signature-based patterns from real-world attack data.
                """)
            else:
                st.success("✅ No obvious threat patterns detected")

            
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
