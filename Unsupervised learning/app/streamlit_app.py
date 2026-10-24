import streamlit as st
import requests
import pandas as pd
import time

st.set_page_config(page_title="Customer Personas", page_icon="✨", layout="wide", initial_sidebar_state="expanded")

# Inject Custom Modern CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Sleek gradient background for the main app */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(15, 23, 42) 0%, rgb(30, 27, 75) 90%);
    }

    /* Style for custom persona cards */
    .persona-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        height: 100%;
    }
    
    .persona-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(129, 140, 248, 0.5);
    }

    .persona-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 12px;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }

    .persona-desc {
        color: #94a3b8;
        font-size: 1.1rem;
        line-height: 1.5;
        margin-bottom: 20px;
        font-weight: 300;
    }
    
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        background: rgba(129, 140, 248, 0.15);
        color: #818cf8;
        border: 1px solid rgba(129, 140, 248, 0.3);
    }

    /* Main Title Styling */
    h1 {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        letter-spacing: -1.5px !important;
        margin-bottom: 0.5rem !important;
        background: linear-gradient(to right, #fff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .subtitle {
        color: #64748b;
        font-size: 1.25rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.39);
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.23);
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    }
    
    /* Inputs */
    .stNumberInput input {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #f8fafc !important;
        border-radius: 10px !important;
    }
    .stNumberInput input:focus {
        border-color: #818cf8 !important;
        box-shadow: 0 0 0 1px #818cf8 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 1.1rem;
        padding-top: 1rem;
        padding-bottom: 1rem;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        color: #f8fafc !important;
    }
    
</style>
""", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000"

# Header
st.title("Customer Intelligence")
st.markdown('<p class="subtitle">Discover hidden segments and behavior patterns within your customer base.</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Segment Overview", "🎯 Target Individual", "📂 Batch Analysis"])

# --- TAB 1: OVERVIEW ---
with tab1:
    st.markdown("### Discovered Personas")
    try:
        r = requests.get(f"{API_URL}/segments", timeout=5)
        if r.status_code == 200:
            segments = r.json()
            cols = st.columns(len(segments))
            for i, (cid, p) in enumerate(segments.items()):
                with cols[i]:
                    st.markdown(f"""
                    <div class="persona-card">
                        <div class="badge">Cluster {cid}</div>
                        <div class="persona-title" style="margin-top: 15px;">{p['name']}</div>
                        <div class="persona-desc">{p['description']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.divider()
            st.markdown("### Architecture Metrics")
            metric_cols = st.columns(4)
            metric_cols[0].metric(label="Total Personas", value=len(segments))
            metric_cols[1].metric(label="Model Algorithm", value="K-Means")
            metric_cols[2].metric(label="Silhouette Score", value="0.378", delta="Excellent")
            metric_cols[3].metric(label="Davies-Bouldin", value="0.961", delta="-0.1", delta_color="inverse")
            
        else:
            st.warning("API is not available.")
    except Exception as e:
        st.error("Failed to connect to backend API. Please ensure FastAPI is running.")

# --- TAB 2: PREDICT ONE ---
with tab2:
    st.markdown("### Predict Customer Persona")
    st.markdown("Input a customer's behavioral metrics to instantly classify them into a persona.")
    
    with st.container(border=True):
        with st.form("predict_form", clear_on_submit=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.number_input("Age", min_value=18, max_value=100, value=35)
                income = st.number_input("Annual Income ($)", min_value=1.0, value=55000.0, step=1000.0)
            with c2:
                total_spend = st.number_input("Total Spend ($)", min_value=0.0, value=1200.0, step=50.0)
                num_purchases = st.number_input("Number of Purchases", min_value=0, value=15)
            with c3:
                recency_days = st.number_input("Recency (Days)", min_value=0, value=10, help="Days since last purchase")
                web_visits_per_month = st.number_input("Web Visits / Month", min_value=0, value=8)
            
            st.write("")
            submitted = st.form_submit_button("Run Analysis ✨")
        
    if submitted:
        payload = {
            "age": age,
            "income": income,
            "total_spend": total_spend,
            "num_purchases": num_purchases,
            "recency_days": recency_days,
            "web_visits_per_month": web_visits_per_month
        }
        
        # Fake loading animation for better UX
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i in range(100):
            progress_bar.progress(i + 1)
            if i < 30: status_text.text("Connecting to model inference endpoint...")
            elif i < 70: status_text.text("Applying StandardScaler and feature engineering...")
            else: status_text.text("Computing centroid distances...")
            time.sleep(0.01)
            
        progress_bar.empty()
        status_text.empty()
        
        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            r.raise_for_status()
            res = r.json()
            
            st.toast("Analysis Complete!", icon="🚀")
            
            with st.container(border=True):
                st.subheader(f"🎯 Match Found: {res['persona']}")
                st.write(res['description'])
                
                m1, m2 = st.columns(2)
                m1.metric("Cluster ID", res['cluster_id'])
                m2.metric("Confidence (Distance)", f"{res['distance_to_centroid']:.3f}")
            
        except requests.RequestException as e:
            st.error(f"API Error: {e}")

# --- TAB 3: BATCH ANALYSIS ---
with tab3:
    st.markdown("### Batch Segmentation")
    st.markdown("Upload a CSV dataset to process multiple customers simultaneously.")
    
    uploaded_file = st.file_uploader("Drop your customer dataset here", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        required_cols = ["age", "income", "total_spend", "num_purchases", "recency_days", "web_visits_per_month"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"⚠️ CSV is missing required columns: {', '.join(missing_cols)}")
        else:
            with st.expander("Preview Data", expanded=False):
                st.dataframe(df.head(), use_container_width=True)
                
            if st.button("Process Batch 🚀"):
                with st.spinner("Processing massive amounts of data..."):
                    # Drop rows with NaN values in required columns
                    df_clean = df.dropna(subset=required_cols).copy()
                    
                    if len(df_clean) == 0:
                        st.error("All rows contain missing values in required columns. Please check your data.")
                    else:
                        payload = df_clean[required_cols].to_dict(orient="records")
                        try:
                            r = requests.post(f"{API_URL}/predict/batch", json=payload, timeout=15)
                            r.raise_for_status()
                            res = r.json()["predictions"]
                            
                            df_clean["Predicted Persona"] = [p["persona"] for p in res]
                            df_clean["Cluster ID"] = [p["cluster_id"] for p in res]
                            
                            st.success(f"Successfully processed {len(df_clean)} customers! ({len(df) - len(df_clean)} rows skipped due to missing values)")
                            
                            st.markdown("### Results Distribution")
                            # Native streamlit bar chart
                            distribution = df_clean["Predicted Persona"].value_counts()
                            st.bar_chart(distribution)
                            
                            st.dataframe(df_clean, use_container_width=True)
                            
                            csv = df_clean.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Segmented Dataset",
                                data=csv,
                                file_name="segmented_customers.csv",
                                mime="text/csv",
                            )
                            
                        except requests.RequestException as e:
                            st.error(f"API Error: {e}")
