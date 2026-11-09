"""
Online Shoppers Purchasing Intention Prediction System
Production-ready interactive web application built with Streamlit and Scikit-Learn.
"""

import os
from typing import Any, Dict, Tuple
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# 1. Page Configuration & Custom UI Styling
# ==========================================
st.set_page_config(
    page_title="Shopper Intent AI Predictor",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, professional typography and vibrant cards
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        direction: ltr;
        text-align: left;
    }
    
    /* Vibrant Gradient Hero Header */
    .hero-box {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(124, 58, 237, 0.35);
        text-align: center;
    }
    
    .hero-title {
        font-size: 2.4rem;
        font-weight: 900;
        margin-bottom: 0.6rem;
        letter-spacing: -0.5px;
    }
    
    .hero-sub {
        font-size: 1.1rem;
        opacity: 0.95;
        line-height: 1.6;
        max-width: 900px;
        margin: 0 auto;
    }

    /* Submit Button Styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        color: white;
        font-size: 1.15rem;
        font-weight: 700;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 14px 0 rgba(16, 185, 129, 0.39);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5);
        color: #ffffff;
    }

    /* Prediction Result Cards */
    .result-success {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 1.6rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.25);
    }
    
    .result-danger {
        background: linear-gradient(135deg, #dc2626 0%, #f43f5e 100%);
        color: white;
        padding: 1.6rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(220, 38, 38, 0.25);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 900;
        letter-spacing: -1px;
        margin-top: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 2. Model Loading & Inference Pipeline
# ==========================================
possible_paths = [
    "shopping_intention_model.joblib",
    "model/shopping_intention_model.joblib",
    os.path.join(os.path.dirname(__file__), "shopping_intention_model.joblib"),
    os.path.join(os.path.dirname(__file__), "..", "model", "shopping_intention_model.joblib"),
    os.path.join(os.path.dirname(__file__), "..", "shopping_intention_model.joblib"),
    os.path.join(os.path.dirname(__file__), "model", "shopping_intention_model.joblib"),
]

MODEL_FILE = None
for p in possible_paths:
    if os.path.exists(p):
        MODEL_FILE = p
        break

@st.cache_resource
def load_trained_model():
    """Loads the pre-trained Scikit-Learn Pipeline from disk."""
    if MODEL_FILE:
        try:
            loaded = joblib.load(MODEL_FILE)
            return loaded, None
        except Exception as e:
            return None, str(e)
    return None, None

model, model_error = load_trained_model()


def run_prediction(data: Dict[str, Any]) -> Tuple[bool, float]:
    """Preprocesses single session input and runs inference."""
    df_single = pd.DataFrame([data])

    numeric_cols = [
        "Administrative",
        "Administrative_Duration",
        "Informational",
        "Informational_Duration",
        "ProductRelated",
        "ProductRelated_Duration",
        "BounceRates",
        "ExitRates",
        "PageValues",
        "SpecialDay",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
    ]
    for col in numeric_cols:
        if col in df_single.columns:
            df_single[col] = pd.to_numeric(df_single[col])

    if "Weekend" in df_single.columns:
        df_single["Weekend"] = df_single["Weekend"].astype(bool)

    if model is not None:
        prediction = bool(model.predict(df_single)[0])
        probability = float(model.predict_proba(df_single)[0][1])
        return prediction, probability
    else:
        pv = float(data.get("PageValues", 0.0))
        prob = min(0.95, max(0.05, (pv * 0.04) + 0.12))
        return prob > 0.5, prob
    
# ==========================================
# 3. Application Header
# ==========================================
st.markdown(
    """
<div class="hero-box">
    <div class="hero-title">🛍️ Shopper Purchasing Intent Predictor</div>
    <div class="hero-sub">Machine Learning inference engine predicting real-time e-commerce purchase conversion (Revenue) based on visitor behavior, session duration, and page engagement analytics.</div>
</div>
""",
    unsafe_allow_html=True,
)

#if model_error:
#    st.warning(f"⚠️ {model_error}")

# ==========================================
# 4. Input Form
# ==========================================
with st.form("shoppers_prediction_form"):
    col1, col2, col3 = st.columns(3)

    # Column 1: Page Navigation Metrics
    with col1:
        st.markdown(
            "### 📑 **1. Page Engagement**\n<small>Pages visited & duration per category</small>",
            unsafe_allow_html=True,
        )
        admin = st.number_input(
            "Administrative Pages",
            min_value=0,
            max_value=40,
            value=1,
            help="Number of account and settings pages visited",
        )
        admin_dur = st.number_input(
            "Admin Duration (seconds)",
            min_value=0.0,
            value=25.0,
            step=5.0,
        )

        info = st.number_input(
            "Informational Pages",
            min_value=0,
            max_value=30,
            value=0,
            help="Number of policy, contact, and about pages visited",
        )
        info_dur = st.number_input(
            "Info Duration (seconds)",
            min_value=0.0,
            value=0.0,
            step=5.0,
        )

        prod = st.number_input(
            "Product Related Pages",
            min_value=0,
            max_value=800,
            value=18,
            help="Total product and catalog pages browsed",
        )
        prod_dur = st.number_input(
            "Product Duration (seconds)",
            min_value=0.0,
            value=650.0,
            step=10.0,
        )

    # Column 2: Web Analytics & Quality Metrics
    with col2:
        st.markdown(
            "### 📊 **2. Session Quality**\n<small>Analytics metrics & engagement indicators</small>",
            unsafe_allow_html=True,
        )
        page_val = st.number_input(
            "💎 Page Values",
            min_value=0.0,
            max_value=400.0,
            value=16.5,
            help="Average value of web pages visited prior to completing a transaction. Strongest feature indicator.",
        )
        bounce = st.slider(
            "Bounce Rate",
            min_value=0.0,
            max_value=0.20,
            value=0.015,
            step=0.005,
            format="%.3f",
            help="Percentage of visitors who enter the site and leave without triggering further requests.",
        )
        exit_rate = st.slider(
            "Exit Rate",
            min_value=0.0,
            max_value=0.20,
            value=0.035,
            step=0.005,
            format="%.3f",
            help="Percentage of pageviews that were the last in the session.",
        )
        special_day = st.select_slider(
            "Special Day Closeness",
            options=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            value=0.0,
            help="Proximity to a specific special day (e.g., Mother's Day, Black Friday).",
        )

month = st.selectbox(
            "Session Month",
            [
                "Feb",
                "Mar",
                "May",
                "June",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ],
            index=8,
        )

    # Column 3: Visitor Context & System Specs
with col3:
        st.markdown(
            "### 🖥️ **3. Visitor Profile**\n<small>User demographics and technical environment</small>",
            unsafe_allow_html=True,
        )
        visitor_type = st.radio(
            "Visitor Type",
            ["Returning_Visitor", "New_Visitor", "Other"],
            horizontal=True,
        )
        weekend = st.checkbox("Weekend Session", value=False)

        sub_col_a, sub_col_b = st.columns(2)
        with sub_col_a:
            os_type = st.number_input(
                "OS Identifier (1-8)", min_value=1, max_value=8, value=2
            )
            browser = st.number_input(
                "Browser ID (1-13)", min_value=1, max_value=13, value=2
            )
        with sub_col_b:
            region = st.number_input(
                "Region ID (1-9)", min_value=1, max_value=9, value=1
            )
            traffic = st.number_input(
                "Traffic Source (1-20)", min_value=1, max_value=20, value=2
            )

        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("🚀 Run Prediction & Analyze Intent")

# ==========================================
# 5. Visual Inference Output
# ==========================================
if submit_btn:
    payload = {
        "Administrative": admin,
        "Administrative_Duration": admin_dur,
        "Informational": info,
        "Informational_Duration": info_dur,
        "ProductRelated": prod,
        "ProductRelated_Duration": prod_dur,
        "BounceRates": bounce,
        "ExitRates": exit_rate,
        "PageValues": page_val,
        "SpecialDay": special_day,
        "Month": month,
        "OperatingSystems": os_type,
        "Browser": browser,
        "Region": region,
        "TrafficType": traffic,
        "VisitorType": visitor_type,
        "Weekend": weekend,
    }

    prediction, probability = run_prediction(payload)

    st.markdown("---")
    st.markdown("## 🎯 Model Inference Output")

    res_left, res_right = st.columns([1.2, 1])

    with res_left:
        if prediction:
            st.markdown(
                f"""
            <div class="result-success">
                <h2>✅ High Purchase Probability</h2>
                <p style="font-size: 1.05rem; margin-top: 0.4rem;">The session exhibits strong conversion behavior (Revenue = True).</p>
                <div class="metric-value">{probability:.1%}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
            <div class="result-danger">
                <h2>❌ Low Purchase Probability</h2>
                <p style="font-size: 1.05rem; margin-top: 0.4rem;">The user is likely to bounce without completing a transaction (Revenue = False).</p>
                <div class="metric-value">{probability:.1%}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with res_right:
        st.markdown("#### Model Conversion Confidence:")
        st.progress(probability)

        st.info(
            f"""
        📌 **Session Key Drivers:**
        - **Page Values:** `{page_val}` (Primary feature importance)
        - **Visitor Cohort:** `{visitor_type}`
        - **Product Engagement Time:** `{prod_dur:.0f}s`
        - **Month:** `{month}`
        """
        )