"""
Streamlit dashboard for X Education Lead Scoring.
Features:
- Single Lead Prioritizer (interactive input form, SVG gauge, top scoring factors).
- Batch Scoring & Analytics (CSV uploader, metric cards, download annotated CSV, interactive charts).
- Model Governance & Performance (CV comparison table, evaluation charts: ROC, PR, feature importances).
"""

from __future__ import annotations

import logging
import io
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

# Add parent directory of 'app' to sys.path to enable imports of 'src'
# regardless of the working directory the script is launched from.
app_dir = Path(__file__).resolve().parent
project_root = app_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set page config before importing other modules
st.set_page_config(
    page_title="X Education Lead Scoring Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.predict import LeadPredictor, clean_feature_name
from src.config import (
    PROJECT_ROOT,
    HOT_THRESHOLD,
    WARM_THRESHOLD,
    FIGURES_DIR,
    MODEL_COMPARISON_PATH
)

logger = logging.getLogger(__name__)

# Premium Custom CSS
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        /* Font styling */
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        
        /* Card styling */
        .metric-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        
        .metric-value {
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 5px;
        }
        
        .metric-label {
            font-size: 14px;
            text-transform: uppercase;
            color: #888888;
            font-weight: 600;
        }
        
        /* Explanations card styling */
        .factor-card-pos {
            background-color: rgba(16, 185, 129, 0.1);
            border-left: 5px solid #10B981;
            padding: 10px 15px;
            margin: 8px 0;
            border-radius: 4px;
        }
        
        .factor-card-neg {
            background-color: rgba(239, 68, 68, 0.1);
            border-left: 5px solid #EF4444;
            padding: 10px 15px;
            margin: 8px 0;
            border-radius: 4px;
        }
        
        /* Dashboard title gradient */
        .title-gradient {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 40px;
            font-weight: 700;
            margin-bottom: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


def render_gauge_svg(score: int, tier: str) -> str:
    """Generate inline SVG for a premium circular lead score gauge."""
    if tier == "Hot Lead":
        color = "#10B981"  # Emerald
        bg = "rgba(16, 185, 129, 0.15)"
    elif tier == "Warm Lead":
        color = "#F59E0B"  # Amber
        bg = "rgba(245, 158, 11, 0.15)"
    else:
        color = "#3B82F6"  # Blue
        bg = "rgba(59, 130, 246, 0.15)"
        
    circumference = 2 * 3.14159 * 40
    stroke_offset = circumference - (circumference * score / 100)
    
    svg = f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 25px; background: {bg}; border-radius: 20px; border: 1px solid {color}; max-width: 250px; margin: auto;">
        <svg width="150" height="150" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="40" stroke="#2D3748" stroke-width="8" fill="none" />
            <circle cx="50" cy="50" r="40" stroke="{color}" stroke-width="8" fill="none"
                    stroke-dasharray="{circumference}" stroke-dashoffset="{stroke_offset}"
                    stroke-linecap="round" transform="rotate(-90 50 50)" style="transition: stroke-dashoffset 0.5s ease-in-out;" />
            <text x="50" y="56" text-anchor="middle" font-size="20" font-family="'Outfit', sans-serif" font-weight="700" fill="white">{score}</text>
        </svg>
        <div style="margin-top: 15px; font-family: 'Outfit', sans-serif; font-size: 18px; font-weight: 700; color: {color}; text-transform: uppercase; letter-spacing: 1px;">
            {tier}
        </div>
    </div>
    """
    return svg


@st.cache_resource
def load_predictors() -> dict[str, LeadPredictor]:
    """Cache models and preprocessors loading."""
    predictors = {}
    try:
        predictors["A"] = LeadPredictor(variant="A")
    except Exception as e:
        st.error(f"Error loading Model A: {e}. Please ensure models are trained.")
    try:
        predictors["B"] = LeadPredictor(variant="B")
    except Exception as e:
        predictors["B"] = None
    return predictors


def main():
    # Sidebar Logo and Navigation
    st.sidebar.markdown("<h2 style='text-align: center; color: #764ba2;'>🎯 X Education</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center; font-size: 14px;'>Lead Prioritisation System</p>", unsafe_allow_html=True)
    st.sidebar.divider()
    
    # Model Selection toggle in sidebar
    model_variant = st.sidebar.selectbox(
        "Active Lead Scoring Model",
        options=["Model A (Primary - No Leakage)", "Model B (Comparison - With Leakage)"],
        index=0,
        help="Model A is the primary model for scoring live inbound leads. Model B is for comparison only."
    )
    variant_code = "A" if "Model A" in model_variant else "B"
    
    st.sidebar.divider()
    
    # Quick definitions
    st.sidebar.markdown(f"**Lead Tiers Rules:**")
    st.sidebar.markdown(f"- 🟢 **Hot Leads**: Score &ge; {HOT_THRESHOLD} (High priority, contact immediately)")
    st.sidebar.markdown(f"- 🟡 **Warm Leads**: Score {WARM_THRESHOLD} - {HOT_THRESHOLD-1} (Medium priority)")
    st.sidebar.markdown(f"- 🔵 **Cold Leads**: Score &lt; {WARM_THRESHOLD} (Low priority)")
    
    # Load predictors
    predictors = load_predictors()
    predictor = predictors.get(variant_code)
    
    if predictor is None:
        st.error(f"Failed to initialize the active predictor for Variant {variant_code}. Ensure models/ files exist.")
        return
        
    # Main Title Area
    st.markdown("<div class='title-gradient'>X Education Lead Prioritisation</div>", unsafe_allow_html=True)
    st.markdown("Convert more leads by focusing sales efforts on the prospects most likely to close.")
    st.divider()
    
    # Navigation Tabs
    tab_single, tab_batch, tab_gov = st.tabs([
        "👤 Single Lead Prioritiser", 
        "📁 Batch Scoring & Analytics", 
        "⚙️ Model Governance & Metrics"
    ])
    
    # =========================================================================
    # TAB 1: SINGLE LEAD PRIORITISER
    # =========================================================================
    with tab_single:
        st.subheader("Evaluate a New Prospect")
        st.write("Fill in the prospect's details below to calculate their conversion probability and key score drivers.")
        
        col_form, col_gauge = st.columns([2, 1])
        
        with col_form:
            with st.form("single_lead_form"):
                st.markdown("##### 1. Referral & Origin Details")
                col_fo_1, col_fo_2 = st.columns(2)
                with col_fo_1:
                    lead_origin = st.selectbox(
                        "Lead Origin",
                        options=["Landing Page Submission", "API", "Lead Add Form", "Lead Import"],
                        index=0
                    )
                with col_fo_2:
                    lead_source = st.selectbox(
                        "Lead Source",
                        options=["Google", "Direct Traffic", "Olark Chat", "Organic Search", "Reference", "Welingak Website", "Referral Sites", "Other"],
                        index=0
                    )
                    
                st.divider()
                st.markdown("##### 2. Website Engagement")
                col_eng_1, col_eng_2, col_eng_3 = st.columns(3)
                with col_eng_1:
                    total_visits = st.number_input("Total Website Visits", min_value=0, max_value=100, value=3, step=1)
                with col_eng_2:
                    time_spent = st.number_input("Time Spent on Website (Seconds)", min_value=0, max_value=5000, value=250, step=10)
                with col_eng_3:
                    page_views = st.number_input("Page Views Per Visit", min_value=0.0, max_value=30.0, value=2.0, step=0.5)
                    
                st.divider()
                st.markdown("##### 3. Professional Profile")
                col_prof_1, col_prof_2 = st.columns(2)
                with col_prof_1:
                    occupation = st.selectbox(
                        "Current Occupation",
                        options=["Unemployed", "Working Professional", "Student", "Housewife", "Other"],
                        index=0
                    )
                with col_prof_2:
                    specialization = st.selectbox(
                        "Specialization",
                        options=[
                            "Marketing Management", "Finance Management", "Human Resource Management",
                            "Operations Management", "Business Administration", "IT Projects Management",
                            "Supply Chain Management", "Travel and Tourism", "Media and Media",
                            "Banking, Investment And Insurance", "E-Commerce", "International Business",
                            "Retail Management", "Hospitality Management", "Other Specialization", "None"
                        ],
                        index=0
                    )
                    
                st.divider()
                st.markdown("##### 4. Restrictions & Communications")
                col_comm_1, col_comm_2 = st.columns(2)
                with col_comm_1:
                    do_not_email = st.selectbox("Do Not Email (Opt-Out)", options=["No", "Yes"], index=0)
                with col_comm_2:
                    do_not_call = st.selectbox("Do Not Call (Opt-Out)", options=["No", "Yes"], index=0)
                    
                submit_button = st.form_submit_button("Calculate Lead Score", use_container_width=True)
                
        with col_gauge:
            if submit_button:
                # Compile input dictionary
                lead_dict = {
                    "Lead Origin": lead_origin,
                    "Lead Source": lead_source,
                    "TotalVisits": total_visits,
                    "Total Time Spent on Website": time_spent,
                    "Page Views Per Visit": page_views,
                    "What is your current occupation": occupation,
                    "Specialization": specialization if specialization != "None" else "Select",
                    "Do Not Email": do_not_email,
                    "Do Not Call": do_not_call,
                }
                
                with st.spinner("Analyzing profile..."):
                    result = predictor.predict_single(lead_dict)
                    
                score = result["lead_score"]
                tier = result["tier"]
                
                # Render SVG Gauge
                st.markdown(render_gauge_svg(score, tier), unsafe_allow_html=True)
                st.write("")
                
                # Show factors
                st.markdown("### Score Attribution")
                
                pos_factors = result["factors"]["positive"]
                neg_factors = result["factors"]["negative"]
                
                if not pos_factors and not neg_factors:
                    st.write("No major scoring drivers detected. Scoring is near the baseline.")
                else:
                    if pos_factors:
                        st.markdown("**Positive Drivers (increases score):**")
                        for f in pos_factors:
                            st.markdown(
                                f"<div class='factor-card-pos'><b>{f['clean_name']}</b>: +{f['change']*100:.1f}% probability</div>",
                                unsafe_allow_html=True
                            )
                    if neg_factors:
                        st.markdown("**Negative Drivers (decreases score):**")
                        for f in neg_factors:
                            st.markdown(
                                f"<div class='factor-card-neg'><b>{f['clean_name']}</b>: {f['change']*100:.1f}% probability</div>",
                                unsafe_allow_html=True
                            )
            else:
                # Default landing state
                st.markdown(
                    """
                    <div style="text-align: center; color: #888888; padding: 50px 20px; border: 2px dashed rgba(255,255,255,0.15); border-radius: 10px;">
                        <h4>No Lead Evaluated Yet</h4>
                        <p>Fill out the form on the left and click <b>Calculate Lead Score</b> to view priorities and details.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # =========================================================================
    # TAB 2: BATCH SCORING & ANALYTICS
    # =========================================================================
    with tab_batch:
        st.subheader("Process a Lead List")
        st.write("Upload a CSV file containing multiple leads to calculate lead scores and prioritize them at scale.")
        
        uploaded_file = st.file_uploader("Upload CSV List", type="csv")
        
        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded {len(raw_df)} leads from CSV!")
                
                with st.spinner("Scoring leads..."):
                    scored_df = predictor.predict_batch(raw_df)
                    
                # Calculate metrics
                total_leads = len(scored_df)
                hot_leads = int((scored_df["Score Tier"] == "Hot Lead").sum())
                warm_leads = int((scored_df["Score Tier"] == "Warm Lead").sum())
                cold_leads = int((scored_df["Score Tier"] == "Cold Lead").sum())
                avg_score = float(scored_df["Lead Score"].mean())
                
                # Conversion rate prediction
                # In evaluate, we saw Model A prioritises leads with a ~70% conversion rate
                expected_conversions = int(round(hot_leads * 0.70 + warm_leads * 0.30))
                expected_conversion_rate = expected_conversions / total_leads if total_leads > 0 else 0.0
                
                # Display metrics
                st.write("")
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                
                with col_m1:
                    st.markdown(
                        f"<div class='metric-card'><div class='metric-value'>{total_leads:,}</div><div class='metric-label'>Total Leads</div></div>",
                        unsafe_allow_html=True
                    )
                with col_m2:
                    st.markdown(
                        f"<div class='metric-card'><div class='metric-value' style='color:#10B981;'>{hot_leads:,} ({100*hot_leads/total_leads:.1f}%)</div><div class='metric-label'>Hot Leads</div></div>",
                        unsafe_allow_html=True
                    )
                with col_m3:
                    st.markdown(
                        f"<div class='metric-card'><div class='metric-value'>{avg_score:.1f}</div><div class='metric-label'>Average Lead Score</div></div>",
                        unsafe_allow_html=True
                    )
                with col_m4:
                    st.markdown(
                        f"<div class='metric-card'><div class='metric-value' style='color:#764ba2;'>{expected_conversion_rate*100:.1f}%</div><div class='metric-label'>Est. Target Conversion</div></div>",
                        unsafe_allow_html=True
                    )
                
                st.write("")
                st.divider()
                
                # Show table & download
                st.subheader("Annotated Prioritisation List")
                st.write("Here are the prioritized leads. You can sort and filter using the table headers.")
                
                # Drop un-needed large columns for UI display clean layout
                clean_display_cols = [
                    "Prospect ID", "Lead Number", "Lead Score", "Score Tier", 
                    "Lead Origin", "Lead Source", "TotalVisits", 
                    "Total Time Spent on Website", "What is your current occupation"
                ]
                display_df = scored_df[[c for c in clean_display_cols if c in scored_df.columns]].copy()
                st.dataframe(display_df, use_container_width=True)
                
                # Download scored CSV
                csv_buffer = io.StringIO()
                scored_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download Full Scored Leads CSV",
                    data=csv_buffer.getvalue(),
                    file_name="scored_leads_export.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                st.divider()
                st.subheader("Batch Analytics Insights")
                
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    # Score distribution histogram
                    st.markdown("**Lead Score Distribution**")
                    hist_data = pd.DataFrame({"Lead Score": scored_df["Lead Score"]})
                    st.bar_chart(hist_data["Lead Score"].value_counts().sort_index())
                    
                with col_chart2:
                    # Lead tier bar chart
                    st.markdown("**Lead Tiers Count**")
                    tier_counts = scored_df["Score Tier"].value_counts().reset_index()
                    tier_counts.columns = ["Tier", "Count"]
                    st.bar_chart(data=tier_counts, x="Tier", y="Count")
                    
            except Exception as e:
                st.error(f"Error processing batch CSV file: {e}")
                logger.exception("Batch scoring streamlit error")
        else:
            st.markdown(
                """
                <div style="text-align: center; color: #888888; padding: 80px 20px; border: 2px dashed rgba(255,255,255,0.15); border-radius: 10px;">
                    <h3>Upload Lead List</h3>
                    <p>Drag and drop a <code>.csv</code> file containing lead records to score them instantly in batch mode.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # =========================================================================
    # TAB 3: MODEL GOVERNANCE & METRICS
    # =========================================================================
    with tab_gov:
        st.subheader("Model Evaluation & Compliance Summary")
        st.write(
            "We enforce strict model governance to avoid data leakage in production. "
            "Model A uses pre-contact characteristics only and is deployable. "
            "Model B uses post-contact attributes (e.g. Tags, Lead Quality) and is for reference only."
        )
        
        # Display CV results
        if Path(MODEL_COMPARISON_PATH).exists():
            st.markdown("#### 1. Cross-Validation Performance Comparison")
            comp_df = pd.read_csv(MODEL_COMPARISON_PATH)
            
            # Format comparison table
            st.dataframe(comp_df.style.highlight_max(axis=0, subset=["cv_roc_auc", "cv_f1", "cv_accuracy"]), use_container_width=True)
        
        st.divider()
        
        st.markdown("#### 2. Evaluation Curve Comparison")
        col_fig1, col_fig2 = st.columns(2)
        
        fig_roc_path = FIGURES_DIR / "06_roc_curve.png"
        fig_pr_path = FIGURES_DIR / "07_precision_recall_curve.png"
        fig_dist_path = FIGURES_DIR / "08_lead_score_distribution.png"
        
        with col_fig1:
            if fig_roc_path.exists():
                st.image(str(fig_roc_path), caption="ROC Curve (True Positive Rate vs False Positive Rate)")
            else:
                st.warning("ROC curve figure not found. Run evaluation script.")
                
        with col_fig2:
            if fig_pr_path.exists():
                st.image(str(fig_pr_path), caption="Precision-Recall Curve (Precision vs Recall)")
            else:
                st.warning("PR curve figure not found. Run evaluation script.")
                
        st.divider()
        st.markdown("#### 3. Lead Score Distribution & Feature Importances (Model A)")
        col_fig3, col_fig4 = st.columns(2)
        
        fig_feat_a_path = FIGURES_DIR / "09_feature_importance_model_a.png"
        
        with col_fig3:
            if fig_dist_path.exists():
                st.image(str(fig_dist_path), caption="Lead Score Distribution by Class")
            else:
                st.warning("Lead score distribution figure not found. Run evaluation script.")
                
        with col_fig4:
            if fig_feat_a_path.exists():
                st.image(str(fig_feat_a_path), caption="Model A Feature Importance (Top 15)")
            else:
                st.warning("Feature importance figure not found. Run evaluation script.")


if __name__ == "__main__":
    main()
