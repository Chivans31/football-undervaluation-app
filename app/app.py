import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st
from fpdf import FPDF, XPos, YPos
import base64

# Path alignment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference import predict
from src.explainability.shap_explainer import explain_prediction

# --- PDF GENERATION ---
def generate_pdf(result, input_data):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("helvetica", "B", 20)
    pdf.cell(200, 10, "Football Scout AI: Valuation Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.ln(10)
    
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(200, 10, "Player Profile Parameters:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("helvetica", "", 12)
    for key, val in input_data.items():
        clean_key = key.replace('_', ' ').title()
        pdf.cell(200, 8, f"- {clean_key}: {val}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.ln(10)
    
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(200, 10, "Model Predictions:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("helvetica", "", 12)
    pdf.cell(200, 8, f"Estimated Fair Value: EUR {result['expected_market_value']:,.2f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(200, 8, f"Mispricing Log-Score: {result['mispricing_score']:.4f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(200, 8, f"Undervaluation Probability: {result['undervalued_probability']:.1%}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.ln(5)
    verdict = "UNDERVALUED" if result["is_undervalued"] else "FAIRLY PRICED/OVERVALUED"
    
    # Set color based on verdict
    if result["is_undervalued"]:
        pdf.set_text_color(0, 128, 0)
    else:
        pdf.set_text_color(200, 0, 0)
        
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(200, 10, f"FINAL VERDICT: {verdict}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    return bytes(pdf.output())

# --- STREAMLIT UI ---
st.set_page_config(page_title="Football Scout AI", layout="wide", page_icon="⚽")
st.title("⚽ Undervalued Player Detection")
st.markdown("---")

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.header("📋 Player Profile")
    with st.container(border=True):
        val = st.number_input("Current Market Value (€)", value=5000000, step=500000)
        age = st.slider("Age", 16, 40, 24)
        s_mean = st.slider("Sentiment Mean (News)", -1.0, 1.0, 0.2)
        s_std = st.slider("Sentiment Volatility", 0.0, 1.0, 0.1)
        m_std = st.slider("Market Value Volatility (€)", 0.0, 1000000.0, 50000.0)
        
        st.markdown("###")
        btn = st.button("🔍 Analyze Player", use_container_width=True, type="primary")

if btn:
    input_data = {
        "current_market_value": val,
        "Age": age,
        "sentiment_mean": s_mean,
        "sentiment_std": s_std,
        "mv_std": m_std
    }
    
    # Run Inference
    with st.spinner("Calculating valuation metrics..."):
        result = predict(input_data)
    
    with col2:
        st.header("📊 Evaluation Results")
        
        # Calculate Delta
        price_diff = result['expected_market_value'] - val
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Expected Fair Value", f"€{result['expected_market_value']:,.0f}", delta=f"€{price_diff:,.0f}")
        m2.metric("Mispricing Score", f"{result['mispricing_score']:.3f}")
        m3.metric("AI Confidence", f"{result['undervalued_probability']:.1%}")

        if result["is_undervalued"]:
            st.success("🎯 Verdict: UNDERVALUED ASSET")
            st.balloons()
        else:
            st.warning("⚖️ Verdict: FAIRLY PRICED / OVERVALUED")

        st.divider()
        st.subheader("💡 Feature Explainer (SHAP)")
        st.info("This chart shows how each feature influenced the model's price prediction.")
        
        # SHAP Visualization
        shap_values = explain_prediction(result["full_df"])
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values, max_display=10, show=False) # Changed from shap_values[0]
        st.pyplot(fig)
        plt.clf()

        # --- DOWNLOAD BUTTON ---
        st.divider()
        pdf_bytes = generate_pdf(result, input_data)
        
        st.download_button(
            label="📄 Download Detailed Scouting Report",
            data=pdf_bytes,
            file_name=f"scout_report_{val}.pdf",
            mime="application/pdf",
            use_container_width=True
        )