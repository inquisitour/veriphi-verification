# src/frontend/app.py
import os
import pandas as pd
import torch
import streamlit as st
import plotly.express as px

from core import create_core_system
from core.models import create_test_model, create_sample_input

st.set_page_config(page_title="Veriphi Robustness Dashboard", layout="wide")
st.title("🔒 Veriphi – Neural Network Robustness Verifier")

# =====================================================
# --- Sidebar controls: Run verification ---
# =====================================================
st.sidebar.header("⚙️ Run Verification")
model_name = st.sidebar.selectbox("Model", ["tiny", "linear", "conv"])
epsilon = st.sidebar.slider("ε (perturbation bound)", 0.01, 0.2, 0.1, 0.01)
norm = st.sidebar.selectbox("Norm", ["inf", "2"])
device = "cuda" if torch.cuda.is_available() else "cpu"

if st.sidebar.button("Run Verification"):
    core = create_core_system(use_attacks=True, device=device)
    model = create_test_model(model_name)
    x = create_sample_input(model_name)

    with st.spinner("Running attack-guided verification..."):
        res = core.verify_robustness(model, x, epsilon=epsilon, norm=norm, timeout=30)

    st.success("✅ Verification Finished")
    st.metric("Status", res.status.value)
    st.metric("Verified", str(res.verified))
    st.metric("Time (s)", f"{res.verification_time:.3f}")

    if res.additional_info:
        st.subheader("Additional Info")
        st.json(res.additional_info)

# =====================================================
# --- Sidebar controls: Baseline results viewer ---
# =====================================================
st.sidebar.header("📊 Baselines")
baseline_type = st.sidebar.selectbox("Choose baseline", ["CPU", "GPU"])
baseline_dir = f"data/baselines/{baseline_type.lower()}/summary"

if os.path.exists(baseline_dir):
    files = [f for f in os.listdir(baseline_dir) if f.endswith(".csv")]
    if files:
        choice = st.sidebar.selectbox("Pick summary file", files)
        if choice:
            df = pd.read_csv(os.path.join(baseline_dir, choice))
            st.subheader(f"Baseline Results ({baseline_type})")

            # epsilon filter
            if "epsilon" in df.columns:
                epsilons = sorted(df["epsilon"].unique())
                eps_choice = st.sidebar.selectbox("Filter by ε", epsilons)
                df = df[df["epsilon"] == eps_choice]
                st.caption(f"Showing results for ε = {eps_choice}")

            # Show raw table
            st.dataframe(df)

            # --- Verification Rate Chart ---
            if "verification_rate" in df.columns:
                fig_rate = px.bar(
                    df,
                    x="model",
                    y="verification_rate",
                    color="norm",
                    barmode="group",
                    title="Verification Rate by Model/Norm",
                )
                st.plotly_chart(fig_rate, use_container_width=True)

            # --- Average Time Chart ---
            if "avg_time_s" in df.columns:
                fig_time = px.bar(
                    df,
                    x="model",
                    y="avg_time_s",
                    color="norm",
                    barmode="group",
                    title="Average Verification Time (s)",
                )
                st.plotly_chart(fig_time, use_container_width=True)

            # --- Memory Usage Chart ---
            if "avg_mem_mb" in df.columns:
                fig_mem = px.bar(
                    df,
                    x="model",
                    y="avg_mem_mb",
                    color="norm",
                    barmode="group",
                    title="Average Memory Usage (MB)",
                )
                st.plotly_chart(fig_mem, use_container_width=True)
