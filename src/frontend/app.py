# src/frontend/app.py (Simplified for Login Node)
import os
import sys
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import glob

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Veriphi - TRM Verification Results",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: #2b2b2b;
        border-radius: 5px 5px 0 0;
        font-weight: 600;
        color: #ffffff;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #3b3b3b;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_trm_results():
    """Load TRM robustness sweep results."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        return None
    
    csvs = glob.glob(os.path.join(log_dir, "trm_robustness_sweep*.csv"))
    
    if not csvs:
        return None
    
    dfs = []
    for csv_file in csvs:
        try:
            df = pd.read_csv(csv_file)
            
            # Add bound method if missing
            if "bound" not in df.columns:
                if "alpha" in csv_file.lower():
                    df["bound"] = "α-CROWN"
                elif "beta" in csv_file.lower():
                    df["bound"] = "β-CROWN"
                else:
                    df["bound"] = "CROWN"
            
            # Calculate verified fraction if missing
            if "verified_fraction" not in df.columns:
                if "verified" in df.columns and "total" in df.columns:
                    df["verified_fraction"] = df["verified"] / df["total"].replace(0, 1)
            
            dfs.append(df)
        except Exception as e:
            st.warning(f"⚠️ Could not load {csv_file}: {str(e)}")
            continue
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<h1 class="main-header">🛡️ Veriphi TRM Verification</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Attack-Guided Verification Results for Tiny Recursive Models</p>', unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("## 🧠 Veriphi Project")
    
    st.markdown("""
    **Neural Network Robustness Verification**
    
    Combining fast adversarial attacks with formal verification methods 
    to certify neural network robustness.
    """)
    
    st.markdown("---")
    
    st.markdown("### 🎯 Key Components")
    st.markdown("""
    - **Attack-Guided Verification**  
      FGSM + I-FGSM → α,β-CROWN
      
    - **TRM Architecture**  
      Tiny Recursive Models on MNIST
      
    - **Adversarial Training**  
      ε=0.15 PGD training
      
    - **GPU-Accelerated**  
      VSC-5 A100 cluster
    """)
    
    st.markdown("---")
    
    st.markdown("### 📊 Current Results")
    st.metric("Verified @ ε=0.01", "80.1%", delta="+67x vs standard")
    st.metric("Avg Time/Sample", "0.25s", delta="GPU optimized")
    st.metric("Total Samples", "512", help="Statistically significant")
    
    st.markdown("---")
    
    st.markdown("### 🔗 Resources")
    st.markdown("""
    - [GitHub Repo](https://github.com/inquisitour/veriphi-verification)
    - [TRM Paper](https://arxiv.org/abs/2510.04871)
    - [α,β-CROWN](https://arxiv.org/abs/2103.06624)
    - [VSC-5 Cluster](https://vsc.ac.at/systems/vsc-5/)
    """)
    
    st.markdown("---")
    st.caption("**AI Safety Hackathon 2025**  \nTU Wien | Team Veriphi")

# ============================================================================
# MAIN TABS
# ============================================================================

tab1, tab2 = st.tabs([
    "📚 Project Overview", 
    "🧠 TRM Results"
])

# ============================================================================
# TAB 1: PROJECT OVERVIEW
# ============================================================================

with tab1:
    st.header("📚 Project Overview")
    
    # Problem Statement
    st.markdown("### ❗ The Problem")
    st.markdown("""
    Neural networks are vulnerable to **adversarial attacks** - small, imperceptible 
    perturbations that cause misclassification. This is critical for:
    
    - 🚗 Autonomous vehicles
    - 🏥 Medical diagnosis systems
    - 🔒 Security applications
    - 💰 Financial AI systems
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.error("""
        **❌ Standard Training**
        - No robustness guarantees
        - Vulnerable to attacks
        - ~1% verified @ ε=0.01
        """)
    
    with col2:
        st.success("""
        **✅ Adversarial Training**
        - Certified robustness
        - Attack-resistant
        - **80% verified @ ε=0.01**
        """)
    
    st.markdown("---")
    
    # Solution
    st.markdown("### 💡 Our Solution: Attack-Guided Verification")
    
    st.markdown("""
    We combine **fast adversarial attacks** with **formal verification** for efficient certification:
    """)
    
    # Workflow diagram using columns
    col_step1, col_arrow1, col_step2, col_arrow2, col_step3 = st.columns([3, 1, 3, 1, 3])
    
    with col_step1:
        st.info("""
        **Phase 1: Attack**
        - FGSM, I-FGSM
        - 20-50ms execution
        - Find vulnerabilities fast
        """)
    
    with col_arrow1:
        st.markdown("### →")
    
    with col_step2:
        st.warning("""
        **Decision Point**
        - Attack found bug? → FALSIFIED ❌
        - No bug found? → Proceed to Phase 2
        """)
    
    with col_arrow2:
        st.markdown("### →")
    
    with col_step3:
        st.success("""
        **Phase 2: Formal**
        - α,β-CROWN bounds
        - 100-500ms execution
        - Mathematical proof ✓
        """)
    
    st.markdown("---")
    
    # TRM Architecture
    st.markdown("### 🧠 Tiny Recursive Models (TRM)")
    
    st.markdown("""
    TRM is a novel architecture using **recursive computation** instead of deep stacking:
    """)
    
    trm_col1, trm_col2 = st.columns(2)
    
    with trm_col1:
        st.markdown("""
        **Traditional Deep Networks:**
        - 32+ layers stacked
        - 500M+ parameters
        - Hard to verify
        """)
    
    with trm_col2:
        st.markdown("""
        **TRM Approach:**
        - **2 layers** recursively applied
        - **~1M parameters** (100x smaller)
        - **Easier to verify** ✅
        """)
    
    st.info("""
    **Key Insight:** Recursive depth (H_cycles × L_cycles) achieves deep network 
    performance with shallow network verifiability.
    """)
    
    st.markdown("---")
    
    # Methodology
    st.markdown("### 🔬 Our Methodology")
    
    st.markdown("""
    **Training Pipeline:**
    1. Train standard TRM-MLP on MNIST (28×28 grayscale)
    2. Adversarially fine-tune with PGD (ε=0.15)
    3. Verify robustness across multiple ε values
    
    **Verification Setup:**
    - **Hardware:** VSC-5 A100 GPU (80GB)
    - **Framework:** auto-LiRPA + α,β-CROWN
    - **Dataset:** MNIST test set (512 samples)
    - **Perturbation:** L∞ norm (ε: 0.01 → 0.1)
    """)
    
    st.code("""
# Training
python scripts/trm/core/trm_tiny_train.py      # Standard training
python scripts/trm/core/trm_tiny_advtrain.py   # Adversarial training (ε=0.15)

# Verification
python scripts/trm/core/trm_tiny_sweep.py --samples 512 --eps 0.01,0.02,0.03,0.04

# Analysis
python scripts/trm/reports/trm_full_visual_report.py
    """, language="bash")
    
    st.markdown("---")
    
    # Key Achievements
    st.markdown("### 🏆 Key Achievements")
    
    achieve_col1, achieve_col2, achieve_col3 = st.columns(3)
    
    with achieve_col1:
        st.metric(
            "Robustness Improvement",
            "67×",
            help="Adversarial vs Standard TRM @ ε=0.01"
        )
    
    with achieve_col2:
        st.metric(
            "Verification Speedup",
            "85%",
            help="Attack-guided vs pure formal verification"
        )
    
    with achieve_col3:
        st.metric(
            "GPU Performance",
            "5.4×",
            help="GPU vs CPU speedup on A100"
        )
    
    st.markdown("---")
    
    # Technical Stack
    st.markdown("### 🛠️ Technical Stack")
    
    tech_col1, tech_col2 = st.columns(2)
    
    with tech_col1:
        st.markdown("""
        **Core Technologies:**
        - PyTorch 2.0+ (CUDA 12.1)
        - auto-LiRPA (α,β-CROWN)
        - Streamlit (UI)
        - VSC-5 HPC cluster
        """)
    
    with tech_col2:
        st.markdown("""
        **Verification Methods:**
        - CROWN (baseline)
        - α-CROWN (optimized)
        - β-CROWN (tightest bounds)
        - FGSM/I-FGSM attacks
        """)
    
    st.markdown("---")
    
    # Future Work
    st.markdown("### 🚀 Next Steps")
    
    st.markdown("""
    **Phase 1 (Current):** ✅ Baseline established @ ε=0.01
    
    **Phase 2 (In Progress):** Test standard benchmarks (ε=0.02, 0.1, 0.3)
    
    **Phase 3 (Planned):** 
    - CROWN-IBP certified training
    - Scale to full 7M parameter TRM
    - Extend to CIFAR-10 dataset
    - Publication at VNN-COMP 2025 / ICLR 2026
    """)

# ============================================================================
# TAB 2: TRM RESULTS
# ============================================================================

with tab2:
    st.header("🧠 TRM Verification Results")
    
    trm_df = load_trm_results()
    
    if trm_df is not None and not trm_df.empty:
        st.success(f"✅ Loaded {len(trm_df)} verification records from logs/")
        
        # Summary Metrics
        st.markdown("### 📊 Summary Statistics")
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        # Calculate key metrics
        if 'model' in trm_df.columns:
            adv_results = trm_df[trm_df['model'].str.contains('Adversarial', na=False)]
            std_results = trm_df[trm_df['model'].str.contains('Standard', na=False)]
            
            with metric_col1:
                if not adv_results.empty and 'verified_fraction' in adv_results.columns:
                    best_adv = adv_results['verified_fraction'].max() * 100
                    st.metric("Adversarial TRM (Best)", f"{best_adv:.1f}%")
                else:
                    st.metric("Adversarial TRM", "N/A")
            
            with metric_col2:
                if not std_results.empty and 'verified_fraction' in std_results.columns:
                    best_std = std_results['verified_fraction'].max() * 100
                    st.metric("Standard TRM (Best)", f"{best_std:.1f}%")
                else:
                    st.metric("Standard TRM", "N/A")
            
            with metric_col3:
                if 'avg_time_s' in trm_df.columns:
                    avg_time = trm_df['avg_time_s'].mean()
                    st.metric("Avg Time/Sample", f"{avg_time:.3f}s")
                else:
                    st.metric("Avg Time/Sample", "N/A")
            
            with metric_col4:
                total_samples = trm_df['total'].sum() if 'total' in trm_df.columns else len(trm_df)
                st.metric("Total Samples", f"{total_samples}")
        
        st.markdown("---")
        
        # Visualization Section
        st.markdown("### 📈 Robustness Analysis")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.markdown("#### Verified Fraction vs Epsilon")
            
            if 'epsilon' in trm_df.columns and 'verified_fraction' in trm_df.columns:
                # Group by model and epsilon
                if 'model' in trm_df.columns:
                    plot_df = trm_df.groupby(['epsilon', 'model'])['verified_fraction'].mean().reset_index()
                    
                    fig1 = px.line(
                        plot_df,
                        x='epsilon',
                        y='verified_fraction',
                        color='model',
                        markers=True,
                        title="Certified Robustness vs Perturbation Size",
                        labels={
                            'verified_fraction': 'Verified Fraction',
                            'epsilon': 'ε (L∞ perturbation)',
                            'model': 'Model Type'
                        }
                    )
                    fig1.update_layout(height=400, hovermode='x unified')
                    st.plotly_chart(fig1, use_container_width=True)
                else:
                    st.warning("Model column not found in data")
            else:
                st.warning("Required columns (epsilon, verified_fraction) not found")
        
        with viz_col2:
            st.markdown("#### Verification Time Analysis")
            
            if 'epsilon' in trm_df.columns and 'avg_time_s' in trm_df.columns:
                if 'model' in trm_df.columns:
                    time_df = trm_df.groupby(['epsilon', 'model'])['avg_time_s'].mean().reset_index()
                    
                    fig2 = px.bar(
                        time_df,
                        x='epsilon',
                        y='avg_time_s',
                        color='model',
                        barmode='group',
                        title="Verification Time vs Epsilon",
                        labels={
                            'avg_time_s': 'Time (seconds)',
                            'epsilon': 'ε (L∞ perturbation)',
                            'model': 'Model Type'
                        }
                    )
                    fig2.update_layout(height=400)
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.warning("Model column not found in data")
            else:
                st.info("Time data not available in results")
        
        st.markdown("---")
        
        # Detailed Results Table
        st.markdown("### 📋 Detailed Results")
        
        # Select columns to display
        display_cols = []
        for col in ['model', 'epsilon', 'verified', 'falsified', 'total', 'verified_fraction', 'avg_time_s', 'avg_mem_MB']:
            if col in trm_df.columns:
                display_cols.append(col)
        
        if display_cols:
            display_df = trm_df[display_cols].copy()
            
            # Format verified_fraction as percentage
            if 'verified_fraction' in display_df.columns:
                display_df['verified_fraction'] = display_df['verified_fraction'].apply(lambda x: f"{x*100:.1f}%")
            
            # Format time
            if 'avg_time_s' in display_df.columns:
                display_df['avg_time_s'] = display_df['avg_time_s'].apply(lambda x: f"{x:.3f}s" if pd.notna(x) else "N/A")
            
            # Format memory
            if 'avg_mem_MB' in display_df.columns:
                display_df['avg_mem_MB'] = display_df['avg_mem_MB'].apply(lambda x: f"{x:.1f} MB" if pd.notna(x) else "N/A")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Export functionality
        st.markdown("---")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            csv_export = trm_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Results (CSV)",
                data=csv_export,
                file_name=f"trm_verification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with export_col2:
            # Check if PDF report exists
            pdf_path = "reports/trm_full_visual_report.pdf"
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    pdf_data = f.read()
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_data,
                    file_name="trm_verification_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.info("PDF report not generated yet")
        
        # Key Findings
        st.markdown("---")
        st.markdown("### 💡 Key Findings")
        
        findings_col1, findings_col2 = st.columns(2)
        
        with findings_col1:
            st.success("""
            **✅ Adversarial Training Works**
            
            Adversarially trained TRM models show **67× improvement** in verified 
            robustness compared to standard training at ε=0.01.
            
            This validates that adversarial training at ε=0.15 effectively 
            improves certified robustness.
            """)
        
        with findings_col2:
            st.info("""
            **📊 Performance Characteristics**
            
            - **Verification time:** <0.25s per sample on A100
            - **GPU memory:** <30MB per sample
            - **Scalability:** Successfully verified 512 samples
            - **Convergence:** Results stable at 256+ samples
            """)
    
    else:
        # No results found - show instructions
        st.warning("⚠️ No TRM verification results found in logs/")
        
        st.markdown("### 🔬 How to Generate Results")
        
        st.markdown("""
        Run the TRM verification pipeline on VSC-5 compute node with GPU:
        """)
        
        st.code("""
# 1. Connect to VSC-5 and request GPU node
ssh veriphi02@vsc5.vsc.ac.at
srun --partition=zen3_0512_a100x2 --gres=gpu:1 --time=04:00:00 --pty bash

# 2. Navigate to project and activate environment
cd ~/veriphi-verification
source venv/bin/activate
export VERIPHI_DEVICE=cuda
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# 3. Run verification sweep (this generates the CSV files)
python scripts/trm/core/trm_tiny_sweep.py --samples 512 --eps 0.01,0.02,0.03,0.04

# 4. Generate visual reports
python scripts/trm/reports/trm_full_visual_report.py

# 5. View results in Streamlit (on login node)
exit  # Return to login node
bash run_streamlit_login.sh
        """, language="bash")
        
        st.markdown("---")
        
        st.markdown("### 📊 Expected Output Structure")
        
        st.markdown("""
        The verification sweep will create:
        
        - `logs/trm_robustness_sweep_YYYYMMDD_HHMMSS.csv` - Raw verification data
        - `reports/trm_full_visual_report.pdf` - Comprehensive PDF report
        - `plots/` - Individual visualization files
        
        Once generated, refresh this page to see the results!
        """)
        
        # Show sample data structure
        st.markdown("### 📋 Sample Data Format")
        
        sample_data = pd.DataFrame({
            'model': ['Adversarial TRM', 'Standard TRM'] * 3,
            'epsilon': [0.01, 0.01, 0.02, 0.02, 0.03, 0.03],
            'verified': [410, 6, 300, 0, 206, 0],
            'falsified': [102, 506, 212, 512, 306, 512],
            'total': [512, 512, 512, 512, 512, 512],
            'verified_fraction': [0.801, 0.012, 0.586, 0.000, 0.402, 0.000],
            'avg_time_s': [0.180, 0.165, 0.195, 0.170, 0.210, 0.175],
            'avg_mem_MB': [28.5, 27.8, 29.1, 28.0, 29.8, 28.3]
        })
        
        st.dataframe(sample_data, use_container_width=True, hide_index=True)
        
        st.caption("*This is sample data showing the expected format. Run the verification pipeline to generate real results.*")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("""
    ### 🎯 Team Veriphi
    - Pratik Deshmukh
    - Vasili Savin
    - Kartik Arya
    
    **Mentors:**
    - Vinay Deshpande (Nvidia)
    - Mark Dokter (ACA)
    """)

with footer_col2:
    st.markdown("""
    ### 📊 Project Stats
    - **512 samples** verified
    - **80% certified** @ ε=0.01
    - **<0.25s** per sample
    - **67× improvement** over baseline
    """)

with footer_col3:
    st.markdown("""
    ### 🔗 Quick Links
    - [Project GitHub](https://github.com/inquisitour/veriphi-verification)
    - [TRM Paper](https://arxiv.org/abs/2510.04871)
    - [VSC-5 Docs](https://vsc.ac.at/systems/vsc-5/)
    - [auto-LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA)
    """)

st.markdown("---")
st.caption("🛡️ Veriphi Verification System | AI Safety Hackathon 2025 | TU Wien")