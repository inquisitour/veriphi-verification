# src/frontend/app.py (Updated for MNIST + CIFAR-10 Results)
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

def load_dataset_results(dataset_name):
    """Load TRM verification results for specific dataset (mnist or cifar10)."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        return None
    
    pattern = f"trm_{dataset_name}_sweep*.csv"
    csvs = glob.glob(os.path.join(log_dir, pattern))
    
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
st.markdown('<p class="sub-header">Cross-Dataset Robustness Verification: MNIST & CIFAR-10</p>', unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("## 🧠 Veriphi Project")
    
    st.markdown("""
    **Neural Network Robustness Verification**
    
    Combining fast adversarial attacks with formal verification methods 
    to certify neural network robustness across datasets.
    """)
    
    st.markdown("---")
    
    st.markdown("### 🎯 Key Results")
    
    st.markdown("**MNIST (28×28):**")
    st.metric("IBP @ ε=0.1", "75%", help="IBP dominates on simple data")
    
    st.markdown("**CIFAR-10 (32×32):**")
    st.metric("PGD @ ε=0.001", "94%", help="PGD dominates on complex data")
    
    st.markdown("---")
    
    st.markdown("### 📊 Training Methods")
    st.markdown("""
    - **Baseline:** Standard training
    - **IBP:** Certified training (1-2/255)
    - **PGD:** Adversarial training (2-8/255)
    """)
    
    st.markdown("---")
    
    st.markdown("### 🔗 Resources")
    st.markdown("""
    - [GitHub Repo](https://github.com/inquisitour/veriphi-verification)
    - [TRM Paper](https://arxiv.org/abs/2510.04871)
    - [α,β-CROWN](https://arxiv.org/abs/2103.06624)
    """)
    
    st.markdown("---")
    st.caption("**AI Safety Hackathon 2025**  \nTU Wien | Team Veriphi")

# ============================================================================
# MAIN TABS
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📚 Project Overview",
    "🔢 MNIST Results", 
    "🖼️ CIFAR-10 Results",
    "🔄 Cross-Dataset Analysis"
])

# ============================================================================
# TAB 1: PROJECT OVERVIEW
# ============================================================================

with tab1:
    st.header("📚 Project Overview")
    
    # Problem Statement
    st.markdown("### ❗ The Challenge")
    st.markdown("""
    **Training method effectiveness depends on dataset complexity:**
    
    - Simple datasets (MNIST) → IBP training excels
    - Complex datasets (CIFAR-10) → PGD training dominates
    - Verification difficulty scales exponentially with input dimensions
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **MNIST (784 dims)**
        - 🥇 IBP: 75-78% @ ε=0.06-0.1
        - 🥈 PGD: 60-65% @ ε=0.08-0.1
        - ❌ Baseline: 0-3%
        """)
    
    with col2:
        st.info("""
        **CIFAR-10 (3072 dims)**
        - 🥇 PGD: 48-95% across all ε
        - 🥈 IBP: Failed (no improvement)
        - ❌ Baseline: 0-82% (small ε only)
        """)
    
    st.markdown("---")
    
    # Key Findings
    st.markdown("### 💡 Key Findings")
    
    findings_col1, findings_col2, findings_col3 = st.columns(3)
    
    with findings_col1:
        st.success("""
        **IBP Works on Simple Data**
        
        Certified training effective on MNIST, achieving 78% verified @ ε=0.08.
        """)
    
    with findings_col2:
        st.success("""
        **PGD Generalizes Better**
        
        Adversarial training robust across both MNIST and CIFAR-10.
        """)
    
    with findings_col3:
        st.warning("""
        **Bounds Have Limits**
        
        α/β-CROWN provide <5% improvement over CROWN.
        """)
    
    st.markdown("---")
    
    # Methodology
    st.markdown("### 🔬 Methodology")
    
    st.markdown("""
    **Training Pipeline:**
    - **Baseline:** Standard supervised training
    - **IBP:** Interval bound propagation training (ε=1-2/255)
    - **PGD:** FGSM adversarial training (ε=2-8/255)
    
    **Verification Setup:**
    - **Hardware:** VSC-5 A100 GPU (80GB)
    - **Framework:** auto-LiRPA (CROWN, α-CROWN, β-CROWN)
    - **Samples:** 512 per model per epsilon
    - **Bounds:** Tested all 3 methods per dataset
    """)

# ============================================================================
# TAB 2: MNIST RESULTS
# ============================================================================

with tab2:
    st.header("🔢 MNIST Verification Results")
    
    mnist_df = load_dataset_results("mnist")
    
    if mnist_df is not None and not mnist_df.empty:
        st.success(f"✅ Loaded {len(mnist_df)} MNIST verification records")
        
        # Summary Table
        st.markdown("### 📊 Summary (512 samples, β-CROWN)")
        
        summary_data = pd.DataFrame({
            'Model': ['Baseline', 'IBP (ε=1/255)', 'PGD (ε=2/255)'],
            'ε=0.01': ['3%', '18%', '30%'],
            'ε=0.04': ['0%', '47%', '43%'],
            'ε=0.06': ['0%', '77%', '63%'],
            'ε=0.08': ['0%', '78%', '65%'],
            'ε=0.1': ['0%', '75%', '60%'],
            'Winner': ['❌', '🥇', '🥈']
        })
        st.dataframe(summary_data, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Visualizations
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.markdown("#### Verified Fraction vs Epsilon")
            
            if 'epsilon' in mnist_df.columns and 'verified_fraction' in mnist_df.columns and 'model' in mnist_df.columns:
                # Use beta-CROWN results
                plot_df = mnist_df[mnist_df['bound'] == 'beta-CROWN'].groupby(['epsilon', 'model'])['verified_fraction'].mean().reset_index()
                
                fig1 = px.line(
                    plot_df,
                    x='epsilon',
                    y='verified_fraction',
                    color='model',
                    markers=True,
                    title="MNIST: Certified Robustness",
                    labels={'verified_fraction': 'Verified Fraction', 'epsilon': 'ε (L∞)'}
                )
                fig1.update_layout(height=400)
                st.plotly_chart(fig1, use_container_width=True)
        
        with viz_col2:
            st.markdown("#### Verification Time")
            
            if 'epsilon' in mnist_df.columns and 'avg_time_s' in mnist_df.columns:
                time_df = mnist_df[mnist_df['bound'] == 'CROWN'].groupby(['epsilon', 'model'])['avg_time_s'].mean().reset_index()
                
                fig2 = px.bar(
                    time_df,
                    x='epsilon',
                    y='avg_time_s',
                    color='model',
                    barmode='group',
                    title="Verification Time (CROWN)",
                    labels={'avg_time_s': 'Time (s)', 'epsilon': 'ε (L∞)'}
                )
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 💡 MNIST Key Finding")
        st.success("""
        **IBP training dominates on simple MNIST**, achieving 75-78% verified accuracy at ε=0.06-0.1.
        PGD competitive but ~15% lower. Baseline completely fails beyond ε=0.01.
        """)
    
    else:
        st.warning("⚠️ No MNIST results found in logs/trm_mnist_sweep*.csv")

# ============================================================================
# TAB 3: CIFAR-10 RESULTS
# ============================================================================

with tab3:
    st.header("🖼️ CIFAR-10 Verification Results")
    
    cifar_df = load_dataset_results("cifar10")
    
    if cifar_df is not None and not cifar_df.empty:
        st.success(f"✅ Loaded {len(cifar_df)} CIFAR-10 verification records")
        
        # Summary Table
        st.markdown("### 📊 Summary (512 samples, β-CROWN)")
        
        summary_data = pd.DataFrame({
            'Model': ['Baseline', 'IBP (ε=2/255)', 'PGD (ε=8/255)'],
            'ε=0.001': ['82%', '78%', '94%'],
            'ε=0.002': ['55%', '51%', '90%'],
            'ε=0.004': ['13%', '10%', '80%'],
            'ε=0.006': ['1%', '1%', '67%'],
            'ε=0.008': ['0%', '0%', '58%'],
            'Winner': ['🥉', '❌', '🥇']
        })
        st.dataframe(summary_data, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Visualizations
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.markdown("#### Verified Fraction vs Epsilon")
            
            if 'epsilon' in cifar_df.columns and 'verified_fraction' in cifar_df.columns and 'model' in cifar_df.columns:
                plot_df = cifar_df[cifar_df['bound'] == 'beta-CROWN'].groupby(['epsilon', 'model'])['verified_fraction'].mean().reset_index()
                
                fig1 = px.line(
                    plot_df,
                    x='epsilon',
                    y='verified_fraction',
                    color='model',
                    markers=True,
                    title="CIFAR-10: Certified Robustness",
                    labels={'verified_fraction': 'Verified Fraction', 'epsilon': 'ε (L∞)'}
                )
                fig1.update_layout(height=400)
                st.plotly_chart(fig1, use_container_width=True)
        
        with viz_col2:
            st.markdown("#### Verification Time")
            
            if 'epsilon' in cifar_df.columns and 'avg_time_s' in cifar_df.columns:
                time_df = cifar_df[cifar_df['bound'] == 'CROWN'].groupby(['epsilon', 'model'])['avg_time_s'].mean().reset_index()
                
                fig2 = px.bar(
                    time_df,
                    x='epsilon',
                    y='avg_time_s',
                    color='model',
                    barmode='group',
                    title="Verification Time (CROWN)",
                    labels={'avg_time_s': 'Time (s)', 'epsilon': 'ε (L∞)'}
                )
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 💡 CIFAR-10 Key Finding")
        st.success("""
        **PGD adversarial training dominates on complex CIFAR-10**, achieving 48-95% verified accuracy 
        across all epsilons. IBP training completely failed - no improvement over baseline.
        """)
    
    else:
        st.warning("⚠️ No CIFAR-10 results found in logs/trm_cifar10_sweep*.csv")

# ============================================================================
# TAB 4: CROSS-DATASET ANALYSIS
# ============================================================================

with tab4:
    st.header("🔄 Cross-Dataset Analysis")
    
    mnist_df = load_dataset_results("mnist")
    cifar_df = load_dataset_results("cifar10")
    
    if mnist_df is not None and cifar_df is not None:
        st.markdown("### 📊 Comparative Analysis")
        
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            st.markdown("#### Dataset Characteristics")
            
            comp_data = pd.DataFrame({
                'Property': ['Input Dimensions', 'Complexity', 'Best Method', 'Best ε Range', 'Peak Verified'],
                'MNIST': ['784', 'Simple', 'IBP @ 1/255', '0.06-0.1', '78% @ ε=0.08'],
                'CIFAR-10': ['3072', 'Complex', 'PGD @ 8/255', '0.001-0.006', '94% @ ε=0.001']
            })
            st.dataframe(comp_data, use_container_width=True, hide_index=True)
        
        with comp_col2:
            st.markdown("#### Training Method Effectiveness")
            
            method_data = pd.DataFrame({
                'Method': ['Baseline', 'IBP', 'PGD'],
                'MNIST': ['Fails', '🥇 Wins', '🥈 Good'],
                'CIFAR-10': ['Poor', '❌ Fails', '🥇 Wins']
            })
            st.dataframe(method_data, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Key Insights
        st.markdown("### 💡 Critical Insights")
        
        insight_col1, insight_col2, insight_col3 = st.columns(3)
        
        with insight_col1:
            st.info("""
            **Dataset Complexity Matters**
            
            IBP works on simple MNIST but fails on complex CIFAR-10. 
            Training method effectiveness depends on data complexity.
            """)
        
        with insight_col2:
            st.success("""
            **PGD Generalizes**
            
            Adversarial training (PGD) works across both datasets, 
            making it more robust for real-world applications.
            """)
        
        with insight_col3:
            st.warning("""
            **Epsilon Scaling**
            
            CIFAR-10 requires 10× smaller ε for same verification rates. 
            Verification difficulty increases exponentially.
            """)
        
        st.markdown("---")
        
        # Recommendations
        st.markdown("### 🎯 Recommendations")
        
        st.success("""
        **For Practitioners:**
        
        1. **Simple data (≤1000 dims):** Use IBP certified training for best verified robustness
        2. **Complex data (>1000 dims):** Use PGD adversarial training - IBP will likely fail
        3. **General case:** PGD is safer choice - works across dataset complexities
        4. **Bound methods:** CROWN sufficient - α/β-CROWN add <5% improvement at higher computational cost
        """)
        
        # Future Work
        st.markdown("---")
        st.markdown("### 🚀 Future Work")
        
        st.markdown("""
        **Next Steps:**
        - Test hybrid IBP+PGD training approaches
        - Scale to ImageNet (224×224, 150K dims)
        - Explore architecture-specific certified training
        - Investigate why IBP fails on complex data
        """)
    
    else:
        st.warning("⚠️ Load both MNIST and CIFAR-10 results to see cross-dataset analysis")

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
    - **1024 samples** verified (512 × 2 datasets)
    - **3 training methods** compared
    - **3 bound methods** tested
    - **18 models** evaluated total
    """)

with footer_col3:
    st.markdown("""
    ### 🔗 Quick Links
    - [GitHub Repo](https://github.com/inquisitour/veriphi-verification)
    - [TRM Paper](https://arxiv.org/abs/2510.04871)
    - [VSC-5 Docs](https://vsc.ac.at/systems/vsc-5/)
    - [auto-LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA)
    """)

st.markdown("---")
st.caption("🛡️ Veriphi Verification System | AI Safety Hackathon 2025 | TU Wien")