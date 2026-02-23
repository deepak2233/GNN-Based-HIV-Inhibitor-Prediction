import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors, Descriptors
from dataset_featurizer import MoleculeDataset
from model.GNN1 import GNN1
from model.GNN2 import GNN2
from model.GNN3 import GNN3
import os
from torch_geometric.data import Data
import io

# Page configuration
st.set_page_config(
    page_title="HIV Inhibitor Predictor", 
    page_icon="🧬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Premium Dark Theme CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background: transparent;
        border-radius: 8px;
        padding: 8px 24px;
        font-weight: 500;
        color: #8892b0;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #e94560, #c23152) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.3);
    }
    
    .glass-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    .prediction-active {
        background: linear-gradient(135deg, rgba(233,69,96,0.15), rgba(194,49,82,0.1));
        border: 1px solid rgba(233, 69, 96, 0.3);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    .prediction-inactive {
        background: linear-gradient(135deg, rgba(0,204,102,0.15), rgba(0,153,76,0.1));
        border: 1px solid rgba(0, 204, 102, 0.3);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    
    .pred-label {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .pred-score {
        font-size: 1.1rem;
        color: #8892b0;
    }
    
    .stat-card {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .stat-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #e94560, #0ea5e9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #8892b0;
        margin-top: 4px;
    }
    
    .model-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .badge-gnn1 { background: rgba(14,165,233,0.2); color: #0ea5e9; }
    .badge-gnn2 { background: rgba(168,85,247,0.2); color: #a855f7; }
    .badge-gnn3 { background: rgba(233,69,96,0.2); color: #e94560; }
    
    .descriptor-table {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        overflow: hidden;
    }
    
    h1, h2, h3 { color: #e6e6e6 !important; }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a1a2e, #16213e);
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #e94560, #0ea5e9) !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Feature Sizes ───────────────────────────────────────────────────────────
FEATURE_SIZE = 9
EDGE_FEATURE_SIZE = 2

# ─── Model Loading ───────────────────────────────────────────────────────────
@st.cache_resource
def load_trained_model(m_type, path):
    if not os.path.exists(path):
        return None
    
    if m_type == 'GNN1':
        model = GNN1(feature_size=FEATURE_SIZE)
    elif m_type == 'GNN2':
        model = GNN2(feature_size=FEATURE_SIZE)
    else:
        model = GNN3(feature_size=FEATURE_SIZE, edge_feature_size=EDGE_FEATURE_SIZE)
        
    try:
        checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model
    except RuntimeError:
        return None
    except Exception:
        return None

def featurize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None
    
    node_feats = []
    for atom in mol.GetAtoms():
        node_feats.append([
            atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(),
            int(atom.GetHybridization()), int(atom.GetIsAromatic()),
            atom.GetTotalNumHs(), atom.GetNumRadicalElectrons(),
            int(atom.IsInRing()), int(atom.GetChiralTag())
        ])
    
    edge_indices = []
    edge_feats = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices += [[i, j], [j, i]]
        ef = [bond.GetBondTypeAsDouble(), int(bond.IsInRing())]
        edge_feats += [ef, ef]
    
    if len(edge_indices) == 0:
        return None
        
    x = torch.tensor(np.array(node_feats), dtype=torch.float)
    edge_index = torch.tensor(edge_indices).t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(np.array(edge_feats), dtype=torch.float)
    batch = torch.zeros(x.shape[0], dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)


def predict_single(model, smiles):
    """Run prediction on a single SMILES. Returns (prob, prediction) or (None, None)."""
    data = featurize_smiles(smiles)
    if data is None:
        return None, None
    with torch.no_grad():
        out = model(data.x, data.edge_attr, data.edge_index, data.batch)
        prob = torch.sigmoid(out).item()
        pred = 1 if prob > 0.5 else 0
    return prob, pred


@st.cache_data
def load_dataset_for_eda():
    """Load the HIV dataset for EDA, using real data."""
    paths = ['data/split_data/HIV_train.csv', 'data/split_data/HIV_test.csv', 'data/raw_data/HIV_data.csv']
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

@st.cache_data
def compute_molecular_weights(smiles_list, max_samples=2000):
    """Compute real molecular weights from SMILES."""
    weights = []
    for sm in smiles_list[:max_samples]:
        mol = Chem.MolFromSmiles(sm)
        if mol:
            weights.append(Descriptors.MolWt(mol))
    return weights

@st.cache_data
def compute_logp_values(smiles_list, max_samples=2000):
    """Compute LogP values from SMILES."""
    logps = []
    for sm in smiles_list[:max_samples]:
        mol = Chem.MolFromSmiles(sm)
        if mol:
            logps.append(Descriptors.MolLogP(mol))
    return logps

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h2 style='background: linear-gradient(135deg, #e94560, #0ea5e9); 
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    font-weight: 700; font-size: 1.6rem;'>
            🧬 HIV-GNN Explorer
        </h2>
        <p style='color: #8892b0; font-size: 0.85rem;'>
            Graph Neural Network Drug Discovery
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### Model Selection")
    sel_model = st.selectbox(
        "Architecture", ["GNN1", "GNN2", "GNN3"],
        help="GNN1: Baseline GAT | GNN2: GIN+Transformer | GNN3: Edge-Aware"
    )
    
    default_path = os.path.join("outputs", sel_model, "best_model.pth")
    use_custom = st.checkbox("Custom weights path", False)
    if use_custom:
        sel_path = st.text_input("Path", default_path)
    else:
        sel_path = default_path
    
    model_loaded = os.path.exists(sel_path)
    if model_loaded:
        st.success(f"Weights: `{sel_path}`")
    else:
        st.warning("No weights found. Train first.")
    
    st.markdown("---")
    st.markdown("""
    <div class='glass-card' style='padding: 16px;'>
        <p style='color: #8892b0; font-size: 0.8rem; margin: 0;'>
            <strong>Architecture Details</strong><br>
            <span class='model-badge badge-gnn1'>GNN1</span> 3× GAT + TopK Pool<br>
            <span class='model-badge badge-gnn2'>GNN2</span> GIN → Transformer → GAT<br>
            <span class='model-badge badge-gnn3'>GNN3</span> Edge-aware GIN+Trans+GAT
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("Built with PyTorch Geometric")

# ─── Main Tabs ───────────────────────────────────────────────────────────────
tab_predict, tab_batch, tab_eda, tab_arch = st.tabs([
    "🎯 Predict", "📋 Batch Predict", "📊 Dataset Analysis", "🏗️ Architecture"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Single Prediction
# ═══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("<h2 style='margin-bottom: 4px;'>🧬 HIV Inhibitor Prediction System</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #8892b0; margin-top: 0;'>Enter a SMILES string to predict HIV inhibitor activity</p>", unsafe_allow_html=True)
    
    # Quick examples
    st.markdown("##### Quick Examples")
    ex_cols = st.columns(4)
    examples = {
        "Efavirenz (Active)": "C1=CC=C2C(=C1)C(OC(=O)N2)(C#CC3CC3)C(F)(F)F",
        "Tenofovir (Active)": "CC(COP(=O)(O)O)N1C=NC2=C1N=CN=C2N",
        "Aspirin (Inactive)": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Caffeine (Inactive)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    }
    selected_smiles = None
    for idx, (name, sm) in enumerate(examples.items()):
        with ex_cols[idx]:
            if st.button(name, key=f"ex_{idx}"):
                selected_smiles = sm

    input_smiles = st.text_input(
        "SMILES String", 
        selected_smiles if selected_smiles else "CCC1=C(C)C=C(C=C1)C2=CC=CC=C2",
        placeholder="Enter a valid SMILES string..."
    )
    
    if input_smiles:
        mol = Chem.MolFromSmiles(input_smiles)
        if mol:
            col_mol, col_pred, col_desc = st.columns([1, 1, 1])
            
            with col_mol:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown("##### Molecular Structure")
                img = Draw.MolToImage(mol, size=(400, 350))
                st.image(img, use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col_pred:
                model = load_trained_model(sel_model, sel_path)
                if model:
                    prob, pred = predict_single(model, input_smiles)
                    if prob is not None:
                        css_class = "prediction-active" if pred == 1 else "prediction-inactive"
                        label = "HIV ACTIVE" if pred == 1 else "HIV INACTIVE"
                        color = "#e94560" if pred == 1 else "#00cc66"
                        
                        st.markdown(f"""
                        <div class='{css_class}'>
                            <div class='pred-label' style='color: {color};'>{label}</div>
                            <div class='pred-score'>Confidence: {prob:.4f}</div>
                            <div style='margin-top: 12px;'>
                                <span class='model-badge badge-{"gnn1" if sel_model=="GNN1" else "gnn2" if sel_model=="GNN2" else "gnn3"}'>
                                    {sel_model}
                                </span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("**Probability Score**")
                        st.progress(prob)
                    else:
                        st.error("Could not featurize molecule (no bonds found).")
                else:
                    st.info(f"Train {sel_model} first: `python main.py --mode train --model_type {sel_model}`")
            
            with col_desc:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown("##### Molecular Properties")
                
                props = {
                    "Formula": rdMolDescriptors.CalcMolFormula(mol),
                    "Mol. Weight": f"{Descriptors.MolWt(mol):.2f} Da",
                    "LogP": f"{Descriptors.MolLogP(mol):.2f}",
                    "H-Bond Donors": rdMolDescriptors.CalcNumHBD(mol),
                    "H-Bond Acceptors": rdMolDescriptors.CalcNumHBA(mol),
                    "Rotatable Bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
                    "TPSA": f"{Descriptors.TPSA(mol):.1f} Å²",
                    "Rings": rdMolDescriptors.CalcNumRings(mol),
                    "Aromatic Rings": rdMolDescriptors.CalcNumAromaticRings(mol),
                    "Atoms": mol.GetNumAtoms(),
                    "Bonds": mol.GetNumBonds(),
                }
                
                for k, v in props.items():
                    st.markdown(f"**{k}:** {v}")
                
                # Lipinski Rule of 5 check
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                hbd = rdMolDescriptors.CalcNumHBD(mol)
                hba = rdMolDescriptors.CalcNumHBA(mol)
                violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
                
                if violations == 0:
                    st.success(f"Lipinski Ro5: PASS (0 violations)")
                elif violations <= 1:
                    st.warning(f"Lipinski Ro5: {violations} violation")
                else:
                    st.error(f"Lipinski Ro5: {violations} violations")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Performance section
            st.markdown("---")
            st.markdown("##### Model Performance")
            cm_path = os.path.join("outputs", sel_model, "confusion_matrix.png")
            if os.path.exists(cm_path):
                cc1, cc2 = st.columns([1, 2])
                with cc1:
                    st.image(cm_path, caption=f"Confusion Matrix — {sel_model}", use_column_width=True)
                with cc2:
                    st.markdown("""
                    <div class='glass-card'>
                        <h4>Understanding the Confusion Matrix</h4>
                        <p style='color: #8892b0;'>
                        <strong>True Negatives (top-left):</strong> Inactive molecules correctly predicted as inactive.<br>
                        <strong>False Positives (top-right):</strong> Inactive molecules incorrectly predicted as active.<br>
                        <strong>False Negatives (bottom-left):</strong> Active molecules missed by the model.<br>
                        <strong>True Positives (bottom-right):</strong> Active molecules correctly identified.
                        </p>
                        <p style='color: #8892b0;'>
                        For drug discovery, <strong>minimizing False Negatives</strong> is critical — we don't want to miss 
                        potentially active compounds. The <code>pos_weight</code> in BCEWithLogitsLoss controls this tradeoff.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info(f"No confusion matrix found for {sel_model}. Run training first.")
        else:
            st.error("Invalid SMILES string. Please check the format.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Batch Prediction
# ═══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("<h2 style='margin-bottom: 4px;'>Batch Prediction</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #8892b0;'>Upload a CSV with a 'smiles' column to predict activity for multiple molecules at once.</p>", unsafe_allow_html=True)
    
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload")
    
    if uploaded:
        df = pd.read_csv(uploaded)
        smiles_col = None
        for col in df.columns:
            if col.lower() in ['smiles', 'smile', 'smi', 'canonical_smiles']:
                smiles_col = col
                break
        
        if smiles_col is None:
            st.error("No 'smiles' column found. Please ensure your CSV has a column named 'smiles'.")
        else:
            st.markdown(f"**Found {len(df)} molecules** in column `{smiles_col}`")
            
            model = load_trained_model(sel_model, sel_path)
            if model:
                if st.button("Run Batch Prediction", type="primary"):
                    results = []
                    progress = st.progress(0)
                    status = st.empty()
                    
                    for i, smiles in enumerate(df[smiles_col]):
                        prob, pred = predict_single(model, str(smiles))
                        results.append({
                            'smiles': smiles,
                            'probability': prob if prob is not None else np.nan,
                            'prediction': 'Active' if pred == 1 else 'Inactive' if pred == 0 else 'Error',
                        })
                        progress.progress((i + 1) / len(df))
                        status.text(f"Processing molecule {i+1}/{len(df)}...")
                    
                    progress.empty()
                    status.empty()
                    
                    results_df = pd.DataFrame(results)
                    
                    # Summary stats
                    valid = results_df.dropna(subset=['probability'])
                    n_active = (valid['prediction'] == 'Active').sum()
                    n_inactive = (valid['prediction'] == 'Inactive').sum()
                    n_error = (results_df['prediction'] == 'Error').sum()
                    
                    sc1, sc2, sc3, sc4 = st.columns(4)
                    with sc1:
                        st.markdown(f"<div class='stat-card'><div class='stat-value'>{len(df)}</div><div class='stat-label'>Total Molecules</div></div>", unsafe_allow_html=True)
                    with sc2:
                        st.markdown(f"<div class='stat-card'><div class='stat-value' style='background: linear-gradient(135deg, #e94560, #ff6b6b); -webkit-background-clip: text;'>{n_active}</div><div class='stat-label'>Predicted Active</div></div>", unsafe_allow_html=True)
                    with sc3:
                        st.markdown(f"<div class='stat-card'><div class='stat-value' style='background: linear-gradient(135deg, #00cc66, #00ff88); -webkit-background-clip: text;'>{n_inactive}</div><div class='stat-label'>Predicted Inactive</div></div>", unsafe_allow_html=True)
                    with sc4:
                        st.markdown(f"<div class='stat-card'><div class='stat-value'>{n_error}</div><div class='stat-label'>Failed</div></div>", unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.dataframe(results_df, height=400)
                    
                    # Download button
                    csv_buf = results_df.to_csv(index=False)
                    st.download_button(
                        "Download Results CSV",
                        csv_buf,
                        file_name=f"hiv_predictions_{sel_model}.csv",
                        mime="text/csv"
                    )
            else:
                st.info(f"Please train {sel_model} first: `python main.py --mode train --model_type {sel_model}`")
    else:
        st.markdown("""
        <div class='glass-card' style='text-align: center; padding: 60px 20px;'>
            <h3 style='color: #8892b0;'>Upload a CSV file to get started</h3>
            <p style='color: #555;'>Your CSV should have a column named <code>smiles</code> containing SMILES strings.</p>
            <p style='color: #555; font-size: 0.85rem;'>Example: <code>CC(=O)OC1=CC=CC=C1C(=O)O</code></p>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Dataset EDA
# ═══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown("<h2 style='margin-bottom: 4px;'>Exploratory Data Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #8892b0;'>Insights from the DTP AIDS Antiviral Screen dataset</p>", unsafe_allow_html=True)
    
    eda_df = load_dataset_for_eda()
    
    if eda_df is not None and 'HIV_active' in eda_df.columns and 'smiles' in eda_df.columns:
        n_total = len(eda_df)
        n_inactive = (eda_df['HIV_active'] == 0).sum()
        n_active = (eda_df['HIV_active'] == 1).sum()
        ratio = n_inactive / max(n_active, 1)
        
        # Top stats
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(f"<div class='stat-card'><div class='stat-value'>{n_total:,}</div><div class='stat-label'>Total Molecules</div></div>", unsafe_allow_html=True)
        with s2:
            st.markdown(f"<div class='stat-card'><div class='stat-value'>{n_inactive:,}</div><div class='stat-label'>Inactive (Class 0)</div></div>", unsafe_allow_html=True)
        with s3:
            st.markdown(f"<div class='stat-card'><div class='stat-value'>{n_active:,}</div><div class='stat-label'>Active (Class 1)</div></div>", unsafe_allow_html=True)
        with s4:
            st.markdown(f"<div class='stat-card'><div class='stat-value'>1:{ratio:.0f}</div><div class='stat-label'>Imbalance Ratio</div></div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_eda1, col_eda2 = st.columns(2)
        
        with col_eda1:
            st.markdown("##### Class Distribution")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            fig1.patch.set_facecolor('#0f0f1a')
            ax1.set_facecolor('#0f0f1a')
            bars = ax1.bar(['Inactive (0)', 'Active (1)'], [n_inactive, n_active], 
                          color=['#0ea5e9', '#e94560'], width=0.5, edgecolor='none')
            for bar, val in zip(bars, [n_inactive, n_active]):
                ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 200, 
                        f'{val:,}', ha='center', va='bottom', color='#8892b0', fontweight='bold')
            ax1.set_ylabel('Count', color='#8892b0')
            ax1.tick_params(colors='#8892b0')
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['bottom'].set_color('#333')
            ax1.spines['left'].set_color('#333')
            st.pyplot(fig1)
            plt.close(fig1)
        
        with col_eda2:
            st.markdown("##### Molecular Weight Distribution (Real Data)")
            smiles_list = eda_df['smiles'].dropna().tolist()
            weights = compute_molecular_weights(smiles_list)
            
            if weights:
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                fig2.patch.set_facecolor('#0f0f1a')
                ax2.set_facecolor('#0f0f1a')
                ax2.hist(weights, bins=50, color='#e94560', alpha=0.7, edgecolor='none')
                ax2.axvline(x=500, color='#ffd700', linestyle='--', alpha=0.7, label='Ro5 Limit (500 Da)')
                ax2.set_xlabel('Molecular Weight (Da)', color='#8892b0')
                ax2.set_ylabel('Count', color='#8892b0')
                ax2.tick_params(colors='#8892b0')
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                ax2.spines['bottom'].set_color('#333')
                ax2.spines['left'].set_color('#333')
                ax2.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='#8892b0')
                st.pyplot(fig2)
                plt.close(fig2)
        
        col_eda3, col_eda4 = st.columns(2)
        
        with col_eda3:
            st.markdown("##### LogP Distribution (Real Data)")
            logps = compute_logp_values(smiles_list)
            if logps:
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                fig3.patch.set_facecolor('#0f0f1a')
                ax3.set_facecolor('#0f0f1a')
                
                # Split by class
                active_smiles = eda_df[eda_df['HIV_active'] == 1]['smiles'].dropna().tolist()[:1000]
                inactive_smiles = eda_df[eda_df['HIV_active'] == 0]['smiles'].dropna().tolist()[:1000]
                active_logp = compute_logp_values(active_smiles)
                inactive_logp = compute_logp_values(inactive_smiles)
                
                ax3.hist(inactive_logp, bins=40, color='#0ea5e9', alpha=0.5, label='Inactive', edgecolor='none')
                ax3.hist(active_logp, bins=40, color='#e94560', alpha=0.6, label='Active', edgecolor='none')
                ax3.set_xlabel('LogP', color='#8892b0')
                ax3.set_ylabel('Count', color='#8892b0')
                ax3.tick_params(colors='#8892b0')
                ax3.spines['top'].set_visible(False)
                ax3.spines['right'].set_visible(False)
                ax3.spines['bottom'].set_color('#333')
                ax3.spines['left'].set_color('#333')
                ax3.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='#8892b0')
                st.pyplot(fig3)
                plt.close(fig3)
        
        with col_eda4:
            st.markdown("##### Sample Molecules")
            sample_df = eda_df.head(50)[['smiles', 'HIV_active']].copy()
            sample_df['HIV_active'] = sample_df['HIV_active'].map({0: 'Inactive', 1: 'Active'})
            st.dataframe(sample_df, height=300)
    else:
        st.warning("Dataset not found. Place `HIV_data.csv` in `data/raw_data/` or split CSVs in `data/split_data/`.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: Architecture Blog
# ═══════════════════════════════════════════════════════════════════════════════
with tab_arch:
    st.markdown("<h2 style='margin-bottom: 4px;'>GNN Architecture Deep Dive</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #8892b0;'>How Graph Neural Networks process molecules for drug discovery</p>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='glass-card'>
        <h3>1. Molecule → Graph Representation</h3>
        <p style='color: #ccd6f6;'>
        Molecules are naturally represented as graphs where <strong>atoms = nodes</strong> and <strong>bonds = edges</strong>.
        </p>
        <table style='width: 100%; color: #8892b0; border-collapse: separate; border-spacing: 0 4px;'>
        <tr><td style='padding: 6px 12px; background: rgba(255,255,255,0.03); border-radius: 6px;'><strong>Node Features (9)</strong></td>
            <td style='padding: 6px 12px;'>Atomic number, Degree, Formal charge, Hybridization, Aromaticity, H-count, Radical electrons, Ring membership, Chirality</td></tr>
        <tr><td style='padding: 6px 12px; background: rgba(255,255,255,0.03); border-radius: 6px;'><strong>Edge Features (2)</strong></td>
            <td style='padding: 6px 12px;'>Bond type (single/double/triple/aromatic), Ring membership</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='glass-card'>
        <h3>2. Model Architectures</h3>
    </div>
    """, unsafe_allow_html=True)
    
    arch_cols = st.columns(3)
    arch_info = [
        ("GNN1", "badge-gnn1", "Baseline GAT", 
         "3× GAT layers with multi-head attention (3 heads each). Each block includes BatchNorm → ReLU → TopKPooling. Multi-scale graph representations are concatenated for classification.",
         "Simple, fast, good baseline"),
        ("GNN2", "badge-gnn2", "GIN + Transformer + GAT",
         "GINConv (Graph Isomorphism Network) for WL-test expressiveness, followed by TransformerConv for long-range dependencies, then 3× GAT blocks with TopKPooling.",
         "Better structural awareness"),
        ("GNN3", "badge-gnn3", "Edge-Aware Transformer",
         "Same as GNN2 but explicitly passes edge attributes (bond types) through the GAT attention mechanism. This allows the model to weight neighbor contributions differently based on bond chemistry.",
         "Best design — uses bond info"),
    ]
    
    for i, (name, badge, title, desc, note) in enumerate(arch_info):
        with arch_cols[i]:
            st.markdown(f"""
            <div class='glass-card' style='height: 100%;'>
                <span class='model-badge {badge}'>{name}</span>
                <h4>{title}</h4>
                <p style='color: #8892b0; font-size: 0.9rem;'>{desc}</p>
                <p style='color: #ccd6f6; font-size: 0.8rem; font-style: italic;'>→ {note}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='glass-card'>
        <h3>3. Handling Class Imbalance</h3>
        <p style='color: #ccd6f6;'>
        The HIV dataset is severely imbalanced (~3.5% active). Standard training would predict "inactive" every time and achieve 96.5% accuracy.
        We use <strong>BCEWithLogitsLoss</strong> with a positive class weight:
        </p>
        <div style='background: rgba(0,0,0,0.3); padding: 16px; border-radius: 8px; font-family: monospace; color: #0ea5e9;'>
        Loss = -w<sub>pos</sub> · y · log σ(x) - (1-y) · log(1 - σ(x))
        </div>
        <p style='color: #8892b0; margin-top: 12px;'>
        where <code>w_pos ≈ 15-26</code> is auto-computed from the train label ratio. This forces the model to pay 
        much more attention to the rare active compounds, trading some precision for better recall.
        </p>
    </div>
    
    <div class='glass-card'>
        <h3>4. Training Strategy</h3>
        <p style='color: #ccd6f6;'>Key improvements in the training pipeline:</p>
        <ul style='color: #8892b0;'>
            <li><strong>ReduceLROnPlateau scheduler</strong> — halves learning rate when F1 stalls</li>
            <li><strong>Early stopping</strong> — prevents overfitting with configurable patience</li>
            <li><strong>Data-driven pos_weight</strong> — auto-computed from label distribution</li>
            <li><strong>BatchNorm</strong> — stabilizes training with aggressive TopKPooling</li>
            <li><strong>Multi-scale concatenation</strong> — [x1; x2; x3] preserves more information than sum</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
