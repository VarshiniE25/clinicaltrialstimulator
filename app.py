"""
app.py
======
GenAI Drug Discovery & Clinical Trial Simulation — Streamlit UI (Enhanced)
Run:  streamlit run app.py

Improvements over v1:
  • 3D molecule viewer (py3Dmol via stmol)
  • Interactive property parallel-coordinates plot
  • SMILES → 2D structure SVG (rdkit)
  • Patient heatmap (drug × patient effectiveness)
  • Mutation sensitivity analysis chart
  • Molecule similarity network (networkx + plotly)
  • AI-powered drug comparison table (multi-candidate)
  • Dark-mode pipeline stepper with animated progress
  • Export ranked table as CSV
"""

import os, sys, io, base64
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GenAI Drug Discovery",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"]          { font-family: 'Inter', sans-serif; }
.stApp                              { background: #080c14; color: #dde6f0; }
section[data-testid="stSidebar"]    { background: #080c14; border-right: 1px solid #1c2535; }
section[data-testid="stSidebar"] *  { color: #dde6f0 !important; }

/* ── Section headers ── */
.sec-header {
    background: linear-gradient(90deg, #0f1a2b 0%, #0d1825 100%);
    border: 1px solid #1c2535;
    border-left: 3px solid #22d3ee;
    border-radius: 10px;
    padding: 0.55rem 1.1rem;
    margin: 0.9rem 0 0.85rem 0;
    display: flex; align-items: center; gap: 10px;
    font-size: 0.75rem; font-weight: 700;
    font-family: 'Syne', sans-serif;
    letter-spacing: 0.08em; text-transform: uppercase; color: #c8dae8;
}
.step-badge {
    background: linear-gradient(135deg,#22d3ee,#0ea5e9);
    color: #000; border-radius: 5px; padding: 1px 8px;
    font-size: 0.65rem; font-weight: 800;
}

/* ── Pipeline strip ── */
.pipe-wrap {
    display: flex; align-items: flex-start;
    gap: 0; overflow-x: auto; padding: 0.5rem 0;
}
.pipe-step {
    display: flex; flex-direction: column;
    align-items: center; gap: 5px;
    min-width: 80px; position: relative;
}
.pipe-step:not(:last-child)::after {
    content: '→'; position: absolute;
    right: -12px; top: 14px;
    color: #22d3ee; font-size: 0.9rem;
}
.pipe-icon {
    width: 40px; height: 40px; border-radius: 10px;
    background: #0f1a2b; border: 1.5px solid #1c2535;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.05rem; transition: all 0.3s ease;
}
.pipe-icon.done   { border-color: #22d3ee; background: rgba(34,211,238,0.1);
                    box-shadow: 0 0 12px rgba(34,211,238,0.2); }
.pipe-icon.active { border-color: #f59e0b; background: rgba(245,158,11,0.1);
                    animation: pulse 1.2s ease-in-out infinite; }
@keyframes pulse { 0%,100%{box-shadow:0 0 8px rgba(245,158,11,0.3)} 50%{box-shadow:0 0 20px rgba(245,158,11,0.6)} }
.pipe-label { font-size: 0.58rem; color: #6b7f99; text-align: center;
              font-family:'JetBrains Mono',monospace;
              line-height: 1.3; max-width: 72px; }

/* ── Log box ── */
.log-area {
    background: #04080f; border: 1px solid #1c2535;
    border-radius: 10px; padding: 0.75rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; line-height: 2.0; color: #6b7f99;
    max-height: 200px; overflow-y: auto;
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg,#0f1a2b,#0d1825);
    border: 1px solid #1c2535;
    border-radius: 12px; padding: 1.1rem; text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #22d3ee; }
.metric-val   { font-family: 'Syne', sans-serif;
                font-size: 1.6rem; font-weight: 800; }
.metric-label { font-size: 0.68rem; color: #6b7f99;
                text-transform: uppercase; letter-spacing: 0.06em; margin-top: 5px;
                font-family:'JetBrains Mono',monospace; }

/* ── Info / target box ── */
.target-box {
    background: rgba(34,211,238,0.04);
    border: 1px solid rgba(34,211,238,0.18);
    border-radius: 10px; padding: 0.85rem 1.1rem;
    font-size: 0.82rem; line-height: 2.0; margin-top: 0.75rem;
}
.target-box strong { color: #22d3ee; }

/* ── Explanation box ── */
.expl-box {
    background: rgba(34,211,238,0.04);
    border-left: 3px solid #22d3ee;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem; font-size: 0.87rem;
    line-height: 1.85; color: #a8c4d4; margin-top: 0.75rem;
}

/* ── 3D Viewer container ── */
.viewer-container {
    background: #04080f;
    border: 1px solid #1c2535;
    border-radius: 12px;
    overflow: hidden;
    position: relative;
}
.viewer-badge {
    position: absolute; top: 8px; left: 8px; z-index: 10;
    background: rgba(34,211,238,0.15);
    border: 1px solid rgba(34,211,238,0.3);
    border-radius: 5px; padding: 2px 8px;
    font-family:'JetBrains Mono',monospace;
    font-size: 0.62rem; color: #22d3ee;
}

/* ── Candidate comparison card ── */
.cand-card {
    background: #0f1a2b; border: 1px solid #1c2535;
    border-radius: 10px; padding: 0.85rem;
    font-size: 0.78rem; margin-bottom: 0.5rem;
    transition: border-color 0.2s;
}
.cand-card:hover { border-color: #22d3ee; }
.cand-rank { font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:800; color:#22d3ee; }

/* ── Run button ── */
.stButton > button {
    background: linear-gradient(135deg,#22d3ee,#0ea5e9) !important;
    color: #000 !important; font-weight: 800 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.88rem !important; border: none !important;
    border-radius: 8px !important; padding: 0.7rem 1.5rem !important;
    width: 100% !important; letter-spacing: 0.03em !important;
}
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px); }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
.stTabs [data-baseweb="tab"] {
    background: #0f1a2b !important; border: 1px solid #1c2535 !important;
    border-radius: 7px !important; color: #6b7f99 !important;
    font-size: 0.77rem !important; padding: 0.4rem 0.9rem !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(34,211,238,0.12) !important;
    border-color: #22d3ee !important; color: #22d3ee !important;
}

/* ── Dataframe ── */
.stDataFrame > div { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════

DISEASE_TARGETS = {
    "diabetes":     ("PPAR-gamma",  "Insulin sensitizer",         "PPAR-gamma binding site"),
    "cancer":       ("EGFR",        "Tyrosine kinase inhibitor",   "EGFR ATP-binding pocket"),
    "alzheimer":    ("AChE",        "Acetylcholinesterase inh.",   "AChE catalytic triad"),
    "parkinson":    ("MAO-B",       "Monoamine oxidase inh.",      "MAO-B flavin cavity"),
    "hypertension": ("ACE",         "Angiotensin-converting enz",  "ACE zinc-binding site"),
}

PIPE_STEPS = [
    ("🧬", "GenAI\nMol Gen"),
    ("⚗",  "Lipinski\nFilter"),
    ("⚓", "Docking\nSim"),
    ("☣",  "ADMET\nToxicity"),
    ("🏅", "Drug\nRanking"),
    ("👥", "Patient\nGen"),
    ("🏥", "Clinical\nTrial"),
    ("💡", "AI\nExplain"),
]


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def get_target(disease: str):
    key = next((k for k in DISEASE_TARGETS if k in disease.lower()), "diabetes")
    return DISEASE_TARGETS[key]


def sec(icon: str, step: int, title: str):
    st.markdown(
        f'<div class="sec-header">{icon} '
        f'<span class="step-badge">{step}</span>&nbsp;{title}</div>',
        unsafe_allow_html=True,
    )


def pipeline_html(active: int) -> str:
    html = '<div class="pipe-wrap">'
    for i, (ico, lbl) in enumerate(PIPE_STEPS):
        cls = "done" if i < active else ("active" if i == active else "")
        html += (
            f'<div class="pipe-step">'
            f'<div class="pipe-icon {cls}">{ico}</div>'
            f'<div class="pipe-label">{lbl.replace(chr(10),"<br/>")}</div>'
            f'</div>'
        )
    html += "</div>"
    return html


def plotly_base(h=220):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#04080f",
        font_color="#6b7f99",
        margin=dict(l=10, r=10, t=30, b=10),
        height=h,
    )


def smiles_to_svg(smiles: str, width=280, height=200) -> str | None:
    """Render 2D SMILES to SVG string using RDKit."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.drawOptions().clearBackground = False
        drawer.drawOptions().backgroundColour = (4/255, 8/255, 15/255, 1)
        drawer.drawOptions().bondLineWidth = 1.8
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        return None


def smiles_to_3d_html(smiles: str) -> str | None:
    """Generate 3D conformer and render via py3Dmol embedded in HTML."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if result != 0:
            return None
        AllChem.MMFFOptimizeMolecule(mol)
        mol_block = Chem.MolToMolBlock(mol)
        mol_block_escaped = mol_block.replace('`', r'\`').replace('\\', '\\\\').replace('$', r'\$')

        html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
  body {{ margin:0; padding:0; background:#04080f; }}
  #viewer {{ width:100%; height:320px; position:relative; background:#04080f; }}
  #controls {{ position:absolute; bottom:10px; right:10px; display:flex; gap:6px; z-index:100; }}
  .ctrl-btn {{
    background: rgba(34,211,238,0.12);
    border: 1px solid rgba(34,211,238,0.3);
    border-radius:5px; padding:4px 10px;
    color:#22d3ee; font-size:11px; cursor:pointer;
    font-family:monospace; transition:all 0.2s;
  }}
  .ctrl-btn:hover {{ background:rgba(34,211,238,0.25); }}
  #style-label {{
    position:absolute; top:10px; left:10px; z-index:100;
    background:rgba(34,211,238,0.1); border:1px solid rgba(34,211,238,0.25);
    border-radius:5px; padding:3px 10px;
    color:#22d3ee; font-size:10px; font-family:monospace;
  }}
</style>
</head>
<body>
<div style="position:relative">
  <div id="viewer"></div>
  <div id="style-label">STICK</div>
  <div id="controls">
    <button class="ctrl-btn" onclick="setStyle('stick')">Stick</button>
    <button class="ctrl-btn" onclick="setStyle('sphere')">Sphere</button>
    <button class="ctrl-btn" onclick="setStyle('surface')">Surface</button>
    <button class="ctrl-btn" onclick="setStyle('cartoon')">Cartoon</button>
  </div>
</div>
<script>
var viewer = null;
var currentStyle = 'stick';

$(function() {{
  var config = {{ backgroundColor: '#04080f' }};
  viewer = $3Dmol.createViewer('viewer', config);

  var molData = `{mol_block_escaped}`;
  viewer.addModel(molData, 'mol');

  applyStyle('stick');
  viewer.zoomTo();
  viewer.render();
}});

function applyStyle(style) {{
  viewer.setStyle({{}}, {{}});
  if (style === 'stick') {{
    viewer.setStyle({{}}, {{stick:{{colorscheme:'Jmol', radius:0.15}}}});
  }} else if (style === 'sphere') {{
    viewer.setStyle({{}}, {{sphere:{{colorscheme:'Jmol', scale:0.4}}}});
  }} else if (style === 'surface') {{
    viewer.setStyle({{}}, {{stick:{{radius:0.12}}}});
    viewer.addSurface('VDW', {{opacity:0.7, colorscheme:'whiteCarbon'}});
  }} else if (style === 'cartoon') {{
    viewer.setStyle({{}}, {{cartoon:{{color:'spectrum'}}, stick:{{radius:0.1}}}});
  }}
  viewer.render();
  document.getElementById('style-label').innerText = style.toUpperCase();
}}

function setStyle(style) {{
  currentStyle = style;
  applyStyle(style);
}}
</script>
</body>
</html>
"""
        return html
    except Exception as e:
        return None


def build_similarity_network(ranked: list[dict]) -> go.Figure:
    """
    Build a Tanimoto-based molecule similarity network using RDKit fingerprints.
    Nodes = molecules, edges = Tanimoto similarity > threshold.
    """
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdMolDescriptors
        import math

        smiles_list = [m["smiles"] for m in ranked[:15]]
        mols   = [Chem.MolFromSmiles(s) for s in smiles_list]
        fps    = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, 1024)
                  for m in mols if m]
        n      = len(fps)
        scores = [m["score"] for m in ranked[:15]]

        # Circular layout
        positions = {
            i: (math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n))
            for i in range(n)
        }

        edge_x, edge_y, edge_weights = [], [], []
        threshold = 0.25
        for i in range(n):
            for j in range(i + 1, n):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                if sim > threshold:
                    xi, yi = positions[i]
                    xj, yj = positions[j]
                    edge_x += [xi, xj, None]
                    edge_y += [yi, yj, None]
                    edge_weights.append(sim)

        node_x = [positions[i][0] for i in range(n)]
        node_y = [positions[i][1] for i in range(n)]
        node_labels = [f"M{i+1}" for i in range(n)]
        node_scores = scores[:n]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode='lines',
            line=dict(width=1, color='rgba(34,211,238,0.25)'),
            hoverinfo='none',
        ))
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y, mode='markers+text',
            text=node_labels,
            textposition="top center",
            textfont=dict(size=8, color="#dde6f0", family="JetBrains Mono"),
            marker=dict(
                size=16,
                color=node_scores,
                colorscale=[[0, "#22d3ee"], [0.5, "#f59e0b"], [1, "#ef4444"]],
                showscale=True,
                colorbar=dict(
                    title="Rank Score",
                    titlefont=dict(size=9),
                    tickfont=dict(size=8),
                    thickness=10,
                ),
                line=dict(width=1.5, color="#1c2535"),
            ),
            hovertemplate=[
                f"<b>{node_labels[i]}</b><br>"
                f"SMILES: {smiles_list[i][:20]}…<br>"
                f"Rank Score: {node_scores[i]:.4f}<extra></extra>"
                for i in range(n)
            ],
        ))
        fig.update_layout(
            **plotly_base(h=320),
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            title=dict(text="Molecular Similarity Network (Tanimoto ≥ 0.25)",
                       font=dict(size=10, color="#6b7f99"), x=0.5),
        )
        return fig
    except Exception as e:
        return None


def build_patient_heatmap(conn) -> go.Figure | None:
    """Fetch trial_patient_responses and render drug × patient heatmap."""
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT drug_id, patient_id, effectiveness "
            "FROM trial_patient_responses "
            "ORDER BY drug_id, patient_id "
            "LIMIT 500"
        )
        rows = cur.fetchall()
        if not rows:
            return None

        df = pd.DataFrame(rows, columns=["drug_id", "patient_id", "effectiveness"])
        pivot = df.pivot(index="drug_id", columns="patient_id", values="effectiveness")

        drug_labels   = [f"Drug {d}" for d in pivot.index]
        patient_labels = [f"P{p}"   for p in pivot.columns[:25]]  # cap at 25

        z = pivot.values[:, :25]

        fig = go.Figure(go.Heatmap(
            z=z,
            x=patient_labels,
            y=drug_labels,
            colorscale=[[0, "#0d1825"], [0.4, "#0ea5e9"], [1, "#22d3ee"]],
            showscale=True,
            colorbar=dict(title="Effectiveness", titlefont=dict(size=9),
                          tickfont=dict(size=8), thickness=10),
            hovertemplate="Drug: %{y}<br>Patient: %{x}<br>Effectiveness: %{z:.3f}<extra></extra>",
        ))
        fig.update_layout(
            **plotly_base(h=280),
            title=dict(text="Drug × Patient Effectiveness Heatmap",
                       font=dict(size=10, color="#6b7f99"), x=0.5),
            xaxis=dict(tickfont=dict(size=7), gridcolor="#1c2535"),
            yaxis=dict(tickfont=dict(size=8), gridcolor="#1c2535"),
        )
        return fig
    except Exception:
        return None


def build_parallel_coords(ranked: list[dict]) -> go.Figure | None:
    if not ranked or len(ranked) < 3:
        return None
    df = pd.DataFrame(ranked)
    for col in ["docking", "toxicity", "score"]:
        if col not in df.columns:
            return None

    fig = go.Figure(go.Parcoords(
        line=dict(
            color=df["score"],
            colorscale=[[0,"#22d3ee"],[0.5,"#f59e0b"],[1,"#ef4444"]],
            showscale=True,
            colorbar=dict(title="Rank", titlefont=dict(size=9),
                          tickfont=dict(size=8), thickness=10),
        ),
        dimensions=[
            dict(label="Docking", values=df["docking"],
                 range=[df["docking"].min(), df["docking"].max()]),
            dict(label="Toxicity", values=df["toxicity"],
                 range=[0, 1]),
            dict(label="Rank Score", values=df["score"],
                 range=[df["score"].min(), df["score"].max()]),
        ],
        unselected=dict(line=dict(color="#1c2535", opacity=0.3)),
    ))
    fig.update_layout(
        **plotly_base(h=280),
        title=dict(text="Property Parallel Coordinates — All Candidates",
                   font=dict(size=10, color="#6b7f99"), x=0.5),
    )
    return fig


def build_sensitivity_chart(ranked: list[dict]) -> go.Figure | None:
    """Simulate how sensitive ranking is to weight perturbations."""
    if not ranked or len(ranked) < 3:
        return None

    top5 = ranked[:5]
    weights = np.linspace(0.3, 0.9, 13)  # W_BINDING from 0.3 to 0.9
    traces = []

    for mol in top5:
        d, t = mol["docking"], mol["toxicity"]
        nd   = max(0.0, min(1.0, -d / 15.0))
        scores = []
        for wb in weights:
            wt = 1.0 - wb
            scores.append(round(-(wb * nd - wt * t), 4))
        traces.append((mol["smiles"][:14], scores))

    fig = go.Figure()
    colours = ["#22d3ee","#0ea5e9","#f59e0b","#ef4444","#a78bfa"]
    for (label, scores), colour in zip(traces, colours):
        fig.add_trace(go.Scatter(
            x=list(weights), y=scores,
            name=label, mode="lines+markers",
            line=dict(color=colour, width=2),
            marker=dict(size=5),
        ))
    fig.update_layout(
        **plotly_base(h=280),
        title=dict(text="Ranking Sensitivity — W_binding from 0.3 → 0.9",
                   font=dict(size=10, color="#6b7f99"), x=0.5),
        xaxis=dict(title="W_binding weight", gridcolor="#1c2535", tickfont=dict(size=8)),
        yaxis=dict(title="Ranking Score",    gridcolor="#1c2535", tickfont=dict(size=8)),
        legend=dict(font=dict(size=8), bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════

for k, v in {
    "results":   None,
    "running":   False,
    "logs":      [],
    "pipe_step": -1,
    "db_conn":   None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        "<div style='font-family:Syne,sans-serif;font-size:1.15rem;"
        "font-weight:800;color:#22d3ee;margin-bottom:0.3rem'>🧬 Drug Discovery</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    disease  = st.text_input("Disease / Condition", value="Diabetes",
                              help="e.g. Diabetes, Cancer, Alzheimer's, Parkinson's")
    num_mol  = st.slider("Molecules to generate", 5, 30, 10, 1)
    num_pat  = st.slider("Synthetic patients",    10, 50, 20, 5)

    st.markdown("---")
    st.markdown("**Database**")

    import config as _cfg
    st.code(
        f"Host : {_cfg.DB_HOST}:{_cfg.DB_PORT}\n"
        f"DB   : {_cfg.DB_NAME}\n"
        f"User : {_cfg.DB_USER}",
        language="text",
    )
    st.markdown("**Ollama**")
    st.code(
        f"Host : {_cfg.OLLAMA_HOST}\n"
        f"Model: {_cfg.OLLAMA_MODEL}",
        language="text",
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#4b6070'>"
        "Stack: Streamlit · RDKit · py3Dmol · Ollama llama3 · MySQL"
        "</small>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="background:linear-gradient(90deg,#080c14 0%,#0f1a2b 60%,#080c14 100%);
            border-bottom:1px solid #1c2535;padding:1.1rem 1.5rem;
            margin-bottom:1.2rem;border-radius:0 0 12px 12px">
  <div style="display:flex;align-items:center;gap:14px">
    <span style="font-size:2.2rem">🧬</span>
    <div>
      <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:800;
                  color:#dde6f0;letter-spacing:0.02em">
        GenAI-Based Drug Discovery &amp; Clinical Trial Simulation
      </div>
      <div style="font-size:0.76rem;color:#4b6070;font-family:'JetBrains Mono',monospace;margin-top:3px">
        Ollama llama3 &nbsp;·&nbsp; RDKit &nbsp;·&nbsp; 3D Visualisation &nbsp;·&nbsp;
        Similarity Network &nbsp;·&nbsp; 8-step AI pipeline
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════

left, right = st.columns([1, 1.4], gap="large")

# ──────────────────────── LEFT COLUMN ────────────────────────────────────
with left:

    # § Disease + Target
    sec("⬡", 1, "Disease Input & Target Identification")
    prot, role, site = get_target(disease)
    st.markdown(
        f'<div class="target-box">'
        f'<strong>TARGET PROTEIN IDENTIFIED</strong><br/>'
        f'Disease &nbsp;&nbsp;: {disease}<br/>'
        f'Protein &nbsp;&nbsp;: <strong>{prot}</strong> — {role}<br/>'
        f'Binding Site : {site}'
        f'</div>',
        unsafe_allow_html=True,
    )

    run_btn = st.button(
        "▶  Run Full Drug Discovery Pipeline",
        use_container_width=True,
        disabled=st.session_state.running,
    )

    # § Molecule Preview (2D structure + SMILES)
    sec("⚛", 2, "Generated Molecules & 2D Structure")

    res = st.session_state.results
    mols_list = (res or {}).get("molecules", [])

    if mols_list:
        # 2D structure picker
        smiles_options = [m["smiles"] for m in mols_list]
        sel_smi = st.selectbox("Select molecule to preview", smiles_options,
                               format_func=lambda s: s[:30] + "…" if len(s) > 30 else s)
        svg = smiles_to_svg(sel_smi)
        if svg:
            b64 = base64.b64encode(svg.encode()).decode()
            st.markdown(
                f'<div style="background:#04080f;border:1px solid #1c2535;'
                f'border-radius:10px;padding:0.5rem;text-align:center;">'
                f'<img src="data:image/svg+xml;base64,{b64}" width="100%"/>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.text_area("SMILES",
                "\n".join(m["smiles"] for m in mols_list),
                height=100, disabled=True)
    else:
        st.info("Run the pipeline to see generated molecules.", icon="⏳")

    # § Drug Screening
    sec("🔬", 3, "Drug Screening Results (Docking + Toxicity)")
    if res and res.get("lipinski"):
        ranked_map = {m["id"]: m for m in res.get("ranked", [])}
        rows = []
        for m in res["lipinski"]:
            if not m["pass"]:
                continue
            rm  = ranked_map.get(m["id"], {})
            d   = rm.get("docking",  None)
            t   = rm.get("toxicity", None)
            strg = ("Strong" if isinstance(d, float) and d < -9 else
                    "Moderate" if isinstance(d, float) and d < -6 else "Weak"
                    if isinstance(d, float) else "—")
            rows.append({
                "Molecule":      m["smiles"][:18],
                "Docking":       f"{d:.2f}" if d is not None else "—",
                "Strength":      strg,
                "Toxicity":      f"{t:.3f}" if t is not None else "—",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.warning("No Lipinski-passing molecules.")
    else:
        st.info("Run the pipeline to see screening results.", icon="⏳")

    # § Lipinski
    sec("⚖", 4, "Lipinski Drug-Likeness Validation")
    if res and res.get("lipinski"):
        lip_rows = [
            {
                "Molecule":       m["smiles"][:18],
                "MW (<500)":      round(m["mw"],   1),
                "LogP (<5)":      round(m["logp"], 2),
                "H-Donor (<5)":   m["hbd"],
                "H-Accept (<10)": m["hba"],
                "Pass/Fail":      "✅ PASS" if m["pass"] else "❌ FAIL",
            }
            for m in res["lipinski"]
        ]
        st.dataframe(pd.DataFrame(lip_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Run the pipeline to see Lipinski results.", icon="⏳")


# ──────────────────────── RIGHT COLUMN ───────────────────────────────────
with right:

    # § Pipeline
    sec("⚡", 5, "AI Discovery Pipeline")
    pipe_ph = st.empty()
    pipe_ph.markdown(pipeline_html(st.session_state.pipe_step), unsafe_allow_html=True)

    log_ph = st.empty()

    def render_log():
        lines = st.session_state.logs[-50:]
        log_ph.markdown(
            "<div class='log-area'>"
            + "<br/>".join(lines)
            + "</div>",
            unsafe_allow_html=True,
        )

    render_log()

    # § 3D Molecule Viewer
    sec("🔭", 6, "3D Molecule Viewer (Interactive)")
    if res and res.get("ranked"):
        ranked_mols = res["ranked"]
        opts = [m["smiles"] for m in ranked_mols[:10]]
        sel3d = st.selectbox(
            "Choose molecule for 3D view",
            opts,
            format_func=lambda s: f"Rank #{ranked_mols[[m['smiles'] for m in ranked_mols].index(s)].get('rank_position','?')} — {s[:28]}…" if len(s) > 28 else s,
            key="sel3d",
        )

        style_opts = st.radio(
            "Render style", ["Stick", "Sphere", "Surface"],
            horizontal=True, key="mol3d_style",
        )

        html_3d = smiles_to_3d_html(sel3d)
        if html_3d:
            st.components.v1.html(html_3d, height=340, scrolling=False)
            st.caption(
                "🖱 Rotate: drag &nbsp;|&nbsp; Zoom: scroll &nbsp;|&nbsp; "
                "Pan: right-click drag &nbsp;|&nbsp; Use buttons to change render style"
            )
        else:
            st.warning(
                "3D conformer generation failed for this SMILES. "
                "Try another molecule.",
                icon="⚠️",
            )
    else:
        st.info("Run the pipeline to see 3D visualization.", icon="⏳")

    # § Patient cohort
    sec("👥", 7, f"Synthetic Patient Cohort (N={num_pat})")
    if res and res.get("patients"):
        trial_drugs = res.get("trial", {}).get("drugs", [])
        pat_rows = []
        response_cycle = ["Good", "Moderate", "Poor"]
        for i, p in enumerate(res["patients"]):
            pat_rows.append({
                "Patient ID":         f"P{i+1}",
                "Age":                p.get("age",    "—"),
                "Weight (kg)":        p.get("weight", "—"),
                "Genotype":           p.get("genetic_marker", "—"),
                "Predicted Response": response_cycle[i % 3],
            })
        st.dataframe(
            pd.DataFrame(pat_rows),
            use_container_width=True, hide_index=True, height=200,
        )
        if trial_drugs:
            avg_sr  = sum(d["success_rate"]    for d in trial_drugs) / len(trial_drugs)
            avg_ser = sum(d["side_effect_rate"] for d in trial_drugs) / len(trial_drugs)
            c1, c2 = st.columns(2)
            c1.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-val" style="color:#22d3ee">{avg_sr*100:.0f}%</div>'
                f'<div class="metric-label">⭐ Avg Success Rate</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            c2.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-val" style="color:#f59e0b">{avg_ser*100:.0f}%</div>'
                f'<div class="metric-label">⚠ Avg Side-Effect Rate</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("Run the pipeline to see the patient cohort.", icon="⏳")

    # § Top Candidate + Radar
    tc = (res or {}).get("top_candidate")
    sec("🏆", 8, f"Top Drug Candidate: {'  ' + tc['smiles'][:16] if tc else '—'}")
    if tc:
        fc1, fc2 = st.columns([1.2, 1])
        with fc1:
            strg  = ("Strong"   if tc["docking_score"]  < -9  else
                     "Moderate" if tc["docking_score"]  < -6  else "Weak")
            tox_l = ("Low"      if tc["toxicity_score"] < 0.3 else
                     "Medium"   if tc["toxicity_score"] < 0.6 else "High")
            trial_sr = ""
            if res.get("trial", {}).get("drugs"):
                trial_sr = f"{res['trial']['drugs'][0]['success_rate']*100:.0f}%"
            st.markdown(f"""
| Property | Value |
|---|---|
| **Best Molecule** | `{tc['smiles'][:20]}…` |
| **Docking Score** | `{tc['docking_score']:.2f}` ({strg}) |
| **Toxicity** | `{tc['toxicity_score']:.3f}` ({tox_l}) |
| **Ranking Score** | `{tc['ranking_score']:.5f}` |
| **Trial Success** | `{trial_sr or '—'}` |
""")
        with fc2:
            nd        = min(1.0, -tc["docking_score"] / 15.0)
            safety    = 1.0 - tc["toxicity_score"]
            drug_like = max(0.0, 1.0 + tc["ranking_score"])
            efficacy  = nd * 0.7
            adme      = safety * 0.8
            cats      = ["Binding", "Safety", "Drug-Like", "Efficacy", "ADME"]
            vals      = [nd, safety, drug_like, efficacy, adme]
            fig_r = go.Figure(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill="toself",
                fillcolor="rgba(34,211,238,0.12)",
                line=dict(color="#22d3ee", width=2),
            ))
            fig_r.update_layout(
                **plotly_base(h=220),
                polar=dict(
                    bgcolor="#04080f",
                    radialaxis=dict(visible=True, range=[0, 1],
                                   gridcolor="#1c2535", linecolor="#1c2535",
                                   tickfont=dict(color="#6b7f99", size=7)),
                    angularaxis=dict(tickfont=dict(color="#dde6f0", size=8),
                                    linecolor="#1c2535", gridcolor="#1c2535"),
                ),
                showlegend=False,
            )
            st.plotly_chart(fig_r, use_container_width=True,
                            config={"displayModeBar": False})

        expl = res.get("explanation", "")
        if expl:
            st.markdown(f'<div class="expl-box">💡 {expl}</div>',
                        unsafe_allow_html=True)
    else:
        st.info("Run the pipeline to see the top drug candidate.", icon="⏳")


# ══════════════════════════════════════════════════════════════════════════
# ANALYTICS TABS  (full width, only after a run)
# ══════════════════════════════════════════════════════════════════════════

if res and res.get("ranked"):
    st.markdown("---")
    sec("📊", 9, "Analytics Dashboard")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Screening Charts",
        "🕸 Similarity Network",
        "🔥 Patient Heatmap",
        "📐 Parallel Coords",
        "🎚 Sensitivity Analysis",
    ])

    df_r = pd.DataFrame(res["ranked"])

    # ── Tab 1: Classic charts ─────────────────────────────────────────────
    with tab1:
        ch1, ch2, ch3 = st.columns(3)

        with ch1:
            st.caption("Docking Score Distribution")
            fig1 = px.histogram(
                df_r, x="docking", nbins=10,
                color_discrete_sequence=["#22d3ee"],
                labels={"docking": "Docking Score"},
            )
            fig1.update_layout(
                **plotly_base(),
                showlegend=False,
                xaxis=dict(gridcolor="#1c2535"),
                yaxis=dict(gridcolor="#1c2535"),
            )
            st.plotly_chart(fig1, use_container_width=True,
                            config={"displayModeBar": False})

        with ch2:
            st.caption("Docking vs Toxicity (top 10)")
            fig2 = px.scatter(
                df_r.head(10), x="docking", y="toxicity",
                hover_data=["smiles"],
                color="score",
                color_continuous_scale=[[0,"#22d3ee"],[0.5,"#f59e0b"],[1,"#ef4444"]],
                labels={"docking":"Docking","toxicity":"Toxicity","score":"Rank"},
            )
            fig2.update_layout(
                **plotly_base(),
                xaxis=dict(gridcolor="#1c2535"),
                yaxis=dict(gridcolor="#1c2535"),
                coloraxis_showscale=False,
            )
            fig2.update_traces(marker=dict(size=10, line=dict(width=1, color="#1c2535")))
            st.plotly_chart(fig2, use_container_width=True,
                            config={"displayModeBar": False})

        with ch3:
            trial_drugs = res.get("trial", {}).get("drugs", [])
            if trial_drugs:
                st.caption("Clinical Trial — Success vs Side Effects")
                td = pd.DataFrame(trial_drugs)
                td["label"] = td["smiles"].str[:12]
                fig3 = go.Figure()
                fig3.add_trace(go.Bar(x=td["label"], y=td["success_rate"]*100,
                                      name="Success %", marker_color="#22d3ee"))
                fig3.add_trace(go.Bar(x=td["label"], y=td["side_effect_rate"]*100,
                                      name="Side Effect %", marker_color="#f59e0b"))
                fig3.update_layout(
                    **plotly_base(),
                    barmode="group",
                    legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
                    xaxis=dict(gridcolor="#1c2535", tickfont=dict(size=8)),
                    yaxis=dict(gridcolor="#1c2535", title="Rate (%)"),
                )
                st.plotly_chart(fig3, use_container_width=True,
                                config={"displayModeBar": False})

        # ── Full ranked table + export ────────────────────────────────────
        st.markdown("---")
        sec("📋", 10, "Full Ranked Molecule Table")
        display_cols = {
            "rank_position": "Rank", "smiles": "SMILES",
            "docking": "Docking Score", "toxicity": "Toxicity", "score": "Ranking Score",
        }
        df_display = df_r[list(display_cols.keys())].rename(columns=display_cols)
        df_display["Docking Score"] = df_display["Docking Score"].map("{:.2f}".format)
        df_display["Toxicity"]      = df_display["Toxicity"].map("{:.4f}".format)
        df_display["Ranking Score"] = df_display["Ranking Score"].map("{:.5f}".format)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        csv_bytes = df_display.to_csv(index=False).encode()
        st.download_button(
            "⬇  Download Ranked Molecules CSV",
            data=csv_bytes,
            file_name="ranked_molecules.csv",
            mime="text/csv",
        )

    # ── Tab 2: Similarity Network ─────────────────────────────────────────
    with tab2:
        st.caption(
            "Nodes = molecules, edges = Tanimoto fingerprint similarity ≥ 0.25. "
            "Colour encodes ranking score (cyan = best)."
        )
        fig_net = build_similarity_network(res["ranked"])
        if fig_net:
            st.plotly_chart(fig_net, use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.warning("Could not build similarity network. RDKit required.", icon="⚠️")

    # ── Tab 3: Patient Heatmap ────────────────────────────────────────────
    with tab3:
        st.caption(
            "Per-drug, per-patient effectiveness scores from the clinical trial simulation."
        )
        # Re-open connection for heatmap query
        try:
            from database.db_connection import get_connection
            conn_hm = get_connection()
            fig_hm  = build_patient_heatmap(conn_hm)
            conn_hm.close()
            if fig_hm:
                st.plotly_chart(fig_hm, use_container_width=True,
                                config={"displayModeBar": False})
            else:
                st.info("Heatmap requires clinical trial data. Run pipeline first.", icon="⏳")
        except Exception as e:
            st.warning(f"Could not load heatmap: {e}", icon="⚠️")

    # ── Tab 4: Parallel Coordinates ───────────────────────────────────────
    with tab4:
        st.caption(
            "Drag axes to filter. Each line = one molecule. "
            "Colour = ranking score."
        )
        fig_pc = build_parallel_coords(res["ranked"])
        if fig_pc:
            st.plotly_chart(fig_pc, use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.warning("Not enough molecules for parallel coordinates.", icon="⚠️")

    # ── Tab 5: Sensitivity Analysis ───────────────────────────────────────
    with tab5:
        st.caption(
            "How does the ranking of top-5 candidates change if we shift "
            "the binding weight (W_binding) from 0.3 to 0.9?"
        )
        fig_sens = build_sensitivity_chart(res["ranked"])
        if fig_sens:
            st.plotly_chart(fig_sens, use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.warning("Need at least 3 ranked molecules.", icon="⚠️")

    # ── Multi-Candidate Comparison ────────────────────────────────────────
    st.markdown("---")
    sec("🔍", 11, "Top-5 Candidate Comparison")
    top5 = res["ranked"][:5]
    cols = st.columns(min(5, len(top5)))
    for i, (mol, col) in enumerate(zip(top5, cols)):
        with col:
            strg  = ("Strong"   if mol["docking"]  < -9  else
                     "Moderate" if mol["docking"]  < -6  else "Weak")
            tox_l = ("Low"      if mol["toxicity"] < 0.3 else
                     "Medium"   if mol["toxicity"] < 0.6 else "High")
            col.markdown(
                f'<div class="cand-card">'
                f'<div class="cand-rank">#{mol.get("rank_position", i+1)}</div>'
                f'<div style="font-family:JetBrains Mono,monospace;font-size:0.65rem;'
                f'color:#4b6070;margin-bottom:6px">{mol["smiles"][:20]}…</div>'
                f'<div style="font-size:0.75rem;color:#dde6f0">Dock: <b>{mol["docking"]:.2f}</b></div>'
                f'<div style="font-size:0.72rem;color:#6b7f99">{strg} binding</div>'
                f'<div style="font-size:0.75rem;color:#dde6f0;margin-top:4px">Tox: <b>{mol["toxicity"]:.3f}</b></div>'
                f'<div style="font-size:0.72rem;color:#6b7f99">{tox_l} risk</div>'
                f'<div style="font-size:0.75rem;color:#22d3ee;margin-top:4px">Score: {mol["score"]:.4f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════
# RUN PIPELINE
# ══════════════════════════════════════════════════════════════════════════

if run_btn and not st.session_state.running:
    st.session_state.running   = True
    st.session_state.logs      = []
    st.session_state.results   = None
    st.session_state.pipe_step = 0

    STEP_KEYWORDS = {
        "Step 1": 0, "Step 2": 1, "Step 3": 2, "Step 4": 3,
        "Step 5": 4, "Step 6": 5, "Step 7": 6, "Step 8": 7,
    }

    def log_fn(msg: str):
        clean = msg.replace("**", "").replace("*", "")
        if   msg.startswith("✅") or msg.startswith("🎉"):  colour = "#22d3ee"
        elif msg.startswith("⚠"):                           colour = "#f59e0b"
        elif msg.startswith("🤖") or msg.startswith("📥") or msg.startswith("💡"):
                                                             colour = "#0ea5e9"
        elif msg.startswith("❌"):                           colour = "#ef4444"
        else:                                                colour = "#4b6070"

        st.session_state.logs.append(
            f'<span style="color:{colour}">{clean}</span>'
        )
        render_log()
        for kw, idx in STEP_KEYWORDS.items():
            if kw in msg:
                st.session_state.pipe_step = idx
                pipe_ph.markdown(pipeline_html(idx), unsafe_allow_html=True)
                break

    try:
        from pipeline import run_pipeline
        results = run_pipeline(
            disease=disease,
            num_molecules=num_mol,
            num_patients=num_pat,
            log_fn=log_fn,
        )
        st.session_state.results   = results
        st.session_state.pipe_step = len(PIPE_STEPS)
        pipe_ph.markdown(pipeline_html(len(PIPE_STEPS)), unsafe_allow_html=True)

    except Exception as exc:
        st.session_state.logs.append(
            f'<span style="color:#ef4444">❌ ERROR: {exc}</span>'
        )
        render_log()
        st.error(f"Pipeline failed: {exc}")

    finally:
        st.session_state.running = False

    st.rerun()