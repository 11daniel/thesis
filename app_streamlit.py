"""
Streamlit Transit Deserts Prototype App
======================================
Interactive dashboard displaying Rips complexes, overlap count maps,
and divergence analysis using synthetic data.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Set

# Import topological engine
from topological_engine import (
    SyntheticTransitNetwork,
    compute_h1_persistence,
    get_trapped_nodes,
    evaluate_divergence
)

# Page configuration
st.set_page_config(
    page_title="Topological Transit Deserts",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize network (cached to prevent recreation on rerun)
@st.cache_resource
def get_network():
    return SyntheticTransitNetwork(spacing=400.0)

network = get_network()

# Helper to draw flat-topped hexagons
def get_hex_polygon(cx: float, cy: float, radius: float) -> List[Tuple[float, float]]:
    vertices = []
    for i in range(6):
        angle_rad = i * np.pi / 3
        x = cx + radius * np.cos(angle_rad)
        y = cy + radius * np.sin(angle_rad)
        vertices.append((x, y))
    return vertices

# --- 1. Custom CSS Theme ---
st.markdown("""
<style>
    /* Dark Theme styling for Streamlit container */
    .stApp {
        background-color: #1e1e2e;
        color: #cdd6f4;
    }
    .stSidebar {
        background-color: #181825;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #cba6f7 !important;
    }
    .stMarkdown p {
        color: #cdd6f4;
    }
    .stMetricLabel {
        color: #a6adc8 !important;
    }
    .stMetricValue {
        color: #f38ba8 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Sidebar Controls ---
st.sidebar.title("🎛️ Scenario Simulation")
st.sidebar.markdown("Adjust commuter preferences and routing parameters below to sweep topological complexes.")

dep_time = st.sidebar.selectbox(
    "Departure Time Window (Traffic)",
    options=['6AM', '7AM', '12PM', '6PM', '7PM'],
    index=1  # Default to 7AM Peak
)

st.sidebar.subheader("Commuter Friction Thresholds")
r_travel = st.sidebar.slider("Travel Time (minutes)", min_value=5, max_value=60, value=20, step=5)
r_transfers = st.sidebar.slider("Transfers Count", min_value=1, max_value=5, value=2, step=1)
r_modes = st.sidebar.slider("Unique Modes Count", min_value=1, max_value=5, value=2, step=1)
r_wait = st.sidebar.slider("Walking/Waiting Time (minutes)", min_value=5, max_value=45, value=15, step=5)

st.sidebar.subheader("Traditional Proximity Model")
prox_threshold = st.sidebar.slider("Walk buffer to stops (meters)", min_value=200, max_value=2000, value=600, step=100)

# --- 3. Compute Engine Call ---
# We compute matrices and run PH based on the selected departure time
matrices = network.generate_matrices(dep_time)

# Run H1 persistence on the four metrics
holes_travel = compute_h1_persistence(network.coords, matrices['travel'])
holes_transfers = compute_h1_persistence(network.coords, matrices['transfers'])
holes_modes = compute_h1_persistence(network.coords, matrices['modes'])
holes_wait = compute_h1_persistence(network.coords, matrices['wait'])

# Evaluate trapped nodes per layer
trapped_travel = get_trapped_nodes(holes_travel, network.coords, r_travel)
trapped_transfers = get_trapped_nodes(holes_transfers, network.coords, r_transfers)
trapped_modes = get_trapped_nodes(holes_modes, network.coords, r_modes)
trapped_wait = get_trapped_nodes(holes_wait, network.coords, r_wait)

# Compute overlap counts
overlap_counts = np.zeros(network.num_nodes, dtype=int)
for idx in range(network.num_nodes):
    cnt = 0
    if idx in trapped_travel: cnt += 1
    if idx in trapped_transfers: cnt += 1
    if idx in trapped_modes: cnt += 1
    if idx in trapped_wait: cnt += 1
    overlap_counts[idx] = cnt

# --- 4. Main Panel UI Layout ---
st.title("🚇 Topological Transit Deserts Prototype")
st.markdown("Replicating the demand-independent spatial analysis proposed in Chapters 3 and 4 of the undergraduate thesis.")

# Create Streamlit tabs
tab_overlap, tab_rips, tab_divergence = st.tabs([
    "🗺️ Overlap Heatmap", 
    "🕸️ Vietoris-Rips Complexes", 
    "⚖️ Proximity vs. Topology"
])

# --- TAB 1: Overlap Heatmap ---
with tab_overlap:
    st.header("Transit Desert Overlap Map")
    st.markdown(
        "Overlays verified topological holes from all four friction metrics. Cells are color-coded "
        "by the number of layers they exhibit a transit desert signature in."
    )
    
    col_plot, col_stats = st.columns([2, 1])
    
    with col_plot:
        # Plotting Overlap map
        fig, ax = plt.subplots(figsize=(8, 7), facecolor='#1e1e2e')
        ax.set_facecolor('#1e1e2e')
        
        hex_radius = 400.0 / np.sqrt(3)
        
        # Catppuccin Macchiato Palette
        color_map = {
            0: '#313244', # Slate Base
            1: '#f9e2af', # Yellow
            2: '#fab387', # Peach
            3: '#eba0ac', # Pink/Maroon
            4: '#f38ba8'  # Red (Severe)
        }
        text_color_map = {0: '#cdd6f4', 1: '#11111b', 2: '#11111b', 3: '#11111b', 4: '#11111b'}
        
        for idx, (cx, cy) in enumerate(network.coords):
            layers = overlap_counts[idx]
            face_col = color_map[layers]
            edge_col = '#45475a'
            
            hex_poly = get_hex_polygon(cx, cy, hex_radius)
            polygon_patch = patches.Polygon(hex_poly, closed=True, facecolor=face_col, edgecolor=edge_col, linewidth=1.5)
            ax.add_patch(polygon_patch)
            
            # Label
            ax.text(cx, cy, f"H{idx}\n({layers}L)", color=text_color_map[layers],
                    ha='center', va='center', fontsize=9, fontweight='bold')
            
        # Draw physical stops
        for stop_idx, name in network.transit_stops.items():
            sx, sy = network.coords[stop_idx]
            ax.plot(sx, sy, marker='*', color='#a6e3a1', markersize=14, markeredgecolor='#11111b', markeredgewidth=1.5, zorder=5)
            
        xs = [c[0] for c in network.coords]
        ys = [c[1] for c in network.coords]
        ax.set_xlim(min(xs) - 250, max(xs) + 250)
        ax.set_ylim(min(ys) - 250, max(ys) + 250)
        ax.set_aspect('equal')
        ax.axis('off')
        
        legend_elements = [
            patches.Patch(facecolor=color_map[0], edgecolor='#45475a', label='Connected (0 Layers)'),
            patches.Patch(facecolor=color_map[1], edgecolor='#45475a', label='Desert in 1 Layer'),
            patches.Patch(facecolor=color_map[2], edgecolor='#45475a', label='Desert in 2 Layers'),
            patches.Patch(facecolor=color_map[3], edgecolor='#45475a', label='Desert in 3 Layers'),
            patches.Patch(facecolor=color_map[4], edgecolor='#45475a', label='Desert in 4 Layers (Severe)'),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#a6e3a1', markersize=14, markeredgecolor='#11111b', label='Transit Stops')
        ]
        ax.legend(handles=legend_elements, loc='upper right', facecolor='#1e1e2e', edgecolor='#45475a', labelcolor='#cdd6f4', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
    with col_stats:
        st.subheader("📊 Grid Analysis & Stats")
        total_deserts = np.sum(overlap_counts > 0)
        severe_deserts = np.sum(overlap_counts == 4)
        
        # Streamlit metrics
        st.metric("Total Grid Cells", network.num_nodes)
        st.metric("Total Transit Deserts (1+ Layers)", f"{total_deserts} ({total_deserts/network.num_nodes*100:.1f}%)")
        st.metric("Severe Deserts (All 4 Layers)", f"{severe_deserts} ({severe_deserts/network.num_nodes*100:.1f}%)")
        
        st.info(
            "The **overlap-count logic** isolates cells based on the dimensional complexity "
            "of their transit frustration. High-overlap count cells (e.g., inside the center) "
            "suffer from severe multi-modal fragmentation, requiring combined improvements."
        )

# --- TAB 2: Rips Complexes ---
with tab_rips:
    st.header("Vietoris-Rips Complexes")
    st.markdown(
        "Displays the active edges (connections) below the threshold in blue. "
        "Red hexagons show nodes trapped inside topological loops (Transit Deserts). "
        "Yellow dashed loops highlight the boundaries of active cycles."
    )
    
    # 2x2 plotting of individual Rips complexes
    metrics_info = [
        ('travel', r_travel, 'Travel Time (min)', 'min', holes_travel, trapped_travel),
        ('transfers', r_transfers, 'Transfers Count', 'transfers', holes_transfers, trapped_transfers),
        ('modes', r_modes, 'Unique Modes Count', 'modes', holes_modes, trapped_modes),
        ('wait', r_wait, 'Walking/Waiting Time (min)', 'min', holes_wait, trapped_wait)
    ]
    
    fig, axs = plt.subplots(2, 2, figsize=(11, 10), facecolor='#1e1e2e')
    axs = axs.ravel()
    
    for idx, (metric_name, threshold, title, unit, holes, trapped) in enumerate(metrics_info):
        ax = axs[idx]
        ax.set_facecolor('#1e1e2e')
        dist_mat = matrices[metric_name]
        
        # Hexagons
        for node_idx, (cx, cy) in enumerate(network.coords):
            is_trapped = node_idx in trapped
            face_col = '#f38ba8' if is_trapped else '#313244'
            edge_col = '#585b70'
            
            hex_poly = get_hex_polygon(cx, cy, hex_radius)
            polygon_patch = patches.Polygon(hex_poly, closed=True, facecolor=face_col, edgecolor=edge_col, linewidth=1.0)
            ax.add_patch(polygon_patch)
            ax.text(cx, cy, str(node_idx), color='#cdd6f4' if not is_trapped else '#11111b',
                    ha='center', va='center', fontsize=8, fontweight='semibold')
            
        # Active edges
        for i in range(network.num_nodes):
            for j in range(i + 1, network.num_nodes):
                if dist_mat[i, j] <= threshold:
                    p1 = network.coords[i]
                    p2 = network.coords[j]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#89b4fa', linewidth=1.5, alpha=0.7, zorder=3)
                    
        # Active holes
        for h in holes:
            if h['birth'] <= threshold < h['death']:
                poly_coords = [network.coords[n] for n in h['cycle']]
                poly_coords.append(poly_coords[0])
                px, py = zip(*poly_coords)
                ax.plot(px, py, color='#f9e2af', linewidth=2.5, linestyle='--', zorder=4)
                
        # Stops
        for stop_idx, name in network.transit_stops.items():
            sx, sy = network.coords[stop_idx]
            ax.plot(sx, sy, marker='*', color='#a6e3a1', markersize=10, markeredgecolor='#11111b', zorder=5)
            
        ax.set_xlim(min(xs) - 250, max(xs) + 250)
        ax.set_ylim(min(ys) - 250, max(ys) + 250)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"{title} (Threshold: {threshold} {unit})", color='#cdd6f4', fontsize=11, fontweight='bold')
        
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# --- TAB 3: Divergence Analysis ---
with tab_divergence:
    st.header("Divergence Analysis: Proximity vs. Topology")
    st.markdown(
        "Compares physical walk proximity against topological network isolation. "
        "This explicitly captures **False Oases**—areas visually close to stops on paper, "
        "but isolated in reality due to transfers and waiting friction."
    )
    
    col_div_plot, col_div_analysis = st.columns([2, 1])
    
    # Union of trapped cells across all 4 metrics
    union_trapped = trapped_travel.union(trapped_transfers).union(trapped_modes).union(trapped_wait)
    proximity = network.get_proximity_to_nearest_stop()
    classifications = evaluate_divergence(union_trapped, proximity, prox_threshold)
    
    with col_div_plot:
        # Divergence Plot
        fig, ax = plt.subplots(figsize=(8, 7), facecolor='#1e1e2e')
        ax.set_facecolor('#1e1e2e')
        
        divergence_colors = {
            'Well-Served': '#a6e3a1',  # Soft Green
            'True Desert': '#f38ba8',  # Soft Red
            'False Oasis': '#fab387',  # Peach
            'False Desert': '#89b4fa'  # Sky Blue
        }
        text_colors = {k: '#11111b' for k in divergence_colors.keys()}
        
        for idx, (cx, cy) in enumerate(network.coords):
            cls = classifications[idx]
            face_col = divergence_colors[cls]
            edge_col = '#45475a'
            
            hex_poly = get_hex_polygon(cx, cy, hex_radius)
            polygon_patch = patches.Polygon(hex_poly, closed=True, facecolor=face_col, edgecolor=edge_col, linewidth=1.5)
            ax.add_patch(polygon_patch)
            
            ax.text(cx, cy, f"H{idx}\n{cls.split()[0]}", color=text_colors[cls],
                    ha='center', va='center', fontsize=8, fontweight='bold')
            
        # Draw physical transit stops
        for stop_idx, name in network.transit_stops.items():
            sx, sy = network.coords[stop_idx]
            ax.plot(sx, sy, marker='*', color='#f9e2af', markersize=14, markeredgecolor='#11111b', markeredgewidth=1.5, zorder=5)
            
        ax.set_xlim(min(xs) - 250, max(xs) + 250)
        ax.set_ylim(min(ys) - 250, max(ys) + 250)
        ax.set_aspect('equal')
        ax.axis('off')
        
        legend_elements = [
            patches.Patch(facecolor=divergence_colors['Well-Served'], edgecolor='#45475a', label='Well-Served (Close, Not Trapped)'),
            patches.Patch(facecolor=divergence_colors['True Desert'], edgecolor='#45475a', label='True Desert (Far, Trapped)'),
            patches.Patch(facecolor=divergence_colors['False Oasis'], edgecolor='#45475a', label='False Oasis (Close, Trapped)'),
            patches.Patch(facecolor=divergence_colors['False Desert'], edgecolor='#45475a', label='False Desert (Far, Not Trapped)'),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#f9e2af', markersize=14, markeredgecolor='#11111b', label='Transit Stops')
        ]
        ax.legend(handles=legend_elements, loc='upper right', facecolor='#1e1e2e', edgecolor='#45475a', labelcolor='#cdd6f4', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
    with col_div_analysis:
        st.subheader("📐 Divergence Classification Breakdown")
        
        # Display breakdown list
        st.write(f"- 🟢 **Well-Served**: {classifications.count('Well-Served')} cells")
        st.write(f"- 🔴 **True Desert**: {classifications.count('True Desert')} cells")
        st.write(f"- 🟠 **False Oasis**: {classifications.count('False Oasis')} cells")
        st.write(f"- 🔵 **False Desert**: {classifications.count('False Desert')} cells")
        
        st.subheader("💡 Strategic Planning Insights")
        
        # Focus on False Oases
        false_oases_nodes = [idx for idx, c in enumerate(classifications) if c == 'False Oasis']
        if len(false_oases_nodes) > 0:
            st.warning(
                f"**Warning: False Oases Detected**\n\n"
                f"Cells `{false_oases_nodes}` sit within physical proximity of transitstops (i.e. buffer <= {prox_threshold}m) "
                f"but are topologically cut off due to high transfers or wait times. "
                "These require frequency optimization and pedestrian route integration, rather than building new rail lines."
            )
        else:
            st.success("No False Oases detected at these thresholds. Topological access matches stop proximity.")
