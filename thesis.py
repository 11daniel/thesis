"""
Vietoris-Rips Complexes for Transit Friction Metrics
====================================================

This script models and visualizes four Vietoris-Rips complexes, representing transit accessibility
based on four friction metrics: travel time, transfer count, mode count, and walking/waiting time.

It contains:
1. A topological model of the boundary ring and interior points.
2. Logic to evaluate edge connectivity and point trapping based on friction thresholds.
3. A command-line simulation interface.
4. A visualization function using Matplotlib (if installed).
5. A lightweight, pure-Python SVG generator that exports the visual state as an HTML page.
"""

import os
import sys
import json
from typing import Dict, List, Tuple, Set

# --- 1. Topological Configuration (Dummy / Calibration Data) ---
# Coordinates matching the SVG layout
R_COORDS = [
    (150.0, 47.0),   # Node 0
    (208.9, 81.0),   # Node 1
    (208.9, 149.0),  # Node 2
    (150.0, 183.0),  # Node 3
    (91.1, 149.0),   # Node 4
    (91.1, 81.0)     # Node 5
]

RING_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]

COORD_A = (128.0, 115.0)
COORD_B = (172.0, 115.0)

# Connective mappings: which ring node indices points A & B connect to
A_RING_INDICES = [4, 5]
B_RING_INDICES = [1, 2]

# The birth threshold of the outer boundary ring edges for each metric
RING_BIRTH = {
    'travel': 5,
    'transfers': 1,
    'modes': 1,
    'wait': 5
}

# The threshold (friction) at which point A or B connects to the boundary
NODE_ESCAPE = {
    'A': {'travel': 24, 'transfers': 5, 'modes': 3, 'wait': 26},
    'B': {'travel': 22, 'transfers': 1, 'modes': 1, 'wait': 24}
}

METRIC_TITLES = {
    'travel': 'Travel time',
    'transfers': 'Transfers',
    'modes': 'Modes',
    'wait': 'Walking/waiting'
}

METRIC_UNITS = {
    'travel': 'min',
    'transfers': 'transfers',
    'modes': 'modes',
    'wait': 'min'
}


# --- 2. Core Topological Model ---
class FrictionComplex:
    """
    Models a Vietoris-Rips complex for a specific friction metric and threshold.
    """
    def __init__(self, metric: str, threshold: float):
        if metric not in METRIC_TITLES:
            raise ValueError(f"Unknown metric: {metric}")
        self.metric = metric
        self.threshold = threshold
        self.title = METRIC_TITLES[metric]
        
    @property
    def ring_connected(self) -> bool:
        """Checks if the outer boundary ring edges are formed."""
        return self.threshold >= RING_BIRTH[self.metric]
        
    @property
    def is_a_trapped(self) -> bool:
        """Point A is trapped if the threshold is below its escape value."""
        return self.threshold < NODE_ESCAPE['A'][self.metric]
        
    @property
    def is_b_trapped(self) -> bool:
        """Point B is trapped if the threshold is below its escape value."""
        return self.threshold < NODE_ESCAPE['B'][self.metric]
        
    def get_edges_state(self) -> List[Dict]:
        """
        Returns all edges in the complex, their coordinate endpoints,
        and whether they are active (below/at current threshold).
        """
        edges = []
        
        # 1. Ring edges
        ring_active = self.ring_connected
        for i, j in RING_EDGES:
            edges.append({
                'type': 'ring',
                'p1': R_COORDS[i],
                'p2': R_COORDS[j],
                'active': ring_active,
                'birth': RING_BIRTH[self.metric]
            })
            
        # 2. Point A edges
        a_active = not self.is_a_trapped
        for idx in A_RING_INDICES:
            edges.append({
                'type': 'node_a',
                'p1': COORD_A,
                'p2': R_COORDS[idx],
                'active': a_active,
                'birth': NODE_ESCAPE['A'][self.metric]
            })
            
        # 3. Point B edges
        b_active = not self.is_b_trapped
        for idx in B_RING_INDICES:
            edges.append({
                'type': 'node_b',
                'p1': COORD_B,
                'p2': R_COORDS[idx],
                'active': b_active,
                'birth': NODE_ESCAPE['B'][self.metric]
            })
            
        return edges


class MultiComplexModel:
    """
    Combines all four metric complexes and tracks overall trapping metrics.
    """
    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds
        self.complexes = {
            metric: FrictionComplex(metric, val)
            for metric, val in thresholds.items()
        }
        
    def get_trapped_counts(self) -> Dict[str, int]:
        """Counts how many complexes nodes A and B are trapped in."""
        a_trapped_count = sum(1 for c in self.complexes.values() if c.is_a_trapped)
        b_trapped_count = sum(1 for c in self.complexes.values() if c.is_b_trapped)
        return {
            'A': a_trapped_count,
            'B': b_trapped_count
        }

    def print_summary(self):
        """Prints a text-based representation of the state in the console."""
        print("-" * 50)
        print("VIETORIS-RIPS FRICTION METRICS STATUS")
        print("-" * 50)
        for metric, comp in self.complexes.items():
            unit = METRIC_UNITS[metric]
            print(f"{comp.title:<20} | Threshold: {comp.threshold:>2} {unit:<9} | "
                  f"A: {'TRAPPED' if comp.is_a_trapped else 'ESCAPED':<7} | "
                  f"B: {'TRAPPED' if comp.is_b_trapped else 'ESCAPED':<7}")
        
        counts = self.get_trapped_counts()
        print("-" * 50)
        print(f"Point A - trapped in {counts['A']}/4 complexes")
        print(f"Point B - trapped in {counts['B']}/4 complexes")
        print("-" * 50)


# --- 3. Matplotlib Visualization Generator ---
def plot_complexes(model: MultiComplexModel, save_path: str = None):
    """
    Plots the four complexes side-by-side using matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("\n[Notice] matplotlib is not installed. Skipping graphical plot.")
        print("To view matplotlib plots, install it via: pip install matplotlib\n")
        return

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Vietoris-Rips Complexes per Friction Metric", fontsize=16, fontweight='bold')
    
    positions = [
        ('travel', axs[0, 0]),
        ('transfers', axs[0, 1]),
        ('modes', axs[1, 0]),
        ('wait', axs[1, 1])
    ]
    
    for metric, ax in positions:
        comp = model.complexes[metric]
        ax.set_title(f"{comp.title} (Threshold: {comp.threshold} {METRIC_UNITS[metric]})", fontsize=12, fontweight='semibold')
        
        # Draw background boundary box
        rect = patches.Rectangle((0, 20), 300, 200, linewidth=1, edgecolor='#cccccc', facecolor='none', rx=8)
        ax.add_patch(rect)
        
        # Draw edges
        for edge in comp.get_edges_state():
            p1, p2 = edge['p1'], edge['p2']
            if edge['active']:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#2c3e50', linewidth=1.5, linestyle='-')
            else:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#95a5a6', linewidth=1.0, linestyle='--')
                
        # Draw outer nodes
        for pt in R_COORDS:
            ax.plot(pt[0], pt[1], 'o', markerfacecolor='#ffffff', markeredgecolor='#7f8c8d', markersize=8, markeredgewidth=1.5)
            
        # Draw interior point A
        color_a = '#e74c3c' if comp.is_a_trapped else '#ffffff'
        border_a = '#c0392b' if comp.is_a_trapped else '#7f8c8d'
        ax.plot(COORD_A[0], COORD_A[1], 'o', markerfacecolor=color_a, markeredgecolor=border_a, markersize=10, markeredgewidth=2)
        ax.text(COORD_A[0] - 12, COORD_A[1] - 12, 'A', fontweight='bold', color=border_a)
        
        # Draw interior point B
        color_b = '#e74c3c' if comp.is_b_trapped else '#ffffff'
        border_b = '#c0392b' if comp.is_b_trapped else '#7f8c8d'
        ax.plot(COORD_B[0], COORD_B[1], 'o', markerfacecolor=color_b, markeredgecolor=border_b, markersize=10, markeredgewidth=2)
        ax.text(COORD_B[0] + 8, COORD_B[1] - 12, 'B', fontweight='bold', color=border_b)
        
        # Setup plot coordinates (matching SVG layout flipped vertically for normal cartesian viewing, or direct pixel mapping)
        ax.set_xlim(-10, 310)
        ax.set_ylim(230, -10)  # Invert Y to match SVG coordinates where 0,0 is top-left
        ax.axis('off')
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Matplotlib plot saved to: {save_path}")
    else:
        plt.show()


# --- 4. Standalone HTML/SVG Dashboard Generator ---
def export_to_html(model: MultiComplexModel, filepath: str = "rips_visualization.html"):
    """
    Generates a standalone, beautiful HTML dashboard mirroring the original layout,
    pre-rendered with the current model's thresholds.
    """
    trapped = model.get_trapped_counts()
    
    # CSS helper variables mirroring premium styling guidelines
    badge_style = "display:inline-flex; align-items:center; gap:6px; padding:8px 16px; border-radius:8px; font-size:13px; font-weight:600; font-family:sans-serif;"
    
    def get_badge_bg(count: int) -> str:
        if count >= 3:
            return "background: #fde8e8; color: #9b1c1c; border: 1px solid #f8b4b4;"
        elif count >= 1:
            return "background: #fef3c7; color: #92400e; border: 1px solid #fde68a;"
        return "background: #f3f4f6; color: #374151; border: 1px solid #e5e7eb;"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Vietoris-Rips Complexes (Python Export)</title>
    <style>
        :root {{
            --bg: #1e1e2e;
            --surface-1: #252538;
            --surface-2: #303046;
            --text: #cdd6f4;
            --text-secondary: #a6adc8;
            --border: #45475a;
            --border-strong: #585b70;
            --danger: #f38ba8;
            --danger-bg: #5a2e37;
            --danger-border: #f38ba8;
        }}
        body {{
            background-color: var(--bg);
            color: var(--text);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 40px 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .card {{
            background: var(--surface-1);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 32px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            max-width: 720px;
            width: 100%;
        }}
        h1 {{
            font-size: 22px;
            margin-top: 0;
            margin-bottom: 24px;
            text-align: center;
            font-weight: 700;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 24px;
        }}
        .metric-item {{
            background: var(--surface-2);
            padding: 12px 16px;
            border-radius: 8px;
            border: 1px solid var(--border);
            font-size: 13px;
        }}
        .metric-name {{
            color: var(--text-secondary);
            font-weight: bold;
        }}
        .metric-value {{
            float: right;
            font-weight: bold;
        }}
        .svg-container {{
            background: var(--surface-2);
            border-radius: 12px;
            padding: 16px;
            border: 1px solid var(--border);
        }}
        svg text {{
            font-family: inherit;
            fill: var(--text);
            font-weight: 600;
            font-size: 13px;
        }}
        .legend {{
            display: flex;
            justify-content: center;
            gap: 16px;
            margin-top: 24px;
            font-size: 12px;
            color: var(--text-secondary);
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .badge-container {{
            display: flex;
            gap: 16px;
            justify-content: center;
            margin-top: 24px;
        }}
    </style>
</head>
<body>
    <div class="card">
        <h1>Vietoris-Rips Complexes (Static Export)</h1>
        
        <div class="metrics-grid">
"""
    
    # Fill metric grid values
    for m, val in model.thresholds.items():
        unit = METRIC_UNITS[m]
        html_content += f"""            <div class="metric-item">
                <span class="metric-name">{METRIC_TITLES[m]} Threshold:</span>
                <span class="metric-value">{val} {unit}</span>
            </div>\n"""
            
    html_content += f"""        </div>

        <div class="svg-container">
            <svg width="100%" viewBox="0 0 640 490" role="img">
                <title>Four Rips complexes</title>
"""

    # Panel positions in the 2x2 layout
    panel_positions = {
        'travel': (0, 0),
        'transfers': (340, 0),
        'modes': (0, 260),
        'wait': (340, 260)
    }

    for metric, (tx, ty) in panel_positions.items():
        comp = model.complexes[metric]
        html_content += f"""                <g id="panel-{metric}" transform="translate({tx},{ty})">
                    <!-- Border -->
                    <rect x="0" y="20" width="300" height="200" rx="8" fill="none" stroke="var(--border)" stroke-width="1.5"/>
                    <!-- Title -->
                    <text x="150" y="12" text-anchor="middle">{comp.title}</text>
"""
        
        # Render edges
        for edge in comp.get_edges_state():
            p1, p2 = edge['p1'], edge['p2']
            if edge['active']:
                html_content += f'                    <line x1="{p1[0]}" y1="{p1[1]}" x2="{p2[0]}" y2="{p2[1]}" stroke="var(--text)" stroke-width="1.75"/>\n'
            else:
                html_content += f'                    <line x1="{p1[0]}" y1="{p1[1]}" x2="{p2[0]}" y2="{p2[1]}" stroke="var(--border)" stroke-width="1.0" stroke-dasharray="3 3"/>\n'

        # Render boundary nodes
        for pt in R_COORDS:
            html_content += f'                    <circle cx="{pt[0]}" cy="{pt[1]}" r="5" fill="var(--surface-1)" stroke="var(--border-strong)" stroke-width="1.5"/>\n'

        # Render Node A
        fill_a = "var(--danger)" if comp.is_a_trapped else "var(--surface-1)"
        stroke_a = "var(--danger-border)" if comp.is_a_trapped else "var(--border-strong)"
        html_content += f'                    <circle cx="{COORD_A[0]}" cy="{COORD_A[1]}" r="7" fill="{fill_a}" stroke="{stroke_a}" stroke-width="2"/>\n'

        # Render Node B
        fill_b = "var(--danger)" if comp.is_b_trapped else "var(--surface-1)"
        stroke_b = "var(--danger-border)" if comp.is_b_trapped else "var(--border-strong)"
        html_content += f'                    <circle cx="{COORD_B[0]}" cy="{COORD_B[1]}" r="7" fill="{fill_b}" stroke="{stroke_b}" stroke-width="2"/>\n'

        html_content += "                </g>\n"

    # Add legend, badges and end of HTML
    html_content += f"""            </svg>
        </div>

        <div class="badge-container">
            <span style="{badge_style} {get_badge_bg(trapped['A'])}">Point A — trapped in {trapped['A']}/4 complexes</span>
            <span style="{badge_style} {get_badge_bg(trapped['B'])}">Point B — trapped in {trapped['B']}/4 complexes</span>
        </div>

        <div class="legend">
            <div class="legend-item">
                <span style="width:16px; height:0; border-top:2px solid var(--text); display:inline-block;"></span>
                <span>Connected (below threshold)</span>
            </div>
            <div class="legend-item">
                <span style="width:16px; height:0; border-top:2px dashed var(--border); display:inline-block;"></span>
                <span>Not yet connected</span>
            </div>
            <div class="legend-item">
                <span style="width:11px; height:11px; border-radius:50%; background:var(--danger); border:1px solid var(--danger-border); display:inline-block;"></span>
                <span>Trapped in this complex</span>
            </div>
        </div>
    </div>
</body>
</html>
"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Interactive-looking static SVG dashboard saved to: {os.path.abspath(filepath)}")


# --- 5. Main Dummy Data Run Execution ---
if __name__ == "__main__":
    # Custom thresholds (Dummy values that can be adjusted)
    # Default values chosen to showcase trapped & escaped nodes:
    # A is trapped in: wait (12 < 26), travel (15 < 24), transfers (2 < 5). Escaped in modes (2 >= 3? No, wait: 2 < 3, so A is trapped in 4/4)
    # B is trapped in: wait (12 < 24), travel (15 < 22). Escaped in transfers (2 >= 1), modes (2 >= 1). (B is trapped in 2/4)
    dummy_thresholds = {
        'travel': 15,     # min
        'transfers': 2,  # count
        'modes': 2,      # count
        'wait': 12        # min
    }

    # If parameters are passed in command line, override defaults
    # format: travel transfers modes wait
    if len(sys.argv) == 5:
        try:
            dummy_thresholds['travel'] = float(sys.argv[1])
            dummy_thresholds['transfers'] = float(sys.argv[2])
            dummy_thresholds['modes'] = float(sys.argv[3])
            dummy_thresholds['wait'] = float(sys.argv[4])
        except ValueError:
            print("Usage: python3 thesis.py [travel_val] [transfers_val] [modes_val] [wait_val]")
            print("Using default dummy values...")

    # Run the model
    model = MultiComplexModel(dummy_thresholds)
    
    # Output to console
    model.print_summary()
    
    # Generate standalone SVG dashboard HTML
    export_to_html(model, "rips_complexes.html")
    
    # Plot using matplotlib (if library exists, will display or save)
    plot_complexes(model, save_path="rips_complexes.png")
