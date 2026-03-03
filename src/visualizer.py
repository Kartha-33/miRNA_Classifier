"""
Interactive RNA Graph Visualization with Obsidian-style interactivity
"""

import plotly.graph_objects as go
import networkx as nx
import torch
from typing import Optional, List
import numpy as np
import pandas as pd

from .graph_builder import sequence_to_graph


class RNAGraphVisualizer:
    """Interactive visualization of RNA sequence and graph structure"""
    
    def __init__(self):
        self.nucleotide_colors = {
            'A': '#FF6B6B',  # Red
            'C': '#4ECDC4',  # Cyan
            'G': '#FFE66D',  # Yellow
            'U': '#95E1D3',  # Green
            'N': '#CCCCCC'   # Gray
        }
    
    def visualize_rna(
        self,
        sequence: str,
        structure: Optional[str] = None,
        title: str = "RNA Structure Graph",
        layout: str = "spring",
        predicted_label: Optional[int] = None,
        confidence: Optional[float] = None
    ) -> go.Figure:
        """Create interactive Obsidian-style graph visualization"""
        
        # Build graph
        graph_data = sequence_to_graph(sequence, structure)
        G = self._to_networkx(graph_data, sequence)
        
        # Get layout
        if layout == "spring":
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Create figure
        fig = go.Figure()
        
        # Add edges with different colors based on type
        self._add_edges(fig, G, pos, graph_data)
        
        # Add nodes
        self._add_nodes(fig, G, pos, sequence, structure)
        
        # Update layout for interactivity
        title_text = self._create_title(title, sequence, structure, predicted_label, confidence)
        
        fig.update_layout(
            title={
                'text': title_text,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            showlegend=True,
            hovermode='closest',
            height=800,
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='white',
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title=''
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title=''
            ),
            # Add drag mode for interactivity
            dragmode='pan',
            # Add modebar buttons
            modebar=dict(
                orientation='v',
                bgcolor='rgba(255,255,255,0.8)'
            )
        )
        
        # Add legend
        self._add_legend(fig)
        
        return fig
    
    def _to_networkx(self, graph_data, sequence: str) -> nx.Graph:
        """Convert PyG graph to NetworkX"""
        G = nx.Graph()
        
        for i, nuc in enumerate(sequence):
            G.add_node(i, nucleotide=nuc, color=self.nucleotide_colors.get(nuc, '#CCCCCC'))
        
        edge_index = graph_data.edge_index.cpu().numpy()
        edge_attr = graph_data.edge_attr.cpu().numpy() if hasattr(graph_data, 'edge_attr') else None
        
        for idx, (src, dst) in enumerate(edge_index.T):
            if src < dst:
                edge_type = edge_attr[idx] if edge_attr is not None else 0
                G.add_edge(int(src), int(dst), edge_type=int(edge_type))
        
        return G
    
    def _add_edges(self, fig, G, pos, graph_data):
        """Add edges with different styles based on type"""
        
        # Separate edges by type
        backbone_edges = []
        basepair_edges = []
        selfloop_edges = []
        
        for edge in G.edges(data=True):
            src, dst, data = edge
            edge_type = data.get('edge_type', 0)
            
            if edge_type == 0:  # Backbone
                backbone_edges.append((src, dst))
            elif edge_type == 1:  # Base pair
                basepair_edges.append((src, dst))
            else:  # Self-loop
                selfloop_edges.append((src, dst))
        
        # Draw backbone edges (gray, thin)
        if backbone_edges:
            edge_x, edge_y = [], []
            for src, dst in backbone_edges:
                x0, y0 = pos[src]
                x1, y1 = pos[dst]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=2, color='#95a5a6'),
                hoverinfo='none',
                showlegend=True,
                name='Backbone',
                legendgroup='backbone'
            ))
        
        # Draw base-pairing edges (blue, thick, dashed)
        if basepair_edges:
            edge_x, edge_y = [], []
            for src, dst in basepair_edges:
                x0, y0 = pos[src]
                x1, y1 = pos[dst]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=3, color='#3498db', dash='dash'),
                hoverinfo='none',
                showlegend=True,
                name='Base Pairing',
                legendgroup='basepair'
            ))
    
    def _add_nodes(self, fig, G, pos, sequence, structure):
        """Add nodes with colors and labels"""
        
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_colors = [G.nodes[node]['color'] for node in G.nodes()]
        
        # Create hover text
        hover_text = []
        for i in G.nodes():
            nuc = G.nodes[i]['nucleotide']
            struct_char = structure[i] if structure and i < len(structure) else '.'
            degree = G.degree(i)
            hover_text.append(
                f"<b>Position {i}</b><br>"
                f"Nucleotide: <b>{nuc}</b><br>"
                f"Structure: {struct_char}<br>"
                f"Connections: {degree}"
            )
        
        # Add node trace
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=25,
                color=node_colors,
                line=dict(width=2, color='white'),
                opacity=0.9
            ),
            text=[G.nodes[node]['nucleotide'] for node in G.nodes()],
            textposition="middle center",
            textfont=dict(size=12, color='black', family='Arial Black'),
            hovertext=hover_text,
            hoverinfo='text',
            showlegend=False,
            name='Nucleotides'
        ))
    
    def _add_legend(self, fig):
        """Add color legend for nucleotides"""
        legend_items = [
            ('A', '#FF6B6B'),
            ('C', '#4ECDC4'),
            ('G', '#FFE66D'),
            ('U', '#95E1D3')
        ]
        
        # Position legend items
        for idx, (nuc, color) in enumerate(legend_items):
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=15, color=color, line=dict(width=2, color='white')),
                showlegend=True,
                name=nuc,
                legendgroup='nucleotides'
            ))
    
    def _create_title(self, title, sequence, structure, predicted_label, confidence):
        """Create formatted title"""
        title_parts = [title]
        
        title_parts.append(f"<br><sub>Sequence Length: {len(sequence)}</sub>")
        
        if structure:
            pairs = structure.count('(')
            title_parts.append(f"<sub> | Base Pairs: {pairs}</sub>")
        
        if predicted_label is not None:
            label_str = "✓ miRNA" if predicted_label == 1 else "✗ non-miRNA"
            conf_str = f" ({confidence:.1%})" if confidence else ""
            title_parts.append(f"<br><sub>Prediction: <b>{label_str}</b>{conf_str}</sub>")
        
        return ''.join(title_parts)


def create_interactive_html(
    csv_path: str,
    output_html: str = "rna_visualizations.html",
    num_samples: int = 10,
    layout: str = "spring"
):
    """
    Generate interactive HTML with all visualizations
    """
    df = pd.read_csv(csv_path)
    df = df.head(num_samples)
    
    viz = RNAGraphVisualizer()
    
    # Create HTML template
    html_parts = ["""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>RNA Structure Visualizations</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        h1 {
            color: white;
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .subtitle {
            color: rgba(255,255,255,0.9);
            text-align: center;
            font-size: 1.2em;
            margin-bottom: 30px;
        }
        
        .viz-container {
            background: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            transition: transform 0.3s ease;
        }
        
        .viz-container:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 50px rgba(0,0,0,0.4);
        }
        
        .sample-info {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            border-left: 5px solid #667eea;
        }
        
        .sample-info h3 {
            margin: 0 0 10px 0;
            color: #2c3e50;
        }
        
        .info-row {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            margin-top: 10px;
        }
        
        .info-item {
            background: white;
            padding: 8px 15px;
            border-radius: 5px;
            font-size: 0.9em;
        }
        
        .label-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }
        
        .label-mirna {
            background: #2ecc71;
            color: white;
        }
        
        .label-non-mirna {
            background: #e74c3c;
            color: white;
        }
        
        .controls {
            background: white;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.2);
        }
        
        .controls h3 {
            margin-top: 0;
            color: #2c3e50;
        }
        
        .controls p {
            color: #7f8c8d;
            margin: 5px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧬 RNA Structure Graph Visualizations</h1>
        <p class="subtitle">Interactive network visualization of RNA secondary structures</p>
        
        <div class="controls">
            <h3>📌 How to Interact</h3>
            <p>🖱️ <b>Drag</b>: Pan the view</p>
            <p>🔍 <b>Scroll</b>: Zoom in/out</p>
            <p>👆 <b>Hover</b>: See nucleotide details</p>
            <p>🎨 <b>Colors</b>: A=Red, C=Cyan, G=Yellow, U=Green</p>
            <p>📊 <b>Edges</b>: Gray=Backbone, Blue Dashed=Base Pairing</p>
        </div>
"""]
    
    # Generate each visualization
    for idx, row in df.iterrows():
        sequence = row['sequence']
        structure = row.get('structure', None)
        if pd.isna(structure) or structure == '':
            structure = None
        label = row.get('label', None)
        sample_id = row.get('id', f'Sample_{idx}')
        
        # Create figure
        label_int = int(label) if label is not None else None
        title = f"{sample_id}"
        
        fig = viz.visualize_rna(
            sequence=sequence,
            structure=structure,
            title=title,
            layout=layout,
            predicted_label=label_int
        )
        
        # Add sample info
        label_str = "miRNA" if label == 1 else "non-miRNA"
        label_class = "label-mirna" if label == 1 else "label-non-mirna"
        
        html_parts.append(f"""
        <div class="viz-container">
            <div class="sample-info">
                <h3>{sample_id}</h3>
                <div class="info-row">
                    <div class="info-item"><b>Length:</b> {len(sequence)}</div>
                    <div class="info-item"><b>Structure:</b> {'Yes' if structure else 'No'}</div>
                    <div class="info-item"><span class="label-badge {label_class}">{label_str}</span></div>
                </div>
            </div>
            <div id="viz_{idx}"></div>
        </div>
""")
        
        # Add plotly script
        fig_html = fig.to_html(include_plotlyjs=False, div_id=f"viz_{idx}")
        html_parts.append(f"<script>{fig_html}</script>\n")
    
    html_parts.append("""
    </div>
</body>
</html>
""")
    
    # Write file
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(''.join(html_parts))
    
    print(f"✓ Interactive visualizations saved to {output_html}")
    print(f"  Generated {num_samples} visualizations")
    print(f"  Open in browser to interact!")