"""
Professional network visualization showing relationships between RNA samples
"""

import plotly.graph_objects as go
import networkx as nx
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


class RNANetworkVisualizer:
    """Create professional network of RNA samples with clustering analysis"""
    
    def __init__(self):
        # Modern professional color scheme
        self.mirna_color = '#3b82f6'      # Blue for miRNA
        self.non_mirna_color = '#1e293b'  # Dark slate for non-miRNA
        self.edge_color = 'rgba(100, 116, 139, 0.15)'  # Subtle gray for edges
        self.bg_color = '#0f172a'         # Deep navy background
        self.text_color = '#e2e8f0'       # Light text
    
    def create_network(
        self,
        csv_path: str,
        num_samples: int = 100,
        similarity_threshold: float = 0.3
    ) -> go.Figure:
        """
        Create interactive network where:
        - Nodes = RNA samples
        - Edges = Sequence similarity
        - Clusters = Groups of similar sequences
        - Node size = Sequence length
        - Node color = miRNA (blue) vs non-miRNA (dark)
        """
        
        # Load data
        df = pd.read_csv(csv_path).head(num_samples)
        
        # Build network graph
        G = nx.Graph()
        
        # Add nodes
        for idx, row in df.iterrows():
            sequence = row['sequence']
            label = row.get('label', 0)
            sample_id = row.get('id', f'Sample_{idx}')
            structure = row.get('structure', '')
            
            # Calculate node attributes
            gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence) * 100
            
            G.add_node(
                idx,
                id=sample_id,
                sequence=sequence,
                structure=structure,
                label=label,
                length=len(sequence),
                gc_content=gc_content,
                composition=self._get_composition(sequence)
            )
        
        # Add edges based on sequence similarity
        print("Computing sequence similarities...")
        sequences = [df.iloc[i]['sequence'] for i in range(len(df))]
        similarities = self._compute_similarities(sequences)
        
        edge_count = 0
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                sim = similarities[i][j]
                if sim > similarity_threshold:
                    G.add_edge(i, j, weight=sim)
                    edge_count += 1
        
        print(f"✓ Network: {len(G.nodes)} nodes, {edge_count} edges")
        
        # Detect communities (clusters)
        print("Detecting clusters...")
        clusters = self._detect_clusters(G)
        
        # Create visualization
        fig = self._create_figure(G, df, clusters, similarities)
        
        return fig
    
    def _get_composition(self, sequence: str) -> Dict[str, float]:
        """Get nucleotide composition"""
        counts = Counter(sequence)
        total = len(sequence)
        return {nuc: counts.get(nuc, 0) / total for nuc in ['A', 'C', 'G', 'U']}
    
    def _compute_similarities(self, sequences: List[str]) -> np.ndarray:
        """
        Compute pairwise sequence similarities using k-mer vectors
        
        WHY: Sequences with similar k-mer patterns likely have:
        - Similar biological function
        - Common evolutionary origin
        - Similar secondary structure
        """
        def sequence_to_kmer_vector(seq: str, k: int = 3) -> np.ndarray:
            """Convert sequence to k-mer frequency vector"""
            kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
            kmer_counts = Counter(kmers)
            
            # Standard RNA k-mer vocabulary
            vocab = ['AAA', 'AAC', 'AAG', 'AAU', 'ACA', 'ACC', 'ACG', 'ACU',
                    'AGA', 'AGC', 'AGG', 'AGU', 'AUA', 'AUC', 'AUG', 'AUU',
                    'CAA', 'CAC', 'CAG', 'CAU', 'CCA', 'CCC', 'CCG', 'CCU',
                    'CGA', 'CGC', 'CGG', 'CGU', 'CUA', 'CUC', 'CUG', 'CUU',
                    'GAA', 'GAC', 'GAG', 'GAU', 'GCA', 'GCC', 'GCG', 'GCU',
                    'GGA', 'GGC', 'GGG', 'GGU', 'GUA', 'GUC', 'GUG', 'GUU',
                    'UAA', 'UAC', 'UAG', 'UAU', 'UCA', 'UCC', 'UCG', 'UCU',
                    'UGA', 'UGC', 'UGG', 'UGU', 'UUA', 'UUC', 'UUG', 'UUU']
            
            vec = np.array([kmer_counts.get(k, 0) for k in vocab], dtype=float)
            # Normalize
            if vec.sum() > 0:
                vec = vec / vec.sum()
            return vec
        
        # Convert sequences to vectors
        vectors = np.array([sequence_to_kmer_vector(seq) for seq in sequences])
        
        # Compute cosine similarity
        similarities = cosine_similarity(vectors)
        
        return similarities
    
    def _detect_clusters(self, G: nx.Graph) -> Dict[int, int]:
        """
        Detect clusters using community detection
        
        CLUSTERS MEAN:
        - Group of sequences with high mutual similarity
        - May share biological function
        - Could be from same miRNA family
        - Similar evolutionary origin
        """
        try:
            from networkx.algorithms import community
            communities = community.greedy_modularity_communities(G)
            
            # Create node -> cluster mapping
            cluster_map = {}
            for cluster_id, nodes in enumerate(communities):
                for node in nodes:
                    cluster_map[node] = cluster_id
            
            print(f"✓ Found {len(communities)} clusters")
            return cluster_map
        except:
            # Fallback: no clustering
            return {node: 0 for node in G.nodes()}
    
    def _create_figure(self, G: nx.Graph, df: pd.DataFrame, clusters: Dict, similarities: np.ndarray) -> go.Figure:
        """Create professional Plotly figure"""
        
        # Layout - use force-directed for organic clustering
        print("Computing layout...")
        pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)
        
        # Prepare node data
        node_x = []
        node_y = []
        node_sizes = []
        node_colors = []
        node_text = []
        hover_text = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node attributes
            data = G.nodes[node]
            label = data['label']
            length = data['length']
            sample_id = data['id']
            gc_content = data['gc_content']
            comp = data['composition']
            cluster_id = clusters.get(node, 0)
            
            # Size based on sequence length
            size = 15 + (length / 8)
            node_sizes.append(size)
            
            # Color based on label (modern blue/dark scheme)
            if label == 1:
                color = self.mirna_color  # Blue for miRNA
            else:
                color = self.non_mirna_color  # Dark for non-miRNA
            node_colors.append(color)
            
            # Text label (shortened)
            node_text.append(sample_id[:8])
            
            # Calculate average similarity to connected nodes
            neighbors = list(G.neighbors(node))
            if neighbors:
                avg_sim = np.mean([similarities[node][n] for n in neighbors])
                connectivity = "High" if avg_sim > 0.5 else "Medium" if avg_sim > 0.3 else "Low"
            else:
                connectivity = "Isolated"
                avg_sim = 0
            
            # Hover info with cluster explanation
            label_str = "miRNA" if label == 1 else "non-miRNA"
            hover = (
                f"<b>{sample_id}</b><br>"
                f"<b>Type:</b> {label_str}<br>"
                f"<b>Cluster:</b> {cluster_id}<br>"
                f"<b>Length:</b> {length} nt<br>"
                f"<b>GC Content:</b> {gc_content:.1f}%<br>"
                f"<b>Composition:</b> A:{comp['A']:.1%} C:{comp['C']:.1%} G:{comp['G']:.1%} U:{comp['U']:.1%}<br>"
                f"<b>Connections:</b> {G.degree(node)}<br>"
                f"<b>Avg Similarity:</b> {avg_sim:.2f}<br>"
                f"<b>Connectivity:</b> {connectivity}"
            )
            hover_text.append(hover)
        
        # Prepare edge data
        edge_x = []
        edge_y = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create figure
        fig = go.Figure()
        
        # Add edges (subtle)
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=0.5, color=self.edge_color),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=1.5, color='rgba(255,255,255,0.3)'),
                opacity=0.85
            ),
            text=node_text,
            textposition="middle center",
            textfont=dict(size=7, color='white', family='Inter, sans-serif'),
            hovertext=hover_text,
            hoverinfo='text',
            showlegend=False
        ))
        
        # Professional layout
        fig.update_layout(
            title={
                'text': "RNA Sequence Network Analysis<br><sub>Clustering based on sequence similarity</sub>",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 28, 'color': self.text_color, 'family': 'Inter, sans-serif'}
            },
            showlegend=False,
            hovermode='closest',
            height=900,
            paper_bgcolor=self.bg_color,
            plot_bgcolor=self.bg_color,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
            dragmode='pan',
            font=dict(family='Inter, sans-serif', color=self.text_color)
        )
        
        return fig


def create_network_html(
    csv_path: str,
    output_html: str = "rna_network.html",
    num_samples: int = 100,
    similarity_threshold: float = 0.3
):
    """Generate professional HTML with network visualization and explanation"""
    
    viz = RNANetworkVisualizer()
    fig = viz.create_network(csv_path, num_samples, similarity_threshold)
    
    # Professional HTML template
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>RNA Network Analysis</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #e2e8f0;
            overflow-x: hidden;
        }}
        
        .container {{
            display: flex;
            height: 100vh;
        }}
        
        .sidebar {{
            width: 380px;
            background: rgba(15, 23, 42, 0.95);
            backdrop-filter: blur(10px);
            padding: 30px;
            overflow-y: auto;
            border-right: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 2px 0 20px rgba(0, 0, 0, 0.3);
        }}
        
        .main-content {{
            flex: 1;
            position: relative;
        }}
        
        #network {{
            width: 100%;
            height: 100vh;
        }}
        
        h1 {{
            font-size: 1.5em;
            font-weight: 700;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        h2 {{
            font-size: 1.1em;
            font-weight: 600;
            color: #94a3b8;
            margin: 25px 0 15px 0;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .subtitle {{
            color: #64748b;
            font-size: 0.9em;
            margin-bottom: 25px;
            line-height: 1.6;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 12px 0;
            font-size: 0.9em;
        }}
        
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
            margin-right: 12px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }}
        
        .info-box {{
            background: rgba(30, 41, 59, 0.6);
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 3px solid #3b82f6;
            font-size: 0.85em;
            line-height: 1.7;
        }}
        
        .info-box strong {{
            color: #3b82f6;
            font-weight: 600;
        }}
        
        .info-box ul {{
            margin: 10px 0 0 20px;
        }}
        
        .info-box li {{
            margin: 6px 0;
            color: #cbd5e1;
        }}
        
        .controls {{
            background: rgba(30, 41, 59, 0.4);
            padding: 12px;
            border-radius: 8px;
            margin: 15px 0;
            font-size: 0.85em;
        }}
        
        .controls p {{
            margin: 6px 0;
            color: #94a3b8;
        }}
        
        .highlight {{
            color: #3b82f6;
            font-weight: 600;
        }}
        
        ::-webkit-scrollbar {{
            width: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: rgba(0, 0, 0, 0.2);
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: rgba(59, 130, 246, 0.5);
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: rgba(59, 130, 246, 0.7);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h1>🧬 RNA Network Analysis</h1>
            <p class="subtitle">Interactive visualization of sequence relationships and clustering patterns</p>
            
            <h2>🎨 Legend</h2>
            <div class="legend-item">
                <div class="legend-color" style="background: #3b82f6;"></div>
                <span><strong>Blue</strong> = miRNA sequences</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #1e293b;"></div>
                <span><strong>Dark</strong> = non-miRNA sequences</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: transparent; border: 2px solid #64748b; width: 12px; height: 12px;"></div>
                <span>Node size = Sequence length</span>
            </div>
            
            <h2>📊 What Are The Clusters?</h2>
            <div class="info-box">
                <strong>Central clusters (dense regions) represent:</strong>
                <ul>
                    <li><strong>Sequence Families:</strong> Groups of RNA sequences with similar k-mer patterns</li>
                    <li><strong>Functional Similarity:</strong> Sequences that likely share biological functions</li>
                    <li><strong>Structural Similarity:</strong> Similar secondary structure motifs</li>
                    <li><strong>Evolutionary Relationships:</strong> May have common evolutionary origin</li>
                </ul>
            </div>
            
            <div class="info-box">
                <strong>Why sequences cluster together:</strong>
                <ul>
                    <li>High <span class="highlight">k-mer similarity</span> (3-base patterns)</li>
                    <li>Similar <span class="highlight">GC content</span></li>
                    <li>Shared <span class="highlight">sequence motifs</span></li>
                    <li>Related <span class="highlight">biological function</span></li>
                </ul>
            </div>
            
            <h2>🔍 How To Interpret</h2>
            <div class="info-box">
                <strong>The connections (edges) show:</strong>
                <ul>
                    <li>Sequences with >30% similarity</li>
                    <li>Thicker clusters = more similarity</li>
                    <li>Isolated nodes = unique sequences</li>
                    <li>Bridge nodes = connect different families</li>
                </ul>
            </div>
            
            <h2>⌨️ Navigation</h2>
            <div class="controls">
                <p>🖱️ <strong>Drag:</strong> Pan the view</p>
                <p>🔍 <strong>Scroll:</strong> Zoom in/out</p>
                <p>👆 <strong>Hover:</strong> See sequence details</p>
                <p>🎯 <strong>Click:</strong> Select node</p>
            </div>
            
            <h2>📈 Technical Details</h2>
            <div class="info-box">
                <strong>Similarity Calculation:</strong>
                <ul>
                    <li>Method: Cosine similarity of k-mer vectors</li>
                    <li>K-mer size: 3 nucleotides</li>
                    <li>Layout: Force-directed (spring)</li>
                    <li>Clustering: Greedy modularity</li>
                </ul>
            </div>
        </div>
        
        <div class="main-content">
            <div id="network"></div>
        </div>
    </div>
    
    {plot_div}
</body>
</html>
"""
    
    plot_div = fig.to_html(include_plotlyjs=False, div_id='network')
    html_content = html_template.format(plot_div=plot_div)
    
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✓ Professional network visualization saved to {output_html}")
    print(f"\n📊 Understanding the clusters:")
    print(f"  • Central dense regions = sequence families with high similarity")
    print(f"  • Connections = shared k-mer patterns (3-base sequences)")
    print(f"  • Node size = sequence length")
    print(f"  • Blue = miRNA, Dark = non-miRNA")
    print(f"\n  Open in browser: open {output_html}")