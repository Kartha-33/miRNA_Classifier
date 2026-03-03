"""
Generate Obsidian-style network visualization of RNA samples
"""

import argparse
from src.network_visualizer import create_network_html


def main():
    parser = argparse.ArgumentParser(
        description="Create Obsidian-style network visualization"
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to CSV dataset'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='rna_network.html',
        help='Output HTML file'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=100,
        help='Number of samples to visualize'
    )
    parser.add_argument(
        '--similarity_threshold',
        type=float,
        default=0.3,
        help='Minimum similarity to draw edges (0-1)'
    )
    
    args = parser.parse_args()
    
    create_network_html(
        csv_path=args.data_path,
        output_html=args.output,
        num_samples=args.num_samples,
        similarity_threshold=args.similarity_threshold
    )


if __name__ == "__main__":
    main()