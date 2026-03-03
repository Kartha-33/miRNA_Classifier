"""
Generate interactive RNA structure visualizations
"""

import argparse
from src.visualizer import RNAGraphVisualizer, create_interactive_html


def main():
    parser = argparse.ArgumentParser(
        description="Create interactive RNA graph visualizations (Obsidian-style)"
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
        default='rna_visualizations.html',
        help='Output HTML file'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=10,
        help='Number of samples to visualize'
    )
    parser.add_argument(
        '--layout',
        type=str,
        default='spring',
        choices=['spring', 'circular', 'kamada_kawai'],
        help='Graph layout algorithm'
    )
    parser.add_argument(
        '--sequence',
        type=str,
        default=None,
        help='Visualize single sequence (optional)'
    )
    parser.add_argument(
        '--structure',
        type=str,
        default=None,
        help='Structure for single sequence (optional)'
    )
    
    args = parser.parse_args()
    
    if args.sequence:
        # Single sequence visualization
        viz = RNAGraphVisualizer()
        fig = viz.visualize_rna(
            sequence=args.sequence,
            structure=args.structure,
            title="RNA Structure Graph",
            layout=args.layout
        )
        fig.write_html(args.output)
        print(f"✓ Visualization saved to {args.output}")
        print(f"  Open in browser: open {args.output}")
    else:
        # Batch visualization from CSV
        create_interactive_html(
            csv_path=args.data_path,
            output_html=args.output,
            num_samples=args.num_samples,
            layout=args.layout
        )
        print(f"\n✓ Open in browser: open {args.output}")


if __name__ == "__main__":
    main()