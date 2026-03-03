"""
RNA secondary structure utilities
"""

from typing import List, Tuple, Optional
import subprocess


def parse_structure(structure: str) -> List[Tuple[int, int]]:
    """
    Parse dot-bracket notation to get base pairs

    Args:
        structure: Dot-bracket string like '(((...)))'

    Returns:
        List of (i, j) base pair indices

    Example:
        >>> parse_structure("(((...)))")
        [(0, 8), (1, 7), (2, 6)]
    """
    pairs = []
    stack = []

    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                pairs.append((j, i))

    return pairs


def predict_structure_rnafold(sequence: str) -> Optional[str]:
    """
    Predict RNA secondary structure using RNAfold

    Args:
        sequence: RNA sequence (ACGU)

    Returns:
        Dot-bracket structure string, or None if RNAfold fails
    """
    try:
        process = subprocess.Popen(
            ['RNAfold', '--noPS'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        output, error = process.communicate(input=sequence, timeout=10)

        # Parse output
        lines = output.strip().split('\n')
        if len(lines) >= 2:
            # Structure line format: "structure (energy)"
            structure_line = lines[1].split()[0]
            return structure_line

        return None

    except FileNotFoundError:
        print("⚠️  RNAfold not found. Install: brew install viennarna")
        return None
    except Exception as e:
        print(f"RNAfold error: {e}")
        return None


def validate_structure(sequence: str, structure: str) -> bool:
    """
    Validate that structure matches sequence length and is well-formed

    Args:
        sequence: RNA sequence
        structure: Dot-bracket structure

    Returns:
        True if valid, False otherwise
    """
    if len(sequence) != len(structure):
        return False

    # Check balanced parentheses
    count = 0
    for char in structure:
        if char == '(':
            count += 1
        elif char == ')':
            count -= 1
        if count < 0:
            return False

    return count == 0