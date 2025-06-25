import re
from collections import Counter
from typing import Dict, List, Union


def formula_to_element_counts(formula_string: str) -> Dict[str, int]:
    """
    Converts a chemical formula to a dictionary of element counts.
    Handles nested groupings with parentheses (e.g., Fe2(SO4)3).

    Parameters
    ----------
    formula_string : str
        A valid chemical formula string (e.g., "C6H12O6", "Fe2(SO4)3")

    Returns
    -------
    dict
        Dictionary of element -> count (e.g., {'Fe': 2, 'S': 3, 'O': 12})
    """
    
    def multiply_counts(base: Dict[str, int], multiplier: int) -> Dict[str, int]:return {element: count * multiplier for element, count in base.items()}

    def parse(tokens: str) -> Counter:
        stack = []
        counts = Counter()
        i = 0
        while i < len(tokens):
            if tokens[i] == '(':
                stack.append((counts, i))
                counts = Counter()
                i += 1
            elif tokens[i] == ')':
                i += 1
                m = re.match(r'\d+', tokens[i:])
                multiplier = int(m.group()) if m else 1
                if m:
                    i += len(m.group())
                prev_counts, _ = stack.pop()
                for k, v in counts.items():
                    prev_counts[k] += v * multiplier
                counts = prev_counts
            else:
                m = re.match(r'([A-Z][a-z]?)(\d*)', tokens[i:])
                if not m:
                    raise ValueError(f"Invalid formula at position {i}: {tokens[i:]}")
                element, num = m.groups()
                count = int(num) if num else 1
                counts[element] += count
                i += len(m.group())
        return counts

    return dict(parse(formula_string))


def count_atoms(formula_string: str) -> int:
    """
    Counts the total number of atoms in a chemical formula.

    Parameters
    ----------
    formula_string : str
        A valid chemical formula string

    Returns
    -------
    int
        Total number of atoms
    """
    return sum(formula_to_element_counts(formula_string).values())


def count_atoms_in_set(formula_string: str, which_atoms: List[str]) -> int:
    """
    Counts the total number of atoms in a formula that belong to a specified set of elements.

    Parameters
    ----------
    formula_string : str
        A valid chemical formula string
    which_atoms : list of str
        List of elements to include in the count (e.g., ['C', 'H'])

    Returns
    -------
    int
        Number of atoms in the specified subset
    """
    counts = formula_to_element_counts(formula_string)
    return sum(count for element, count in counts.items() if element in which_atoms)


def get_first_entry(entry: Union[List, str]) -> str:
    """
    Returns the first entry from a list or the string itself if not a list.

    Parameters
    ----------
    entry : list or str
        Possibly a list of strings or a single string.

    Returns
    -------
    str
        First string entry
    """
    if isinstance(entry, list):
        return entry[0]
    return entry
