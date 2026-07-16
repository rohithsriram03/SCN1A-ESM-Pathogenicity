from pathlib import Path

from sklearn.model_selection import train_test_split

from .config import MAX_AA_LENGTH, SEED, TEST_SIZE


def read_fasta(path) -> str:
    """Concatenate the residues of a single-record FASTA file."""
    lines = Path(path).read_text().splitlines()
    return "".join(l.strip() for l in lines if l and not l.startswith(">"))


def apply_mutation(wildtype: str, position: int, ref: str, alt: str) -> str:
    """Substitute `alt` at 1-indexed `position`; raise if `ref` doesn't match."""
    i = int(position) - 1
    if wildtype[i] != ref:
        raise ValueError(f"ref {ref} != {wildtype[i]} at position {position}")
    return wildtype[:i] + alt + wildtype[i + 1:]


def sequence_window(sequence: str, position: int, max_len: int = MAX_AA_LENGTH):
    """Up to `max_len` residues centred on 1-indexed `position`.

    Returns (window, index_within_window, start, end) with 1-indexed bounds.
    """
    pos = int(position) - 1
    start = max(0, pos - max_len // 2)
    end = min(len(sequence), start + max_len)
    start = max(0, end - max_len)
    return sequence[start:end], pos - start, start + 1, end


def stratified_split(*arrays, labels, test_size: float = TEST_SIZE, seed: int = SEED):
    """train_test_split with the project's fixed seed and label stratification."""
    return train_test_split(*arrays, test_size=test_size, random_state=seed, stratify=labels)
