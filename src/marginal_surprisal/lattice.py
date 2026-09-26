def build_lattice(s: bytes, surfaces: set[bytes], max_len: int) -> list[list[tuple[int, bytes]]]:
    """Build lattice[j] = list of (i, token) such that s[j:i] == token.

    Positions are byte offsets. Edges are indexed by their start position so the DP can run
    forwards and skip positions that no tokenization reaches.
    """
    n = len(s)
    lattice: list[list[tuple[int, bytes]]] = [[] for _ in range(n + 1)]
    for j in range(n):
        for i in range(j + 1, min(n, j + max_len) + 1):
            tok = s[j:i]
            if tok in surfaces:
                lattice[j].append((i, tok))
    return lattice
