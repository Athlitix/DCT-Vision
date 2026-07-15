"""Duplicate / near-duplicate image detection via DCT perceptual hashing.

pHash is a DCT algorithm, and we already hold the DCT coefficients, so hashing
skips the usual decode -> resize -> DCT steps entirely.
"""

from __future__ import annotations

from pathlib import Path

from dct_vision.core.dct_image import DCTImage
from dct_vision.ops.phash import hamming_distance, perceptual_hash

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _load_dct(path: str) -> DCTImage:
    if path.lower().endswith((".jpg", ".jpeg")):
        return DCTImage.from_file(path)
    from dct_vision.io.convert import convert_to_dct
    return convert_to_dct(path)


def hash_folder(root: str, hash_size: int = 8) -> list[tuple[str, int]]:
    """Compute a perceptual hash for every image under ``root`` (recursively)."""
    paths = sorted(
        str(p) for p in Path(root).rglob("*") if p.suffix.lower() in _IMAGE_EXTS
    )
    out = []
    for p in paths:
        try:
            out.append((p, perceptual_hash(_load_dct(p), hash_size)))
        except Exception:  # noqa: BLE001 - skip unreadable files, keep going
            continue
    return out


def find_duplicates(
    root: str,
    max_distance: int = 5,
    hash_size: int = 8,
) -> list[list[str]]:
    """Group near-duplicate images in a folder.

    Two images are considered near-duplicates when the Hamming distance between
    their perceptual hashes is <= ``max_distance``. Grouping is transitive
    (single-linkage) via union-find.

    Parameters
    ----------
    root : str
        Directory to scan (recursively).
    max_distance : int
        Maximum Hamming distance (in bits) to treat two images as duplicates.
        0 = identical hash only; larger = looser matching.
    hash_size : int
        Perceptual hash dimension (hash_size**2 bits).

    Returns
    -------
    list[list[str]]
        Groups of 2+ paths that are mutually near-duplicate, each group sorted.
        Images with no duplicate are not returned.
    """
    hashes = hash_folder(root, hash_size)
    n = len(hashes)

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    for i in range(n):
        for j in range(i + 1, n):
            if hamming_distance(hashes[i][1], hashes[j][1]) <= max_distance:
                union(i, j)

    groups: dict[int, list[str]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(hashes[i][0])

    return sorted(
        (sorted(g) for g in groups.values() if len(g) > 1),
        key=lambda g: g[0],
    )
