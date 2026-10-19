"""Primordial sound decomposition: rhythm and tonality.

A primal sonority is decomposed along two axes placed in parallel,
respecting the geometric triangularisation (see triangle.py):
- rhythm (temporal, percussive axis)
- tonality (spectral, harmonic axis)

Both decompositions are placed side by side on the same triangular tree,
mirroring the light decomposition (genesis/spectrum.py for the visual pole).
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

Tonalité_SYSTEMS = {
    "pythagoreen": [1.0, 9 / 8, 81 / 64, 4 / 3, 3 / 2, 27 / 16, 243 / 128],
    "pentatonique": [1.0, 9 / 8, 81 / 64, 4 / 3, 3 / 2],
    "octaves": [1.0, 2.0, 4.0, 8.0],
}

RHYTHM_HIERARCHY = ["pulsation", "temps", "mesure", "phrase", "cycle"]


@dataclass
class RhythmNode:
    """One level of the rhythmic decomposition."""
    level: str
    bpm: float
    poids: float


@dataclass
class TonalityNode:
    """One partial of the tonal decomposition."""
    degree: int
    ratio: float
    poids: float


@dataclass
class SonorityDecomposition:
    """A primal sonority split into rhythm and tonality in parallel."""
    rhythm: List[RhythmNode] = field(default_factory=list)
    tonality: List[TonalityNode] = field(default_factory=list)


class SoundDecomposer:
    """Decompose a vector along rhythm and tonality axes.

    Deterministic mapping, parallel to LightDecomposer:
    - even dimensions -> rhythm (temporal structure)
    - odd dimensions  -> tonality (harmonic structure)
    """

    def __init__(self, base_bpm: float = 60.0, system: str = "pythagoreen"):
        self.base_bpm = base_bpm
        self.ratios = Tonalité_SYSTEMS.get(system, Tonalité_SYSTEMS["pythagoreen"])

    def decompose(self, vector: np.ndarray) -> SonorityDecomposition:
        v = np.asarray(vector, dtype=float).ravel()
        if v.size == 0:
            return SonorityDecomposition()
        even = v[0::2]
        odd = v[1::2]

        rhythm: List[RhythmNode] = []
        n_levels = min(len(RHYTHM_HIERARCHY), max(2, even.size))
        for i in range(n_levels):
            chunk = even[i::n_levels] if n_levels else even
            poids = float(np.linalg.norm(chunk)) if chunk.size else 0.0
            bpm = self.base_bpm * (2.0 ** i)
            rhythm.append(RhythmNode(level=RHYTHM_HIERARCHY[i], bpm=bpm, poids=poids))

        tonality: List[TonalityNode] = []
        for i in range(len(self.ratios)):
            chunk = odd[i::len(self.ratios)]
            poids = float(np.linalg.norm(chunk)) if chunk.size else 0.0
            tonality.append(TonalityNode(degree=i, ratio=self.ratios[i], poids=poids))

        return SonorityDecomposition(rhythm=rhythm, tonality=tonality)


def tempo_couleur(bpm: float) -> float:
    """Map a rhythm to a wavelength in parallel with tonality (fast=short waves)."""
    ref = 120.0
    base_nm = 550.0
    return float(np.clip(base_nm * (ref / max(bpm, 1e-6)) ** 0.5, 380.0, 780.0))
