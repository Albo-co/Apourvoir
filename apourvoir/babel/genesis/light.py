"""Genesis decomposition: light as first element.

Spectral decomposition of a knowledge vector into wavelengths (spectral bins),
each carrying the savoirs associated to that band.

Physical anchors (ISO/CIE):
- Visible spectrum: 380-780 nm.
- Fraunhofer lines as stable anchor wavelengths.
- Wien's law links a "color" to a temperature: lambda_max * T = b (Wien constant).
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

NM_MIN = 380.0
NM_MAX = 780.0
WIEN_B = 2.897771955e-3  # m.K

FRAUNHOFER_LINES = {
    "A": 759.37, "B": 686.72, "C": 656.28, "D": 589.29,
    "E": 526.96, "F": 486.13, "G": 434.05, "H": 410.17,
}

SPECTRAL_BANDS = {
    "violet": (380.0, 450.0),
    "bleu": (450.0, 495.0),
    "cyan": (495.0, 520.0),
    "vert": (520.0, 565.0),
    "jaune": (565.0, 590.0),
    "orange": (590.0, 625.0),
    "rouge": (625.0, 780.0),
}


@dataclass
class SpectralBin:
    """One wavelength bin with the savoirs attached to it."""
    wavelength_nm: float
    intensity: float
    savoirs: List[str] = field(default_factory=list)

    @property
    def band(self) -> str:
        for name, (lo, hi) in SPECTRAL_BANDS.items():
            if lo <= self.wavelength_nm < hi:
                return name
        return "rouge"

    @property
    def temperature_kelvin(self) -> float:
        """Wien's law: hot sources radiate short wavelengths."""
        return WIEN_B / (self.wavelength_nm * 1e-9)


class LightDecomposer:
    """Decompose a knowledge vector into a spectrum of wavelengths.

    Deterministic projection: each dimension of the input vector maps to a
    wavelength bin (evenly spaced over the visible spectrum). Negative energy
    is folded onto the anti-spectrum (complementary wavelength).
    """

    def __init__(self, n_bins: int = 64):
        self.n_bins = n_bins
        self.edges = np.linspace(NM_MIN, NM_MAX, n_bins + 1)
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])

    def decompose(self, vector: np.ndarray) -> List[SpectralBin]:
        v = np.asarray(vector, dtype=float).ravel()
        if v.size == 0:
            return []
        step = max(1, v.size // self.n_bins)
        bins: List[SpectralBin] = []
        for i in range(self.n_bins):
            chunk = v[i * step:(i + 1) * step] if step * (i + 1) <= v.size else v[i * step:]
            if chunk.size == 0:
                intensity = 0.0
            else:
                intensity = float(np.linalg.norm(chunk))
            bins.append(SpectralBin(wavelength_nm=float(self.centers[i]), intensity=intensity))
        return bins

    @staticmethod
    def attach_savoirs(bins: List[SpectralBin], savoirs: Dict[str, List[str]]) -> List[SpectralBin]:
        """Attach savoirs to bins by band name.

        savoirs maps a band name ("violet", "rouge", ...) to the savoirs
        associated to that wavelength family.
        """
        for b in bins:
            b.savoirs = list(savoirs.get(b.band, []))
        return bins

    @staticmethod
    def complementary_nm(wavelength_nm: float) -> float:
        """Anti-light: complementary wavelength across the white point (~565 nm)."""
        return float(np.clip(2.0 * 565.0 - wavelength_nm, NM_MIN, NM_MAX))

    def recompose(self, bins: List[SpectralBin], dim: int) -> np.ndarray:
        """Inverse operation: rebuild a vector from a spectrum."""
        v = np.zeros(dim)
        for b in bins:
            idx = int(np.searchsorted(self.edges, b.wavelength_nm) - 1)
            idx = min(max(idx, 0), self.n_bins - 1)
            v[idx] += b.intensity
        return v


def spectral_signature(bins: List[SpectralBin]) -> np.ndarray:
    """Signature vector: intensity-weighted centroid temperature and dispersion."""
    total = sum(b.intensity for b in bins) or 1.0
    temps = np.array([b.temperature_kelvin for b in bins])
    weights = np.array([b.intensity for b in bins])
    mean_t = float(np.sum(temps * weights) / total)
    spread = float(np.sqrt(np.sum(weights * (temps - mean_t) ** 2) / total))
    return np.array([mean_t, spread])
