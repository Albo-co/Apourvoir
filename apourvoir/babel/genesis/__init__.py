"""Genesis pipeline for the canonical space.

Enriches the Babel canonical space beyond its nodes, by decomposing its
first elements (light -> wavelengths, primal sounds -> rhythm/tonality),
placing them in triangular trees, distilling knowledge into a concentric
chronological datagraph (no quantization), and compressing it into a
recursive Tree of Experts (TOE), each expert carrying a more granular TOE.
The kernel (noyau) then illuminates roots and leaves to echo its "pour voir".
"""
from .light import LightDecomposer, SpectralBin, FRAUNHOFER_LINES, SPECTRAL_BANDS
from .sound import SoundDecomposer, SonorityDecomposition, tempo_couleur
from .triangle import TriangleNode, triangulate, barycentric_coordinates, barycenter
from .datagraph import ConcentricDatagraph, ChronoNode
from .toe import TreeOfExperts, Expert
from .noyau import Noyau, Echo
from .pipeline import genesis_pipeline

__all__ = [
    "LightDecomposer", "SpectralBin", "FRAUNHOFER_LINES", "SPECTRAL_BANDS",
    "SoundDecomposer", "SonorityDecomposition", "tempo_couleur",
    "TriangleNode", "triangulate", "barycentric_coordinates", "barycenter",
    "ConcentricDatagraph", "ChronoNode",
    "TreeOfExperts", "Expert",
    "Noyau", "Echo",
    "genesis_pipeline",
]
