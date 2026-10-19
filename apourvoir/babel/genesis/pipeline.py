"""End-to-end genesis pipeline: vector -> spectrum/sound -> triangle ->
concentric datagraph -> TOE -> kernel echo."""
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .light import LightDecomposer, SpectralBin
from .sound import SoundDecomposer, SonorityDecomposition
from .triangle import TriangleNode, triangulate
from .datagraph import ConcentricDatagraph
from .toe import TreeOfExperts
from .noyau import Noyau, Echo


@dataclass
class GenesisResult:
    spectrum: List[SpectralBin]
    sonority: SonorityDecomposition
    tree: TriangleNode
    datagraph: ConcentricDatagraph
    toe: TreeOfExperts
    echo: Echo

    def summary(self) -> Dict[str, object]:
        return {
            "n_bins": len(self.spectrum),
            "n_rhythm_levels": len(self.sonority.rhythm),
            "n_tonal_partials": len(self.sonority.tonality),
            "tree_depth": self.tree.depth(),
            "n_leaves": len(self.tree.leaves()),
            "n_layers": len(self.datagraph.layers()),
            "n_experts": self.toe.count(),
            "resonance": self.echo.resonance,
        }


def genesis_pipeline(vectors: np.ndarray, timestamps: Optional[List[float]] = None,
                     n_bins: int = 64, n_layers: int = 5,
                     k: int = 3) -> GenesisResult:
    """Run the full genesis decomposition over a batch of knowledge vectors."""
    X = np.asarray(vectors, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    light = LightDecomposer(n_bins=n_bins)
    sound = SoundDecomposer()
    dg = ConcentricDatagraph(n_layers=n_layers)

    for i, vec in enumerate(X):
        bins = light.decompose(vec)
        soro = sound.decompose(vec)
        ts = timestamps[i] if timestamps else float(i)
        dg.add(vec, timestamp=ts, node_id=f"v{i}")

    spectrum = light.decompose(X.mean(axis=0))
    sonority = sound.decompose(X.mean(axis=0))

    items = []
    for i, vec in enumerate(X):
        w = sonority.rhythm[0].poids if sonority.rhythm else 0.0
        items.append({"name": f"v{i}", "light": float(np.linalg.norm(vec)),
                      "rhythm": w, "tonality": float(np.mean(vec))})
    tree = triangulate(items)

    toe = TreeOfExperts(k=k)
    toe.build(X)

    noyau = Noyau()
    echo = noyau.echo(toe, dg)

    return GenesisResult(spectrum=spectrum, sonority=sonority, tree=tree,
                         datagraph=dg, toe=toe, echo=echo)
