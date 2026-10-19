"""Noyau (kernel): shed light on roots and leaves to echo its "pour voir".

The kernel traverses a TOE (roots) and its datagraph (leaves), projects both
poles onto the canonical space (pou / voir), and produces an "echo": a compact
signature of what the space can "pour voir" (provide-for-seeing).
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .datagraph import ConcentricDatagraph
from .toe import TreeOfExperts


@dataclass
class Echo:
    """The kernel's echo: roots illuminated, leaves illuminated, resonance."""
    roots_signature: np.ndarray
    leaves_signature: np.ndarray
    resonance: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "roots": self.roots_signature.tolist(),
            "leaves": self.leaves_signature.tolist(),
            "resonance": self.resonance,
        }


class Noyau:
    """Kernel illuminating the roots (TOE centers) and leaves (datagraph nodes)."""

    def __init__(self, dim: Optional[int] = None):
        self.dim = dim

    @staticmethod
    def _canonical(p: np.ndarray) -> np.ndarray:
        """Project on the canonical axes: x=pou (possibility), y=voir (perception)."""
        v = np.asarray(p, dtype=float).ravel()
        pou = float(np.linalg.norm(v))
        voir = float(np.mean(v)) if v.size else 0.0
        return np.array([pou, voir])

    def illuminate_roots(self, toe: TreeOfExperts) -> np.ndarray:
        """Shed light on the roots: centroids of the top-level experts."""
        if not toe.experts:
            return np.zeros(2)
        roots = np.stack([self._canonical(e.payload) for e in toe.experts])
        return roots.mean(axis=0)

    def illuminate_leaves(self, datagraph: ConcentricDatagraph) -> np.ndarray:
        """Shed light on the leaves: outermost layer of the concentric datagraph."""
        layers = datagraph.layers()
        if not layers:
            return np.zeros(2)
        outer = max(layers.keys())
        nodes = layers[outer] or datagraph.nodes[list(datagraph.nodes)[0]]
        leaves = np.stack([self._canonical(n.payload) for n in nodes])
        return leaves.mean(axis=0)

    def echo(self, toe: TreeOfExperts, datagraph: ConcentricDatagraph) -> Echo:
        """Resonance between roots and leaves: cosine of their canonical poles."""
        roots = self.illuminate_roots(toe)
        leaves = self.illuminate_leaves(datagraph)
        denom = (np.linalg.norm(roots) * np.linalg.norm(leaves))
        resonance = float(roots @ leaves / denom) if denom > 1e-12 else 0.0
        return Echo(roots_signature=roots, leaves_signature=leaves,
                    resonance=resonance)
