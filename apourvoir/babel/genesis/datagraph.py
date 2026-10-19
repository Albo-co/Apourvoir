"""Chronological concentric datagraph: distillation, not quantization.

Knowledge is distilled into concentric layers ordered chronologically:
- the center (layer 0) is the most distilled, oldest essence
- outer layers keep increasingly finer chronological detail

Unlike quantization (lossy projection onto a fixed grid), distillation
preserves the *relative* structure while compressing the number of nodes:
each layer keeps the same topology, only the resolution changes.
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ChronoNode:
    """A node of the concentric datagraph."""
    id: str
    timestamp: float
    payload: np.ndarray
    layer: int
    children: List[str] = field(default_factory=list)
    parent: Optional[str] = None


class ConcentricDatagraph:
    """Datagraph with concentric chronological layers.

    Insertion assigns each node to a layer based on a chronological decay:
    recent knowledge sits on outer layers, distilled knowledge migrates
    toward the center as it survives successive compressions.
    """

    def __init__(self, n_layers: int = 5, halflife: float = 3600.0):
        self.n_layers = n_layers
        self.halflife = halflife
        self.nodes: Dict[str, ChronoNode] = {}
        self._counter = 0

    def _layer_for(self, timestamp: float, now: Optional[float] = None) -> int:
        now = now if now is not None else max(
            (n.timestamp for n in self.nodes.values()), default=timestamp)
        age = max(0.0, now - timestamp)
        decay = 1.0 - 2.0 ** (-age / self.halflife)
        return int(decay * (self.n_layers - 1))

    def add(self, payload: np.ndarray, timestamp: Optional[float] = None,
            node_id: Optional[str] = None, parent: Optional[str] = None) -> ChronoNode:
        ts = timestamp if timestamp is not None else time.time()
        self._counter += 1
        nid = node_id or f"n{self._counter}"
        layer = self._layer_for(ts)
        node = ChronoNode(id=nid, timestamp=ts,
                          payload=np.asarray(payload, dtype=float),
                          layer=layer, parent=parent)
        self.nodes[nid] = node
        if parent and parent in self.nodes:
            self.nodes[parent].children.append(nid)
        return node

    def distill(self, target_factor: float = 0.5, now: Optional[float] = None) -> Dict[int, int]:
        """Compress the datagraph toward the center.

        Nodes of the same layer close to each other are merged into a single
        distilled node on the layer just below (one step toward the center).
        Payloads are averaged, earliest timestamp kept. Returns the mapping
        layer -> number of merged nodes.
        """
        if not self.nodes:
            return {}
        now = now if now is not None else max(n.timestamp for n in self.nodes.values())
        merged: Dict[int, int] = {}
        by_layer: Dict[int, List[ChronoNode]] = {}
        for n in self.nodes.values():
            by_layer.setdefault(n.layer, []).append(n)

        for layer, nodes in by_layer.items():
            if layer == 0 or len(nodes) < 2:
                continue
            payloads = np.stack([n.payload for n in nodes])
            norms = np.linalg.norm(payloads, axis=1)
            order = np.argsort(norms)
            n_keep = max(1, int(len(nodes) * target_factor))
            keep_idx = set(order[:n_keep].tolist())
            new_layer = layer - 1
            for i, node in enumerate(nodes):
                if i in keep_idx:
                    node.layer = new_layer
                else:
                    donor = nodes[min(keep_idx)]
                    donor.payload = 0.5 * (donor.payload + node.payload)
                    donor.timestamp = min(donor.timestamp, node.timestamp)
                    self._absorb_children(node, donor)
                    self.nodes.pop(node.id)
            merged[layer] = len(nodes) - len(keep_idx)
        return merged

    def _absorb_children(self, node: ChronoNode, donor: ChronoNode) -> None:
        for cid in node.children:
            child = self.nodes.get(cid)
            if child is not None:
                child.parent = donor.id
                donor.children.append(cid)

    def layers(self) -> Dict[int, List[ChronoNode]]:
        out: Dict[int, List[ChronoNode]] = {}
        for n in self.nodes.values():
            out.setdefault(n.layer, []).append(n)
        return out

    def radial_signature(self) -> np.ndarray:
        """Concentric signature: node mass per layer, from center outward."""
        sig = np.zeros(self.n_layers)
        for n in self.nodes.values():
            sig[n.layer] += float(np.linalg.norm(n.payload))
        total = sig.sum() or 1.0
        return sig / total
