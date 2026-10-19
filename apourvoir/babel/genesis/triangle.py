"""Geometric triangularisation in trees.

Each element of knowledge is placed in a triangle (light / rhythm / tonality),
and each triangle becomes a node of an arborescence. The parallel placement
respects the triangular geometry: the three poles are the vertices, the
arborescence grows from their barycenter.
"""
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Canonical triangle vertices (2D barycentric frame)
LIGHT_POLE = np.array([1.0, 0.0])
RHYTHM_POLE = np.array([0.0, 1.0])
TONALITY_POLE = np.array([0.0, 0.0])


def barycenter(*points: np.ndarray) -> np.ndarray:
    pts = [np.asarray(p, dtype=float) for p in points]
    return np.mean(pts, axis=0)


def barycentric_coordinates(point: np.ndarray) -> Tuple[float, float, float]:
    """Return (light, rhythm, tonality) barycentric weights of a 2D point."""
    p = np.asarray(point, dtype=float)
    a, b, c = LIGHT_POLE, RHYTHM_POLE, TONALITY_POLE
    v0, v1, v2 = b - c, a - c, p - c
    d00, d01 = v0 @ v0, v0 @ v1
    d11, d20, d21 = v1 @ v1, v2 @ v0, v2 @ v1
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return (1 / 3, 1 / 3, 1 / 3)
    u = (d00 * d21 - d01 * d20) / denom  # weight of LIGHT_POLE
    v = (d11 * d20 - d01 * d21) / denom  # weight of RHYTHM_POLE
    w = 1.0 - u - v
    return (float(u), float(v), float(w))


@dataclass
class TriangleNode:
    """A triangle placed in the arborescence."""
    name: str
    light: float = 0.0
    rhythm: float = 0.0
    tonality: float = 0.0
    children: List["TriangleNode"] = field(default_factory=list)

    def point(self) -> np.ndarray:
        return (self.light * LIGHT_POLE
                + self.rhythm * RHYTHM_POLE
                + self.tonality * TONALITY_POLE)

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def leaves(self) -> List["TriangleNode"]:
        if not self.children:
            return [self]
        out: List[TriangleNode] = []
        for c in self.children:
            out.extend(c.leaves())
        return out


def triangulate(items: List[Dict[str, float]], max_children: int = 3) -> TriangleNode:
    """Build a triangle tree from weighted items.

    Items are sorted by their barycenter norm, then grouped in triples:
    each triple forms a triangle whose parent is their barycenter, giving a
    geometric arborescence (each internal node has up to max_children children).
    """
    if not items:
        return TriangleNode(name="racine")

    def key(it: Dict[str, float]) -> float:
        p = (it.get("light", 0.0) * LIGHT_POLE
             + it.get("rhythm", 0.0) * RHYTHM_POLE)
        return float(np.linalg.norm(p))

    ordered = sorted(items, key=key)

    def build(level_items: List[Dict[str, float]], depth: int) -> List[TriangleNode]:
        nodes: List[TriangleNode] = []
        for i, it in enumerate(level_items):
            nodes.append(TriangleNode(
                name=str(it.get("name", f"nœud_{depth}_{i}")),
                light=float(it.get("light", 0.0)),
                rhythm=float(it.get("rhythm", 0.0)),
                tonality=float(it.get("tonality", 0.0)),
            ))
        while len(nodes) > 1:
            parents: List[TriangleNode] = []
            for i in range(0, len(nodes), max_children):
                group = nodes[i:i + max_children]
                if len(group) == 1:
                    parents.append(group[0])
                    continue
                bc = barycenter(*[g.point() for g in group])
                w = barycentric_coordinates(bc)
                parent = TriangleNode(
                    name=f"triangle_d{depth}_{len(parents)}",
                    light=w[0], rhythm=w[1], tonality=w[2],
                    children=group,
                )
                parents.append(parent)
            nodes = parents
        return nodes

    tree = build(ordered, 0)
    return tree[0] if tree else TriangleNode(name="racine")
