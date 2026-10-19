"""Tree of Experts (TOE): recursive distillation of the datagraph.

The concentric datagraph is optimized/compressed into a TOE (tree of experts).
Each expert carries its own, more granular TOE: a nested tree where every
node is itself a tree of sub-experts. Recursion terminates when an expert's
payload cannot be distilled further without losing its signature.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Expert:
    """An expert of the TOE. Can itself hold a granular sub-TOE."""
    id: str
    payload: np.ndarray
    children: List["Expert"] = field(default_factory=list)

    def sub_toe(self) -> Optional["TreeOfExperts"]:
        return self._sub_toe

    _sub_toe: Optional["TreeOfExperts"] = None


class TreeOfExperts:
    """A TOE built by recursive k-division of a payload matrix.

    At each level, the payloads are split into k clusters (deterministic:
    projection on the leading principal direction, then split at the median).
    Each cluster centroid becomes an expert; each expert recursively owns a
    granular sub-TOE built on its own members, until fewer than k members
    remain or the payloads collapse (no variance).
    """

    def __init__(self, k: int = 3, min_members: int = 2, max_depth: int = 6):
        self.k = max(2, k)
        self.min_members = min_members
        self.max_depth = max_depth
        self.experts: List[Expert] = []

    def build(self, payloads: np.ndarray, ids: Optional[List[str]] = None,
              depth: int = 0) -> List[Expert]:
        X = np.asarray(payloads, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n = X.shape[0]
        if n == 0:
            return []
        ids = ids or [f"e{depth}_{i}" for i in range(n)]

        if n < self.min_members or depth >= self.max_depth or n < self.k:
            return [Expert(id=ids[i], payload=X[i]) for i in range(n)]

        center = X.mean(axis=0)
        centered = X - center
        # leading direction via power iteration on the covariance
        if centered.shape[1] == 0:
            axis = np.ones(1)
        else:
            cov = centered.T @ centered
            v = np.random.default_rng(0).normal(size=cov.shape[0])
            for _ in range(32):
                v = cov @ v
                norm = np.linalg.norm(v)
                if norm < 1e-12:
                    break
                v = v / norm
            axis = v / (np.linalg.norm(v) or 1.0)
        proj = centered @ axis
        order = np.argsort(proj, kind="stable")

        groups: List[np.ndarray] = []
        group_ids: List[List[str]] = []
        edges = np.array_split(order, self.k)
        for e in edges:
            if e.size == 0:
                continue
            groups.append(X[e])
            group_ids.append([ids[i] for i in e])

        experts: List[Expert] = []
        for gi, (gX, gids) in enumerate(zip(groups, group_ids)):
            centroid = gX.mean(axis=0)
            exp = Expert(id=f"{ids[0]}_c{gi}_d{depth}", payload=centroid)
            if gX.shape[0] >= self.min_members and not self._collapsed(gX):
                sub = TreeOfExperts(self.k, self.min_members, self.max_depth)
                sub.experts = sub.build(gX, gids, depth + 1)
                exp._sub_toe = sub
            else:
                exp.children = [Expert(id=gids[j], payload=gX[j])
                                for j in range(gX.shape[0])]
            experts.append(exp)
        self.experts = experts
        return experts

    @staticmethod
    def _collapsed(X: np.ndarray) -> bool:
        if X.shape[0] < 2:
            return True
        return float(np.linalg.norm(X - X.mean(axis=0))) < 1e-9

    def count(self) -> int:
        def cnt(exps: List[Expert]) -> int:
            total = 0
            for e in exps:
                total += 1
                if e._sub_toe is not None:
                    total += cnt(e._sub_toe.experts)
                total += len(e.children)
            return total
        return cnt(self.experts)

    def path_to(self, target: np.ndarray) -> List[str]:
        """Descend the TOE (and its sub-TOEs) to the closest expert path."""
        path: List[str] = []
        exps = self.experts
        t = np.asarray(target, dtype=float)
        while exps:
            best = min(exps, key=lambda e: np.linalg.norm(e.payload - t))
            path.append(best.id)
            if best._sub_toe is not None:
                exps = best._sub_toe.experts
            elif best.children:
                exps = best.children
            else:
                break
        return path
