"""Tests for the genesis module: light/sound decomposition, triangularisation,
concentric datagraph distillation, recursive TOE, and kernel echo."""
import numpy as np
import pytest

from apourvoir.babel.genesis import (
    LightDecomposer, SoundDecomposer, triangulate, barycentric_coordinates,
    ConcentricDatagraph, TreeOfExperts, Noyau, genesis_pipeline,
)


class TestLightDecomposer:
    def test_decompose_covers_spectrum(self):
        d = LightDecomposer(n_bins=16)
        bins = d.decompose(np.random.default_rng(0).normal(size=32))
        assert len(bins) == 16
        assert all(380.0 <= b.wavelength_nm <= 780.0 for b in bins)
        assert bins[0].wavelength_nm < bins[-1].wavelength_nm

    def test_bands_and_savoirs(self):
        d = LightDecomposer(n_bins=7)
        bins = d.decompose(np.ones(7))
        bands = {b.band for b in bins}
        assert "violet" in bands and "rouge" in bands
        d.attach_savoirs(bins, {"violet": ["savoir-violet"]})
        assert bins[0].savoirs == ["savoir-violet"]

    def test_roundtrip(self):
        d = LightDecomposer(n_bins=8)
        bins = d.decompose(np.arange(8, dtype=float))
        rec = d.recompose(bins, dim=8)
        assert rec.shape == (8,)
        assert rec.sum() > 0

    def test_complementary(self):
        assert 380.0 <= LightDecomposer.complementary_nm(400.0) <= 780.0


class TestSoundDecomposer:
    def test_parallel_axes(self):
        s = SoundDecomposer().decompose(np.arange(10, dtype=float))
        assert len(s.rhythm) >= 2
        assert len(s.tonality) >= 5
        levels = [r.level for r in s.rhythm]
        assert levels == ["pulsation", "temps", "mesure", "phrase", "cycle"][:len(levels)]

    def test_tempo_couleur_parallel(self):
        fast = SoundDecomposer.__dict__  # sanity: module importable
        from apourvoir.babel.genesis.sound import tempo_couleur
        assert tempo_couleur(240.0) < tempo_couleur(60.0)


class TestTriangle:
    def test_barycentric_on_vertices(self):
        from apourvoir.babel.genesis.triangle import LIGHT_POLE, RHYTHM_POLE
        l, r, t = barycentric_coordinates(LIGHT_POLE)
        assert l == pytest.approx(1.0)
        assert r == pytest.approx(0.0, abs=1e-9)
        l2, r2, t2 = barycentric_coordinates(RHYTHM_POLE)
        assert r2 == pytest.approx(1.0)

    def test_triangulate_groups(self):
        items = [{"name": f"n{i}", "light": i, "rhythm": 1.0, "tonality": 0.5}
                 for i in range(9)]
        tree = triangulate(items)
        assert tree.depth() >= 2
        assert len(tree.leaves()) == 9


class TestConcentricDatagraph:
    def test_layers_and_distill(self):
        dg = ConcentricDatagraph(n_layers=4, halflife=1.0)
        for i in range(8):
            dg.add(np.array([float(i), 1.0]), timestamp=float(i))
        before = len(dg.nodes)
        merged = dg.distill(target_factor=0.5)
        after = len(dg.nodes)
        assert after <= before
        assert sum(merged.values()) == before - after
        sig = dg.radial_signature()
        assert np.isclose(sig.sum(), 1.0)

    def test_parent_child(self):
        dg = ConcentricDatagraph(n_layers=3)
        a = dg.add(np.ones(2), node_id="a")
        b = dg.add(np.ones(2), node_id="b", parent="a")
        assert "b" in dg.nodes["a"].children


class TestToe:
    def test_recursive_sub_toe(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(27, 4))
        toe = TreeOfExperts(k=3)
        toe.build(X)
        assert len(toe.experts) == 3
        subs = [e for e in toe.experts if e._sub_toe is not None]
        assert len(subs) >= 1
        assert all(e._sub_toe.k == 3 for e in subs)

    def test_path_to(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(9, 3))
        toe = TreeOfExperts(k=3)
        toe.build(X)
        path = toe.path_to(X[0])
        assert len(path) >= 1


class TestNoyau:
    def test_echo(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(12, 4))
        res = genesis_pipeline(X)
        assert -1.0 <= res.echo.resonance <= 1.0
        s = res.summary()
        assert s["n_bins"] == 64
        assert s["n_experts"] >= 1
        assert s["n_leaves"] == 12
