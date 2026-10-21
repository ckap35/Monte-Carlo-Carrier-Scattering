import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from carrier_sim import ScatteringModel
from carrier_sim.physics import K_B_EV_PER_K
from carrier_sim.diagnostics import (absolute_histograms, accessible_equilibrium,
                                     continuum_equilibrium)


class DiagnosticTests(unittest.TestCase):
    def test_absolute_energy_alignment_preserves_valley_counts(self):
        density = np.zeros((2, 3, 10), dtype=int)
        density[:, 0, 0] = 2
        density[:, 1, 3] = 4
        density[:, 2, 1] = 6
        edges, shifted = absolute_histograms(density, np.arange(11) * 0.01, [0.03, 0.01, 0])
        assert_allclose(edges, np.arange(14) * 0.01)
        assert_array_equal(shifted.sum(axis=2), density.sum(axis=2))
        self.assertEqual(shifted[0, 0, 3], 2)
        self.assertEqual(shifted[0, 1, 4], 4)

    def test_non_aligned_histograms_are_not_silently_smeared(self):
        with self.assertRaises(ValueError):
            absolute_histograms(np.ones((2, 2, 10)), np.arange(11) * 0.01, [0, 0.015])

    def test_continuum_normalization_populations_and_mean(self):
        model = ScatteringModel()
        edges = np.linspace(0, 5, 501)
        reference = continuum_equilibrium(model, edges, edges)
        kt = K_B_EV_PER_K * model.temperature_K
        expected = np.exp(-np.array(model.valley_floors_eV) / kt)
        expected /= expected.sum()
        assert_allclose(reference.valley_probability, expected, rtol=1e-13)
        assert_allclose(reference.kinetic_probability.sum(), 1, atol=1e-13)
        assert_allclose(reference.total_probability.sum(), 1, atol=1e-13)
        self.assertAlmostEqual(reference.mean_kinetic_energy_eV, 1.5 * kt)

    def test_accessible_reference_matches_known_ladder_partition(self):
        model = ScatteringModel(
            valley_names=("a", "b"), valley_floors_eV=(0, 0.06),
            dos_mass_ratios=(1, 2), valley_degeneracies=(1, 2),
            temperature_K=300, coupling_ps_inv=np.ones((2, 2)),
        )
        edges = np.linspace(0, 3, 601)
        reference = accessible_equilibrium(model, [0.315], edges, edges)
        kt = K_B_EV_PER_K * 300
        ratio = 2 * 2 ** 1.5 * np.exp(-0.06 / kt)
        assert_allclose(reference.valley_probability, [1 / (1 + ratio), ratio / (1 + ratio)], rtol=1e-12)
        energies = 0.015 + np.arange(200) * 0.03
        weights = np.sqrt(energies) * np.exp(-energies / kt)
        expected = np.sum(weights * energies) / weights.sum()
        self.assertAlmostEqual(reference.mean_kinetic_energy_eV, expected, places=13)
        assert_allclose(reference.kinetic_categories.sum(), 1, atol=1e-13)

    def test_initial_energy_classes_are_normalized_individually(self):
        model = ScatteringModel(
            valley_names=("a",), valley_floors_eV=(0,), dos_mass_ratios=(1,),
            valley_degeneracies=(1,), coupling_ps_inv=[[1]], temperature_K=300,
        )
        edges = np.linspace(0, 3, 601)
        initial = [0.005, 0.025]
        reference = accessible_equilibrium(model, initial, edges, edges)
        kt = K_B_EV_PER_K * 300
        means = []
        for residue in initial:
            energy = residue + np.arange(200) * 0.03
            weights = np.sqrt(energy) * np.exp(-energy / kt)
            means.append(np.sum(weights * energy) / weights.sum())
        self.assertAlmostEqual(reference.mean_total_energy_eV, np.mean(means), places=13)

    def test_equilibrium_draws_preserve_probability_and_match_reference_mean(self):
        model = ScatteringModel()
        edges = np.linspace(0, 2, 501)
        initial = np.full(2000, 0.325)
        reference = accessible_equilibrium(model, initial, edges, edges, samples=8)
        assert_allclose(reference.sample_kinetic_categories.sum(axis=(1, 2)), 1)
        self.assertLess(abs(reference.sample_mean_total_energy_eV.mean()
                            - reference.mean_total_energy_eV), 0.004)

    def test_unconnected_model_does_not_claim_a_unique_ladder_equilibrium(self):
        model = ScatteringModel(coupling_ps_inv=np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            accessible_equilibrium(model, [0.325], np.linspace(0, 1, 101),
                                   np.linspace(0, 1, 101))


if __name__ == "__main__":
    unittest.main()
