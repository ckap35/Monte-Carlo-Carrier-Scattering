import unittest

import numpy as np
from numpy.testing import assert_allclose

from carrier_sim.physics import (ScatteringModel, K_B_EV_PER_K,
                                 EMISSION, ABSORPTION)


class PhysicsTests(unittest.TestCase):
    def test_bose_occupation_uses_phonon_energy(self):
        model = ScatteringModel()
        self.assertAlmostEqual(model.phonon_occupation, 1.2715496621843811, places=13)
        changed_floors = ScatteringModel(valley_floors_eV=(0.8, 0.2, 0))
        self.assertEqual(model.phonon_occupation, changed_floors.phonon_occupation)
        self.assertEqual(ScatteringModel(temperature_K=0).phonon_occupation, 0)

    def test_every_channel_obeys_energy_balance(self):
        model = ScatteringModel()
        energies = np.array([0.0301204819277108, 0.4, 0.8])
        valleys = np.arange(3)
        final = model.destination_energies(valleys, energies)
        for source in range(3):
            initial_total = energies[source] + model.valley_floors_eV[source]
            for target in range(3):
                for kind, sign in ((EMISSION, -1), (ABSORPTION, 1)):
                    change = final[source, target, kind] + model.valley_floors_eV[target] - initial_total
                    self.assertAlmostEqual(change, sign * 0.03, places=14)
        self.assertAlmostEqual(final[0, 2, EMISSION], 0.3001204819277108)

    def test_forbidden_thresholds_have_zero_rates(self):
        model = ScatteringModel()
        assert_allclose(model.channel_rates([2], [0])[0, 0], 0)
        self.assertEqual(model.channel_rates([2], [0.26])[0, 0, ABSORPTION], 0)
        self.assertGreater(model.channel_rates([2], [0.271])[0, 0, ABSORPTION], 0)
        self.assertEqual(model.channel_rates([2], [0.001])[0, 2, EMISSION], 0)

    def test_zero_temperature_forbids_absorption(self):
        rates = ScatteringModel(temperature_K=0).channel_rates([0, 1, 2], [0.5, 0.5, 0.5])
        assert_allclose(rates[:, :, ABSORPTION], 0)
        self.assertTrue(np.any(rates[:, :, EMISSION] > 0))

    def test_pairwise_detailed_balance_with_mass_and_degeneracy(self):
        model = ScatteringModel(
            dos_mass_ratios=(0.5, 1, 2), valley_degeneracies=(1, 3, 4),
            coupling_ps_inv=[[0.03, 0.02, 0.07],
                             [0.02, 0.04, 0.01], [0.07, 0.01, 0.05]],
        )
        kt = K_B_EV_PER_K * model.temperature_K
        def boltzmann_weight(valley, kinetic):
            dos = (model.valley_degeneracies[valley]
                   * model.dos_mass_ratios[valley] ** 1.5 * np.sqrt(kinetic))
            return dos * np.exp(-(kinetic + model.valley_floors_eV[valley]) / kt)
        for source in range(3):
            kinetic = 0.5
            rates = model.channel_rates([source], [kinetic])[0]
            final = model.destination_energies([source], [kinetic])[0]
            for target in range(3):
                for kind in (EMISSION, ABSORPTION):
                    energy = final[target, kind]
                    reverse = model.channel_rates([target], [energy])[0, source, 1 - kind]
                    left = boltzmann_weight(source, kinetic) * rates[target, kind]
                    right = boltzmann_weight(target, energy) * reverse
                    assert_allclose(left, right, rtol=1e-12, atol=0)

    def test_couplings_are_copied_and_readonly(self):
        coupling = np.ones((3, 3))
        model = ScatteringModel(coupling_ps_inv=coupling)
        coupling[:] = 0
        assert_allclose(model.coupling_ps_inv, 1)
        with self.assertRaises(ValueError):
            model.coupling_ps_inv[0, 0] = 0

    def test_invalid_parameters_and_states_are_rejected(self):
        for settings in (
            {"temperature_K": -1}, {"phonon_energy_eV": 0},
            {"dos_mass_ratios": (1, 0, 1)}, {"valley_degeneracies": (1, 1.2, 1)},
            {"coupling_ps_inv": [[0, 1, 0], [0, 0, 0], [0, 0, 0]]},
            {"coupling_ps_inv": np.full((3, 3), np.nan)},
        ):
            with self.subTest(settings=list(settings)), self.assertRaises(ValueError):
                ScatteringModel(**settings)
        model = ScatteringModel()
        for valleys, energies in (([0.5], [0.1]), ([3], [0.1]), ([0], [-0.1]),
                                  ([0], [np.nan]), ([0, 1], [0.1])):
            with self.assertRaises(ValueError):
                model.channel_rates(valleys, energies)


if __name__ == "__main__":
    unittest.main()
