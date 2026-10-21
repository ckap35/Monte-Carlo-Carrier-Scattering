import json
from dataclasses import replace
from math import gamma
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from carrier_sim import OpticalExcitation, SimulationConfig, run_simulation
from carrier_sim.excitation import EV_IN_J, HBAR_J_S, ELECTRON_MASS_KG


class OpticalExcitationTests(unittest.TestCase):
    def test_jdos_integrates_to_k_space_state_count(self):
        optical = OpticalExcitation(band_gap_eV=1, photon_center_eV=2,
                                    electron_mass_ratio=0.2, hole_mass_ratio=0.8)
        # Count all states in a k-space sphere, with two spins, independently
        # of the density per photon energy. Integrate using q=u^2.
        nodes, weights = np.polynomial.legendre.leggauss(32)
        q_max = 0.7
        u = np.sqrt(q_max) * (nodes + 1) / 2
        integrated = np.sum(weights * np.sqrt(q_max) * u
                            * optical.joint_dos(optical.band_gap_eV + u ** 2))
        k_max = np.sqrt(2 * 0.16 * ELECTRON_MASS_KG * q_max * EV_IN_J) / HBAR_J_S
        assert_allclose(integrated, k_max ** 3 / (3 * np.pi ** 2) / 1e6, rtol=2e-14)
        assert_array_equal(optical.joint_dos([0.5, 1.0]), [0, 0])

    def test_photon_energy_and_same_k_energy_partition(self):
        optical = OpticalExcitation(1, 2, electron_mass_ratio=0.2, hole_mass_ratio=0.8)
        pairs = optical.sample_pairs(2000, np.random.default_rng(2))
        assert_allclose(pairs.photon_energies_eV,
                        1 + pairs.electron_kinetic_energies_eV + pairs.hole_kinetic_energies_eV)
        # Equal k implies m_e*epsilon_e = m_h*epsilon_h.
        assert_allclose(0.2 * pairs.electron_kinetic_energies_eV,
                        0.8 * pairs.hole_kinetic_energies_eV, rtol=1e-14)
        self.assertTrue(np.all(pairs.photon_energies_eV > 1))

    def test_gaussian_intensity_fwhm(self):
        optical = OpticalExcitation(1, 2)
        assert_allclose(optical.spectrum([1.5, 2, 2.5]), [0.5, 1, 0.5], atol=1e-15)

    def test_threshold_pump_matches_analytic_moments(self):
        optical = OpticalExcitation(1, 1)
        q = optical.sample_pairs(80_000, np.random.default_rng(3)).photon_energies_eV - 1
        mean = np.sqrt(2) * optical.sigma_eV * gamma(1.25) / gamma(0.75)
        second = 2 * optical.sigma_eV ** 2 * gamma(1.75) / gamma(0.75)
        self.assertLess(abs(q.mean() - mean), 5 * np.sqrt((second - mean ** 2) / len(q)))

    def test_samples_and_pdf_against_independent_quadrature(self):
        # Test both rejection proposals and a far-subgap tail. The reference
        # uses JDOS times spectrum, rather than the sampler's envelope.
        nodes, weights = np.polynomial.legendre.leggauss(300)
        for center in (0.1, 1, 2, 4):
            with self.subTest(center=center):
                optical = OpticalExcitation(1, center)
                maximum = max(center - 1, 0) + 10 * optical.sigma_eV
                u = np.sqrt(maximum) * (nodes + 1) / 2
                q = u ** 2
                dq = weights * np.sqrt(maximum) * u
                photon = 1 + q
                density = optical.joint_dos(photon) * optical.spectrum(photon)
                norm = np.sum(dq * density)
                mean = np.sum(dq * q * density) / norm
                variance = np.sum(dq * (q - mean) ** 2 * density) / norm
                # Subtraction at the gap rounds tiny q values; compare the PDF
                # at the same representable excess energy as the JDOS input.
                assert_allclose(optical.excess_pdf(photon - 1), density / norm, rtol=2e-11)
                assert_allclose(np.sum(dq * optical.excess_pdf(q)), 1, atol=2e-12)
                fraction = optical.electron_energy_fraction
                assert_allclose(np.sum(dq * fraction * optical.electron_pdf(q * fraction)), 1,
                                atol=2e-12)
                draws = optical.sample_pairs(40_000, np.random.default_rng(4))
                excess = draws.photon_energies_eV - 1
                self.assertLess(abs(excess.mean() - mean), 5 * np.sqrt(variance / len(excess)))
                for threshold in (0.2, 0.6, 1.2):
                    z = np.sqrt(threshold) * (nodes + 1) / 2
                    density_z = optical.joint_dos(1 + z ** 2) * optical.spectrum(1 + z ** 2)
                    probability = np.sum(weights * np.sqrt(threshold) * z * density_z) / norm
                    observed = np.mean(excess < threshold)
                    standard_error = np.sqrt(probability * (1 - probability) / len(excess))
                    self.assertLess(abs(observed - probability), 5 * standard_error + 1 / len(excess))

    def test_optical_simulation_ledger_histograms_and_reproducibility(self):
        config = SimulationConfig(num_carriers=500, duration_ps=30,
                                  optical_excitation=OpticalExcitation(1, 2))
        first = run_simulation(config, seed=5)
        second = run_simulation(replace(config, num_energy_bins=10,
                                        histogram_max_kinetic_energy_eV=0.05), seed=5)
        assert_array_equal(first.final_valleys, second.final_valleys)
        assert_array_equal(first.final_kinetic_energies_eV, second.final_kinetic_energies_eV)
        pairs = first.initial_optical_pairs
        assert_allclose(first.initial_total_energies_eV,
                        pairs.electron_kinetic_energies_eV + config.model.valley_floors_eV[0])
        assert_allclose(first.carrier_energy_balance_errors_eV, 0, atol=1e-13)
        assert_array_equal(first.valley_population_history.sum(axis=1), config.num_carriers)
        assert_array_equal(second.density_history.sum(axis=2) + second.overflow_history,
                           second.valley_population_history)
        restored = json.loads(json.dumps(config.to_dict()))
        restored.pop("model")
        self.assertEqual(SimulationConfig(**restored).optical_excitation, config.optical_excitation)

    def test_rejects_inconsistent_optical_setup(self):
        for settings in ({"band_gap_eV": -1, "photon_center_eV": 2},
                         {"band_gap_eV": 1, "photon_center_eV": 2, "spectral_fwhm_eV": 0},
                         {"band_gap_eV": 1, "photon_center_eV": 2, "hole_mass_ratio": np.nan}):
            with self.assertRaises(ValueError):
                OpticalExcitation(**settings)
        with self.assertRaises(ValueError):
            SimulationConfig(optical_excitation=OpticalExcitation(1, 2, electron_mass_ratio=0.2))
        config = SimulationConfig(num_carriers=1, optical_excitation=OpticalExcitation(1, 2))
        with self.assertRaises(ValueError):
            run_simulation(config, initial_kinetic_energies_eV=[0.1])
        with self.assertRaises(ValueError):
            run_simulation(config, initial_valleys=[1])


if __name__ == "__main__":
    unittest.main()
