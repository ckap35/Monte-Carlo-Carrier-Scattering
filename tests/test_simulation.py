import unittest
from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from carrier_sim import ScatteringModel, SimulationConfig, run_simulation
from carrier_sim.physics import K_B_EV_PER_K, EMISSION, ABSORPTION


class SimulationTests(unittest.TestCase):
    def test_original_gamma_to_l_bug_is_fixed_in_actual_events(self):
        model = ScatteringModel(temperature_K=0,
                                coupling_ps_inv=[[0, 0, 10], [0, 0, 0], [10, 0, 0]])
        config = SimulationConfig(model=model, num_carriers=1000, duration_ps=10,
                                  initial_kinetic_range_eV=(0.0301204819277108,) * 2)
        result = run_simulation(config, seed=1)
        assert_array_equal(result.final_valleys, np.full(1000, 2))
        assert_allclose(result.final_kinetic_energies_eV, 0.3001204819277108, atol=1e-15)
        self.assertEqual(result.event_counts[0, 2, EMISSION], 1000)
        self.assertEqual(result.event_counts.sum(), 1000)
        assert_allclose(result.carrier_energy_balance_errors_eV, 0, atol=1e-14)

    def test_impossible_uphill_jump_never_occurs(self):
        model = ScatteringModel(coupling_ps_inv=[[0, 0, 10], [0, 0, 0], [10, 0, 0]])
        config = SimulationConfig(model=model, num_carriers=200, duration_ps=1000,
                                  sample_interval_ps=1000, initial_valley=2,
                                  initial_kinetic_range_eV=(0, 0))
        result = run_simulation(config, seed=2)
        self.assertEqual(result.event_counts.sum(), 0)
        assert_allclose(result.final_kinetic_energies_eV, 0)
        assert_array_equal(result.final_valleys, np.full(200, 2))

    def test_population_histograms_flux_and_energy_conservation(self):
        config = SimulationConfig(
            model=ScatteringModel(coupling_ps_inv=np.ones((3, 3))),
            num_carriers=500, duration_ps=50, sample_interval_ps=0.7,
        )
        result = run_simulation(config, seed=3)
        assert_array_equal(result.valley_population_history.sum(axis=1), 500)
        assert_array_equal(result.density_history.sum(axis=2) + result.overflow_history,
                           result.valley_population_history)
        transfers = result.event_counts.sum(axis=2)
        flux_change = transfers.sum(axis=0) - transfers.sum(axis=1)
        assert_array_equal(result.valley_population_history[-1]
                           - result.valley_population_history[0], flux_change)
        assert_allclose(result.energy_balance_error_eV, 0, atol=1e-10)
        assert_allclose(result.carrier_energy_balance_errors_eV, 0, atol=1e-12)
        self.assertGreater(result.event_counts[:, :, EMISSION].sum(), 0)
        self.assertGreater(result.event_counts[:, :, ABSORPTION].sum(), 0)
        self.assertTrue(np.all(result.final_kinetic_energies_eV >= 0))

    def test_histogram_limits_do_not_clip_carrier_energies(self):
        model = ScatteringModel(coupling_ps_inv=np.zeros((3, 3)))
        config = SimulationConfig(model=model, num_carriers=2, duration_ps=0)
        result = run_simulation(config, initial_valleys=[0, 1],
                                initial_kinetic_energies_eV=[0.9, 0.1])
        assert_allclose(result.final_kinetic_energies_eV, [0.9, 0.1])
        assert_array_equal(result.overflow_history[0], [1, 0, 0])
        self.assertEqual(result.density_history.sum(), 1)

    def test_histogram_binning_does_not_change_trajectories(self):
        config = SimulationConfig(model=ScatteringModel(coupling_ps_inv=np.ones((3, 3))),
                                  num_carriers=200, duration_ps=5)
        first = run_simulation(config, seed=4)
        second = run_simulation(replace(config, num_energy_bins=3,
                                       histogram_max_kinetic_energy_eV=0.01), seed=4)
        assert_array_equal(first.final_valleys, second.final_valleys)
        assert_array_equal(first.final_kinetic_energies_eV, second.final_kinetic_energies_eV)
        assert_array_equal(first.event_counts, second.event_counts)

    def test_reproducibility_and_no_global_rng_changes(self):
        config = SimulationConfig(num_carriers=200, duration_ps=5)
        before = np.random.get_state()
        first = run_simulation(config, seed=5)
        second = run_simulation(config, seed=5)
        after = np.random.get_state()
        assert_array_equal(first.final_kinetic_energies_eV, second.final_kinetic_energies_eV)
        assert_array_equal(first.density_history, second.density_history)
        assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_exponential_survival_probability(self):
        coupling = 1 / np.sqrt(0.005)  # upper -> lower emission rate = 1 / ps
        model = ScatteringModel(
            valley_names=("upper", "lower"), valley_floors_eV=(0.03, 0),
            dos_mass_ratios=(1, 1), valley_degeneracies=(1, 1), temperature_K=0,
            coupling_ps_inv=[[0, coupling], [coupling, 0]],
        )
        config = SimulationConfig(model=model, num_carriers=20_000, duration_ps=1,
                                  initial_kinetic_range_eV=(0.005, 0.005))
        result = run_simulation(config, seed=6)
        probability = np.exp(-1)
        expected = config.num_carriers * probability
        sigma = np.sqrt(config.num_carriers * probability * (1 - probability))
        self.assertLess(abs(result.valley_population_history[-1, 0] - expected), 5 * sigma)

    def test_competing_channels_follow_rate_ratios(self):
        model = ScatteringModel(
            valley_floors_eV=(0, 0, 0), temperature_K=0,
            coupling_ps_inv=[[0, 10, 30], [10, 0, 0], [30, 0, 0]],
        )
        config = SimulationConfig(model=model, num_carriers=20_000, duration_ps=10,
                                  sample_interval_ps=10, initial_kinetic_range_eV=(0.04, 0.04))
        result = run_simulation(config, seed=7)
        self.assertEqual(result.valley_population_history[-1, 0], 0)
        probability = 0.25
        sigma = np.sqrt(config.num_carriers * probability * (1 - probability))
        self.assertLess(abs(result.valley_population_history[-1, 1]
                            - config.num_carriers * probability), 5 * sigma)

    def test_snapshot_spacing_does_not_change_single_carrier_trajectory(self):
        config = SimulationConfig(model=ScatteringModel(coupling_ps_inv=np.full((3, 3), 20.0)),
                                  num_carriers=1, duration_ps=10, sample_interval_ps=0.1)
        first = run_simulation(config, seed=8)
        second = run_simulation(replace(config, sample_interval_ps=7), seed=8)
        assert_array_equal(first.final_valleys, second.final_valleys)
        assert_array_equal(first.final_kinetic_energies_eV, second.final_kinetic_energies_eV)
        assert_array_equal(first.event_counts, second.event_counts)
        assert_allclose(second.times_ps, [0, 7, 10])

    def test_equilibrium_on_accessible_energy_ladder(self):
        # Floors differ by two phonons, so both valleys share the same kinetic
        # energy ladder: 0.015 + n*0.03 eV. Its partition sums are analytic.
        model = ScatteringModel(
            valley_names=("a", "b"), valley_floors_eV=(0, 0.06),
            dos_mass_ratios=(1, 2), valley_degeneracies=(1, 2), temperature_K=300,
            coupling_ps_inv=np.full((2, 2), 10.0),
        )
        config = SimulationConfig(model=model, num_carriers=5000, duration_ps=30,
                                  sample_interval_ps=30, initial_kinetic_range_eV=(0.315, 0.315))
        result = run_simulation(config, seed=9)
        kt = K_B_EV_PER_K * 300
        relative_weight = 2 * 2 ** 1.5 * np.exp(-0.06 / kt)
        probability = relative_weight / (1 + relative_weight)
        observed = np.mean(result.final_valleys == 1)
        standard_error = np.sqrt(probability * (1 - probability) / config.num_carriers)
        self.assertLess(abs(observed - probability), 5 * standard_error)
        energies = 0.015 + np.arange(200) * 0.03
        weights = np.sqrt(energies) * np.exp(-energies / kt)
        weights /= weights.sum()
        mean = np.sum(weights * energies)
        variance = np.sum(weights * (energies - mean) ** 2)
        self.assertLess(abs(result.final_kinetic_energies_eV.mean() - mean),
                        5 * np.sqrt(variance / config.num_carriers))

    def test_invalid_configuration_and_initial_states(self):
        for settings in ({"num_carriers": 0}, {"duration_ps": -1},
                         {"sample_interval_ps": 0}, {"initial_valley": 3},
                         {"initial_kinetic_range_eV": (0.2, 0.1)}):
            with self.assertRaises(ValueError):
                SimulationConfig(**settings)
        with self.assertRaises(ValueError):
            run_simulation(SimulationConfig(num_carriers=2), initial_valleys=[0],
                           initial_kinetic_energies_eV=[0.1])


if __name__ == "__main__":
    unittest.main()
