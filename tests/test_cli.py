import json
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose


ROOT = Path(__file__).resolve().parents[1]


class CommandLineTests(unittest.TestCase):
    def test_optical_excitation_saved_pairs_and_config_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first", Path(directory) / "second"
            environment = dict(os.environ, MPLBACKEND="Agg")
            subprocess.run(
                [sys.executable, "-m", "carrier_sim", "--config",
                 str(ROOT / "examples/optical_excitation_config.json"),
                 "--carriers", "200", "--duration-ps", "5", "--seed", "12",
                 "--output", str(first)], env=environment, capture_output=True, text=True, check=True)
            subprocess.run(
                [sys.executable, "-m", "carrier_sim", "--config", str(first / "parameters.json"),
                 "--output", str(second)], env=environment, capture_output=True, text=True, check=True)
            with np.load(first / "arrays.npz") as a, np.load(second / "arrays.npz") as b:
                for key in a.files:
                    assert_array_equal(a[key], b[key])
                assert_allclose(a["initial_photon_energies_eV"],
                                1 + a["initial_electron_kinetic_energies_eV"]
                                + a["initial_hole_kinetic_energies_eV"])

    def test_import_has_no_output_or_files(self):
        with tempfile.TemporaryDirectory() as directory:
            process = subprocess.run(
                [sys.executable, "-c", "import carrier_sim; import carrier_sim.cli"],
                cwd=directory, capture_output=True, text=True, check=True,
            )
            self.assertEqual(process.stdout, "")
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_saved_arrays_energy_ledger_and_config_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            environment = os.environ.copy()
            environment["MPLBACKEND"] = "Agg"
            command = [sys.executable, str(ROOT / "examples/run_simulation.py"),
                       "--carriers", "200", "--duration-ps", "5", "--seed", "12",
                       "--output", str(first)]
            process = subprocess.run(command, env=environment, capture_output=True,
                                     text=True, check=True)
            metrics = json.loads(process.stdout)
            self.assertEqual(metrics, json.loads((first / "metrics.json").read_text()))
            self.assertTrue(metrics["population_conserved"])
            self.assertTrue(metrics["histograms_account_for_all_carriers"])
            self.assertLess(metrics["max_carrier_energy_balance_error_eV"], 1e-12)
            subprocess.run(
                [sys.executable, "-m", "carrier_sim", "--config", str(first / "parameters.json"),
                 "--output", str(second)], env=environment, capture_output=True, text=True, check=True,
            )
            with np.load(first / "arrays.npz") as a, np.load(second / "arrays.npz") as b:
                self.assertEqual(a.files, b.files)
                for key in a.files:
                    assert_array_equal(a[key], b[key])
                assert_allclose(a["total_carrier_energy_eV"] + a["phonon_energy_to_bath_eV"],
                                a["total_carrier_energy_eV"][0], atol=1e-10)
            self.assertTrue((first / "summary.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))


if __name__ == "__main__":
    unittest.main()
