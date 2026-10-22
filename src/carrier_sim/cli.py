"""Run carrier scattering from JSON settings and save the result."""

import argparse
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from .physics import ScatteringModel, EMISSION, ABSORPTION
from .simulation import SimulationConfig, run_simulation, plot_results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON configuration or saved parameters.json")
    parser.add_argument("--carriers", type=int)
    parser.add_argument("--duration-ps", type=float)
    parser.add_argument("--sample-ps", type=float)
    parser.add_argument("--temperature-K", type=float, dest="temperature_K")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, default=Path("results/carrier_scattering"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)
    saved = json.loads(args.config.read_text()) if args.config else {}
    settings = dict(saved.get("config", saved))
    model_settings = dict(settings.pop("model", {}))
    if args.temperature_K is not None:
        model_settings["temperature_K"] = args.temperature_K
    for key, value in (("num_carriers", args.carriers), ("duration_ps", args.duration_ps),
                       ("sample_interval_ps", args.sample_ps)):
        if value is not None:
            settings[key] = value
    seed = saved.get("seed", 0) if args.seed is None else args.seed
    config = SimulationConfig(model=ScatteringModel(**model_settings), **settings)
    started = perf_counter()
    result = run_simulation(config, seed=seed)
    elapsed = perf_counter() - started
    metrics = {
        "seed": seed, "num_carriers": config.num_carriers,
        "duration_ps": config.duration_ps,
        "events": int(result.event_counts.sum()),
        "phonons_emitted": int(result.event_counts[:, :, EMISSION].sum()),
        "phonons_absorbed": int(result.event_counts[:, :, ABSORPTION].sum()),
        "population_conserved": bool(np.all(result.valley_population_history.sum(axis=1) == config.num_carriers)),
        "histograms_account_for_all_carriers": bool(np.array_equal(
            result.density_history.sum(axis=2) + result.overflow_history,
            result.valley_population_history)),
        "max_total_energy_balance_error_eV": float(np.max(np.abs(result.energy_balance_error_eV))),
        "max_carrier_energy_balance_error_eV": float(np.max(np.abs(result.carrier_energy_balance_errors_eV))),
        "initial_mean_total_energy_eV": float(result.total_carrier_energy_eV[0] / config.num_carriers),
        "final_mean_total_energy_eV": float(result.total_carrier_energy_eV[-1] / config.num_carriers),
        "final_populations": dict(zip(config.model.valley_names, map(int, result.valley_population_history[-1]))),
        "minimum_final_kinetic_energy_eV": float(result.final_kinetic_energies_eV.min()),
        "final_histogram_overflow": int(result.overflow_history[-1].sum()),
        "simulation_seconds": elapsed,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    parameters = {
        "config": config.to_dict(), "seed": seed,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
    }
    (args.output / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    optical_arrays = {}
    if result.initial_optical_pairs is not None:
        pairs = result.initial_optical_pairs
        optical_arrays = {
            "initial_photon_energies_eV": pairs.photon_energies_eV,
            "initial_electron_kinetic_energies_eV": pairs.electron_kinetic_energies_eV,
            "initial_hole_kinetic_energies_eV": pairs.hole_kinetic_energies_eV,
        }
    np.savez_compressed(
        args.output / "arrays.npz", times_ps=result.times_ps,
        energy_bin_edges_eV=result.energy_bin_edges_eV,
        density_history=result.density_history,
        valley_population_history=result.valley_population_history,
        overflow_history=result.overflow_history,
        total_carrier_energy_eV=result.total_carrier_energy_eV,
        phonon_energy_to_bath_eV=result.phonon_energy_to_bath_eV,
        final_valleys=result.final_valleys,
        final_kinetic_energies_eV=result.final_kinetic_energies_eV,
        initial_total_energies_eV=result.initial_total_energies_eV,
        net_phonons_emitted=result.net_phonons_emitted,
        event_counts=result.event_counts,
        **optical_arrays,
    )
    plot_results(result, args.output / "summary.png", show=args.show)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
