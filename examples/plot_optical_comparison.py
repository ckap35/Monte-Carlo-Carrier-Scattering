"""Plot JDOS-weighted optical preparation and its subsequent relaxation."""

import argparse
import json
from pathlib import Path
import re
import zipfile

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from carrier_sim import OpticalExcitation, ScatteringModel
from carrier_sim.excitation import ELECTRON_MASS_KG, EV_IN_J, HBAR_J_S
from carrier_sim.diagnostics import absolute_histograms, accessible_equilibrium, continuum_equilibrium
from plot_style import BLUE, ORANGE, GREEN, GRAY, style, time_axis


def load_run(path):
    parameters = json.loads((path / "parameters.json").read_text())
    with np.load(path / "arrays.npz") as saved:
        arrays = {key: saved[key] for key in saved.files}
    model = ScatteringModel(**parameters["config"]["model"])
    edges, density = absolute_histograms(arrays["density_history"], arrays["energy_bin_edges_eV"],
                                         model.valley_floors_eV)
    return parameters, arrays, model, edges, density.sum(axis=1) / len(arrays["final_valleys"])


def save(figure, output, name, footer):
    figure.text(0.015, 0.013, footer, fontsize=8, color="#64748b")
    figure.savefig(output / (name + ".png"), dpi=180)
    figure.savefig(output / (name + ".svg"))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results/optical_excitation"))
    parser.add_argument("--output", type=Path, default=Path("results/optical_figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    parameters, a, model, total_edges, total_pdf = load_run(args.run)
    optical = OpticalExcitation(**parameters["config"]["optical_excitation"])
    tests_log = (args.run / "test_results.txt").read_text()
    match = re.search(r"Ran (\d+) tests", tests_log)
    if not match or not tests_log.rstrip().endswith("OK"):
        raise ValueError("save a successful test log in the optical run folder first")
    tests = int(match.group(1))
    times, count = a["times_ps"], len(a["final_valleys"])
    initial_electron = a["initial_electron_kinetic_energies_eV"]
    reference = accessible_equilibrium(model, a["initial_total_energies_eV"],
                                       a["energy_bin_edges_eV"], total_edges, samples=24, seed=123)
    continuum = continuum_equilibrium(model, a["energy_bin_edges_eV"], total_edges)
    categories = np.concatenate((a["density_history"], a["overflow_history"][:, :, None]), axis=2) / count
    to_model = np.abs(categories - reference.kinetic_categories).sum(axis=(1, 2)) / 2
    to_continuum = np.abs(categories - continuum.kinetic_categories).sum(axis=(1, 2)) / 2
    equilibrium_sampling = np.abs(reference.sample_kinetic_categories
                                   - reference.kinetic_categories).sum(axis=(1, 2)) / 2
    style()
    footer = (f"Direct excitation into {model.valley_names[parameters['config']['initial_valley']]} | "
              f"{count:,} electrons | {model.temperature_K:g} K bath | "
              f"{model.phonon_energy_eV * 1000:g} meV phonon | illustrative bands and rates")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.045, 1, 0.94))
    fig.suptitle("Optical preparation from parabolic bands and a broad pulse", fontsize=17, weight="bold")
    # k is shared by the electron and the missing valence electron.
    k_max = np.sqrt(2 * optical.reduced_mass_ratio * ELECTRON_MASS_KG
                    * 2 * EV_IN_J) / HBAR_J_S
    k = np.linspace(-k_max, k_max, 700)
    conduction = optical.band_gap_eV + HBAR_J_S ** 2 * k ** 2 / (2 * optical.electron_mass_ratio * ELECTRON_MASS_KG * EV_IN_J)
    valence = -HBAR_J_S ** 2 * k ** 2 / (2 * optical.hole_mass_ratio * ELECTRON_MASS_KG * EV_IN_J)
    axes[0, 0].plot(k / 1e9, conduction, color=BLUE, label="Conduction band")
    axes[0, 0].plot(k / 1e9, valence, color=GREEN, label="Valence band")
    center_excess = max(0, optical.photon_center_eV - optical.band_gap_eV)
    k_center = np.sqrt(2 * optical.reduced_mass_ratio * ELECTRON_MASS_KG * center_excess * EV_IN_J) / HBAR_J_S
    e_center = optical.band_gap_eV + optical.electron_energy_fraction * center_excess
    h_center = -(1 - optical.electron_energy_fraction) * center_excess
    axes[0, 0].annotate("", xy=(k_center / 1e9, e_center), xytext=(k_center / 1e9, h_center),
                         arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 2})
    axes[0, 0].text(0.04, 0.63, f"Direct gap = {optical.band_gap_eV:g} eV\n"
                    f"Electron / hole masses = {optical.electron_mass_ratio:g} / {optical.hole_mass_ratio:g} m₀",
                    transform=axes[0, 0].transAxes, va="top", fontsize=9)
    axes[0, 0].set(title="A   Vertical transitions conserve crystal momentum",
                   xlabel=r"Wavevector $k$ (nm$^{-1}$)", ylabel="Band energy relative to valence maximum (eV)")
    axes[0, 0].legend(loc="lower center")
    photons = np.linspace(0, optical.photon_center_eV + 1.5, 2000)
    generated = optical.excess_pdf(photons - optical.band_gap_eV)
    jdos = optical.joint_dos(photons)
    axes[0, 1].plot(photons, optical.spectrum(photons), color=ORANGE,
                    label=f"Pulse spectrum: {optical.spectral_fwhm_eV:g} eV FWHM")
    axes[0, 1].plot(photons, jdos / jdos.max(), color=GRAY, ls=":", label="JDOS ∝ √(Eγ − Eg)")
    axes[0, 1].plot(photons, generated / generated.max(), color=BLUE, label="Generated pairs ∝ spectrum × JDOS")
    axes[0, 1].axvline(optical.band_gap_eV, color=GREEN, lw=1, ls="--", label="Direct gap")
    axes[0, 1].set(title="B   The JDOS weights the optical spectrum",
                   xlabel="Photon energy (eV)", ylabel="Each curve normalized to its own maximum",
                   xlim=(max(0, optical.band_gap_eV - 0.5), optical.photon_center_eV + 1.5))
    axes[0, 1].legend()
    initial_edges = np.linspace(0, 1.5, 76)
    initial_histogram = np.histogram(initial_electron, initial_edges)[0] / count / np.diff(initial_edges)
    axes[1, 0].stairs(initial_histogram, initial_edges, color=BLUE, lw=1.3, label="Sampled electrons (20 meV bins)")
    electron_grid = np.linspace(0, 1.5, 1000)
    axes[1, 0].plot(electron_grid, optical.electron_pdf(electron_grid), "--", color="#172033",
                    label="Calculated electron PDF")
    axes[1, 0].text(0.97, 0.93, f"Mean εe = {initial_electron.mean():.3f} eV\n"
                    f"Electron excess-energy share = {optical.electron_energy_fraction:.0%}",
                    transform=axes[1, 0].transAxes, ha="right", va="top", fontsize=9)
    axes[1, 0].set(title="C   The electron and hole share the excess energy",
                   xlabel="Initial electron kinetic energy above injected valley (eV)",
                   ylabel="Probability density (1/eV)", xlim=(0, 1.5))
    axes[1, 0].legend(loc="center right")
    residue_edges = np.linspace(0, model.phonon_energy_eV, 31)
    residue = np.remainder(a["initial_total_energies_eV"] - min(model.valley_floors_eV), model.phonon_energy_eV)
    residue_probability = np.histogram(residue, residue_edges)[0] / count
    axes[1, 1].stairs(residue_probability / np.diff(residue_edges * 1000), residue_edges * 1000,
                      color=BLUE, lw=1.6, label="Optical preparation")
    axes[1, 1].axhline(1 / (model.phonon_energy_eV * 1000), color=GRAY, ls=":",
                       label="Uniform remainder density")
    axes[1, 1].set(title="D   Initial energy remainders under single-phonon scattering",
                   xlabel="Total energy modulo phonon energy (meV)",
                   ylabel="Probability density (1/meV)",
                   xlim=(0, model.phonon_energy_eV * 1000), ylim=(0, None))
    axes[1, 1].legend(loc="lower center")
    save(fig, args.output, "optical_initialization", footer + " | holes are recorded at preparation; only electrons evolve")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.045, 1, 0.94))
    fig.suptitle("Carrier relaxation after optical excitation", fontsize=17, weight="bold")
    image = axes[0, 0].pcolormesh(times, (total_edges[1:] + total_edges[:-1]) / 2,
                                 np.ma.masked_less_equal((total_pdf / np.diff(total_edges)).T, 0),
                                 shading="auto", cmap="magma", norm=LogNorm(vmin=0.02, vmax=50), rasterized=True)
    axes[0, 0].set(title="A   Total electron energy distribution over time",
                   ylabel="Total electron energy (eV)", ylim=(0, 1.6))
    time_axis(axes[0, 0], times[-1])
    fig.colorbar(image, ax=axes[0, 0], label="Probability density (1/eV)", pad=0.02)
    # Use 20 meV bins for the broad time slices; keep the 2 meV bins below.
    starts = np.arange(0, total_pdf.shape[1], 10)
    slice_edges = np.append(total_edges[starts], total_edges[-1])
    indices = sorted(set(np.argmin(abs(times - t)) for t in (0, 50, 500, times[-1])))
    for index, color in zip(indices, (ORANGE, GRAY, GREEN, BLUE)):
        probability = np.add.reduceat(total_pdf[index], starts)
        axes[0, 1].stairs(probability / np.diff(slice_edges), slice_edges,
                          color=color, lw=1.5, label=f"{times[index]:g} ps")
    probability = np.add.reduceat(reference.total_probability.sum(axis=0), starts)
    axes[0, 1].stairs(probability / np.diff(slice_edges), slice_edges, color="#172033",
                      ls="--", lw=1.6, label="Model equilibrium")
    axes[0, 1].set(title="B   Energy distributions at successive times (20 meV bins)",
                   xlabel="Total electron energy (eV)", ylabel="Probability density (1/eV)",
                   xlim=(0, 1.6))
    axes[0, 1].legend()
    for probability, edges, color, label, line in (
        (total_pdf[-1], total_edges, BLUE, "Simulation", "-"),
        (reference.total_probability.sum(axis=0), total_edges, "#172033", "Model equilibrium", "--"),
        (continuum.total_probability.sum(axis=0), total_edges, GREEN,
         f"Unrestricted {model.temperature_K:g} K thermal reference", ":"),
    ):
        axes[1, 0].stairs(probability / np.diff(edges), edges, color=color, ls=line, lw=1.3, label=label)
    axes[1, 0].set(title=f"C   Final distributions at {times[-1]:,.0f} ps (2 meV bins)",
                   xlabel="Total electron energy (eV)", ylabel="Probability density (1/eV)", xlim=(0, 0.45))
    axes[1, 0].legend()
    axes[1, 1].plot(times, to_model, color=BLUE, label="Distance to model equilibrium")
    axes[1, 1].plot(times, to_continuum, color=GREEN, ls="--", label="Distance to unrestricted thermal reference")
    axes[1, 1].axhspan(equilibrium_sampling.min(), equilibrium_sampling.max(), color=BLUE, alpha=0.15,
                      label="Range of 24 model-equilibrium draws")
    axes[1, 1].set(title="D   Relaxation toward the accessible equilibrium",
                   ylabel="Total variation distance (joint valley and kinetic bins)", ylim=(0, 1.01))
    time_axis(axes[1, 1], times[-1])
    axes[1, 1].legend()
    save(fig, args.output, "optical_relaxation_comparison", footer + f" | {tests} automated tests passed | no energy smoothing")

    numerical = json.loads((args.run / "metrics.json").read_text())
    numerical.update({
        "automated_tests_passed": tests,
        "optical_excitation": optical.to_dict(),
        "initial_mean_electron_kinetic_energy_eV": float(initial_electron.mean()),
        "initial_std_electron_kinetic_energy_eV": float(initial_electron.std()),
        "initial_mean_photon_energy_eV": float(a["initial_photon_energies_eV"].mean()),
        "final_mean_kinetic_energy_eV": float(a["final_kinetic_energies_eV"].mean()),
        "model_equilibrium_mean_total_energy_eV": reference.mean_total_energy_eV,
        "continuum_mean_total_energy_eV": continuum.mean_total_energy_eV,
        "model_equilibrium_valley_fractions": reference.valley_probability.tolist(),
        "continuum_valley_fractions": continuum.valley_probability.tolist(),
        "final_total_variation_to_model": float(to_model[-1]),
        "final_total_variation_to_continuum": float(to_continuum[-1]),
        "model_vs_continuum_total_variation": float(np.abs(reference.kinetic_categories - continuum.kinetic_categories).sum() / 2),
        "equilibrium_draw_total_variation_range": [float(equilibrium_sampling.min()), float(equilibrium_sampling.max())],
        "equilibrium_draw_mean_total_energy_range_eV": [float(reference.sample_mean_total_energy_eV.min()), float(reference.sample_mean_total_energy_eV.max())],
        "optical_initial_empty_remainder_bins_of_30": int(np.count_nonzero(residue_probability == 0)),
        "max_histogram_overflow": int(a["overflow_history"].sum(axis=1).max()),
    })
    (args.output / "optical_comparison_metrics.json").write_text(json.dumps(numerical, indent=2) + "\n")
    (args.output / "test_results.txt").write_text(tests_log)
    np.savez_compressed(args.output / "optical_comparison_data.npz",
                        photon_grid_eV=photons, pulse_spectrum=optical.spectrum(photons),
                        joint_dos_per_eV_cm3=jdos, generated_photon_pdf=generated,
                        initial_electron_energies_eV=initial_electron,
                        initial_hole_energies_eV=a["initial_hole_kinetic_energies_eV"],
                        total_energy_edges_eV=total_edges, optical_total_probability=total_pdf,
                        optical_times_ps=times,
                        model_total_probability=reference.total_probability,
                        continuum_total_probability=continuum.total_probability,
                        distance_to_model=to_model, distance_to_continuum=to_continuum)
    notes = f"""# Optical excitation and carrier relaxation

Direct gap: {optical.band_gap_eV:g} eV. Pulse center: {optical.photon_center_eV:g} eV.
Gaussian spectral-intensity FWHM: {optical.spectral_fwhm_eV:g} eV.
Electron and hole masses: {optical.electron_mass_ratio:g} and {optical.hole_mass_ratio:g} m0.
The gap is measured to the injected valley. Only electrons are propagated.

## Preparation

For isotropic 3D parabolic bands, 1/mu = 1/me + 1/mh. The spin-inclusive JDOS
is (1/(2 pi^2)) (2 mu/hbar^2)^(3/2) sqrt(Egamma-Eg) above the gap and zero below.
Pairs are drawn from this JDOS times the Gaussian electric-field spectrum,
assuming a constant dipole, a filled valence band, and an empty conduction band.

The electron receives mh/(me+mh) of the excess photon energy and the hole
receives me/(me+mh). The pulse prepares the state at t=0. The pair count is
specified by the run; predicting the yield would also require fluence and a
dipole magnitude. Sampling uses continuous energies.

The single {model.phonon_energy_eV * 1000:g} meV phonon preserves each electron's
energy remainder. The equilibrium reference keeps those initial class weights;
the unrestricted thermal distribution is shown separately.

## Results

At {times[-1]:g} ps, mean total energy is {numerical['final_mean_total_energy_eV']:.8f} eV.
The accessible-equilibrium value is {reference.mean_total_energy_eV:.8f} eV.
The final joint-histogram total variation distance is {to_model[-1]:.5f};
24 independent equilibrium draws give {equilibrium_sampling.min():.5f} to
{equilibrium_sampling.max():.5f}. This sampling range is not a confidence interval
or a test that every observable has equilibrated. The test log records
{tests} passing tests.

The archive contains PNG/SVG figures, optical-run arrays, metrics, and the test
log. Simulation commands and source code are in the repository README.

JDOS reference: David Miller, Stanford, Lecture 32,
https://web.stanford.edu/group/dabmgroup/cgi-bin/dabm/wp-content/uploads/2021/12/Lecture_32.pdf
"""
    (args.output / "README.md").write_text(notes)
    with zipfile.ZipFile(args.output / "Optical-excitation-comparison.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(args.output.iterdir()):
            if path.suffix in (".png", ".svg", ".npz", ".json", ".txt", ".md"):
                archive.write(path, path.name)
    print(json.dumps(numerical, indent=2))


if __name__ == "__main__":
    main()
