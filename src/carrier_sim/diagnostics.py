"""Thermal references and mass-preserving marginals for the fixed-phonon model."""

from dataclasses import dataclass
from math import erf

import numpy as np

from .physics import K_B_EV_PER_K


@dataclass
class EquilibriumReference:
    kinetic_probability: np.ndarray  # joint probability by valley and kinetic bin
    total_probability: np.ndarray  # joint probability by valley and total-energy bin
    valley_probability: np.ndarray
    mean_kinetic_energy_eV: float
    mean_total_energy_eV: float
    sample_kinetic_categories: np.ndarray | None = None
    sample_mean_total_energy_eV: np.ndarray | None = None

    @property
    def kinetic_categories(self):
        overflow = self.valley_probability - self.kinetic_probability.sum(axis=1)
        return np.column_stack((self.kinetic_probability, np.maximum(overflow, 0)))


def absolute_histograms(density, kinetic_edges, floors):
    """Place valley histograms on a common total-energy grid.

    Bin widths must be uniform, with valley-floor differences aligned to the
    bins. Overflow counts must be handled separately.
    """
    density = np.asarray(density)
    edges, floors = np.asarray(kinetic_edges), np.asarray(floors)
    widths = np.diff(edges)
    if not len(widths) or np.any(widths <= 0) or not np.allclose(widths, widths[0]):
        raise ValueError("uniform, increasing kinetic-energy bins are required")
    offsets = (floors - floors.min()) / widths[0]
    shifts = np.rint(offsets).astype(int)
    if not np.allclose(offsets, shifts, atol=1e-8, rtol=0):
        raise ValueError("valley-floor differences must align with the histogram bins")
    if density.ndim != 3 or density.shape[1:] != (len(floors), len(widths)):
        raise ValueError("density must have shape (time, valley, kinetic bin)")
    size = len(widths) + shifts.max()
    output = np.zeros((density.shape[0], len(floors), size), dtype=density.dtype)
    for valley, offset in enumerate(shifts):
        output[:, valley, offset:offset + len(widths)] = density[:, valley]
    absolute_edges = floors.min() + edges[0] + np.arange(size + 1) * widths[0]
    return absolute_edges, output


def _gamma_energy_cdf(energy, thermal_energy):
    x = np.maximum(np.asarray(energy, dtype=float), 0) / thermal_energy
    root = np.sqrt(x)
    error_function = np.fromiter((erf(float(v)) for v in root.flat), float,
                                 count=root.size).reshape(root.shape)
    return error_function - 2 / np.sqrt(np.pi) * root * np.exp(-x)


def continuum_equilibrium(model, kinetic_edges, total_edges):
    """Unrestricted nondegenerate 3D parabolic-band Boltzmann reference."""
    if model.temperature_K <= 0:
        raise ValueError("positive temperature is required")
    kt = K_B_EV_PER_K * model.temperature_K
    floors = np.asarray(model.valley_floors_eV)
    masses = np.asarray(model.dos_mass_ratios)
    weights = (np.asarray(model.valley_degeneracies) * masses ** 1.5
               * np.exp(-(floors - floors.min()) / kt))
    weights /= weights.sum()
    kinetic = weights[:, None] * np.diff(_gamma_energy_cdf(kinetic_edges, kt))
    total = np.array([weight * np.diff(_gamma_energy_cdf(np.asarray(total_edges) - floor, kt))
                      for weight, floor in zip(weights, floors)])
    return EquilibriumReference(kinetic, total, weights, 1.5 * kt,
                                float(1.5 * kt + np.dot(weights, floors)))


def accessible_equilibrium(model, initial_total_energies, kinetic_edges, total_edges,
                           *, samples=0, seed=123, batch_size=512):
    """Boltzmann equilibrium conditional on each carrier's conserved E modulo hw.

    Normalize DOS_v(E-floor_v)*exp(-E/kBT) on each carrier's energy ladder,
    then mix using the initial class weights. The energy grid extends at least
    40 kBT above the highest valley floor. Optional samples draw one state per
    carrier from its own ladder. Temperature and all couplings must be positive.
    """
    if model.temperature_K <= 0 or np.any(model.coupling_ps_inv <= 0):
        raise ValueError("positive temperature and strictly positive couplings are required")
    initial = np.asarray(initial_total_energies, dtype=float)
    if initial.ndim != 1 or not len(initial) or not np.all(np.isfinite(initial)):
        raise ValueError("initial total energies must be a finite nonempty vector")
    floors = np.asarray(model.valley_floors_eV)
    if np.any(initial < floors.min()):
        raise ValueError("initial total energy is below every valley")
    if not isinstance(samples, int) or samples < 0 or batch_size < 1:
        raise ValueError("invalid sampling or batch count")
    kinetic_edges, total_edges = np.asarray(kinetic_edges), np.asarray(total_edges)
    if np.any(np.diff(kinetic_edges) <= 0) or np.any(np.diff(total_edges) <= 0):
        raise ValueError("reference histogram edges must increase")
    kt, phonon = K_B_EV_PER_K * model.temperature_K, model.phonon_energy_eV
    levels = int(np.ceil((np.ptp(floors) + 40 * kt) / phonon)) + 2
    count, valleys = len(initial), model.num_valleys
    bins = len(kinetic_edges) - 1
    kinetic = np.zeros((valleys, bins))
    total = np.zeros((valleys, len(total_edges) - 1))
    population = np.zeros(valleys)
    mean_kinetic = mean_total = 0.0
    sample_counts = np.zeros((samples, valleys, bins + 1), dtype=np.int64)
    sample_means = np.zeros(samples)
    rng = np.random.default_rng(seed)
    dos_factor = np.asarray(model.valley_degeneracies) * np.asarray(model.dos_mass_ratios) ** 1.5
    for start in range(0, count, batch_size):
        residues = np.remainder(initial[start:start + batch_size] - floors.min(), phonon)
        residues[np.isclose(residues, phonon, atol=phonon * 1e-12, rtol=0)] = 0
        grid = floors.min() + residues[:, None] + np.arange(levels) * phonon
        energies = grid[:, None, :] - floors[None, :, None]
        dos = dos_factor[None, :, None] * np.sqrt(np.maximum(energies, 0))
        baseline = np.min(np.where(dos > 0, grid[:, None, :], np.inf), axis=(1, 2))
        boltzmann = np.exp(-np.maximum(grid - baseline[:, None], 0) / kt)
        weights = dos * boltzmann[:, None, :]
        weights /= weights.sum(axis=(1, 2), keepdims=True)
        population += weights.sum(axis=(0, 2)) / count
        mean_total += np.sum(weights * grid[:, None, :]) / count
        mean_kinetic += np.sum(weights * energies) / count
        for valley in range(valleys):
            kinetic[valley] += np.histogram(energies[:, valley].ravel(), kinetic_edges,
                                            weights=weights[:, valley].ravel() / count)[0]
            total[valley] += np.histogram(grid.ravel(), total_edges,
                                          weights=weights[:, valley].ravel() / count)[0]
        if samples:
            cumulative = np.cumsum(weights.reshape(len(grid), -1), axis=1)
            draws = rng.random((samples, len(grid))) * cumulative[None, :, -1]
            selected = np.sum(cumulative[None, :, :] <= draws[:, :, None], axis=2)
            selected_valley, level = selected // levels, selected % levels
            selected_total = grid[np.arange(len(grid))[None, :], level]
            selected_kinetic = selected_total - floors[selected_valley]
            index = np.searchsorted(kinetic_edges, selected_kinetic, side="right") - 1
            index[selected_kinetic == kinetic_edges[-1]] = bins - 1
            index[(index < 0) | (index >= bins)] = bins
            np.add.at(sample_counts, (np.arange(samples)[:, None], selected_valley, index), 1)
            sample_means += selected_total.sum(axis=1) / count
    return EquilibriumReference(
        kinetic, total, population, float(mean_kinetic), float(mean_total),
        sample_counts / count if samples else None, sample_means if samples else None,
    )
