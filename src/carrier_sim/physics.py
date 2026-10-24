"""Phonon rates for parabolic conduction valleys.

Energies are in eV, time in ps, and coupling coefficients in ps^-1.
The destination DOS is scaled by a reference energy.
"""

from dataclasses import dataclass, field

import numpy as np

K_B_EV_PER_K = 8.617333262145e-5
EMISSION = 0
ABSORPTION = 1
ENERGY_SIGNS = np.array([-1.0, 1.0])
ENERGY_SIGNS.setflags(write=False)


@dataclass(frozen=True, eq=False)
class ScatteringModel:
    valley_names: tuple = ("Gamma", "X", "L")
    valley_floors_eV: tuple = (0.3, 0.05, 0.0)
    dos_mass_ratios: tuple = (1.0, 1.0, 1.0)
    valley_degeneracies: tuple = (1, 1, 1)
    temperature_K: float = 600.0
    phonon_energy_eV: float = 0.03
    dos_reference_energy_eV: float = 1.0
    coupling_ps_inv: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        names = tuple(self.valley_names)
        floors = tuple(float(x) for x in self.valley_floors_eV)
        masses = tuple(float(x) for x in self.dos_mass_ratios)
        degeneracies = tuple(float(x) for x in self.valley_degeneracies)
        count = len(names)
        if count == 0 or len(set(names)) != count:
            raise ValueError("valley_names must be nonempty and unique")
        if any(len(values) != count for values in (floors, masses, degeneracies)):
            raise ValueError("one floor, DOS mass, and degeneracy are required per valley")
        if not np.all(np.isfinite(floors)):
            raise ValueError("valley floors must be finite")
        if not np.all(np.isfinite(masses)) or np.any(np.array(masses) <= 0):
            raise ValueError("DOS mass ratios must be finite and positive")
        if (not np.all(np.isfinite(degeneracies))
                or any(x < 1 or not x.is_integer() for x in degeneracies)):
            raise ValueError("valley degeneracies must be positive integers")
        if not np.isfinite(self.temperature_K) or self.temperature_K < 0:
            raise ValueError("temperature_K must be finite and nonnegative")
        for value in (self.phonon_energy_eV, self.dos_reference_energy_eV):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("phonon and reference energies must be finite and positive")
        if self.coupling_ps_inv is None:
            coupling = np.full((count, count), 0.01)
            np.fill_diagonal(coupling, 0.05)
        else:
            if np.iscomplexobj(self.coupling_ps_inv):
                raise ValueError("couplings must be real")
            coupling = np.array(self.coupling_ps_inv, dtype=float, copy=True)
        if (coupling.shape != (count, count) or not np.all(np.isfinite(coupling))
                or np.any(coupling < 0)):
            raise ValueError("couplings must be a finite nonnegative square matrix")
        if not np.allclose(coupling, coupling.T, rtol=1e-12, atol=0):
            raise ValueError("symmetric couplings are required for detailed balance")
        coupling.setflags(write=False)
        for name, value in (("valley_names", names), ("valley_floors_eV", floors),
                            ("dos_mass_ratios", masses),
                            ("valley_degeneracies", degeneracies),
                            ("coupling_ps_inv", coupling)):
            object.__setattr__(self, name, value)

    @property
    def num_valleys(self):
        return len(self.valley_names)

    @property
    def phonon_occupation(self):
        if self.temperature_K == 0:
            return 0.0
        exponent = self.phonon_energy_eV / (K_B_EV_PER_K * self.temperature_K)
        return 0.0 if exponent > 700 else float(1.0 / np.expm1(exponent))

    def validate_states(self, valleys, kinetic_energies_eV):
        valleys = np.asarray(valleys)
        if not np.issubdtype(valleys.dtype, np.integer) or valleys.ndim != 1:
            raise ValueError("valleys must be a one-dimensional integer array")
        if np.any((valleys < 0) | (valleys >= self.num_valleys)):
            raise ValueError("valley index is out of range")
        if np.iscomplexobj(kinetic_energies_eV):
            raise ValueError("kinetic energies must be real")
        energies = np.asarray(kinetic_energies_eV, dtype=float)
        if (energies.shape != valleys.shape or not np.all(np.isfinite(energies))
                or np.any(energies < 0)):
            raise ValueError("kinetic energies must be finite, nonnegative, and match valleys")
        return valleys, energies

    def destination_energies(self, valleys, kinetic_energies_eV):
        """Return energies for every (carrier, destination, emission/absorption)."""
        valleys, energies = self.validate_states(valleys, kinetic_energies_eV)
        floors = np.asarray(self.valley_floors_eV)
        return (energies[:, None, None] + floors[valleys, None, None]
                - floors[None, :, None]
                + self.phonon_energy_eV * ENERGY_SIGNS[None, None, :])

    def channel_rates(self, valleys, kinetic_energies_eV):
        """Rates in ps^-1; kinematically forbidden channels have zero rate.

        The dimensionless final-state DOS is g_v * m_v^(3/2) * sqrt(E/E_ref).
        Symmetric couplings and Bose factors (N+1, N) enforce pairwise detailed
        balance with DOS-weighted Boltzmann occupations on accessible states.
        """
        valleys, energies = self.validate_states(valleys, kinetic_energies_eV)
        destination = self.destination_energies(valleys, energies)
        factors = (np.asarray(self.valley_degeneracies)
                   * np.asarray(self.dos_mass_ratios) ** 1.5)
        # Keep sqrt real, then zero the forbidden-channel rates.
        dos = (factors[None, :, None]
               * np.sqrt(np.maximum(destination, 0) / self.dos_reference_energy_eV))
        occupation = self.phonon_occupation
        return (self.coupling_ps_inv[valleys, :, None] * dos
                * np.array([occupation + 1, occupation])[None, None, :])

    def to_dict(self):
        return {
            "valley_names": self.valley_names,
            "valley_floors_eV": self.valley_floors_eV,
            "dos_mass_ratios": self.dos_mass_ratios,
            "valley_degeneracies": self.valley_degeneracies,
            "temperature_K": self.temperature_K,
            "phonon_energy_eV": self.phonon_energy_eV,
            "dos_reference_energy_eV": self.dos_reference_energy_eV,
            "coupling_ps_inv": self.coupling_ps_inv.tolist(),
        }
