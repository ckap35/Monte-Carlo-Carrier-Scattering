"""Weak direct optical excitation of isotropic, three-dimensional parabolic bands.

The Gaussian is the electric-field spectral intensity |E_tilde(E_gamma)|^2.
With a constant interband dipole, pair generation is this spectrum times JDOS.
The pulse is treated as preparation at t=0; holes are recorded but not evolved.
"""

from dataclasses import asdict, dataclass
from functools import cached_property

import numpy as np

EV_IN_J = 1.602176634e-19
HBAR_J_S = 1.054571817e-34
ELECTRON_MASS_KG = 9.1093837139e-31


@dataclass(frozen=True)
class OpticalPairs:
    photon_energies_eV: np.ndarray
    electron_kinetic_energies_eV: np.ndarray
    hole_kinetic_energies_eV: np.ndarray


@dataclass(frozen=True)
class OpticalExcitation:
    band_gap_eV: float
    photon_center_eV: float
    spectral_fwhm_eV: float = 1.0
    electron_mass_ratio: float = 1.0
    hole_mass_ratio: float = 1.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def sigma_eV(self):
        return self.spectral_fwhm_eV / np.sqrt(8 * np.log(2))

    @property
    def reduced_mass_ratio(self):
        return 1 / (1 / self.electron_mass_ratio + 1 / self.hole_mass_ratio)

    @property
    def electron_energy_fraction(self):
        return self.hole_mass_ratio / (self.electron_mass_ratio + self.hole_mass_ratio)

    @property
    def excess_mode_eV(self):
        center = self.photon_center_eV - self.band_gap_eV
        root = np.hypot(center, np.sqrt(2) * self.sigma_eV)
        return ((center + root) / 2 if center >= 0
                else self.sigma_eV ** 2 / (root - center))

    def spectrum(self, photon_energy_eV):
        """Peak-normalized Gaussian electric-field spectral intensity."""
        photon = np.asarray(photon_energy_eV, dtype=float)
        return np.exp(-0.5 * ((photon - self.photon_center_eV) / self.sigma_eV) ** 2)

    def joint_dos(self, photon_energy_eV):
        """JDOS per eV per cm^3 for one band pair, including two spin states.

        In SI units: (1/(2*pi^2))*(2*mu/hbar^2)^(3/2)*sqrt(E_gamma-E_gap).
        This is a density of allowed transitions, not an absolute excitation yield.
        """
        excess_J = np.maximum(np.asarray(photon_energy_eV) - self.band_gap_eV, 0) * EV_IN_J
        reduced_kg = self.reduced_mass_ratio * ELECTRON_MASS_KG
        return ((2 * reduced_kg / HBAR_J_S ** 2) ** 1.5 / (2 * np.pi ** 2)
                * np.sqrt(excess_J) * EV_IN_J / 1e6)

    def _relative_weight(self, excess):
        # Subtract the exponent at the mode; this remains usable for subgap pumps.
        center, mode = self.photon_center_eV - self.band_gap_eV, self.excess_mode_eV
        return (np.sqrt(np.maximum(excess, 0))
                * np.exp(-(excess - mode) * (excess + mode - 2 * center)
                         / (2 * self.sigma_eV ** 2)))

    @cached_property
    def _normalization(self):
        center, sigma = self.photon_center_eV - self.band_gap_eV, self.sigma_eV
        low = max(0, center - 12 * sigma)
        high = max(center, 0) + 12 * sigma
        if center < -sigma:
            high = min(high, 40 * sigma ** 2 / -center)
        # q = low + (high-low)*u^2 resolves the square-root onset. Tails omitted
        # only in PDF quadrature are negligible; the random sampler is unbounded.
        nodes, weights = np.polynomial.legendre.leggauss(128)
        u = (nodes + 1) / 2
        q = low + (high - low) * u ** 2
        return np.sum(weights * (high - low) * u * self._relative_weight(q))

    def excess_pdf(self, excess_energy_eV):
        """Normalized pair excess-energy PDF, proportional to sqrt(q)*S(Eg+q)."""
        excess = np.asarray(excess_energy_eV, dtype=float)
        safe = np.maximum(excess, 0)
        return np.where(excess >= 0, self._relative_weight(safe) / self._normalization, 0)

    def electron_pdf(self, kinetic_energy_eV):
        """Electron marginal, including the energy-sharing Jacobian."""
        fraction = self.electron_energy_fraction
        return self.excess_pdf(np.asarray(kinetic_energy_eV) / fraction) / fraction

    def sample_pairs(self, count, rng):
        """Sample JDOS-weighted excitation with no energy grid or hard tail cutoff.

        For a pump near/above gap, a Gaussian centered at the target mode is an
        envelope because log(q) <= log(mode)+(q-mode)/mode. For far-subgap pumps,
        a gamma(3/2) proposal absorbs the linear exponential. Both are exact
        rejection samplers for sqrt(q)*exp(-(q-center)^2/(2*sigma^2)), q>0.
        """
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError("count must be a positive integer")
        center, sigma, mode = (self.photon_center_eV - self.band_gap_eV,
                               self.sigma_eV, self.excess_mode_eV)
        excess = np.empty(count)
        filled = 0
        while filled < count:
            size = max(32, 2 * (count - filled))
            if center < -sigma:
                proposed = rng.gamma(1.5, sigma ** 2 / -center, size=size)
                log_acceptance = -0.5 * (proposed / sigma) ** 2
            else:
                proposed = rng.normal(mode, sigma, size=size)
                proposed = proposed[proposed > 0]
                ratio = proposed / mode
                log_acceptance = 0.5 * (np.log(ratio) - ratio + 1)
            accepted = proposed[np.log(rng.random(len(proposed))) < log_acceptance]
            take = min(count - filled, len(accepted))
            excess[filled:filled + take] = accepted[:take]
            filled += take
        electron = self.electron_energy_fraction * excess
        return OpticalPairs(self.band_gap_eV + excess, electron, excess - electron)

    def to_dict(self):
        return asdict(self)
