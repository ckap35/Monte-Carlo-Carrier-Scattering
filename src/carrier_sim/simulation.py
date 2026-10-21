"""Independent carriers with continuous energies and exponential event times."""

from dataclasses import dataclass, field

import numpy as np

from .physics import ScatteringModel, ENERGY_SIGNS
from .excitation import OpticalExcitation, OpticalPairs


@dataclass(frozen=True)
class SimulationConfig:
    model: ScatteringModel = field(default_factory=ScatteringModel)
    num_carriers: int = 10_000
    duration_ps: float = 500.0
    sample_interval_ps: float = 1.0
    num_energy_bins: int = 250
    histogram_max_kinetic_energy_eV: float = 0.5
    initial_valley: int = 0
    initial_kinetic_range_eV: tuple = (0.02, 0.04)
    optical_excitation: OpticalExcitation | None = None

    def __post_init__(self):
        for name in ("num_carriers", "num_energy_bins"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.duration_ps) or self.duration_ps < 0:
            raise ValueError("duration_ps must be finite and nonnegative")
        for name in ("sample_interval_ps", "histogram_max_kinetic_energy_eV"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (isinstance(self.initial_valley, bool)
                or not isinstance(self.initial_valley, (int, np.integer))
                or not 0 <= self.initial_valley < self.model.num_valleys):
            raise ValueError("initial_valley is out of range")
        low, high = self.initial_kinetic_range_eV
        if not np.all(np.isfinite([low, high])) or not 0 <= low <= high:
            raise ValueError("initial kinetic range must be finite with 0 <= low <= high")
        object.__setattr__(self, "initial_kinetic_range_eV", (float(low), float(high)))
        optical = self.optical_excitation
        if isinstance(optical, dict):
            optical = OpticalExcitation(**optical)
            object.__setattr__(self, "optical_excitation", optical)
        if optical is not None:
            if not isinstance(optical, OpticalExcitation):
                raise ValueError("optical_excitation must be an OpticalExcitation or settings dict")
            if not np.isclose(optical.electron_mass_ratio,
                              self.model.dos_mass_ratios[self.initial_valley], rtol=1e-12, atol=0):
                raise ValueError("optical electron mass must match the initial valley's DOS mass")

    def to_dict(self):
        return {
            "model": self.model.to_dict(), "num_carriers": self.num_carriers,
            "duration_ps": self.duration_ps, "sample_interval_ps": self.sample_interval_ps,
            "num_energy_bins": self.num_energy_bins,
            "histogram_max_kinetic_energy_eV": self.histogram_max_kinetic_energy_eV,
            "initial_valley": self.initial_valley,
            "initial_kinetic_range_eV": self.initial_kinetic_range_eV,
            "optical_excitation": (None if self.optical_excitation is None
                                   else self.optical_excitation.to_dict()),
        }


@dataclass
class SimulationResult:
    config: SimulationConfig
    times_ps: np.ndarray
    energy_bin_edges_eV: np.ndarray
    density_history: np.ndarray
    valley_population_history: np.ndarray
    overflow_history: np.ndarray
    total_carrier_energy_eV: np.ndarray
    phonon_energy_to_bath_eV: np.ndarray
    final_valleys: np.ndarray
    final_kinetic_energies_eV: np.ndarray
    initial_total_energies_eV: np.ndarray
    net_phonons_emitted: np.ndarray
    event_counts: np.ndarray
    initial_optical_pairs: OpticalPairs | None = None

    @property
    def energy_balance_error_eV(self):
        return (self.total_carrier_energy_eV + self.phonon_energy_to_bath_eV
                - self.total_carrier_energy_eV[0])

    @property
    def carrier_energy_balance_errors_eV(self):
        model = self.config.model
        final_total = (self.final_kinetic_energies_eV
                       + np.asarray(model.valley_floors_eV)[self.final_valleys])
        return (final_total + self.net_phonons_emitted * model.phonon_energy_eV
                - self.initial_total_energies_eV)


def _waiting_times(rates, rng):
    total = rates.sum(axis=(1, 2))
    waiting = np.full(len(total), np.inf)
    active = total > 0
    waiting[active] = rng.exponential(size=np.count_nonzero(active)) / total[active]
    return waiting


def _sample_times(duration, interval):
    times = np.arange(0, duration, interval, dtype=float)
    return np.append(times[times < duration], duration)


def run_simulation(config=None, *, seed=0, initial_valleys=None,
                   initial_kinetic_energies_eV=None):
    """Run exact event-time sampling for independent, time-homogeneous carriers.

    Each channel has a constant rate between events. Snapshot spacing controls
    output sampling only; it is not used to approximate event probabilities.
    Energies are never rounded to bins or clipped to the histogram range.
    """
    config = SimulationConfig() if config is None else config
    model = config.model
    rng = np.random.default_rng(seed)
    count, valleys_count = config.num_carriers, model.num_valleys
    valleys = (np.full(count, config.initial_valley, dtype=int)
               if initial_valleys is None else np.array(initial_valleys, copy=True))
    pairs = None
    if config.optical_excitation is not None:
        if initial_kinetic_energies_eV is not None:
            raise ValueError("choose optical excitation or explicit initial energies")
        if np.any(valleys != config.initial_valley):
            raise ValueError("optical excitation prepares only the configured initial valley")
        pairs = config.optical_excitation.sample_pairs(count, rng)
        energies = pairs.electron_kinetic_energies_eV.copy()
    else:
        energies = (rng.uniform(*config.initial_kinetic_range_eV, size=count)
                    if initial_kinetic_energies_eV is None
                    else np.array(initial_kinetic_energies_eV, copy=True))
    valleys, energies = model.validate_states(valleys, energies)
    if len(valleys) != count:
        raise ValueError("initial states must match num_carriers")
    floors = np.asarray(model.valley_floors_eV)
    initial_total = energies + floors[valleys]
    rates = model.channel_rates(valleys, energies)
    next_event_times = _waiting_times(rates, rng)
    net_emitted = np.zeros(count, dtype=np.int64)
    event_counts = np.zeros((valleys_count, valleys_count, 2), dtype=np.int64)

    times = _sample_times(config.duration_ps, config.sample_interval_ps)
    edges = np.linspace(0, config.histogram_max_kinetic_energy_eV,
                        config.num_energy_bins + 1)
    density = np.zeros((len(times), valleys_count, config.num_energy_bins), dtype=np.int64)
    population = np.zeros((len(times), valleys_count), dtype=np.int64)
    overflow = np.zeros_like(population)
    total_energy = np.zeros(len(times))
    bath_energy = np.zeros(len(times))

    for snapshot, time in enumerate(times):
        active = np.flatnonzero(next_event_times <= time)
        while len(active):
            source = valleys[active].copy()
            cumulative = np.cumsum(rates[active].reshape(len(active), -1), axis=1)
            draws = rng.random(len(active)) * cumulative[:, -1]
            channels = np.sum(cumulative <= draws[:, None], axis=1)
            destination, kind = channels // 2, channels % 2
            possible = model.destination_energies(source, energies[active])
            new_energy = possible[np.arange(len(active)), destination, kind]
            if np.any(new_energy < 0):
                raise RuntimeError("a forbidden negative-energy channel was selected")
            valleys[active], energies[active] = destination, new_energy
            net_emitted[active] -= ENERGY_SIGNS[kind].astype(np.int64)
            np.add.at(event_counts, (source, destination, kind), 1)
            rates[active] = model.channel_rates(destination, new_energy)
            next_event_times[active] += _waiting_times(rates[active], rng)
            active = np.flatnonzero(next_event_times <= time)

        for valley in range(valleys_count):
            selected = energies[valleys == valley]
            population[snapshot, valley] = len(selected)
            density[snapshot, valley] = np.histogram(selected, bins=edges)[0]
            overflow[snapshot, valley] = np.count_nonzero(selected > edges[-1])
        total_energy[snapshot] = np.sum(energies + floors[valleys])
        bath_energy[snapshot] = net_emitted.sum() * model.phonon_energy_eV

    return SimulationResult(
        config=config, times_ps=times, energy_bin_edges_eV=edges,
        density_history=density, valley_population_history=population,
        overflow_history=overflow, total_carrier_energy_eV=total_energy,
        phonon_energy_to_bath_eV=bath_energy, final_valleys=valleys,
        final_kinetic_energies_eV=energies, initial_total_energies_eV=initial_total,
        net_phonons_emitted=net_emitted, event_counts=event_counts,
        initial_optical_pairs=pairs,
    )


def plot_results(result, output_path=None, *, show=False):
    """Plot a result without relying on global simulation settings."""
    from matplotlib import pyplot as plt

    model = result.config.model
    count = model.num_valleys
    figure, axes = plt.subplots(2, max(count, 3), figsize=(15, 8),
                                constrained_layout=True, squeeze=False)
    centers = (result.energy_bin_edges_eV[:-1] + result.energy_bin_edges_eV[1:]) / 2
    for valley in range(count):
        image = axes[0, valley].pcolormesh(
            centers + model.valley_floors_eV[valley], result.times_ps,
            result.density_history[:, valley], shading="auto", cmap="viridis")
        axes[0, valley].set(title=f"{model.valley_names[valley]} energy distribution",
                           xlabel="Total carrier energy (eV)", ylabel="Time (ps)")
        figure.colorbar(image, ax=axes[0, valley], label="Carriers per energy bin")
        axes[1, 0].plot(result.times_ps,
                        result.valley_population_history[:, valley] / result.config.num_carriers,
                        label=model.valley_names[valley])
        axes[1, 2].plot(centers + model.valley_floors_eV[valley],
                        result.density_history[-1, valley], label=model.valley_names[valley])
    axes[1, 0].set(title="Valley populations", xlabel="Time (ps)", ylabel="Carrier fraction")
    axes[1, 0].legend()
    axes[1, 1].plot(result.times_ps,
                    result.total_carrier_energy_eV / result.config.num_carriers)
    axes[1, 1].set(title="Mean total carrier energy", xlabel="Time (ps)", ylabel="Energy (eV)")
    axes[1, 2].set(title="Final energy distributions", xlabel="Total carrier energy (eV)",
                   ylabel="Carriers per energy bin")
    axes[1, 2].legend()
    for column in range(count, axes.shape[1]):
        axes[0, column].set_visible(False)
    for column in range(3, axes.shape[1]):
        axes[1, column].set_visible(False)
    figure.suptitle("Carrier scattering | illustrative rates, exact event times")
    if output_path is not None:
        figure.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(figure)
