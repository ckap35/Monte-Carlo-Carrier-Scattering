import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ----------------------------
# Simulation Parameters
# ----------------------------
NUM_CARRIERS = 10_000
NUM_VALLEYS = 3
NUM_ENERGY_BINS = 250
MAX_ENERGY = 0.5  # eV
DT = 0.1  # ps
N_STEPS = 5000
PHONON_ENERGY = 0.03  # eV
TEMPERATURE = 600  # K
K_B = 8.617333262145e-5  # eV/K

VALLEY_FLOORS = np.array([0.3, 0.05, 0.0])  # Γ, X, L
ENERGY_BINS = np.linspace(0, MAX_ENERGY, NUM_ENERGY_BINS)
DELTA_E = ENERGY_BINS[1] - ENERGY_BINS[0]


def intravalley_rate(E, T):
    """Energy and temperature-dependent intra-valley scattering rate."""
    return 0.05 * (1 + E) * (T / 300)


def intervalley_rate(E_from, valley_from, valley_to, T):
    """Scattering rate for intervalley transitions."""
    if valley_from == valley_to:
        return 0.0
    delta_E = VALLEY_FLOORS[valley_to] - VALLEY_FLOORS[valley_from]
    N_ph = 0 if delta_E == 0 else 1 / (np.exp(abs(delta_E) / (K_B * T)) - 1)
    base = 0.01 * (1 + E_from)
    return base * (N_ph + 1 if delta_E < 0 else N_ph)


def initialize_carriers():
    """Initialize carriers with random energy bins in the Γ valley."""
    carriers = np.zeros(NUM_CARRIERS, dtype=[('valley', int), ('energy_bin', int)])
    carriers['valley'][:] = 0
    carriers['energy_bin'][:] = np.random.randint(10, 20, size=NUM_CARRIERS)
    return carriers


def run_simulation():
    """Run the carrier scattering simulation."""
    carriers = initialize_carriers()
    density_history = np.zeros((N_STEPS + 1, NUM_VALLEYS, NUM_ENERGY_BINS))
    valley_population_history = np.zeros((N_STEPS + 1, NUM_VALLEYS))

    # Initial state
    for c in carriers:
        density_history[0, c['valley'], c['energy_bin']] += 1
    for v in range(NUM_VALLEYS):
        valley_population_history[0, v] = np.sum(carriers['valley'] == v)

    # Main time evolution
    for step in range(1, N_STEPS + 1):
        for i in range(NUM_CARRIERS):
            valley_idx = carriers['valley'][i]
            energy_idx = carriers['energy_bin'][i]
            energy = ENERGY_BINS[energy_idx] + VALLEY_FLOORS[valley_idx]

            # Intra-valley scattering
            if np.random.rand() < intravalley_rate(energy, TEMPERATURE) * DT:
                if energy_idx == 0:
                    energy_idx = 1
                elif energy_idx == NUM_ENERGY_BINS - 1:
                    energy_idx = NUM_ENERGY_BINS - 2
                else:
                    energy_idx += np.random.choice([-1, 1])
                carriers['energy_bin'][i] = energy_idx
                energy = ENERGY_BINS[energy_idx] + VALLEY_FLOORS[valley_idx]

            # Inter-valley scattering
            for valley_target in range(NUM_VALLEYS):
                if valley_idx == valley_target:
                    continue
                if np.random.rand() < intervalley_rate(energy, valley_idx, valley_target, TEMPERATURE) * DT:
                    delta_E = VALLEY_FLOORS[valley_target] - VALLEY_FLOORS[valley_idx]
                    energy_shift = int(PHONON_ENERGY / DELTA_E)
                    energy_idx = max(0, energy_idx - energy_shift) if delta_E < 0 else min(NUM_ENERGY_BINS - 1, energy_idx + energy_shift)
                    carriers['valley'][i] = valley_target
                    carriers['energy_bin'][i] = energy_idx
                    break

        # Save step data
        for v in range(NUM_VALLEYS):
            mask = (carriers['valley'] == v)
            valley_population_history[step, v] = np.sum(mask)
            density_history[step, v, :] = np.bincount(
                carriers['energy_bin'][mask], minlength=NUM_ENERGY_BINS
            )

    return density_history, valley_population_history


def plot_results(density_history, valley_population_history):
    """Generate plots for carrier evolution."""
    times = np.arange(N_STEPS + 1) * DT
    effective_energy = np.array([ENERGY_BINS + floor for floor in VALLEY_FLOORS])

    # 1. Carrier Count Heatmaps
    for v in range(NUM_VALLEYS):
        fig, ax = plt.subplots(figsize=(8, 5))
        energy_mesh, time_mesh = np.meshgrid(effective_energy[v], times)
        c = ax.pcolormesh(energy_mesh, time_mesh, density_history[:, v, :], shading='auto', cmap='jet')
        ax.set_title(f'Carrier Count Evolution - Valley {v} (Floor = {VALLEY_FLOORS[v]:.2f} eV)')
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Time (ps)')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(c, cax=cax, label='Carrier Count')
        plt.tight_layout()
        plt.show()

    # 2. Total Population per Valley vs Time
    plt.figure(figsize=(10, 6))
    for v in range(NUM_VALLEYS):
        plt.plot(times, valley_population_history[:, v], label=f'Valley {v} (Floor {VALLEY_FLOORS[v]:.1f} eV)')
    plt.xlabel('Time (ps)')
    plt.ylabel('Carrier Count')
    plt.title('Total Carrier Population per Valley Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3. Final Carrier Distribution per Valley
    plt.figure(figsize=(10, 6))
    for v in range(NUM_VALLEYS):
        plt.plot(effective_energy[v], density_history[-1, v], label=f'Valley {v}')
    plt.xlabel('Energy (eV)')
    plt.ylabel('Carrier Count')
    plt.title('Final Carrier Energy Distribution per Valley')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    density_history, valley_population_history = run_simulation()
    plot_results(density_history, valley_population_history)
