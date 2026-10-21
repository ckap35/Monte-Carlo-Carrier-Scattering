# Carrier scattering simulation

A Monte Carlo model of independent carriers moving between semiconductor valleys
through intra- and intervalley phonon emission and absorption. Carriers have
continuous kinetic energies; energy bins are used only for output histograms.

The rate coefficients, DOS masses, degeneracies, and valley floors are
illustrative. This version is not calibrated to a particular material.
There is no electric field, spatial transport, carrier-carrier interaction,
or Pauli blocking. The phonon bath stays at a fixed temperature.

## Install and run

Requires Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
python -m carrier_sim --config examples/optical_excitation_config.json --seed 0 --output results/example
```

The installed `carrier-sim` command and `python examples/run_simulation.py`
accept the same options, including `--config` for the optical example.
The `PYTHONPATH=src python examples/run_simulation.py` entry point also works.
Importing the package does not run a simulation or create plots.

Useful options:

```bash
carrier-sim --config examples/optical_excitation_config.json --carriers 10000 --duration-ps 500 --sample-ps 1 --temperature-K 600 --seed 0
carrier-sim --config results/example/parameters.json --output results/repeated
```

The sample interval controls output snapshots. Event times are sampled
continuously and do not use the sample interval as an approximation.
Use `--show` to also open the saved figure interactively.

Each run saves `parameters.json`, `metrics.json`, `arrays.npz`, and
`summary.png`. Saved parameters contain the model, simulation settings,
seed, Python version, and NumPy version. Repeating a run with the same settings,
seed, and software environment reproduces the arrays; timing can vary.

## Energy conservation and rates

Kinetic energy is measured from the occupied valley's floor. For a transition
from valley i to j:

```text
E_kin_final = E_kin_initial + floor_i - floor_j - phonon_energy  (emission)
E_kin_final = E_kin_initial + floor_i - floor_j + phonon_energy  (absorption)
```

Channels with negative final kinetic energy have zero rate. Emission and
absorption are separate channels for every destination, including the starting
valley. A downhill valley transition can absorb a phonon, and an uphill
transition can emit one when sufficient kinetic energy is available.

With phonon occupation `N = 1 / expm1(phonon_energy / (k_B T))`, the rates are:

```text
rate_emission   = C_ij * DOS_j(E_kin_final) * (N + 1)
rate_absorption = C_ij * DOS_j(E_kin_final) * N
DOS_j(E)       = degeneracy_j * mass_ratio_j^(3/2) * sqrt(E / E_reference)
```

Here DOS is a dimensionless parabolic-band shape and `C_ij` is in ps^-1.
The reference energy defaults to 1 eV. Symmetric, nonnegative coupling
coefficients are required so reverse channels satisfy detailed balance.
At zero temperature, absorption vanishes. The square-root DOS vanishes
exactly at a valley floor.

The default floors are 0.30, 0.05, and 0 eV (Gamma, X, L), the phonon
energy is 0.03 eV, temperature is 600 K, DOS masses and degeneracies are one,
and coupling coefficients are 0.05 within a valley and 0.01 between valleys.
Carriers start in Gamma with kinetic energies uniformly sampled from 0.02
to 0.04 eV. These defaults retain the original energy scales, while replacing
the earlier illustrative rate laws and energy-bin random walk.

Event waiting times are exponential with the total outgoing rate. The next
channel is selected in proportion to its rate. All events before each output
snapshot are processed, including multiple events for one carrier.

## Equilibrium and model limits

For positive temperature, reverse channels obey detailed balance with
DOS-weighted Boltzmann occupations. A single fixed phonon energy confines
each carrier's total energy to a ladder separated by that phonon energy.
The equilibrium tests therefore compare against the distribution on the
accessible ladder. They do not establish full continuum thermalization.
Additional phonon modes or acoustic energy exchange would be needed to
remove this restriction.

Kinetic energies are neither clipped to the plotting range nor rounded to
bins. Histogram overflow is counted separately. Recorded populations obey:

```text
histogram_count + overflow_count = valley_population
```

The simulation records emitted and absorbed phonons. At each snapshot,
`total_carrier_energy + net_energy_to_phonon_bath` equals its initial value
up to floating-point roundoff. Per-carrier energy balances and a cumulative
(source valley, destination valley, emission/absorption) event count are also
available in the saved arrays.

## Python API and configuration

```python
from carrier_sim import ScatteringModel, SimulationConfig, run_simulation

model = ScatteringModel(temperature_K=300)
config = SimulationConfig(model=model, num_carriers=2000, duration_ps=100)
result = run_simulation(config, seed=42)
populations = result.valley_population_history
errors = result.carrier_energy_balance_errors_eV
```

The old tuple return is replaced by a `SimulationResult` with explicit times,
bin edges, histories, final states, and diagnostics. Custom initial carrier
arrays can be supplied to `run_simulation`. Custom floors, masses,
degeneracies, and symmetric couplings can be supplied to `ScatteringModel`.
JSON configuration uses the fields returned by `SimulationConfig.to_dict()`;
the CLI can also reload its own saved `parameters.json`.

## Tests

After installation:

```bash
python -m unittest discover -s tests -v
```

Tests cover single-event energy conservation, forbidden transitions, phonon
occupations, zero-temperature absorption, detailed balance, particle and
energy conservation, continuous energies independent of histogram bins,
exponential survival, competing-channel probabilities, reproducibility,
snapshot spacing, a known equilibrium distribution, and CLI output reloads.
Stochastic checks use fixed seeds and five-standard-error tolerances.
Optical-preparation checks also compare the JDOS integral with a k-space state
count, enforce electron/hole energy and momentum sharing, verify the pulse
FWHM, and compare sampled energies with analytic moments and quadrature.

The physics follows the standard energy-conserving phonon-channel formulation;
the numerical coefficients still require material-specific calibration.
References: [Jacoboni and Reggiani, Rev. Mod. Phys. 55, 645 (1983)](https://doi.org/10.1103/RevModPhys.55.645)
and [Eric Pop, Monte Carlo Transport and Heat Generation in Semiconductors (2014)](https://poplab.stanford.edu/pdfs/Pop-HeatGenSemic-arht14.pdf).

## Files

### Optical preparation from the JDOS

`examples/optical_excitation_config.json` prepares carriers through a weak,
allowed direct transition between isotropic 3D
parabolic valence and conduction bands. The direct gap is measured from the
valence maximum to the injected conduction valley, independently of the valley
floor offsets used in transport. The example excites Gamma only.

For positive electron and hole masses, define `1/mu = 1/me + 1/mh`.
At a shared wavevector k:

```text
E_gamma = Eg + hbar^2 k^2 / (2 mu)
JDOS(E_gamma) = (1/(2 pi^2)) (2 mu/hbar^2)^(3/2) sqrt(E_gamma - Eg), E_gamma > Eg
S(E_gamma) = exp[-4 ln(2) ((E_gamma - E_center)/FWHM)^2]
P(E_gamma) proportional to JDOS(E_gamma) * S(E_gamma)
epsilon_e = mh/(me + mh) * (E_gamma - Eg)
epsilon_h = me/(me + mh) * (E_gamma - Eg)
```

The JDOS includes two spin states. `S` is the electric-field spectral intensity
`|E_tilde|^2`, and the interband dipole matrix element is taken constant.
The valence band starts filled and the conduction band empty. Sampling uses
continuous, unbounded rejection sampling; photon energies below the gap have
zero generation weight. `joint_dos` reports states per eV per cm^3. The chosen
carrier count sets the number of generated pairs; an absolute fluence and
dipole magnitude would be needed to predict the yield.

The illustrative example uses a 1 eV direct gap, 2 eV pulse center, 1 eV
Gaussian spectral-intensity FWHM, and equal electron/hole masses of m0. These
are editable parameters, not values assigned to a particular material.
The initial valley's DOS mass must match the optical electron mass. With
equal masses, each particle receives half the excess photon energy, so the
electron's width before JDOS weighting is 0.5 eV. JDOS weighting then shifts
and reshapes it. The pulse is treated as instantaneous preparation at t=0;
the paired hole energies are saved but holes are not propagated.

### Distribution and equilibrium figures

The following commands prepare 20,000 carriers from the optical pulse, evolve
them for 5,000 ps, run the tests, and create PNG/SVG figures and reference data:

```bash
python -m carrier_sim --config examples/optical_excitation_config.json --seed 0 --output results/optical_excitation
python -m unittest discover -s tests -v > results/optical_excitation/test_results.txt 2>&1
python examples/plot_optical_comparison.py --run results/optical_excitation --output results/optical_figures
```

The figures show the parabolic bands, pulse spectrum, JDOS-weighted initial
distribution, energy distributions over time, and late-time equilibrium
comparisons. Total energy modulo the single phonon energy is conserved; the
stationary reference therefore preserves the actual initial class weights.
Independent equilibrium draws provide a finite-ensemble sampling reference.
The unrestricted continuum Maxwell-Boltzmann distribution is shown separately.
Histograms use matched bin widths when comparing simulated and reference
distributions. No temperature or rate parameters are fitted to the plots.

JDOS derivation: [David Miller, Stanford, Optical absorption in semiconductors,
Lecture 32](https://web.stanford.edu/group/dabmgroup/cgi-bin/dabm/wp-content/uploads/2021/12/Lecture_32.pdf).

### Validated optical example

The [validation report](docs/validation.md) records the 35-test result, model
parameters, conservation checks, and equilibrium comparison. The complete
[test output](docs/test_results.txt) and
[optical initialization figure](docs/figures/optical_initialization.png) are included.

![Carrier relaxation after optical excitation](docs/figures/optical_relaxation.png)

### Source files

- `src/carrier_sim/physics.py`: channel energies, rates, and model parameters.
- `src/carrier_sim/excitation.py`: parabolic-band JDOS and optical pair sampling.
- `src/carrier_sim/simulation.py`: event-time simulation, diagnostics, and plotting.
- `src/carrier_sim/diagnostics.py`: marginals and equilibrium reference distributions.
- `src/carrier_sim/cli.py`: configuration, saved results, and command-line interface.
- `examples/run_simulation.py`: example entry point.
- `examples/plot_optical_comparison.py`: optical initialization and relaxation figures.
- `tests/`: physics, simulation, and output checks.

## License

MIT; see `LICENSE`.
