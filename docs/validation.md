# Optical excitation: reference run

This run uses the optical configuration in `examples/optical_excitation_config.json`
with seed 0. The [saved metrics](validation_metrics.json) and
[full test output](test_results.txt) accompany the figures.

## Tests

`python -m unittest discover -s tests -v`: 35 tests passed on Python 3.12.14,
NumPy 2.3.5, and Matplotlib 3.10.8.

The checks cover energy balance, uphill thresholds, Bose occupation, detailed
balance, event statistics, population counts, histogram independence, and an
analytically known ladder equilibrium. Optical sampling is checked against
k-space state counts, analytic moments, and independent quadrature. Saved-run
reloads and seed repeatability are also tested.

## Parameters and results

The simulation contains 20,000 electrons evolved for 5,000 ps, with snapshots
every 5 ps. Histograms use 2 meV bins over 0–2 eV kinetic energy. The pulse has
a 2 eV center and 1 eV spectral-intensity FWHM; the direct gap is 1 eV and the
electron/hole masses are both m0. Excitation prepares Gamma electrons.

The bath is 600 K with one 30 meV phonon. Gamma/X/L floors are 0.30/0.05/0 eV;
DOS masses and degeneracies are one. Intra- and intervalley couplings are
0.05 and 0.01 ps^-1. These are the model's example parameters.

| Quantity | Result |
| --- | ---: |
| Scattering events | 6,090,895 |
| Carrier count | Conserved at every snapshot |
| Histogram counts plus overflow | Equal to each valley population |
| Maximum histogram overflow | 0 |
| Maximum per-electron energy-balance error | 2.887e-15 eV |
| Maximum ensemble energy-balance error | 7.276e-12 eV |
| Initial mean kinetic energy | 0.55150010 eV |
| Final mean kinetic energy | 0.07713960 eV |
| Unrestricted 600 K mean kinetic energy, 3 kBT/2 | 0.07755600 eV |
| Final mean total energy | 0.09187460 eV |
| Accessible-equilibrium mean total energy | 0.09203757 eV |

The final joint valley/kinetic-energy histogram is at total variation distance
0.04715 from the accessible equilibrium. Twenty-four independent equilibrium
draws give distances of 0.03891–0.04673. This is a sampling comparison, not a
confidence interval or a test that every observable has equilibrated. The
[model notes](model.md) explain the conserved energy classes and thermal reference.

## Figures

- [Optical preparation](figures/optical_initialization.png): bands, pulse spectrum,
  JDOS weighting, initial electron energies, and energy remainders.
- [Carrier relaxation](figures/optical_relaxation.png): the energy distribution
  over time, time slices, final spectrum, and equilibrium comparison.

Reproduce the run from the repository root after installation:

```bash
python -m carrier_sim --config examples/optical_excitation_config.json --seed 0 --output results/optical_excitation
python -m unittest discover -s tests -v > results/optical_excitation/test_results.txt 2>&1
python examples/plot_optical_comparison.py --run results/optical_excitation --output results/optical_figures
```

The plotting command writes PNG/SVG figures, reference arrays, metrics, and a
report under `results/optical_figures/`.
