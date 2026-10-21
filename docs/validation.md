# Validation of the optical excitation example

The validation run uses independent electrons, three parabolic conduction valleys,
and a fixed phonon bath. The band parameters and scattering coefficients are
illustrative and are not fitted to a material.

## Automated checks

`python -m unittest discover -s tests -v`: **35 tests passed**.
See [the complete output](test_results.txt).

Environment: Python 3.12.14, NumPy 2.3.5,
Matplotlib 3.10.8.

The tests cover energy conservation in emission and absorption, forbidden and
allowed uphill transitions, Bose occupation, detailed balance, exponential
event times, competing-channel probabilities, number conservation, histogram
independence, seeded repeatability, saved-configuration reload, and a known
accessible-ladder equilibrium. Optical tests check the JDOS against a k-space
state count, spectral FWHM, electron/hole energy sharing, and sampling against
analytic moments and independent quadrature.

## Optical run

Use `examples/optical_excitation_config.json` with seed 0: 20,000 electrons,
5,000 ps duration, 5 ps snapshots, and 2 meV output bins over 0–2 eV kinetic
energy. The direct gap is 1 eV, the Gaussian pulse center 2 eV, its spectral
intensity FWHM 1 eV, and the electron and hole masses both m0. Excitation
prepares Gamma electrons; holes are recorded at preparation but not evolved.

The bath is 600 K with a 30 meV phonon. Gamma/X/L valley floors are 0.30/0.05/0 eV,
DOS masses and degeneracies are one, and symmetric coupling coefficients are
0.05 ps^-1 within a valley and 0.01 ps^-1 between valleys.

| Check | Result |
| --- | ---: |
| Scattering events | 6,090,895 |
| Carrier population conserved at every snapshot | Yes |
| Histogram counts plus overflow equal valley populations | Yes |
| Maximum histogram overflow | 0 |
| Maximum per-electron energy-ledger error | 2.887e-15 eV |
| Maximum ensemble energy-ledger error | 7.276e-12 eV |
| Initial mean electron kinetic energy | 0.55150010 eV |
| Final mean electron kinetic energy | 0.07713960 eV |
| Unrestricted 600 K mean kinetic energy, 3 kBT/2 | 0.07755600 eV |
| Final mean total electron energy | 0.09187460 eV |
| Accessible-equilibrium mean total energy | 0.09203757 eV |

## Equilibrium interpretation

A single fixed phonon energy conserves each electron's total energy modulo
30 meV. The model equilibrium therefore normalizes DOS-weighted Boltzmann
weights separately on each electron's accessible ladder and preserves the
initial class weights. The unrestricted continuum thermal distribution is
shown separately; a smooth histogram does not establish full thermalization.

The final joint valley/kinetic-histogram total variation distance to model
equilibrium is 0.04715. Twenty-four
independent draws from the same equilibrium give a descriptive range of
0.03891–0.04673.
This range is not a confidence interval or proof that all observables have
equilibrated. There is no temperature or scattering-rate fit.

## Figures and reproduction

- [Optical preparation](figures/optical_initialization.png): parabolic bands,
  pulse spectrum, JDOS weighting, sampled electron energies, and energy classes.
- [Carrier relaxation](figures/optical_relaxation.png): energy distributions over
  time, time slices, final equilibrium comparison, and convergence diagnostic.

```bash
python -m pip install -e .
python -m carrier_sim --config examples/optical_excitation_config.json --seed 0 --output results/optical_excitation
python -m unittest discover -s tests -v > results/optical_excitation/test_results.txt 2>&1
python examples/plot_optical_comparison.py --run results/optical_excitation --output results/optical_figures
```

PNG/SVG figures, reference arrays, numerical metrics, and a report are generated
by the plotting command. Simulation results and regenerated figures are saved
under the ignored `results/` directory.
