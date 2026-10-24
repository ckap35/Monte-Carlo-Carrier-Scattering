# Carrier model

Energies are in eV and times in ps. Kinetic energy is measured from the floor
of the occupied valley. Total electron energy is the sum of the floor and
kinetic energy; its zero is independent of the optical direct gap.

## Phonon scattering

For a transition from valley i to j,

```text
emission:   E_final = E_initial + floor_i - floor_j - phonon_energy
absorption: E_final = E_initial + floor_i - floor_j + phonon_energy
```

Here `E_initial` and `E_final` are kinetic energies. Both channels are considered
for every destination valley, including the starting valley. Negative final
kinetic energies have zero rate. An uphill valley transition can absorb a
phonon, or emit one if the initial electron has enough kinetic energy.

For phonon occupation `N = 1 / expm1(phonon_energy / (k_B T))`,

```text
rate_emission   = C_ij * DOS_j(E_final) * (N + 1)
rate_absorption = C_ij * DOS_j(E_final) * N
DOS_j(E)       = g_j * mass_ratio_j^(3/2) * sqrt(E / E_reference)
```

The DOS is dimensionless, `C_ij` is in ps^-1, and `E_reference` defaults to
1 eV. The coupling matrix is symmetric and nonnegative. These rates satisfy
detailed balance with DOS-weighted Boltzmann occupations. Absorption vanishes
at zero temperature.

The default Gamma/X/L floors are 0.30/0.05/0 eV; DOS masses and degeneracies
are one. Coupling coefficients are 0.05 ps^-1 within a valley and 0.01 ps^-1
between valleys. The optical example uses a 600 K bath and a 30 meV phonon.

The waiting time is exponential with the total outgoing rate. A channel is
then drawn in proportion to its rate, and the next clock starts at the event
time. Every event before an output snapshot is processed. Histogram bins
are used only for recording the result; electrons outside the histogram range
are counted separately in `overflow_history`.

## Optical preparation

Assume isotropic 3D parabolic valence and conduction bands and an allowed
vertical transition. With positive electron and hole masses,

$$
\frac{1}{\mu}=\frac{1}{m_e}+\frac{1}{m_h},\qquad
E_\gamma=E_g+\frac{\hbar^2 k^2}{2\mu}.
$$

The spin-inclusive JDOS for one band pair is

$$
J(E_\gamma)=\frac{1}{2\pi^2}
\left(\frac{2\mu}{\hbar^2}\right)^{3/2}
\sqrt{E_\gamma-E_g},\qquad E_\gamma>E_g,
$$

and zero below the gap. `OpticalExcitation.joint_dos` converts this expression
to states per eV per cm³.

The pulse spectrum is the electric-field spectral intensity,

$$
S(E_\gamma)=\exp\left[-4\ln 2
\left(\frac{E_\gamma-E_0}{\Delta E}\right)^2\right],
$$

where `Delta E` is the spectral-intensity FWHM. With a constant interband
dipole, a filled valence band, and an empty conduction band, the generated
pair distribution is proportional to `J(E_gamma) * S(E_gamma)`.

The electron and hole share the excess energy:

$$
\varepsilon_e=\frac{m_h}{m_e+m_h}(E_\gamma-E_g),\qquad
\varepsilon_h=\frac{m_e}{m_e+m_h}(E_\gamma-E_g).
$$

The direct gap is measured from the valence maximum to the injected conduction
valley. Valley floors use a common conduction-band energy reference. The optical
electron mass matches the injected valley's DOS mass. All masses are expressed
as ratios to the free-electron mass.

The example uses equal masses, so each particle receives half the excess
energy. A 1 eV pulse bandwidth therefore maps to a 0.5 eV electron bandwidth
before JDOS weighting. Pair energies are drawn by rejection sampling with
unbounded energy support. `num_carriers` sets the number of sampled pairs.
The pulse prepares the state at t=0, after which the electrons undergo phonon
scattering.

## Equilibrium and conservation

For the connected, positive-temperature example, equilibrium has DOS-weighted
Boltzmann weights on each electron's accessible energy ladder. A single phonon
energy preserves total energy modulo that phonon energy, so the reference
normalizes each ladder separately before combining the electrons.

The continuum Boltzmann kinetic-energy distribution is proportional to
`sqrt(E) * exp(-E / (k_B T))`, with mean `3 k_B T / 2`. The accessible
reference conditions the Boltzmann weights on the conserved energy classes.

At each snapshot, electron energy plus net energy transferred to the phonon
bath equals the initial electron energy, up to roundoff. The saved arrays
also record this balance per electron, channel event counts, valley populations,
and histogram overflow.

## References

- [Jacoboni and Reggiani, Rev. Mod. Phys. 55, 645 (1983)](https://doi.org/10.1103/RevModPhys.55.645).
- [Eric Pop, Monte Carlo Transport and Heat Generation in Semiconductors (2014)](https://poplab.stanford.edu/pdfs/Pop-HeatGenSemic-arht14.pdf).
- [David Miller, Optical absorption in semiconductors, Lecture 32](https://web.stanford.edu/group/dabmgroup/cgi-bin/dabm/wp-content/uploads/2021/12/Lecture_32.pdf).
