# Carrier Scattering Simulation

This project simulates carrier dynamics across semiconductor valleys with intra- and inter-valley phonon scattering. It uses a Monte Carlo-like approach and produces time-resolved energy distributions.

## Features
- Temperature- and energy-dependent scattering rates
- Multi-valley transitions with phonon emission/absorption
- Visualizations of carrier dynamics

## Installation

Use Python 3.8+ and install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Simulation

```bash
PYTHONPATH=src python examples/run_simulation.py
```

## Project Structure

- `src/carrier_sim/simulation.py`: Main simulation logic
- `examples/run_simulation.py`: How to run the simulation
- `README.md`: Project overview
- `requirements.txt`: Required Python libraries
- `LICENSE`: MIT license

## License
MIT
