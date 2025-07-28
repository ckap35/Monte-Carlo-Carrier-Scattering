import sys
import os

# Ensure 'src' is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from carrier_sim.simulation import run_simulation, plot_results

if __name__ == "__main__":
    density_history, valley_population_history = run_simulation()
    plot_results(density_history, valley_population_history)
