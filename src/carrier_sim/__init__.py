"""Energy-conserving, fixed-bath carrier scattering with illustrative rates."""

from .physics import ScatteringModel
from .excitation import OpticalExcitation, OpticalPairs
from .simulation import SimulationConfig, SimulationResult, run_simulation, plot_results

__all__ = ["ScatteringModel", "SimulationConfig", "SimulationResult",
           "run_simulation", "plot_results", "OpticalExcitation", "OpticalPairs"]
