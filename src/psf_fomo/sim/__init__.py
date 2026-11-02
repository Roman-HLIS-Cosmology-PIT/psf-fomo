"""Simulation module: Roman star images with known truth and controlled PSF errors.

Main entry points: `RomanSimulation` (builder), `Simulation` (result) and `load_simulation`.
"""
from .simulation import RomanSimulation, Simulation, load_simulation

__all__ = ["RomanSimulation", "Simulation", "load_simulation"]
