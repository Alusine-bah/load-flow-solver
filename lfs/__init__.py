"""
lfs — Load Flow Solver
======================

A power system analysis toolkit for balanced steady-state analysis.

Modules
-------
models      : Bus, Line, Generator dataclasses
per_unit    : base conversion and per-unit transforms
ybus        : bus admittance matrix construction
gauss_seidel: Gauss-Seidel power flow solver
newton_raphson: Newton-Raphson power flow solver

Author: Alusine Bah
License: MIT
"""

from .models import (
    Bus, Line, Generator, Shunt, PowerSystem,
    SLACK, PV, PQ,
)

__version__ = "0.1.0"
__all__ = [
    "Bus", "Line", "Generator", "Shunt", "PowerSystem",
    "SLACK", "PV", "PQ",
]