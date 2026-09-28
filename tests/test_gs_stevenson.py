"""
Verify the Gauss-Seidel solver against Stevenson's 4-bus example.

The system is the same one used for the Y-bus self-test:
4 buses, 4 lines, 2 shunts.

Bus 1 is the slack (V = 1.05 ∠ 0°).
Buses 2, 3, 4 are load buses with the loads shown below.

Expected results (from Stevenson and the lecture notes):
After convergence the voltage magnitudes should be approximately:

    Bus 1 : 1.0500
    Bus 2 : 1.0200  (this is a PV bus, magnitude is fixed)
    Bus 3 : ~0.98 - 1.00
    Bus 4 : ~0.95 - 1.00

We print the results so they can be compared against the textbook.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lfs import Bus, Line, Shunt, PowerSystem, SLACK, PV, PQ
from lfs.ybus import build_ybus, print_ybus
from lfs.gauss_seidel import solve_gauss_seidel, print_results


def main():
    sys_ps = PowerSystem(name="Stevenson 4-bus", base_mva=100)

    # Buses
    sys_ps.add_bus(Bus(id=1, name="Slack", type=SLACK, V=1.05, delta=0.0))
    sys_ps.add_bus(Bus(id=2, name="Gen2",  type=PV,    V=1.02, P_gen=0.5))
    sys_ps.add_bus(Bus(id=3, name="Load3", type=PQ,    P_load=0.6, Q_load=0.3))
    sys_ps.add_bus(Bus(id=4, name="Load4", type=PQ,    P_load=0.4, Q_load=0.2))

    # Lines (all per-unit; R = 0)
    sys_ps.add_line(Line(from_bus=1, to_bus=2, R=0.0, X=0.4))
    sys_ps.add_line(Line(from_bus=1, to_bus=3, R=0.0, X=0.2))
    sys_ps.add_line(Line(from_bus=2, to_bus=3, R=0.0, X=0.2))
    sys_ps.add_line(Line(from_bus=3, to_bus=4, R=0.0, X=0.08))

    # Shunts
    sys_ps.add_shunt(Shunt(bus_id=1, G=0.0, B=-1.0))
    sys_ps.add_shunt(Shunt(bus_id=2, G=0.0, B=-1.25))

    print(sys_ps.summary())
    print()

    Ybus = build_ybus(sys_ps)
    print_ybus(sys_ps, Ybus)

    result = solve_gauss_seidel(
        sys_ps, Ybus,
        tol=1e-8,
        max_iter=200,
        verbose=True,
        acceleration=1.4,
    )

    print_results(sys_ps, result)

    # Assertions
    assert result["converged"], "Solver did not converge"
    V = result["V_magnitude"]

    assert abs(V[0] - 1.05) < 1e-9, "Slack magnitude should stay at 1.05"
    assert abs(V[1] - 1.02) < 1e-9, "PV magnitude should stay at 1.02"
    assert 0.90 < V[2] < 1.05, f"Bus 3 magnitude {V[2]} out of plausible range"
    assert 0.90 < V[3] < 1.05, f"Bus 4 magnitude {V[3]} out of plausible range"

    print("All assertions passed.")


if __name__ == "__main__":
    main()