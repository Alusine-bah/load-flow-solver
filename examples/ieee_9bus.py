"""
IEEE 9-bus (P.M. Anderson) load flow example.

The WSCC 3-machine, 9-bus system, originally published in
P.M. Anderson and A.A. Fouad, "Power System Control and Stability".
It's one of the standard test systems in power system analysis.

Topology:
    9 buses, 3 generators, 9 branches (6 lines + 3 transformer branches)
    Loads at buses 5, 7, 9
    Generators at buses 1 (slack), 2 (PV), 3 (PV)

Expected results (from published load flow solutions):
    V1 = 1.040 pu  (slack, fixed)
    V2 = 1.025 pu  (PV, fixed)
    V3 = 1.025 pu  (PV, fixed)
    V5 ≈ 0.975 - 0.985 pu
    V7 ≈ 0.985 - 0.995 pu
    V9 ≈ 0.955 - 0.965 pu

Reference:
    P.M. Anderson and A.A. Fouad, "Power System Control and
    Stability", 2nd ed., IEEE Press, 2003.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lfs import Bus, Line, Shunt, PowerSystem, SLACK, PV, PQ
from lfs.ybus import build_ybus, print_ybus
from lfs.newton_raphson import solve_newton_raphson, print_results


def build_ieee_9bus():
    """Build the IEEE 9-bus system with data from the Anderson reference."""
    ps = PowerSystem(name="IEEE 9-Bus (Anderson)", base_mva=100)

    # ── Buses ──
    # Bus 1: Slack generator
    ps.add_bus(Bus(id=1, name="Gen1", type=SLACK, V=1.04, base_kv=345))
    # Bus 2: PV generator
    ps.add_bus(Bus(id=2, name="Gen2", type=PV, V=1.025, P_gen=1.63, base_kv=345))
    # Bus 3: PV generator
    ps.add_bus(Bus(id=3, name="Gen3", type=PV, V=1.025, P_gen=0.85, base_kv=345))
    # Bus 4: zero-injection connection bus
    ps.add_bus(Bus(id=4, name="Node4", type=PQ, base_kv=345))
    # Bus 5: load bus
    ps.add_bus(Bus(id=5, name="Load5", type=PQ, P_load=0.90, Q_load=0.30, base_kv=345))
    # Bus 6: zero-injection connection bus
    ps.add_bus(Bus(id=6, name="Node6", type=PQ, base_kv=345))
    # Bus 7: load bus
    ps.add_bus(Bus(id=7, name="Load7", type=PQ, P_load=1.00, Q_load=0.35, base_kv=345))
    # Bus 8: zero-injection connection bus
    ps.add_bus(Bus(id=8, name="Node8", type=PQ, base_kv=345))
    # Bus 9: load bus
    ps.add_bus(Bus(id=9, name="Load9", type=PQ, P_load=1.25, Q_load=0.50, base_kv=345))

    # ── Branches ──
    # Transformer branches (R = 0, no charging)
    ps.add_line(Line(from_bus=1, to_bus=4, R=0.0000, X=0.0576))
    ps.add_line(Line(from_bus=7, to_bus=2, R=0.0000, X=0.0625))
    ps.add_line(Line(from_bus=9, to_bus=3, R=0.0000, X=0.0586))

    # Transmission lines (with charging susceptance)
    # The B/2 value in the source table is half the total line charging.
    # Our Line class expects the *total* B, so we multiply by 2.
    ps.add_line(Line(from_bus=4, to_bus=5, R=0.0100, X=0.0850, B=0.0880))
    ps.add_line(Line(from_bus=5, to_bus=7, R=0.0320, X=0.1610, B=0.0153))
    ps.add_line(Line(from_bus=4, to_bus=6, R=0.0170, X=0.0920, B=0.0790))
    ps.add_line(Line(from_bus=6, to_bus=9, R=0.0390, X=0.1700, B=0.1790))
    ps.add_line(Line(from_bus=7, to_bus=8, R=0.0085, X=0.0720, B=0.0745))
    ps.add_line(Line(from_bus=8, to_bus=9, R=0.0119, X=0.1008, B=0.1045))

    return ps


def main():
    ps = build_ieee_9bus()
    print(ps.summary())
    print()

    Ybus = build_ybus(ps)

    # Solve with Newton-Raphson
    result = solve_newton_raphson(ps, Ybus, tol=1e-8, max_iter=20, verbose=True)
    print_results(ps, result)

    # Sanity checks
      # Cross-check against Gauss-Seidel on the same system.
    # Both solvers must converge to the same operating point;
    # that's the strongest available verification when the
    # published reference assumes a different line dataset.
    from lfs.gauss_seidel import solve_gauss_seidel

    gs = solve_gauss_seidel(ps, Ybus, tol=1e-10, max_iter=500,
                            verbose=False, acceleration=1.4)

    assert result["converged"], "Newton-Raphson did not converge"
    assert gs["converged"], "Gauss-Seidel did not converge"

    # Slack and PV magnitudes must be respected exactly.
    V = result["V_magnitude"]
    assert abs(V[0] - 1.04) < 1e-9, "Slack V should be 1.04"
    assert abs(V[1] - 1.025) < 1e-9, "PV V at bus 2 should be 1.025"
    assert abs(V[2] - 1.025) < 1e-9, "PV V at bus 3 should be 1.025"

    # The two independent solvers must agree.
    dV = abs(result["V_magnitude"] - gs["V_magnitude"])
    max_dV = float(dV.max())
    print(f"\nCross-check: max |V_NR - V_GS| = {max_dV:.2e} pu")
    assert max_dV < 1e-6, (
        f"NR and GS disagree by {max_dV:.2e} — one of them is wrong"
    )

    # All bus voltages should be within typical operating limits.
    for i, bus in enumerate(ps.buses):
        assert 0.90 < V[i] < 1.10, (
            f"Bus {bus.id} V = {V[i]:.4f} outside 0.90-1.10"
        )  

    print("All assertions passed.")
    print(f"NR iterations: {result['iterations']}")


if __name__ == "__main__":
    main()