"""
Verify the Newton-Raphson solver on Stevenson's 4-bus system.

The same system used for the Gauss-Seidel test. Expected results
should match GS to 4+ decimal places (both converge to the same
solution — the physics is the same), but NR should reach it in
4–6 iterations instead of 17.

We also compare directly against GS to confirm both solvers agree.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lfs import Bus, Line, Shunt, PowerSystem, SLACK, PV, PQ
from lfs.ybus import build_ybus
from lfs.gauss_seidel import solve_gauss_seidel
from lfs.newton_raphson import solve_newton_raphson, print_results


def build_system():
    """Same 4-bus system used in the GS test."""
    ps = PowerSystem(name="Stevenson 4-bus", base_mva=100)

    ps.add_bus(Bus(id=1, name="Slack", type=SLACK, V=1.05, delta=0.0))
    ps.add_bus(Bus(id=2, name="Gen2",  type=PV,    V=1.02, P_gen=0.5))
    ps.add_bus(Bus(id=3, name="Load3", type=PQ,    P_load=0.6, Q_load=0.3))
    ps.add_bus(Bus(id=4, name="Load4", type=PQ,    P_load=0.4, Q_load=0.2))

    ps.add_line(Line(from_bus=1, to_bus=2, R=0.0, X=0.4))
    ps.add_line(Line(from_bus=1, to_bus=3, R=0.0, X=0.2))
    ps.add_line(Line(from_bus=2, to_bus=3, R=0.0, X=0.2))
    ps.add_line(Line(from_bus=3, to_bus=4, R=0.0, X=0.08))

    ps.add_shunt(Shunt(bus_id=1, G=0.0, B=-1.0))
    ps.add_shunt(Shunt(bus_id=2, G=0.0, B=-1.25))

    return ps


def main():
    ps = build_system()
    Ybus = build_ybus(ps)

    print("=" * 65)
    print("Newton-Raphson")
    print("=" * 65)
    nr = solve_newton_raphson(ps, Ybus, tol=1e-8, max_iter=20, verbose=True)
    print_results(ps, nr)

    print("=" * 65)
    print("Gauss-Seidel (for comparison)")
    print("=" * 65)
    gs = solve_gauss_seidel(ps, Ybus, tol=1e-8, max_iter=200,
                            verbose=False, acceleration=1.4)

    # ── Assertions ──
    assert nr["converged"], "Newton-Raphson did not converge"
    assert nr["iterations"] <= 8, (
        f"NR took {nr['iterations']} iterations — expected 4–6"
    )

    # NR should be faster than GS.
    assert nr["iterations"] < gs["iterations"], (
        f"NR ({nr['iterations']}) should beat GS ({gs['iterations']})"
    )

    # The two solvers should agree on the final voltages.
    dV = abs(nr["V_magnitude"] - gs["V_magnitude"])
    max_dV = float(dV.max())
    print(f"\nMax |V| difference between NR and GS: {max_dV:.2e} pu")
    assert max_dV < 1e-5, (
        f"NR and GS disagree by {max_dV:.2e} — should be < 1e-5"
    )

    # Slack and PV magnitudes must be respected.
    assert abs(nr["V_magnitude"][0] - 1.05) < 1e-9
    assert abs(nr["V_magnitude"][1] - 1.02) < 1e-9

    print("\nAll assertions passed.")
    print(f"  NR iterations : {nr['iterations']}")
    print(f"  GS iterations : {gs['iterations']}")
    print(f"  Speedup       : {gs['iterations'] / nr['iterations']:.1f}x")


if __name__ == "__main__":
    main()