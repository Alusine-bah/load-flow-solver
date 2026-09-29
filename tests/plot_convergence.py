"""
Generate a convergence plot comparing Gauss-Seidel and Newton-Raphson.

Both solvers run on the same 4-bus system used in the verification
tests. The mismatch magnitude at each iteration is plotted on a
semi-log scale. The difference in convergence rate — linear for GS,
quadratic for NR — becomes visible immediately.

The output figure is saved to figures/convergence.png and is
embedded in the README.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt

from lfs import Bus, Line, Shunt, PowerSystem, SLACK, PV, PQ
from lfs.ybus import build_ybus
from lfs.gauss_seidel import solve_gauss_seidel
from lfs.newton_raphson import solve_newton_raphson


def build_system():
    """Stevenson 4-bus system used in all verification tests."""
    ps = PowerSystem(name="Stevenson 4-bus", base_mva=100)
    ps.add_bus(Bus(id=1, name="Slack", type=SLACK, V=1.05))
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

    # Run both solvers, same tolerance, and capture the histories.
    gs = solve_gauss_seidel(ps, Ybus, tol=1e-10, max_iter=200,
                            verbose=False, acceleration=1.4)
    nr = solve_newton_raphson(ps, Ybus, tol=1e-10, max_iter=20,
                              verbose=False)

    gs_hist = gs["convergence_history"]
    nr_hist = nr["convergence_history"]

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogy(range(1, len(gs_hist) + 1), gs_hist,
                marker="o", markersize=4, linewidth=1.8,
                color="#2563eb", label=f"Gauss-Seidel ({len(gs_hist)} iterations)")
    ax.semilogy(range(1, len(nr_hist) + 1), nr_hist,
                marker="s", markersize=5, linewidth=2.2,
                color="#dc2626", label=f"Newton-Raphson ({len(nr_hist)} iterations)")

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Maximum mismatch (per-unit)", fontsize=11)
    ax.set_title("Convergence of Load Flow Solvers — Stevenson 4-Bus System",
                 fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=10, loc="upper right")

    # Annotate the quadratic drop after iteration 2 for NR
    if len(nr_hist) >= 4:
        ax.annotate(
            "quadratic drop\n(~20× per step)",
            xy=(4, nr_hist[3]),
            xytext=(6.5, 1e-6),
            fontsize=9,
            color="#7f1d1d",
            arrowprops=dict(arrowstyle="->", color="#7f1d1d", lw=0.8),
        )

    plt.tight_layout()

    # Save
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "figures", "convergence.png"
    )
    out_path = os.path.abspath(out_path)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # Print the numbers, for reference
    print(f"\nGS iterations: {len(gs_hist)}")
    print(f"NR iterations: {len(nr_hist)}")
    print(f"Speedup: {len(gs_hist) / len(nr_hist):.1f}x")


if __name__ == "__main__":
    main()