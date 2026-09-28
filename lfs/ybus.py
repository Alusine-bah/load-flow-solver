"""
Bus admittance matrix construction.

The bus admittance matrix (Y-bus) is the foundation of every
load flow and fault analysis algorithm. It encodes the entire
network topology as a single n × n complex matrix.

Formation rules (Stevenson §1.12, Saadat §6.2):

    Y_ii = sum of admittances connected directly to bus i
    Y_ij = negative of the admittance between buses i and j   (i ≠ j)

For each transmission line modeled as a π-equivalent:

    Series admittance   y_s  = 1 / (R + jX)
    Shunt admittance    y_sh = G + jB
    Half of y_sh is placed at each end of the line.

So a line between buses p and q contributes:

    Y_pp += y_s + y_sh / 2
    Y_qq += y_s + y_sh / 2
    Y_pq -= y_s
    Y_qp -= y_s

This function is verified against Stevenson's Example 7.1
(a 4-bus system with a known Y-bus).

References
----------
Stevenson, "Elements of Power System Analysis", §1.12, §7.1
Saadat, "Power System Analysis", §6.2
Kothari & Nagrath, "Modern Power System Analysis", §4.3
"""

import numpy as np

from .models import PowerSystem, Line


def build_ybus(system: PowerSystem) -> np.ndarray:
    """
    Construct the bus admittance matrix for a power system.

    Parameters
    ----------
    system : PowerSystem
        The system whose buses and lines define the network.

    Returns
    -------
    Ybus : (n, n) complex numpy array
        The bus admittance matrix, where n = system.n_buses.
        Bus k in the system maps to row/column index k (in the
        order the buses were added — use system.bus_index(id)
        if you need to look up the position of a specific bus).

    Notes
    -----
    - The matrix is symmetric for a network of passive branches.
    - It is complex. Its real part is the conductance matrix G,
      its imaginary part is the susceptance matrix B.
    - The matrix is singular unless at least one shunt element
      is present (which is why the slack bus matters).
    """
    n = system.n_buses
    Ybus = np.zeros((n, n), dtype=complex)

    for line in system.lines:
        _add_line_to_ybus(Ybus, system, line)

    for shunt in system.shunts:
        _add_shunt_to_ybus(Ybus, system, shunt)

    return Ybus


def _add_line_to_ybus(Ybus: np.ndarray, system: PowerSystem, line: Line) -> None:
    """Add one line's contribution to the Y-bus matrix."""
    i = system.bus_index(line.from_bus)
    j = system.bus_index(line.to_bus)

    # Series admittance
    Z = complex(line.R, line.X)
    if Z == 0:
        raise ValueError(
            f"Line {line.from_bus}→{line.to_bus} has zero series impedance; "
            "this would cause a singular Y-bus."
        )
    y_series = 1.0 / Z

    # Shunt admittance (half at each end)
    y_shunt = complex(line.G, line.B)
    y_half_shunt = y_shunt / 2.0

    # Diagonal and off-diagonal contributions
    Ybus[i, i] += y_series + y_half_shunt
    Ybus[j, j] += y_series + y_half_shunt
    Ybus[i, j] -= y_series
    Ybus[j, i] -= y_series


def _add_shunt_to_ybus(Ybus: np.ndarray, system: PowerSystem, shunt) -> None:
    """Add a shunt admittance to the diagonal of the Y-bus."""
    i = system.bus_index(shunt.bus_id)
    Ybus[i, i] += shunt.Y


# ── Reporting helpers ────────────────────────────────────────

def print_ybus(system: PowerSystem, Ybus: np.ndarray) -> None:
    """Pretty-print the Y-bus matrix with row and column labels."""
    n = Ybus.shape[0]
    labels = [f"{b.id}:{b.name[:8]}" for b in system.buses]

    print(f"\nY-bus matrix ({n} × {n}) for '{system.name}'")
    print("Values shown as G + jB, in per-unit\n")

    # Header row
    header = "        " + "".join(f"{lbl:>16}" for lbl in labels)
    print(header)
    print("        " + "-" * (16 * n))

    for i, row_label in enumerate(labels):
        cells = []
        for j in range(n):
            y = Ybus[i, j]
            cells.append(f"{y.real:+.4f}{y.imag:+.4f}j".rjust(16))
        print(f"{row_label:>6} |" + "".join(cells))

    print()
    _print_ybus_summary(Ybus)


def _print_ybus_summary(Ybus: np.ndarray) -> None:
    """Print a short structural summary of the Y-bus."""
    n = Ybus.shape[0]
    nonzero = int(np.count_nonzero(Ybus))
    density = nonzero / (n * n) * 100

    # Check symmetry
    is_symmetric = np.allclose(Ybus, Ybus.T)

    print("Structure:")
    print(f"  Non-zero entries : {nonzero} / {n*n}  ({density:.1f}%)")
    print(f"  Symmetric        : {is_symmetric}")
    print(f"  Real part range  : "
          f"{Ybus.real.min():.4f} to {Ybus.real.max():.4f}")
    print(f"  Imag part range  : "
          f"{Ybus.imag.min():.4f} to {Ybus.imag.max():.4f}")


# ── Quick numeric self-test ──────────────────────────────────

def _self_test() -> None:
    """
    Reproduce Stevenson's Example 7.1 (4-bus Y-bus) and compare.

    From the lecture notes: the expected Y-bus for
    the small 4-bus network is

        [ -j8.50   j2.50   j5.00   0      ]
        [  j2.50  -j8.75   j5.00   0      ]
        [  j5.00   j5.00  -j22.50  j12.50 ]
        [  0       0       j12.50  -j12.50]

    The buses have no resistance, only reactance.
    """
    from .models import Bus, Line, Shunt, PowerSystem, SLACK, PQ

    sys = PowerSystem(name="Stevenson Example 7.1", base_mva=100)
    # Four buses, slack = 1, rest are PQ
    sys.add_bus(Bus(id=1, type=SLACK))
    sys.add_bus(Bus(id=2, type=PQ))
    sys.add_bus(Bus(id=3, type=PQ))
    sys.add_bus(Bus(id=4, type=PQ))

    # Line data from the lecture: R=0 for all, X in pu.
    # Line 1-2: X = 0.4   → y = -j2.5
    # Line 1-3: X = 0.2   → y = -j5.0
    # Line 2-3: X = 0.2   → y = -j5.0
    # Line 3-4: X = 0.08  → y = -j12.5
    sys.add_line(Line(from_bus=1, to_bus=2, R=0.0, X=0.4))
    sys.add_line(Line(from_bus=1, to_bus=3, R=0.0, X=0.2))
    sys.add_line(Line(from_bus=2, to_bus=3, R=0.0, X=0.2))
    sys.add_line(Line(from_bus=3, to_bus=4, R=0.0, X=0.08))

    # Shunt admittances at buses 1 and 2 (from the lecture slide).
    # y10 = -j1.0, y20 = -j1.25
    sys.add_shunt(Shunt(bus_id=1, G=0.0, B=-1.0))
    sys.add_shunt(Shunt(bus_id=2, G=0.0, B=-1.25))

    Ybus = build_ybus(sys)
    print_ybus(sys, Ybus)


if __name__ == "__main__":
    _self_test()