"""
Gauss-Seidel power flow solver.

The Gauss-Seidel method solves the power flow equations iteratively.
For each PQ bus, it updates the complex voltage using the calculated
injection P and Q from the previous iteration:

    V_i^(k+1) = (1 / Y_ii) * [ (P_i - jQ_i) / conj(V_i^(k))
                              - Σ_{j≠i} Y_ij · V_j ]

For PV buses, only the angle is updated; the magnitude stays fixed.
The slack bus voltage is never updated.

Convergence is declared when the largest change in any voltage
magnitude between iterations falls below a tolerance.

References
----------
Saadat, "Power System Analysis", 2nd ed., §6.3.1, §6.5
Stevenson, "Elements of Power System Analysis", §9.4
"""

import numpy as np

from .models import PowerSystem, Bus, SLACK, PV, PQ


def solve_gauss_seidel(
    system: PowerSystem,
    Ybus: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 100,
    verbose: bool = False,
    acceleration: float = 1.0,
) -> dict:
    """
    Solve the power flow problem using the Gauss-Seidel method.

    Parameters
    ----------
    system : PowerSystem
        The system to solve. Must have exactly one slack bus.
    Ybus : np.ndarray
        The bus admittance matrix from build_ybus(system).
    tol : float
        Convergence tolerance on the maximum voltage magnitude change.
    max_iter : int
        Maximum number of iterations before giving up.
    verbose : bool
        If True, print iteration-by-iteration progress.
    acceleration : float
        Relaxation factor. 1.0 is the standard Gauss-Seidel update.
        Values slightly above 1 (e.g. 1.2 to 1.6) often accelerate
        convergence.

    Returns
    -------
    result : dict with keys
        'converged'      : bool
        'iterations'     : int
        'V'              : np.ndarray of complex bus voltages
        'V_magnitude'    : np.ndarray of |V| for each bus
        'delta_rad'      : np.ndarray of voltage angles in radians
        'delta_deg'      : np.ndarray of voltage angles in degrees
        'P_injected'     : np.ndarray of net P at each bus (pu)
        'Q_injected'     : np.ndarray of net Q at each bus (pu)
        'convergence_history' : list of max |ΔV| per iteration
    """
    n = system.n_buses
    slack = system.slack_bus()
    slack_idx = system.bus_index(slack.id)

    # Build the scheduled power injection vectors.
    P_sched, Q_sched = _scheduled_injections(system)

    # Initial voltage estimate: flat start (all V = 1 ∠ 0)
    V = np.array([bus.V * np.exp(1j * bus.delta) for bus in system.buses],
                 dtype=complex)

    # Find PV and PQ bus indices.
    pq_indices = [i for i, b in enumerate(system.buses) if b.type == PQ]
    pv_indices = [i for i, b in enumerate(system.buses) if b.type == PV]

    convergence_history = []

    for iteration in range(1, max_iter + 1):
        V_old = V.copy()

        # ── Update PQ buses (both magnitude and angle free) ──
        for i in pq_indices:
            _update_pq_bus(V, Ybus, P_sched, Q_sched, i, acceleration)

        # ── Update PV buses (angle free, magnitude fixed) ──
        for i in pv_indices:
            _update_pv_bus(V, Ybus, P_sched, i, system.buses[i].V, acceleration)

        # ── Check convergence ──
        max_change = float(np.max(np.abs(V - V_old)))
        convergence_history.append(max_change)

        if verbose:
            print(f"  iter {iteration:3d}   max|ΔV| = {max_change:.3e}")

        if max_change < tol:
            break

    # Compute final P and Q injections from the converged voltages.
    P_calc, Q_calc = _calculate_injections(V, Ybus)

    result = {
        "converged": max_change < tol,
        "iterations": iteration,
        "V": V,
        "V_magnitude": np.abs(V),
        "delta_rad": np.angle(V),
        "delta_deg": np.degrees(np.angle(V)),
        "P_injected": P_calc,
        "Q_injected": Q_calc,
        "convergence_history": convergence_history,
    }
    return result


# ── Internal helpers ─────────────────────────────────────────

def _scheduled_injections(system: PowerSystem):
    """Build arrays of scheduled net P and Q injections at every bus."""
    n = system.n_buses
    P = np.zeros(n)
    Q = np.zeros(n)
    for i, bus in enumerate(system.buses):
        P[i] = bus.P_injected
        Q[i] = bus.Q_injected
    return P, Q


def _update_pq_bus(V, Ybus, P_sched, Q_sched, i, acceleration):
    """
    Apply one Gauss-Seidel update to a PQ bus.

    From Saadat Eq. 6.31:
        V_i = (1/Y_ii) * [ (P_i - jQ_i)/conj(V_i) - Σ_{j≠i} Y_ij V_j ]
    """
    # Sum of off-diagonal terms: Σ_{j≠i} Y_ij V_j
    off_sum = Ybus[i, :] @ V - Ybus[i, i] * V[i]

    # The scheduled complex power S_i = P_i + jQ_i
    S_conj = P_sched[i] - 1j * Q_sched[i]

    # Proposed update
    V_new = (S_conj / np.conj(V[i]) - off_sum) / Ybus[i, i]

    # Apply acceleration (relaxation)
    if acceleration != 1.0:
        V_new = V[i] + acceleration * (V_new - V[i])

    V[i] = V_new


def _update_pv_bus(V, Ybus, P_sched, i, V_setpoint, acceleration):
    """
    Apply one Gauss-Seidel update to a PV bus.

    The voltage magnitude is fixed at V_setpoint. Only the angle is
    updated. The standard technique is to compute the complex
    update, then rescale to the set magnitude.
    """
    # Compute Q from the current voltage estimate.
    S_calc = V[i] * np.conj(Ybus[i, :] @ V)
    Q_calc = S_calc.imag

    off_sum = Ybus[i, :] @ V - Ybus[i, i] * V[i]
    S_conj = P_sched[i] - 1j * Q_calc
    V_new = (S_conj / np.conj(V[i]) - off_sum) / Ybus[i, i]

    # Rescale magnitude to the setpoint, keep the new angle.
    mag = np.abs(V_new)
    if mag > 0:
        V_new = V_new * (V_setpoint / mag)

    if acceleration != 1.0:
        V_new = V[i] + acceleration * (V_new - V[i])

    V[i] = V_new


def _calculate_injections(V, Ybus):
    """
    Given converged voltages, compute the net P and Q injected at
    each bus:  S_i = V_i · conj(Σ_j Y_ij V_j)
    """
    I = Ybus @ V
    S = V * np.conj(I)
    return S.real, S.imag


# ── Reporting helper ─────────────────────────────────────────

def print_results(system: PowerSystem, result: dict) -> None:
    """Print a formatted table of results."""
    print(f"\nGauss-Seidel Results for '{system.name}'")
    status = "CONVERGED" if result["converged"] else "NOT CONVERGED"
    print(f"Status   : {status}")
    print(f"Iterations: {result['iterations']}")
    print(f"Tolerance : max|ΔV| = {result['convergence_history'][-1]:.3e}\n")

    header = (f"{'Bus':>5} {'Name':<12} {'Type':<7} "
              f"{'|V| (pu)':>10} {'Angle (°)':>12} "
              f"{'P (pu)':>10} {'Q (pu)':>10}")
    print(header)
    print("-" * len(header))

    for i, bus in enumerate(system.buses):
        print(f"{bus.id:>5} {bus.name[:12]:<12} {bus.type:<7} "
              f"{result['V_magnitude'][i]:>10.4f} "
              f"{result['delta_deg'][i]:>12.4f} "
              f"{result['P_injected'][i]:>10.4f} "
              f"{result['Q_injected'][i]:>10.4f}")
    print()